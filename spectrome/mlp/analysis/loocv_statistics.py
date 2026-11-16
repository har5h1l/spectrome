"""Generate a LOOCV ranking or a named comparison with the analytical filter."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from tempfile import TemporaryDirectory

from .helpers import load_loocv_scores, load_run_config, load_saved_object, write_csv
from .statistics import apply_bonferroni, paired_fisher_z_test

REPO = Path(__file__).resolve().parents[3]
HISTORICAL_RUNS = tuple(f"basic_mlp_{index}" for index in range(16))


def _path_label(path):
    return str(path.relative_to(REPO) if path.is_relative_to(REPO) else path)


def _collect_scores(experiments, run_ids, reference_id):
    configs = {run_id: load_run_config(experiments / run_id) for run_id in run_ids}
    reference_config = configs[reference_id]
    reference_dir = experiments / reference_id
    reference = load_loocv_scores(reference_dir)
    ids = reference["provenance"]["subject_ids"]
    analytical = reference["analytical"]
    reference_cache = None
    runs = []
    for run_id, config in configs.items():
        run_dir = experiments / run_id
        if config.get("status") != "complete":
            raise ValueError(f"{run_id}: expected a completed LOOCV run")
        if run_id == reference_id:
            scores = reference
        elif config["cache_sha256"] == reference_config["cache_sha256"]:
            scores = load_loocv_scores(run_dir)
        else:
            if reference_cache is None:
                reference_cache = load_saved_object(reference_dir / reference_config["cache_path"])
            scores = load_loocv_scores(run_dir, reference_cache=reference_cache)
        if (set(scores["analytical"]) != set(ids) or set(scores["learned"]) != set(ids)
                or any(not math.isclose(scores["analytical"][key], analytical[key],
                                        rel_tol=1e-12, abs_tol=1e-12) for key in ids)):
            raise ValueError(f"{run_id}: subject IDs or analytical scores differ from {reference_id}")
        model = config["model_config"]
        training = config["training_config"]
        learned = [scores["learned"][key] for key in ids]
        baseline = [analytical[key] for key in ids]
        gains = [left - right for left, right in zip(learned, baseline)]
        row = {
            "run_id": run_id, "hidden_sizes": json.dumps(model["hidden_sizes"]),
            "activation": model.get("activation", "tanh"), "omega_scale": model["omega_scale"],
            **{key: training[key] for key in ("epochs", "learning_rate", "seed")},
            **{key: training.get(key) for key in ("optimizer", "device", "dtype")},
            "n_subjects": len(ids), "mean_r": mean(learned), "sd_r": stdev(learned),
            "analytical_mean_r": mean(baseline), "analytical_sd_r": stdev(baseline),
            "mean_gain_r": mean(gains), "sd_gain_r": stdev(gains),
            "n_improved": sum(gain > 0 for gain in gains), "run_dir": _path_label(run_dir),
        }
        source = {
            "run_id": run_id, "run_dir": row["run_dir"],
            "config_sha256": hashlib.sha256((run_dir / "config.json").read_bytes()).hexdigest(),
            "cache_sha256": config["cache_sha256"],
            "forward_sha256": scores["provenance"]["forward_sha256"],
            "scores_sha256": hashlib.sha256(
                json.dumps([scores["learned"], scores["analytical"]], sort_keys=True).encode()
            ).hexdigest(),
        }
        runs.append((row, scores["learned"], source))
    return sorted(runs, key=lambda run: (-run[0]["mean_gain_r"], run[0]["run_id"])), analytical, ids


def _snapshot_files(directory, snapshot_id, runs, analytical, ids, reference_id):
    contrasts = [f"{row['run_id']}_vs_analytical" for row, _, _ in runs]
    tests = [
        paired_fisher_z_test(
            learned, analytical, left_label=row["run_id"], right_label="analytical",
            contrast=contrast, expected_subject_ids=ids,
        )
        for (row, learned, _), contrast in zip(runs, contrasts)
    ]
    family = apply_bonferroni(tests, family_name=snapshot_id, planned_contrasts=contrasts)
    rows = []
    for (row, _, _), comparison in zip(runs, family["comparisons"]):
        interval = comparison["fisher_z_difference_95_percent_ci"] or {}
        rows.append({
            **row, "contrast": comparison["contrast"],
            "mean_difference_z": comparison["fisher_z_difference_mean"],
            "sd_difference_z": comparison["fisher_z_difference_sd"],
            "ci95_low_z": interval.get("low"), "ci95_high_z": interval.get("high"),
            "t_statistic": comparison["t_statistic"],
            "degrees_of_freedom": comparison["degrees_of_freedom"],
            "p_raw": comparison["p_value_unadjusted"],
            "p_bonferroni": comparison["adjustment"]["p_value"],
            "family_size": family["planned_tests"], "test_status": comparison["status"],
        })
    manifest = {
        "snapshot_id": snapshot_id,
        "run_ids": [row["run_id"] for row in rows],
        "family_size": family["planned_tests"], "alpha": family["alpha"],
        "correction": family["method"], "method": tests[0]["method"],
        "confidence_intervals": "Unadjusted 95% Fisher-z intervals",
        "scope": "Exploratory fixed-configuration LOOCV on a reused cohort",
        "reference_run_id": reference_id, "subject_ids": ids,
        "inputs": [source for _, _, source in runs],
    }
    write_csv(directory / "comparisons.csv", rows)
    (directory / "family.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", choices=("ranking", "snapshot"))
    parser.add_argument("--run-ids", nargs="+", default=HISTORICAL_RUNS)
    parser.add_argument("--reference-run", default="basic_mlp_1")
    parser.add_argument("--snapshot-id")
    parser.add_argument("--experiments", type=Path, default=REPO / "results/experiments")
    parser.add_argument("--output", type=Path, default=REPO / "results/analysis/loocv-vs-analytical")
    args = parser.parse_args(argv)
    if (len(set(args.run_ids)) != len(args.run_ids)
            or any(not name or Path(name).name != name or name in {".", ".."} for name in args.run_ids)
            or args.reference_run not in args.run_ids):
        parser.error("provide unique run folder names, including the reference run")
    if args.report == "snapshot":
        name = args.snapshot_id
        if not name or Path(name).name != name or name in {".", ".."}:
            parser.error("snapshot requires a single folder name via --snapshot-id")
        destination = args.output / "snapshots" / name
    else:
        if args.snapshot_id is not None:
            parser.error("--snapshot-id applies only to snapshot")
        destination = args.output

    runs, analytical, ids = _collect_scores(args.experiments.resolve(), args.run_ids, args.reference_run)
    parent = destination.parent if args.report == "snapshot" else destination
    parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=parent) as temporary:
        stage = Path(temporary) / "report"
        stage.mkdir()
        if args.report == "ranking":
            write_csv(stage / "ranking.csv", [row for row, _, _ in runs]).replace(destination / "ranking.csv")
        else:
            _snapshot_files(stage, name, runs, analytical, ids, args.reference_run)
            if destination.exists():
                if any((destination / path.name).read_bytes() != path.read_bytes() for path in stage.iterdir()):
                    raise ValueError(f"{destination}: snapshot differs; use a new --snapshot-id")
            else:
                stage.rename(destination)
    print(destination)


if __name__ == "__main__":
    main()

"""Export score tables and regenerate reports from saved results."""

import json
from pathlib import Path

import numpy as np
import torch

from .helpers import (
    _load_inputs, _write_csv, _region_scores, _analytical_filter,
    load_run_config, load_loocv_run, load_nested_run, summarize_outer_scores,
)
from .plots import plot_results


@torch.no_grad()
def export_results(results, cache, output_dir):
    """Write subject, region, epoch, and complex-filter CSV tables.

    Inputs may be saved paths or loaded dictionaries. Region indices are
    zero-based model nodes, not guessed anatomical labels. Subject scores are
    checked against the saved scalar metrics before any tables are written.
    """
    results, cache, subjects = _load_inputs(results, cache)
    output_dir = Path(output_dir)
    subject_rows, region_rows, history_rows, filter_rows = [], [], [], []
    for fold in results["folds"]:
        subject = subjects[fold["held_out_subject_id"]]
        base = {"fold": fold["fold"], "subject_id": subject["subject_id"]}
        analytical = _region_scores(fold["analytical_prediction"], subject["empirical_meg"])
        learned = _region_scores(fold["learned_prediction"], subject["empirical_meg"])
        if (not np.isclose(analytical.mean().item(), fold["analytical_correlation"], atol=1e-10, rtol=1e-8)
                or not np.isclose(learned.mean().item(), fold["learned_correlation"], atol=1e-10, rtol=1e-8)):
            raise ValueError("Recomputed scores differ from saved results; check the cache")
        subject_rows.append({**base, "n_training_subjects": len(fold["training_subject_ids"]),
                             "n_regions": analytical.numel(),
                             "analytical_training_correlation": fold.get("analytical_training_correlation", ""),
                             "learned_training_correlation": fold.get("learned_training_correlation", ""),
                             "analytical_correlation": analytical.mean().item(),
                             "learned_correlation": learned.mean().item(),
                             "correlation_difference": (learned.mean() - analytical.mean()).item()})
        for row, node in enumerate(subject["observed_regions"].tolist()):
            region_rows.append({**base, "meg_region_row": row, "model_node_index": node,
                                "analytical_correlation": analytical[row].item(),
                                "learned_correlation": learned[row].item(),
                                "correlation_difference": (learned[row] - analytical[row]).item()})
        history_rows.extend({**base, **entry} for entry in fold["history"])
        control = _analytical_filter(subject, cache["frequencies_hz"])
        for frequency, baseline, value in zip(cache["frequencies_hz"], control, fold["Fe_graph"]):
            filter_rows.append({**base, "frequency_hz": frequency.item(),
                                "analytical_real": baseline.real.item(),
                                "analytical_imaginary": baseline.imag.item(),
                                "analytical_magnitude": baseline.abs().item(),
                                "analytical_phase_rad": baseline.angle().item(),
                                "learned_real": value.real.item(),
                                "learned_imaginary": value.imag.item(),
                                "learned_magnitude": value.abs().item(),
                                "learned_phase_rad": value.angle().item()})
    output_dir.mkdir(parents=True, exist_ok=True)
    return {name: _write_csv(output_dir / f"{name}.csv", rows) for name, rows in (
        ("subject_scores", subject_rows), ("region_scores", region_rows),
        ("training_history", history_rows), ("filter_values", filter_rows),
    )}


def _write_nested_report(run_dir, results, verified_rows):
    """Write tables and a report from checked nested-evaluation results."""
    run_dir = Path(run_dir)
    outer_rows = [{
        "outer_fold": row["outer_fold"],
        "subject_id": row["subject_id"],
        "selected_candidate_id": row["selected_candidate_id"],
        "n_regions": row["n_regions"],
        "analytical_r": row["analytical_r"],
        "selected_mlp_r": row["learned_r"],
        "raw_r_difference": row["raw_r_difference"],
        "prediction_path": row["prediction_path"],
    } for row in verified_rows]
    summary_metrics = summarize_outer_scores(
        verified_rows, [candidate["candidate_id"] for candidate in results["candidates"]]
    )
    candidate_rows, inner_rows, history_rows = [], [], []
    for outer in results["outer_folds"]:
        for candidate in results["candidates"]:
            candidate_id = candidate["candidate_id"]
            selected = candidate_id == outer["selected_candidate_id"]
            candidate_rows.append({
                "outer_fold": outer["outer_fold"],
                "candidate_id": candidate_id,
                "selected": selected,
                "mean_inner_raw_r": outer["candidate_mean_raw_r"][candidate_id],
                "model_config": json.dumps(candidate["model_config"], sort_keys=True),
                "training_config": json.dumps(candidate["training_config"], sort_keys=True),
            })
            inner_fold_for_subject = {
                subject_id: inner["inner_fold"]
                for inner in outer["inner_splits"]
                for subject_id in inner["validation_subject_ids"]
            }
            inner_rows.extend({
                "outer_fold": outer["outer_fold"],
                "inner_fold": inner_fold_for_subject[row["subject_id"]],
                "candidate_id": candidate_id,
                "subject_id": row["subject_id"],
                "validation_r": row["r"],
                "selected": selected,
            } for row in outer["candidate_scores"][candidate_id])
        history_rows.extend({
            "outer_fold": outer["outer_fold"], **entry
        } for entry in outer["outer_refit"]["training_history"])

    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "outer_subject_scores": _write_csv(tables_dir / "outer_subject_scores.csv", outer_rows),
        "candidate_selection": _write_csv(tables_dir / "candidate_selection.csv", candidate_rows),
        "inner_subject_scores": _write_csv(tables_dir / "inner_subject_scores.csv", inner_rows),
        "outer_refit_history": _write_csv(tables_dir / "outer_refit_history.csv", history_rows),
    }
    selected_counts = summary_metrics["selected_candidate_counts"]
    lines = [
        "# Nested evaluation summary", "",
        f"Run: `{run_dir.name}`.", "",
        f"Outer evaluation: {len(outer_rows)} subject predictions across {len(results['outer_folds'])} folds.",
        f"Inner selection: {results['protocol']['inner_folds']} balanced subject folds per outer fold.",
        "Candidates are ranked by the pooled mean raw subject spectral correlation. "
        "Each subject has equal weight. The first declared candidate wins an exact tie.", "",
        "| Outer measure | Value |", "|---|---:|",
        f"| Mean analytical r | {summary_metrics['mean_analytical_r']:.6f} |",
        f"| Mean selected MLP r | {summary_metrics['mean_selected_mlp_r']:.6f} |",
        f"| Mean raw-r difference | {summary_metrics['mean_raw_r_difference']:.6f} |", "",
        "## Selected configurations", "",
        "| Candidate | Outer subjects |", "|---|---:|",
        *[f"| `{candidate_id}` | {count} |" for candidate_id, count in selected_counts.items()], "",
        "Outer predictions are evaluated once after selection and fresh refitting. "
        "Inner scores are selection evidence; outer scores are evaluation evidence.",
        "The cohort has already informed prior model choices, and fitted SGM parameters "
        "come from MEG, so nested evaluation does not create an untouched cohort or "
        "establish structure-only prediction.", "",
        "Across-subject means weight subjects equally. Correlations describe spectral shape, "
        "not calibrated absolute power.", "",
    ]
    if results.get("statistics"):
        lines.extend(["## Declared comparisons", ""])
        for item in results["statistics"]:
            pvalue = item.get("p_value_unadjusted")
            value = "undefined" if pvalue is None else f"{pvalue:.6g}"
            lines.append(
                f"- `{item['contrast']}`: n={item['n_subject_pairs']}, "
                f"t({item['degrees_of_freedom']})={item['t_statistic']}, "
                f"unadjusted p={value}; exploratory because outer training sets overlap."
            )
        lines.append("")
    lines.extend([
        "## Saved tables", "",
        "- [Outer subject scores](tables/outer_subject_scores.csv)",
        "- [Candidate selection and inner means](tables/candidate_selection.csv)",
        "- [Inner subject scores](tables/inner_subject_scores.csv)",
        "- [Outer-refit training histories](tables/outer_refit_history.csv)",
        "- [Versioned result record](nested_results.json)",
        "- [Configuration and provenance](config.json)", "",
    ])
    summary = run_dir / "summary.md"
    summary.write_text("\n".join(lines), encoding="utf-8")
    return {"tables": tables, "figures": [], "summary": summary}



def _write_loocv_report(run_dir, config, results, cache):
    """Write tables, figures, and a report from loaded LOOCV results."""
    tables = export_results(results, cache, run_dir / "tables")
    figures = plot_results(results, cache, run_dir / "figures")
    analytical = np.array([fold["analytical_correlation"] for fold in results["folds"]])
    learned = np.array([fold["learned_correlation"] for fold in results["folds"]])
    difference = learned - analytical
    training = config["training_config"]
    lines = ["# Run summary", "",
             f"Run: `{run_dir.name}`.", "",
             f"LOOCV: {len(difference)} subjects; {training['epochs']} fixed epochs; seed {training['seed']}.",
             f"Model settings: `{json.dumps(config['model_config'], sort_keys=True)}`. "
             f"Adam learning rate: {training['learning_rate']}.", "",
             "| Held-out measure | Value |", "|---|---:|",
             f"| Mean analytical r | {analytical.mean():.6f} |",
             f"| Mean MLP r | {learned.mean():.6f} |",
             f"| Mean difference (MLP − analytical) | {difference.mean():.6f} |",
             f"| Median difference | {np.median(difference):.6f} |",
             f"| Subjects improved | {int((difference > 0).sum())}/{len(difference)} |", "",
             "Each score is the mean spectral correlation over the subject's observed regions. "
             "The seven fitted SGM parameters and local analytical filter stay fixed. "
             "This evaluates filter transfer conditional on those supplied fitted parameters.", "",
             "## Figures", "",
             "- [Held-out scores](figures/summary/held_out_scores.png): "
             "A, paired scores; B, identity comparison with equal axes; C, subject differences.",
             "- [Training curves](figures/summary/training_curves.png): "
             "gray fold curves and black mean; loss is measured before each optimizer step.",
             "- [Across-subject spectra](figures/summary/spectra_across_subjects.png): "
             "normalize each regional spectrum, average regions within each subject, then show "
             "the mean ± one sample SD across subjects (equal subject weights).",
             "- [Regional spectra](figures/summary/spectra_by_region.png): "
             "each observed region separately, with mean ± one sample SD across subjects. "
             "Both summaries show MEG, analytical and held-out MLP shapes, not absolute power; "
             "bands describe subject variation, not confidence intervals.",
             "- [Subject figures](figures/subjects/): spectra and filter magnitude/phase for every subject. "
             "Spectra use the first, middle and last observed rows (zero-based). "
             "Magnitude, smoothing, square root, demeaning and unit norm produce the displayed shape; "
             "these are not absolute power spectra. Filter phase is unwrapped, and the analytical "
             "filter uses the held-out subject's fixed tau_e.", "",
             "## Saved data", "",
             "- [Subject scores](tables/subject_scores.csv)",
             "- [Region scores and model-node indices](tables/region_scores.csv)",
             "- [Training history](tables/training_history.csv)",
             "- [Complex filter values](tables/filter_values.csv)",
             "- [Configuration and original training-source hashes](config.json)", ""]
    summary = run_dir / "summary.md"
    summary.write_text("\n".join(lines), encoding="utf-8")
    return {"tables": tables, "figures": figures, "summary": summary}


def analyze_run(run_dir):
    """Load and check one saved run, then write its analysis outputs."""
    run_dir = Path(run_dir)
    config = load_run_config(run_dir)
    if config.get("evaluation") == "nested_cross_validation":
        results, verified_rows = load_nested_run(run_dir, config)
        return _write_nested_report(run_dir, results, verified_rows)
    results, cache = load_loocv_run(run_dir, config)
    return _write_loocv_report(run_dir, config, results, cache)

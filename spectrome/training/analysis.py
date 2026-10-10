"""Export and plot saved held-out results without retraining any models."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .filters import transform_spectra


ANALYTICAL_COLOR = "#27618D"
LEARNED_COLOR = "#CC7A29"
OBSERVED_COLOR = "#30343B"


def _load_inputs(results, cache):
    """Match every held-out prediction to its cached empirical subject."""
    if not isinstance(results, dict):
        results = torch.load(results, map_location="cpu", weights_only=True)
    if not isinstance(cache, dict):
        cache = torch.load(cache, map_location="cpu", weights_only=True)
    if results.get("evaluation") != "leave_one_subject_out" or cache.get("format_version") != 2:
        raise ValueError("Analysis requires LOOCV results and a version 2 cohort cache")
    if (results["preprocessing_id"] != cache["preprocessing_id"]
            or results["forward_sha256"] != cache["forward_sha256"]
            or not torch.equal(results["frequencies_hz"], cache["frequencies_hz"])):
        raise ValueError("Results and cache have different preprocessing, forward source, or frequencies")
    subjects = {subject["subject_id"]: subject for subject in cache["subjects"]}
    ids = [fold["held_out_subject_id"] for fold in results["folds"]]
    if (len(subjects) != len(cache["subjects"]) or len(ids) != len(set(ids))
            or set(ids) != set(subjects)):
        raise ValueError("Results must contain exactly one held-out fold per cached subject")
    for fold in results["folds"]:
        subject = subjects[fold["held_out_subject_id"]]
        training_ids = fold["training_subject_ids"]
        if (len(training_ids) != len(subjects) - 1
                or set(training_ids) != set(subjects) - {subject["subject_id"]}):
            raise ValueError("Fold training IDs do not match leave-one-subject-out membership")
        shape = subject["empirical_meg"].shape
        if (fold["analytical_prediction"].shape != shape
                or fold["learned_prediction"].shape != shape
                or fold["Fe_graph"].shape != cache["frequencies_hz"].shape):
            raise ValueError("Saved prediction/filter dimensions do not match the cache")
    return results, cache, subjects


def _normalized_spectra(values):
    transformed = transform_spectra(values)
    norms = transformed.norm(dim=-1, keepdim=True)
    if not torch.isfinite(transformed).all() or (norms <= 1e-12).any():
        raise ValueError("Analysis requires finite, nonconstant regional spectra")
    return transformed / norms


def _region_scores(prediction, target):
    return (_normalized_spectra(prediction) * _normalized_spectra(target)).sum(dim=-1)


def _analytical_filter(subject, frequencies):
    tau_e = subject["parameters"]["tau_e"]
    omega = 2 * torch.pi * frequencies
    return (1 / tau_e**2) / (1j * omega + 1 / tau_e)**2


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


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


@torch.no_grad()
def plot_results(results, cache, output_dir, *, region_rows=None):
    """Save PNG/PDF diagnostics for every subject, without outcome selection.

    Spectrum overlays use the loss transformation followed by unit-norm
    normalization, matching Pearson correlation's shape comparison. Default
    example regions are the first, middle, and last observed rows in every fold.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results, cache, subjects = _load_inputs(results, cache)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    def save(figure, relative_path):
        path = output_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        for extension in ("png", "pdf"):
            target = path.with_suffix(f".{extension}")
            figure.savefig(target, dpi=160, bbox_inches="tight")
            paths.append(target)
        plt.close(figure)

    style = {"font.family": "DejaVu Sans", "font.size": 10,
             "axes.spines.top": False, "axes.spines.right": False,
             "axes.labelcolor": OBSERVED_COLOR, "text.color": OBSERVED_COLOR,
             "axes.titleweight": "semibold", "axes.grid": True,
             "grid.alpha": 0.15, "grid.color": "#808080"}
    with plt.rc_context(style):
        folds = results["folds"]
        figure, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        for fold in folds:
            ax.plot([row["epoch"] for row in fold["history"]],
                    [row["train_loss_before_step"] for row in fold["history"]],
                    color=ANALYTICAL_COLOR, alpha=0.35, linewidth=1)
        ax.set(title=f"Training loss across {len(folds)} folds", xlabel="Epoch",
               ylabel="1 − mean training-subject spectral correlation")
        figure.suptitle("Each curve is one fold; losses are measured before the optimizer step", fontsize=10)
        save(figure, "training_curves")

        figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
        analytical = np.array([fold["analytical_correlation"] for fold in folds])
        learned = np.array([fold["learned_correlation"] for fold in folds])
        axes[0].scatter(analytical, learned, color=LEARNED_COLOR, edgecolor=OBSERVED_COLOR)
        axes[0].plot([-1, 1], [-1, 1], color="#808080", linestyle="--", linewidth=1)
        axes[0].set(xlim=(-1, 1), ylim=(-1, 1), xlabel="Analytical spectral correlation",
                    ylabel="Learned spectral correlation", title="Held-out subject scores")
        axes[0].set_aspect("equal", adjustable="box")
        axes[1].scatter(np.arange(1, len(folds) + 1), learned - analytical,
                        color=LEARNED_COLOR, edgecolor=OBSERVED_COLOR)
        axes[1].axhline(0, color="#808080", linestyle="--", linewidth=1)
        axes[1].set(xlabel="Fold (source subject order)", ylabel="Learned − analytical correlation",
                    title="Paired differences")
        figure.suptitle(f"One held-out score per subject · n = {len(folds)}", fontsize=11)
        save(figure, "held_out_scores")

        frequencies = cache["frequencies_hz"].numpy()
        for fold in folds:
            subject = subjects[fold["held_out_subject_id"]]
            n_regions = subject["observed_regions"].numel()
            rows = list(region_rows) if region_rows is not None else sorted({0, n_regions // 2, n_regions - 1})
            if not rows or any(not isinstance(row, int) or row < 0 or row >= n_regions for row in rows):
                raise ValueError("region_rows must be nonempty valid zero-based observed rows")
            figure, axes = plt.subplots(1, len(rows), figsize=(4 * len(rows), 4),
                                        squeeze=False, layout="constrained")
            series = [("MEG", subject["empirical_meg"], OBSERVED_COLOR, ":"),
                      ("Analytical", fold["analytical_prediction"], ANALYTICAL_COLOR, "--"),
                      ("Learned", fold["learned_prediction"], LEARNED_COLOR, "-")]
            for label, values, color, linestyle in series:
                shapes = _normalized_spectra(values).numpy()
                for axis, row in zip(axes[0], rows):
                    axis.plot(frequencies, shapes[row], label=label, color=color, linestyle=linestyle)
            for axis, row in zip(axes[0], rows):
                node = subject["observed_regions"][row].item()
                axis.set(title=f"Observed row {row} · model node {node}", xlabel="Frequency (Hz)",
                         ylabel="Normalized spectral shape")
            axes[0, -1].legend(frameon=False, fontsize=9)
            figure.suptitle(f"Subject {subject['subject_id']} · held-out fold {fold['fold']}\n"
                           "Magnitude → smoothing → square root → demeaning → unit norm", fontsize=11)
            save(figure, f"spectra/fold-{fold['fold']:03d}")

            figure, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
            baseline = _analytical_filter(subject, cache["frequencies_hz"]).numpy()
            for label, values, color, linestyle in (
                ("Analytical", baseline, ANALYTICAL_COLOR, "--"),
                ("Learned", fold["Fe_graph"].numpy(), LEARNED_COLOR, "-"),
            ):
                axes[0].plot(frequencies, np.abs(values), label=label, color=color, linestyle=linestyle)
                axes[1].plot(frequencies, np.unwrap(np.angle(values)), label=label, color=color, linestyle=linestyle)
            axes[0].set(xlabel="Frequency (Hz)", ylabel="Filter magnitude", title="Graph-filter magnitude")
            axes[1].set(xlabel="Frequency (Hz)", ylabel="Unwrapped phase (rad)", title="Graph-filter phase")
            axes[1].legend(frameon=False)
            figure.suptitle(f"Fold {fold['fold']} · analytical control uses held-out subject's fixed tau_e", fontsize=11)
            save(figure, f"filters/fold-{fold['fold']:03d}")
    return paths


def analyze_run(run_dir):
    """Regenerate this run's tables/figures using its local results and cache."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    cache_path = run_dir / config["cache_path"]
    if hashlib.sha256(cache_path.read_bytes()).hexdigest() != config["cache_sha256"]:
        raise ValueError("Run cache hash differs from config.json")
    results, cache, _ = _load_inputs(run_dir / config["results_path"], cache_path)
    tables = export_results(results, cache, run_dir / "tables")
    figures = plot_results(results, cache, run_dir / "figures")
    return {"tables": tables, "figures": figures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    analyze_run(args.run_dir)


if __name__ == "__main__":
    main()

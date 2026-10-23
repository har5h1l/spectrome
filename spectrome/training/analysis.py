"""Export and plot saved held-out results without retraining any models."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .spectra import transform_spectra


ANALYTICAL_COLOR = "C0"
LEARNED_COLOR = "C1"
OBSERVED_COLOR = "black"


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


@torch.no_grad()
def plot_results(results, cache, output_dir, *, region_rows=None):
    """Save 300-dpi PNG diagnostics for every subject, without outcome selection.

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
        target = path.with_suffix(".png")
        figure.savefig(target, dpi=300, bbox_inches="tight")
        paths.append(target)
        plt.close(figure)

    def panels(axes):
        for index, axis in enumerate(np.asarray(axes).ravel()):
            axis.text(-0.15, 1.04, chr(65 + index), transform=axis.transAxes,
                      fontweight="bold", va="bottom")

    # Start from Matplotlib defaults so notebook/user themes cannot leak in.
    with plt.rc_context(matplotlib.rcParamsDefault):
        plt.rcParams.update({"font.size": 9, "axes.titlesize": 9,
                             "axes.labelsize": 9, "legend.fontsize": 8,
                             "lines.linewidth": 1.2})
        folds = results["folds"]
        figure, ax = plt.subplots(figsize=(5, 3.2), layout="constrained")
        histories = []
        for fold in folds:
            epochs = [row["epoch"] for row in fold["history"]]
            losses = [row["train_loss_before_step"] for row in fold["history"]]
            ax.plot(epochs, losses, color="0.7", alpha=0.6, linewidth=0.6)
            histories.append(losses)
        if all([row["epoch"] for row in fold["history"]] == epochs for fold in folds):
            ax.plot(epochs, np.mean(histories, axis=0), color="black", label="Mean")
        ax.set(xlabel="Epoch", ylabel="Training loss (1 − r)")
        ax.legend(frameon=False)
        save(figure, "summary/training_curves")

        analytical = np.array([fold["analytical_correlation"] for fold in folds])
        learned = np.array([fold["learned_correlation"] for fold in folds])
        figure, axes = plt.subplots(1, 3, figsize=(9, 3.1), layout="constrained")
        for before, after in zip(analytical, learned):
            axes[0].plot([0, 1], [before, after], color="0.75", linewidth=0.6)
        axes[0].scatter(np.zeros(len(folds)), analytical, color=ANALYTICAL_COLOR, s=12, zorder=3)
        axes[0].scatter(np.ones(len(folds)), learned, color=LEARNED_COLOR, s=12, zorder=3)
        axes[0].set(xticks=[0, 1], xticklabels=["Analytical", "MLP"],
                    xlim=(-0.3, 1.3), ylabel="Spectral correlation, r")
        lower, upper = min(analytical.min(), learned.min()), max(analytical.max(), learned.max())
        padding = max((upper - lower) * 0.1, 0.01)
        limits = (max(-1, lower - padding), min(1, upper + padding))
        axes[1].scatter(analytical, learned, color="black", s=14)
        axes[1].plot(limits, limits, color="0.5", linestyle="--", linewidth=0.8)
        axes[1].set(xlim=limits, ylim=limits, xlabel="Analytical r", ylabel="MLP r")
        axes[1].set_aspect("equal", adjustable="box")
        axes[2].scatter(np.arange(1, len(folds) + 1), learned - analytical, color="black", s=14)
        axes[2].axhline(0, color="0.5", linestyle="--", linewidth=0.8)
        axes[2].set(xlabel="Subject (source order)", ylabel="Δr (MLP − analytical)")
        panels(axes)
        save(figure, "summary/held_out_scores")

        frequencies = cache["frequencies_hz"].numpy()
        # Normalize each region before either averaging operation. The subject
        # axis remains the sampling unit for both standard-deviation bands.
        nodes = subjects[folds[0]["held_out_subject_id"]]["observed_regions"]
        if any(not torch.equal(subjects[f["held_out_subject_id"]]["observed_regions"], nodes)
               for f in folds):
            raise ValueError("Across-subject spectra require identical observed region order")
        if len(folds) < 2:
            raise ValueError("Across-subject sample SD requires at least two subjects")
        cohort_series = []
        for label, key, color, linestyle in (
            ("MEG", "empirical_meg", OBSERVED_COLOR, ":"),
            ("Analytical", "analytical_prediction", ANALYTICAL_COLOR, "--"),
            ("MLP", "learned_prediction", LEARNED_COLOR, "-"),
        ):
            values = torch.stack([
                _normalized_spectra(subjects[f["held_out_subject_id"]][key]
                                    if label == "MEG" else f[key]) for f in folds
            ]).numpy()  # subjects x regions x frequencies
            cohort_series.append((label, values, color, linestyle))

        def draw_subject_summary(axis, values, label, color, linestyle):
            mean = values.mean(axis=0)
            sd = values.std(axis=0, ddof=1)
            axis.fill_between(frequencies, mean - sd, mean + sd, color=color, alpha=0.12)
            axis.plot(frequencies, mean, label=label, color=color, linestyle=linestyle)

        figure, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        for label, values, color, linestyle in cohort_series:
            draw_subject_summary(ax, values.mean(axis=1), label, color, linestyle)
        ax.set(xlabel="Frequency (Hz)", ylabel="Normalized spectral shape",
               title=f"Region-averaged spectra across {len(folds)} held-out subjects\n"
                     "Mean ± 1 sample SD across subjects")
        ax.legend(frameon=False)
        save(figure, "summary/spectra_across_subjects")

        n_regions = len(nodes)
        ncols = min(4, n_regions)
        nrows = (n_regions + ncols - 1) // ncols
        figure, axes = plt.subplots(nrows, ncols, figsize=(12, 1.9 * nrows),
                                    sharex=True, sharey=True, squeeze=False)
        for row, axis in enumerate(axes.ravel()):
            if row >= n_regions:
                axis.set_visible(False)
                continue
            for label, values, color, linestyle in cohort_series:
                draw_subject_summary(axis, values[:, row, :], label, color, linestyle)
            axis.set_title(f"Region row {row} · node {nodes[row].item()}")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.suptitle(f"Regional spectra across {len(folds)} held-out subjects\n"
                        "Mean ± 1 sample SD across subjects; region and node indices are zero-based",
                        y=0.995)
        figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.978),
                      ncol=3, frameon=False)
        figure.supxlabel("Frequency (Hz)")
        figure.supylabel("Normalized spectral shape")
        figure.tight_layout(rect=(0.025, 0.015, 1, 0.955))
        save(figure, "summary/spectra_by_region")

        for fold in folds:
            subject = subjects[fold["held_out_subject_id"]]
            subject_id = subject["subject_id"]
            # IDs become directory names; reject paths rather than sanitize ambiguously.
            if Path(subject_id).name != subject_id or subject_id in (".", "..") or "\\" in subject_id:
                raise ValueError("Subject ID must be a safe directory name")
            folder = f"subjects/subject-{subject_id}"
            n_regions = subject["observed_regions"].numel()
            rows = list(region_rows) if region_rows is not None else sorted({0, n_regions // 2, n_regions - 1})
            if not rows or any(not isinstance(row, int) or row < 0 or row >= n_regions for row in rows):
                raise ValueError("region_rows must be nonempty valid zero-based observed rows")
            figure, axes = plt.subplots(1, len(rows), figsize=(3 * len(rows), 2.8),
                                        sharex=True, sharey=True, squeeze=False, layout="constrained")
            series = [("MEG", subject["empirical_meg"], OBSERVED_COLOR, ":"),
                      ("Analytical", fold["analytical_prediction"], ANALYTICAL_COLOR, "--"),
                      ("MLP", fold["learned_prediction"], LEARNED_COLOR, "-")]
            for label, values, color, linestyle in series:
                shapes = _normalized_spectra(values).numpy()
                for axis, row in zip(axes[0], rows):
                    axis.plot(frequencies, shapes[row], label=label, color=color, linestyle=linestyle)
            for axis, row in zip(axes[0], rows):
                axis.set(title=f"Region row {row}", xlabel="Frequency (Hz)")
            axes[0, 0].set_ylabel("Normalized spectral shape")
            axes[0, -1].legend(frameon=False)
            panels(axes)
            save(figure, f"{folder}/spectra")

            figure, axes = plt.subplots(1, 2, figsize=(6, 2.8), sharex=True, layout="constrained")
            baseline = _analytical_filter(subject, cache["frequencies_hz"]).numpy()
            for label, values, color, linestyle in (
                ("Analytical", baseline, ANALYTICAL_COLOR, "--"),
                ("MLP", fold["Fe_graph"].numpy(), LEARNED_COLOR, "-"),
            ):
                axes[0].plot(frequencies, np.abs(values), label=label, color=color, linestyle=linestyle)
                axes[1].plot(frequencies, np.unwrap(np.angle(values)), label=label, color=color, linestyle=linestyle)
            axes[0].set(xlabel="Frequency (Hz)", ylabel="Filter magnitude")
            axes[1].set(xlabel="Frequency (Hz)", ylabel="Phase (rad)")
            axes[1].legend(frameon=False)
            panels(axes)
            save(figure, f"{folder}/filter")
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

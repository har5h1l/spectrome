"""Render spectra, learned filters, training histories, and subject scores."""

from pathlib import Path

import numpy as np
import torch

from .helpers import (
    validate_loocv_results, normalize_spectra, analytical_graph_filter, load_saved_object,
)

ANALYTICAL_COLOR = "C0"
LEARNED_COLOR = "C1"
OBSERVED_COLOR = "black"


@torch.no_grad()
def plot_results(results, cache, output_dir, *, region_rows=None):
    """Save 300-dpi PNG diagnostics for every subject, without outcome selection.

    Spectrum overlays use the loss transformation followed by unit-norm
    normalization, matching Pearson correlation's shape comparison. Default
    example regions are the first, middle, and last observed rows in every fold.
    """
    results, cache, subjects = validate_loocv_results(results, cache)
    histories = [fold["history"] for fold in results["folds"]]
    return _plot_evaluation(results["folds"], histories, cache, subjects, output_dir,
                            region_rows=region_rows)


@torch.no_grad()
def plot_nested_results(run_dir, results, cache, output_dir):
    """Draw checked outer predictions; count each refit once in loss curves.

    The report loader must verify checkpoints and saved predictions first.
    Inner validation scores do not enter these plots.
    """
    subjects = {subject["subject_id"]: subject for subject in cache["subjects"]}
    subject_records, histories = [], []
    for outer in results["outer_folds"]:
        predictions = load_saved_object(Path(run_dir) / outer["prediction_path"])
        histories.append(outer["outer_refit"]["training_history"])
        for row in outer["outer_subject_records"]:
            subject_records.append({
                "held_out_subject_id": row["subject_id"],
                "analytical_correlation": row["analytical_r"],
                "learned_correlation": row["learned_r"],
                **predictions["subject_predictions"][row["subject_id"]],
            })
    # Keep the source cohort order, including when an outer fold has several subjects.
    by_id = {row["held_out_subject_id"]: row for row in subject_records}
    subject_records = [by_id[subject["subject_id"]] for subject in cache["subjects"]]
    return _plot_evaluation(subject_records, histories, cache, subjects, output_dir)


@torch.no_grad()
def _plot_evaluation(subject_records, histories, cache, subjects, output_dir, *, region_rows=None):
    """Share figure styling across fixed and nested held-out evaluations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
        folds = subject_records
        figure, ax = plt.subplots(figsize=(5, 3.2), layout="constrained")
        for history in histories:
            epochs = [row["epoch"] for row in history]
            losses = [row["train_loss_before_step"] for row in history]
            ax.plot(epochs, losses, color="0.7", alpha=0.6, linewidth=0.6,
                    marker="o" if len(epochs) == 1 else None)
        same_epochs = all([row["epoch"] for row in history] == epochs for history in histories)
        if same_epochs:
            ax.plot(epochs, np.mean([[row["train_loss_before_step"] for row in history]
                                    for history in histories], axis=0), color="black", label="Mean",
                    marker="o" if len(epochs) == 1 else None)
        ax.set(xlabel="Epoch", ylabel="Training loss (1 − r)")
        ax.set_title("Outer-refit training loss")
        if all(len(history) == 1 for history in histories):
            ax.set_xticks([1])
        if same_epochs:
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
                normalize_spectra(subjects[f["held_out_subject_id"]][key]
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
                shapes = normalize_spectra(values).numpy()
                for axis, row in zip(axes[0], rows):
                    axis.plot(frequencies, shapes[row], label=label, color=color, linestyle=linestyle)
            for axis, row in zip(axes[0], rows):
                axis.set(title=f"Region row {row}", xlabel="Frequency (Hz)")
            axes[0, 0].set_ylabel("Normalized spectral shape")
            axes[0, -1].legend(frameon=False)
            panels(axes)
            save(figure, f"{folder}/spectra")

            figure, axes = plt.subplots(1, 2, figsize=(6, 2.8), sharex=True, layout="constrained")
            baseline = analytical_graph_filter(subject, cache["frequencies_hz"]).numpy()
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

"""Run saved graph-filter experiments and subject-level LOOCV from Python.

Read run_experiment for the full preparation → training → analysis workflow.
Read train_filter for fold initialization, optimization, evaluation, and saving.
Logging, run records, and validation are in helpers.py; spectral scoring is
in spectra.py. This module has no command-line entry point.
"""

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from zoneinfo import ZoneInfo

import torch

from .analysis import analyze_run
from .data import generate_training_data, prepare_spectral_subject
from .helpers import Tee, create_run_record, save_run_record, validate_training_cache
from .model import FeGraphMLP
from .spectra import predict_spectra, subject_correlation


# Default settings. Each run records the values actually supplied by its caller.
HIDDEN_SIZES = (8,)
OMEGA_SCALE = 2 * math.pi * 45  # rad/s, corresponding to 45 Hz
EPOCHS = 100
LEARNING_RATE = 1e-3
SEED = 7


def run_experiment(run_dir=None, *, data_dir=None, hidden_sizes=HIDDEN_SIZES,
                   omega_scale=OMEGA_SCALE, epochs=EPOCHS,
                   learning_rate=LEARNING_RATE, seed=SEED,
                   subjects=None, frequencies_hz=None, preprocessing_id=None,
                   activation="tanh", cache_source=None):
    """Create a new run folder containing the cache, results, CSVs, and figures.

    Defaults are the accepted 100-epoch, seed-7 settings. Explicit subject
    records/frequencies support bounded verification runs. Refuse an existing
    directory to preserve earlier runs. No notebook is modified or executed.
    """
    if cache_source is not None and any(value is not None for value in
                                         (subjects, frequencies_hz, preprocessing_id, data_dir)):
        raise ValueError("cache_source cannot be combined with data preparation inputs")
    # 1. Create a new run folder and record the supplied settings.
    repo = Path(__file__).resolve().parents[2]
    hidden_sizes = tuple(hidden_sizes)
    if run_dir is None:
        timestamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S_%f")
        run_dir = repo / "results" / "experiments" / f"fe_graph_loocv_{timestamp}_seed{seed}"
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    cache_path = run_dir / "cache" / "cohort_training_v2.pt"
    results_path = run_dir / "results.pt"
    config = create_run_record(
        repo, hidden_sizes=hidden_sizes, omega_scale=omega_scale,
        activation=activation, epochs=epochs, learning_rate=learning_rate, seed=seed,
    )

    save_run_record(run_dir, config)
    with (run_dir / "training.log").open("w", encoding="utf-8") as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                print(f"Run folder: {run_dir}")
                started = time.perf_counter()
                # 2. Prepare or reuse the fixed cohort cache.
                if cache_source is None:
                    cache = generate_training_data(
                        subjects, frequencies_hz, cache_path, data_dir=data_dir,
                        preprocessing_id=preprocessing_id,
                    )
                else:
                    cache = torch.load(cache_source, map_location="cpu", weights_only=True)
                    validate_training_cache(cache)
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(cache_source, cache_path)
                    config["cache_source"] = str(Path(cache_source).resolve())
                config.update({
                    "cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
                    "subject_ids": [subject["subject_id"] for subject in cache["subjects"]],
                    "n_subjects": len(cache["subjects"]),
                    "frequencies_hz": cache["frequencies_hz"].tolist(),
                    "preprocessing_id": cache["preprocessing_id"],
                    "status": "training",
                    "preparation_seconds": time.perf_counter() - started,
                })
                save_run_record(run_dir, config)
                print(f"Prepared {config['n_subjects']} subjects; starting LOOCV")
                started = time.perf_counter()
                # 3. Train and save all LOOCV folds.
                results = train_filter(
                    cache, results_path, checkpoint_dir=run_dir / "checkpoints",
                    hidden_sizes=hidden_sizes, omega_scale=omega_scale,
                    epochs=epochs, learning_rate=learning_rate, seed=seed, activation=activation,
                )
                config["status"] = "analyzing"
                config["training_seconds"] = time.perf_counter() - started
                save_run_record(run_dir, config)
                print("Exporting tables and figures")
                started = time.perf_counter()
                # 4. Export tables and figures from the saved results.
                analyze_run(run_dir)
                config.update({"status": "complete",
                               "analysis_seconds": time.perf_counter() - started,
                               "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                               "mean_analytical_correlation": results["mean_analytical_correlation"],
                               "mean_learned_correlation": results["mean_learned_correlation"],
                               "mean_correlation_difference": results["mean_correlation_difference"]})
                save_run_record(run_dir, config)
                print(f"Complete: {run_dir}")
            except Exception as error:
                config.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
                save_run_record(run_dir, config)
                traceback.print_exc()
                raise
    return run_dir


def train_filter(cache, output_path, *, hidden_sizes, omega_scale,
                 epochs, learning_rate, seed, checkpoint_dir=None, activation="tanh"):
    """Run one leave-one-subject-out fold per cached subject on CPU float64.

    Each fold starts a fresh model and optimizer, trains on the other subjects
    for the fixed epoch budget, and then evaluates the held-out subject once.
    Save fold checkpoints in checkpoint_dir (beside output_path by default),
    save combined results at output_path, and return that results dictionary.
    """
    # 1. Validate inputs and prepare fixed responses and targets once.
    forward_hash = validate_training_cache(cache)
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if not 0 < learning_rate < float("inf"):
        raise ValueError("learning_rate must be finite and positive")
    subjects = cache["subjects"]
    subject_ids = [subject["subject_id"] for subject in subjects]
    if len(subjects) < 2 or len(set(subject_ids)) != len(subject_ids):
        raise ValueError("LOOCV requires at least two subjects with unique IDs")
    subjects = [prepare_spectral_subject(subject) for subject in subjects]
    with torch.no_grad():
        analytical_scores = [subject_correlation(subject).item() for subject in subjects]
    hidden_sizes = tuple(hidden_sizes)
    omega = cache["omega"]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else output_path.parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "format_version": 1,
        "evaluation": "leave_one_subject_out",
        "model_config": {"hidden_sizes": list(hidden_sizes), "omega_scale": float(omega_scale),
                         "activation": activation},
        "training_config": {"epochs": epochs, "learning_rate": learning_rate,
                            "seed": seed, "optimizer": "Adam"},
        "selection_rule": "final epoch of fixed budget; no held-out selection",
        "frequencies_hz": cache["frequencies_hz"],
        "preprocessing_id": cache["preprocessing_id"],
        "loss_preprocessing": "magnitude_conv5_sqrt_floor1e-12_demean",
        "forward_sha256": forward_hash,
        "subjects": [{key: subject[key] for key in (
            "subject_id", "provenance", "observed_regions", "parameters"
        )} for subject in subjects],
        "torch_version": str(torch.__version__),
    }
    folds = []
    for held_out_index, held_out in enumerate(subjects):
        training_subjects = [subject for index, subject in enumerate(subjects)
                             if index != held_out_index]
        # 2. Initialize a fresh model and optimizer for this fold.
        # Use the same predeclared initialization seed in every fold. Both
        # model weights and Adam state are new, with no carryover between folds.
        torch.manual_seed(seed)
        model = FeGraphMLP(hidden_sizes, omega_scale, activation=activation)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        history = []
        # 3. Optimize on training subjects for the fixed epoch budget.
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            filters = model(omega)
            train_scores = torch.stack([
                subject_correlation(subject, filters)
                for subject in training_subjects
            ])
            loss = 1 - train_scores.mean()
            if not torch.isfinite(loss):
                raise ValueError(f"Fold {held_out_index + 1}, epoch {epoch}: nonfinite loss")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError(f"Fold {held_out_index + 1}, epoch {epoch}: missing/nonfinite gradient")
            optimizer.step()
            history.append({"epoch": epoch, "train_loss_before_step": loss.item()})
            print(f"fold={held_out_index + 1}/{len(subjects)} epoch={epoch} "
                  f"train_loss={loss.item():.6f}", flush=True)

        # 4. Evaluate final training and held-out performance.
        model.eval()
        with torch.no_grad():
            filters = model(omega)
            analytical = predict_spectra(held_out)
            learned = predict_spectra(held_out, filters)
            analytical_score = analytical_scores[held_out_index]
            learned_score = subject_correlation(held_out, filters).item()
            analytical_training_score = sum(score for index, score in enumerate(analytical_scores)
                                            if index != held_out_index) / len(training_subjects)
            learned_training_score = torch.stack([
                subject_correlation(subject, filters) for subject in training_subjects]).mean().item()
        # 5. Save this fold before moving to the next held-out subject.
        fold_path = checkpoint_dir / f"{output_path.stem}.fold-{held_out_index + 1:03d}.pt"
        fold = {
            "fold": held_out_index + 1,
            "held_out_subject_id": held_out["subject_id"],
            "training_subject_ids": [subject["subject_id"] for subject in training_subjects],
            "history": history,
            "trained_epochs": epochs,
            "analytical_training_correlation": analytical_training_score,
            "learned_training_correlation": learned_training_score,
            "analytical_correlation": analytical_score,
            "learned_correlation": learned_score,
            "correlation_difference": learned_score - analytical_score,
            "analytical_prediction": analytical.detach().cpu().clone(),
            "learned_prediction": learned.detach().cpu().clone(),
            "Fe_graph": filters.detach().cpu().clone(),
            "checkpoint_path": os.path.relpath(fold_path.resolve(), output_path.resolve().parent),
        }
        state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        torch.save({**metadata, **fold, "state_dict": state}, fold_path)
        folds.append(fold)

    results = {
        **metadata,
        "folds": folds,
        "mean_analytical_correlation": sum(f["analytical_correlation"] for f in folds) / len(folds),
        "mean_learned_correlation": sum(f["learned_correlation"] for f in folds) / len(folds),
        "mean_correlation_difference": sum(f["correlation_difference"] for f in folds) / len(folds),
    }
    torch.save(results, output_path)
    return results

"""Train a fixed configuration with leave-one-subject-out folds."""

import math
import os
from pathlib import Path

import torch

from ..data.data import prepare_spectral_subject
from ..data.helpers import validate_training_cache
from ..spectra import SCORING_ID, evaluate_filter, subject_correlation
from .helpers import save_fold_checkpoint
from .training import train_filter


def run_loocv(cache, output_path, *, hidden_sizes, omega_scale,
                 epochs, learning_rate, seed, checkpoint_dir=None, activation="tanh"):
    """Fit and evaluate a fixed configuration with leave-one-subject-out folds."""
    forward_hash = validate_training_cache(cache)
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if isinstance(learning_rate, bool) or not math.isfinite(float(learning_rate)) or learning_rate <= 0:
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

    model_config = {"hidden_sizes": hidden_sizes, "omega_scale": omega_scale,
                    "activation": activation}
    training_config = {"epochs": epochs, "learning_rate": learning_rate, "seed": seed}
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
        "scoring_id": SCORING_ID,
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

        def report_epoch(entry):
            print(f"fold={held_out_index + 1}/{len(subjects)} epoch={entry['epoch']} "
                  f"train_loss={entry['train_loss_before_step']:.6f}", flush=True)

        model, history = train_filter(
            training_subjects, omega, model_config, training_config,
            on_epoch=report_epoch,
        )
        evaluation = evaluate_filter(model, [held_out], omega)[0]
        with torch.no_grad():
            filters = model(omega)
            analytical_training_score = sum(
                score for index, score in enumerate(analytical_scores)
                if index != held_out_index
            ) / len(training_subjects)
            learned_training_score = torch.stack([
                subject_correlation(subject, filters) for subject in training_subjects
            ]).mean().item()

        fold_path = checkpoint_dir / f"{output_path.stem}.fold-{held_out_index + 1:03d}.pt"
        fold = {
            "fold": held_out_index + 1,
            "held_out_subject_id": held_out["subject_id"],
            "training_subject_ids": [subject["subject_id"] for subject in training_subjects],
            "history": history,
            "trained_epochs": epochs,
            "analytical_training_correlation": analytical_training_score,
            "learned_training_correlation": learned_training_score,
            "analytical_correlation": analytical_scores[held_out_index],
            "learned_correlation": evaluation["learned_correlation"],
            "correlation_difference": (evaluation["learned_correlation"]
                                       - analytical_scores[held_out_index]),
            "analytical_prediction": evaluation["analytical_prediction"],
            "learned_prediction": evaluation["learned_prediction"],
            "Fe_graph": evaluation["Fe_graph"],
            "checkpoint_path": os.path.relpath(fold_path.resolve(), output_path.resolve().parent),
        }
        save_fold_checkpoint(
            fold_path, metadata, fold, model.state_dict()
        )
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

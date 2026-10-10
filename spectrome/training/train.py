"""Train the shared frequency-only FeGraphMLP with subject-level LOOCV."""

import hashlib
import math
from pathlib import Path

import torch

from .filters import FeGraphMLP, predict_spectra, spectral_correlation
from spectrome.forward import network_transfer_torch as nt


# Accepted first-run settings. The epoch budget is fixed in every fold and
# never selected using held-out scores. run.py owns the experiment folders.
HIDDEN_SIZES = (8,)
OMEGA_SCALE = 2 * math.pi * 45  # rad/s, corresponding to 45 Hz
EPOCHS = 100
LEARNING_RATE = 1e-3
SEED = 7


def train_filter(cache, output_path, *, hidden_sizes, omega_scale,
                 epochs, learning_rate, seed, checkpoint_dir=None):
    """Run one leave-one-subject-out fold per cached subject on CPU float64.

    Each fold starts a fresh model and optimizer, trains on the other subjects
    for the fixed epoch budget, and then evaluates the held-out subject once.
    Save fold checkpoints in checkpoint_dir (beside output_path by default),
    save combined results at output_path, and return that results dictionary.
    """
    if cache["purpose"] != "training":
        raise ValueError("Training requires a verified training cache, not historical replay")
    if cache["format_version"] != 2:
        raise ValueError("Regenerate the data as a version 2 cache without fixed splits")
    forward_hash = hashlib.sha256(Path(nt.__file__).read_bytes()).hexdigest()
    if forward_hash != cache["forward_sha256"]:
        raise ValueError("Forward source changed; regenerate the frozen data")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if not 0 < learning_rate < float("inf"):
        raise ValueError("learning_rate must be finite and positive")
    subjects = cache["subjects"]
    subject_ids = [subject["subject_id"] for subject in subjects]
    if len(subjects) < 2 or len(set(subject_ids)) != len(subject_ids):
        raise ValueError("LOOCV requires at least two subjects with unique IDs")
    hidden_sizes = tuple(hidden_sizes)
    omega = cache["omega"]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else output_path.parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "format_version": 1,
        "evaluation": "leave_one_subject_out",
        "model_config": {"hidden_sizes": list(hidden_sizes), "omega_scale": float(omega_scale)},
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
        # Use the same predeclared initialization seed in every fold. Both
        # model weights and Adam state are new, with no carryover between folds.
        torch.manual_seed(seed)
        model = FeGraphMLP(hidden_sizes, omega_scale)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        history = []
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            filters = model(omega)
            train_scores = torch.stack([
                spectral_correlation(predict_spectra(subject, filters), subject["empirical_meg"])
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

        model.eval()
        with torch.no_grad():
            filters = model(omega)
            analytical = predict_spectra(held_out)
            learned = predict_spectra(held_out, filters)
            analytical_score = spectral_correlation(analytical, held_out["empirical_meg"]).item()
            learned_score = spectral_correlation(learned, held_out["empirical_meg"]).item()
        fold_path = checkpoint_dir / f"{output_path.stem}.fold-{held_out_index + 1:03d}.pt"
        fold = {
            "fold": held_out_index + 1,
            "held_out_subject_id": held_out["subject_id"],
            "training_subject_ids": [subject["subject_id"] for subject in training_subjects],
            "history": history,
            "trained_epochs": epochs,
            "analytical_correlation": analytical_score,
            "learned_correlation": learned_score,
            "correlation_difference": learned_score - analytical_score,
            "analytical_prediction": analytical.detach().cpu().clone(),
            "learned_prediction": learned.detach().cpu().clone(),
            "Fe_graph": filters.detach().cpu().clone(),
            "checkpoint_path": str(fold_path.resolve()),
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


def main():
    from .run import main as run_main
    run_main()


if __name__ == "__main__":
    main()

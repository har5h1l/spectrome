"""Train FeGraphMLP using the starter settings below and a verified data cache."""

import hashlib
import math
from pathlib import Path

import torch

from .filters import FeGraphMLP, predict_spectra, spectral_correlation
from spectrome.forward import network_transfer_torch as nt


# Edit these for a first run. They are starter settings, not selected
# experiment settings; training still requires a verified cohort cache.
CACHE_PATH = Path("training_cache.pt")
OUTPUT_PATH = Path("fe_graph_mlp.pt")
HIDDEN_SIZES = (8,)
OMEGA_SCALE = 2 * math.pi * 45  # rad/s, corresponding to a 45 Hz upper frequency
EPOCHS = 100
LEARNING_RATE = 1e-3
SEED = 7


def train_filter(cache, output_path, *, hidden_sizes, omega_scale,
                 epochs, learning_rate, seed):
    """Train with full-cohort Adam steps and select weights by validation loss.

    Final split scores, including the analytical baseline, are computed after
    selection. Runs on CPU in float64; the loop does not recompute eigenpairs.
    """
    if cache["format_version"] != 1 or cache["purpose"] != "training":
        raise ValueError("Training requires a verified training cache, not historical replay")
    forward_hash = hashlib.sha256(Path(nt.__file__).read_bytes()).hexdigest()
    if forward_hash != cache["forward_sha256"]:
        raise ValueError("Forward source changed; regenerate the frozen data")
    if epochs < 1 or not 0 < learning_rate < float("inf"):
        raise ValueError("Supply positive epochs and a finite positive learning rate")
    splits = {split: [s for s in cache["subjects"] if s["split"] == split]
              for split in ("train", "validation", "test")}
    if not splits["train"] or not splits["validation"]:
        raise ValueError("Explicit nonempty train and validation splits are required")
    torch.manual_seed(seed)
    model = FeGraphMLP(hidden_sizes, omega_scale)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    omega = cache["omega"]

    def correlations(subjects, filters):
        return torch.stack([
            spectral_correlation(predict_spectra(subject, filters), subject["empirical_meg"])
            for subject in subjects
        ])

    history, best_validation_loss, best_state, best_epoch = [], float("inf"), None, None
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_scores = correlations(splits["train"], model(omega))
        loss = 1 - train_scores.mean()
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite training loss at epoch {epoch}")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError(f"Missing or nonfinite MLP gradient at epoch {epoch}")
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = 1 - correlations(
                splits["validation"], model(omega)
            ).mean().item()
        history.append({"epoch": epoch,
                        "train_loss_before_step": loss.item(),
                        "validation_loss_after_step": validation_loss})
        if validation_loss < best_validation_loss:
            best_validation_loss, best_epoch = validation_loss, epoch
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        print(
            f"epoch={epoch} train_loss={loss.item():.6f} "
            f"validation_loss={validation_loss:.6f}",
            flush=True,
        )

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        filters = model(omega)
        analytical_scores, learned_scores = {}, {}
        for split, subjects in splits.items():
            if subjects:
                subject_ids = [s["subject_id"] for s in subjects]
                analytical_scores[split] = dict(zip(
                    subject_ids, correlations(subjects, None).tolist()
                ))
                learned_scores[split] = dict(zip(
                    subject_ids, correlations(subjects, filters).tolist()
                ))
    checkpoint = {
        "state_dict": best_state, "model_config": {"hidden_sizes": list(hidden_sizes),
                                                   "omega_scale": float(omega_scale)},
        "training_config": {"epochs": epochs, "learning_rate": learning_rate,
                            "seed": seed, "optimizer": "Adam"},
        "best_epoch": best_epoch,
        "selection_rule": "minimum validation loss (one minus mean subject correlation)",
        "history": history,
        "analytical_correlations": analytical_scores,
        "learned_correlations": learned_scores,
        "frequencies_hz": cache["frequencies_hz"],
        "preprocessing_id": cache["preprocessing_id"],
        "loss_preprocessing": "magnitude_conv5_sqrt_floor1e-12_demean",
        "forward_sha256": forward_hash,
        "subjects": [{key: s[key] for key in ("subject_id", "split", "provenance", "observed_regions", "parameters")}
                     for s in cache["subjects"]],
        "torch_version": str(torch.__version__),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    return model, checkpoint


def main():
    cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=True)
    train_filter(cache, OUTPUT_PATH, hidden_sizes=HIDDEN_SIZES,
                 omega_scale=OMEGA_SCALE, epochs=EPOCHS,
                 learning_rate=LEARNING_RATE, seed=SEED)


if __name__ == "__main__":
    main()

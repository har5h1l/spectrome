"""Train and save FeGraphMLP using explicitly split, frozen SGM data.

Run from the repo root: python -m spectrome.training.train --help
TODO: settle architecture, scaling, split, epochs, and learning rate before
the first cohort experiment. This script does not choose them automatically.
"""

import argparse
import hashlib
from pathlib import Path

import torch

from .filters import FeGraphMLP, predict_spectra, spectral_correlation
from spectrome.forward import network_transfer_torch as nt


def train_filter(cache, output_path, *, hidden_sizes, omega_scale,
                 epochs, learning_rate, seed):
    """Full-cohort Adam steps; validation correlation selects the checkpoint.

    Test subjects are evaluated once after loading the selected weights.
    Runs on CPU in float64; no graph eigendecomposition occurs in this loop.
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

    with torch.no_grad():
        analytical = {
            split: dict(zip([s["subject_id"] for s in subjects],
                            correlations(subjects, None).tolist()))
            for split, subjects in splits.items() if subjects
        }
    history, best_score, best_state, best_epoch = [], -float("inf"), None, None
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
            validation = correlations(splits["validation"], model(omega)).mean().item()
        history.append({"epoch": epoch, "train_loss_before_step": loss.item(),
                        "validation_correlation_after_step": validation})
        if validation > best_score:
            best_score, best_epoch = validation, epoch
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        print(f"epoch={epoch} loss={loss.item():.6f} validation={validation:.6f}", flush=True)

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        filters = model(omega)
        final_scores, predictions = {}, {}
        for split, subjects in splits.items():
            if subjects:
                final_scores[split] = dict(zip([s["subject_id"] for s in subjects],
                                              correlations(subjects, filters).tolist()))
                predictions[split] = {s["subject_id"]: predict_spectra(s, filters)
                                      for s in subjects}
    checkpoint = {
        "state_dict": best_state, "model_config": {"hidden_sizes": list(hidden_sizes),
                                                   "omega_scale": float(omega_scale)},
        "training_config": {"epochs": epochs, "learning_rate": learning_rate,
                            "seed": seed, "optimizer": "Adam"},
        "best_epoch": best_epoch, "selection_rule": "maximum validation mean subject correlation",
        "history": history, "analytical_correlations": analytical,
        "learned_correlations": final_scores, "raw_predictions": predictions,
        "Fe_graph": filters, "frequencies_hz": cache["frequencies_hz"],
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--omega-scale", type=float, required=True, help="Frequency divisor in rad/s")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    train_filter(cache, args.output, hidden_sizes=args.hidden_sizes,
                 omega_scale=args.omega_scale, epochs=args.epochs,
                 learning_rate=args.learning_rate, seed=args.seed)


if __name__ == "__main__":
    main()

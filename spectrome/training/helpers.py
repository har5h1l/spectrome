"""Validation, checkpoint loading, logging, and automatic run records."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from spectrome.forward import network_transfer_torch as nt


def validate_frequencies(frequencies_hz):
    """Copy and validate the frequencies used to prepare the graph cache."""
    frequencies = torch.as_tensor(
        frequencies_hz, dtype=torch.float64
    ).detach().cpu().clone()
    if frequencies.ndim != 1:
        raise ValueError("frequencies_hz must be a one-dimensional sequence")
    if frequencies.numel() < 5:
        raise ValueError("frequencies_hz must contain at least five values")
    if not torch.isfinite(frequencies).all():
        raise ValueError("frequencies_hz must contain only finite values")
    if not (frequencies > 0).all():
        raise ValueError("frequencies_hz must contain only positive values")
    if not (frequencies.diff() > 0).all():
        raise ValueError("frequencies_hz must be strictly increasing")
    return frequencies


def validate_subject_metadata(subject, purpose, seen_ids):
    """Validate one subject's ID, alignment confirmation, and provenance."""
    if not isinstance(subject, dict):
        raise ValueError("Each subject input must be a dictionary")

    subject_id = subject.get("subject_id")
    if not isinstance(subject_id, str) or not subject_id:
        raise ValueError("subject_id must be a nonempty string")
    if subject_id in seen_ids:
        raise ValueError(f"{subject_id}: duplicate subject_id")
    seen_ids.add(subject_id)

    if purpose == "training" and subject.get("mapping_verified") is not True:
        raise ValueError(f"{subject_id}: unverified subject and region alignment")

    provenance = subject.get("provenance")
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError(f"{subject_id}: provenance must include source and alignment details")
    try:
        # Keep saved metadata portable and compatible with weights_only loading.
        provenance = json.loads(json.dumps(provenance))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{subject_id}: provenance must contain JSON-compatible values") from error

    return subject_id, provenance


def validate_parameters(subject_id, raw_parameters):
    """Convert and validate one subject's fixed SGM parameters."""
    parameter_names = ("tau_e", "tau_i", "alpha", "speed", "gei", "gii", "tauC")
    positive_parameters = ("tau_e", "tau_i", "tauC", "alpha", "speed")
    if not isinstance(raw_parameters, dict):
        raise ValueError(f"{subject_id}: parameters must be a dictionary")

    parameters = {}
    for name in parameter_names:
        if name not in raw_parameters:
            raise ValueError(f"{subject_id}: missing SGM parameter '{name}'")
        try:
            value = torch.as_tensor(raw_parameters[name], dtype=torch.float64)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError(f"{subject_id}: SGM parameter '{name}' must be numeric") from error
        if value.numel() != 1:
            raise ValueError(f"{subject_id}: SGM parameter '{name}' must be a scalar")
        value = value.detach().cpu().clone().reshape(())
        if not torch.isfinite(value):
            raise ValueError(f"{subject_id}: SGM parameter '{name}' must be finite")
        if name in positive_parameters and not value > 0:
            raise ValueError(f"{subject_id}: SGM parameter '{name}' must be positive")
        parameters[name] = value
    return parameters


def validate_brain_matrices(subject_id, brain):
    """Check brain matrix inputs and return their tensor views."""
    if brain is None:
        raise ValueError(f"{subject_id}: missing brain with connectivity and distance matrices")
    if not hasattr(brain, "reducedConnectome"):
        raise ValueError(f"{subject_id}: brain is missing reducedConnectome")
    if not hasattr(brain, "distance_matrix"):
        raise ValueError(f"{subject_id}: brain is missing distance_matrix")

    C = torch.as_tensor(brain.reducedConnectome)
    D = torch.as_tensor(brain.distance_matrix)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"{subject_id}: reducedConnectome must be a square matrix")
    if D.shape != C.shape:
        raise ValueError(f"{subject_id}: distance_matrix must match reducedConnectome shape")
    if not torch.isfinite(C).all():
        raise ValueError(f"{subject_id}: reducedConnectome contains nonfinite values")
    if not torch.isfinite(D).all():
        raise ValueError(f"{subject_id}: distance_matrix contains nonfinite values")
    if (C < 0).any():
        raise ValueError(f"{subject_id}: reducedConnectome must be nonnegative")
    if (D < 0).any():
        raise ValueError(f"{subject_id}: distance_matrix must be nonnegative")
    return C, D


def validate_observations(subject, subject_id, n_nodes, n_frequencies):
    """Validate and copy the observed-region map and empirical spectra."""
    try:
        raw_regions = torch.as_tensor(subject["observed_regions"])
    except KeyError as error:
        raise ValueError(f"{subject_id}: missing observed_regions") from error
    if raw_regions.ndim != 1 or raw_regions.numel() == 0:
        raise ValueError(f"{subject_id}: observed_regions must be a nonempty 1-D sequence")
    regions = raw_regions.to(dtype=torch.long).detach().cpu().clone()
    if not torch.equal(raw_regions.cpu(), regions):
        raise ValueError(f"{subject_id}: observed_regions must contain integer indices")
    if regions.unique().numel() != regions.numel():
        raise ValueError(f"{subject_id}: observed_regions contains duplicate indices")
    if (regions < 0).any() or (regions >= n_nodes).any():
        raise ValueError(f"{subject_id}: observed_regions contains indices outside the connectome")

    try:
        target = torch.as_tensor(
            subject["empirical_meg"], dtype=torch.float64
        ).detach().cpu().clone()
    except KeyError as error:
        raise ValueError(f"{subject_id}: missing empirical_meg") from error
    expected_shape = (regions.numel(), n_frequencies)
    if target.shape != expected_shape:
        raise ValueError(
            f"{subject_id}: empirical_meg must have shape {expected_shape}, got {tuple(target.shape)}"
        )
    if not torch.isfinite(target).all():
        raise ValueError(f"{subject_id}: empirical_meg contains nonfinite values")
    if (target < 0).any():
        raise ValueError(f"{subject_id}: empirical_meg must be nonnegative")
    return regions, target


def validate_prepared_graph(graph, subject_id, frequency_hz):
    """Reject nonfinite values in a graph prepared for one frequency."""
    for name, value in graph.items():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise ValueError(
                f"{subject_id}: prepared graph field '{name}' is nonfinite "
                f"at {frequency_hz} Hz"
            )


def validate_training_cache(cache):
    """Reject historical, split, or stale forward caches before reuse."""
    if cache["purpose"] != "training":
        raise ValueError("Training requires a verified training cache, not historical replay")
    if cache["format_version"] != 2:
        raise ValueError("Regenerate the data as a version 2 cache without fixed splits")
    forward_hash = hashlib.sha256(Path(nt.__file__).read_bytes()).hexdigest()
    if forward_hash != cache["forward_sha256"]:
        raise ValueError("Forward source changed; regenerate the frozen data")
    return forward_hash


def load_fold_checkpoint(results_path, fold):
    """Load paths relative to results.pt, or historical absolute paths."""
    path = Path(fold["checkpoint_path"])
    if not path.is_absolute():
        path = Path(results_path).resolve().parent / path
    return torch.load(path, map_location="cpu", weights_only=True)


class Tee:
    """Keep progress visible while retaining it in the run's log."""

    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, value):
        self.console.write(value)
        self.log.write(value)
        self.log.flush()
        return len(value)

    def flush(self):
        self.console.flush()
        self.log.flush()


def create_run_record(repo, *, run_kind, hidden_sizes, omega_scale,
                      activation, epochs, learning_rate, seed):
    """Capture settings and source provenance for a new experiment run."""
    record = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_kind": run_kind, "status": "preparing",
        "cache_path": "cache/cohort_training_v2.pt", "results_path": "results.pt",
        "checkpoint_dir": "checkpoints",
        "model_config": {"hidden_sizes": list(hidden_sizes), "omega_scale": float(omega_scale),
                         "activation": activation},
        "training_config": {"epochs": epochs, "learning_rate": learning_rate, "seed": seed,
                            "optimizer": "Adam", "device": "cpu", "dtype": "float64"},
        "selection_rule": "final epoch of fixed budget; no held-out selection",
        "torch_version": str(torch.__version__), "torch_threads": torch.get_num_threads(),
        "source_sha256": {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in [*(repo / "spectrome/training").glob("*.py"),
                                       repo / "spectrome/forward/network_transfer_torch.py"]},
        "plot_region_policy": "First, middle and last observed rows for every held-out subject",
    }
    # Git metadata is useful when available, but does not prevent use of a
    # downloaded source tree. Exact participating source hashes are saved above.
    try:
        record["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL).strip()
        record["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True, stderr=subprocess.DEVNULL).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        record["git_commit"], record["git_dirty"] = None, None

    return record


def save_run_record(run_dir, record):
    """Atomically replace the run's automatic config.json record."""
    run_dir = Path(run_dir)
    temporary = run_dir / "config.json.tmp"
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(run_dir / "config.json")

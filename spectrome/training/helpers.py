"""Input validation and conversion helpers for graph-filter training."""

import json

import torch


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
    """Validate one subject's ID, alignment evidence, provenance, and split."""
    if not isinstance(subject, dict):
        raise ValueError("Each subject input must be a dictionary")

    subject_id = subject.get("subject_id")
    if not isinstance(subject_id, str) or not subject_id:
        raise ValueError("subject_id must be a nonempty string")
    if subject_id in seen_ids:
        raise ValueError(f"{subject_id}: duplicate subject_id")
    seen_ids.add(subject_id)

    if purpose == "training" and subject.get("mapping_verified") is not True:
        raise ValueError(f"{subject_id}: subject and region mapping are not verified")

    provenance = subject.get("provenance")
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError(f"{subject_id}: provenance must include source and alignment details")
    try:
        # Keep saved metadata portable and compatible with weights_only loading.
        provenance = json.loads(json.dumps(provenance))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{subject_id}: provenance must contain JSON-compatible values") from error

    split = subject.get("split")
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"{subject_id}: split must be 'train', 'validation', or 'test'")
    return subject_id, provenance, split


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

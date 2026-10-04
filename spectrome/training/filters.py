"""Frozen SGM data, a frequency-only MLP, and spectrum prediction.

Graph equations stay in forward/network_transfer_torch.py. This module does
not infer the subject ordering or the structural-to-MEG region mapping.
"""

from pathlib import Path
import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F

from spectrome.forward import network_transfer_torch as nt


@torch.no_grad()
def generate_training_data(subjects, frequencies_hz, output_path, *,
                           preprocessing_id, use_smalleigs=True,
                           purpose="training"):
    """Prepare and save fixed graphs and raw empirical targets on CPU.

    Each subject is a dictionary with subject_id, brain (already in model
    order), parameters, empirical_meg [R, F], observed_regions [R], split
    (train/validation/test), and provenance. Provenance must contain the basis
    for subject/region alignment and source identifiers. Training records must
    explicitly declare mapping_verified=True. Historical replay can be saved
    with purpose='historical_replay'; it is not a corrected training dataset.

    TODO: construct these records from an authoritative real-data manifest.
    Do not zip compacted MEG/parameter arrays with unfiltered connectivity.
    """
    if purpose not in {"training", "historical_replay"}:
        raise ValueError("purpose must be training or historical_replay")
    frequencies = torch.as_tensor(frequencies_hz, dtype=torch.float64).detach().cpu().clone()
    if (frequencies.ndim != 1 or frequencies.numel() < 5
            or not torch.isfinite(frequencies).all()
            or not (frequencies > 0).all()
            or not (frequencies.diff() > 0).all()):
        raise ValueError("Supply at least five finite, positive, increasing frequencies in Hz")
    if not preprocessing_id:
        raise ValueError("preprocessing_id must identify the supplied input preparation")
    records, seen = [], set()
    names = ("tau_e", "tau_i", "alpha", "speed", "gei", "gii", "tauC")
    for subject in subjects:
        subject_id = subject["subject_id"]
        if not isinstance(subject_id, str) or not subject_id or subject_id in seen:
            raise ValueError("subject_id must be a unique nonempty string")
        seen.add(subject_id)
        if purpose == "training" and subject.get("mapping_verified") is not True:
            raise ValueError(f"{subject_id}: subject and region mapping remain unverified")
        provenance = subject["provenance"]
        if not isinstance(provenance, dict) or not provenance:
            raise ValueError(f"{subject_id}: supply source and alignment provenance")
        # Keep saved metadata portable and compatible with weights_only loading.
        provenance = json.loads(json.dumps(provenance))
        split = subject["split"]
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"{subject_id}: split must be train, validation, or test")
        parameters = {
            name: torch.as_tensor(subject["parameters"][name], dtype=torch.float64)
            .detach().cpu().clone().reshape(()) for name in names
        }
        if not all(torch.isfinite(value) for value in parameters.values()):
            raise ValueError(f"{subject_id}: nonfinite SGM parameters")
        if not all(parameters[name] > 0 for name in ("tau_e", "tau_i", "tauC", "alpha", "speed")):
            raise ValueError(f"{subject_id}: time constants, alpha, and speed must be positive")
        brain = subject["brain"]
        C = torch.as_tensor(brain.reducedConnectome)
        D = torch.as_tensor(brain.distance_matrix)
        if (C.ndim != 2 or C.shape[0] != C.shape[1] or D.shape != C.shape
                or not torch.isfinite(C).all() or not torch.isfinite(D).all()
                or (C < 0).any() or (D < 0).any()):
            raise ValueError(f"{subject_id}: C and D must be finite nonnegative square matrices")
        raw_regions = torch.as_tensor(subject["observed_regions"])
        regions = raw_regions.to(dtype=torch.long).detach().cpu().clone()
        target = torch.as_tensor(subject["empirical_meg"], dtype=torch.float64).detach().cpu().clone()
        if (regions.ndim != 1 or regions.numel() == 0
                or not torch.equal(raw_regions.cpu(), regions)
                or regions.unique().numel() != regions.numel()
                or (regions < 0).any() or (regions >= C.shape[0]).any()
                or target.shape != (regions.numel(), frequencies.numel())
                or not torch.isfinite(target).all() or (target < 0).any()):
            raise ValueError(f"{subject_id}: invalid observed-region map or empirical MEG [R, F]")
        graphs = []
        for frequency in frequencies:
            graph = nt.prepare_graph_torch(brain, parameters, 2 * torch.pi * frequency,
                                           use_smalleigs=use_smalleigs)
            graph = {key: value.detach().cpu().clone() if torch.is_tensor(value)
                     else value for key, value in graph.items()}
            if not all(torch.isfinite(value).all() for value in graph.values()
                       if torch.is_tensor(value)):
                raise ValueError(f"{subject_id}: nonfinite prepared graph at {frequency.item()} Hz")
            graphs.append(graph)
        records.append({"subject_id": subject_id, "parameters": parameters,
                        "graphs": graphs, "empirical_meg": target,
                        "observed_regions": regions, "split": split,
                        "provenance": provenance})
    if not records:
        raise ValueError("Supply at least one subject")
    cache = {"format_version": 1, "purpose": purpose,
             "frequencies_hz": frequencies, "omega": 2 * torch.pi * frequencies,
             "preprocessing_id": preprocessing_id, "use_smalleigs": use_smalleigs,
             "forward_sha256": hashlib.sha256(Path(nt.__file__).read_bytes()).hexdigest(),
             "subjects": records}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output_path)
    return cache


class FeGraphMLP(nn.Module):
    """Map omega [F] in rad/s to a shared complex graph filter [F].

    TODO: choose hidden widths and omega_scale for the first experiment.
    They are required inputs, rather than implicit scientific choices.
    Tanh hidden layers and PyTorch's default linear initialization are a
    provisional implementation; this is not analytical-filter initialization.
    """

    def __init__(self, hidden_sizes, omega_scale):
        super().__init__()
        hidden_sizes = tuple(hidden_sizes)
        if not hidden_sizes or any(not isinstance(size, int) or size < 1 for size in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integer widths")
        if not 0 < float(omega_scale) < float("inf"):
            raise ValueError("omega_scale must be finite and positive (rad/s)")
        self.hidden_sizes = hidden_sizes
        self.register_buffer("omega_scale", torch.tensor(float(omega_scale), dtype=torch.float64))
        layers, width = [], 1
        for next_width in hidden_sizes:
            layers.extend([nn.Linear(width, next_width), nn.Tanh()])
            width = next_width
        layers.append(nn.Linear(width, 2))
        self.network = nn.Sequential(*layers).double()

    def forward(self, omega):
        omega = torch.as_tensor(omega, dtype=self.omega_scale.dtype,
                                device=self.omega_scale.device)
        components = self.network((omega / self.omega_scale).unsqueeze(-1))
        return torch.complex(components[..., 0], components[..., 1])


def predict_spectra(subject, Fe_graph=None):
    """Return raw complex regional response [R, F] using cached graphs.

    Fe_graph is a tensor [F] from FeGraphMLP; None uses the subject's analytical
    filter. Only fixed data are moved to its device; filter gradients stay live.
    """
    graphs = subject["graphs"]
    if Fe_graph is not None:
        if not torch.is_tensor(Fe_graph) or Fe_graph.shape != (len(graphs),):
            raise ValueError("Fe_graph must be a tensor with shape [F]")
        device = Fe_graph.device
    else:
        device = graphs[0]["eigenvalues"].device
    parameters = {name: value.to(device) for name, value in subject["parameters"].items()}
    predictions = []
    for index, stored_graph in enumerate(graphs):
        graph = {key: value.to(device) if torch.is_tensor(value) else value
                 for key, value in stored_graph.items()}
        value = None if Fe_graph is None else Fe_graph[index]
        predictions.append(nt.evaluate_response_torch(graph, parameters, value)[3])
    return torch.stack(predictions, dim=-1)[subject["observed_regions"].to(device)]


def transform_spectra(values):
    """Historical magnitude, five-bin smoothing, square root, and demeaning.

    A 1e-12 floor before sqrt gives finite gradients at zero. Record this
    numerical choice separately from exact historical replay.
    """
    magnitude = values.abs()
    kernel = magnitude.new_tensor([1, 2, 5, 2, 1]).reshape(1, 1, 5) / 11
    smoothed = F.conv1d(magnitude.unsqueeze(1), kernel, padding=2).squeeze(1)
    transformed = smoothed.clamp_min(1e-12).sqrt()
    return transformed - transformed.mean(dim=-1, keepdim=True)


def spectral_correlation(prediction, target):
    """Mean regional Pearson correlation over frequency for one subject.

    Fail on nonfinite or near-constant spectra instead of dropping regions.
    """
    x, y = transform_spectra(prediction), transform_spectra(target)
    xnorm, ynorm = x.norm(dim=-1), y.norm(dim=-1)
    if (not torch.isfinite(x).all() or not torch.isfinite(y).all()
            or (xnorm <= 1e-12).any() or (ynorm <= 1e-12).any()):
        raise ValueError("Spectral correlation requires finite, nonconstant regional spectra")
    return ((x * y).sum(dim=-1) / (xnorm * ynorm)).mean()
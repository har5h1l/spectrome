"""Frozen SGM data, a frequency-only MLP, and spectrum prediction.

Graph equations stay in forward/network_transfer_torch.py. This module does
not infer the subject ordering or the structural-to-MEG region mapping.
"""

from pathlib import Path
import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from spectrome.forward import network_transfer_torch as nt
from .helpers import (
    validate_brain_matrices,
    validate_frequencies,
    validate_observations,
    validate_parameters,
    validate_prepared_graph,
    validate_subject_metadata,
)


class FeGraphMLP(nn.Module):
    """Map angular frequencies to shared complex graph-filter values.

    Input:
        omega: [F] angular frequencies in radians per second.

    Output:
        A complex tensor [F], with one Fe_graph value per frequency.
        predict_spectra() uses these values to predict each subject's spectra.
    """

    def __init__(self, hidden_sizes, omega_scale):
        """Initialize the MLP.

        Args:
            hidden_sizes: Units in each hidden layer, e.g. (8,).
            omega_scale: Positive frequency divisor applied to omega.
        """
        super().__init__()
        hidden_sizes = tuple(hidden_sizes)
        self.hidden_sizes = hidden_sizes

        if not hidden_sizes or any(not isinstance(size, int) or size < 1 for size in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integer widths")
        if not 0 < float(omega_scale) < float("inf"):
            raise ValueError("omega_scale must be finite and positive (rad/s)")

        self.register_buffer("omega_scale", torch.tensor(float(omega_scale), dtype=torch.float64))

        # Hidden layers use Tanh; the final layer outputs real and imaginary parts.
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


@torch.no_grad()
def generate_training_data(subjects, frequencies_hz, output_path, *,
                           preprocessing_id, use_smalleigs=True,
                           purpose="training"):
    """Prepare and save fixed graphs and raw empirical targets on CPU.

    ``subjects`` must contain already-paired records with subject IDs, brains,
    parameters, empirical MEG, observed regions, splits, and provenance. This
    function validates those records and prepares a graph for each frequency;
    it does not load or align raw cohort files.

    TODO: build the input records from an authoritative subject/region
    manifest. Do not pair compacted MEG/parameter arrays with unfiltered
    connectivity by position.
    """
    if purpose not in {"training", "historical_replay"}:
        raise ValueError("purpose must be 'training' or 'historical_replay'")
    if not preprocessing_id:
        raise ValueError("preprocessing_id must identify the input preparation")

    frequencies = validate_frequencies(frequencies_hz)
    records, seen_ids = [], set()

    # Validate each subject, prepare its graph at each frequency, and cache it.
    for subject in subjects:
        subject_id, provenance, split = validate_subject_metadata(
            subject, purpose, seen_ids
        )
        parameters = validate_parameters(subject_id, subject.get("parameters"))
        brain = subject.get("brain")
        C, _ = validate_brain_matrices(subject_id, brain)
        regions, target = validate_observations(
            subject, subject_id, C.shape[0], frequencies.numel()
        )

        graphs = []
        for frequency in frequencies:
            graph = nt.prepare_graph_torch(
                brain, parameters, 2 * torch.pi * frequency,
                use_smalleigs=use_smalleigs,
            )
            graph = {
                name: value.detach().cpu().clone() if torch.is_tensor(value) else value
                for name, value in graph.items()
            }
            validate_prepared_graph(graph, subject_id, frequency.item())
            graphs.append(graph)

        records.append({
            "subject_id": subject_id,
            "parameters": parameters,
            "graphs": graphs,
            "empirical_meg": target,
            "observed_regions": regions,
            "split": split,
            "provenance": provenance,
        })

    if not records:
        raise ValueError("subjects must contain at least one subject record")

    cache = {
        "format_version": 1,
        "purpose": purpose,
        "frequencies_hz": frequencies,
        "omega": 2 * torch.pi * frequencies,
        "preprocessing_id": preprocessing_id,
        "use_smalleigs": use_smalleigs,
        "forward_sha256": hashlib.sha256(Path(nt.__file__).read_bytes()).hexdigest(),
        "subjects": records,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output_path)
    return cache


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

"""Regional spectral prediction and the spectral-correlation objective.

The forward module owns the SGM equations. These functions select observed
regions and apply the shared loss transformation.
"""

import torch
from torch.nn import functional as F

from spectrome.forward import network_transfer_torch as nt

SCORING_ID = "mean_regional_pearson_magnitude_conv5_sqrt_floor1e-12_frequency_demean_v1"


def predict_spectra(subject, Fe_graph=None):
    """Return raw complex regional response [R, F] without computing FC.

    Prepared training subjects batch frequencies; raw cached subjects remain
    supported. Only filter gradients are live in the prepared training path.
    """
    graphs = subject["graphs"]
    if Fe_graph is not None and (not torch.is_tensor(Fe_graph)
                                or Fe_graph.shape != (len(graphs),)):
        raise ValueError("Fe_graph must be a tensor with shape [F]")
    if "spectral_response" in subject:
        response = subject["spectral_response"]
        device = response["eigenvalues"].device if Fe_graph is None else Fe_graph.device
        response = {key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in response.items()}
        _, output = nt.evaluate_spectral_response_torch(response, Fe_graph)
        return output.T[subject["observed_regions"].to(device)]
    device = graphs[0]["eigenvalues"].device if Fe_graph is None else Fe_graph.device
    parameters = {name: value.to(device) for name, value in subject["parameters"].items()}
    predictions = []
    for index, stored_graph in enumerate(graphs):
        graph = {key: value.to(device) if torch.is_tensor(value) else value
                 for key, value in stored_graph.items()}
        value = None if Fe_graph is None else Fe_graph[index]
        predictions.append(nt.evaluate_response_torch(graph, parameters, value, compute_fc=False)[3])
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

def spectral_correlation(prediction, target, *, target_norm=None):
    """Mean regional Pearson correlation over frequency for one subject.

    Fail on nonfinite or near-constant spectra instead of dropping regions.
    With target_norm supplied, target must already be transformed and demeaned.
    """
    x = transform_spectra(prediction)
    y = transform_spectra(target) if target_norm is None else target
    xnorm = x.norm(dim=-1)
    ynorm = y.norm(dim=-1) if target_norm is None else target_norm
    if (not torch.isfinite(x).all() or not torch.isfinite(y).all()
            or (xnorm <= 1e-12).any() or (ynorm <= 1e-12).any()):
        raise ValueError("Spectral correlation requires finite, nonconstant regional spectra")
    return ((x * y).sum(dim=-1) / (xnorm * ynorm)).mean()


def subject_correlation(subject, Fe_graph=None):
    """Score one prepared subject using its fixed transformed empirical target.

    prepare_spectral_subject supplies the transformed_target and target_norm.
    Fe_graph selects learned graph propagation; None selects the analytical
    baseline. Return mean regional correlation, preserving filter gradients.
    """
    return spectral_correlation(
        predict_spectra(subject, Fe_graph), subject["transformed_target"],
        target_norm=subject["target_norm"],
    )


@torch.no_grad()
def evaluate_filter(model, subjects, omega):
    """Evaluate one fitted model on supplied subjects and return subject records."""
    if not subjects:
        raise ValueError("subjects must not be empty")
    model.eval()
    filters = model(omega)
    if not torch.isfinite(filters.real).all() or not torch.isfinite(filters.imag).all():
        raise ValueError("evaluate_filter received a nonfinite graph filter")
    records = []
    for subject in subjects:
        analytical = predict_spectra(subject)
        learned = predict_spectra(subject, filters)
        analytical_score = subject_correlation(subject).item()
        learned_score = subject_correlation(subject, filters).item()
        records.append({
            "subject_id": subject["subject_id"],
            "analytical_prediction": analytical.detach().cpu().clone(),
            "learned_prediction": learned.detach().cpu().clone(),
            "Fe_graph": filters.detach().cpu().clone(),
            "analytical_correlation": analytical_score,
            "learned_correlation": learned_score,
            "correlation_difference": learned_score - analytical_score,
        })
    return records

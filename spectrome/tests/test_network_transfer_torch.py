"""Parity and gradient checks for the cohort local-alpha Torch port."""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spectrome.forward.network_transfer import network_transfer_local_alpha
from spectrome.forward.network_transfer_torch import (
    evaluate_response_torch,
    network_transfer_local_alpha_torch,
    prepare_graph_torch,
    run_local_coupling_forward_torch,
)


PARAMETERS = {
    "tau_e": 0.012,
    "tau_i": 0.003,
    "alpha": 0.8,
    "speed": 5.0,
    "gei": 4.0,
    "gii": 1.0,
    "tauC": 0.006,
}
W = 2 * np.pi * 10


def _synthetic_brain(nroi=6):
    rng = np.random.default_rng(100 + nroi)
    C = rng.uniform(0.05, 1, (nroi, nroi))
    C = (C + C.T) / 2
    np.fill_diagonal(C, 0)
    D = rng.uniform(10, 120, (nroi, nroi))
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    return SimpleNamespace(reducedConnectome=C, distance_matrix=D)


@pytest.mark.parametrize("use_smalleigs", [True, False])
def test_local_alpha_torch_matches_numpy_on_synthetic_graph(use_smalleigs):
    brain = _synthetic_brain()
    expected = network_transfer_local_alpha(brain, PARAMETERS, W, use_smalleigs)
    actual = network_transfer_local_alpha_torch(brain, PARAMETERS, W, use_smalleigs)

    assert len(actual) == len(expected) == 5
    for numpy_value, torch_value in zip(expected, actual):
        assert torch_value.dtype == torch.complex128
        np.testing.assert_allclose(torch_value.detach().numpy(), numpy_value, atol=1e-9, rtol=1e-9)


def test_local_alpha_torch_gradient_matches_finite_difference():
    brain = _synthetic_brain()
    C = torch.tensor(brain.reducedConnectome, dtype=torch.float64, requires_grad=True)
    torch_brain = SimpleNamespace(reducedConnectome=C, distance_matrix=brain.distance_matrix)
    model_out = network_transfer_local_alpha_torch(torch_brain, PARAMETERS, W)[3]
    model_out.abs().sum().backward()

    gradient = C.grad[0, 1].item()
    h = 1e-5
    plus = brain.reducedConnectome.copy()
    minus = brain.reducedConnectome.copy()
    plus[0, 1] += h
    minus[0, 1] -= h
    plus_out = network_transfer_local_alpha(
        SimpleNamespace(reducedConnectome=plus, distance_matrix=brain.distance_matrix),
        PARAMETERS,
        W,
    )[3]
    minus_out = network_transfer_local_alpha(
        SimpleNamespace(reducedConnectome=minus, distance_matrix=brain.distance_matrix),
        PARAMETERS,
        W,
    )[3]
    finite_difference = (np.abs(plus_out).sum() - np.abs(minus_out).sum()) / (2 * h)
    assert np.isfinite(gradient)
    np.testing.assert_allclose(gradient, finite_difference, atol=1e-5, rtol=1e-4)


def test_prepared_and_explicit_analytical_filter_match_default():
    brain = _synthetic_brain()
    graph = prepare_graph_torch(brain, PARAMETERS, W)
    analytical = (1 / PARAMETERS["tau_e"]**2) / (1j * W + 1 / PARAMETERS["tau_e"])**2
    expected = network_transfer_local_alpha_torch(brain, PARAMETERS, W)
    for actual in (
        evaluate_response_torch(graph, PARAMETERS),
        evaluate_response_torch(graph, PARAMETERS, analytical),
        network_transfer_local_alpha_torch(brain, PARAMETERS, W, Fe_graph=analytical),
    ):
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference)


def test_replacement_changes_only_graph_response():
    graph = prepare_graph_torch(_synthetic_brain(), PARAMETERS, W)
    filter_value = 0.4 - 0.3j
    response = evaluate_response_torch(graph, PARAMETERS, filter_value)[0]
    # Independent local expression: replacement must not enter these terms.
    alpha, te, ti = PARAMETERS["alpha"], PARAMETERS["tau_e"], PARAMETERS["tau_i"]
    gei, gii, tc = PARAMETERS["gei"], PARAMETERS["gii"], PARAMETERS["tauC"]
    fe = (1 / te**2) / (1j * W + 1 / te)**2
    fi = (gii / ti**2) / (1j * W + 1 / ti)**2
    local = (0.5 * (alpha / te) / (1j * W + alpha / te * fe)
             + 0.25 * (alpha / ti) / (1j * W + alpha / ti * fi)
             + 0.25 * gei * fe * fi / (1 + gei * fe * fi))
    q = (tc / alpha) * (1j * W + alpha / tc * filter_value * graph.eigenvalues)
    q = torch.maximum(q.abs(), 0.05 * q.abs().max()) * torch.exp(1j * q.angle())
    torch.testing.assert_close(response * q, torch.full_like(response, local))
    assert not torch.allclose(response, evaluate_response_torch(graph, PARAMETERS)[0])
    with pytest.raises(ValueError, match="scalar"):
        evaluate_response_torch(graph, PARAMETERS, torch.ones(6))


def test_filter_gradient_through_sweep_matches_finite_difference():
    brain = _synthetic_brain()
    freqs = [8.0, 10.0]
    components = torch.tensor([[0.4, -0.3], [0.3, -0.2]], dtype=torch.float64,
                              requires_grad=True)

    def loss(values):
        filters = torch.complex(values[:, 0], values[:, 1])
        outputs = run_local_coupling_forward_torch(
            brain, PARAMETERS, freqs, graph_filter_values=filters
        )
        assert outputs[0].shape == (6, 2)
        return outputs[0].abs().sum()

    loss(components).backward()
    for index in ((0, 0), (0, 1)):
        plus, minus = components.detach().clone(), components.detach().clone()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        finite_difference = (loss(plus) - loss(minus)) / 2e-6
        torch.testing.assert_close(components.grad[index], finite_difference,
                                   atol=1e-5, rtol=1e-4)
    prepared = [prepare_graph_torch(brain, PARAMETERS, 2 * np.pi * f) for f in freqs]
    filters = torch.complex(components.detach()[:, 0], components.detach()[:, 1])
    cached = torch.stack([
        evaluate_response_torch(graph, PARAMETERS, value)[3]
        for graph, value in zip(prepared, filters)
    ], dim=1)
    uncached = run_local_coupling_forward_torch(
        brain, PARAMETERS, freqs, graph_filter_values=filters
    )[0]
    torch.testing.assert_close(cached, uncached)

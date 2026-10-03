"""Torch counterpart of the cohort notebook's local-alpha SGM forward path."""

import torch


def prepare_graph_torch(brain, parameters, w, use_smalleigs=True):
    """Compute the frequency-specific delayed graph and its eigendecomposition."""
    C = torch.as_tensor(brain.reducedConnectome, dtype=torch.float64)
    D = torch.as_tensor(brain.distance_matrix, dtype=torch.float64, device=C.device)
    w = torch.as_tensor(w, dtype=torch.float64, device=C.device)
    speed = torch.as_tensor(parameters["speed"], dtype=torch.float64, device=C.device)
    alpha = torch.as_tensor(parameters["alpha"], dtype=torch.float64, device=C.device)

    rowdegree = C.sum(dim=1)
    coldegree = C.sum(dim=0)
    low_degree = rowdegree + coldegree < 0.2 * (rowdegree + coldegree).mean()
    rowdegree = torch.where(low_degree, torch.inf, rowdegree)
    coldegree = torch.where(low_degree, torch.inf, coldegree)

    nroi = C.shape[0]
    K = round(2 / 3 * nroi) if use_smalleigs else nroi
    tau = 0.001 * D / speed
    Cc = C * torch.exp(-1j * tau * w)
    degree_scale = 1 / (
        torch.sqrt(rowdegree * coldegree) + torch.finfo(torch.float64).eps
    )
    L = torch.eye(nroi, dtype=torch.complex128, device=C.device) - alpha * (
        torch.diag(degree_scale).to(torch.complex128) @ Cc
    )

    values, vectors = torch.linalg.eig(L)
    order = torch.argsort(torch.abs(values))
    eigenvalues = values[order]
    eigenvectors = vectors[:, order][:, :K]
    anchors = eigenvectors[
        torch.argmax(torch.abs(eigenvectors), dim=0),
        torch.arange(K, device=C.device),
    ]
    eigenvectors = eigenvectors * (anchors / torch.abs(anchors)).conj()[None, :]

    return {
        "w": w,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "K": K,
    }


def evaluate_response_torch(graph, parameters, Fe_graph=None):
    """Evaluate local and graph responses from a prepared graph."""
    device = graph["eigenvalues"].device
    params = {
        name: torch.as_tensor(parameters[name], dtype=torch.float64, device=device)
        for name in ("tau_e", "tau_i", "gei", "gii", "tauC", "alpha")
    }
    tau_e, tau_i = params["tau_e"], params["tau_i"]
    gei, gii, tauC, alpha = params["gei"], params["gii"], params["tauC"], params["alpha"]
    w = graph["w"]

    Fe_local = (1 / tau_e**2) / (1j * w + 1 / tau_e) ** 2
    Fi = (gii / tau_i**2) / (1j * w + 1 / tau_i) ** 2
    Hed = (alpha / tau_e) / (1j * w + alpha / tau_e * Fe_local)
    Hid = (alpha / tau_i) / (1j * w + alpha / tau_i * Fi)
    Heid = gei * Fe_local * Fi / (1 + gei * Fe_local * Fi)
    Htotal = 0.5 * Hed + 0.25 * Hid + 0.25 * Heid

    if Fe_graph is None:
        Fe_graph = Fe_local
    else:
        Fe_graph = torch.as_tensor(Fe_graph, device=device)
        if Fe_graph.numel() != 1:
            raise ValueError("Fe_graph must be a scalar complex value")
        Fe_graph = Fe_graph.reshape(()).to(torch.complex128)

    q1 = (tauC / alpha) * (
        1j * w + alpha / tauC * Fe_graph * graph["eigenvalues"]
    )
    q_threshold = 0.05 * torch.max(torch.abs(q1))
    q1 = torch.maximum(torch.abs(q1), q_threshold) * torch.exp(1j * torch.angle(q1))
    frequency_response = Htotal / q1

    eigenvectors = graph["eigenvectors"]
    K = graph["K"]
    model_out = torch.sum(
        eigenvectors[:, 1:K] * frequency_response[None, 1:K], dim=1
    )
    selected = eigenvectors[:, 1:K]
    FCmodel = selected @ torch.diag(frequency_response[1:K] ** 2) @ selected.T
    inverse_denominator = torch.diag(1 / torch.sqrt(torch.abs(model_out))).to(
        torch.complex128
    )
    FCmodel = inverse_denominator @ FCmodel @ inverse_denominator

    return frequency_response, graph["eigenvalues"], eigenvectors, model_out, FCmodel


def network_transfer_local_alpha_torch(
    brain, parameters, w, use_smalleigs=True, Fe_graph=None
):
    """Single-frequency compatibility entry point."""
    graph = prepare_graph_torch(brain, parameters, w, use_smalleigs)
    return evaluate_response_torch(graph, parameters, Fe_graph)


def run_local_coupling_forward_torch(
    brain, parameters, frequencies, graph_filter_values=None, use_smalleigs=True
):
    """Run the local-coupling forward model over frequencies in Hz."""
    frequencies = list(frequencies)
    if graph_filter_values is None:
        graph_filter_values = [None] * len(frequencies)
    else:
        graph_filter_values = torch.as_tensor(graph_filter_values)
        if graph_filter_values.numel() != len(frequencies):
            raise ValueError("graph_filter_values must contain one value per frequency")
        graph_filter_values = list(graph_filter_values.reshape(-1).unbind())

    outputs = []
    for frequency, graph_filter in zip(frequencies, graph_filter_values):
        w = 2 * torch.pi * torch.as_tensor(frequency, dtype=torch.float64)
        graph = prepare_graph_torch(brain, parameters, w, use_smalleigs)
        outputs.append(evaluate_response_torch(graph, parameters, graph_filter))

    if not outputs:
        raise ValueError("frequencies must contain at least one frequency")
    model_out = torch.stack([output[3] for output in outputs], dim=-1)
    frequency_response = torch.stack([output[0] for output in outputs], dim=0)
    eigenvalues = torch.stack([output[1] for output in outputs], dim=0)
    eigenvectors = torch.stack([output[2] for output in outputs], dim=0)
    return model_out, frequency_response, eigenvalues, eigenvectors

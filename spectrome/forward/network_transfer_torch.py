"""Torch counterpart of the cohort notebook's local-alpha SGM forward path."""

import torch


def prepare_graph_torch(brain, parameters, w, use_smalleigs=True):
    """Prepare eigenpairs at angular frequency ``w`` in radians/second.

    ``brain`` supplies model-order connectivity and distances in millimeters.
    Parameters ``alpha`` and ``speed`` (meters/second) determine the graph.
    Reuse this result only while connectivity, distances, frequency, alpha,
    speed, and mode selection remain fixed. Prepare once per subject/frequency
    for training a filter with frozen SGM parameters.
    """
    # Convert the connectome, distance matrix, and angular frequency to tensors.
    C = torch.as_tensor(brain.reducedConnectome, dtype=torch.float64)
    D = torch.as_tensor(brain.distance_matrix, dtype=torch.float64, device=C.device)
    w = torch.as_tensor(w, dtype=torch.float64, device=C.device)  # w is omega, not Hz

    # Graph parameters use the same device and precision as the connectome.
    speed = torch.as_tensor(parameters["speed"], dtype=torch.float64, device=C.device)
    alpha = torch.as_tensor(parameters["alpha"], dtype=torch.float64, device=C.device)

    # Compute row/column degrees and suppress low-degree nodes in normalization.
    rowdegree = C.sum(dim=1)
    coldegree = C.sum(dim=0)
    low_degree = rowdegree + coldegree < 0.2 * (rowdegree + coldegree).mean()
    # torch.inf makes the corresponding inverse degree normalization zero.
    rowdegree = torch.where(low_degree, torch.inf, rowdegree)
    coldegree = torch.where(low_degree, torch.inf, coldegree)

    nroi = C.shape[0]
    # Truncate to two-thirds of the modes when requested.
    K = round(2 / 3 * nroi) if use_smalleigs is True else nroi

    # Complex normalized graph operator used by the legacy SGM:
    # L = I - alpha * diag(1 / sqrt(rowdegree * coldegree)) @ Cc.
    Tau = 0.001 * D / speed
    Cc = C * torch.exp(-1j * Tau * w)
    degree_scale = 1 / (
        torch.sqrt(rowdegree * coldegree) + torch.finfo(torch.float64).eps
    )
    L = torch.eye(nroi, dtype=torch.complex128, device=C.device) - alpha * (
        torch.diag(degree_scale).to(torch.complex128) @ Cc
    )

    # Eigenvalue decomposition; order modes by increasing absolute eigenvalue.
    values, vectors = torch.linalg.eig(L)
    order = torch.argsort(torch.abs(values))
    eigenvalues = values[order]
    eigenvectors = vectors[:, order][:, :K]
    # Complex eigenvectors are defined only up to a global phase. Fix that
    # phase deterministically for the phase-sensitive legacy modal readout.
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
        # Snapshot values so later in-place parameter changes are detectable.
        "alpha": alpha.detach().clone(),
        "speed": speed.detach().clone(),
    }


def evaluate_response_torch(graph, parameters, Fe_graph=None):
    """Evaluate a prepared graph with the same alpha and speed used to prepare it.

    Parameters are ``tau_e``, ``tau_i``, ``gei``, ``gii``, ``tauC``, ``alpha``,
    and ``speed``; time constants are in seconds. ``Fe_graph`` is one shared
    complex scalar at the prepared frequency; None uses analytical ``Fe_local``.

    Returns modal response [N], eigenvalues [N], eigenvectors [N, K], regional
    response [N], and normalized FC [N, N], all complex128 tensors.
    """
    device = graph["eigenvalues"].device
    for name in ("alpha", "speed"):
        value = torch.as_tensor(parameters[name], dtype=torch.float64, device=device)
        if not torch.equal(value.detach(), graph[name]):
            raise ValueError(f"{name} differs from the prepared graph; prepare it again")
    # Convert local-response parameters to the graph's device.
    params = {
        name: torch.as_tensor(parameters[name], dtype=torch.float64, device=device)
        for name in ("tau_e", "tau_i", "gei", "gii", "tauC", "alpha")
    }
    tau_e, tau_i = params["tau_e"], params["tau_i"]
    gei, gii = params["gei"], params["gii"]
    tauC, alpha = params["tauC"], params["alpha"]
    w = graph["w"]

    zero_thr = 0.05  # Numerical safeguard for the graph denominator.
    a = 0.5  # Fraction of signal at a node that is recurrent excitatory.

    # Analytical local excitatory filter; it remains unchanged in the MLP experiment.
    Fe_local = (1 / tau_e**2) / (1j * w + 1 / tau_e) ** 2
    Fi = (gii / tau_i**2) / (1j * w + 1 / tau_i) ** 2
    Hed = (alpha / tau_e) / (1j * w + alpha / tau_e * Fe_local)
    Hid = (alpha / tau_i) / (1j * w + alpha / tau_i * Fi)
    Heid = gei * Fe_local * Fi / (1 + gei * Fe_local * Fi)
    Htotal = a * Hed + (1 - a) / 2 * Hid + (1 - a) / 2 * Heid

    # Graph-propagation filter defaults to the analytical filter for baseline parity.
    # Later, a frequency-only MLP will supply one shared complex scalar per frequency.
    if Fe_graph is None:
        Fe_graph = Fe_local
    else:
        if torch.is_tensor(Fe_graph):
            Fe_graph = Fe_graph.to(device=device, dtype=torch.complex128)
        else:
            Fe_graph = torch.tensor(
                Fe_graph, dtype=torch.complex128, device=device
            )
        if Fe_graph.numel() != 1:
            raise ValueError("Fe_graph must be one complex scalar per frequency")
        Fe_graph = Fe_graph.reshape(())

    # Keep Fe_local in the local terms; Fe_graph changes only graph propagation.
    q1 = (tauC / alpha) * (
        1j * w + alpha / tauC * Fe_graph * graph["eigenvalues"]
    )
    q_threshold = zero_thr * torch.max(torch.abs(q1))
    q1 = torch.maximum(torch.abs(q1), q_threshold) * torch.exp(1j * torch.angle(q1))
    # Frequency response gives each graph mode's response at this frequency.
    frequency_response = Htotal / q1

    # Preserve the original modal readout and normalized FC calculation.
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
    """Evaluate at ``w`` in radians/second, returning the five response outputs.

    Uses the repository's local-alpha equations and parameter names. An optional
    scalar ``Fe_graph`` replaces only the graph-propagation filter.
    """
    graph = prepare_graph_torch(brain, parameters, w, use_smalleigs)
    return evaluate_response_torch(graph, parameters, Fe_graph)


def run_local_coupling_forward_torch(
    brain, parameters, frequencies, graph_filter_values=None, use_smalleigs=True
):
    """Run the local-coupling forward model over frequencies in Hz.

    ``graph_filter_values`` may be a tensor or a sequence of scalar values or
    tensors, with one shared complex filter value per frequency. Gradients are
    preserved. None selects the analytical baseline.

    Returns regional output [N, F], modal response [F, N], eigenvalues [F, N],
    and eigenvectors [F, N, K], following ``run_local_coupling_forward``.
    This convenience wrapper prepares graphs on every call; repeated training
    evaluations should reuse ``prepare_graph_torch`` results instead.
    """
    frequencies = list(frequencies)
    if not frequencies:
        raise ValueError("frequencies must contain at least one frequency")
    if graph_filter_values is None:
        graph_filter_values = [None] * len(frequencies)
    else:
        if torch.is_tensor(graph_filter_values):
            graph_filter_values = graph_filter_values.to(dtype=torch.complex128)
        else:
            device = torch.as_tensor(brain.reducedConnectome).device
            values = [
                torch.as_tensor(value, dtype=torch.complex128, device=device)
                for value in graph_filter_values
            ]
            graph_filter_values = (
                torch.stack(values) if values else torch.empty(0, dtype=torch.complex128)
            )
        if graph_filter_values.numel() != len(frequencies):
            raise ValueError("graph_filter_values must contain one value per frequency")
        graph_filter_values = list(graph_filter_values.reshape(-1).unbind())

    outputs = []
    for frequency, graph_filter in zip(frequencies, graph_filter_values):
        # Convert Hz to angular frequency, then prepare one graph per frequency.
        w = 2 * torch.pi * torch.as_tensor(frequency, dtype=torch.float64)
        graph = prepare_graph_torch(brain, parameters, w, use_smalleigs)
        outputs.append(evaluate_response_torch(graph, parameters, graph_filter))

    model_out = torch.stack([output[3] for output in outputs], dim=-1)
    frequency_response = torch.stack([output[0] for output in outputs], dim=0)
    eigenvalues = torch.stack([output[1] for output in outputs], dim=0)
    eigenvectors = torch.stack([output[2] for output in outputs], dim=0)
    # Match the NumPy wrapper: regional output is [region, frequency].
    return model_out, frequency_response, eigenvalues, eigenvectors

"""Torch counterpart of the cohort notebook's local-alpha SGM forward path.

The four calculation steps appear in order below: prepare the graph, prepare
local responses, evaluate spectra, and optionally compute functional
connectivity (FC). Training reuses steps 1–2 and repeats step 3. The entry
points after step 4 compose these steps for ordinary forward calls.
"""

import torch


def prepare_graph_torch(brain, parameters, w, use_smalleigs=True):
    """Step 1/4: prepare graph eigenpairs at angular frequency ``w`` in radians/second.

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


def prepare_local_response_torch(graph, parameters):
    """Step 2/4: prepare fixed local responses and denominator coefficients.

    Use the graph from step 1 and the subject's fitted SGM parameters. Return
    analytical Fe_local, Htotal, and the graph data needed by step 3. Reuse
    this result only while the graph and all fitted parameters remain fixed.
    The learned Fe_graph never replaces Fe_local in these local terms.
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

    a = 0.5  # Fraction of signal at a node that is recurrent excitatory.

    # Analytical local excitatory filter; it remains unchanged in the MLP experiment.
    Fe_local = (1 / tau_e**2) / (1j * w + 1 / tau_e) ** 2
    Fi = (gii / tau_i**2) / (1j * w + 1 / tau_i) ** 2
    Hed = (alpha / tau_e) / (1j * w + alpha / tau_e * Fe_local)
    Hid = (alpha / tau_i) / (1j * w + alpha / tau_i * Fi)
    Heid = gei * Fe_local * Fi / (1 + gei * Fe_local * Fi)
    Htotal = a * Hed + (1 - a) / 2 * Hid + (1 - a) / 2 * Heid

    return {"Fe_local": Fe_local, "Htotal": Htotal,
            "tauC_over_alpha": tauC / alpha, "alpha_over_tauC": alpha / tauC,
            "w": w, "eigenvalues": graph["eigenvalues"],
            "eigenvectors": graph["eigenvectors"], "K": graph["K"]}


def evaluate_spectral_response_torch(response, Fe_graph=None):
    """Step 3/4: evaluate modal responses and regional spectra without FC.

    Use the fixed terms from step 2 at one frequency or a frequency batch.
    Fe_graph supplies the shared learned filter; None uses analytical Fe_local.
    Return modal responses [..., N] and regional spectra [..., N]. Gradients
    pass through Fe_graph. The denominator floor uses all eigenvalues at each
    frequency before selecting readout modes.
    """
    value = response["Fe_local"] if Fe_graph is None else torch.as_tensor(
        Fe_graph, dtype=torch.complex128, device=response["eigenvalues"].device)
    q1 = response["tauC_over_alpha"][..., None] * (
        1j * response["w"][..., None]
        + response["alpha_over_tauC"][..., None] * value[..., None]
        * response["eigenvalues"])
    threshold = 0.05 * q1.abs().amax(dim=-1, keepdim=True)
    q1 = torch.maximum(q1.abs(), threshold) * torch.exp(1j * torch.angle(q1))
    frequency_response = response["Htotal"][..., None] / q1
    K = response["K"]
    model_out = (response["eigenvectors"][..., :, 1:K]
                 * frequency_response[..., None, 1:K]).sum(dim=-1)
    return frequency_response, model_out


def compute_fc_torch(graph, frequency_response, model_out):
    """Step 4/4: compute normalized functional connectivity at one frequency.

    Use the graph from step 1 and both outputs of step 3. Return the historical
    complex FC matrix [N, N]. This optional step is omitted during spectral
    training because its loss uses only regional spectra.
    """
    eigenvectors, K = graph["eigenvectors"], graph["K"]
    selected = eigenvectors[:, 1:K]
    FCmodel = selected @ torch.diag(frequency_response[1:K] ** 2) @ selected.T
    inverse_denominator = torch.diag(1 / torch.sqrt(torch.abs(model_out))).to(
        torch.complex128)
    return inverse_denominator @ FCmodel @ inverse_denominator


def evaluate_response_torch(graph, parameters, Fe_graph=None, *, compute_fc=True):
    """Compose steps 2–4 using a graph already prepared in step 1.

    Return the historical five outputs: modal response, eigenvalues,
    eigenvectors, regional response, and FC. With compute_fc=False, omit
    step 4 and return None in the FC position.
    """
    response = prepare_local_response_torch(graph, parameters)
    if Fe_graph is not None and torch.as_tensor(Fe_graph).numel() != 1:
        raise ValueError("Fe_graph must be one complex scalar per frequency")
    if Fe_graph is not None:
        Fe_graph = torch.as_tensor(Fe_graph, dtype=torch.complex128,
                                   device=graph["eigenvalues"].device).reshape(())
    frequency_response, model_out = evaluate_spectral_response_torch(response, Fe_graph)
    FCmodel = compute_fc_torch(graph, frequency_response, model_out) if compute_fc else None
    return frequency_response, graph["eigenvalues"], graph["eigenvectors"], model_out, FCmodel


def network_transfer_local_alpha_torch(
    brain, parameters, w, use_smalleigs=True, Fe_graph=None
):
    """Compose steps 1–4 at ``w`` in radians/second; return five outputs.

    Uses the repository's local-alpha equations and parameter names. An optional
    scalar ``Fe_graph`` replaces only the graph-propagation filter.
    """
    graph = prepare_graph_torch(brain, parameters, w, use_smalleigs)
    return evaluate_response_torch(graph, parameters, Fe_graph)


def run_local_coupling_forward_torch(
    brain, parameters, frequencies, graph_filter_values=None, use_smalleigs=True
):
    """Compose steps 1–4 over a frequency grid in Hz and collect outputs.

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

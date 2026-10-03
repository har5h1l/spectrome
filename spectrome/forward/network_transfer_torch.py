"""Torch counterpart of the cohort notebook's local-alpha SGM forward path."""

import torch


def network_transfer_local_alpha_torch(brain, parameters, w, use_smalleigs=True):
    """Match ``network_transfer_local_alpha`` at one angular frequency.

    ``brain`` supplies ``reducedConnectome`` and ``distance_matrix`` in the
    same node order as the NumPy function. The five return values are Torch
    tensors in the same order as that function. This preserves its modal
    readout and numerical conventions; it does not use the equations in the
    separate deconvolution Torch reference.
    """

    # convert to torhc tensors & set device to match C
    C = torch.as_tensor(brain.reducedConnectome, dtype=torch.float64)
    D = torch.as_tensor(brain.distance_matrix, dtype=torch.float64, device=C.device)
    w = torch.as_tensor(w, dtype=torch.float64, device=C.device) # this is omega = 2(pi)f, not frequency

    # extract parameters and convert to torch tensors on the same device as C
    params = {
        name: torch.as_tensor(parameters[name], dtype=torch.float64, device=C.device)
        for name in ("tau_e", "tau_i", "speed", "gei", "gii", "tauC", "alpha")
    }
    tau_e = params["tau_e"]
    tau_i = params["tau_i"]
    speed = params["speed"]
    gei = params["gei"]
    gii = params["gii"]
    tauC = params["tauC"]
    alpha = params["alpha"]


    zero_thr = 0.05 # numerical safeguard
    a = 0.5 # used as a weight for H_total

    # compute the degree of each node and filter out nodes with low degree; essentially prerprocessing / filtering the connectome beforehand (this is like an implementation choice)
    rowdegree = torch.sum(C, dim=1) # 
    coldegree = torch.sum(C, dim=0)
    qind = rowdegree + coldegree < 0.2 * torch.mean(rowdegree + coldegree)
    # need torch.inf b/c this could approach 1/sqrt(infinity) = 0
    rowdegree = torch.where(qind, torch.inf, rowdegree)
    coldegree = torch.where(qind, torch.inf, coldegree)

    nroi = C.shape[0]
    K = round(2 / 3 * nroi) if use_smalleigs is True else nroi # trunucating eigenvalues to 2/3 of the total number of ROIs (if enabled)

    Tau = 0.001 * D / speed
    Cc = C * torch.exp(-1j * Tau * w)

    # compute complex laplacian matrix L = I - alpha * D^(-1/2) * Cc * D^(-1/2)
    L2 = 1 / (torch.sqrt(rowdegree * coldegree) + torch.finfo(torch.float64).eps)
    L = torch.eye(nroi, dtype=torch.complex128, device=C.device) - alpha * (
        torch.diag(L2).to(torch.complex128) @ Cc
    )

    # eigenvalue decomposition of the complex laplacian matrix L
    d, v = torch.linalg.eig(L)
    eig_ind = torch.argsort(torch.abs(d))
    eigenvalues = d[eig_ind] 
    eigenvectors = v[:, eig_ind][:, :K] # trunucate based on K (calculated earlier)
   
   # phase shift for consistency with numpy code
    anchors = eigenvectors[
        torch.argmax(torch.abs(eigenvectors), dim=0),
        torch.arange(K, device=C.device),
    ]
    eigenvectors = eigenvectors * (anchors / torch.abs(anchors)).conj()[None, :]

    # local filters/responses
    Fe_local = (1 / tau_e**2) / (1j * w + 1 / tau_e) ** 2 # for MLP 
    Fi = (gii / tau_i**2) / (1j * w + 1 / tau_i) ** 2
    Hed = (alpha / tau_e) / (1j * w + alpha / tau_e * Fe_local)
    Hid = (alpha / tau_i) / (1j * w + alpha / tau_i * Fi)
    Heid = gei * Fe_local * Fi / (1 + gei * Fe_local * Fi)
    Htotal = a * Hed + (1 - a) / 2 * Hid + (1 - a) / 2 * Heid

    Fe_graph = Fe_local # this is temporary until we get the learned Fe_graph; later it will be a parameter of the function, or we will call the mlp here

    q1 = (tauC / alpha) * (1j * w + alpha / tauC * Fe_graph * eigenvalues) # we wil replace Fe_local here with a learned filter later on, but for now we are using the same Fe_local as in the original code
    qthr = zero_thr * torch.max(torch.abs(q1))
    magq1 = torch.maximum(torch.abs(q1), qthr)
    q1 = magq1 * torch.exp(1j * torch.angle(q1))
    frequency_response = Htotal / q1 # how strongly does the graph mode respond to the frequency input? (this is the main output of the forward model)

    model_out = torch.sum(
        eigenvectors[:, 1:K] * frequency_response[None, 1:K], dim=1
    )
    selected = eigenvectors[:, 1:K]
    FCmodel = selected @ torch.diag(frequency_response[1:K] ** 2) @ selected.T
    den = torch.sqrt(torch.abs(model_out))
    inv_den = torch.diag(1 / den).to(torch.complex128)
    FCmodel = inv_den @ FCmodel @ inv_den
    return frequency_response, eigenvalues, eigenvectors, model_out, FCmodel

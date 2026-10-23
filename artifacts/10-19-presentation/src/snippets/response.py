value = response["Fe_local"] if Fe_graph is None else torch.as_tensor(
    Fe_graph, dtype=torch.complex128, device=response["eigenvalues"].device)
q1 = response["tauC_over_alpha"][..., None] * (
    1j * response["w"][..., None]
    + response["alpha_over_tauC"][..., None] * value[..., None]
    * response["eigenvalues"])

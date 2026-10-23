components = self.network(
    (omega / self.omega_scale).unsqueeze(-1))
return torch.complex(
    components[..., 0], components[..., 1])

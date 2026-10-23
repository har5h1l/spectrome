threshold = 0.05 * q1.abs().amax(dim=-1, keepdim=True)
q1 = torch.maximum(q1.abs(), threshold) * torch.exp(1j * torch.angle(q1))
frequency_response = response["Htotal"][..., None] / q1

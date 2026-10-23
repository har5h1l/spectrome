magnitude = values.abs()
kernel = magnitude.new_tensor([1, 2, 5, 2, 1]).reshape(1, 1, 5) / 11
smoothed = F.conv1d(magnitude.unsqueeze(1), kernel, padding=2).squeeze(1)
transformed = smoothed.clamp_min(1e-12).sqrt()
return transformed - transformed.mean(dim=-1, keepdim=True)

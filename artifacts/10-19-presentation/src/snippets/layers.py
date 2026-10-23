layers, width = [], 1
for next_width in hidden_sizes:
    layers.extend([nn.Linear(width, next_width), activations[activation]()])
    width = next_width
layers.append(nn.Linear(width, 2))

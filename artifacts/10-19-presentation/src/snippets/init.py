torch.manual_seed(seed)
model = FeGraphMLP(hidden_sizes, omega_scale, activation=activation)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

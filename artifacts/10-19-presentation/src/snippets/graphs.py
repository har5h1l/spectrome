graphs = []
for frequency in frequencies:
    graph = nt.prepare_graph_torch(
        brain, parameters, 2 * torch.pi * frequency,
        use_smalleigs=use_smalleigs,
    )

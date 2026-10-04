# Learning

This package is reserved for the future learned-filter experiment. It contains
no model or training implementation yet.

The first filter will be one shared MLP taking angular frequency alone and
returning one complex graph-propagation value per frequency. Each subject's
seven fitted SGM parameters stay fixed; the local analytical filter stays
unchanged. Train the shared filter jointly across training subjects and compare
against the analytical baseline on held-out subjects.

Add filter definitions, training objectives, and training code here as that work
begins. Forward equations, graph preparation, and frequency sweeps remain in
`spectrome/forward/network_transfer_torch.py`. Reuse prepared graphs during
training while their inputs stay fixed. Resolve subject/region alignment and
the evaluation split before real-data training.

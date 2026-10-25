"""Fit one fresh frequency-only graph-filter MLP on supplied subjects."""

import torch

from ..model import FeGraphMLP
from ..spectra import subject_correlation
from .helpers import validate_training_inputs, validate_loss, validate_gradients


def train_filter(training_subjects, omega, model_config, training_config, *, on_epoch=None):
    """Fit one model and return its final state plus pre-update loss history.

    ``training_subjects`` must already contain only the subjects used for the
    fit. This function does not make splits, inspect held-out subjects, choose
    settings, write files, or create plots.
    """
    validate_training_inputs(training_subjects, model_config, training_config)
    epochs = training_config["epochs"]
    learning_rate = training_config["learning_rate"]
    seed = training_config["seed"]

    # Keep the fixed-configuration initialization contract. Reset immediately before every
    # new model so split construction and candidate order cannot consume it.
    torch.manual_seed(seed)
    model = FeGraphMLP(**model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        # Score the same shared filter across every training subject.
        optimizer.zero_grad()
        filters = model(omega)
        train_scores = torch.stack([
            subject_correlation(subject, filters)
            for subject in training_subjects
        ])
        loss = 1 - train_scores.mean()
        validate_loss(loss, epoch)

        # Differentiate the mean subject loss and update only the MLP weights.
        loss.backward()
        validate_gradients(model, epoch)
        optimizer.step()

        # Record the loss measured before this epoch's optimizer update.
        entry = {"epoch": epoch, "train_loss_before_step": loss.item()}
        history.append(entry)
        if on_epoch is not None:
            on_epoch(entry)
    model.eval()
    return model, history

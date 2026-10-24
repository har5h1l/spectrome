"""Frequency-only MLP for the shared complex graph-propagation filter."""

import torch
from torch import nn


class FeGraphMLP(nn.Module):
    """Map angular frequencies to shared complex graph-filter values.

    Input:
        omega: [F] angular frequencies in radians per second.

    Output:
        A complex tensor [F], with one Fe_graph value per frequency.
        predict_spectra() uses these values to predict each subject's spectra.
    """

    def __init__(self, hidden_sizes, omega_scale, activation="tanh"):
        """Initialize the MLP.

        Args:
            hidden_sizes: Units in each hidden layer, e.g. (8,).
            omega_scale: Positive frequency divisor applied to omega.
            activation: Hidden activation, tanh (default) or sigmoid.
        """
        super().__init__()
        hidden_sizes = tuple(hidden_sizes)
        self.hidden_sizes = hidden_sizes
        activations = {"tanh": nn.Tanh, "sigmoid": nn.Sigmoid}
        if activation not in activations:
            raise ValueError("activation must be tanh or sigmoid")
        self.activation = activation

        if not hidden_sizes or any(not isinstance(size, int) or size < 1 for size in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integer widths")
        if not 0 < float(omega_scale) < float("inf"):
            raise ValueError("omega_scale must be finite and positive (rad/s)")

        self.register_buffer("omega_scale", torch.tensor(float(omega_scale), dtype=torch.float64))

        # The final linear layer outputs unrestricted real and imaginary parts.
        layers, width = [], 1
        for next_width in hidden_sizes:
            layers.extend([nn.Linear(width, next_width), activations[activation]()])
            width = next_width
        layers.append(nn.Linear(width, 2))

        self.network = nn.Sequential(*layers).double()

    def forward(self, omega):
        omega = torch.as_tensor(omega, dtype=self.omega_scale.dtype,
                                device=self.omega_scale.device)
        components = self.network((omega / self.omega_scale).unsqueeze(-1))
        return torch.complex(components[..., 0], components[..., 1])

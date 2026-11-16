"""MLP for the shared complex graph-propagation filter."""

import torch
from torch import nn


class FeGraphMLP(nn.Module):
    """Map angular frequencies, and optionally subject tau_e, to filter values.

    Input:
        omega: [F] angular frequencies in radians per second.
        tau_e: Required finite positive scalar in seconds when conditioned.

    Output:
        A complex tensor [F], with one Fe_graph value per frequency.
        predict_spectra() uses these values to predict each subject's spectra.
    """

    def __init__(self, hidden_sizes, omega_scale, activation="tanh",
                 tau_e_conditioned=False):
        """Initialize the MLP.

        Args:
            hidden_sizes: Units in each hidden layer, e.g. (8,).
            omega_scale: Positive frequency divisor applied to omega.
            activation: Hidden activation, tanh (default) or sigmoid.
            tau_e_conditioned: Also condition on the subject's fixed tau_e (seconds).
        """
        super().__init__()
        hidden_sizes = tuple(hidden_sizes)
        self.hidden_sizes = hidden_sizes
        activations = {"tanh": nn.Tanh, "sigmoid": nn.Sigmoid}
        if activation not in activations:
            raise ValueError("activation must be tanh or sigmoid")
        self.activation = activation
        if not isinstance(tau_e_conditioned, bool):
            raise ValueError("tau_e_conditioned must be a boolean")
        self.tau_e_conditioned = tau_e_conditioned

        if not hidden_sizes or any(not isinstance(size, int) or size < 1 for size in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integer widths")
        if not 0 < float(omega_scale) < float("inf"):
            raise ValueError("omega_scale must be finite and positive (rad/s)")

        self.register_buffer("omega_scale", torch.tensor(float(omega_scale), dtype=torch.float64))

        # The final linear layer outputs unrestricted real and imaginary parts.
        layers, width = [], (2 if tau_e_conditioned else 1)
        for next_width in hidden_sizes:
            layers.extend([nn.Linear(width, next_width), activations[activation]()])
            width = next_width
        layers.append(nn.Linear(width, 2))

        self.network = nn.Sequential(*layers).double()

    def forward(self, omega, tau_e=None):
        omega = torch.as_tensor(omega, dtype=self.omega_scale.dtype,
                                device=self.omega_scale.device)
        normalized_omega = omega / self.omega_scale
        if self.tau_e_conditioned:
            try:
                tau_e = torch.as_tensor(tau_e, dtype=normalized_omega.dtype,
                                        device=normalized_omega.device)
            except (TypeError, ValueError, RuntimeError) as error:
                raise ValueError("tau_e_conditioned model requires a finite positive scalar tau_e") from error
            if (tau_e.numel() != 1 or not torch.isfinite(tau_e).all()
                    or not (tau_e > 0).all()):
                raise ValueError("tau_e_conditioned model requires a finite positive scalar tau_e")
            tau_feature = (tau_e.reshape(()) / 0.01).expand_as(normalized_omega)
            inputs = torch.stack((normalized_omega, tau_feature), dim=-1)
        else:
            inputs = normalized_omega.unsqueeze(-1)
        components = self.network(inputs)
        return torch.complex(components[..., 0], components[..., 1])

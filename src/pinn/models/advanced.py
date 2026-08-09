"""Advanced PINN architectures.

This module collects several neural network architectures that extend the
basic :class:`~pinn.models.MLP` model with additional capabilities tailored
for challenging PDE problems.  The implementations are lightweight and focus
on demonstrating the interfaces required for experimentation rather than being
state of the art.  Each model is fully differentiable and can be composed with
standard training loops.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn

from .mlp import MLP
from .multiscale import MultiScalePINN
from ..utils.validation import check_range


class SineLayer(nn.Module):
    """Linear layer followed by a scaled sine activation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        omega_0: float = 30.0,
        is_first: bool = False,
    ) -> None:
        """Initialize a SIREN layer with paper-style weight bounds."""
        super().__init__()
        check_range("in_features", in_features, min=1)
        check_range("out_features", out_features, min=1)
        if omega_0 <= 0.0:
            raise ValueError(f"omega_0 must be positive, got {omega_0}")
        self.in_features = in_features
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights using SIREN frequency-preserving bounds."""
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the scaled sine activation."""
        return torch.sin(self.omega_0 * self.linear(x))


class SirenPINN(nn.Module):
    """SIREN network for representing high-frequency PDE solutions.

    Sinusoidal representation networks are useful when the target field has
    sharp oscillations or when accurate coordinate derivatives are important.
    The initialization follows the SIREN paper's separate first-layer and
    hidden-layer bounds.
    """

    def __init__(
        self,
        in_dim: int = 2,
        hidden_layers: int = 3,
        width: int = 64,
        out_dim: int = 1,
        *,
        omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
    ) -> None:
        """Initialize the sinusoidal network."""
        super().__init__()
        check_range("in_dim", in_dim, min=1)
        check_range("hidden_layers", hidden_layers, min=0)
        check_range("width", width, min=1)
        check_range("out_dim", out_dim, min=1)
        if omega_0 <= 0.0:
            raise ValueError(f"omega_0 must be positive, got {omega_0}")
        if hidden_omega_0 <= 0.0:
            raise ValueError(f"hidden_omega_0 must be positive, got {hidden_omega_0}")

        layers: list[nn.Module] = []
        previous_dim = in_dim
        if hidden_layers > 0:
            layers.append(
                SineLayer(
                    previous_dim,
                    width,
                    omega_0=omega_0,
                    is_first=True,
                )
            )
            previous_dim = width
            for _ in range(hidden_layers - 1):
                layers.append(
                    SineLayer(
                        previous_dim,
                        width,
                        omega_0=hidden_omega_0,
                    )
                )
        self.net = nn.Sequential(*layers)
        self.final_linear = nn.Linear(previous_dim, out_dim)
        self._init_final_layer(hidden_omega_0)

    def _init_final_layer(self, omega_0: float) -> None:
        """Initialize the output layer consistently with hidden SIREN layers."""
        with torch.no_grad():
            bound = math.sqrt(6.0 / self.final_linear.in_features) / omega_0
            self.final_linear.weight.uniform_(-bound, bound)
            self.final_linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the SIREN network."""
        features = self.net(x) if len(self.net) > 0 else x
        return self.final_linear(features)


class FourierFeaturePINN(nn.Module):
    """Embed inputs using Fourier features before applying an ``MLP``.

    Useful for problems with periodic structure where high-frequency
    components are difficult to learn with standard networks.
    """

    def __init__(
        self,
        in_features: int = 1,
        mapping_size: int = 256,
        sigma: float = 1.0,
        base_network=MLP,
    ):
        """Initialize Fourier features and the downstream base network."""
        super().__init__()
        self.B = nn.Parameter(
            sigma * torch.randn(in_features, mapping_size), requires_grad=False
        )
        self.network = base_network(in_dim=2 * mapping_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial
        """Evaluate the model after sine/cosine Fourier embedding."""
        # (batch, in_features) @ (in_features, mapping_size) -> (batch, mapping_size)
        projections = 2 * torch.pi * x @ self.B
        features = torch.cat([torch.sin(projections), torch.cos(projections)], dim=-1)
        return self.network(features)


class AttentionPINN(nn.Module):
    """Apply a self-attention mechanism over the feature dimension.

    The input is first projected to an embedding space, attended, and then fed
    through a small ``MLP``.  This is a simplified attention model intended to
    showcase the idea rather than provide a highly optimised transformer.
    """

    def __init__(
        self,
        in_features: int = 2,
        embed_dim: int = 32,
        num_heads: int = 4,
        base_network=MLP,
    ):
        """Initialize embedding, attention and prediction layers."""
        super().__init__()
        self.embed = nn.Linear(in_features, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.mlp = base_network(in_dim=embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial
        """Evaluate a single-token self-attention PINN."""
        # Treat the feature dimension as sequence length 1
        x_e = self.embed(x).unsqueeze(1)
        attn_out, _ = self.attn(x_e, x_e, x_e)
        attn_out = attn_out.squeeze(1)
        return self.mlp(attn_out)


class EnsemblePINN(nn.Module):
    """Average predictions from a list of models to obtain uncertainty.

    Parameters
    ----------
    networks:
        Sequence of neural networks with identical input/output shapes.
    """

    def __init__(self, networks: Sequence[nn.Module]):
        """Store ensemble members."""
        super().__init__()
        if not networks:
            raise ValueError("Ensemble requires at least one network")
        self.networks = nn.ModuleList(networks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial
        """Return the mean prediction across ensemble members."""
        preds = [net(x) for net in self.networks]
        return torch.stack(preds).mean(dim=0)

    def predict_with_uncertainty(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mean and standard deviation of ensemble predictions."""
        preds = torch.stack([net(x) for net in self.networks])
        mean = preds.mean(dim=0)
        std = preds.std(dim=0)
        return mean, std


__all__ = [
    "MultiScalePINN",
    "SineLayer",
    "SirenPINN",
    "FourierFeaturePINN",
    "AttentionPINN",
    "EnsemblePINN",
]

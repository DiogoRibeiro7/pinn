"""Advanced PINN architectures.

This module collects several neural network architectures that extend the
basic :class:`~pinn.models.MLP` model with additional capabilities tailored
for challenging PDE problems.  The implementations are lightweight and focus
on demonstrating the interfaces required for experimentation rather than being
state of the art.  Each model is fully differentiable and can be composed with
standard training loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from .mlp import MLP
from .multiscale import MultiScalePINN


class FourierFeaturePINN(nn.Module):
    """Embed inputs using Fourier features before applying an ``MLP``.

    Useful for problems with periodic structure where high-frequency
    components are difficult to learn with standard networks.
    """

    def __init__(self, in_features: int = 1, mapping_size: int = 256, sigma: float = 1.0, base_network=MLP):
        super().__init__()
        self.B = nn.Parameter(sigma * torch.randn(in_features, mapping_size), requires_grad=False)
        self.network = base_network(in_dim=2 * mapping_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial
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

    def __init__(self, in_features: int = 2, embed_dim: int = 32, num_heads: int = 4, base_network=MLP):
        super().__init__()
        self.embed = nn.Linear(in_features, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.mlp = base_network(in_dim=embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial
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
        super().__init__()
        if not networks:
            raise ValueError("Ensemble requires at least one network")
        self.networks = nn.ModuleList(networks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial
        preds = [net(x) for net in self.networks]
        return torch.stack(preds).mean(dim=0)

    def predict_with_uncertainty(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mean and standard deviation of ensemble predictions."""

        preds = torch.stack([net(x) for net in self.networks])
        mean = preds.mean(dim=0)
        std = preds.std(dim=0)
        return mean, std


__all__ = [
    "MultiScalePINN",
    "FourierFeaturePINN",
    "AttentionPINN",
    "EnsemblePINN",
]

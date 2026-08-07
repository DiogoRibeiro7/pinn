"""Tests for importance sampling utilities."""

from __future__ import annotations

import numpy as np
import torch

from pinn.sampling import GradientBasedImportanceSampler
from pinn.utils.sampling import Domain, LatinHypercubeSampler


class IdentityModel(torch.nn.Module):
    """Simple model returning the input as output with a dummy parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial
        return x * self.weight


def simple_loss(model: torch.nn.Module, pts: torch.Tensor) -> torch.Tensor:
    """Loss whose gradient norm increases with x."""
    preds = model(pts)
    return (preds**2).mean()


def test_importance_sampler_updates_and_samples() -> None:
    domain = Domain([(0.0, 1.0)])
    base = LatinHypercubeSampler(domain, seed=0)
    sampler = GradientBasedImportanceSampler(
        domain, base, update_frequency=1, n_candidates=32
    )

    model = IdentityModel()
    pts = sampler.sample(8, model=model, loss_fn=simple_loss)
    assert pts.shape == (8, 1)

    weights = sampler.importance_weights.cpu().numpy()
    assert np.isclose(weights.sum(), 1.0)
    # Points with larger coordinate should have larger weight
    candidates = sampler.last_points.cpu().numpy().ravel()
    order = np.argsort(candidates)
    assert np.all(np.diff(weights[order]) >= -1e-6)

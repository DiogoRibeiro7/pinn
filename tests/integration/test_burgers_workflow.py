"""Integration tests for a minimal Burgers PINN workflow."""

from __future__ import annotations

import pytest

from pinn.solvers.raissi_improved import (
    BurgersConfig,
    ContinuousPINN,
    TrainConfig,
    burgers_residual,
)
from pinn.sampling import GradientBasedImportanceSampler
from pinn.utils.sampling import Domain, LatinHypercubeSampler
from tests.fixtures import simple_mlp


@pytest.mark.integration
def test_burgers_end_to_end(device):
    """Train a tiny model for a few steps to verify end-to-end execution."""

    model = simple_mlp().to(device)
    pinn = ContinuousPINN(
        model=model,
        device=device,
        pde_residual_fn=burgers_residual,
        cfg_burgers=BurgersConfig(),
    )

    domain = Domain([(0.0, 1.0), (-1.0, 1.0)])
    base = LatinHypercubeSampler(domain, seed=0)
    imp_sampler = GradientBasedImportanceSampler(
        domain, base, update_frequency=1, n_candidates=64
    )

    tcfg = TrainConfig(
        n_u0=5,
        n_bc=5,
        n_f=10,
        lr=1e-3,
        adam_steps=2,
        lbfgs_max_iter=0,
        seed=42,
        print_frequency=1,
        track_losses=True,
        importance_sampler=imp_sampler,
    )

    history = pinn.train(tcfg)
    assert "total" in history and len(history["total"]) >= 1


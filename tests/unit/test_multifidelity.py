"""Tests for multi-fidelity sampling and training."""

from __future__ import annotations

import numpy as np
import torch

from pinn.sampling import (
    HierarchicalConfig,
    HierarchicalPINNTrainer,
    MultiFidelityGaussianProcess,
    MultiFidelitySampler,
)
from pinn.utils.sampling import Domain, LatinHypercubeSampler


def test_hierarchical_schedule_allocates_more_to_low_fidelity() -> None:
    domain = Domain([(0.0, 1.0)])
    sampler = MultiFidelitySampler(
        domain, [5, 20], [1.0, 4.0], base_sampler=LatinHypercubeSampler(domain, seed=1)
    )
    schedule = sampler.hierarchical_sampling_schedule(100)
    assert len(schedule) == 2
    # first level should receive more points than the second
    assert schedule[0][1] > schedule[1][1]


def _make_model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(1, 16),
        torch.nn.Tanh(),
        torch.nn.Linear(16, 1),
    )


def test_hierarchical_training_improves_accuracy() -> None:
    torch.manual_seed(0)
    domain = Domain([(0.0, np.pi)])
    sampler = MultiFidelitySampler(
        domain, [5, 20], [1.0, 4.0], base_sampler=LatinHypercubeSampler(domain, seed=2)
    )
    models = [_make_model(), _make_model()]
    analytic_fn = lambda x: torch.sin(x)
    trainer = HierarchicalPINNTrainer(models, sampler, analytic_fn)
    trainer.train_hierarchical(
        HierarchicalConfig(total_budget=40, epochs=50, validate_points=20)
    )
    err_low = trainer.validate(0, 20)
    err_high = trainer.validate(1, 20)
    assert err_high < err_low
    assert err_high < 0.1


def test_multifidelity_gp_learns_linear_relation() -> None:
    y_low = np.array([0.0, 1.0, 2.0])
    y_high = 2 * y_low + 1
    mf_gp = MultiFidelityGaussianProcess()
    mf_gp.fit(y_low, y_high)
    preds = mf_gp.predict(np.array([3.0, 4.0]))
    assert np.allclose(preds, 2 * np.array([3.0, 4.0]) + 1)

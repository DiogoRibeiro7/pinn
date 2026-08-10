"""Tests for adaptive loss weighting utilities."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pinnkit.models.mlp import MLP
from pinnkit.solvers.raissi import (
    BurgersConfig,
    ContinuousPINN,
    TrainConfig,
    burgers_residual,
)
from pinnkit.training.adaptive_weighting import (
    AdaptiveLossWeighting,
    AdaptiveWeightingConfig,
    WeightEvolutionVisualizer,
    compute_gradient_norms,
)


def test_gradnorm_updates_weights() -> None:
    cfg = AdaptiveWeightingConfig(
        enabled=True,
        method="gradnorm",
        adaptation_rate=0.5,
        adaptation_frequency=1,
    )
    weighting = AdaptiveLossWeighting(("ic", "bc", "pde"), cfg)
    losses = {
        "ic": torch.tensor(0.1),
        "bc": torch.tensor(0.2),
        "pde": torch.tensor(1.0),
        "total": torch.tensor(1.3),
    }
    grads = {
        "ic": torch.tensor(0.2),
        "bc": torch.tensor(0.2),
        "pde": torch.tensor(1.2),
    }
    before = weighting.weights.clone()
    after = weighting.update_weights(losses, step=1, gradient_norms=grads)
    assert not torch.allclose(before, after)
    assert pytest.approx(after.sum().item(), rel=1e-5) == 1.0
    assert float(after[2]) != float(before[2])


def test_dwa_history_adjusts_weights() -> None:
    cfg = AdaptiveWeightingConfig(enabled=True, method="dwa", adaptation_frequency=1)
    weighting = AdaptiveLossWeighting(("ic", "bc", "pde"), cfg)
    for step, (ic_val, pde_val) in enumerate(
        [(0.6, 0.8), (0.5, 0.7), (0.4, 0.5)], start=1
    ):
        losses = {
            "ic": torch.tensor(ic_val),
            "bc": torch.tensor(ic_val),
            "pde": torch.tensor(pde_val),
            "total": torch.tensor(ic_val * 2 + pde_val),
        }
        weighting.update_weights(losses, step=step)
    weights = weighting.weights.detach().cpu().numpy()
    assert pytest.approx(np.sum(weights), rel=1e-5) == 1.0
    assert weights[2] != pytest.approx(1.0 / 3.0)


def test_visualizer_payload() -> None:
    viz = WeightEvolutionVisualizer(("ic", "bc", "pde"))
    payload = viz.create_payload([[0.3, 0.3, 0.4], [0.25, 0.35, 0.4]], [1.0, 0.8])
    assert payload is not None
    assert payload["weights"]["pde"][-1] == pytest.approx(0.4)


def test_compute_gradient_norms_returns_expected() -> None:
    model = torch.nn.Linear(2, 1)
    params = list(model.parameters())
    inputs = torch.randn(4, 2, requires_grad=True)
    outputs = model(inputs)
    losses = {"l1": outputs.pow(2).mean()}
    norms = compute_gradient_norms(losses, params)
    assert "l1" in norms
    assert norms["l1"].item() >= 0.0


def test_continuous_pinn_uses_adaptive_weighting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = TrainConfig(
        n_u0=8,
        n_bc=8,
        n_f=32,
        lr=1e-3,
        adam_steps=2,
        lbfgs_max_iter=0,
        adaptive_weighting=AdaptiveWeightingConfig(
            enabled=True,
            method="dwa",
            adaptation_frequency=1,
        ),
    )
    model = MLP(in_dim=2, out_dim=1, hidden_layers=2, width=16)
    pinn = ContinuousPINN(model, "cpu", burgers_residual, BurgersConfig())

    calls: list[int] = []
    original_update = AdaptiveLossWeighting.update_weights

    def wrapped_update(
        self: AdaptiveLossWeighting,
        losses,
        *,
        step: int,
        gradient_norms=None,
    ):
        calls.append(step)
        return original_update(self, losses, step=step, gradient_norms=gradient_norms)

    monkeypatch.setattr(AdaptiveLossWeighting, "update_weights", wrapped_update)
    pinn.train(cfg)
    assert calls, "adaptive weighting should be invoked at least once"

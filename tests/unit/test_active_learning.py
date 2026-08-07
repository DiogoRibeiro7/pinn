"""Tests for active learning utilities."""

from __future__ import annotations

import math

import torch

from pinn.sampling import (
    ActiveLearningConfig,
    ActiveLearningStrategy,
    ActivePINNTrainer,
    ExpectedImprovementAcquisition,
    InformationGainAcquisition,
    QueryByCommitteeAcquisition,
    UncertaintyAcquisition,
)
from pinn.solvers.raissi_improved import TrainConfig


class StubUncertainModel(torch.nn.Module):
    """Model exposing deterministic uncertainty for testing."""

    def __init__(self, means: list[float], stds: list[float]) -> None:
        super().__init__()
        self.means = torch.tensor(means, dtype=torch.float32)
        self.stds = torch.tensor(stds, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial
        return x.sum(dim=1, keepdim=True)

    def predict_with_uncertainty(
        self, points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = points.shape[0]
        return self.means[:n, None], self.stds[:n, None]


def test_uncertainty_acquisition_prefers_large_variance() -> None:
    model = StubUncertainModel([0.0, 0.0], [0.1, 0.6])
    acq = UncertaintyAcquisition(mode="variance")
    pts = torch.zeros(2, 2)
    scores = acq(pts, model=model)
    assert scores.shape == (2,)
    assert scores[1] > scores[0]


def test_expected_improvement_uses_best_value() -> None:
    model = StubUncertainModel([0.1, 0.4], [0.2, 0.2])
    acq = ExpectedImprovementAcquisition(
        best_value=0.25,
        maximize=False,
        custom_predict=lambda m, pts: m.predict_with_uncertainty(pts),
    )
    pts = torch.zeros(2, 2)
    scores = acq(pts, model=model)
    assert scores[0] > scores[1]


def test_query_by_committee_measures_disagreement() -> None:
    class ConstantModel(torch.nn.Module):
        def __init__(self, value: float) -> None:
            super().__init__()
            self.value = torch.tensor(value, dtype=torch.float32)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.full((x.shape[0], 1), self.value)

    committee = [ConstantModel(0.0), ConstantModel(1.0)]
    acq = QueryByCommitteeAcquisition(committee)
    pts = torch.zeros(5, 2)
    scores = acq(pts, model=committee)
    expected = torch.full((5,), math.sqrt(0.5))
    assert torch.allclose(scores, expected)


def test_information_gain_is_positive() -> None:
    model = StubUncertainModel([0.0, 0.0, 0.0], [0.1, 0.2, 0.3])
    acq = InformationGainAcquisition()
    pts = torch.zeros(3, 2)
    scores = acq(pts, model=model)
    assert torch.all(scores > 0)


def test_active_learning_strategy_selects_top_points() -> None:
    def acquisition(
        points: torch.Tensor, *, model: torch.nn.Module, **_: dict
    ) -> torch.Tensor:
        return points[:, 0]

    strategy = ActiveLearningStrategy(acquisition, batch_size=1)
    pts = torch.tensor([[0.2, 0.0], [0.7, 0.0], [0.4, 0.0]])
    selected = strategy.select_points(
        pts, model=StubUncertainModel([0, 0, 0], [1, 1, 1]), n_select=2
    )
    assert selected.shape == (2, 2)
    assert torch.allclose(selected[:, 0], torch.tensor([0.7, 0.4]))


def test_active_pinn_trainer_configures_train_config() -> None:
    class DummyPINN:
        def __init__(self) -> None:
            self.received_config: TrainConfig | None = None

        def train(self, cfg: TrainConfig):  # pragma: no cover - simple storage
            self.received_config = cfg
            return {"total": [0.0]}

    def acquisition(
        points: torch.Tensor, *, model: torch.nn.Module, **_: dict
    ) -> torch.Tensor:
        return torch.linspace(0, 1, points.shape[0])

    strategy = ActiveLearningStrategy(acquisition, batch_size=2)
    trainer = ActivePINNTrainer(DummyPINN(), strategy)
    tcfg = TrainConfig(
        n_u0=4, n_bc=4, n_f=16, adam_steps=1, lbfgs_max_iter=0, track_losses=False
    )
    cfg = ActiveLearningConfig(
        max_active_iterations=2,
        points_per_iteration=3,
        candidate_pool_size=8,
        update_frequency=1,
        convergence_tol=1e-4,
        patience=2,
        max_points=64,
    )

    trainer.train_with_active_learning(tcfg, cfg)
    assert trainer.pinn.received_config is not None
    received = trainer.pinn.received_config
    assert received.active_strategy is strategy
    assert received.active_points_per_iteration == 3
    assert received.active_candidate_pool == 8
    assert received.active_update_interval == 1
    assert received.active_max_iterations == 2
    assert received.active_max_points == 64

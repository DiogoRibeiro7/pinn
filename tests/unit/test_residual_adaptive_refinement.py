"""Tests for residual-based adaptive refinement in the generic trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from pinn.constraints import Constraint
from pinn.geometry import Rectangle
from pinn.models import MLP
from pinn.problems import HeatProblem
from pinn.residuals import (
    AutogradDerivativeBackend,
    DerivativeBackend,
    StrongFormResidual,
)
from pinn.solvers.heat import HeatConfig
from pinn.training import (
    OptimizerConfig,
    ResidualAdaptiveConfig,
    ResidualAdaptiveRefiner,
    Trainer,
    TrainerConfig,
)


@dataclass(frozen=True)
class CoordinateResidualProblem:
    """Problem whose residual increases with the second coordinate."""

    geometry: Rectangle = Rectangle((0.0, 0.0), (1.0, 1.0), ("t", "x"))

    @property
    def input_dim(self) -> int:
        """Return coordinate dimension."""
        return 2

    @property
    def output_dim(self) -> int:
        """Return scalar output dimension."""
        return 1

    def strong_residual(
        self,
        model: nn.Module,
        coordinates: Tensor,
        derivative_backend: DerivativeBackend,
    ) -> Tensor:
        """Return a residual that ranks points by ``x``."""
        return coordinates[:, [1]]

    def constraints(self) -> Sequence[Constraint]:
        """Return no constraints for this synthetic problem."""
        return ()


def test_residual_adaptive_refiner_selects_high_residual_points() -> None:
    """RAR keeps in-domain, high-residual candidates with the requested dtype."""
    problem = CoordinateResidualProblem()
    refiner = ResidualAdaptiveRefiner(
        ResidualAdaptiveConfig(
            candidate_count=128,
            points_per_refinement=16,
            refresh_every=1,
            max_refined_points=16,
        )
    )
    generator = torch.Generator().manual_seed(7)

    diagnostics = refiner.refine(
        problem,
        nn.Linear(2, 1),
        StrongFormResidual(),
        AutogradDerivativeBackend(),
        generator=generator,
        device="cpu",
        dtype=torch.float64,
    )
    points = refiner.current_points(device="cpu", dtype=torch.float64)

    assert points is not None
    assert points.shape == (16, 2)
    assert points.dtype == torch.float64
    assert torch.all(problem.geometry.contains(points))
    assert torch.min(points[:, 1]).item() > 0.75
    assert diagnostics["rar_points"] == 16.0
    assert diagnostics["rar_new_points"] == 16.0
    assert diagnostics["rar_candidate_max_residual"] <= 1.0


def test_generic_trainer_records_residual_adaptive_diagnostics() -> None:
    """Generic trainer can add RAR points and report refinement diagnostics."""
    problem = HeatProblem(HeatConfig(alpha=0.1), n_constraint_samples=4)
    model = MLP(2, 1, 8, 1)
    trainer = Trainer(
        TrainerConfig(
            seed=5,
            dtype=torch.float64,
            collocation_count=8,
            constraint_sample_counts={"ic": 4, "bc_left": 4, "bc_right": 4},
            adaptive_refinement=ResidualAdaptiveConfig(
                candidate_count=16,
                points_per_refinement=4,
                refresh_every=1,
                max_refined_points=8,
            ),
            optimizer=OptimizerConfig(adam_steps=2, lr=1e-3),
            log_every=1,
        )
    )

    result = trainer.train(problem, model)

    assert result.final_loss >= 0.0
    assert result.final_state.diagnostics["rar_points"] == 8.0
    assert result.final_state.diagnostics["rar_new_points"] == 4.0
    assert "rar_candidate_mean_residual" in result.final_state.diagnostics
    assert next(model.parameters()).dtype == torch.float64


def test_residual_adaptive_config_rejects_invalid_options() -> None:
    """RAR configuration validates sample counts and retention limits."""
    for config in (
        ResidualAdaptiveConfig(candidate_count=0),
        ResidualAdaptiveConfig(points_per_refinement=0),
        ResidualAdaptiveConfig(candidate_count=2, points_per_refinement=3),
        ResidualAdaptiveConfig(refresh_every=0),
        ResidualAdaptiveConfig(start_step=0),
        ResidualAdaptiveConfig(points_per_refinement=4, max_refined_points=3),
    ):
        try:
            config.validate()
        except ValueError:
            continue
        raise AssertionError("invalid ResidualAdaptiveConfig was accepted")

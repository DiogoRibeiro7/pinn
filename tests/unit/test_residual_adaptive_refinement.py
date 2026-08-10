"""Tests for residual-based adaptive refinement in the generic trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from pinnkit.constraints import Constraint
from pinnkit.geometry import Geometry, Rectangle
from pinnkit.models import MLP
from pinnkit.problems import HeatProblem
from pinnkit.residuals import (
    AutogradDerivativeBackend,
    DerivativeBackend,
    StrongFormResidual,
)
from pinnkit.solvers.heat import HeatConfig
from pinnkit.training import (
    OptimizerConfig,
    ResidualAdaptiveConfig,
    ResidualAdaptiveRefiner,
    Trainer,
    TrainerConfig,
)


@dataclass(frozen=True)
class CoordinateResidualProblem:
    """Problem whose residual increases with the second coordinate."""

    geometry: Geometry = Rectangle((0.0, 0.0), (1.0, 1.0), ("t", "x"))

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


@dataclass(frozen=True)
class FixedCandidateGeometry:
    """Geometry that returns a fixed candidate set for deterministic RAR tests."""

    candidates: Tensor

    @property
    def dimension(self) -> int:
        """Return coordinate dimension."""
        return int(self.candidates.shape[1])

    @property
    def lower_bounds(self) -> Tensor:
        """Return lower coordinate bounds."""
        return torch.zeros(self.dimension, dtype=torch.float64)

    @property
    def upper_bounds(self) -> Tensor:
        """Return upper coordinate bounds."""
        return torch.ones(self.dimension, dtype=torch.float64)

    @property
    def names(self) -> Sequence[str]:
        """Return coordinate names."""
        return ("t", "x")

    def sample_interior(
        self,
        n: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Return the first ``n`` fixed candidates."""
        del generator
        return self.candidates[:n].to(device=device, dtype=dtype)

    def sample_boundary(
        self,
        n: int,
        *,
        dim: int,
        side: str,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Boundary sampling is not needed for these tests."""
        del n, dim, side, generator, device, dtype
        raise NotImplementedError

    def contains(self, points: Tensor, *, atol: float = 1e-7) -> Tensor:
        """Return a mask for points in the unit square."""
        return torch.all((points >= -atol) & (points <= 1.0 + atol), dim=1)

    def boundary_mask(
        self,
        points: Tensor,
        *,
        dim: int | None = None,
        side: str | None = None,
        atol: float = 1e-7,
    ) -> Tensor:
        """Boundary masks are not needed for these tests."""
        del points, dim, side, atol
        raise NotImplementedError


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


def test_residual_adaptive_refiner_can_select_diverse_points() -> None:
    """RAR can trade residual score for spread across the geometry."""
    candidates = torch.tensor(
        [
            [0.0, 0.90],
            [0.0, 0.91],
            [0.0, 0.92],
            [0.0, 0.10],
            [0.0, 0.50],
            [0.0, 0.99],
        ],
        dtype=torch.float64,
    )
    problem = CoordinateResidualProblem(FixedCandidateGeometry(candidates))
    refiner = ResidualAdaptiveRefiner(
        ResidualAdaptiveConfig(
            candidate_count=6,
            points_per_refinement=3,
            refresh_every=1,
            max_refined_points=3,
            diversity_weight=2.0,
        )
    )

    diagnostics = refiner.refine(
        problem,
        nn.Linear(2, 1),
        StrongFormResidual(),
        AutogradDerivativeBackend(),
        generator=torch.Generator().manual_seed(7),
        device="cpu",
        dtype=torch.float64,
    )
    points = refiner.current_points(device="cpu", dtype=torch.float64)

    assert points is not None
    selected_x = set(torch.round(points[:, 1], decimals=2).tolist())
    assert selected_x == {0.1, 0.5, 0.99}
    assert diagnostics["rar_selection_diversity_weight"] == 2.0
    assert diagnostics["rar_selected_min_distance"] > 0.39


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
        ResidualAdaptiveConfig(diversity_weight=-1.0),
    ):
        try:
            config.validate()
        except ValueError:
            continue
        raise AssertionError("invalid ResidualAdaptiveConfig was accepted")

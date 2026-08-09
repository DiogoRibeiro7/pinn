"""Tests for the composable core PINN API."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from pinn.constraints import (
    DirichletBoundary,
    InitialCondition,
    NeumannBoundary,
    PeriodicBoundary,
)
from pinn.geometry import Interval, Rectangle
from pinn.models import MLP
from pinn.problems import BurgersProblem, HeatProblem, WaveProblem
from pinn.residuals import AutogradDerivativeBackend, StrongFormResidual
from pinn.sampling import UniformInteriorSampler
from pinn.solvers import HeatPINN, WavePINN
from pinn.solvers._base import TrainConfig as LegacyExactTrainConfig
from pinn.solvers.raissi_improved import BurgersConfig
from pinn.solvers.heat import HeatConfig
from pinn.solvers.wave import WaveConfig
from pinn.training import (
    OptimizerConfig,
    ResidualAdaptiveConfig,
    Trainer,
    TrainerConfig,
)


class AnalyticHeatModel(nn.Module):
    """Torch model returning the heat equation analytic solution."""

    def __init__(self, alpha: float) -> None:
        """Store the heat diffusivity."""
        super().__init__()
        self.alpha = alpha

    def forward(self, coordinates: Tensor) -> Tensor:
        """Evaluate the analytic heat field."""
        t = coordinates[:, [0]]
        x = coordinates[:, [1]]
        return torch.exp(-self.alpha * np.pi**2 * t) * torch.sin(np.pi * x)


class AnalyticWaveModel(nn.Module):
    """Torch model returning the wave equation analytic solution."""

    def __init__(self, c: float) -> None:
        """Store the wave speed."""
        super().__init__()
        self.c = c

    def forward(self, coordinates: Tensor) -> Tensor:
        """Evaluate the analytic wave field."""
        t = coordinates[:, [0]]
        x = coordinates[:, [1]]
        return torch.cos(self.c * np.pi * t) * torch.sin(np.pi * x)


class ZeroModel(nn.Module):
    """Torch model returning a differentiable zero field."""

    def forward(self, coordinates: Tensor) -> Tensor:
        """Return zeros that keep a connection to input coordinates."""
        return coordinates[:, [0]] * 0.0


class CoordinateModel(nn.Module):
    """Simple differentiable model for constraint tests."""

    def forward(self, coordinates: Tensor) -> Tensor:
        """Return a linear field with known derivatives."""
        return coordinates[:, [0]] + 2.0 * coordinates[:, [1]]


class RecordingSampler:
    """Interior sampler that records how often the trainer calls it."""

    def __init__(self) -> None:
        """Initialize call counters."""
        self.calls = 0
        self.last_count = 0
        self.last_dtype: torch.dtype | None = None

    def sample(
        self,
        geometry: Rectangle,
        n: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Record the request and delegate to uniform geometry sampling."""
        self.calls += 1
        self.last_count = n
        self.last_dtype = dtype
        return geometry.sample_interior(
            n,
            generator=generator,
            device=device,
            dtype=dtype,
        )


def test_interval_sampling_and_boundary_masks_are_typed() -> None:
    """Interval sampling preserves requested dtype and boundary semantics."""
    interval = Interval(-2.0, 3.0)
    points = interval.sample_interior(16, dtype=torch.float64)
    assert points.shape == (16, 1)
    assert points.dtype == torch.float64
    assert torch.all(interval.contains(points))

    lower = interval.sample_boundary(4, dim=0, side="lower", dtype=torch.float64)
    upper = interval.sample_boundary(4, dim=0, side="upper", dtype=torch.float64)
    assert torch.allclose(lower, torch.full_like(lower, -2.0))
    assert torch.allclose(upper, torch.full_like(upper, 3.0))
    assert torch.all(interval.boundary_mask(torch.cat([lower, upper], dim=0)))


def test_rectangle_boundary_sampling_fixes_requested_face() -> None:
    """Rectangle boundary sampling fixes exactly the selected face."""
    rectangle = Rectangle((0.0, -1.0), (1.0, 1.0), ("t", "x"))
    points = rectangle.sample_boundary(10, dim=1, side="upper")
    assert points.shape == (10, 2)
    assert torch.allclose(points[:, 1], torch.ones(10))
    assert torch.all(rectangle.contains(points))
    assert torch.all(rectangle.boundary_mask(points, dim=1, side="upper"))


def test_soft_constraints_evaluate_expected_losses() -> None:
    """Initial, Dirichlet and Neumann soft losses vanish when satisfied."""
    geometry = Rectangle((0.0, 0.0), (1.0, 1.0), ("t", "x"))
    backend = AutogradDerivativeBackend()
    model = CoordinateModel()

    initial = InitialCondition(
        name="ic",
        time_dim=0,
        time_value=0.0,
        target=lambda coords: 2.0 * coords[:, [1]],
        default_sample_count=8,
    )
    assert initial.loss(model, geometry, backend).value.item() == pytest.approx(0.0)

    boundary = DirichletBoundary(
        name="right",
        boundary_dim=1,
        boundary_value=1.0,
        target=lambda coords: coords[:, [0]] + 2.0,
        default_sample_count=8,
    )
    assert boundary.loss(model, geometry, backend).value.item() == pytest.approx(0.0)

    neumann = NeumannBoundary(
        name="du_dt",
        boundary_dim=0,
        boundary_value=0.0,
        derivative_dim=0,
        target=1.0,
        default_sample_count=8,
    )
    assert neumann.loss(model, geometry, backend).value.item() == pytest.approx(0.0)


def test_periodic_boundary_matches_opposite_faces() -> None:
    """Periodic constraints compare values at paired opposite faces."""

    class PeriodicModel(nn.Module):
        """Field that is periodic in the spatial coordinate."""

        def forward(self, coordinates: Tensor) -> Tensor:
            """Evaluate a periodic scalar field."""
            return torch.sin(2.0 * np.pi * coordinates[:, [1]])

    geometry = Rectangle((0.0, 0.0), (1.0, 1.0), ("t", "x"))
    constraint = PeriodicBoundary(name="periodic_x", periodic_dim=1)
    loss = constraint.loss(PeriodicModel(), geometry, AutogradDerivativeBackend())
    assert loss.value.item() < 1e-12


def test_heat_problem_residual_vanishes_on_exact_solution() -> None:
    """The heat problem residual is zero on the analytic solution."""
    problem = HeatProblem(HeatConfig(alpha=0.2), n_constraint_samples=8)
    coords = problem.geometry.sample_interior(32, dtype=torch.float64)
    coords.requires_grad_(True)
    residual = StrongFormResidual().evaluate(
        problem,
        AnalyticHeatModel(alpha=0.2),
        coords,
        AutogradDerivativeBackend(),
    )
    assert torch.max(torch.abs(residual)).item() < 1e-8


def test_wave_problem_residual_vanishes_on_exact_solution() -> None:
    """The wave problem residual is zero on the analytic solution."""
    problem = WaveProblem(WaveConfig(c=1.5), t_final=0.5, n_constraint_samples=8)
    coords = problem.geometry.sample_interior(32, dtype=torch.float64)
    coords.requires_grad_(True)
    residual = StrongFormResidual().evaluate(
        problem,
        AnalyticWaveModel(c=1.5),
        coords,
        AutogradDerivativeBackend(),
    )
    assert torch.max(torch.abs(residual)).item() < 1e-8


def test_burgers_problem_matches_composable_problem_contract() -> None:
    """The Burgers adapter exposes residuals, constraints and reference values."""
    config = BurgersConfig(tmax=0.2, nu=0.01 / np.pi)
    problem = BurgersProblem(config, n_constraint_samples=8)
    coords = torch.tensor([[0.0, -0.5], [0.0, 0.0], [0.0, 0.5]], dtype=torch.float64)

    residual = StrongFormResidual().evaluate(
        problem,
        ZeroModel(),
        coords.clone().detach().requires_grad_(True),
        AutogradDerivativeBackend(),
    )
    initial = problem.initial_condition(coords)
    exact = problem.exact(np.array([0.0]), np.array([0.5]))

    assert residual.shape == (3, 1)
    assert torch.max(torch.abs(residual)).item() == pytest.approx(0.0)
    assert torch.allclose(initial, -torch.sin(torch.pi * coords[:, [1]]))
    assert exact[0] == pytest.approx(-1.0)
    assert [constraint.name for constraint in problem.constraints()] == [
        "ic",
        "bc_left",
        "bc_right",
    ]


def test_generic_trainer_runs_core_problems_without_equation_specific_api() -> None:
    """One generic trainer can optimize Burgers, heat and wave problem instances."""
    config = TrainerConfig(
        seed=1,
        dtype=torch.float64,
        collocation_count=16,
        constraint_sample_counts={
            "ic": 8,
            "initial_velocity": 8,
            "bc_left": 8,
            "bc_right": 8,
        },
        optimizer=OptimizerConfig(adam_steps=3, lbfgs_max_iter=0, lr=1e-3),
        log_every=1,
    )
    trainer = Trainer(config)

    burgers = BurgersProblem(BurgersConfig(tmax=0.1), n_constraint_samples=8)
    burgers_model = MLP(2, 1, 8, 1)
    burgers_result = trainer.train(burgers, burgers_model)

    heat = HeatProblem(HeatConfig(alpha=0.1), n_constraint_samples=8)
    heat_model = MLP(2, 1, 8, 1)
    heat_result = trainer.train(heat, heat_model)

    wave = WaveProblem(WaveConfig(c=1.0), t_final=0.2, n_constraint_samples=8)
    wave_model = MLP(2, 1, 8, 1)
    wave_result = trainer.train(wave, wave_model)

    assert burgers_result.final_loss >= 0.0
    assert heat_result.final_loss >= 0.0
    assert wave_result.final_loss >= 0.0
    assert {"pde", "ic", "bc_left", "bc_right"} <= set(
        burgers_result.final_state.named_losses
    )
    assert {"pde", "ic", "bc_left", "bc_right"} <= set(
        heat_result.final_state.named_losses
    )
    assert {"pde", "ic", "initial_velocity", "bc_left", "bc_right"} <= set(
        wave_result.final_state.named_losses
    )
    assert next(burgers_model.parameters()).dtype == torch.float64
    assert next(heat_model.parameters()).dtype == torch.float64
    assert next(wave_model.parameters()).dtype == torch.float64


def test_generic_trainer_uses_configured_interior_sampler() -> None:
    """Trainer residual points are supplied by the configured sampler."""
    sampler = RecordingSampler()
    problem = HeatProblem(HeatConfig(alpha=0.1), n_constraint_samples=4)
    model = MLP(2, 1, 8, 1)
    trainer = Trainer(
        TrainerConfig(
            seed=2,
            dtype=torch.float64,
            collocation_count=6,
            sampler=sampler,
            constraint_sample_counts={"ic": 4, "bc_left": 4, "bc_right": 4},
            optimizer=OptimizerConfig(adam_steps=0, lbfgs_max_iter=0),
        )
    )

    result = trainer.train(problem, model)

    assert result.final_loss >= 0.0
    assert sampler.calls == 1
    assert sampler.last_count == 6
    assert sampler.last_dtype == torch.float64


def test_new_training_config_rejects_invalid_options() -> None:
    """Trainer configuration reports invalid numerical options early."""
    assert isinstance(TrainerConfig().sampler, UniformInteriorSampler)
    with pytest.raises(ValueError, match="collocation_count"):
        TrainerConfig(collocation_count=0).validate()
    with pytest.raises(ValueError, match="dtype"):
        TrainerConfig(dtype="float16").validate()
    with pytest.raises(ValueError, match="non-negative"):
        TrainerConfig(loss_weights={"pde": -1.0}).validate()
    with pytest.raises(ValueError, match="sampler"):
        TrainerConfig(sampler=object()).validate()  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="candidate_count"):
        TrainerConfig(
            adaptive_refinement=ResidualAdaptiveConfig(candidate_count=0)
        ).validate()


def test_legacy_heat_wave_imports_still_work() -> None:
    """Old heat and wave solver imports remain usable."""
    heat = HeatPINN(HeatConfig(), model=MLP(2, 1, 8, 1), device="cpu")
    wave = WavePINN(WaveConfig(), model=MLP(2, 1, 8, 1), device="cpu", t_final=0.1)
    cfg = LegacyExactTrainConfig(adam_steps=1, n_f=8, n_ic=4, n_bc=4, log_every=1)
    assert heat.train(cfg)
    assert wave.train(cfg)

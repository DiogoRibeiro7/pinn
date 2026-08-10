"""Tests for physical scaling and dtype handling in the core trainer."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from pinnlab.models import MLP
from pinnlab.problems import HeatProblem
from pinnlab.residuals import AutogradDerivativeBackend, StrongFormResidual
from pinnlab.scaling import CharacteristicScales, Nondimensionalizer, ScaledModel
from pinnlab.solvers.heat import HeatConfig
from pinnlab.training import OptimizerConfig, Trainer, TrainerConfig


class QuadraticDimensionlessModel(nn.Module):
    """Dimensionless scalar model ``u* = x*^2``."""

    def forward(self, coordinates: Tensor) -> Tensor:
        """Evaluate the quadratic field."""
        return coordinates[:, [0]] ** 2


class DimensionlessHeatModel(nn.Module):
    """Dimensionless heat solution for a scaled ``[0, T] x [0, L]`` problem."""

    def __init__(self, *, alpha: float, time_scale: float, length_scale: float) -> None:
        """Store physical scales needed for the dimensionless heat coefficient."""
        super().__init__()
        self.dimensionless_alpha = alpha * time_scale / (length_scale**2)

    def forward(self, coordinates: Tensor) -> Tensor:
        """Evaluate ``exp(-alpha* pi^2 t*) sin(pi x*)``."""
        t_star = coordinates[:, [0]]
        x_star = coordinates[:, [1]]
        return torch.exp(-self.dimensionless_alpha * np.pi**2 * t_star) * torch.sin(
            np.pi * x_star
        )


class RecordingModel(nn.Module):
    """Small trainable model that records the dtype of its inputs."""

    def __init__(self) -> None:
        """Create one scalar parameter so optimizers have state to update."""
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[0.5]]))
        self.last_input_dtype: torch.dtype | None = None

    def forward(self, coordinates: Tensor) -> Tensor:
        """Record input dtype and return a scalar field."""
        self.last_input_dtype = coordinates.dtype
        return self.weight.to(dtype=coordinates.dtype) * coordinates[:, [1]]


def test_dimensionless_round_trips_inputs_outputs_and_parameters() -> None:
    """Dimensional and dimensionless transforms are inverse maps."""
    scales = CharacteristicScales(
        input_scales=(2.0, 4.0),
        output_scales=(3.0,),
        input_offsets=(10.0, -1.0),
        output_offsets=(5.0,),
        parameter_scales={"alpha": 0.25},
    )
    transform = Nondimensionalizer(scales)
    x = torch.tensor([[10.0, -1.0], [12.0, 3.0]], dtype=torch.float64)
    u = torch.tensor([[5.0], [8.0]], dtype=torch.float64)

    x_star = transform.to_dimensionless_inputs(x)
    u_star = transform.to_dimensionless_outputs(u)

    torch.testing.assert_close(transform.to_dimensional_inputs(x_star), x)
    torch.testing.assert_close(transform.to_dimensional_outputs(u_star), u)
    torch.testing.assert_close(transform.to_dimensionless(x, kind="input"), x_star)
    torch.testing.assert_close(transform.to_dimensional(u_star, kind="output"), u)
    assert transform.parameter_to_dimensionless(
        "alpha", torch.tensor(0.5)
    ).item() == pytest.approx(2.0)
    assert transform.parameter_to_dimensional(
        "alpha", torch.tensor(2.0)
    ).item() == pytest.approx(0.5)


def test_derivative_scale_factors_match_physical_units() -> None:
    """Derivative scale factors implement U/L and U/L^2."""
    transform = Nondimensionalizer(
        CharacteristicScales(input_scales=(4.0,), output_scales=(3.0,))
    )
    first = transform.derivative_scale_factor(
        output_index=0, coordinate_index=0, order=1
    )
    second = transform.derivative_scale_factor(
        output_index=0, coordinate_index=0, order=2
    )
    assert first.item() == pytest.approx(3.0 / 4.0)
    assert second.item() == pytest.approx(3.0 / 16.0)


def test_scaled_model_derivatives_match_chain_rule() -> None:
    """Autograd through ``ScaledModel`` gives dimensional derivatives."""
    transform = Nondimensionalizer(
        CharacteristicScales(input_scales=(4.0,), output_scales=(3.0,))
    )
    model = ScaledModel(QuadraticDimensionlessModel(), transform)
    backend = AutogradDerivativeBackend()
    x = torch.tensor([[2.0], [4.0]], dtype=torch.float64, requires_grad=True)
    u = model(x)

    du_dx = backend.partial_derivative(u, x, coordinate_index=0, order=1)
    d2u_dx2 = backend.partial_derivative(u, x, coordinate_index=0, order=2)

    expected_first = 3.0 * 2.0 * (x / 4.0) / 4.0
    expected_second = torch.full_like(x, 2.0 * 3.0 / 16.0)
    torch.testing.assert_close(du_dx, expected_first)
    torch.testing.assert_close(d2u_dx2, expected_second)


def test_heat_residual_is_equivalent_under_nondimensionalization() -> None:
    """Scaled heat model gives a zero dimensional residual after wrapping."""
    alpha = 0.2
    time_scale = 2.0
    length_scale = 3.0
    problem = HeatProblem(
        HeatConfig(alpha=alpha, length=length_scale),
        t_final=time_scale,
        n_constraint_samples=8,
    )
    transform = Nondimensionalizer(
        CharacteristicScales(
            input_scales=(time_scale, length_scale),
            output_scales=(1.0,),
        )
    )
    model = ScaledModel(
        DimensionlessHeatModel(
            alpha=alpha, time_scale=time_scale, length_scale=length_scale
        ),
        transform,
    )
    coords = problem.geometry.sample_interior(32, dtype=torch.float64)
    coords.requires_grad_(True)
    residual = StrongFormResidual().evaluate(
        problem, model, coords, AutogradDerivativeBackend()
    )
    assert torch.max(torch.abs(residual)).item() < 1e-8


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_trainer_smoke_preserves_configured_dtype(dtype: torch.dtype) -> None:
    """Trainer supports FP32 and FP64 without silently changing model dtype."""
    problem = HeatProblem(HeatConfig(alpha=0.1), n_constraint_samples=4)
    model = MLP(2, 1, 8, 1)
    trainer = Trainer(
        TrainerConfig(
            seed=3,
            dtype=dtype,
            collocation_count=8,
            constraint_sample_counts={"ic": 4, "bc_left": 4, "bc_right": 4},
            optimizer=OptimizerConfig(adam_steps=2, lr=1e-3),
            log_every=1,
        )
    )

    result = trainer.train(problem, model)

    assert result.final_loss >= 0.0
    assert next(model.parameters()).dtype == dtype


def test_scaled_trainer_feeds_dimensionless_model_with_configured_dtype() -> None:
    """Scaling does not downcast raw model inputs in the trainer."""
    dtype = torch.float64
    transform = Nondimensionalizer(
        CharacteristicScales(input_scales=(1.0, 1.0), output_scales=(1.0,))
    )
    problem = HeatProblem(HeatConfig(alpha=0.1), n_constraint_samples=4)
    model = RecordingModel()
    trainer = Trainer(
        TrainerConfig(
            seed=4,
            dtype=dtype,
            scaling=transform,
            collocation_count=8,
            constraint_sample_counts={"ic": 4, "bc_left": 4, "bc_right": 4},
            optimizer=OptimizerConfig(adam_steps=1, lr=1e-3),
            log_every=1,
        )
    )

    trainer.train(problem, model)

    assert model.last_input_dtype == dtype
    assert next(model.parameters()).dtype == dtype

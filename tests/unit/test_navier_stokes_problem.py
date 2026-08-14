"""Tests for the composable Navier-Stokes problem.

This is the first composable problem with more than one output, so it is also
the standing check that the core has not quietly regressed to scalar
assumptions. Before it existed, ``causal_residual_loss`` flattened its residual
and rejected any system; nothing noticed, because every shipped problem was
scalar.

The decisive test here is that the residual vanishes on the closed-form
Taylor-Green solution. It is run in float64 against a stand-in model that
returns the exact fields, with a wrong-viscosity control so that "the residual
is small" cannot pass by accident.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from pinnlab.constraints import InitialCondition, PeriodicBoundary
from pinnlab.problems import NavierStokesConfig, NavierStokesProblem, taylor_green
from pinnlab.residuals import AutogradDerivativeBackend

NU = 0.01


class _ExactField(nn.Module):
    """Returns the Taylor-Green solution, so the residual must vanish on it."""

    def __init__(self, nu: float = NU) -> None:
        super().__init__()
        self.nu = nu

    def forward(self, coordinates: Tensor) -> Tensor:
        t, x, y = coordinates[:, [0]], coordinates[:, [1]], coordinates[:, [2]]
        decay = torch.exp(-2.0 * self.nu * t)
        u = torch.sin(x) * torch.cos(y) * decay
        v = -torch.cos(x) * torch.sin(y) * decay
        p = (
            0.25
            * (torch.cos(2.0 * x) + torch.cos(2.0 * y))
            * torch.exp(-4.0 * self.nu * t)
        )
        return torch.cat([u, v, p], dim=1)


def _coordinates(n: int = 256) -> Tensor:
    problem = NavierStokesProblem()
    generator = torch.Generator().manual_seed(0)
    points = problem.geometry.sample_interior(n, generator=generator)
    return points.double().requires_grad_(True)


class TestResidualVanishesOnTheExactSolution:
    @pytest.mark.parametrize(
        "index, name", [(0, "momentum_x"), (1, "momentum_y"), (2, "continuity")]
    )
    def test_component_is_zero(self, index: int, name: str):
        problem = NavierStokesProblem(config=NavierStokesConfig(nu=NU))
        residual = problem.strong_residual(
            _ExactField().double(), _coordinates(), AutogradDerivativeBackend()
        )
        largest = float(residual[:, index].detach().abs().max())
        assert largest < 1e-12, f"{name} is {largest:.3e}, not zero"

    def test_residual_has_one_column_per_equation(self):
        problem = NavierStokesProblem()
        residual = problem.strong_residual(
            _ExactField().double(), _coordinates(64), AutogradDerivativeBackend()
        )
        assert residual.shape == (64, 3)

    def test_a_wrong_viscosity_does_not_vanish(self):
        """Negative control. Without it the test above proves nothing."""
        problem = NavierStokesProblem(config=NavierStokesConfig(nu=5.0 * NU))
        residual = problem.strong_residual(
            _ExactField(NU).double(), _coordinates(), AutogradDerivativeBackend()
        )
        assert float(residual[:, 0].detach().abs().max()) > 1e-3


class TestShape:
    def test_dimensions(self):
        problem = NavierStokesProblem()
        assert problem.input_dim == 3
        assert problem.output_dim == 3

    def test_reference_is_labelled_analytic(self):
        """Unlike Allen-Cahn, this benchmark has a closed form and says so."""
        assert NavierStokesProblem().reference_kind == "analytic"

    def test_geometry_spans_the_periodic_box(self):
        problem = NavierStokesProblem()
        lower = problem.geometry.lower_bounds.tolist()
        upper = problem.geometry.upper_bounds.tolist()
        assert lower == pytest.approx([0.0, 0.0, 0.0])
        assert upper == pytest.approx([1.0, 2 * math.pi, 2 * math.pi])

    def test_a_two_output_model_is_rejected(self):
        problem = NavierStokesProblem()
        with pytest.raises(ValueError, match="3 outputs"):
            problem.strong_residual(
                nn.Linear(3, 2).double(), _coordinates(8), AutogradDerivativeBackend()
            )

    def test_wrongly_shaped_coordinates_are_rejected(self):
        problem = NavierStokesProblem()
        with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
            problem.strong_residual(
                _ExactField().double(),
                torch.rand(8, 2, dtype=torch.float64, requires_grad=True),
                AutogradDerivativeBackend(),
            )


class TestConstraints:
    def test_every_output_is_matched_across_both_periodic_directions(self):
        """Six periodic constraints: three fields, two directions."""
        periodic = [
            c
            for c in NavierStokesProblem().constraints()
            if isinstance(c, PeriodicBoundary)
        ]
        assert len(periodic) == 6
        assert {(c.periodic_dim, c.output_index) for c in periodic} == {
            (dim, index) for dim in (1, 2) for index in (0, 1, 2)
        }

    def test_pressure_is_supervised_by_default(self):
        """Measured, not assumed: see the class docstring. Leaving it free gave
        a mean-centred relative L2 of 0.98, no better than a constant."""
        names = [c.name for c in NavierStokesProblem().constraints()]
        assert "ic_p" in names

    def test_pressure_supervision_can_be_switched_off(self):
        names = [
            c.name for c in NavierStokesProblem(constrain_pressure=False).constraints()
        ]
        assert "ic_p" not in names
        assert "ic_u" in names and "ic_v" in names

    def test_initial_conditions_target_the_exact_field(self):
        problem = NavierStokesProblem()
        coords = torch.zeros(16, 3, dtype=torch.float64)
        coords[:, 1] = torch.linspace(0.0, 2 * math.pi, 16, dtype=torch.float64)
        initial = [c for c in problem.constraints() if isinstance(c, InitialCondition)]
        expected = problem.initial_state(coords)
        for index, constraint in enumerate(initial):
            torch.testing.assert_close(constraint.target(coords), expected[:, [index]])


class TestReferenceGrid:
    def test_grid_shapes_agree(self):
        grid = NavierStokesProblem().reference_grid(n_t=5, n_x=8, n_y=6)
        for key in ("t", "x", "y", "u", "v", "p"):
            assert grid[key].shape == (5, 8, 6), key

    def test_grid_excludes_the_duplicate_periodic_edge(self):
        """``xmax`` is the same physical point as ``xmin`` on a periodic box."""
        grid = NavierStokesProblem().reference_grid(n_t=2, n_x=4, n_y=4)
        assert float(grid["x"].max()) < 2 * math.pi

    def test_grid_matches_the_closed_form(self):
        problem = NavierStokesProblem()
        grid = problem.reference_grid(n_t=3, n_x=5, n_y=5)
        u, v, p = taylor_green(grid["t"], grid["x"], grid["y"], problem.config.nu)
        np.testing.assert_allclose(grid["u"], u)
        np.testing.assert_allclose(grid["v"], v)
        np.testing.assert_allclose(grid["p"], p)

    def test_a_degenerate_grid_is_rejected(self):
        with pytest.raises(ValueError, match=">= 2"):
            NavierStokesProblem().reference_grid(n_t=1)


class TestConfigValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"nu": 0.0},
            {"nu": -1.0},
            {"tmax": 0.0},
            {"xmax": -1.0},
            {"ymax": -1.0},
        ],
    )
    def test_unphysical_settings_are_rejected(self, kwargs):
        with pytest.raises(ValueError):
            NavierStokesProblem(config=NavierStokesConfig(**kwargs))

    def test_sample_count_must_be_positive(self):
        with pytest.raises(ValueError, match="n_constraint_samples"):
            NavierStokesProblem(n_constraint_samples=0)

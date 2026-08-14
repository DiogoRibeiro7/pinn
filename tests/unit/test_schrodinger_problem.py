"""Tests for the composable nonlinear Schrodinger problem.

The second multi-output problem, and the one that shows the two outputs need not
be physically distinct fields: here they are the real and imaginary parts of one
complex field. ROADMAP.md recorded this problem as blocked on "complex-valued
residuals, which the current StrongFormResidual contract does not express". It
was not blocked; the split is ordinary two-output work and belongs to the
problem, not the core.

The decisive test is that the residual vanishes on the soliton. The amplitude
matters more than it looks: ``sech(x)`` is a stationary solution with a closed
form, while ``2 sech(x)`` -- the familiar benchmark initial condition -- is a
breather with none, and leaves a residual of order 1. Both are asserted, so the
distinction cannot quietly erode.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from pinnlab.problems import (
    SOLITON_AMPLITUDE,
    SchrodingerConfig,
    SchrodingerProblem,
    soliton,
)
from pinnlab.residuals import AutogradDerivativeBackend


class _Soliton(nn.Module):
    """Returns ``sech(x) exp(i t / 2)`` split into real and imaginary parts."""

    def __init__(self, amplitude: float = SOLITON_AMPLITUDE) -> None:
        super().__init__()
        self.amplitude = amplitude

    def forward(self, coordinates: Tensor) -> Tensor:
        t, x = coordinates[:, [0]], coordinates[:, [1]]
        envelope = self.amplitude / torch.cosh(x)
        return torch.cat(
            [envelope * torch.cos(t / 2.0), envelope * torch.sin(t / 2.0)], dim=1
        )


def _coordinates(n: int = 256) -> Tensor:
    generator = torch.Generator().manual_seed(0)
    points = SchrodingerProblem().geometry.sample_interior(n, generator=generator)
    return points.double().requires_grad_(True)


class TestResidualVanishesOnTheSoliton:
    @pytest.mark.parametrize("index, name", [(0, "real"), (1, "imaginary")])
    def test_component_is_zero(self, index: int, name: str):
        residual = SchrodingerProblem().strong_residual(
            _Soliton().double(), _coordinates(), AutogradDerivativeBackend()
        )
        largest = float(residual[:, index].detach().abs().max())
        assert largest < 1e-12, f"{name} residual is {largest:.3e}, not zero"

    def test_residual_has_one_column_per_equation(self):
        residual = SchrodingerProblem().strong_residual(
            _Soliton().double(), _coordinates(32), AutogradDerivativeBackend()
        )
        assert residual.shape == (32, 2)

    def test_the_benchmark_amplitude_is_not_a_stationary_solution(self):
        """``2 sech(x)`` is a breather, and this is why it has no closed form.

        Also the negative control for the test above: without it, "the residual
        is small" would prove nothing.
        """
        residual = SchrodingerProblem().strong_residual(
            _Soliton(2.0).double(), _coordinates(), AutogradDerivativeBackend()
        )
        assert float(residual[:, 0].detach().abs().max()) > 1.0


class TestSolitonClosedForm:
    def test_profile_is_preserved_and_only_the_phase_turns(self):
        """The defining property: ``|h|`` is time-independent."""
        x = np.linspace(-5.0, 5.0, 64)
        magnitudes = []
        for t in (0.0, 0.5, 1.0, math.pi / 2):
            real, imaginary = soliton(np.full_like(x, t), x)
            magnitudes.append(np.hypot(real, imaginary))
        for later in magnitudes[1:]:
            np.testing.assert_allclose(later, magnitudes[0], atol=1e-12)

    def test_phase_advances_at_half_the_rate_of_time(self):
        real, imaginary = soliton(np.array([math.pi]), np.array([0.0]))
        # At t = pi the phase is pi / 2, so h is purely imaginary.
        assert real[0] == pytest.approx(0.0, abs=1e-12)
        assert imaginary[0] == pytest.approx(1.0, abs=1e-12)

    def test_initial_data_is_real(self):
        problem = SchrodingerProblem()
        coords = torch.zeros(16, 2, dtype=torch.float64)
        coords[:, 1] = torch.linspace(-5.0, 5.0, 16, dtype=torch.float64)
        initial = problem.initial_profile(coords)
        torch.testing.assert_close(initial[:, 1], torch.zeros(16, dtype=torch.float64))
        expected = 1.0 / torch.cosh(coords[:, 1])
        torch.testing.assert_close(initial[:, 0], expected)


class TestReferenceAvailability:
    def test_soliton_amplitude_reports_analytic(self):
        assert SchrodingerProblem().reference_kind == "analytic"

    def test_breather_amplitude_reports_unavailable(self):
        """Better than claiming a reference the problem does not have."""
        problem = SchrodingerProblem(config=SchrodingerConfig(amplitude=2.0))
        assert problem.reference_kind == "unavailable"

    def test_exact_refuses_the_breather_amplitude(self):
        problem = SchrodingerProblem(config=SchrodingerConfig(amplitude=2.0))
        with pytest.raises(ValueError, match="no closed form"):
            problem.exact(np.zeros(1), np.zeros(1))

    def test_reference_grid_refuses_the_breather_amplitude(self):
        problem = SchrodingerProblem(config=SchrodingerConfig(amplitude=2.0))
        with pytest.raises(ValueError, match="no closed form"):
            problem.reference_grid(n_t=2, n_x=2)


class TestConstraints:
    def test_both_components_are_constrained_initially_and_periodically(self):
        names = [c.name for c in SchrodingerProblem().constraints()]
        assert names == ["ic_real", "ic_imag", "periodic_real", "periodic_imag"]

    def test_no_flux_constraint_is_imposed(self):
        """``sech'`` is odd, so derivative periodicity is genuinely violated.

        Imposing it would contradict the exact solution rather than describe it.
        """
        derivative_orders = {
            getattr(c, "derivative_order", 0)
            for c in SchrodingerProblem().constraints()
        }
        assert derivative_orders == {0}

    def test_value_periodicity_holds_exactly_for_the_soliton(self):
        """Why value matching is the right condition: ``sech`` is even."""
        problem = SchrodingerProblem()
        left = 1.0 / math.cosh(problem.config.xmin)
        right = 1.0 / math.cosh(problem.config.xmax)
        assert left == pytest.approx(right, abs=1e-15)

    def test_dirichlet_would_contradict_the_solution(self):
        """Recorded because zero looks like the obvious boundary value here."""
        assert 1.0 / math.cosh(5.0) > 0.01


class TestShape:
    def test_dimensions(self):
        problem = SchrodingerProblem()
        assert problem.input_dim == 2
        assert problem.output_dim == 2

    def test_a_single_output_model_is_rejected(self):
        with pytest.raises(ValueError, match="2 outputs"):
            SchrodingerProblem().strong_residual(
                nn.Linear(2, 1).double(), _coordinates(8), AutogradDerivativeBackend()
            )

    def test_wrongly_shaped_coordinates_are_rejected(self):
        with pytest.raises(ValueError, match=r"shape \(N, 2\)"):
            SchrodingerProblem().strong_residual(
                _Soliton().double(),
                torch.rand(8, 3, dtype=torch.float64, requires_grad=True),
                AutogradDerivativeBackend(),
            )

    def test_grid_excludes_the_duplicate_periodic_edge(self):
        grid = SchrodingerProblem().reference_grid(n_t=3, n_x=8)
        assert float(grid["x"].max()) < 5.0

    def test_grid_magnitude_matches_its_components(self):
        grid = SchrodingerProblem().reference_grid(n_t=3, n_x=8)
        np.testing.assert_allclose(
            grid["magnitude"], np.hypot(grid["real"], grid["imag"])
        )


class TestConfigValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [{"amplitude": 0.0}, {"amplitude": -1.0}, {"tmax": -1.0}, {"xmax": -6.0}],
    )
    def test_unphysical_settings_are_rejected(self, kwargs):
        with pytest.raises(ValueError):
            SchrodingerProblem(config=SchrodingerConfig(**kwargs))

    def test_sample_count_must_be_positive(self):
        with pytest.raises(ValueError, match="n_constraint_samples"):
            SchrodingerProblem(n_constraint_samples=0)

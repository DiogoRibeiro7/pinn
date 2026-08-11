"""Tests for the composable Allen-Cahn problem and its numerical reference.

Allen-Cahn has no closed-form solution, so the strategy used for heat, wave and
Burgers -- evaluate the residual on the exact solution and require it to vanish
-- does not apply wholesale. Two special cases rescue most of it:

* ``u(x) = tanh(x sqrt(gamma / 2 nu))`` is an exact *stationary* solution, since
  ``nu u_xx + gamma (u - u^3) = u (1 - u^2) (gamma - 2 nu a^2)`` vanishes when
  ``a^2 = gamma / (2 nu)``. It pins down the diffusion and reaction terms.
* A spatially uniform field obeys the logistic ODE ``u' = gamma (u - u^3)``,
  whose solution is known in closed form. It pins down the time derivative.

Between them every term of the residual is checked against something exact, and
each check carries a negative control so that a residual which is merely small
cannot pass. The spectral reference is verified the same way: with ``gamma = 0``
it must reproduce the heat equation, and with ``nu = 0`` the logistic ODE.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from pinnlab.constraints import PeriodicBoundary
from pinnlab.geometry import Rectangle
from pinnlab.models.mlp import MLP
from pinnlab.problems import (
    AllenCahnConfig,
    AllenCahnProblem,
    benchmark_initial_profile,
)
from pinnlab.residuals import AutogradDerivativeBackend
from pinnlab.sampling import LatinHypercubeInteriorSampler
from pinnlab.solvers.reference import (
    allen_cahn_reference_grid,
    allen_cahn_spectral,
    relative_l2_error,
)
from pinnlab.training import OptimizerConfig, Trainer, TrainerConfig
from pinnlab.training.adaptive_refinement import ResidualAdaptiveConfig
from pinnlab.training.causal_weighting import CausalWeightConfig


class _AnalyticModel(nn.Module):
    """A stand-in network returning a closed-form field."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, tx: Tensor) -> Tensor:
        return self.fn(tx[:, 0:1], tx[:, 1:2])


def _coordinates(n: int = 20) -> Tensor:
    """Return interior ``(t, x)`` coordinates in float64."""
    t = torch.linspace(0.05, 0.95, n, dtype=torch.float64)
    x = torch.linspace(-0.9, 0.9, n, dtype=torch.float64)
    tt, xx = torch.meshgrid(t, x, indexing="ij")
    return torch.stack([tt.reshape(-1), xx.reshape(-1)], dim=1)


def _stationary_kink(nu: float, gamma: float):
    """Return the exact stationary solution ``tanh(x sqrt(gamma / 2 nu))``."""
    rate = math.sqrt(gamma / (2.0 * nu))

    def field(t: Tensor, x: Tensor) -> Tensor:
        del t
        return torch.tanh(rate * x)

    return field


def _logistic(gamma: float, initial: float):
    """Return the exact spatially uniform solution of ``u' = gamma (u - u^3)``."""

    def field(t: Tensor, x: Tensor) -> Tensor:
        growth = torch.exp(gamma * t)
        denominator = torch.sqrt(
            1.0 - initial**2 + initial**2 * torch.exp(2.0 * gamma * t)
        )
        return torch.full_like(x, 1.0) * initial * growth / denominator

    return field


class TestResidualVanishesOnExactSolutions:
    """The residual must be zero on fields that solve the equation."""

    def test_stationary_kink_has_vanishing_residual(self):
        # nu is raised well above the benchmark 1e-4 so the kink is resolved by
        # ordinary float64 arithmetic rather than by a near-step function.
        nu, gamma = 0.1, 2.0
        problem = AllenCahnProblem(config=AllenCahnConfig(nu=nu, gamma=gamma))
        model = _AnalyticModel(_stationary_kink(nu, gamma)).double()
        residual = problem.strong_residual(
            model, _coordinates(), AutogradDerivativeBackend()
        )
        assert torch.max(torch.abs(residual)) < 1e-9

    def test_stationary_kink_fails_against_the_wrong_diffusivity(self):
        """Negative control: the same field is not a solution for another nu."""
        nu, gamma = 0.1, 2.0
        model = _AnalyticModel(_stationary_kink(nu, gamma)).double()
        wrong = AllenCahnProblem(config=AllenCahnConfig(nu=2.0 * nu, gamma=gamma))
        residual = wrong.strong_residual(
            model, _coordinates(), AutogradDerivativeBackend()
        )
        assert torch.max(torch.abs(residual)) > 1e-2

    def test_uniform_logistic_field_has_vanishing_residual(self):
        """A spatially uniform field checks ``u_t`` and the reaction term."""
        gamma = 3.0
        problem = AllenCahnProblem(config=AllenCahnConfig(nu=1e-2, gamma=gamma))
        model = _AnalyticModel(_logistic(gamma, 0.3)).double()
        residual = problem.strong_residual(
            model, _coordinates(), AutogradDerivativeBackend()
        )
        assert torch.max(torch.abs(residual)) < 1e-10

    def test_uniform_logistic_field_fails_against_the_wrong_reaction(self):
        """Negative control: the growth rate has to match gamma."""
        gamma = 3.0
        model = _AnalyticModel(_logistic(gamma, 0.3)).double()
        wrong = AllenCahnProblem(config=AllenCahnConfig(nu=1e-2, gamma=1.5 * gamma))
        residual = wrong.strong_residual(
            model, _coordinates(), AutogradDerivativeBackend()
        )
        assert torch.max(torch.abs(residual)) > 1e-2

    def test_constant_phases_are_steady_states(self):
        """``u = +-1`` are the equilibria the equation drives solutions toward."""
        problem = AllenCahnProblem()
        backend = AutogradDerivativeBackend()
        for phase in (1.0, -1.0):
            model = _AnalyticModel(lambda t, x, v=phase: torch.full_like(x, v)).double()
            residual = problem.strong_residual(model, _coordinates(), backend)
            assert torch.max(torch.abs(residual)) < 1e-12

    def test_zero_field_satisfies_everything_except_the_initial_condition(self):
        """The trivial solution, pinned down because it is a real training trap.

        ``u = 0`` is the unstable equilibrium between the two phases. It solves
        the PDE exactly and matches both periodic faces exactly, so the only
        term that objects is the initial condition, on one face. That is the
        whole reason plain Adam collapses onto it here: three of the four loss
        components are already at their minimum.
        """
        problem = AllenCahnProblem()
        backend = AutogradDerivativeBackend()
        zero = _AnalyticModel(lambda t, x: torch.zeros_like(x)).double()

        residual = problem.strong_residual(zero, _coordinates(), backend)
        assert torch.max(torch.abs(residual)) == 0.0

        losses = {
            c.name: float(
                c.loss(
                    zero, problem.geometry, backend, dtype=torch.float64
                ).value.detach()
            )
            for c in problem.constraints()
        }
        assert losses["bc_periodic"] == 0.0
        assert losses["bc_periodic_flux"] == 0.0
        # Only the initial condition can tell the trivial solution apart.
        assert losses["ic"] > 0.1

    def test_residual_rejects_wrongly_shaped_coordinates(self):
        problem = AllenCahnProblem()
        model = _AnalyticModel(lambda t, x: x).double()
        bad = torch.zeros(4, 3, dtype=torch.float64)
        with pytest.raises(ValueError, match="shape"):
            problem.strong_residual(model, bad, AutogradDerivativeBackend())


class TestProblemDefinition:
    def test_dimensions_and_geometry(self):
        problem = AllenCahnProblem()
        assert problem.input_dim == 2
        assert problem.output_dim == 1
        assert problem.geometry == Rectangle((0.0, -1.0), (1.0, 1.0), ("t", "x"))

    def test_reference_is_labelled_numerical(self):
        """Milestone 3 requires every benchmark to declare its provenance."""
        assert AllenCahnProblem().reference_kind == "numerical"

    def test_constraints_cover_the_initial_and_periodic_faces(self):
        names = [c.name for c in AllenCahnProblem().constraints()]
        assert names == ["ic", "bc_periodic", "bc_periodic_flux"]

    def test_derivative_matching_can_be_disabled(self):
        problem = AllenCahnProblem(match_boundary_derivative=False)
        names = [c.name for c in problem.constraints()]
        assert names == ["ic", "bc_periodic"]

    def test_constraint_sample_count_is_propagated(self):
        problem = AllenCahnProblem(n_constraint_samples=37)
        assert all(c.default_sample_count == 37 for c in problem.constraints())

    def test_tensor_initial_condition_matches_the_numpy_profile(self):
        """The two must stay one formula, or reference and training disagree."""
        problem = AllenCahnProblem()
        x = np.linspace(-1.0, 1.0, 33)
        coordinates = torch.stack(
            [torch.zeros(33, dtype=torch.float64), torch.from_numpy(x)], dim=1
        )
        tensor_values = problem.initial_condition(coordinates).numpy().reshape(-1)
        np.testing.assert_allclose(tensor_values, benchmark_initial_profile(x))

    def test_benchmark_profile_is_continuous_around_the_circle(self):
        left, right = benchmark_initial_profile(np.array([-1.0, 1.0]))
        assert left == pytest.approx(right)
        assert left == pytest.approx(-1.0)

    def test_benchmark_profile_has_a_derivative_jump_at_the_seam(self):
        """Documented in AllenCahnProblem; pinned here so it stays documented."""
        step = 1e-6
        near_left = benchmark_initial_profile(np.array([-1.0, -1.0 + step]))
        near_right = benchmark_initial_profile(np.array([1.0 - step, 1.0]))
        slope_left = (near_left[1] - near_left[0]) / step
        slope_right = (near_right[1] - near_right[0]) / step
        assert slope_left == pytest.approx(2.0, abs=1e-4)
        assert slope_right == pytest.approx(-2.0, abs=1e-4)

    def test_custom_initial_profile_is_used(self):
        problem = AllenCahnProblem(initial_profile=lambda x: np.cos(np.pi * x))
        coordinates = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
        values = problem.initial_condition(coordinates).numpy().reshape(-1)
        np.testing.assert_allclose(values, [1.0, -1.0], atol=1e-12)

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"nu": 0.0}, "nu must be positive"),
            ({"nu": -1.0}, "nu must be positive"),
            ({"gamma": 0.0}, "gamma must be positive"),
            ({"tmax": 0.0}, "tmax must exceed tmin"),
            ({"xmax": -2.0}, "xmax must exceed xmin"),
        ],
    )
    def test_invalid_configurations_are_rejected(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            AllenCahnProblem(config=AllenCahnConfig(**kwargs))

    def test_invalid_sample_count_is_rejected(self):
        with pytest.raises(ValueError, match="n_constraint_samples"):
            AllenCahnProblem(n_constraint_samples=0)


class TestPeriodicDerivativeMatching:
    """The ``derivative_order`` extension to ``PeriodicBoundary``."""

    geometry = Rectangle((0.0, -1.0), (1.0, 1.0), ("t", "x"))

    def _loss(self, model: nn.Module, order: int) -> float:
        constraint = PeriodicBoundary(
            name="bc", periodic_dim=1, derivative_order=order, default_sample_count=16
        )
        return float(
            constraint.loss(
                model,
                self.geometry,
                AutogradDerivativeBackend(),
                dtype=torch.float64,
            ).value.detach()
        )

    def test_smoothly_periodic_field_satisfies_both_orders(self):
        model = _AnalyticModel(lambda t, x: torch.cos(torch.pi * x)).double()
        assert self._loss(model, 0) < 1e-20
        assert self._loss(model, 1) < 1e-20

    def test_value_matching_alone_admits_a_kink_at_the_seam(self):
        """The reason the flux constraint exists: ``x^2`` matches by value only.

        ``x^2`` takes the same value at both faces, so value matching is
        satisfied, but its slope is ``-2`` and ``+2`` there. Without the
        derivative constraint nothing in the loss objects to that.
        """
        model = _AnalyticModel(lambda t, x: x**2).double()
        assert self._loss(model, 0) < 1e-20
        assert self._loss(model, 1) == pytest.approx(16.0, rel=1e-6)

    def test_default_order_is_value_matching(self):
        assert PeriodicBoundary(name="bc", periodic_dim=1).derivative_order == 0

    @pytest.mark.parametrize("order", [-1, 2])
    def test_unsupported_derivative_order_is_rejected(self, order):
        with pytest.raises(ValueError, match="derivative_order"):
            PeriodicBoundary(name="bc", periodic_dim=1, derivative_order=order)


class TestSpectralReference:
    """The numerical reference, checked against the limits it must reproduce."""

    def test_zero_reaction_reproduces_the_heat_equation(self):
        """With gamma = 0 the integrating factor is the exact solution."""
        nu = 0.05
        grid = allen_cahn_spectral(
            lambda x: np.cos(np.pi * x),
            nu=nu,
            gamma=0.0,
            t_final=1.0,
            n_t=11,
            n_x=64,
            steps_per_snapshot=10,
        )
        exact = np.exp(-nu * np.pi**2 * grid["t"]) * np.cos(np.pi * grid["x"])
        assert np.max(np.abs(grid["u"] - exact)) < 1e-12

    def test_zero_diffusion_reproduces_the_logistic_ode(self):
        initial, gamma = 0.3, 2.0
        grid = allen_cahn_spectral(
            lambda x: np.full_like(x, initial),
            nu=0.0,
            gamma=gamma,
            t_final=1.0,
            n_t=11,
            n_x=8,
            steps_per_snapshot=20,
        )
        growth = np.exp(gamma * grid["t"])
        exact = (
            initial
            * growth
            / np.sqrt(1.0 - initial**2 + initial**2 * np.exp(2.0 * gamma * grid["t"]))
        )
        assert np.max(np.abs(grid["u"] - exact)) < 1e-8

    def test_time_stepping_is_fourth_order(self):
        """Halving the step must cut the error by roughly sixteen."""

        def solve(steps: int) -> np.ndarray:
            return allen_cahn_spectral(
                lambda x: 0.5 * np.cos(np.pi * x),
                nu=1e-2,
                gamma=5.0,
                t_final=0.5,
                n_t=2,
                n_x=32,
                steps_per_snapshot=steps,
            )["u"][-1]

        finest = solve(800)
        coarse = np.max(np.abs(solve(25) - finest))
        fine = np.max(np.abs(solve(50) - finest))
        assert coarse / fine > 8.0

    def test_uniform_phases_are_fixed_points(self):
        grid = allen_cahn_spectral(
            lambda x: np.ones_like(x),
            t_final=1.0,
            n_t=5,
            n_x=16,
            steps_per_snapshot=10,
        )
        assert np.max(np.abs(grid["u"] - 1.0)) < 1e-10

    def test_solution_stays_within_the_phase_bounds(self):
        """A maximum principle the true solution obeys for data in [-1, 1].

        At the default resolution the phases are resolved and the bound holds
        to 1e-5. See the companion test for what happens below it.
        """
        grid = AllenCahnProblem().reference_grid(n_t=4, n_x=512, steps_per_snapshot=60)
        assert np.max(np.abs(grid["reference"])) <= 1.0 + 1e-5

    def test_coarse_grids_overshoot_the_phases_and_refinement_fixes_it(self):
        """The resolution requirement quoted in ``allen_cahn_spectral``.

        With ``nu = 1e-4`` and ``gamma = 5`` the transition layers are about
        ``sqrt(nu / gamma) = 0.0045`` wide. A grid too coarse to resolve them
        violates the maximum principle, which is the visible symptom of an
        under-resolved reference rather than a bug in the stepper.
        """
        problem = AllenCahnProblem()
        overshoots = [
            np.max(
                np.abs(
                    problem.reference_grid(n_t=3, n_x=n, steps_per_snapshot=60)[
                        "reference"
                    ]
                )
            )
            - 1.0
            for n in (128, 512)
        ]
        assert overshoots[0] > 1e-3
        assert overshoots[1] < 1e-4

    def test_grid_shapes_and_periodic_spacing(self):
        grid = allen_cahn_spectral(
            benchmark_initial_profile, n_t=7, n_x=16, steps_per_snapshot=5
        )
        assert grid["t"].shape == grid["x"].shape == grid["u"].shape == (7, 16)
        # The grid omits xmax, which is the same point on the circle as xmin.
        assert grid["x"][0, -1] < 1.0
        spacing = np.diff(grid["x"][0])
        np.testing.assert_allclose(spacing, spacing[0])

    def test_initial_snapshot_is_the_initial_condition(self):
        grid = allen_cahn_spectral(
            benchmark_initial_profile, n_t=3, n_x=32, steps_per_snapshot=2
        )
        np.testing.assert_allclose(
            grid["u"][0], benchmark_initial_profile(grid["x"][0])
        )

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"nu": -1.0}, "nu must be non-negative"),
            ({"gamma": -1.0}, "gamma must be non-negative"),
            ({"t_final": 0.0}, "t_final must be positive"),
            ({"xmin": 1.0, "xmax": -1.0}, "xmax must exceed xmin"),
            ({"n_t": 1}, "n_t must be >= 2"),
            ({"n_x": 15}, "n_x must be even"),
            ({"n_x": 2}, "n_x must be even"),
            ({"steps_per_snapshot": 0}, "steps_per_snapshot must be >= 1"),
        ],
    )
    def test_invalid_arguments_are_rejected(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            allen_cahn_spectral(benchmark_initial_profile, **kwargs)

    def test_mismatched_initial_condition_length_is_rejected(self):
        with pytest.raises(ValueError, match="must return 16 values"):
            allen_cahn_spectral(lambda x: np.zeros(3), n_x=16, n_t=2)


class TestReferenceGridScoring:
    def test_scoring_a_perfect_prediction_gives_zero_error(self):
        problem = AllenCahnProblem()
        grid = problem.reference_grid(n_t=4, n_x=64, steps_per_snapshot=10)

        scored = problem.reference_grid(
            n_t=4,
            n_x=64,
            steps_per_snapshot=10,
            predict=lambda t, x: grid["reference"],
        )
        assert scored["relative_l2_error"] == pytest.approx(0.0, abs=1e-12)

    def test_scoring_a_wrong_prediction_gives_order_one_error(self):
        problem = AllenCahnProblem()
        scored = problem.reference_grid(
            n_t=4,
            n_x=64,
            steps_per_snapshot=10,
            predict=lambda t, x: np.zeros_like(t),
        )
        assert scored["relative_l2_error"] == pytest.approx(1.0)

    def test_grid_without_prediction_omits_the_score(self):
        grid = allen_cahn_reference_grid(
            benchmark_initial_profile, n_t=3, n_x=32, steps_per_snapshot=5
        )
        assert set(grid) == {"t", "x", "reference"}

    def test_relative_error_is_computed_against_the_reference_field(self):
        grid = allen_cahn_reference_grid(
            benchmark_initial_profile,
            n_t=3,
            n_x=32,
            steps_per_snapshot=5,
            predict=lambda t, x: np.zeros_like(t),
        )
        expected = relative_l2_error(np.zeros_like(grid["t"]), grid["reference"])
        assert grid["relative_l2_error"] == pytest.approx(expected)


class TestTrainsThroughTheGenericTrainer:
    """The point of Milestone 3: a nonlinear periodic PDE needs no new trainer."""

    def _trainer(self, steps: int) -> Trainer:
        return Trainer(
            TrainerConfig(
                seed=0,
                dtype=torch.float64,
                collocation_count=256,
                optimizer=OptimizerConfig(lr=1e-3, adam_steps=steps),
                log_every=10,
            )
        )

    def test_training_reduces_the_loss(self):
        problem = AllenCahnProblem(n_constraint_samples=64)
        model = MLP(in_dim=2, out_dim=1, hidden_layers=3, width=32).double()
        result = self._trainer(60).train(problem, model)
        assert result.final_state.total_loss < result.history[0].total_loss

    def test_every_constraint_reports_a_loss_component(self):
        problem = AllenCahnProblem(n_constraint_samples=32)
        model = MLP(in_dim=2, out_dim=1, hidden_layers=2, width=16).double()
        result = self._trainer(5).train(problem, model)
        assert set(result.final_state.named_losses) == {
            "pde",
            "ic",
            "bc_periodic",
            "bc_periodic_flux",
        }

    @pytest.mark.parametrize(
        "name, overrides",
        [
            ("causal", {"causal": CausalWeightConfig(enabled=True, n_chunks=8)}),
            (
                "rar",
                {
                    "adaptive_refinement": ResidualAdaptiveConfig(
                        candidate_count=200,
                        points_per_refinement=10,
                        refresh_every=5,
                    )
                },
            ),
            ("lhs", {"sampler": LatinHypercubeInteriorSampler()}),
        ],
    )
    def test_optional_trainer_strategies_compose_with_it(self, name, overrides):
        """Causal weighting, RAR and Latin hypercube sampling need no changes.

        These are trainer-owned strategies, so a new problem should inherit
        them for free. That only holds if the trainer really is equation
        agnostic, which is the claim this problem exists to test.
        """
        del name
        problem = AllenCahnProblem(n_constraint_samples=32)
        model = MLP(in_dim=2, out_dim=1, hidden_layers=2, width=16).double()
        trainer = Trainer(
            TrainerConfig(
                seed=0,
                dtype=torch.float64,
                collocation_count=256,
                optimizer=OptimizerConfig(lr=1e-3, adam_steps=30),
                log_every=10,
                **overrides,
            )
        )
        result = trainer.train(problem, model)
        assert result.final_state.total_loss < result.history[0].total_loss

    def test_causal_weighting_reports_its_diagnostic(self):
        """A strategy that silently did nothing would still pass the test above."""
        problem = AllenCahnProblem(n_constraint_samples=32)
        model = MLP(in_dim=2, out_dim=1, hidden_layers=2, width=16).double()
        trainer = Trainer(
            TrainerConfig(
                seed=0,
                dtype=torch.float64,
                collocation_count=256,
                optimizer=OptimizerConfig(adam_steps=10),
                log_every=5,
                causal=CausalWeightConfig(enabled=True, n_chunks=8),
            )
        )
        result = trainer.train(problem, model)
        assert "min_causal_weight" in result.final_state.diagnostics

    def test_adaptive_refinement_actually_adds_points(self):
        """Same reasoning: check RAR did work, not merely that it ran."""
        problem = AllenCahnProblem(n_constraint_samples=32)
        model = MLP(in_dim=2, out_dim=1, hidden_layers=2, width=16).double()
        trainer = Trainer(
            TrainerConfig(
                seed=0,
                dtype=torch.float64,
                collocation_count=256,
                optimizer=OptimizerConfig(adam_steps=30),
                log_every=10,
                adaptive_refinement=ResidualAdaptiveConfig(
                    candidate_count=200, points_per_refinement=10, refresh_every=5
                ),
            )
        )
        result = trainer.train(problem, model)
        assert result.final_state.diagnostics["rar_points"] > 0

    def test_predictions_can_be_scored_on_the_reference_grid(self):
        """The full loop: train briefly, then score against numerical data."""
        problem = AllenCahnProblem(n_constraint_samples=32)
        model = MLP(in_dim=2, out_dim=1, hidden_layers=2, width=16).double()
        self._trainer(10).train(problem, model)

        def predict(t: np.ndarray, x: np.ndarray) -> np.ndarray:
            coordinates = torch.tensor(
                np.stack([t.reshape(-1), x.reshape(-1)], axis=1), dtype=torch.float64
            )
            with torch.no_grad():
                return model(coordinates).numpy().reshape(t.shape)

        scored = problem.reference_grid(
            n_t=4, n_x=64, steps_per_snapshot=10, predict=predict
        )
        assert np.isfinite(scored["relative_l2_error"])

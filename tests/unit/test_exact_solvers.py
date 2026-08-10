"""Tests for the solvers that have a closed-form solution to check against.

The central check here does not train anything. Each solver's PDE residual is
evaluated on the analytic solution itself: if the residual is implemented
correctly it must vanish there, to within floating-point error. That pins down
the physics in milliseconds, where a training run takes minutes and conflates a
wrong residual with an under-trained network.

Training is exercised separately, with small budgets, only to confirm that the
optimisation actually reduces the error against the exact solution.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from pinnkit.models.mlp import MLP

from pinnkit.solvers._base import TrainConfig
from pinnkit.solvers.heat import HeatConfig, HeatPINN, heat_exact
from pinnkit.solvers.raissi_generic import latin_hypercube as generic_lhs
from pinnkit.solvers.raissi_improved import latin_hypercube as improved_lhs
from pinnkit.solvers.reference import (
    burgers_cole_hopf,
    burgers_reference_grid,
    relative_l2_error,
)
from pinnkit.solvers.wave import WaveConfig, WavePINN, wave_exact
from pinnkit.training.causal_weighting import (
    CausalWeightConfig,
    causal_residual_loss,
    causal_weights,
    temporal_chunk_index,
)


class _AnalyticModel(nn.Module):
    """A stand-in network that returns the exact solution.

    Substituting this for the trained network lets the residual be evaluated on
    the true solution, where it must be zero.
    """

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, tx: Tensor) -> Tensor:
        return self.fn(tx[:, 0:1], tx[:, 1:2])


def _grid(n: int = 24, tmax: float = 1.0, xmax: float = 1.0):
    """Return interior ``(t, x)`` column tensors in float64 with gradients on."""
    t = torch.linspace(0.05, tmax * 0.95, n, dtype=torch.float64)
    x = torch.linspace(0.05, xmax * 0.95, n, dtype=torch.float64)
    tt, xx = torch.meshgrid(t, x, indexing="ij")
    return (
        tt.reshape(-1, 1).requires_grad_(True),
        xx.reshape(-1, 1).requires_grad_(True),
    )


# ---------------------------------------------------------------------------
# Analytic solutions
# ---------------------------------------------------------------------------


class TestHeatExact:
    def test_initial_condition_is_the_sine_profile(self):
        x = np.linspace(0, 1, 21)
        got = heat_exact(np.zeros_like(x), x, alpha=0.1)
        np.testing.assert_allclose(got, np.sin(np.pi * x), atol=1e-12)

    def test_vanishes_on_the_boundaries(self):
        t = np.linspace(0, 1, 11)
        for edge in (0.0, 1.0):
            got = heat_exact(t, np.full_like(t, edge), alpha=0.1)
            np.testing.assert_allclose(got, 0.0, atol=1e-12)

    def test_decays_monotonically_in_time(self):
        t = np.linspace(0, 2, 25)
        mid = heat_exact(t, np.full_like(t, 0.5), alpha=0.1)
        assert np.all(np.diff(mid) < 0)

    def test_higher_modes_decay_faster(self):
        """Mode n decays like exp(-alpha n^2 pi^2 t), the stiffness this tests."""
        t, x = np.array([0.5]), np.array([0.5])
        first = abs(heat_exact(t, x, alpha=0.1, modes=((1, 1.0),))[0])
        fourth = abs(heat_exact(t, x, alpha=0.1, modes=((4, 1.0),))[0])
        assert fourth < first

    def test_modes_superpose(self):
        t = np.linspace(0, 1, 7)
        x = np.linspace(0, 1, 7)
        combined = heat_exact(t, x, 0.1, modes=((1, 1.0), (3, 0.4)))
        separate = heat_exact(t, x, 0.1, modes=((1, 1.0),)) + heat_exact(
            t, x, 0.1, modes=((3, 0.4),)
        )
        np.testing.assert_allclose(combined, separate, atol=1e-12)

    @pytest.mark.parametrize("kwargs", [{"alpha": 0.0}, {"alpha": -1.0}])
    def test_rejects_invalid_alpha(self, kwargs):
        with pytest.raises(ValueError, match="alpha"):
            heat_exact(np.array([0.0]), np.array([0.5]), **kwargs)

    def test_rejects_invalid_length(self):
        with pytest.raises(ValueError, match="length"):
            heat_exact(np.array([0.0]), np.array([0.5]), alpha=0.1, length=0.0)


class TestWaveExact:
    def test_initial_condition_is_the_displacement(self):
        x = np.linspace(0, 1, 21)
        got = wave_exact(np.zeros_like(x), x, c=1.0)
        np.testing.assert_allclose(got, np.sin(np.pi * x), atol=1e-12)

    def test_vanishes_on_the_boundaries(self):
        t = np.linspace(0, 2, 11)
        for edge in (0.0, 1.0):
            got = wave_exact(t, np.full_like(t, edge), c=1.0)
            np.testing.assert_allclose(got, 0.0, atol=1e-12)

    def test_is_periodic_in_time(self):
        """One period returns the field to its initial shape, unlike the heat
        equation, so nothing damps an early error away."""
        cfg = WaveConfig(c=1.0, length=1.0, modes=((1, 1.0),))
        x = np.linspace(0, 1, 21)
        start = wave_exact(np.zeros_like(x), x, cfg.c, cfg.length, cfg.modes)
        later = wave_exact(np.full_like(x, cfg.period), x, cfg.c, cfg.length, cfg.modes)
        np.testing.assert_allclose(start, later, atol=1e-12)

    def test_starts_from_rest(self):
        """u_t(0, x) = 0, checked with a centred difference in time."""
        x = np.linspace(0.05, 0.95, 15)
        h = 1e-6
        forward = wave_exact(np.full_like(x, h), x, c=1.0)
        backward = wave_exact(np.full_like(x, -h + 2 * h), x, c=1.0)  # t = +h
        # Symmetric about t=0 because cos is even, so the derivative vanishes.
        np.testing.assert_allclose(forward, backward, atol=1e-12)
        derivative = (
            wave_exact(np.full_like(x, h), x, c=1.0)
            - wave_exact(np.zeros_like(x), x, c=1.0)
        ) / h
        np.testing.assert_allclose(derivative, 0.0, atol=1e-5)

    def test_period_tracks_the_slowest_mode(self):
        assert WaveConfig(c=1.0, length=1.0, modes=((1, 1.0),)).period == pytest.approx(
            2.0
        )
        assert WaveConfig(c=2.0, length=1.0, modes=((1, 1.0),)).period == pytest.approx(
            1.0
        )

    def test_rejects_invalid_speed(self):
        with pytest.raises(ValueError, match="c must be positive"):
            wave_exact(np.array([0.0]), np.array([0.5]), c=0.0)


# ---------------------------------------------------------------------------
# The residuals must vanish on the exact solution
# ---------------------------------------------------------------------------


class TestResidualVanishesOnExactSolution:
    """If the residual is right, feeding it the true solution gives zero.

    This is the test that would catch a sign error, a missing coefficient or a
    derivative taken with respect to the wrong variable -- none of which a
    training run distinguishes from slow convergence.
    """

    def test_heat_residual(self):
        cfg = HeatConfig(alpha=0.3, length=1.0, modes=((1, 1.0), (3, 0.5)))
        solver = HeatPINN(cfg, device="cpu")
        solver.model = _AnalyticModel(
            lambda t, x: sum(
                a
                * torch.exp(-cfg.alpha * (n * np.pi) ** 2 * t)
                * torch.sin(n * np.pi * x)
                for n, a in cfg.modes
            )
        )
        t, x = _grid()
        residual = solver.residual(t, x)
        assert torch.max(torch.abs(residual)).item() < 1e-8

    def test_wave_residual(self):
        cfg = WaveConfig(c=1.7, length=1.0, modes=((1, 1.0), (2, 0.3)))
        solver = WavePINN(cfg, device="cpu")
        solver.model = _AnalyticModel(
            lambda t, x: sum(
                a * torch.cos(cfg.c * n * np.pi * t) * torch.sin(n * np.pi * x)
                for n, a in cfg.modes
            )
        )
        t, x = _grid(tmax=cfg.period)
        residual = solver.residual(t, x)
        assert torch.max(torch.abs(residual)).item() < 1e-8

    def test_heat_residual_detects_a_wrong_diffusivity(self):
        """The check has teeth: the same field fails against a different alpha."""
        solver = HeatPINN(HeatConfig(alpha=0.3), device="cpu")
        solver.model = _AnalyticModel(
            # Field built with alpha = 0.9, solver configured for 0.3.
            lambda t, x: torch.exp(-0.9 * np.pi**2 * t)
            * torch.sin(np.pi * x)
        )
        t, x = _grid()
        assert torch.max(torch.abs(solver.residual(t, x))).item() > 1e-2

    def test_relative_l2_error_is_zero_for_the_exact_solution(self):
        cfg = HeatConfig(alpha=0.2)
        solver = HeatPINN(cfg, device="cpu")
        solver.model = _AnalyticModel(
            lambda t, x: torch.exp(-cfg.alpha * np.pi**2 * t) * torch.sin(np.pi * x)
        )
        assert solver.relative_l2_error(n_t=21, n_x=21) < 1e-6


# ---------------------------------------------------------------------------
# Cole-Hopf reference for Burgers
# ---------------------------------------------------------------------------


class TestBurgersColeHopf:
    NU = 0.05

    def test_recovers_the_initial_condition(self):
        x = np.linspace(-1, 1, 41)
        got = burgers_cole_hopf(np.zeros_like(x), x, self.NU)
        np.testing.assert_allclose(got, -np.sin(np.pi * x), atol=1e-12)

    def test_is_antisymmetric(self):
        x = np.linspace(-1, 1, 41)
        t = np.full_like(x, 0.3)
        np.testing.assert_allclose(
            burgers_cole_hopf(t, x, self.NU),
            -burgers_cole_hopf(t, -x, self.NU),
            atol=1e-10,
        )

    def test_vanishes_on_the_boundaries(self):
        t = np.linspace(0.1, 1.0, 5)
        for edge in (-1.0, 1.0):
            got = burgers_cole_hopf(t, np.full_like(t, edge), self.NU)
            np.testing.assert_allclose(got, 0.0, atol=1e-10)

    def test_stays_within_the_initial_bounds(self):
        """The maximum principle: |u| never exceeds its initial maximum of 1."""
        t = np.linspace(0, 1, 25)[:, None]
        x = np.linspace(-1, 1, 101)[None, :]
        assert np.abs(burgers_cole_hopf(t, x, self.NU)).max() <= 1.0 + 1e-9

    def test_matches_an_independent_finite_difference_solve(self):
        """Cross-check against an explicit FD solve of the same problem.

        The quadrature and the time-stepper share no code, so agreement is
        real evidence rather than a restatement of the implementation.
        """
        t_final, nx = 0.3, 400
        xs = np.linspace(-1, 1, nx + 1)
        dx = xs[1] - xs[0]
        dt = 0.2 * min(dx * dx / (2 * self.NU), dx / 1.5)
        steps = int(np.ceil(t_final / dt))
        dt = t_final / steps

        u = -np.sin(np.pi * xs)
        for _ in range(steps):
            ux = (u[2:] - u[:-2]) / (2 * dx)
            uxx = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
            nxt = u.copy()
            nxt[1:-1] = u[1:-1] + dt * (-u[1:-1] * ux + self.NU * uxx)
            nxt[0] = nxt[-1] = 0.0
            u = nxt

        exact = burgers_cole_hopf(np.full_like(xs, t_final), xs, self.NU)
        assert relative_l2_error(exact, u) < 5e-4

    def test_is_converged_in_the_quadrature(self):
        """Doubling the nodes must not move the answer."""
        x = np.linspace(-0.9, 0.9, 25)
        t = np.full_like(x, 0.5)
        coarse = burgers_cole_hopf(t, x, 0.01 / np.pi, n_quad=100)
        fine = burgers_cole_hopf(t, x, 0.01 / np.pi, n_quad=200)
        np.testing.assert_allclose(coarse, fine, atol=1e-8)

    def test_handles_the_benchmark_viscosity(self):
        """nu = 0.01/pi makes F span exp(+-50); the rescaling must hold up."""
        x = np.linspace(-1, 1, 201)
        got = burgers_cole_hopf(np.full_like(x, 0.5), x, 0.01 / np.pi)
        assert np.all(np.isfinite(got))
        assert np.abs(got).max() <= 1.0 + 1e-9

    @pytest.mark.parametrize("n_quad", [1, 400])
    def test_rejects_unusable_node_counts(self, n_quad):
        """Above ~300 nodes numpy's hermgauss returns NaN instead of raising."""
        with pytest.raises(ValueError, match="n_quad"):
            burgers_cole_hopf(np.array([0.5]), np.array([0.0]), self.NU, n_quad=n_quad)

    def test_rejects_negative_time(self):
        with pytest.raises(ValueError, match="non-negative"):
            burgers_cole_hopf(np.array([-0.1]), np.array([0.0]), self.NU)

    def test_rejects_invalid_viscosity(self):
        with pytest.raises(ValueError, match="nu"):
            burgers_cole_hopf(np.array([0.1]), np.array([0.0]), nu=0.0)

    def test_reference_grid_scores_a_prediction(self):
        result = burgers_reference_grid(
            self.NU,
            n_t=5,
            n_x=21,
            predict=lambda t, x: burgers_cole_hopf(t, x, self.NU),
        )
        assert result["relative_l2_error"] < 1e-12
        assert result["exact"].shape == (5, 21)


class TestRelativeL2Error:
    def test_zero_for_identical_inputs(self):
        a = np.linspace(1, 2, 10)
        assert relative_l2_error(a, a) == 0.0

    def test_falls_back_to_absolute_error_for_a_zero_reference(self):
        zeros = np.zeros(5)
        assert relative_l2_error(np.ones(5), zeros) == pytest.approx(np.sqrt(5))

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            relative_l2_error(np.zeros(3), np.zeros(4))


# ---------------------------------------------------------------------------
# Causal weighting
# ---------------------------------------------------------------------------


class TestCausalWeighting:
    def test_chunk_index_spans_the_windows(self):
        t = torch.linspace(0.0, 1.0, 100)
        idx = temporal_chunk_index(t, n_chunks=10, tmin=0.0, tmax=1.0)
        assert int(idx.min()) == 0
        assert int(idx.max()) == 9

    def test_chunk_index_is_non_decreasing_in_time(self):
        t = torch.linspace(0.0, 1.0, 50)
        idx = temporal_chunk_index(t, n_chunks=8, tmin=0.0, tmax=1.0)
        assert torch.all(torch.diff(idx) >= 0)

    def test_final_time_folds_into_the_last_window(self):
        idx = temporal_chunk_index(torch.tensor([1.0]), n_chunks=4, tmin=0.0, tmax=1.0)
        assert int(idx[0]) == 3

    def test_first_window_is_never_suppressed(self):
        w = causal_weights(torch.tensor([5.0, 5.0, 5.0]), eps=1.0)
        assert float(w[0]) == pytest.approx(1.0)

    def test_weights_decrease_with_accumulated_error(self):
        w = causal_weights(torch.tensor([1.0, 1.0, 1.0, 1.0]), eps=1.0)
        assert torch.all(torch.diff(w) < 0)

    def test_zero_eps_recovers_uniform_weights(self):
        w = causal_weights(torch.rand(6), eps=0.0)
        np.testing.assert_allclose(w.numpy(), np.ones(6), atol=1e-12)

    def test_weights_are_detached(self):
        losses = torch.tensor([1.0, 2.0], requires_grad=True)
        assert not causal_weights(losses, eps=1.0).requires_grad

    def test_disabled_config_gives_the_plain_mean(self):
        residual = torch.linspace(0.1, 1.0, 40).reshape(-1, 1)
        t = torch.linspace(0.0, 1.0, 40).reshape(-1, 1)
        loss, weights = causal_residual_loss(
            residual, t, CausalWeightConfig(enabled=False), 0.0, 1.0
        )
        assert loss.item() == pytest.approx(float((residual**2).mean()))
        np.testing.assert_allclose(weights.numpy(), np.ones(1))

    def test_uniform_residual_reduces_to_a_known_value(self):
        """With a constant residual r every window has loss r^2, so the total is
        the mean of the geometric weights times r^2."""
        r, n = 0.5, 8
        residual = torch.full((800, 1), r, dtype=torch.float64)
        t = torch.linspace(0.0, 1.0, 800, dtype=torch.float64).reshape(-1, 1)
        cfg = CausalWeightConfig(n_chunks=n, eps=1.0)
        loss, weights = causal_residual_loss(residual, t, cfg, 0.0, 1.0)

        expected_weights = np.exp(-1.0 * np.arange(n) * r**2)
        np.testing.assert_allclose(weights.numpy(), expected_weights, rtol=1e-9)
        assert loss.item() == pytest.approx(float((expected_weights * r**2).sum() / n))

    def test_empty_windows_do_not_produce_nan(self):
        """All points in the first half; the empty later windows must not divide
        by zero."""
        t = torch.linspace(0.0, 0.4, 50).reshape(-1, 1)
        residual = torch.ones_like(t)
        loss, weights = causal_residual_loss(
            residual, t, CausalWeightConfig(n_chunks=10), 0.0, 1.0
        )
        assert torch.isfinite(loss)
        assert torch.all(torch.isfinite(weights))

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="same points"):
            causal_residual_loss(
                torch.ones(10), torch.ones(5), CausalWeightConfig(), 0.0, 1.0
            )

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"n_chunks": 0}, "n_chunks"),
            ({"eps": -1.0}, "eps"),
            ({"tol": 0.0}, "tol"),
            ({"tol": 1.5}, "tol"),
        ],
    )
    def test_rejects_invalid_config(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            CausalWeightConfig(**kwargs).validate()

    def test_rejects_empty_time_interval(self):
        with pytest.raises(ValueError, match="tmax"):
            temporal_chunk_index(torch.zeros(4), 4, tmin=1.0, tmax=1.0)


# ---------------------------------------------------------------------------
# Regression: the generic sampler was broken for d > 1
# ---------------------------------------------------------------------------


class TestGenericLatinHypercubeRegression:
    """``raissi_generic`` shipped its own copy of ``latin_hypercube`` that left
    the interval bounds at shape ``(n,)`` instead of ``(n, 1)``. The broadcast
    against the ``(n, d)`` sample failed for every ``d > 1``, and since
    ``ContinuousPINNGeneric._make_data`` calls it with ``d=2``, the Allen-Cahn
    and Schrodinger solvers raised before training began.
    """

    @pytest.mark.parametrize("d", [1, 2, 3, 5])
    def test_samples_every_dimensionality(self, d):
        h = generic_lhs(32, d, 0.0, 1.0)
        assert h.shape == (32, d)
        assert h.min() >= 0.0 and h.max() <= 1.0

    def test_respects_the_requested_range(self):
        h = generic_lhs(64, 2, -3.0, 7.0)
        assert h.min() >= -3.0 and h.max() <= 7.0

    def test_is_the_single_shared_implementation(self):
        """One function, so the two copies cannot drift apart again."""
        assert generic_lhs is improved_lhs

    def test_stratifies_each_dimension(self):
        """Latin hypercube sampling puts exactly one point in each of the n
        equal bins, per dimension."""
        n = 50
        h = generic_lhs(n, 2, 0.0, 1.0)
        for column in range(2):
            bins = np.floor(h[:, column] * n).astype(int)
            assert len(np.unique(bins)) == n


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"alpha": 0.0}, "alpha"),
            ({"length": -1.0}, "length"),
            ({"modes": ()}, "modes"),
            ({"modes": ((0, 1.0),)}, "mode numbers"),
        ],
    )
    def test_heat_config_rejects_bad_values(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            HeatConfig(**kwargs).validate()

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"c": 0.0}, "c must be positive"),
            ({"length": 0.0}, "length"),
            ({"modes": ()}, "modes"),
        ],
    )
    def test_wave_config_rejects_bad_values(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            WaveConfig(**kwargs).validate()

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"n_f": 0}, "n_f"),
            ({"n_ic": 0}, "n_ic"),
            ({"adam_steps": -1}, "adam_steps"),
            ({"lr": 0.0}, "lr"),
            ({"lbfgs_max_iter": -5}, "lbfgs_max_iter"),
            ({"weights": (1.0, 1.0)}, "weights"),
            ({"weights": (1.0, -1.0, 1.0)}, "weights"),
        ],
    )
    def test_train_config_rejects_bad_values(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            TrainConfig(**kwargs).validate()

    def test_solver_validates_its_config_on_construction(self):
        with pytest.raises(ValueError, match="alpha"):
            HeatPINN(HeatConfig(alpha=-1.0))

    def test_wave_domain_defaults_to_one_period(self):
        solver = WavePINN(WaveConfig(c=2.0, length=1.0), device="cpu")
        assert solver.domain.tmax == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Training actually reduces the error
# ---------------------------------------------------------------------------


class TestTrainingConverges:
    """Short runs confirming the optimisation moves toward the exact solution.

    The budgets here are sized for CI, not for accuracy: a small network, a few
    hundred Adam steps and no L-BFGS, which costs a few seconds. The thresholds
    are correspondingly loose and exist to catch a solver that does not learn at
    all, not to certify accuracy. For reference, the same solvers reach 1.7e-3
    (heat) and 4.0e-2 (wave) with the budgets used in the examples, which take
    minutes rather than seconds.
    """

    def test_heat_training_reduces_the_error(self):
        solver = HeatPINN(HeatConfig(alpha=0.2), model=MLP(2, 3, 32, 1), device="cpu")
        before = solver.relative_l2_error(31, 31)
        solver.train(
            TrainConfig(
                adam_steps=800,
                lbfgs_max_iter=0,
                n_f=800,
                n_ic=64,
                n_bc=64,
                seed=0,
            )
        )
        after = solver.relative_l2_error(31, 31)
        assert after < 0.15, f"expected a trained error below 0.15, got {after:.3f}"
        assert after < 0.3 * before

    def test_wave_training_reduces_the_error(self):
        """The wave equation is materially harder than the heat equation, so
        this asserts clear improvement rather than an accuracy figure."""
        solver = WavePINN(WaveConfig(c=1.0), model=MLP(2, 3, 32, 1), device="cpu")
        before = solver.relative_l2_error(31, 31)
        solver.train(
            TrainConfig(
                adam_steps=800,
                lbfgs_max_iter=0,
                n_f=800,
                n_ic=64,
                n_bc=64,
                seed=0,
            )
        )
        after = solver.relative_l2_error(31, 31)
        assert after < 0.8 * before, f"error barely moved: {before:.3f} -> {after:.3f}"

    def test_training_records_a_loss_history(self):
        solver = HeatPINN(HeatConfig(), model=MLP(2, 2, 16, 1), device="cpu")
        history = solver.train(
            TrainConfig(adam_steps=50, lbfgs_max_iter=0, n_f=200, log_every=10)
        )
        assert history and history is solver.loss_history
        assert {"total", "ic", "bc", "pde"} <= set(history[0])
        assert history[-1]["total"] < history[0]["total"]

    def test_causal_weighting_runs_end_to_end(self):
        """Exercises the causal path through training; accuracy is compared in
        examples/advanced/causal_weighting.py, not asserted here."""
        solver = WavePINN(WaveConfig(c=1.0), model=MLP(2, 2, 16, 1), device="cpu")
        history = solver.train(
            TrainConfig(
                adam_steps=50,
                lbfgs_max_iter=0,
                n_f=400,
                log_every=25,
                causal=CausalWeightConfig(n_chunks=8, eps=1.0),
            )
        )
        assert all(np.isfinite(h["total"]) for h in history)
        assert 0.0 < history[-1]["min_causal_weight"] <= 1.0

    def test_predict_matches_the_evaluation_grid_shape(self):
        solver = HeatPINN(HeatConfig(), model=MLP(2, 2, 16, 1), device="cpu")
        tt, xx = solver.evaluation_grid(9, 7)
        assert solver.predict(tt, xx).shape == (9, 7)

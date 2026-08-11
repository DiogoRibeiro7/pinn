"""Tests for the discrete-time Runge-Kutta PINN solver.

``DiscreteRKPINN`` is part of the public surface that ROADMAP.md commits to
keeping importable, but until now only its importability was tested -- nothing
ever called ``train_one_step``. It did not work. Two defects sat in it, found
by turning the type checker on the module:

* ``_compute_derivatives`` re-created its coordinate tensor with
  ``x.clone().detach().requires_grad_(True)`` *after* the prediction had been
  made from the original, so ``autograd.grad`` was asked for the gradient with
  respect to a tensor that was not in the graph. Every call raised
  ``RuntimeError``, for every tableau. The solver could not take a single step.
* Once that was fixed, a one-stage tableau -- forward Euler is a legitimate one
  -- crashed with ``AttributeError: 'float' object has no attribute 'item'``,
  because the smoothness term was initialised to ``0.0`` and the loop that
  would have made it a tensor never ran.

These tests exist so neither can come back silently.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pinnlab.models.mlp import MLP
from pinnlab.solvers.raissi_improved import (
    BurgersConfig,
    DiscreteRKPINN,
    RKTableau,
    rk4_tableau,
)


def _initial_condition(x: np.ndarray) -> np.ndarray:
    return -np.sin(np.pi * x).astype(np.float32)


def _solver(tableau: RKTableau, seed: int = 0) -> DiscreteRKPINN:
    torch.manual_seed(seed)
    return DiscreteRKPINN(
        stage_net=MLP(in_dim=2, out_dim=1, hidden_layers=2, width=16),
        device="cpu",
        cfg=BurgersConfig(),
        tableau=tableau,
    )


def _euler_tableau() -> RKTableau:
    """Forward Euler: the smallest legitimate explicit RK tableau."""
    return RKTableau(A=np.zeros((1, 1)), b=np.array([1.0]), c=np.array([0.0]))


def _collocation(n: int = 32) -> np.ndarray:
    return np.linspace(-1.0, 1.0, n).reshape(-1, 1).astype(np.float32)


class TestTrainingRuns:
    """The solver must actually take a step. It previously could not."""

    @pytest.mark.parametrize(
        "tableau_factory, stages",
        [(rk4_tableau, 4), (_euler_tableau, 1)],
    )
    def test_train_one_step_completes(self, tableau_factory, stages):
        solver = _solver(tableau_factory())
        assert len(solver.tableau.b) == stages

        history = solver.train_one_step(
            x_colloc=_collocation(),
            t0=0.0,
            t1=0.01,
            u0_fn=_initial_condition,
            steps=3,
        )
        assert len(history["total"]) == 3
        assert all(np.isfinite(history["total"]))

    def test_training_reduces_the_loss(self):
        solver = _solver(rk4_tableau())
        history = solver.train_one_step(
            x_colloc=_collocation(),
            t0=0.0,
            t1=0.01,
            u0_fn=_initial_condition,
            steps=25,
        )
        assert history["total"][-1] < history["total"][0]

    def test_single_stage_tableau_reports_zero_smoothness(self):
        """The regression: one stage means no inter-stage differences exist.

        The value being reportable at all is the point -- it used to be a bare
        float and ``.item()`` raised ``AttributeError``.
        """
        solver = _solver(_euler_tableau())
        history = solver.train_one_step(
            x_colloc=_collocation(),
            t0=0.0,
            t1=0.01,
            u0_fn=_initial_condition,
            steps=2,
        )
        assert history["smoothness"] == [0.0, 0.0]

    def test_multi_stage_tableau_reports_nonzero_smoothness(self):
        """Negative control for the test above."""
        solver = _solver(rk4_tableau())
        history = solver.train_one_step(
            x_colloc=_collocation(),
            t0=0.0,
            t1=0.01,
            u0_fn=_initial_condition,
            steps=2,
        )
        assert history["smoothness"][-1] > 0.0

    def test_backwards_time_step_is_rejected(self):
        solver = _solver(rk4_tableau())
        with pytest.raises(ValueError, match="greater than initial time"):
            solver.train_one_step(
                x_colloc=_collocation(),
                t0=0.5,
                t1=0.5,
                u0_fn=_initial_condition,
                steps=1,
            )


class TestDerivativeHelper:
    """``_compute_derivatives`` must differentiate the tensor it was given."""

    def test_derivatives_of_a_known_field_are_correct(self):
        """On ``u = x^3``, ``u_x = 3x^2`` and ``u_xx = 6x``."""
        solver = _solver(rk4_tableau())
        x = torch.linspace(-1.0, 1.0, 16).reshape(-1, 1).requires_grad_(True)
        u = x**3

        u_x, u_xx = solver._compute_derivatives(u, x)

        torch.testing.assert_close(u_x, 3.0 * x**2)
        torch.testing.assert_close(u_xx, 6.0 * x)

    def test_a_detached_coordinate_tensor_is_rejected(self):
        """The exact mistake that broke the solver, now caught with a message.

        Passing coordinates that the prediction did not come from used to fail
        deep inside autograd with "One of the differentiated Tensors appears to
        not have been used in the graph".
        """
        solver = _solver(rk4_tableau())
        x = torch.linspace(-1.0, 1.0, 8).reshape(-1, 1).requires_grad_(True)
        u = x**2

        with pytest.raises(ValueError, match="must require grad"):
            solver._compute_derivatives(u, x.clone().detach())

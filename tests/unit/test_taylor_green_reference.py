"""Verify the Taylor-Green vortex actually solves Navier-Stokes.

``tgv_u``, ``tgv_v`` and ``tgv_p`` are advertised as the analytic solution of
the 2-D incompressible Navier-Stokes equations, and are used two ways inside
``NavierStokesPINN``: to supervise the initial condition, and to score the
trained network. Nothing checked that they satisfy the equations.

``tgv_p`` returned the wrong sign. With the momentum residual the solver
actually minimises -- ``u_t + u u_x + v u_y + p_x - nu lap u`` -- the shipped
pressure left a residual of order 1 rather than zero. Because pressure is
supervised at ``t = 0``, the initial condition pulled the network toward one
sign while its own PDE residual required the other, and momentum couples
pressure to both velocity components, so the conflict was not confined to
``p``.

The velocities were always correct: continuity holds for them regardless of the
pressure, which is why the notebook and example -- which score ``u`` and ``v``
only -- were unaffected.

These tests differentiate the closed-form solution with autograd and assert
each residual vanishes, in float64 so "vanishes" means 1e-12 rather than
"small enough to hide a sign error".
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from pinnlab.solvers.navier_stokes import tgv_p, tgv_u, tgv_v

NU = 0.01
TOLERANCE = 1e-12


def _sample(n: int = 128) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random interior points of the periodic box, requiring grad."""
    generator = torch.Generator().manual_seed(0)
    t = torch.rand(n, 1, generator=generator, dtype=torch.float64)
    x = torch.rand(n, 1, generator=generator, dtype=torch.float64) * 2 * math.pi
    y = torch.rand(n, 1, generator=generator, dtype=torch.float64) * 2 * math.pi
    return (t.requires_grad_(True), x.requires_grad_(True), y.requires_grad_(True))


def _analytic_fields(t, x, y):
    """The same closed form the module ships, built differentiably.

    Kept in lockstep with ``tgv_*`` by ``test_matches_the_shipped_numpy_form``
    below, so a change to one without the other is caught.
    """
    decay = torch.exp(-2.0 * NU * t)
    u = torch.sin(x) * torch.cos(y) * decay
    v = -torch.cos(x) * torch.sin(y) * decay
    p = 0.25 * (torch.cos(2.0 * x) + torch.cos(2.0 * y)) * torch.exp(-4.0 * NU * t)
    return u, v, p


def _d(f: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(f, w, torch.ones_like(f), create_graph=True)[0]


def _residuals():
    t, x, y = _sample()
    u, v, p = _analytic_fields(t, x, y)
    u_t, u_x, u_y = _d(u, t), _d(u, x), _d(u, y)
    v_t, v_x, v_y = _d(v, t), _d(v, x), _d(v, y)
    p_x, p_y = _d(p, x), _d(p, y)
    momentum_x = u_t + u * u_x + v * u_y + p_x - NU * (_d(u_x, x) + _d(u_y, y))
    momentum_y = v_t + u * v_x + v * v_y + p_y - NU * (_d(v_x, x) + _d(v_y, y))
    continuity = u_x + v_y
    return momentum_x, momentum_y, continuity


class TestTaylorGreenSolvesTheEquations:
    @pytest.mark.parametrize(
        "index, name", [(0, "momentum_x"), (1, "momentum_y"), (2, "continuity")]
    )
    def test_residual_vanishes(self, index: int, name: str):
        residual = _residuals()[index]
        largest = float(residual.abs().max())
        assert largest < TOLERANCE, f"{name} residual is {largest:.3e}, not zero"

    def test_a_wrong_pressure_sign_is_detected(self):
        """The negative control, and the exact bug this file exists for.

        Without it, a test asserting "the residual is small" could pass on a
        solution that happens to be small everywhere sampled.
        """
        t, x, y = _sample()
        u, v, _ = _analytic_fields(t, x, y)
        wrong_p = (
            -0.25 * (torch.cos(2.0 * x) + torch.cos(2.0 * y)) * torch.exp(-4.0 * NU * t)
        )
        u_t, u_x, u_y = _d(u, t), _d(u, x), _d(u, y)
        momentum_x = (
            u_t + u * u_x + v * u_y + _d(wrong_p, x) - NU * (_d(u_x, x) + _d(u_y, y))
        )
        assert float(momentum_x.abs().max()) > 0.1, "the sign error must be visible"


class TestShippedFormsAgree:
    """The numpy functions callers use must match what was verified above."""

    @staticmethod
    def _grid():
        rng = np.random.RandomState(0)
        t = rng.uniform(0.0, 1.0, size=(64, 1))
        x = rng.uniform(0.0, 2 * math.pi, size=(64, 1))
        y = rng.uniform(0.0, 2 * math.pi, size=(64, 1))
        return t, x, y

    def test_matches_the_shipped_numpy_form(self):
        t, x, y = self._grid()
        expected_u, expected_v, expected_p = _analytic_fields(
            torch.tensor(t), torch.tensor(x), torch.tensor(y)
        )
        for got, want, name in (
            (tgv_u(t, x, y, NU), expected_u, "u"),
            (tgv_v(t, x, y, NU), expected_v, "v"),
            (tgv_p(t, x, y, NU), expected_p, "p"),
        ):
            np.testing.assert_allclose(
                got, want.detach().numpy(), rtol=1e-6, atol=1e-6, err_msg=name
            )

    def test_pressure_is_positive_at_the_origin(self):
        """A cheap, direct assertion on the sign that was wrong.

        At ``t = x = y = 0`` the pressure is ``+(1/4)(1 + 1) = +0.5``.
        """
        value = float(
            tgv_p(np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)), NU)[0, 0]
        )
        assert value == pytest.approx(0.5)

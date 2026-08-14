"""Navier-Stokes problem definition for the generic PINN trainer.

This is the first composable problem with more than one output. Heat, wave,
Burgers and Allen-Cahn are all scalar fields; this one carries three coupled
unknowns -- two velocity components and a pressure -- linked by three residuals,
one of which (continuity) contains no time derivative at all.

Multi-output was recorded in ROADMAP.md as a structural gap in the composable
core. It was not. The residual form already validated ``(N, residual_dim)``,
every constraint already selected its component with ``output_index``, and the
trainer already reduced with a shape-agnostic mean. Only
:func:`~pinnlab.training.causal_weighting.causal_residual_loss` was scalar-only,
and it now reduces components per point. This problem exists partly to keep that
honest: if the core regresses to scalar assumptions, this stops training.

The benchmark is the Taylor-Green vortex, which has a closed-form solution, so
``reference_kind`` is ``"analytic"`` rather than the ``"numerical"`` that
Allen-Cahn must settle for. That solution is verified against the equations
themselves in ``tests/unit/test_taylor_green_reference.py`` -- worth doing, as
its pressure carried a sign error until 2026-08-14.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from ..constraints import Constraint, InitialCondition, PeriodicBoundary
from ..geometry import Rectangle
from ..residuals import DerivativeBackend


@dataclass(frozen=True)
class NavierStokesConfig:
    """Domain and viscosity for the 2-D incompressible Navier-Stokes benchmark.

    The defaults are the Taylor-Green vortex on a ``2*pi``-periodic box with
    ``nu = 0.01``, matching :class:`~pinnlab.solvers.navier_stokes.NSConfig` so
    the composable and legacy paths describe the same physics.

    Args:
        nu: Kinematic viscosity; must be positive.
        tmin: Initial time.
        tmax: Final time; must exceed ``tmin``.
        xmin: Left edge of the periodic box.
        xmax: Right edge; the period in ``x`` is ``xmax - xmin``.
        ymin: Lower edge of the periodic box.
        ymax: Upper edge; the period in ``y`` is ``ymax - ymin``.
    """

    nu: float = 0.01
    tmin: float = 0.0
    tmax: float = 1.0
    xmin: float = 0.0
    xmax: float = 2.0 * math.pi
    ymin: float = 0.0
    ymax: float = 2.0 * math.pi

    def validate(self) -> None:
        """Raise ``ValueError`` when the configuration is unphysical."""
        if self.nu <= 0.0:
            raise ValueError(f"nu must be positive, got {self.nu}")
        if self.tmax <= self.tmin:
            raise ValueError(f"tmax must exceed tmin, got {self.tmin}, {self.tmax}")
        if self.xmax <= self.xmin:
            raise ValueError(f"xmax must exceed xmin, got {self.xmin}, {self.xmax}")
        if self.ymax <= self.ymin:
            raise ValueError(f"ymax must exceed ymin, got {self.ymin}, {self.ymax}")


def taylor_green(
    t: np.ndarray, x: np.ndarray, y: np.ndarray, nu: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the Taylor-Green vortex ``(u, v, p)`` at the given points.

    This is an exact solution of the 2-D incompressible Navier-Stokes equations
    in the form used here, ``u_t + (u.grad)u + grad p - nu lap u = 0`` with
    ``div u = 0``. Verified against those equations rather than asserted; see
    ``tests/unit/test_taylor_green_reference.py``.

    Note the pressure sign. With this velocity field the momentum residual
    vanishes only for ``p = +(1/4)(cos 2x + cos 2y) exp(-4 nu t)``. Pressure is
    free up to an additive constant, not up to a sign.
    """
    decay = np.exp(-2.0 * nu * t)
    u = np.sin(x) * np.cos(y) * decay
    v = -np.cos(x) * np.sin(y) * decay
    p = 0.25 * (np.cos(2.0 * x) + np.cos(2.0 * y)) * np.exp(-4.0 * nu * t)
    return u, v, p


@dataclass(frozen=True)
class NavierStokesProblem:
    """2-D incompressible Navier-Stokes on a doubly periodic space-time box.

    The unknowns are ``(u, v, p)`` and the residual has three components:

    - ``u_t + u u_x + v u_y + p_x - nu (u_xx + u_yy)``
    - ``v_t + u v_x + v v_y + p_y - nu (v_xx + v_yy)``
    - ``u_x + v_y``

    The third is the incompressibility constraint. It carries no time
    derivative, which is what makes this a differential-algebraic system rather
    than three evolution equations, and is the structural reason a
    velocity-pressure PINN behaves differently from the scalar problems here.

    **Pressure is determined only up to an additive constant.** Nothing in the
    equations fixes its level: only ``grad p`` appears, so a solution offset by
    a constant is equally correct and supervising ``p`` pointwise penalises it
    for something that does not matter.

    That argument is sound and still lost to measurement. Leaving pressure free
    was the original default here; scored mean-centred over three seeds at 1500
    Adam steps plus 100 L-BFGS, it gave a median relative L2 of ``0.98``
    (range ``0.97-1.03``) -- indistinguishable from predicting a constant --
    against ``0.64`` (range ``0.42-0.68``) with the initial condition
    supervising it. Those ranges are disjoint. Velocity was unaffected either
    way, the two sets of ranges overlapping (``u`` 0.21-0.30 against 0.21-0.28).

    So the gauge freedom is real but does not bite: absorbing a constant offset
    is cheap for the network, whereas with nothing anchoring ``p`` at this
    budget the optimiser has only ``grad p`` inside the momentum residual to go
    on, and does not converge. Supervision is the default; pass
    ``constrain_pressure=False`` for the formulation without it. Score pressure
    mean-centred whichever you choose, since the constant remains arbitrary.

    Even supervised, pressure is the hard part of this problem at a small
    budget, and neither setting should be read as solving it.

    **Weighting a system is not the same as weighting a scalar problem, and
    getting it wrong is expensive.** The trainer reduces the ``(N, 3)`` residual
    with a single mean, so each equation receives a third of
    ``loss_weights["pde"]``, while each of the nine constraints here is a
    separate named entry at its own full weight. A configuration carried over
    from a scalar problem -- every constraint at ``10``, ``pde`` left at ``1``
    -- therefore weights the constraints thirty times more heavily against the
    residual than weighting each equation and each constraint equally would, and
    the network barely optimises the PDE. Measured at 3000 Adam steps over three
    seeds, that configuration reaches a velocity relative L2 of ``0.139``,
    against ``0.035`` for the balance below::

        loss_weights = {"pde": 3.0}                       # 1 per equation
        loss_weights.update({"ic_u": 1.0, "ic_v": 1.0, "ic_p": 1.0})
        loss_weights.update(
            {c.name: 1.0 / 3.0 for c in problem.constraints()
             if c.name.startswith("periodic_")}
        )

    ``pde`` at ``3`` turns the trainer's mean back into a sum, so each equation
    carries weight one. This is a starting point rather than a tuned optimum: it
    reproduces the per-component balance of
    :class:`~pinnlab.solvers.navier_stokes.NavierStokesPINN`, which is a known
    working configuration for this benchmark, not a claim about the best one.

    Args:
        config: Physical parameters and domain.
        n_constraint_samples: Default sample count for each constraint.
        constrain_pressure: Whether the initial condition also supervises ``p``.
            On by default, on the measurement above rather than on principle.
    """

    config: NavierStokesConfig = NavierStokesConfig()
    n_constraint_samples: int = 256
    constrain_pressure: bool = True

    def __post_init__(self) -> None:
        """Validate the configuration and sample count."""
        self.config.validate()
        if self.n_constraint_samples < 1:
            raise ValueError("n_constraint_samples must be >= 1")

    @property
    def input_dim(self) -> int:
        """Return the coordinate dimension, time plus two spatial coordinates."""
        return 3

    @property
    def output_dim(self) -> int:
        """Return the output dimension, two velocity components and a pressure."""
        return 3

    @property
    def reference_kind(self) -> str:
        """Return ``"analytic"``: the Taylor-Green vortex is a closed form."""
        return "analytic"

    @property
    def geometry(self) -> Rectangle:
        """Return the ``(t, x, y)`` space-time box."""
        return Rectangle(
            (self.config.tmin, self.config.xmin, self.config.ymin),
            (self.config.tmax, self.config.xmax, self.config.ymax),
            ("t", "x", "y"),
        )

    def initial_state(self, coordinates: Tensor) -> Tensor:
        """Evaluate the Taylor-Green field at the coordinates' ``(x, y)``.

        Returns all three components; callers select one with an
        ``output_index``.
        """
        x = coordinates[:, [1]]
        y = coordinates[:, [2]]
        t = coordinates[:, [0]]
        decay = torch.exp(-2.0 * self.config.nu * t)
        u = torch.sin(x) * torch.cos(y) * decay
        v = -torch.cos(x) * torch.sin(y) * decay
        p = (
            0.25
            * (torch.cos(2.0 * x) + torch.cos(2.0 * y))
            * torch.exp(-4.0 * self.config.nu * t)
        )
        return torch.cat([u, v, p], dim=1)

    def _component(self, index: int):
        """Return a target callable selecting one component of the exact field."""

        def target(coordinates: Tensor) -> Tensor:
            return self.initial_state(coordinates)[:, [index]]

        return target

    def strong_residual(
        self,
        model: nn.Module,
        coordinates: Tensor,
        derivative_backend: DerivativeBackend,
    ) -> Tensor:
        """Return the three residual components at ``(t, x, y)``, shape ``(N, 3)``.

        Raises:
            ValueError: If ``coordinates`` is not ``(N, 3)``.
        """
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("Navier-Stokes coordinates must have shape (N, 3)")
        coords = coordinates.requires_grad_(True)
        outputs = model(coords)
        if outputs.shape[1] != 3:
            raise ValueError(
                f"model must produce 3 outputs (u, v, p), got {outputs.shape[1]}"
            )

        def d(output_index: int, coordinate_index: int, order: int = 1) -> Tensor:
            return derivative_backend.partial_derivative(
                outputs,
                coords,
                output_index=output_index,
                coordinate_index=coordinate_index,
                order=order,
            )

        u = outputs[:, [0]]
        v = outputs[:, [1]]
        nu = self.config.nu

        u_t, u_x, u_y = d(0, 0), d(0, 1), d(0, 2)
        v_t, v_x, v_y = d(1, 0), d(1, 1), d(1, 2)
        p_x, p_y = d(2, 1), d(2, 2)
        lap_u = d(0, 1, order=2) + d(0, 2, order=2)
        lap_v = d(1, 1, order=2) + d(1, 2, order=2)

        momentum_x = u_t + u * u_x + v * u_y + p_x - nu * lap_u
        momentum_y = v_t + u * v_x + v * v_y + p_y - nu * lap_v
        continuity = u_x + v_y
        return torch.cat([momentum_x, momentum_y, continuity], dim=1)

    def constraints(self) -> Sequence[Constraint]:
        """Return initial conditions and periodicity in both spatial directions.

        Periodicity is matched on value only. Unlike Allen-Cahn, the initial
        data here is smooth and genuinely periodic in both directions, so the
        derivative match that problem needs to rule out a seam kink is not
        required to make the condition well posed. It is also three times as
        expensive, since it would apply to every output.
        """
        items: list[Constraint] = [
            InitialCondition(
                name="ic_u",
                time_dim=0,
                time_value=self.config.tmin,
                target=self._component(0),
                output_index=0,
                default_sample_count=self.n_constraint_samples,
            ),
            InitialCondition(
                name="ic_v",
                time_dim=0,
                time_value=self.config.tmin,
                target=self._component(1),
                output_index=1,
                default_sample_count=self.n_constraint_samples,
            ),
        ]
        if self.constrain_pressure:
            items.append(
                InitialCondition(
                    name="ic_p",
                    time_dim=0,
                    time_value=self.config.tmin,
                    target=self._component(2),
                    output_index=2,
                    default_sample_count=self.n_constraint_samples,
                )
            )
        for dim, axis in ((1, "x"), (2, "y")):
            for index, field_name in enumerate(("u", "v", "p")):
                items.append(
                    PeriodicBoundary(
                        name=f"periodic_{axis}_{field_name}",
                        periodic_dim=dim,
                        output_index=index,
                        default_sample_count=self.n_constraint_samples,
                    )
                )
        return tuple(items)

    def exact(
        self, t: np.ndarray, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the Taylor-Green solution ``(u, v, p)``."""
        return taylor_green(t, x, y, self.config.nu)

    def reference_grid(
        self, n_t: int = 21, n_x: int = 64, n_y: int = 64
    ) -> dict[str, np.ndarray]:
        """Return a ``(t, x, y)`` grid with the exact fields on it.

        Named ``reference_grid`` for symmetry with the other problems, but this
        one is exact rather than numerical, which ``reference_kind`` reports.
        """
        if min(n_t, n_x, n_y) < 2:
            raise ValueError("n_t, n_x and n_y must each be >= 2")
        t = np.linspace(self.config.tmin, self.config.tmax, n_t)
        # Exclude the duplicate right edge: the box is periodic, so xmax is xmin.
        x = np.linspace(self.config.xmin, self.config.xmax, n_x, endpoint=False)
        y = np.linspace(self.config.ymin, self.config.ymax, n_y, endpoint=False)
        tt, xx, yy = np.meshgrid(t, x, y, indexing="ij")
        u, v, p = self.exact(tt, xx, yy)
        return {"t": tt, "x": xx, "y": yy, "u": u, "v": v, "p": p}

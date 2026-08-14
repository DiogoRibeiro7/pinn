"""Nonlinear Schrodinger problem definition for the generic PINN trainer.

The second composable problem with more than one output, and a different reason
for it than :class:`~pinnlab.problems.NavierStokesProblem`. There the unknowns
were physically distinct fields; here they are the real and imaginary parts of a
single complex field, so the two outputs are the same quantity viewed through a
representation the trainer never has to know about.

That distinction was worth checking rather than assuming. ROADMAP.md recorded
this problem as needing "complex-valued residuals, which the current
``StrongFormResidual`` contract does not express". It does not need them: the
usual real/imaginary split is an ordinary two-output problem and works today.

The equation is ``i h_t + (1/2) h_xx + |h|^2 h = 0`` with ``h = u + i v``,
giving two coupled real residuals::

    -v_t + (1/2) u_xx + (u^2 + v^2) u
     u_t + (1/2) v_xx + (u^2 + v^2) v

Unlike Allen-Cahn and Navier-Stokes, neither carries a diffusion term: this
equation is dispersive rather than dissipative, so errors are carried around the
domain rather than damped out of it.
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

#: Amplitude whose initial data is a stationary soliton with a closed form.
SOLITON_AMPLITUDE = 1.0


@dataclass(frozen=True)
class SchrodingerConfig:
    """Domain and initial amplitude for the nonlinear Schrodinger benchmark.

    The default ``amplitude = 1`` gives the fundamental soliton
    ``h(0, x) = sech(x)``, whose exact solution is ``sech(x) exp(i t / 2)``.

    ``amplitude = 2`` is the more familiar PINN benchmark, from Raissi et al.
    That initial condition is *not* a stationary soliton -- it is a breather,
    and it has no closed form -- so the problem still trains but nothing can
    score it. See :attr:`SchrodingerProblem.reference_kind`.

    Args:
        amplitude: Multiplier on the ``sech`` initial profile; must be positive.
        tmin: Initial time.
        tmax: Final time; must exceed ``tmin``. Defaults to ``pi / 2``.
        xmin: Left edge of the periodic interval.
        xmax: Right edge; the spatial period is ``xmax - xmin``.
    """

    amplitude: float = SOLITON_AMPLITUDE
    tmin: float = 0.0
    tmax: float = math.pi / 2.0
    xmin: float = -5.0
    xmax: float = 5.0

    def validate(self) -> None:
        """Raise ``ValueError`` when the configuration is unphysical."""
        if self.amplitude <= 0.0:
            raise ValueError(f"amplitude must be positive, got {self.amplitude}")
        if self.tmax <= self.tmin:
            raise ValueError(f"tmax must exceed tmin, got {self.tmin}, {self.tmax}")
        if self.xmax <= self.xmin:
            raise ValueError(f"xmax must exceed xmin, got {self.xmin}, {self.xmax}")

    @property
    def has_closed_form(self) -> bool:
        """Whether this amplitude admits the soliton solution."""
        return math.isclose(self.amplitude, SOLITON_AMPLITUDE)


def soliton(t: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(Re h, Im h)`` for ``h = sech(x) exp(i t / 2)``.

    An exact solution of ``i h_t + (1/2) h_xx + |h|^2 h = 0``: the ``sech``
    profile is preserved exactly while its phase rotates uniformly. Verified
    against the equation rather than asserted; see
    ``tests/unit/test_schrodinger_problem.py``.
    """
    envelope = 1.0 / np.cosh(x)
    return envelope * np.cos(t / 2.0), envelope * np.sin(t / 2.0)


@dataclass(frozen=True)
class SchrodingerProblem:
    """Nonlinear Schrodinger equation on a periodic space-time strip.

    The network has two outputs, the real and imaginary parts of ``h``. Nothing
    in the trainer, residual form or constraints needed changing to support a
    complex field: splitting it is enough, and the split is entirely this
    problem's business.

    **Periodicity is matched on value only, and that is exact here.** ``sech``
    is even, so ``sech(xmin) == sech(xmax)`` to the last bit at the default
    ``[-5, 5]``. The derivative is not periodic -- ``sech'`` is odd, so the two
    ends differ in sign -- which is why no flux constraint is imposed. Dirichlet
    conditions would be worse than either: ``sech(5)`` is ``0.0135``, not zero,
    so pinning the ends to zero would contradict the exact solution by more than
    a percent of the peak.

    **Scoring depends on the amplitude.** At the default ``amplitude = 1`` the
    solution is the stationary soliton and ``reference_kind`` is ``"analytic"``.
    At any other amplitude -- including ``2``, the familiar benchmark -- the
    initial data is a breather with no closed form, ``reference_kind`` is
    ``"unavailable"`` and :meth:`exact` raises. The problem still trains; there
    is simply nothing here to score it against, and saying so is better than
    quietly comparing against the wrong thing.

    Args:
        config: Physical parameters and domain.
        n_constraint_samples: Default sample count for each constraint.
    """

    config: SchrodingerConfig = SchrodingerConfig()
    n_constraint_samples: int = 256

    def __post_init__(self) -> None:
        """Validate the configuration and sample count."""
        self.config.validate()
        if self.n_constraint_samples < 1:
            raise ValueError("n_constraint_samples must be >= 1")

    @property
    def input_dim(self) -> int:
        """Return coordinate dimension: time and one spatial coordinate."""
        return 2

    @property
    def output_dim(self) -> int:
        """Return output dimension: the real and imaginary parts of ``h``."""
        return 2

    @property
    def reference_kind(self) -> str:
        """Return ``"analytic"`` for the soliton, ``"unavailable"`` otherwise."""
        return "analytic" if self.config.has_closed_form else "unavailable"

    @property
    def geometry(self) -> Rectangle:
        """Return the ``(t, x)`` space-time strip."""
        return Rectangle(
            (self.config.tmin, self.config.xmin),
            (self.config.tmax, self.config.xmax),
            ("t", "x"),
        )

    def initial_profile(self, coordinates: Tensor) -> Tensor:
        """Return ``(Re h, Im h)`` at ``t = tmin``: a real ``sech`` profile."""
        x = coordinates[:, [1]]
        envelope = self.config.amplitude / torch.cosh(x)
        return torch.cat([envelope, torch.zeros_like(envelope)], dim=1)

    def _component(self, index: int):
        """Return a target callable selecting one component of the initial data."""

        def target(coordinates: Tensor) -> Tensor:
            return self.initial_profile(coordinates)[:, [index]]

        return target

    def strong_residual(
        self,
        model: nn.Module,
        coordinates: Tensor,
        derivative_backend: DerivativeBackend,
    ) -> Tensor:
        """Return the real and imaginary residuals at ``(t, x)``, shape ``(N, 2)``.

        Raises:
            ValueError: If ``coordinates`` is not ``(N, 2)`` or the model does
                not produce two outputs.
        """
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("Schrodinger coordinates must have shape (N, 2)")
        coords = coordinates.requires_grad_(True)
        outputs = model(coords)
        if outputs.shape[1] != 2:
            raise ValueError(
                f"model must produce 2 outputs (Re h, Im h), got {outputs.shape[1]}"
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
        magnitude_squared = u**2 + v**2

        u_t, v_t = d(0, 0), d(1, 0)
        u_xx, v_xx = d(0, 1, order=2), d(1, 1, order=2)

        real = -v_t + 0.5 * u_xx + magnitude_squared * u
        imaginary = u_t + 0.5 * v_xx + magnitude_squared * v
        return torch.cat([real, imaginary], dim=1)

    def constraints(self) -> Sequence[Constraint]:
        """Return initial conditions and value periodicity for both components."""
        items: list[Constraint] = [
            InitialCondition(
                name=f"ic_{name}",
                time_dim=0,
                time_value=self.config.tmin,
                target=self._component(index),
                output_index=index,
                default_sample_count=self.n_constraint_samples,
            )
            for index, name in enumerate(("real", "imag"))
        ]
        items.extend(
            PeriodicBoundary(
                name=f"periodic_{name}",
                periodic_dim=1,
                output_index=index,
                default_sample_count=self.n_constraint_samples,
            )
            for index, name in enumerate(("real", "imag"))
        )
        return tuple(items)

    def exact(self, t: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return the exact ``(Re h, Im h)`` for the soliton amplitude.

        Raises:
            ValueError: If the amplitude is not the soliton amplitude, since no
                closed form exists for it.
        """
        if not self.config.has_closed_form:
            raise ValueError(
                f"no closed form for amplitude {self.config.amplitude}: only the "
                f"soliton amplitude {SOLITON_AMPLITUDE} has one. The initial data "
                "is a breather; score it against an independent numerical "
                "solution or not at all."
            )
        return soliton(t, x)

    def reference_grid(self, n_t: int = 41, n_x: int = 256) -> dict[str, np.ndarray]:
        """Return a ``(t, x)`` grid with the exact fields on it.

        Raises:
            ValueError: If the grid is degenerate, or the amplitude has no
                closed form.
        """
        if min(n_t, n_x) < 2:
            raise ValueError("n_t and n_x must each be >= 2")
        t = np.linspace(self.config.tmin, self.config.tmax, n_t)
        # Exclude the duplicate right edge: the strip is periodic in x.
        x = np.linspace(self.config.xmin, self.config.xmax, n_x, endpoint=False)
        tt, xx = np.meshgrid(t, x, indexing="ij")
        real, imaginary = self.exact(tt, xx)
        return {
            "t": tt,
            "x": xx,
            "real": real,
            "imag": imaginary,
            "magnitude": np.hypot(real, imaginary),
        }

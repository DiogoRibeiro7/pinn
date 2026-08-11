"""Allen-Cahn problem definition for the generic PINN trainer.

Allen-Cahn is the first problem here that is both nonlinear and periodic, which
is why it exists: heat and wave are linear with separable exact solutions, and
Burgers is nonlinear but Dirichlet. Putting this through the same
:class:`~pinnlab.training.Trainer` is the test of whether the composable core
generalises or was quietly shaped around the easy cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from ..constraints import Constraint, InitialCondition, PeriodicBoundary
from ..geometry import Rectangle
from ..residuals import DerivativeBackend
from ..solvers.reference import allen_cahn_reference_grid


@dataclass(frozen=True)
class AllenCahnConfig:
    """Physical parameters and domain for the Allen-Cahn benchmark.

    The defaults are the standard PINN benchmark values: ``nu = 1e-4`` and
    ``gamma = 5`` on ``(t, x)`` in ``[0, 1] x [-1, 1]``. That combination is
    deliberately hard. Diffusion is weak enough that the interfaces between the
    ``u = +1`` and ``u = -1`` phases stay sharp, and the reaction term is stiff
    enough that a solution which drifts off a phase gets pushed to the wrong
    one rather than being gently corrected.

    Args:
        nu: Diffusion coefficient; must be positive.
        gamma: Reaction strength multiplying ``u^3 - u``; must be positive.
        tmin: Initial time.
        tmax: Final time; must exceed ``tmin``.
        xmin: Left edge of the periodic interval.
        xmax: Right edge; the spatial period is ``xmax - xmin``.
    """

    nu: float = 1e-4
    gamma: float = 5.0
    tmin: float = 0.0
    tmax: float = 1.0
    xmin: float = -1.0
    xmax: float = 1.0

    def validate(self) -> None:
        """Raise ``ValueError`` when the configuration is unphysical."""
        if self.nu <= 0.0:
            raise ValueError(f"nu must be positive, got {self.nu}")
        if self.gamma <= 0.0:
            raise ValueError(f"gamma must be positive, got {self.gamma}")
        if self.tmax <= self.tmin:
            raise ValueError(f"tmax must exceed tmin, got {self.tmin}, {self.tmax}")
        if self.xmax <= self.xmin:
            raise ValueError(f"xmax must exceed xmin, got {self.xmin}, {self.xmax}")


def benchmark_initial_profile(x: np.ndarray) -> np.ndarray:
    """Return the benchmark initial condition ``u(0, x) = x^2 cos(pi x)``."""
    return np.asarray(x, dtype=float) ** 2 * np.cos(np.pi * np.asarray(x, dtype=float))


@dataclass(frozen=True)
class AllenCahnProblem:
    """Allen-Cahn reaction-diffusion equation on a periodic space-time strip.

    The equation is ``u_t - nu u_xx + gamma (u^3 - u) = 0``, with the benchmark
    initial condition ``u(0, x) = x^2 cos(pi x)`` and periodic boundaries.

    Periodicity is imposed as two constraints, on the value and on ``u_x``.
    Matching values alone is not periodicity: it permits a kink at the seam,
    which the residual never sees because no collocation point straddles
    ``x = xmax``. Set ``match_boundary_derivative=False`` to drop the second
    one and reproduce the weaker condition.

    One honest caveat about the benchmark initial data. ``x^2 cos(pi x)`` is
    continuous around the circle, but its derivative is not: ``u_x`` is ``2`` at
    ``x = -1`` and ``-2`` at ``x = 1``. So the derivative constraint contradicts
    the initial condition exactly at ``t = 0``, and holds for every ``t > 0``
    once diffusion has smoothed the seam. The two constraints sample disjoint
    faces -- ``t = 0`` and ``x = +-1`` -- so they meet only at the two corner
    points, a measure-zero set that random sampling does not hit. This is a
    property of the benchmark, not of the formulation.

    **The trivial solution.** ``u = 0`` satisfies this PDE exactly -- it is the
    unstable equilibrium between the two phases -- and it satisfies both
    periodic constraints exactly too. Only the initial condition objects to it,
    and only on the ``t = 0`` face. Plain Adam training therefore has a wide,
    easily reached basin that fits the initial condition and then decays to
    zero for ``t > 0``, scoring a relative L2 error near ``1.0`` while every
    loss component except the initial condition looks healthy. Measured here at
    ``nu = 1e-2`` with 1500 Adam steps: ``ic`` reached ``6e-4`` and both
    periodic losses ``2e-5``, while the mean ``|u|`` over the domain was
    ``0.07`` against a reference mean of order one.

    This is a property of Allen-Cahn, not of this implementation, and it is why
    the equation is a standard test for temporal-causality methods. If you see
    it, the fix is not more Adam steps. Reach for
    :class:`~pinnlab.training.causal_weighting.CausalWeightConfig`, which
    refuses to credit late-time residuals until early-time ones are small, or
    for a time-marching scheme.

    Args:
        config: Physical parameters and domain.
        n_constraint_samples: Default sample count for each constraint.
        match_boundary_derivative: Whether to also match ``u_x`` across the
            periodic faces.
        initial_profile: Callable giving ``u(0, x)`` for a NumPy array. Used
            both for the constraint target and for the numerical reference, so
            that the two cannot disagree.
    """

    config: AllenCahnConfig = AllenCahnConfig()
    n_constraint_samples: int = 128
    match_boundary_derivative: bool = True
    initial_profile: Callable[[np.ndarray], np.ndarray] = field(
        default=benchmark_initial_profile
    )

    def __post_init__(self) -> None:
        """Validate physical parameters and sample counts."""
        self.config.validate()
        if self.n_constraint_samples < 1:
            raise ValueError("n_constraint_samples must be >= 1")

    @property
    def input_dim(self) -> int:
        """Return coordinate dimension: time and one spatial coordinate."""
        return 2

    @property
    def output_dim(self) -> int:
        """Return scalar output dimension."""
        return 1

    @property
    def reference_kind(self) -> str:
        """Return ``"numerical"``: Allen-Cahn has no closed-form solution."""
        return "numerical"

    @property
    def geometry(self) -> Rectangle:
        """Return the Allen-Cahn space-time rectangle."""
        return Rectangle(
            (self.config.tmin, self.config.xmin),
            (self.config.tmax, self.config.xmax),
            ("t", "x"),
        )

    def initial_condition(self, coordinates: Tensor) -> Tensor:
        """Evaluate ``u(0, x)`` on tensor coordinates of shape ``(N, 2)``.

        Delegates to ``initial_profile`` so the constraint target and the
        reference solver's initial data are the same function by construction.
        """
        x = coordinates[:, [1]]
        values = self.initial_profile(x.detach().cpu().numpy().reshape(-1))
        return torch.as_tensor(
            np.asarray(values, dtype=float).reshape(-1, 1),
            device=coordinates.device,
            dtype=coordinates.dtype,
        )

    def strong_residual(
        self,
        model: nn.Module,
        coordinates: Tensor,
        derivative_backend: DerivativeBackend,
    ) -> Tensor:
        """Return ``u_t - nu u_xx + gamma (u^3 - u)`` at ``(t, x)``."""
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("Allen-Cahn coordinates must have shape (N, 2)")
        coords = coordinates.requires_grad_(True)
        outputs = model(coords)
        u = outputs[:, [0]]
        u_t = derivative_backend.partial_derivative(
            outputs, coords, output_index=0, coordinate_index=0, order=1
        )
        u_xx = derivative_backend.partial_derivative(
            outputs, coords, output_index=0, coordinate_index=1, order=2
        )
        return u_t - self.config.nu * u_xx + self.config.gamma * (u**3 - u)

    def constraints(self) -> Sequence[Constraint]:
        """Return the initial condition and the periodic boundary constraints."""
        items: list[Constraint] = [
            InitialCondition(
                name="ic",
                time_dim=0,
                time_value=self.config.tmin,
                target=self.initial_condition,
                default_sample_count=self.n_constraint_samples,
            ),
            PeriodicBoundary(
                name="bc_periodic",
                periodic_dim=1,
                default_sample_count=self.n_constraint_samples,
            ),
        ]
        if self.match_boundary_derivative:
            items.append(
                PeriodicBoundary(
                    name="bc_periodic_flux",
                    periodic_dim=1,
                    derivative_order=1,
                    default_sample_count=self.n_constraint_samples,
                )
            )
        return tuple(items)

    def reference_grid(
        self,
        *,
        n_t: int = 101,
        n_x: int = 512,
        steps_per_snapshot: int = 100,
        predict: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Return a numerical reference grid, optionally scoring a prediction.

        Unlike the ``exact`` methods on the heat, wave and Burgers problems,
        this is a discretised solution with its own error. See
        :func:`~pinnlab.solvers.reference.allen_cahn_spectral` for what that
        error depends on.

        Args:
            n_t: Number of time snapshots.
            n_x: Spatial grid points used by the spectral solver.
            steps_per_snapshot: Time steps taken between snapshots.
            predict: Optional callable taking ``(t_grid, x_grid)`` and
                returning predictions of the same shape.

        Returns:
            A dict with ``t``, ``x`` and ``reference``, plus ``predicted`` and
            ``relative_l2_error`` when ``predict`` is supplied.
        """
        return allen_cahn_reference_grid(
            self.initial_profile,
            nu=self.config.nu,
            gamma=self.config.gamma,
            t_final=self.config.tmax - self.config.tmin,
            xmin=self.config.xmin,
            xmax=self.config.xmax,
            n_t=n_t,
            n_x=n_x,
            steps_per_snapshot=steps_per_snapshot,
            predict=predict,
        )

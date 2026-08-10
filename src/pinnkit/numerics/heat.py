"""Classical finite-difference references for the 1-D heat equation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any

import numpy as np


@dataclass(frozen=True)
class HeatFiniteDifferenceSolution:
    """Explicit finite-difference solution of the 1-D heat equation."""

    t: np.ndarray
    x: np.ndarray
    u: np.ndarray
    alpha: float
    length: float
    reference_kind: str = "numerical"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate finite-difference array shapes and provenance."""
        if self.reference_kind != "numerical":
            raise ValueError("finite-difference heat solution must be numerical")
        if self.u.shape != (self.t.size, self.x.size):
            raise ValueError("u must have shape (len(t), len(x))")


def solve_heat_explicit(
    *,
    alpha: float,
    length: float,
    t_final: float,
    n_x: int,
    n_t: int,
    initial_condition: Callable[[np.ndarray], np.ndarray],
    boundary_left: float = 0.0,
    boundary_right: float = 0.0,
) -> HeatFiniteDifferenceSolution:
    """Solve ``u_t = alpha u_xx`` with a stable explicit FTCS scheme.

    The returned reference is labelled as ``numerical`` so benchmark reports do
    not conflate classical solver output with analytic or observational data.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    if t_final <= 0:
        raise ValueError(f"t_final must be positive, got {t_final}")
    if n_x < 3:
        raise ValueError("n_x must be at least 3")
    if n_t < 2:
        raise ValueError("n_t must be at least 2")

    x = np.linspace(0.0, length, n_x, dtype=float)
    t = np.linspace(0.0, t_final, n_t, dtype=float)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    cfl = alpha * dt / dx**2
    if cfl > 0.5:
        raise ValueError(
            "explicit heat solver is unstable: alpha * dt / dx^2 must be <= 0.5"
        )

    u = np.empty((n_t, n_x), dtype=float)
    u[0, :] = np.asarray(initial_condition(x), dtype=float)
    if u[0, :].shape != x.shape:
        raise ValueError("initial_condition must return an array shaped like x")
    u[0, 0] = boundary_left
    u[0, -1] = boundary_right

    for step in range(n_t - 1):
        u_next = u[step].copy()
        u_next[1:-1] = u[step, 1:-1] + cfl * (
            u[step, 2:] - 2.0 * u[step, 1:-1] + u[step, :-2]
        )
        u_next[0] = boundary_left
        u_next[-1] = boundary_right
        u[step + 1] = u_next

    return HeatFiniteDifferenceSolution(
        t=t,
        x=x,
        u=u,
        alpha=alpha,
        length=length,
        metadata={
            "method": "explicit_ftcs",
            "dx": dx,
            "dt": dt,
            "cfl": cfl,
            "boundary_left": boundary_left,
            "boundary_right": boundary_right,
        },
    )

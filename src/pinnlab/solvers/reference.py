"""Reference solutions used to score solvers quantitatively.

The Burgers solvers in this package have until now been judged by eye against a
plot. The Cole-Hopf transform gives the viscous Burgers equation an exact
solution in integral form, which turns "the shock looks about right" into a
relative L2 error.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["burgers_cole_hopf", "relative_l2_error", "burgers_reference_grid"]

# numpy's hermgauss overflows while normalising its weights past roughly this
# many nodes, returning NaN instead of raising. The quadrature converges far
# below it, so the cap costs nothing.
_MAX_QUAD_NODES = 300


def burgers_cole_hopf(
    t: np.ndarray,
    x: np.ndarray,
    nu: float,
    n_quad: int = 200,
) -> np.ndarray:
    """Evaluate the exact solution of the viscous Burgers equation.

    Solves ``u_t + u u_x = nu u_xx`` on ``x in [-1, 1]`` with the initial
    condition ``u(0, x) = -sin(pi x)`` and homogeneous Dirichlet boundaries --
    the benchmark used throughout the PINN literature.

    The Cole-Hopf transform ``u = -2 nu phi_x / phi`` turns Burgers into the heat
    equation, whose solution is a Gaussian convolution. Writing that convolution
    out gives

        u(t, x) = -N / D,
        N = integral sin(pi (x - eta)) F(x - eta) exp(-eta^2 / (4 nu t)) d eta
        D = integral F(x - eta) exp(-eta^2 / (4 nu t)) d eta
        F(y) = exp(-cos(pi y) / (2 pi nu)).

    Substituting ``eta = sqrt(4 nu t) z`` reduces both integrals to the
    Gauss-Hermite form ``integral g(z) exp(-z^2) dz``, evaluated here with
    ``n_quad`` nodes. ``F`` spans roughly ``exp(±1/(2 pi nu))``, which overflows
    for small viscosity, so the integrands are rescaled by their maximum
    exponent -- a factor that cancels in the ratio.

    Args:
        t: Times, any shape. ``t = 0`` returns the initial condition exactly.
        x: Positions, broadcastable against ``t``.
        nu: Viscosity. The literature benchmark is ``0.01 / pi``.
        n_quad: Number of Gauss-Hermite nodes. The default is ample for the
            usual viscosities -- the result is unchanged to six figures between
            100 and 200 nodes at ``nu = 0.01 / pi``. Values much above 300 are
            rejected: :func:`numpy.polynomial.hermite.hermgauss` overflows while
            computing its weights there and returns NaN rather than failing.

    Returns:
        ``u(t, x)`` with the broadcast shape of the inputs.

    Raises:
        ValueError: If ``nu`` is not positive, ``n_quad`` is outside ``[2, 300]``,
            or any time is negative.
    """
    if nu <= 0:
        raise ValueError(f"nu must be positive, got {nu}")
    if n_quad < 2:
        raise ValueError(f"n_quad must be >= 2, got {n_quad}")
    if n_quad > _MAX_QUAD_NODES:
        raise ValueError(
            f"n_quad must be <= {_MAX_QUAD_NODES}, got {n_quad}. numpy's "
            "hermgauss overflows beyond roughly this many nodes and returns NaN "
            "weights; the quadrature has long since converged by then."
        )

    t_arr, x_arr = np.broadcast_arrays(
        np.asarray(t, dtype=float), np.asarray(x, dtype=float)
    )
    if np.any(t_arr < 0):
        raise ValueError("times must be non-negative")

    out = np.empty(t_arr.shape, dtype=float)

    # t = 0 is a removable singularity of the integral form: the Gaussian
    # collapses to a delta and recovers the initial condition.
    at_zero = t_arr == 0.0
    out[at_zero] = -np.sin(np.pi * x_arr[at_zero])

    positive = ~at_zero
    if not np.any(positive):
        return out

    nodes, weights = np.polynomial.hermite.hermgauss(n_quad)

    tp = t_arr[positive][:, None]  # (P, 1)
    xp = x_arr[positive][:, None]  # (P, 1)
    y = xp - np.sqrt(4.0 * nu * tp) * nodes[None, :]  # (P, Q)

    # F(y) = exp(exponent); rescale per point so the largest term is exp(0).
    exponent = -np.cos(np.pi * y) / (2.0 * np.pi * nu)
    exponent = exponent - exponent.max(axis=1, keepdims=True)
    f = np.exp(exponent)

    weighted = weights[None, :] * f
    numerator = np.sum(np.sin(np.pi * y) * weighted, axis=1)
    denominator = np.sum(weighted, axis=1)

    out[positive] = -numerator / denominator
    return out


def relative_l2_error(
    predicted: np.ndarray,
    exact: np.ndarray,
) -> float:
    """Return ``||predicted - exact||_2 / ||exact||_2``.

    Args:
        predicted: Predicted values.
        exact: Reference values of the same shape.

    Returns:
        The relative error, or the absolute error when the reference is
        identically zero.

    Raises:
        ValueError: If the shapes disagree.
    """
    p = np.asarray(predicted, dtype=float)
    e = np.asarray(exact, dtype=float)
    if p.shape != e.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {e.shape}")
    denominator = np.linalg.norm(e)
    if denominator == 0.0:
        return float(np.linalg.norm(p - e))
    return float(np.linalg.norm(p - e) / denominator)


def burgers_reference_grid(
    nu: float,
    n_t: int = 101,
    n_x: int = 201,
    t_final: float = 1.0,
    n_quad: int = 200,
    predict: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
) -> dict:
    """Build a Burgers reference grid, and optionally score a prediction on it.

    Args:
        nu: Viscosity.
        n_t: Grid points in time.
        n_x: Grid points in space.
        t_final: End time.
        n_quad: Gauss-Hermite nodes passed to :func:`burgers_cole_hopf`.
        predict: Optional callable taking ``(t_grid, x_grid)`` and returning
            predictions of the same shape.

    Returns:
        A dict with the ``t`` and ``x`` meshes and the ``exact`` field, plus
        ``predicted`` and ``relative_l2_error`` when ``predict`` is supplied.
    """
    t = np.linspace(0.0, t_final, n_t)
    x = np.linspace(-1.0, 1.0, n_x)
    tt, xx = np.meshgrid(t, x, indexing="ij")
    exact = burgers_cole_hopf(tt, xx, nu, n_quad=n_quad)

    result = {"t": tt, "x": xx, "exact": exact}
    if predict is not None:
        predicted = np.asarray(predict(tt, xx), dtype=float)
        result["predicted"] = predicted
        result["relative_l2_error"] = relative_l2_error(predicted, exact)
    return result

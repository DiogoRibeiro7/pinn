"""Reference solutions used to score solvers quantitatively.

The Burgers solvers in this package have until now been judged by eye against a
plot. The Cole-Hopf transform gives the viscous Burgers equation an exact
solution in integral form, which turns "the shock looks about right" into a
relative L2 error.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "allen_cahn_reference_grid",
    "allen_cahn_spectral",
    "burgers_cole_hopf",
    "burgers_reference_grid",
    "relative_l2_error",
]

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

    result: Dict[str, Any] = {"t": tt, "x": xx, "exact": exact}
    if predict is not None:
        predicted = np.asarray(predict(tt, xx), dtype=float)
        result["predicted"] = predicted
        result["relative_l2_error"] = relative_l2_error(predicted, exact)
    return result


def allen_cahn_spectral(
    initial_condition: Callable[[np.ndarray], np.ndarray],
    *,
    nu: float = 1e-4,
    gamma: float = 5.0,
    t_final: float = 1.0,
    xmin: float = -1.0,
    xmax: float = 1.0,
    n_t: int = 101,
    n_x: int = 512,
    steps_per_snapshot: int = 100,
) -> Dict[str, np.ndarray]:
    """Solve periodic Allen-Cahn numerically by Fourier spectral integration.

    Allen-Cahn, ``u_t - nu u_xx + gamma (u^3 - u) = 0``, has no closed-form
    solution, so unlike :func:`burgers_cole_hopf` this reference is *numerical*:
    it is a discretisation with its own error, not ground truth. Treat its
    output as accurate to roughly the tolerances the tests pin down, not to
    machine precision.

    Space is handled by a real FFT on a uniform periodic grid, which is
    exponentially accurate for smooth data. Time uses integrating-factor RK4:
    the stiff diffusion term is advanced exactly by ``exp(-nu k^2 dt)`` and only
    the reaction term goes through the Runge-Kutta stages, so the step size is
    limited by accuracy rather than by diffusive stability. The cubic term is
    dealiased with the 2/3 rule.

    What sets ``n_x`` is the interface width. Allen-Cahn drives the field
    toward the phases ``u = +-1`` separated by transition layers of width about
    ``sqrt(nu / gamma)``, which for the benchmark ``nu = 1e-4, gamma = 5`` is
    roughly ``0.0045``. Below about ``n_x = 512`` on ``[-1, 1]`` the grid
    spacing exceeds that, the layers are not resolved, and the solution
    overshoots the phases: measured at ``t = 1``, the excess over ``|u| = 1``
    is ``1.0e-2`` at ``n_x = 128``, ``8.2e-4`` at ``256`` and ``8.3e-6`` at
    ``512``. It is discretisation error and it converges away, but a reference
    computed at low ``n_x`` is not one.

    A second, milder caveat: the benchmark initial condition ``x^2 cos(pi x)``
    is continuous around the circle but its derivative is not (``u_x`` is ``2``
    at ``x = -1`` and ``-2`` at ``x = 1``), so its Fourier coefficients decay
    like ``1/k^2`` rather than exponentially and the spatial accuracy is
    algebraic near ``t = 0``. With ``nu = 1e-4`` diffusion smooths the seam
    only over a layer of width about ``sqrt(nu t)``.

    Args:
        initial_condition: Callable mapping the spatial grid to ``u(0, x)``.
        nu: Diffusion coefficient; must be non-negative.
        gamma: Reaction strength; must be non-negative.
        t_final: End time; must be positive.
        xmin: Left edge of the periodic interval.
        xmax: Right edge; the period is ``xmax - xmin``.
        n_t: Number of snapshots returned, including ``t = 0``.
        n_x: Spatial grid points; must be even and at least four.
        steps_per_snapshot: Time steps taken between consecutive snapshots.

    Returns:
        A dict with ``t``, ``x`` and ``u`` meshes of shape ``(n_t, n_x)``. The
        grid excludes ``xmax``, which is the same point as ``xmin``.

    Raises:
        ValueError: If any parameter is out of range, or the initial condition
            does not return one value per grid point.
    """
    if nu < 0.0:
        raise ValueError(f"nu must be non-negative, got {nu}")
    if gamma < 0.0:
        raise ValueError(f"gamma must be non-negative, got {gamma}")
    if t_final <= 0.0:
        raise ValueError(f"t_final must be positive, got {t_final}")
    if xmax <= xmin:
        raise ValueError(f"xmax must exceed xmin, got {xmin} and {xmax}")
    if n_t < 2:
        raise ValueError(f"n_t must be >= 2, got {n_t}")
    if n_x < 4 or n_x % 2 != 0:
        raise ValueError(f"n_x must be even and >= 4, got {n_x}")
    if steps_per_snapshot < 1:
        raise ValueError(f"steps_per_snapshot must be >= 1, got {steps_per_snapshot}")

    length = xmax - xmin
    x = xmin + length * np.arange(n_x) / n_x
    initial = np.asarray(initial_condition(x), dtype=float).reshape(-1)
    if initial.shape != (n_x,):
        raise ValueError(
            f"initial_condition must return {n_x} values, got {initial.shape}"
        )

    wavenumbers = 2.0 * np.pi * np.fft.rfftfreq(n_x, d=length / n_x)
    linear = nu * wavenumbers**2
    # 2/3 rule: the cubic term generates modes up to three times the resolved
    # wavenumber, which would otherwise fold back onto the resolved ones.
    keep = wavenumbers <= (2.0 / 3.0) * wavenumbers[-1]

    t = np.linspace(0.0, t_final, n_t)
    dt = (t[1] - t[0]) / steps_per_snapshot
    full_step = np.exp(-linear * dt)
    half_step = np.exp(-linear * dt / 2.0)

    def reaction(spectrum: np.ndarray) -> np.ndarray:
        """Return the dealiased spectrum of ``gamma (u - u^3)``."""
        field = np.fft.irfft(spectrum, n=n_x)
        return keep * np.fft.rfft(gamma * (field - field**3))

    snapshots = np.empty((n_t, n_x), dtype=float)
    snapshots[0] = initial
    spectrum = np.fft.rfft(initial)
    for index in range(1, n_t):
        for _ in range(steps_per_snapshot):
            stage_a = reaction(spectrum)
            stage_b = reaction(half_step * (spectrum + 0.5 * dt * stage_a))
            stage_c = reaction(half_step * spectrum + 0.5 * dt * stage_b)
            stage_d = reaction(full_step * spectrum + dt * half_step * stage_c)
            spectrum = (
                full_step * spectrum
                + dt
                * (
                    full_step * stage_a
                    + 2.0 * half_step * (stage_b + stage_c)
                    + stage_d
                )
                / 6.0
            )
        snapshots[index] = np.fft.irfft(spectrum, n=n_x)

    tt, xx = np.meshgrid(t, x, indexing="ij")
    return {"t": tt, "x": xx, "u": snapshots}


def allen_cahn_reference_grid(
    initial_condition: Callable[[np.ndarray], np.ndarray],
    *,
    nu: float = 1e-4,
    gamma: float = 5.0,
    t_final: float = 1.0,
    xmin: float = -1.0,
    xmax: float = 1.0,
    n_t: int = 101,
    n_x: int = 512,
    steps_per_snapshot: int = 100,
    predict: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
) -> Dict[str, Any]:
    """Build an Allen-Cahn reference grid, and optionally score a prediction.

    This mirrors :func:`burgers_reference_grid`, with one difference worth
    stating: the ``reference`` field is produced by :func:`allen_cahn_spectral`
    and so carries discretisation error. A reported ``relative_l2_error`` below
    the reference's own accuracy is not meaningful.

    Args:
        initial_condition: Callable mapping the spatial grid to ``u(0, x)``.
        nu: Diffusion coefficient.
        gamma: Reaction strength.
        t_final: End time.
        xmin: Left edge of the periodic interval.
        xmax: Right edge of the periodic interval.
        n_t: Number of time snapshots.
        n_x: Spatial grid points.
        steps_per_snapshot: Time steps between snapshots.
        predict: Optional callable taking ``(t_grid, x_grid)`` and returning
            predictions of the same shape.

    Returns:
        A dict with the ``t`` and ``x`` meshes and the ``reference`` field, plus
        ``predicted`` and ``relative_l2_error`` when ``predict`` is supplied.
    """
    grid = allen_cahn_spectral(
        initial_condition,
        nu=nu,
        gamma=gamma,
        t_final=t_final,
        xmin=xmin,
        xmax=xmax,
        n_t=n_t,
        n_x=n_x,
        steps_per_snapshot=steps_per_snapshot,
    )
    result: Dict[str, Any] = {
        "t": grid["t"],
        "x": grid["x"],
        "reference": grid["u"],
    }
    if predict is not None:
        predicted = np.asarray(predict(grid["t"], grid["x"]), dtype=float)
        result["predicted"] = predicted
        result["relative_l2_error"] = relative_l2_error(predicted, grid["u"])
    return result

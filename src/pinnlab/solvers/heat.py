"""1-D heat equation solved with a PINN, checked against its Fourier solution.

The problem is

    u_t = alpha * u_xx,        x in [0, L],  t in [0, T]
    u(t, 0) = u(t, L) = 0
    u(0, x) = sum_n a_n sin(n pi x / L)

whose solution is a sum of decaying Fourier modes,

    u(t, x) = sum_n a_n exp(-alpha (n pi / L)^2 t) sin(n pi x / L).

Because the answer is known in closed form, a trained network can be scored
rather than merely plotted. Mode ``n`` decays like ``exp(-alpha n^2 pi^2 t / L^2)``,
so a multi-mode initial condition is a compact stiffness test: the high modes
vanish quickly and the network has to resolve very different time scales at
once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn

from ..utils.logging import get_logger
from ._base import Domain1D, ExactlySolvablePINN, TrainConfig

logger = get_logger(__name__)

__all__ = ["HeatConfig", "heat_exact", "HeatPINN", "demo_heat"]


@dataclass(frozen=True)
class HeatConfig:
    """Physical setup for the 1-D heat equation.

    Attributes:
        alpha: Thermal diffusivity. Larger values smooth the field faster.
        length: Domain length ``L``; the domain is ``[0, L]``.
        modes: ``(n, amplitude)`` pairs defining the initial condition as a sum
            of sine modes. The default is the fundamental mode alone.
    """

    alpha: float = 0.1
    length: float = 1.0
    modes: Tuple[Tuple[int, float], ...] = ((1, 1.0),)

    def validate(self) -> None:
        """Raise :class:`ValueError` if the configuration is unusable."""
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")
        if self.length <= 0:
            raise ValueError(f"length must be positive, got {self.length}")
        if not self.modes:
            raise ValueError("modes must contain at least one (n, amplitude) pair")
        for n, _ in self.modes:
            if n < 1:
                raise ValueError(f"mode numbers must be >= 1, got {n}")


def heat_exact(
    t: np.ndarray,
    x: np.ndarray,
    alpha: float,
    length: float = 1.0,
    modes: Sequence[Tuple[int, float]] = ((1, 1.0),),
) -> np.ndarray:
    """Evaluate the analytic solution of the 1-D heat equation.

    Args:
        t: Times, any shape.
        x: Positions, broadcastable against ``t``.
        alpha: Thermal diffusivity.
        length: Domain length.
        modes: ``(n, amplitude)`` pairs of the initial condition.

    Returns:
        ``u(t, x)`` with the broadcast shape of the inputs.

    Raises:
        ValueError: If ``alpha`` or ``length`` is not positive.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")

    t_arr, x_arr = np.broadcast_arrays(
        np.asarray(t, dtype=float), np.asarray(x, dtype=float)
    )
    out = np.zeros(t_arr.shape, dtype=float)
    for n, amplitude in modes:
        k = n * np.pi / length
        out = out + amplitude * np.exp(-alpha * k**2 * t_arr) * np.sin(k * x_arr)
    return out


class HeatPINN(ExactlySolvablePINN):
    """PINN for the 1-D heat equation with homogeneous Dirichlet boundaries."""

    def __init__(
        self,
        config: Optional[HeatConfig] = None,
        model: Optional[nn.Module] = None,
        domain: Optional[Domain1D] = None,
        device: Optional[Union[str, torch.device]] = None,
        t_final: float = 1.0,
    ) -> None:
        """Initialise the solver.

        Args:
            config: Physical setup. Defaults to a single mode with alpha 0.1.
            model: Network mapping ``(t, x) -> u``. Defaults to a 4x64 tanh MLP.
            domain: Space-time rectangle. Defaults to ``[0, t_final] x [0, L]``.
            device: Torch device.
            t_final: End time, used only when ``domain`` is not given.
        """
        self.config = config or HeatConfig()
        self.config.validate()
        if domain is None:
            domain = Domain1D(tmin=0.0, tmax=t_final, xmin=0.0, xmax=self.config.length)
        super().__init__(model=model, domain=domain, device=device)

    def composable_problem(self):
        """Return the equivalent :class:`~pinnlab.problems.HeatProblem`.

        Raises:
            ValueError: If the domain cannot be expressed as that problem's
                geometry, which spans ``[0, t_final] x [0, length]``. A wrapper
                on a shifted domain has no composable equivalent, so it is
                rejected rather than trained against a different region.
        """
        from ..problems import HeatProblem

        self._require_composable_domain()
        return HeatProblem(self.config, t_final=self.domain.tmax)

    def residual(self, t: Tensor, x: Tensor) -> Tensor:
        """Return ``u_t - alpha * u_xx`` at the given points.

        The physics is defined once, by
        :class:`~pinnlab.problems.HeatProblem`; this only adapts the calling
        convention, since the wrapper takes ``t`` and ``x`` as separate columns
        while the problem takes a single ``(N, 2)`` coordinate tensor.

        Gradients flow to the coordinates the problem differentiates, not to
        the ``t`` and ``x`` passed in, so use the returned values rather than
        backpropagating into the arguments.
        """
        return self._problem_residual(t, x)

    def initial_condition(self, x: np.ndarray) -> np.ndarray:
        """Return the sine-series initial temperature profile."""
        return heat_exact(
            np.zeros_like(x),
            x,
            self.config.alpha,
            self.config.length,
            self.config.modes,
        )

    def exact(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Return the analytic solution."""
        return heat_exact(
            t, x, self.config.alpha, self.config.length, self.config.modes
        )


def demo_heat(
    adam_steps: int = 3_000, lbfgs_max_iter: int = 200, seed: int = 0
) -> Tuple[HeatPINN, float]:
    """Train a heat-equation PINN and report its accuracy.

    Args:
        adam_steps: Adam iterations.
        lbfgs_max_iter: L-BFGS iterations after Adam.
        seed: Seed for reproducibility.

    Returns:
        The trained solver and its relative L2 error against the exact solution.
    """
    solver = HeatPINN(HeatConfig(alpha=0.1, modes=((1, 1.0),)))
    solver.train(
        TrainConfig(
            adam_steps=adam_steps,
            lbfgs_max_iter=lbfgs_max_iter,
            seed=seed,
            n_f=4_000,
        )
    )
    error = solver.relative_l2_error()
    logger.info("Heat demo finished", extra={"relative_l2_error": error})
    return solver, error

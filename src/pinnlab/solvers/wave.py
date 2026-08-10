"""1-D wave equation solved with a PINN, checked against its standing-wave solution.

The problem is

    u_tt = c^2 * u_xx,        x in [0, L],  t in [0, T]
    u(t, 0) = u(t, L) = 0
    u(0, x)   = sum_n a_n sin(n pi x / L)
    u_t(0, x) = 0

whose solution is a superposition of standing waves,

    u(t, x) = sum_n a_n cos(c n pi t / L) sin(n pi x / L).

This is a harder target than the heat equation and a more informative one. It is
second order in time, so the initial data constrains both ``u`` and ``u_t``;
nothing decays, so an error made early persists instead of being damped away;
and the solution is periodic in time, which is exactly where a PINN trained
without regard to temporal ordering tends to lock onto the wrong phase. It is
therefore the natural demonstration for the causal weighting in
:mod:`pinnlab.training.causal_weighting`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn

from ..utils.logging import get_logger
from ._base import Domain1D, ExactlySolvablePINN, TrainConfig, gradient

logger = get_logger(__name__)

__all__ = ["WaveConfig", "wave_exact", "WavePINN", "demo_wave"]


@dataclass(frozen=True)
class WaveConfig:
    """Physical setup for the 1-D wave equation.

    Attributes:
        c: Wave speed.
        length: Domain length ``L``; the domain is ``[0, L]``.
        modes: ``(n, amplitude)`` pairs defining the initial displacement.
    """

    c: float = 1.0
    length: float = 1.0
    modes: Tuple[Tuple[int, float], ...] = ((1, 1.0),)

    def validate(self) -> None:
        """Raise :class:`ValueError` if the configuration is unusable."""
        if self.c <= 0:
            raise ValueError(f"c must be positive, got {self.c}")
        if self.length <= 0:
            raise ValueError(f"length must be positive, got {self.length}")
        if not self.modes:
            raise ValueError("modes must contain at least one (n, amplitude) pair")
        for n, _ in self.modes:
            if n < 1:
                raise ValueError(f"mode numbers must be >= 1, got {n}")

    @property
    def period(self) -> float:
        """Return the temporal period of the slowest mode."""
        lowest = min(n for n, _ in self.modes)
        return 2.0 * self.length / (self.c * lowest)


def wave_exact(
    t: np.ndarray,
    x: np.ndarray,
    c: float,
    length: float = 1.0,
    modes: Sequence[Tuple[int, float]] = ((1, 1.0),),
) -> np.ndarray:
    """Evaluate the analytic solution of the 1-D wave equation.

    Args:
        t: Times, any shape.
        x: Positions, broadcastable against ``t``.
        c: Wave speed.
        length: Domain length.
        modes: ``(n, amplitude)`` pairs of the initial displacement.

    Returns:
        ``u(t, x)`` with the broadcast shape of the inputs.

    Raises:
        ValueError: If ``c`` or ``length`` is not positive.
    """
    if c <= 0:
        raise ValueError(f"c must be positive, got {c}")
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")

    t_arr, x_arr = np.broadcast_arrays(
        np.asarray(t, dtype=float), np.asarray(x, dtype=float)
    )
    out = np.zeros(t_arr.shape, dtype=float)
    for n, amplitude in modes:
        k = n * np.pi / length
        out = out + amplitude * np.cos(c * k * t_arr) * np.sin(k * x_arr)
    return out


class WavePINN(ExactlySolvablePINN):
    """PINN for the 1-D wave equation with homogeneous Dirichlet boundaries."""

    def __init__(
        self,
        config: Optional[WaveConfig] = None,
        model: Optional[nn.Module] = None,
        domain: Optional[Domain1D] = None,
        device: Optional[Union[str, torch.device]] = None,
        t_final: Optional[float] = None,
    ) -> None:
        """Initialise the solver.

        Args:
            config: Physical setup. Defaults to unit speed, fundamental mode.
            model: Network mapping ``(t, x) -> u``. Defaults to a 4x64 tanh MLP.
            domain: Space-time rectangle. Defaults to one full period.
            device: Torch device.
            t_final: End time. Defaults to one period of the slowest mode, so
                the solution returns to its initial shape at ``t_final``.
        """
        self.config = config or WaveConfig()
        self.config.validate()
        if domain is None:
            tmax = self.config.period if t_final is None else t_final
            domain = Domain1D(tmin=0.0, tmax=tmax, xmin=0.0, xmax=self.config.length)
        super().__init__(model=model, domain=domain, device=device)

    def residual(self, t: Tensor, x: Tensor) -> Tensor:
        """Return ``u_tt - c^2 * u_xx`` at the given points."""
        u = self.model(torch.cat([t, x], dim=1))
        u_t = gradient(u, t)
        u_tt = gradient(u_t, t)
        u_x = gradient(u, x)
        u_xx = gradient(u_x, x)
        return u_tt - self.config.c**2 * u_xx

    def initial_condition(self, x: np.ndarray) -> np.ndarray:
        """Return the initial displacement."""
        return wave_exact(
            np.zeros_like(x), x, self.config.c, self.config.length, self.config.modes
        )

    def initial_velocity(self, x: np.ndarray) -> np.ndarray:
        """Return the initial velocity, which is zero for a released standing wave."""
        return np.zeros_like(np.asarray(x, dtype=np.float32))

    def exact(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Return the analytic solution."""
        return wave_exact(t, x, self.config.c, self.config.length, self.config.modes)


def demo_wave(
    adam_steps: int = 4_000,
    lbfgs_max_iter: int = 300,
    seed: int = 0,
    causal: bool = False,
) -> Tuple[WavePINN, float]:
    """Train a wave-equation PINN and report its accuracy.

    Args:
        adam_steps: Adam iterations.
        lbfgs_max_iter: L-BFGS iterations after Adam.
        seed: Seed for reproducibility.
        causal: Whether to weight the residual causally in time.

    Returns:
        The trained solver and its relative L2 error against the exact solution.
    """
    from ..training.causal_weighting import CausalWeightConfig

    solver = WavePINN(WaveConfig(c=1.0, modes=((1, 1.0),)))
    solver.train(
        TrainConfig(
            adam_steps=adam_steps,
            lbfgs_max_iter=lbfgs_max_iter,
            seed=seed,
            n_f=6_000,
            causal=CausalWeightConfig(n_chunks=16, eps=1.0) if causal else None,
        )
    )
    error = solver.relative_l2_error()
    logger.info(
        "Wave demo finished", extra={"relative_l2_error": error, "causal": causal}
    )
    return solver, error

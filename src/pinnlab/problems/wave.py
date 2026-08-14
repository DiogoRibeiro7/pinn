"""Wave-equation problem definition for the generic PINN trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn

from ..constraints import (
    Constraint,
    DirichletBoundary,
    InitialCondition,
    NeumannBoundary,
)
from ..geometry import Rectangle
from ..residuals import DerivativeBackend
from ..solvers.wave import WaveConfig, wave_exact


@dataclass(frozen=True)
class WaveProblem:
    """One-dimensional wave equation ``u_tt - c^2 u_xx = 0``.

    Coordinates are ordered as ``(t, x)`` over ``[0, T] x [0, L]``.
    """

    config: WaveConfig = WaveConfig()
    t_final: float | None = None
    n_constraint_samples: int = 128

    def __post_init__(self) -> None:
        """Validate physical parameters and sample counts."""
        self.config.validate()
        if self.final_time <= 0:
            raise ValueError(f"t_final must be positive, got {self.final_time}")
        if self.n_constraint_samples < 1:
            raise ValueError("n_constraint_samples must be >= 1")

    @property
    def final_time(self) -> float:
        """Return the configured final time."""
        return self.config.period if self.t_final is None else self.t_final

    @property
    def input_dim(self) -> int:
        """Return the coordinate dimension, time plus one spatial coordinate."""
        return 2

    @property
    def reference_kind(self) -> str:
        """Return ``"analytic"``: the standing-wave solution is exact."""
        return "analytic"

    @property
    def output_dim(self) -> int:
        """Return scalar output dimension."""
        return 1

    @property
    def geometry(self) -> Rectangle:
        """Return the space-time rectangle."""
        return Rectangle(
            (0.0, 0.0),
            (self.final_time, self.config.length),
            ("t", "x"),
        )

    def initial_displacement(self, coordinates: Tensor) -> Tensor:
        """Evaluate ``u(0, x)`` on tensor coordinates."""
        x = coordinates[:, [1]]
        value = torch.zeros_like(x)
        for n, amplitude in self.config.modes:
            k = n * np.pi / self.config.length
            value = value + float(amplitude) * torch.sin(float(k) * x)
        return value

    def initial_velocity(self, coordinates: Tensor) -> Tensor:
        """Return the zero initial velocity ``u_t(0, x)``."""
        return torch.zeros(
            (coordinates.shape[0], 1),
            device=coordinates.device,
            dtype=coordinates.dtype,
        )

    def strong_residual(
        self,
        model: nn.Module,
        coordinates: Tensor,
        derivative_backend: DerivativeBackend,
    ) -> Tensor:
        """Return ``u_tt - c^2 u_xx`` at ``(t, x)`` coordinates."""
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("wave coordinates must have shape (N, 2)")
        coords = coordinates.requires_grad_(True)
        outputs = model(coords)
        u_tt = derivative_backend.partial_derivative(
            outputs, coords, output_index=0, coordinate_index=0, order=2
        )
        u_xx = derivative_backend.partial_derivative(
            outputs, coords, output_index=0, coordinate_index=1, order=2
        )
        return u_tt - self.config.c**2 * u_xx

    def constraints(self) -> Sequence[Constraint]:
        """Return displacement, velocity and Dirichlet boundary constraints."""
        return (
            InitialCondition(
                name="ic",
                time_dim=0,
                time_value=0.0,
                target=self.initial_displacement,
                default_sample_count=self.n_constraint_samples,
            ),
            NeumannBoundary(
                name="initial_velocity",
                boundary_dim=0,
                boundary_value=0.0,
                derivative_dim=0,
                target=self.initial_velocity,
                default_sample_count=self.n_constraint_samples,
            ),
            DirichletBoundary(
                name="bc_left",
                boundary_dim=1,
                boundary_value=0.0,
                target=0.0,
                default_sample_count=self.n_constraint_samples,
            ),
            DirichletBoundary(
                name="bc_right",
                boundary_dim=1,
                boundary_value=self.config.length,
                target=0.0,
                default_sample_count=self.n_constraint_samples,
            ),
        )

    def exact(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Evaluate the analytic solution."""
        return wave_exact(t, x, self.config.c, self.config.length, self.config.modes)

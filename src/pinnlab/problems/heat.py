"""Heat-equation problem definition for the generic PINN trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn

from ..constraints import Constraint, DirichletBoundary, InitialCondition
from ..geometry import Rectangle
from ..residuals import DerivativeBackend
from ..solvers.heat import HeatConfig, heat_exact


@dataclass(frozen=True)
class HeatProblem:
    """One-dimensional heat equation ``u_t - alpha u_xx = 0``.

    Coordinates are ordered as ``(t, x)`` over ``[0, T] x [0, L]``.
    """

    config: HeatConfig = HeatConfig()
    t_final: float = 1.0
    n_constraint_samples: int = 128

    def __post_init__(self) -> None:
        """Validate physical parameters and sample counts."""
        self.config.validate()
        if self.t_final <= 0:
            raise ValueError(f"t_final must be positive, got {self.t_final}")
        if self.n_constraint_samples < 1:
            raise ValueError("n_constraint_samples must be >= 1")

    @property
    def input_dim(self) -> int:
        """Return the coordinate dimension, time plus one spatial coordinate."""
        return 2

    @property
    def output_dim(self) -> int:
        """Return scalar output dimension."""
        return 1

    @property
    def geometry(self) -> Rectangle:
        """Return the space-time rectangle."""
        return Rectangle(
            (0.0, 0.0),
            (self.t_final, self.config.length),
            ("t", "x"),
        )

    def initial_condition(self, coordinates: Tensor) -> Tensor:
        """Evaluate ``u(0, x)`` on tensor coordinates."""
        x = coordinates[:, [1]]
        value = torch.zeros_like(x)
        for n, amplitude in self.config.modes:
            k = n * np.pi / self.config.length
            value = value + float(amplitude) * torch.sin(float(k) * x)
        return value

    def strong_residual(
        self,
        model: nn.Module,
        coordinates: Tensor,
        derivative_backend: DerivativeBackend,
    ) -> Tensor:
        """Return ``u_t - alpha u_xx`` at ``(t, x)`` coordinates."""
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("heat coordinates must have shape (N, 2)")
        coords = coordinates.requires_grad_(True)
        outputs = model(coords)
        u_t = derivative_backend.partial_derivative(
            outputs, coords, output_index=0, coordinate_index=0, order=1
        )
        u_xx = derivative_backend.partial_derivative(
            outputs, coords, output_index=0, coordinate_index=1, order=2
        )
        return u_t - self.config.alpha * u_xx

    def constraints(self) -> Sequence[Constraint]:
        """Return initial and homogeneous Dirichlet boundary constraints."""
        return (
            InitialCondition(
                name="ic",
                time_dim=0,
                time_value=0.0,
                target=self.initial_condition,
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
        return heat_exact(
            t, x, self.config.alpha, self.config.length, self.config.modes
        )

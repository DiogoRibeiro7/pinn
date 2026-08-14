"""Burgers-equation problem definition for the generic PINN trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn

from ..constraints import Constraint, DirichletBoundary, InitialCondition
from ..geometry import Rectangle
from ..residuals import DerivativeBackend
from ..solvers.raissi_improved import BurgersConfig
from ..solvers.reference import burgers_cole_hopf


@dataclass(frozen=True)
class BurgersProblem:
    """One-dimensional viscous Burgers equation on a space-time rectangle.

    The equation is ``u_t + u u_x - nu u_xx = 0`` with the PINN benchmark
    initial condition ``u(0, x) = -sin(pi x)`` and homogeneous Dirichlet
    boundaries at ``x = xmin`` and ``x = xmax``.
    """

    config: BurgersConfig = BurgersConfig()
    n_constraint_samples: int = 128
    reference_quad_points: int = 200

    def __post_init__(self) -> None:
        """Validate physical parameters and sample counts."""
        if self.config.nu <= 0:
            raise ValueError(f"nu must be positive, got {self.config.nu}")
        if self.n_constraint_samples < 1:
            raise ValueError("n_constraint_samples must be >= 1")
        if self.reference_quad_points < 2:
            raise ValueError("reference_quad_points must be >= 2")

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
        """Return the Burgers space-time rectangle."""
        return Rectangle(
            (self.config.tmin, self.config.xmin),
            (self.config.tmax, self.config.xmax),
            ("t", "x"),
        )

    def initial_condition(self, coordinates: Tensor) -> Tensor:
        """Evaluate ``u(0, x) = -sin(pi x)`` on tensor coordinates."""
        x = coordinates[:, [1]]
        return -torch.sin(torch.pi * x)

    def strong_residual(
        self,
        model: nn.Module,
        coordinates: Tensor,
        derivative_backend: DerivativeBackend,
    ) -> Tensor:
        """Return ``u_t + u u_x - nu u_xx`` at ``(t, x)`` coordinates."""
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("Burgers coordinates must have shape (N, 2)")
        coords = coordinates.requires_grad_(True)
        outputs = model(coords)
        u_t = derivative_backend.partial_derivative(
            outputs, coords, output_index=0, coordinate_index=0, order=1
        )
        u_x = derivative_backend.partial_derivative(
            outputs, coords, output_index=0, coordinate_index=1, order=1
        )
        u_xx = derivative_backend.partial_derivative(
            outputs, coords, output_index=0, coordinate_index=1, order=2
        )
        return u_t + outputs[:, [0]] * u_x - self.config.nu * u_xx

    def constraints(self) -> Sequence[Constraint]:
        """Return initial and homogeneous Dirichlet boundary constraints."""
        return (
            InitialCondition(
                name="ic",
                time_dim=0,
                time_value=self.config.tmin,
                target=self.initial_condition,
                default_sample_count=self.n_constraint_samples,
            ),
            DirichletBoundary(
                name="bc_left",
                boundary_dim=1,
                boundary_value=self.config.xmin,
                target=0.0,
                default_sample_count=self.n_constraint_samples,
            ),
            DirichletBoundary(
                name="bc_right",
                boundary_dim=1,
                boundary_value=self.config.xmax,
                target=0.0,
                default_sample_count=self.n_constraint_samples,
            ),
        )

    def exact(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Evaluate the Cole-Hopf reference solution."""
        return burgers_cole_hopf(
            t,
            x,
            self.config.nu,
            n_quad=self.reference_quad_points,
        )

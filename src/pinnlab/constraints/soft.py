"""Soft penalty constraints for PINN training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn

from ..geometry import Geometry
from ..geometry.base import validate_sample_count
from ..residuals import DerivativeBackend

from .base import ConstraintKind, ConstraintLoss

TargetFunction = Callable[[Tensor], Tensor]


def _as_target(value: Tensor | float | TargetFunction) -> TargetFunction:
    if callable(value):
        return value

    def constant_target(coordinates: Tensor) -> Tensor:
        if isinstance(value, Tensor):
            target = value.to(device=coordinates.device, dtype=coordinates.dtype)
            return torch.broadcast_to(target, (coordinates.shape[0], target.numel()))
        return torch.full(
            (coordinates.shape[0], 1),
            float(value),
            device=coordinates.device,
            dtype=coordinates.dtype,
        )

    return constant_target


def _validate_target(name: str, prediction: Tensor, target: Tensor) -> None:
    if target.shape != prediction.shape:
        raise ValueError(
            f"{name} target shape {tuple(target.shape)} does not match "
            f"prediction shape {tuple(prediction.shape)}"
        )


@dataclass(frozen=True)
class SoftPointConstraint:
    """Base class for soft constraints evaluated on sampled coordinates."""

    name: str
    fixed_dim: int
    fixed_value: float
    target: Tensor | float | TargetFunction
    kind: ConstraintKind
    default_sample_count: int = 128
    output_index: int | None = None

    def __post_init__(self) -> None:
        """Validate common constraint fields."""
        validate_sample_count(self.default_sample_count)

    def sample_coordinates(
        self,
        geometry: Geometry,
        *,
        n_samples: int,
        generator: torch.Generator | None,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> Tensor:
        """Sample points on the configured fixed-coordinate slice."""
        lower = float(geometry.lower_bounds[self.fixed_dim])
        upper = float(geometry.upper_bounds[self.fixed_dim])
        if abs(self.fixed_value - lower) <= 1e-12:
            side = "lower"
        elif abs(self.fixed_value - upper) <= 1e-12:
            side = "upper"
        else:
            points = geometry.sample_interior(
                n_samples, generator=generator, device=device, dtype=dtype
            )
            points[:, self.fixed_dim] = self.fixed_value
            return points
        return geometry.sample_boundary(
            n_samples,
            dim=self.fixed_dim,
            side=side,
            generator=generator,
            device=device,
            dtype=dtype,
        )

    def prediction(
        self,
        model: nn.Module,
        coordinates: Tensor,
        derivative_backend: DerivativeBackend,
    ) -> Tensor:
        """Return model values for this constraint."""
        del derivative_backend
        outputs = model(coordinates)
        if self.output_index is None:
            return outputs
        return outputs[:, [self.output_index]]

    def loss(
        self,
        model: nn.Module,
        geometry: Geometry,
        derivative_backend: DerivativeBackend,
        *,
        n_samples: int | None = None,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> ConstraintLoss:
        """Evaluate the soft MSE penalty for this constraint."""
        count = n_samples if n_samples is not None else self.default_sample_count
        validate_sample_count(count)
        coordinates = self.sample_coordinates(
            geometry,
            n_samples=count,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        prediction = self.prediction(model, coordinates, derivative_backend)
        target = _as_target(self.target)(coordinates)
        _validate_target(self.name, prediction, target)
        return ConstraintLoss(self.name, torch.mean((prediction - target) ** 2))


class InitialCondition(SoftPointConstraint):
    """Soft initial-condition constraint on a fixed time coordinate."""

    def __init__(
        self,
        *,
        name: str,
        time_dim: int,
        time_value: float,
        target: Tensor | float | TargetFunction,
        default_sample_count: int = 128,
        output_index: int | None = None,
    ) -> None:
        """Create an initial-condition loss."""
        super().__init__(
            name=name,
            fixed_dim=time_dim,
            fixed_value=time_value,
            target=target,
            kind=ConstraintKind.INITIAL,
            default_sample_count=default_sample_count,
            output_index=output_index,
        )


class DirichletBoundary(SoftPointConstraint):
    """Soft Dirichlet boundary condition on a fixed coordinate face."""

    def __init__(
        self,
        *,
        name: str,
        boundary_dim: int,
        boundary_value: float,
        target: Tensor | float | TargetFunction,
        default_sample_count: int = 128,
        output_index: int | None = None,
    ) -> None:
        """Create a Dirichlet boundary loss."""
        super().__init__(
            name=name,
            fixed_dim=boundary_dim,
            fixed_value=boundary_value,
            target=target,
            kind=ConstraintKind.DIRICHLET,
            default_sample_count=default_sample_count,
            output_index=output_index,
        )


class NeumannBoundary(SoftPointConstraint):
    """Soft Neumann condition for a selected coordinate derivative."""

    derivative_dim: int

    def __init__(
        self,
        *,
        name: str,
        boundary_dim: int,
        boundary_value: float,
        derivative_dim: int,
        target: Tensor | float | TargetFunction,
        default_sample_count: int = 128,
        output_index: int = 0,
    ) -> None:
        """Create a derivative boundary loss."""
        super().__init__(
            name=name,
            fixed_dim=boundary_dim,
            fixed_value=boundary_value,
            target=target,
            kind=ConstraintKind.NEUMANN,
            default_sample_count=default_sample_count,
            output_index=output_index,
        )
        object.__setattr__(self, "derivative_dim", derivative_dim)

    def prediction(
        self,
        model: nn.Module,
        coordinates: Tensor,
        derivative_backend: DerivativeBackend,
    ) -> Tensor:
        """Return the selected normal or coordinate derivative."""
        coords = coordinates.clone().detach().requires_grad_(True)
        outputs = model(coords)
        return derivative_backend.partial_derivative(
            outputs,
            coords,
            output_index=self.output_index or 0,
            coordinate_index=self.derivative_dim,
            order=1,
        )


@dataclass(frozen=True)
class PeriodicBoundary:
    """Soft periodic constraint matching opposite faces of one coordinate.

    With ``derivative_order=0`` the two faces are matched by value, which is
    what most periodic problems need. Value matching alone does not make a
    solution periodic, though: it admits a field with a kink at the seam. A
    second instance with ``derivative_order=1`` matches the normal derivative
    as well, which is what makes the pair equivalent to solving on a circle.
    Allen-Cahn is the case in point -- see
    :class:`~pinnlab.problems.AllenCahnProblem`.
    """

    name: str
    periodic_dim: int
    kind: ConstraintKind = ConstraintKind.PERIODIC
    default_sample_count: int = 128
    output_index: int | None = None
    derivative_order: int = 0

    def __post_init__(self) -> None:
        """Validate sample count and derivative order."""
        validate_sample_count(self.default_sample_count)
        if self.derivative_order not in (0, 1):
            raise ValueError(
                f"derivative_order must be 0 or 1, got {self.derivative_order}"
            )

    def _face_values(
        self,
        model: nn.Module,
        coordinates: Tensor,
        derivative_backend: DerivativeBackend,
    ) -> Tensor:
        """Return the model output, or its normal derivative, on one face."""
        if self.derivative_order == 0:
            outputs = model(coordinates)
            if self.output_index is not None:
                return outputs[:, [self.output_index]]
            return outputs
        coords = coordinates.clone().detach().requires_grad_(True)
        outputs = model(coords)
        return derivative_backend.partial_derivative(
            outputs,
            coords,
            output_index=self.output_index or 0,
            coordinate_index=self.periodic_dim,
            order=1,
        )

    def loss(
        self,
        model: nn.Module,
        geometry: Geometry,
        derivative_backend: DerivativeBackend,
        *,
        n_samples: int | None = None,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> ConstraintLoss:
        """Match values, or normal derivatives, across the periodic faces."""
        count = n_samples if n_samples is not None else self.default_sample_count
        validate_sample_count(count)
        lower = geometry.sample_boundary(
            count,
            dim=self.periodic_dim,
            side="lower",
            generator=generator,
            device=device,
            dtype=dtype,
        )
        upper = lower.clone()
        upper_bound = geometry.upper_bounds.to(device=lower.device, dtype=lower.dtype)[
            self.periodic_dim
        ]
        upper[:, self.periodic_dim] = upper_bound
        pred_lower = self._face_values(model, lower, derivative_backend)
        pred_upper = self._face_values(model, upper, derivative_backend)
        return ConstraintLoss(self.name, torch.mean((pred_lower - pred_upper) ** 2))

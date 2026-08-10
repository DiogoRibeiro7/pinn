"""Axis-aligned rectangular geometries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
from torch import Tensor

from .base import validate_boundary, validate_points, validate_sample_count


@dataclass(frozen=True)
class Rectangle:
    """Axis-aligned hyperrectangle with tensor-native sampling.

    Args:
        lower: Lower coordinate bounds.
        upper: Upper coordinate bounds.
        names: Optional coordinate names. Defaults to ``x0``, ``x1``, ...
    """

    lower: Tuple[float, ...]
    upper: Tuple[float, ...]
    names_: Tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Validate bounds and coordinate names."""
        if len(self.lower) == 0:
            raise ValueError("rectangle must have at least one dimension")
        if len(self.lower) != len(self.upper):
            raise ValueError("lower and upper bounds must have the same length")
        for low, high in zip(self.lower, self.upper):
            if high <= low:
                raise ValueError(f"upper bound {high} must exceed lower bound {low}")
        if self.names_ is not None and len(self.names_) != len(self.lower):
            raise ValueError("names must match the number of dimensions")

    @property
    def dimension(self) -> int:
        """Return the number of dimensions."""
        return len(self.lower)

    @property
    def lower_bounds(self) -> Tensor:
        """Return lower bounds as a float64 tensor."""
        return torch.tensor(self.lower, dtype=torch.float64)

    @property
    def upper_bounds(self) -> Tensor:
        """Return upper bounds as a float64 tensor."""
        return torch.tensor(self.upper, dtype=torch.float64)

    @property
    def names(self) -> Sequence[str]:
        """Return coordinate names."""
        if self.names_ is None:
            return tuple(f"x{i}" for i in range(self.dimension))
        return self.names_

    def _bounds(
        self, device: torch.device | str | None, dtype: torch.dtype | None
    ) -> tuple[Tensor, Tensor]:
        resolved_dtype = dtype or torch.get_default_dtype()
        lower = torch.tensor(self.lower, device=device, dtype=resolved_dtype)
        upper = torch.tensor(self.upper, device=device, dtype=resolved_dtype)
        return lower, upper

    def sample_interior(
        self,
        n: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Sample uniformly in the rectangle interior."""
        validate_sample_count(n)
        lower, upper = self._bounds(device, dtype)
        unit = torch.rand(
            (n, self.dimension),
            generator=generator,
            device=device,
            dtype=lower.dtype,
        )
        return lower + (upper - lower) * unit

    def sample_boundary(
        self,
        n: int,
        *,
        dim: int,
        side: str,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Sample uniformly on one boundary face."""
        validate_sample_count(n)
        validate_boundary(dim, side, self.dimension)
        points = self.sample_interior(
            n, generator=generator, device=device, dtype=dtype
        )
        lower, upper = self._bounds(device, dtype or points.dtype)
        points[:, dim] = lower[dim] if side == "lower" else upper[dim]
        return points

    def contains(self, points: Tensor, *, atol: float = 1e-7) -> Tensor:
        """Return a mask for points inside or on the rectangle."""
        validate_points(points, self.dimension)
        lower = torch.as_tensor(self.lower, device=points.device, dtype=points.dtype)
        upper = torch.as_tensor(self.upper, device=points.device, dtype=points.dtype)
        return torch.all((points >= lower - atol) & (points <= upper + atol), dim=1)

    def boundary_mask(
        self,
        points: Tensor,
        *,
        dim: int | None = None,
        side: str | None = None,
        atol: float = 1e-7,
    ) -> Tensor:
        """Return a mask for all boundaries or one selected face."""
        validate_points(points, self.dimension)
        lower = torch.as_tensor(self.lower, device=points.device, dtype=points.dtype)
        upper = torch.as_tensor(self.upper, device=points.device, dtype=points.dtype)
        if dim is not None:
            chosen_side = side or "lower"
            validate_boundary(dim, chosen_side, self.dimension)
            bound = lower[dim] if chosen_side == "lower" else upper[dim]
            return torch.isclose(points[:, dim], bound, atol=atol, rtol=0.0)
        return torch.any(
            torch.isclose(points, lower, atol=atol, rtol=0.0)
            | torch.isclose(points, upper, atol=atol, rtol=0.0),
            dim=1,
        )

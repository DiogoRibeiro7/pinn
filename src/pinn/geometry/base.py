"""Geometry contracts for PINN problem domains."""

from __future__ import annotations

from typing import Protocol, Sequence

import torch
from torch import Tensor


class Geometry(Protocol):
    """Protocol for domains that can sample and validate coordinates."""

    @property
    def dimension(self) -> int:
        """Return the number of coordinate dimensions."""

    @property
    def lower_bounds(self) -> Tensor:
        """Return lower coordinate bounds with shape ``(dimension,)``."""

    @property
    def upper_bounds(self) -> Tensor:
        """Return upper coordinate bounds with shape ``(dimension,)``."""

    @property
    def names(self) -> Sequence[str]:
        """Return coordinate names."""

    def sample_interior(
        self,
        n: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Sample coordinates inside the domain."""

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
        """Sample coordinates on a named boundary face."""

    def contains(self, points: Tensor, *, atol: float = 1e-7) -> Tensor:
        """Return a boolean mask identifying points inside the domain."""

    def boundary_mask(
        self,
        points: Tensor,
        *,
        dim: int | None = None,
        side: str | None = None,
        atol: float = 1e-7,
    ) -> Tensor:
        """Return a boolean mask identifying boundary points."""


def validate_sample_count(n: int) -> None:
    """Raise ``ValueError`` if a sample count is not positive."""
    if n < 1:
        raise ValueError(f"sample count must be >= 1, got {n}")


def validate_points(points: Tensor, dimension: int) -> None:
    """Validate that coordinates have shape ``(N, dimension)``."""
    if points.ndim != 2 or points.shape[1] != dimension:
        raise ValueError(
            f"points must have shape (N, {dimension}), got {tuple(points.shape)}"
        )


def validate_boundary(dim: int, side: str, dimension: int) -> None:
    """Validate a boundary face selector."""
    if dim < 0 or dim >= dimension:
        raise ValueError(f"dim must be in [0, {dimension}), got {dim}")
    if side not in {"lower", "upper"}:
        raise ValueError("side must be 'lower' or 'upper'")

"""Composable sampling contracts for trainer-owned collocation points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from torch import Tensor

from ..geometry import Geometry
from ..geometry.base import validate_sample_count


def latin_hypercube(n: int, d: int, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    """Generate Latin-hypercube samples in ``[low, high]^d``.

    This NumPy helper preserves the legacy solver-level function contract while
    keeping the implementation in the sampling package.
    """
    validate_sample_count(n)
    if d < 1:
        raise ValueError(f"dimension count must be >= 1, got {d}")

    cut = np.linspace(0.0, 1.0, n + 1)
    offsets = np.random.rand(n, d)
    lower_edges = cut[:n][:, np.newaxis]
    upper_edges = cut[1 : n + 1][:, np.newaxis]
    stratified = lower_edges + (upper_edges - lower_edges) * offsets

    unit_samples = np.zeros_like(stratified)
    for dim_index in range(d):
        order = np.random.permutation(n)
        unit_samples[:, dim_index] = stratified[order, dim_index]
    return low + (high - low) * unit_samples


class InteriorSampler(Protocol):
    """Protocol for sampling interior collocation coordinates."""

    def sample(
        self,
        geometry: Geometry,
        n: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Return ``n`` interior points for ``geometry``."""


@dataclass(frozen=True)
class UniformInteriorSampler:
    """Uniformly sample interior coordinates from the geometry."""

    def sample(
        self,
        geometry: Geometry,
        n: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Delegate to ``geometry.sample_interior``."""
        return geometry.sample_interior(
            n,
            generator=generator,
            device=device,
            dtype=dtype,
        )


@dataclass(frozen=True)
class LatinHypercubeInteriorSampler:
    """Stratified Latin-hypercube sampler for geometry interiors.

    Args:
        jitter: When ``True``, sample randomly within each stratum. When
            ``False``, place each sample at the midpoint of its stratum.
    """

    jitter: bool = True

    def sample(
        self,
        geometry: Geometry,
        n: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Return Latin-hypercube interior points scaled to geometry bounds."""
        validate_sample_count(n)
        resolved_dtype = dtype or torch.get_default_dtype()
        lower = geometry.lower_bounds.to(device=device, dtype=resolved_dtype)
        upper = geometry.upper_bounds.to(device=device, dtype=resolved_dtype)

        if self.jitter:
            offsets = torch.rand(
                (n, geometry.dimension),
                generator=generator,
                device=device,
                dtype=resolved_dtype,
            )
        else:
            offsets = torch.full(
                (n, geometry.dimension),
                0.5,
                device=device,
                dtype=resolved_dtype,
            )

        base = torch.arange(n, device=device, dtype=resolved_dtype).unsqueeze(1)
        stratified = (base + offsets) / float(n)
        columns = []
        for dim_index in range(geometry.dimension):
            permutation = torch.randperm(n, generator=generator, device=device)
            columns.append(stratified[permutation, dim_index])
        unit_points = torch.stack(columns, dim=1)

        return lower + (upper - lower) * unit_points

"""Composable sampling contracts for trainer-owned collocation points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from ..geometry import Geometry
from ..geometry.base import validate_sample_count


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

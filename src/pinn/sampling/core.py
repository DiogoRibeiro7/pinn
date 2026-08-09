"""Composable sampling contracts for trainer-owned collocation points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from ..geometry import Geometry


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

"""Compatibility adapters between legacy and composable samplers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from ..geometry import Geometry
from ..geometry.base import validate_points, validate_sample_count
from ..utils.sampling import BaseSampler, Domain
from .core import InteriorSampler, UniformInteriorSampler


def _domain_from_geometry(geometry: Geometry) -> Domain:
    """Convert geometry bounds to the legacy sampling domain structure."""
    lower = geometry.lower_bounds.detach().cpu().tolist()
    upper = geometry.upper_bounds.detach().cpu().tolist()
    bounds = [(float(low), float(high)) for low, high in zip(lower, upper)]
    return Domain(bounds=bounds, names=list(geometry.names))


@dataclass(frozen=True)
class BaseSamplerInteriorAdapter:
    """Expose a legacy ``BaseSampler`` through the ``InteriorSampler`` protocol.

    Args:
        base_sampler: Legacy sampler that returns NumPy arrays.
        validate_domain: When ``True``, reject sampled points outside the
            geometry supplied by the trainer.
    """

    base_sampler: BaseSampler
    validate_domain: bool = True

    def sample(
        self,
        geometry: Geometry,
        n: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Return legacy sampler points as a tensor compatible with geometry."""
        del generator
        validate_sample_count(n)
        resolved_dtype = dtype or torch.get_default_dtype()
        points = torch.as_tensor(
            self.base_sampler.sample(n),
            device=device,
            dtype=resolved_dtype,
        )
        validate_points(points, geometry.dimension)
        if self.validate_domain and not bool(torch.all(geometry.contains(points))):
            raise ValueError("base sampler returned points outside the geometry")
        return points


class InteriorSamplerBaseAdapter(BaseSampler):
    """Expose an ``InteriorSampler`` through the legacy ``BaseSampler`` API."""

    def __init__(
        self,
        geometry: Geometry,
        sampler: InteriorSampler | None = None,
        *,
        seed: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize the adapter with geometry-derived legacy domain bounds."""
        self.geometry = geometry
        self.interior_sampler = sampler or UniformInteriorSampler()
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.dtype = dtype
        self._generator: torch.Generator | None = None
        if seed is not None:
            self._generator = torch.Generator(device=self.device)
            self._generator.manual_seed(seed)
        super().__init__(_domain_from_geometry(geometry), seed=seed)

    def sample(self, n_points: int) -> np.ndarray:
        """Generate NumPy samples using the wrapped interior sampler."""
        validate_sample_count(n_points)
        with torch.no_grad():
            points = self.interior_sampler.sample(
                self.geometry,
                n_points,
                generator=self._generator,
                device=self.device,
                dtype=self.dtype,
            )
        return points.detach().cpu().numpy()

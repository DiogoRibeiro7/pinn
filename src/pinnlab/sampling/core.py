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


class FixedInteriorSampler:
    """Draw collocation points once and reuse them for the rest of the run.

    The generic :class:`~pinnlab.training.Trainer` calls its sampler on every
    optimizer step, so the default samplers give a fresh set of collocation
    points each time. That is not the only reasonable strategy, and it is not
    what the solvers in :mod:`pinnlab.solvers` do: they draw one set before
    training and reuse it throughout, which is also what the original
    Raissi-style PINN implementations do. Resampling explores the domain more
    thoroughly; a fixed set makes the loss a deterministic function of the
    parameters, which suits L-BFGS, whose line search assumes the objective
    does not move underneath it.

    This sampler makes that choice available on the composable path, so the
    legacy behaviour can be expressed rather than approximated::

        TrainerConfig(sampler=FixedInteriorSampler())

    Points are cached per requested count, so a caller asking for a different
    ``n`` gets its own fixed set rather than an error or a silent resize. The
    cache is per instance: share one sampler to share the points, or construct
    a new one to start over.

    Args:
        inner: Sampler used for the single draw. Defaults to uniform sampling.
    """

    def __init__(self, inner: InteriorSampler | None = None) -> None:
        """Wrap ``inner``, defaulting to :class:`UniformInteriorSampler`."""
        self.inner: InteriorSampler = inner or UniformInteriorSampler()
        self._cache: dict[int, Tensor] = {}

    def sample(
        self,
        geometry: Geometry,
        n: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Return the cached points for ``n``, drawing them on first use."""
        validate_sample_count(n)
        cached = self._cache.get(n)
        if cached is None:
            cached = self.inner.sample(
                geometry, n, generator=generator, device=device, dtype=dtype
            )
            self._cache[n] = cached
        # Move rather than redraw, so device or dtype changes do not silently
        # produce a different set of points.
        return cached.to(
            device=device if device is not None else cached.device,
            dtype=dtype if dtype is not None else cached.dtype,
        )

    def reset(self) -> None:
        """Discard the cached points so the next call draws a new set."""
        self._cache.clear()

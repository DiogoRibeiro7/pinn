"""Tests for composable sampling policies."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pinnkit.geometry import Rectangle
from pinnkit.sampling import (
    BaseSamplerInteriorAdapter,
    InteriorSamplerBaseAdapter,
    LatinHypercubeInteriorSampler,
    UniformInteriorSampler,
)
from pinnkit.utils.sampling import BaseSampler, Domain, LatinHypercubeSampler


class OutOfBoundsSampler(BaseSampler):
    """Legacy sampler that intentionally returns an invalid point."""

    def sample(self, n_points: int) -> np.ndarray:
        """Return points outside the unit interval."""
        return np.full((n_points, 1), 2.0, dtype=np.float32)


def test_uniform_interior_sampler_delegates_to_geometry() -> None:
    """Uniform sampler returns geometry-compatible points."""
    geometry = Rectangle((-1.0, 2.0), (1.0, 4.0), ("x", "t"))
    sampler = UniformInteriorSampler()

    points = sampler.sample(geometry, 12, dtype=torch.float64)

    assert points.shape == (12, 2)
    assert points.dtype == torch.float64
    assert torch.all(geometry.contains(points))


def test_latin_hypercube_sampler_covers_each_stratum_per_dimension() -> None:
    """Latin-hypercube sampling uses every interval in each dimension once."""
    geometry = Rectangle((0.0, 0.0), (1.0, 1.0), ("x", "t"))
    sampler = LatinHypercubeInteriorSampler(jitter=False)
    generator = torch.Generator().manual_seed(11)

    points = sampler.sample(geometry, 8, generator=generator, dtype=torch.float64)
    strata = torch.floor(points * 8).to(torch.int64)

    assert points.shape == (8, 2)
    assert torch.all(geometry.contains(points))
    for dim_index in range(geometry.dimension):
        assert torch.equal(torch.sort(strata[:, dim_index]).values, torch.arange(8))


def test_latin_hypercube_sampler_scales_to_geometry_bounds() -> None:
    """Latin-hypercube points are scaled from unit strata to physical bounds."""
    geometry = Rectangle((-2.0, 10.0), (2.0, 20.0), ("x", "t"))
    sampler = LatinHypercubeInteriorSampler()
    generator = torch.Generator().manual_seed(7)

    points = sampler.sample(geometry, 16, generator=generator, dtype=torch.float32)

    assert points.shape == (16, 2)
    assert points.dtype == torch.float32
    assert torch.all(geometry.contains(points))


def test_latin_hypercube_sampler_is_seed_deterministic() -> None:
    """Matching generators produce identical Latin-hypercube samples."""
    geometry = Rectangle((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    sampler = LatinHypercubeInteriorSampler()
    first_generator = torch.Generator().manual_seed(23)
    second_generator = torch.Generator().manual_seed(23)

    first = sampler.sample(geometry, 10, generator=first_generator)
    second = sampler.sample(geometry, 10, generator=second_generator)

    assert torch.allclose(first, second)


def test_latin_hypercube_sampler_rejects_empty_sample_count() -> None:
    """Latin-hypercube sampler reports invalid sample counts."""
    geometry = Rectangle((0.0,), (1.0,))
    sampler = LatinHypercubeInteriorSampler()

    with pytest.raises(ValueError, match="sample count"):
        sampler.sample(geometry, 0)


def test_base_sampler_adapter_exposes_legacy_sampler_to_trainer_api() -> None:
    """Legacy NumPy samplers can satisfy the interior sampler protocol."""
    geometry = Rectangle((0.0, -1.0), (1.0, 1.0), ("t", "x"))
    domain = Domain([(0.0, 1.0), (-1.0, 1.0)], ["t", "x"])
    legacy_sampler = LatinHypercubeSampler(domain, seed=5)
    sampler = BaseSamplerInteriorAdapter(legacy_sampler)

    points = sampler.sample(geometry, 10, dtype=torch.float64)

    assert points.shape == (10, 2)
    assert points.dtype == torch.float64
    assert torch.all(geometry.contains(points))


def test_base_sampler_adapter_rejects_points_outside_geometry() -> None:
    """Legacy sampler adapters validate returned points against geometry."""
    geometry = Rectangle((0.0,), (1.0,))
    legacy_sampler = OutOfBoundsSampler(Domain([(0.0, 1.0)]))
    sampler = BaseSamplerInteriorAdapter(legacy_sampler)

    with pytest.raises(ValueError, match="outside the geometry"):
        sampler.sample(geometry, 3)


def test_interior_sampler_adapter_exposes_new_sampler_to_legacy_api() -> None:
    """Composable interior samplers can feed legacy BaseSampler consumers."""
    geometry = Rectangle((0.0, 0.0), (1.0, 1.0), ("t", "x"))
    first = InteriorSamplerBaseAdapter(
        geometry,
        LatinHypercubeInteriorSampler(jitter=False),
        seed=13,
    )
    second = InteriorSamplerBaseAdapter(
        geometry,
        LatinHypercubeInteriorSampler(jitter=False),
        seed=13,
    )

    first_points = first.sample(8)
    second_points = second.sample(8)
    strata = (first_points * 8).astype(int)

    assert first.domain.bounds == [(0.0, 1.0), (0.0, 1.0)]
    assert first.domain.names == ["t", "x"]
    assert first_points.shape == (8, 2)
    assert first_points.dtype == np.float32
    assert (first_points >= 0.0).all()
    assert (first_points <= 1.0).all()
    assert (first_points == second_points).all()
    for dim_index in range(geometry.dimension):
        assert sorted(strata[:, dim_index].tolist()) == list(range(8))

"""Tests for composable sampling policies."""

from __future__ import annotations

import pytest
import torch

from pinn.geometry import Rectangle
from pinn.sampling import LatinHypercubeInteriorSampler, UniformInteriorSampler


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

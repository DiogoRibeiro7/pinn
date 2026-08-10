"""
Advanced sampling utilities for Physics-Informed Neural Networks (PINNs).

This module provides various sampling strategies for generating training points:
- Latin Hypercube Sampling (LHS) for efficient space-filling
- Sobol sequences for low-discrepancy sampling
- Adaptive sampling based on residual magnitudes
- Boundary and interface sampling utilities
- Time-stepping samplers for temporal domains

The sampling strategies are crucial for PINN performance as they determine
the distribution of collocation points used to enforce physics constraints.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, Callable, List, Dict
from abc import ABC, abstractmethod
from dataclasses import dataclass
import warnings

from .logging import get_logger, log_memory_usage, log_gpu_usage
from .profiling import profile_performance
from .validation import check_array, check_range, ValidationError

logger = get_logger(__name__)

try:
    from scipy.stats import qmc

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn(
        "SciPy not available. Some advanced sampling methods will be unavailable."
    )


# ============================================================
# Base Classes and Configuration
# ============================================================


@dataclass(frozen=True)
class Domain:
    """
    Define a computational domain for sampling.

    Attributes:
        bounds: List of (min, max) tuples for each dimension
        names: Optional names for each dimension (e.g., ['t', 'x', 'y'])
    """

    bounds: List[Tuple[float, float]]
    names: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.names is not None and len(self.names) != len(self.bounds):
            raise ValidationError("Number of names must match number of dimensions")
        for low, high in self.bounds:
            if high <= low:
                raise ValidationError("Upper bound must be greater than lower bound")

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.bounds)

    @property
    def volume(self) -> float:
        """Compute domain volume/area/length."""
        vol = 1.0
        for low, high in self.bounds:
            vol *= high - low
        return vol


class BaseSampler(ABC):
    """
    Abstract base class for all sampling strategies.

    All samplers should inherit from this class and implement the sample method.
    """

    def __init__(self, domain: Domain, seed: Optional[int] = None):
        """
        Initialize the sampler.

        Args:
            domain: Computational domain to sample from
            seed: Random seed for reproducibility
        """
        self.domain = domain
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    @abstractmethod
    def sample(self, n_points: int) -> np.ndarray:
        """
        Generate sample points in the domain.

        Args:
            n_points: Number of points to generate

        Returns:
            Array of shape (n_points, ndim) with sample coordinates
        """
        pass

    def _scale_to_domain(self, unit_samples: np.ndarray) -> np.ndarray:
        """
        Scale samples from unit hypercube [0,1]^d to domain bounds.

        Args:
            unit_samples: Samples in [0,1]^d, shape (n_points, ndim)

        Returns:
            Scaled samples within domain bounds
        """
        scaled = np.zeros_like(unit_samples)
        for i, (low, high) in enumerate(self.domain.bounds):
            scaled[:, i] = low + (high - low) * unit_samples[:, i]
        return scaled


# ============================================================
# Latin Hypercube Sampling
# ============================================================


class LatinHypercubeSampler(BaseSampler):
    """
    Latin Hypercube Sampling for efficient space-filling sampling.

    LHS ensures good coverage by dividing each dimension into n equal intervals
    and sampling exactly once from each interval, then applying random permutations.
    """

    def __init__(
        self, domain: Domain, seed: Optional[int] = None, criterion: str = "random"
    ):
        """
        Initialize LHS sampler.

        Args:
            domain: Computational domain
            seed: Random seed
            criterion: Sampling criterion ("random", "centered", "maximin", "correlation")
        """
        super().__init__(domain, seed)
        self.criterion = criterion

        if not SCIPY_AVAILABLE and criterion != "random":
            warnings.warn(
                f"SciPy not available. Using random LHS instead of {criterion}"
            )
            self.criterion = "random"

    @profile_performance
    def sample(self, n_points: int) -> np.ndarray:
        """
        Generate Latin Hypercube samples.

        Args:
            n_points: Number of points to generate

        Returns:
            LHS samples of shape (n_points, ndim)
        """
        check_range("n_points", n_points, min=1)

        logger.info(
            "Sampling points",
            extra={
                "method": "LHS",
                "n_points": n_points,
                "ndim": self.domain.ndim,
            },
        )

        if SCIPY_AVAILABLE and self.criterion != "random":
            # Use SciPy's advanced LHS implementation
            sampler = qmc.LatinHypercube(d=self.domain.ndim, seed=self.seed)
            unit_samples = sampler.random(n=n_points)

            # Apply optimization criteria
            if self.criterion == "maximin":
                # This would require optimization - simplified here
                pass
            elif self.criterion == "correlation":
                # This would require correlation optimization - simplified here
                pass

        else:
            # Manual LHS implementation
            unit_samples = self._manual_lhs(n_points)

        samples = self._scale_to_domain(unit_samples).astype(np.float32)
        log_memory_usage(logger)
        log_gpu_usage(logger)
        check_array("samples", samples, shape=(n_points, self.domain.ndim))
        return samples

    def _manual_lhs(self, n_points: int) -> np.ndarray:
        """Manual implementation of basic LHS."""
        ndim = self.domain.ndim

        # Create equally spaced intervals
        cut = np.linspace(0.0, 1.0, n_points + 1)
        u = np.random.rand(n_points, ndim)

        # Sample within each interval
        a = cut[:n_points]
        b = cut[1 : n_points + 1]
        rdpoints = a[:, np.newaxis] + (b[:, np.newaxis] - a[:, np.newaxis]) * u

        # Apply random permutation to each dimension
        H = np.zeros_like(rdpoints)
        for j in range(ndim):
            order = np.random.permutation(n_points)
            H[:, j] = rdpoints[order, j]

        return H


# ============================================================
# Sobol Low-Discrepancy Sampling
# ============================================================


class SobolSampler(BaseSampler):
    """
    Sobol sequence generator for low-discrepancy sampling.

    Sobol sequences provide better uniformity than random sampling,
    especially useful for high-dimensional problems and convergence studies.
    """

    def __init__(
        self, domain: Domain, seed: Optional[int] = None, scramble: bool = True
    ):
        """
        Initialize Sobol sampler.

        Args:
            domain: Computational domain
            seed: Random seed for scrambling
            scramble: Whether to scramble the sequence for better randomization
        """
        super().__init__(domain, seed)
        self.scramble = scramble

        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy is required for Sobol sampling")

    def sample(self, n_points: int) -> np.ndarray:
        """
        Generate Sobol sequence samples.

        Args:
            n_points: Number of points to generate

        Returns:
            Sobol samples of shape (n_points, ndim)
        """
        if n_points <= 0:
            raise ValueError("Number of points must be positive")

        # Create Sobol sampler
        sampler = qmc.Sobol(d=self.domain.ndim, scramble=self.scramble, seed=self.seed)

        # Generate samples (Sobol requires power-of-2 for optimal properties)
        unit_samples = sampler.random(n=n_points)

        return self._scale_to_domain(unit_samples).astype(np.float32)


# ============================================================
# Adaptive Sampling
# ============================================================


class AdaptiveSampler:
    """
    Adaptive sampling based on PDE residual magnitudes.

    This sampler refines the point distribution by adding more points
    in regions where the PDE residual is large, improving solution accuracy.
    """

    def __init__(
        self,
        domain: Domain,
        residual_fn: Callable[[np.ndarray], np.ndarray],
        base_sampler: BaseSampler,
        refinement_factor: float = 0.2,
        max_iterations: int = 5,
    ):
        """
        Initialize adaptive sampler.

        Args:
            domain: Computational domain
            residual_fn: Function computing PDE residuals at points
            base_sampler: Base sampling strategy
            refinement_factor: Fraction of points to refine each iteration
            max_iterations: Maximum refinement iterations
        """
        self.domain = domain
        self.residual_fn = residual_fn
        self.base_sampler = base_sampler
        self.refinement_factor = refinement_factor
        self.max_iterations = max_iterations

    def sample(self, n_points: int) -> np.ndarray:
        """
        Generate adaptively refined samples.

        Args:
            n_points: Target number of points

        Returns:
            Adaptively refined samples
        """
        # Start with base sampling
        points = self.base_sampler.sample(n_points // 2)

        for iteration in range(self.max_iterations):
            # Compute residuals
            residuals = self.residual_fn(points)
            residual_magnitudes = np.abs(residuals).flatten()

            # Identify high-residual regions
            n_refine = int(len(points) * self.refinement_factor)
            high_residual_indices = np.argsort(residual_magnitudes)[-n_refine:]

            if len(high_residual_indices) == 0:
                break

            # Generate new points around high-residual regions
            new_points = self._refine_around_points(
                points[high_residual_indices], n_refine
            )

            # Add new points
            points = np.vstack([points, new_points])

            # Stop if we have enough points
            if len(points) >= n_points:
                break

        # Return requested number of points
        if len(points) > n_points:
            indices = np.random.choice(len(points), n_points, replace=False)
            points = points[indices]

        return points.astype(np.float32)

    def _refine_around_points(
        self, center_points: np.ndarray, n_new: int
    ) -> np.ndarray:
        """Generate new points around high-residual regions."""
        new_points = []

        for center in center_points:
            # Create local sampling region around center point
            radius = 0.1  # Adaptive radius could be implemented

            # Generate random points in local neighborhood
            n_local = max(1, n_new // len(center_points))
            local_samples = np.random.uniform(-radius, radius, (n_local, len(center)))
            local_points = center + local_samples

            # Clip to domain bounds
            for i, (low, high) in enumerate(self.domain.bounds):
                local_points[:, i] = np.clip(local_points[:, i], low, high)

            new_points.append(local_points)

        return np.vstack(new_points) if new_points else np.empty((0, self.domain.ndim))


# ============================================================
# Boundary Sampling
# ============================================================


class BoundarySampler:
    """
    Specialized sampling for domain boundaries.

    Generates points on domain boundaries for enforcing boundary conditions.
    Supports different boundary types and sampling densities.
    """

    def __init__(self, domain: Domain, seed: Optional[int] = None):
        """
        Initialize boundary sampler.

        Args:
            domain: Computational domain
            seed: Random seed
        """
        self.domain = domain
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def sample_boundary(
        self, n_points_per_face: int, boundary_dim: int, boundary_value: float
    ) -> np.ndarray:
        """
        Sample points on a specific boundary face.

        Args:
            n_points_per_face: Number of points per boundary face
            boundary_dim: Dimension index for the boundary (0 for x, 1 for y, etc.)
            boundary_value: Value at the boundary (min or max of domain)

        Returns:
            Boundary points with specified coordinate fixed
        """
        if boundary_dim >= self.domain.ndim:
            raise ValueError("Boundary dimension exceeds domain dimensions")

        # Sample remaining dimensions
        other_dims = [i for i in range(self.domain.ndim) if i != boundary_dim]

        if not other_dims:
            # 1D case - boundary is just the point
            return np.array([[boundary_value]])

        # Create domain for other dimensions
        other_bounds = [self.domain.bounds[i] for i in other_dims]
        other_domain = Domain(other_bounds)

        # Sample in reduced domain
        sampler = LatinHypercubeSampler(other_domain, self.seed)
        other_coords = sampler.sample(n_points_per_face)

        # Insert boundary coordinate
        boundary_points = np.zeros((n_points_per_face, self.domain.ndim))
        boundary_points[:, boundary_dim] = boundary_value

        other_idx = 0
        for i in range(self.domain.ndim):
            if i != boundary_dim:
                boundary_points[:, i] = other_coords[:, other_idx]
                other_idx += 1

        return boundary_points.astype(np.float32)

    def sample_all_boundaries(
        self, n_points_per_face: int, exclude_corners: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Sample points on all domain boundaries.

        Args:
            n_points_per_face: Points per boundary face
            exclude_corners: Whether to avoid corner/edge overlaps

        Returns:
            Dictionary with boundary names as keys and point arrays as values
        """
        boundaries = {}

        for dim in range(self.domain.ndim):
            low, high = self.domain.bounds[dim]
            dim_name = self.domain.names[dim] if self.domain.names else f"dim_{dim}"

            # Sample min boundary
            boundaries[f"{dim_name}_min"] = self.sample_boundary(
                n_points_per_face, dim, low
            )

            # Sample max boundary
            boundaries[f"{dim_name}_max"] = self.sample_boundary(
                n_points_per_face, dim, high
            )

        return boundaries


# ============================================================
# Time-stepping Samplers
# ============================================================


class TimeSteppingSampler:
    """
    Temporal sampling for time-dependent PDEs.

    Provides various strategies for sampling in time, including uniform,
    adaptive, and progressive time-stepping approaches.
    """

    def __init__(
        self,
        spatial_domain: Domain,
        t_min: float,
        t_max: float,
        spatial_sampler: BaseSampler,
        seed: Optional[int] = None,
    ):
        """
        Initialize time-stepping sampler.

        Args:
            spatial_domain: Spatial domain (excluding time)
            t_min: Minimum time value
            t_max: Maximum time value
            spatial_sampler: Sampler for spatial coordinates
            seed: Random seed
        """
        self.spatial_domain = spatial_domain
        self.t_min = t_min
        self.t_max = t_max
        self.spatial_sampler = spatial_sampler
        self.seed = seed

    def uniform_time_sampling(
        self, n_time_steps: int, n_spatial_per_step: int
    ) -> np.ndarray:
        """
        Uniform sampling in time with spatial sampling at each time step.

        Args:
            n_time_steps: Number of time steps
            n_spatial_per_step: Spatial points per time step

        Returns:
            Spacetime samples of shape (n_time_steps * n_spatial_per_step, ndim+1)
        """
        # Generate time points
        t_points = np.linspace(self.t_min, self.t_max, n_time_steps)

        all_points = []
        for t in t_points:
            # Sample spatial points
            spatial_points = self.spatial_sampler.sample(n_spatial_per_step)

            # Add time coordinate
            t_column = np.full((n_spatial_per_step, 1), t)
            spacetime_points = np.hstack([t_column, spatial_points])

            all_points.append(spacetime_points)

        return np.vstack(all_points).astype(np.float32)

    def progressive_time_sampling(
        self, n_total_points: int, time_concentration: float = 2.0
    ) -> np.ndarray:
        """
        Progressive sampling with higher density near initial time.

        Args:
            n_total_points: Total number of spacetime points
            time_concentration: Parameter controlling time concentration (>1 for early bias)

        Returns:
            Progressively sampled spacetime points
        """
        # Generate spatial points
        spatial_points = self.spatial_sampler.sample(n_total_points)

        # Generate time points with power-law distribution
        u = np.random.random(n_total_points)
        t_normalized = u ** (1.0 / time_concentration)
        t_points = self.t_min + (self.t_max - self.t_min) * t_normalized

        # Combine time and space
        spacetime_points = np.hstack([t_points.reshape(-1, 1), spatial_points])

        return spacetime_points.astype(np.float32)


# ============================================================
# Composite Samplers
# ============================================================


class CompositeSampler:
    """
    Combine multiple sampling strategies for different regions/purposes.

    Useful for problems requiring different sampling densities in different
    regions or for combining initial conditions, boundaries, and interior points.
    """

    def __init__(self):
        """Initialize composite sampler."""
        self.samplers: List[Tuple[BaseSampler, int, str]] = []

    def add_sampler(self, sampler: BaseSampler, n_points: int, name: str = "") -> None:
        """
        Add a sampler to the composite.

        Args:
            sampler: Sampling strategy
            n_points: Number of points to generate with this sampler
            name: Optional name for this sampling component
        """
        self.samplers.append((sampler, n_points, name))

    def sample(self) -> Dict[str, np.ndarray]:
        """
        Generate samples from all component samplers.

        Returns:
            Dictionary with sampler names as keys and point arrays as values
        """
        results = {}

        for i, (sampler, n_points, name) in enumerate(self.samplers):
            sampler_name = name if name else f"sampler_{i}"
            results[sampler_name] = sampler.sample(n_points)

        return results

    def sample_combined(self) -> np.ndarray:
        """
        Generate and combine all samples into a single array.

        Returns:
            Combined samples from all component samplers
        """
        all_samples = []

        for sampler, n_points, _ in self.samplers:
            samples = sampler.sample(n_points)
            all_samples.append(samples)

        return np.vstack(all_samples) if all_samples else np.empty((0, 0))


# ============================================================
# Convenience Functions
# ============================================================


def create_lhs_sampler(
    bounds: List[Tuple[float, float]], seed: Optional[int] = None
) -> LatinHypercubeSampler:
    """
    Convenience function to create LHS sampler.

    Args:
        bounds: List of (min, max) tuples for each dimension
        seed: Random seed

    Returns:
        Configured LHS sampler
    """
    domain = Domain(bounds)
    return LatinHypercubeSampler(domain, seed)


def create_boundary_points(
    bounds: List[Tuple[float, float]],
    n_points_per_face: int,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Convenience function to generate boundary points.

    Args:
        bounds: Domain bounds
        n_points_per_face: Points per boundary face
        seed: Random seed

    Returns:
        Dictionary of boundary points
    """
    check_range("n_points_per_face", n_points_per_face, min=1)
    domain = Domain(bounds)
    sampler = BoundarySampler(domain, seed)
    return sampler.sample_all_boundaries(n_points_per_face)


def create_spacetime_points(
    spatial_bounds: List[Tuple[float, float]],
    time_bounds: Tuple[float, float],
    n_time_steps: int,
    n_spatial_per_step: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Convenience function for spacetime sampling.

    Args:
        spatial_bounds: Spatial domain bounds
        time_bounds: (t_min, t_max) time bounds
        n_time_steps: Number of time steps
        n_spatial_per_step: Spatial points per time step
        seed: Random seed

    Returns:
        Spacetime sample points
    """
    check_range("n_time_steps", n_time_steps, min=1)
    check_range("n_spatial_per_step", n_spatial_per_step, min=1)
    spatial_domain = Domain(spatial_bounds)
    spatial_sampler = LatinHypercubeSampler(spatial_domain, seed)

    time_sampler = TimeSteppingSampler(
        spatial_domain, time_bounds[0], time_bounds[1], spatial_sampler, seed
    )

    return time_sampler.uniform_time_sampling(n_time_steps, n_spatial_per_step)

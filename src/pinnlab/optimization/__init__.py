"""Optimization utilities including caching and precomputation helpers."""

from .caching import (
    AdaptiveCache,
    CacheConfig,
    CacheEntry,
    CachedComputation,
    CachedSampler,
    PrecomputationManager,
    TrainingCache,
    cached_computation,
)

__all__ = [
    "AdaptiveCache",
    "CacheConfig",
    "CacheEntry",
    "CachedComputation",
    "CachedSampler",
    "PrecomputationManager",
    "TrainingCache",
    "cached_computation",
]

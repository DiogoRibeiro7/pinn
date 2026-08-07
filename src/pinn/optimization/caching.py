"""Caching and precomputation utilities for accelerating PINN workloads.

This module provides adaptive caching structures and lightweight
precomputation helpers that can be integrated into training loops to reuse
expensive computations.  The caches are aware of memory budgets, support
optional compression, and can persist entries across training sessions.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import pickle
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, Optional, Tuple, Union
from collections import defaultdict, deque

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Cache entry representation
# ---------------------------------------------------------------------------


@dataclass
class CacheEntry:
    """Container for cached values.

    Attributes
    ----------
    payload:
        Stored value (possibly compressed).
    size:
        Estimated memory footprint of the stored value in bytes.
    priority:
        Priority hint used during eviction (higher values are retained longer).
    timestamp:
        Last access timestamp for LRU-style eviction.
    hits:
        Number of cache hits recorded for this entry.
    metadata:
        Arbitrary metadata such as model version or context.
    compressed:
        Flag indicating whether ``payload`` is a compressed blob.
    """

    payload: Any
    size: int
    priority: int = 0
    timestamp: float = field(default_factory=time.time)
    hits: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    compressed: bool = False


# ---------------------------------------------------------------------------
# Adaptive in-memory cache
# ---------------------------------------------------------------------------


class AdaptiveCache:
    """Memory-aware cache with LRU eviction and optional compression.

    Parameters
    ----------
    max_memory_mb:
        Soft upper bound on memory usage in megabytes.  When exceeded the
        cache evicts low-priority entries until the budget is respected.
    max_items:
        Maximum number of items to keep in the cache.  ``None`` disables the
        limit.
    compression:
        When ``True`` cache entries are compressed using ``gzip`` and pickled
        with the highest protocol to reduce memory consumption.
    persistent_path:
        Optional path used to persist cache entries with :meth:`save` and
        :meth:`load`.
    namespace:
        Logical name used when reporting statistics.
    """

    def __init__(
        self,
        *,
        max_memory_mb: Optional[int] = 512,
        max_items: Optional[int] = None,
        compression: bool = True,
        persistent_path: Optional[Union[str, Path]] = None,
        namespace: str = "default",
    ) -> None:
        self.max_memory = None
        if max_memory_mb is not None:
            self.max_memory = int(max_memory_mb * 1024 * 1024)
        self.max_items = max_items
        self.compression = compression
        self.namespace = namespace
        self._entries: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._current_memory = 0
        self._persistent_path = Path(persistent_path) if persistent_path else None
        if self._persistent_path is not None:
            self._persistent_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _estimate_size(self, obj: Any) -> int:
        if obj is None:
            return 0
        if torch.is_tensor(obj):
            return int(obj.element_size() * obj.nelement())
        if isinstance(obj, np.ndarray):
            return int(obj.nbytes)
        if isinstance(obj, (bytes, bytearray, memoryview)):
            return len(obj)
        if isinstance(obj, str):
            return len(obj.encode("utf-8"))
        if isinstance(obj, dict):
            return sum(
                self._estimate_size(k) + self._estimate_size(v) for k, v in obj.items()
            )
        if isinstance(obj, (list, tuple, set)):
            return sum(self._estimate_size(v) for v in obj)
        return sys.getsizeof(obj)

    def _serialise(self, value: Any) -> Tuple[Any, bool, int]:
        if not self.compression:
            return value, False, self._estimate_size(value)
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        compressed = gzip.compress(payload)
        return compressed, True, len(compressed)

    def _deserialise(self, payload: Any, compressed: bool) -> Any:
        if not compressed:
            return payload
        data = gzip.decompress(payload)
        # Trusted input: this cache only ever deserialises payloads it
        # serialised itself in _serialise. Never point it at foreign data.
        return pickle.loads(data)  # nosec B301

    def _evict_one(self) -> None:
        if not self._entries:
            return
        # Evict the lowest-priority, least-recently-used entry
        key, entry = min(
            self._entries.items(),
            key=lambda item: (item[1].priority, item[1].timestamp, item[1].hits),
        )
        self._current_memory = max(0, self._current_memory - entry.size)
        del self._entries[key]

    def _evict_until_within_budget(self, additional: int = 0) -> None:
        while True:
            over_memory = self.max_memory is not None and (
                self._current_memory + additional > self.max_memory
            )
            over_items = self.max_items is not None and (
                len(self._entries) + (1 if additional else 0) > self.max_items
            )
            if not over_memory and not over_items:
                break
            self._evict_one()

    def _update_memory(self) -> None:
        self._current_memory = sum(entry.size for entry in self._entries.values())

    def __contains__(self, key: str) -> bool:  # pragma: no cover - trivial wrapper
        with self._lock:
            return key in self._entries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        value_meta = self.get_with_metadata(key)
        if value_meta is None:
            return default
        value, _ = value_meta
        return value

    def get_with_metadata(self, key: str) -> Optional[Tuple[Any, Dict[str, Any]]]:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            entry.hits += 1
            entry.timestamp = time.time()
            value = self._deserialise(entry.payload, entry.compressed)
            return value, dict(entry.metadata)

    def put(
        self,
        key: str,
        value: Any,
        *,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        serialised, compressed, size = self._serialise(value)
        with self._lock:
            if key in self._entries:
                old = self._entries.pop(key)
                self._current_memory = max(0, self._current_memory - old.size)
            self._evict_until_within_budget(additional=size)
            self._entries[key] = CacheEntry(
                payload=serialised,
                size=size,
                priority=priority,
                metadata=dict(metadata or {}),
                compressed=compressed,
            )
            self._current_memory += size

    def invalidate(self, key: str) -> None:
        with self._lock:
            if key in self._entries:
                entry = self._entries.pop(key)
                self._current_memory = max(0, self._current_memory - entry.size)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._current_memory = 0

    def get_or_compute(
        self,
        key: str,
        factory: Callable[[], Any],
        *,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        refresh: bool = False,
    ) -> Any:
        if not refresh:
            cached = self.get(key)
            if cached is not None:
                return cached
        value = factory()
        self.put(key, value, priority=priority, metadata=metadata)
        return value

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "namespace": self.namespace,
                "items": len(self._entries),
                "memory_bytes": self._current_memory,
                "max_memory_bytes": self.max_memory,
                "compression": self.compression,
            }

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        target = Path(path) if path is not None else self._persistent_path
        if target is None:
            return
        with self._lock:
            payload = {
                "entries": self._entries,
                "memory": self._current_memory,
                "compression": self.compression,
            }
        with open(target, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: Optional[Union[str, Path]] = None) -> None:
        source = Path(path) if path is not None else self._persistent_path
        if source is None or not source.exists():
            return
        with open(source, "rb") as fh:
            # Trusted input: this file is written by this class's own save().
            payload = pickle.load(fh)  # nosec B301
        with self._lock:
            self._entries = payload.get("entries", {})
            self._current_memory = payload.get("memory", 0)


# ---------------------------------------------------------------------------
# Cache configuration helpers
# ---------------------------------------------------------------------------


@dataclass
class CacheConfig:
    """Configuration options for :class:`TrainingCache`.

    Attributes
    ----------
    enabled:
        Toggle caching behaviour.  When ``False`` the :class:`TrainingCache`
        becomes a no-op wrapper.
    identifier:
        Logical identifier for cache persistence.  When combined with
        ``persistent_path`` this allows training runs to reuse previously saved
        sampling points.
    max_memory_mb:
        Per-cache memory budget applied to each internal :class:`AdaptiveCache`.
    persistent_path:
        Directory used to persist cached artefacts.  Files are stored within a
        ``cache`` sub-directory named after :attr:`identifier`.
    compression:
        Enable value compression before storing entries.
    store_samples, store_gradients, store_losses, store_activations:
        Toggle specialised caches for sampling points, autograd outputs,
        diagnostic loss values, and forward activations respectively.
    refresh_interval:
        Optional number of invocations after which cached sampling points are
        refreshed (re-generated).  ``0`` disables periodic refreshes.
    async_precompute:
        Enable background precomputation utilities for collocation ordering and
        other expensive transformations.
    precompute_workers:
        Number of worker threads used when :attr:`async_precompute` is enabled.
    max_permutation_queue:
        Maximum number of precomputed permutations buffered for reuse.
    persist_samples:
        When ``True`` cached sampling points are written to disk for reuse in
        future runs.
    lazy_evaluation:
        Enable memoisation helpers to lazily compute expensive functions.
    """

    enabled: bool = False
    identifier: str = "default"
    max_memory_mb: int = 512
    persistent_path: Optional[Union[str, Path]] = None
    compression: bool = True
    store_samples: bool = True
    store_gradients: bool = False
    store_losses: bool = False
    store_activations: bool = False
    refresh_interval: int = 0
    async_precompute: bool = False
    precompute_workers: int = 2
    max_permutation_queue: int = 2
    persist_samples: bool = True
    lazy_evaluation: bool = True

    def make_directory(self) -> Optional[Path]:
        if self.persistent_path is None:
            return None
        base = Path(self.persistent_path)
        base.mkdir(parents=True, exist_ok=True)
        cache_dir = base / self.identifier
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def validate(self) -> None:
        if self.max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive")
        if self.precompute_workers <= 0:
            raise ValueError("precompute_workers must be positive")
        if self.max_permutation_queue <= 0:
            raise ValueError("max_permutation_queue must be positive")
        if self.refresh_interval < 0:
            raise ValueError("refresh_interval must be non-negative")


# ---------------------------------------------------------------------------
# Precomputation manager
# ---------------------------------------------------------------------------


class PrecomputationManager:
    """Simple thread-pool wrapper for asynchronous precomputation."""

    def __init__(self, max_workers: int = 2) -> None:
        from concurrent.futures import ThreadPoolExecutor

        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def submit(
        self, key: str, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> None:
        from concurrent.futures import Future

        with self._lock:
            future: Future = self._executor.submit(fn, *args, **kwargs)
            self._futures[key] = future

    def result(self, key: str, default: Any = None) -> Any:
        with self._lock:
            future = self._futures.get(key)
        if future is None:
            return default
        if future.done():
            return future.result()
        return default

    def pop_ready(self, prefix: Optional[str] = None) -> Iterable[Tuple[str, Any]]:
        ready: Dict[str, Any] = {}
        with self._lock:
            for key, future in list(self._futures.items()):
                if prefix is not None and not key.startswith(prefix):
                    continue
                if future.done():
                    ready[key] = future.result()
                    del self._futures[key]
        return ready.items()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Training cache orchestrator
# ---------------------------------------------------------------------------


class TrainingCache:
    """Manage multiple caches for a training run."""

    def __init__(self, config: CacheConfig, *, rng_seed: Optional[int] = None) -> None:
        self.config = config
        self.config.validate()
        self._rng = np.random.default_rng(rng_seed)
        self._sample_cache = (
            AdaptiveCache(
                max_memory_mb=config.max_memory_mb,
                compression=config.compression,
                namespace="samples",
            )
            if config.store_samples
            else None
        )
        self._gradient_cache = (
            AdaptiveCache(
                max_memory_mb=max(16, config.max_memory_mb // 4),
                compression=config.compression,
                namespace="gradients",
            )
            if config.store_gradients
            else None
        )
        self._loss_cache = (
            AdaptiveCache(
                max_memory_mb=64,
                compression=False,
                namespace="losses",
            )
            if config.store_losses
            else None
        )
        self._activation_cache = (
            AdaptiveCache(
                max_memory_mb=max(64, config.max_memory_mb // 2),
                compression=config.compression,
                namespace="activations",
            )
            if config.store_activations
            else None
        )
        self._precompute = (
            PrecomputationManager(config.precompute_workers)
            if config.async_precompute
            else None
        )
        self._permutation_queue: Dict[int, Deque[np.ndarray]] = defaultdict(deque)
        self._permutation_pending: Dict[str, int] = {}
        self._use_count = 0
        self.model_version = 0
        self._cache_dir = config.make_directory()

    # ------------------------------------------------------------------
    # Sampling utilities
    # ------------------------------------------------------------------

    def _training_key(self, context: Dict[str, Any]) -> str:
        context_serialisable = json.dumps(context, sort_keys=True)
        digest = hashlib.blake2b(context_serialisable.encode("utf-8"), digest_size=16)
        return digest.hexdigest()

    def _sample_file(self, key: str) -> Optional[Path]:
        if self._cache_dir is None or not self.config.persist_samples:
            return None
        return self._cache_dir / f"{key}_samples.pt"

    def prepare_training_points(
        self,
        tcfg: Any,
        factory: Callable[[], Dict[str, torch.Tensor]],
        *,
        device: torch.device,
        context: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        if not self.config.enabled:
            return factory()

        key_context = {
            "config": {
                "n_u0": tcfg.n_u0,
                "n_bc": tcfg.n_bc,
                "n_f": tcfg.n_f,
                "seed": tcfg.seed,
            },
            "problem": context,
        }
        key = self._training_key(key_context)
        refresh = (
            self.config.refresh_interval > 0
            and self._use_count >= self.config.refresh_interval
        )
        cached: Optional[Dict[str, torch.Tensor]] = None
        if not refresh and self._sample_cache is not None:
            cached = self._sample_cache.get(key)
        if cached is None and not refresh:
            file_path = self._sample_file(key)
            if file_path is not None and file_path.exists():
                cached = torch.load(file_path, map_location="cpu")
                if self._sample_cache is not None:
                    self._sample_cache.put(key, cached, priority=5)
        if cached is not None and not refresh:
            self._use_count += 1
            return {name: tensor.to(device) for name, tensor in cached.items()}

        data = factory()
        cpu_data = {name: tensor.detach().cpu() for name, tensor in data.items()}
        if self._sample_cache is not None:
            self._sample_cache.put(key, cpu_data, priority=5)
        file_path = self._sample_file(key)
        if file_path is not None:
            torch.save(cpu_data, file_path)
        self._use_count = 1
        return {name: tensor.to(device) for name, tensor in cpu_data.items()}

    # ------------------------------------------------------------------
    # Activation caching
    # ------------------------------------------------------------------

    def _tensor_digest(self, tensor: torch.Tensor) -> str:
        cpu = tensor.detach().to("cpu")
        buffer = cpu.reshape(-1).to(torch.float64)
        array = buffer.numpy()
        digest = hashlib.blake2b(array.tobytes(), digest_size=16)
        return digest.hexdigest()

    def get_activation(self, inputs: torch.Tensor) -> Optional[torch.Tensor]:
        if self._activation_cache is None:
            return None
        key = self._tensor_digest(inputs)
        cached = self._activation_cache.get_with_metadata(key)
        if cached is None:
            return None
        value, metadata = cached
        if metadata.get("version") != self.model_version:
            return None
        return value.to(inputs.device)

    def store_activation(self, inputs: torch.Tensor, outputs: torch.Tensor) -> None:
        if self._activation_cache is None:
            return
        key = self._tensor_digest(inputs)
        payload = outputs.detach().cpu()
        metadata = {"version": self.model_version}
        self._activation_cache.put(key, payload, priority=2, metadata=metadata)

    # ------------------------------------------------------------------
    # Gradient caching
    # ------------------------------------------------------------------

    def coordinates_key(self, *tensors: torch.Tensor) -> str:
        digest = hashlib.blake2b(digest_size=16)
        for tensor in tensors:
            digest.update(self._tensor_digest(tensor).encode("utf-8"))
        digest.update(str(self.model_version).encode("utf-8"))
        return digest.hexdigest()

    def get_gradient(self, key: str) -> Optional[torch.Tensor]:
        if self._gradient_cache is None:
            return None
        cached = self._gradient_cache.get_with_metadata(key)
        if cached is None:
            return None
        value, metadata = cached
        if metadata.get("version") != self.model_version:
            return None
        return value

    def store_gradient(self, key: str, value: torch.Tensor) -> None:
        if self._gradient_cache is None:
            return
        metadata = {"version": self.model_version}
        self._gradient_cache.put(
            key, value.detach().cpu(), priority=1, metadata=metadata
        )

    # ------------------------------------------------------------------
    # Loss history caching
    # ------------------------------------------------------------------

    def record_loss(self, step: int, losses: Dict[str, float]) -> None:
        if self._loss_cache is None:
            return
        key = f"loss:{step}"
        metadata = {"step": step}
        self._loss_cache.put(key, dict(losses), priority=0, metadata=metadata)

    def loss_history(self) -> Dict[int, Dict[str, float]]:
        if self._loss_cache is None:
            return {}
        history: Dict[int, Dict[str, float]] = {}
        for key in list(
            self._loss_cache._entries.keys()
        ):  # pragma: no cover - debug path
            value_meta = self._loss_cache.get_with_metadata(key)
            if value_meta is None:
                continue
            value, metadata = value_meta
            history[int(metadata.get("step", 0))] = value
        return dict(sorted(history.items()))

    # ------------------------------------------------------------------
    # Collocation permutations via precomputation
    # ------------------------------------------------------------------

    def register_collocation_set(self, total: int) -> None:
        if self._precompute is None:
            return
        self._schedule_permutation(total)

    def _schedule_permutation(self, total: int) -> None:
        if self._precompute is None:
            return
        queue = self._permutation_queue[total]
        if len(queue) >= self.config.max_permutation_queue:
            return
        token = self._rng.integers(0, 2**32 - 1)
        key = f"perm:{total}:{token}"
        self._permutation_pending[key] = total

        def _compute(seed: int, n: int) -> np.ndarray:
            rng = np.random.default_rng(seed)
            return rng.permutation(n)

        self._precompute.submit(key, _compute, int(token), total)

    def next_permutation(self, total: int, device: torch.device) -> torch.Tensor:
        if self._precompute is None:
            return torch.randperm(total, device=device)
        # Harvest ready futures
        for key, value in self._precompute.pop_ready(prefix="perm:"):
            count = self._permutation_pending.pop(key, None)
            if count is None:
                continue
            self._permutation_queue[count].append(value)
        queue = self._permutation_queue[total]
        if queue:
            perm = queue.popleft()
        else:
            perm = np.random.permutation(total)
        self._schedule_permutation(total)
        tensor = torch.as_tensor(perm, dtype=torch.long)
        return tensor.to(device)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def advance_model_version(self) -> None:
        self.model_version += 1
        if self._activation_cache is not None:
            self._activation_cache.clear()
        if self._gradient_cache is not None:
            self._gradient_cache.clear()

    def close(self) -> None:
        if self._precompute is not None:
            self._precompute.shutdown()


# ---------------------------------------------------------------------------
# Cached sampler wrapper
# ---------------------------------------------------------------------------


class CachedSampler:
    """Wrap a sampler to reuse generated points across invocations."""

    def __init__(
        self,
        base_sampler: Any,
        *,
        cache: Optional[AdaptiveCache] = None,
        cache_size_mb: int = 128,
    ) -> None:
        self.base_sampler = base_sampler
        self.cache = cache or AdaptiveCache(
            max_memory_mb=cache_size_mb,
            namespace="cached-sampler",
        )
        self.hit_count = 0
        self.miss_count = 0

    def _key(self, n_points: int, **kwargs: Any) -> str:
        payload = {"n_points": n_points, **kwargs}
        digest = hashlib.blake2b(
            json.dumps(payload, sort_keys=True).encode("utf-8"), digest_size=16
        )
        return digest.hexdigest()

    def sample(self, n_points: int, **kwargs: Any) -> np.ndarray:
        key = self._key(n_points, **kwargs)
        cached = self.cache.get(key)
        if cached is not None:
            self.hit_count += 1
            return np.array(cached)
        samples = self.base_sampler.sample(n_points, **kwargs)
        self.cache.put(key, np.asarray(samples), priority=1)
        self.miss_count += 1
        return samples

    def stats(self) -> Dict[str, Any]:
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total else 0.0
        return {
            "hit_rate": hit_rate,
            "hits": self.hit_count,
            "misses": self.miss_count,
            **self.cache.stats(),
        }


# ---------------------------------------------------------------------------
# Memoisation decorator
# ---------------------------------------------------------------------------


class CachedComputation:
    """Callable decorator that memoises results inside an :class:`AdaptiveCache`."""

    def __init__(self, cache: AdaptiveCache) -> None:
        self.cache = cache

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key_payload = {
                "fn": getattr(func, "__qualname__", func.__name__),
                "args": _hashable_args(args),
                "kwargs": _hashable_kwargs(kwargs),
            }
            digest = hashlib.blake2b(
                json.dumps(key_payload, sort_keys=True).encode("utf-8"), digest_size=16
            ).hexdigest()
            return self.cache.get_or_compute(digest, lambda: func(*args, **kwargs))

        return wrapper


def cached_computation(cache: AdaptiveCache) -> CachedComputation:
    """Convenience wrapper returning :class:`CachedComputation`."""

    return CachedComputation(cache)


# ---------------------------------------------------------------------------
# Helper utilities for hashing arguments
# ---------------------------------------------------------------------------


def _hashable_args(args: Tuple[Any, ...]) -> Any:
    result = []
    for value in args:
        if torch.is_tensor(value):
            result.append(
                {
                    "tensor": hashlib.blake2b(
                        value.detach().cpu().numpy().tobytes(), digest_size=16
                    ).hexdigest()
                }
            )
        elif isinstance(value, np.ndarray):
            result.append(
                {
                    "ndarray": hashlib.blake2b(
                        value.tobytes(), digest_size=16
                    ).hexdigest()
                }
            )
        elif isinstance(value, (list, tuple)):
            result.append(_hashable_args(tuple(value)))
        elif isinstance(value, dict):
            result.append(_hashable_kwargs(value))
        else:
            result.append(value)
    return result


def _hashable_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {key: _hashable_args((value,))[0] for key, value in sorted(kwargs.items())}

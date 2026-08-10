"""Performance profiling utilities for PINNs.

Provides lightweight performance monitoring including timing decorators,
CPU/GPU memory tracking, and simple reporting helpers. These utilities are
intended to complement the structured logging utilities and offer a programmatic
way to detect performance bottlenecks and regressions.

Example
-------
>>> from pinnlab.utils.profiling import profile_performance, MemoryTracker
>>> @profile_performance
... def step():
...     pass
>>> with MemoryTracker() as mt:
...     step()
...     print(mt.peak_memory)
"""

from __future__ import annotations

import csv
import json
import time
import tracemalloc
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Callable, List, Optional

try:  # optional
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional
    torch = None

try:  # optional
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional
    psutil = None

__all__ = [
    "profile_performance",
    "MemoryTracker",
    "PerformanceReport",
]


@dataclass
class PerformanceRecord:
    """Single profiling record."""

    name: str
    duration_sec: float
    cpu_mem_mb: Optional[float] = None
    gpu_mem_mb: Optional[float] = None


@dataclass
class PerformanceReport:
    """Collect and export profiling records."""

    records: List[PerformanceRecord] = field(default_factory=list)

    def add(self, record: PerformanceRecord) -> None:
        self.records.append(record)

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------
    def to_json(self, path: Path | str) -> None:
        path = Path(path)
        data = [r.__dict__ for r in self.records]
        path.write_text(json.dumps(data, indent=2))

    def to_csv(self, path: Path | str) -> None:
        path = Path(path)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["name", "duration_sec", "cpu_mem_mb", "gpu_mem_mb"]
            )
            writer.writeheader()
            for r in self.records:
                writer.writerow(r.__dict__)

    # Convenience summary ------------------------------------------------
    def slowest(self) -> Optional[PerformanceRecord]:
        if not self.records:
            return None
        return max(self.records, key=lambda r: r.duration_sec)


class MemoryTracker:
    """Context manager for tracking CPU and GPU memory usage."""

    def __enter__(self) -> "MemoryTracker":
        tracemalloc.start()
        self._start_snapshot = tracemalloc.take_snapshot()
        self.peak_memory = 0.0
        self.gpu_start = None
        self.gpu_peak_memory = None
        if torch is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.gpu_start = torch.cuda.memory_allocated()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        current, peak = tracemalloc.get_traced_memory()
        self.peak_memory = peak / (1024**2)
        tracemalloc.stop()
        if torch is not None and torch.cuda.is_available():
            peak_gpu = torch.cuda.max_memory_allocated() - (self.gpu_start or 0)
            self.gpu_peak_memory = peak_gpu / (1024**2)


def _current_memory_mb() -> Optional[float]:
    if psutil is None:
        return None
    process = psutil.Process()
    return process.memory_info().rss / (1024**2)


def _current_gpu_memory_mb() -> Optional[float]:
    if torch is None or not torch.cuda.is_available():
        return None
    return torch.cuda.memory_allocated() / (1024**2)


def profile_performance(
    func: Optional[Callable] = None,
    *,
    report: Optional[PerformanceReport] = None,
    name: Optional[str] = None,
) -> Callable:
    """Decorator to time function execution and track memory usage.

    Parameters
    ----------
    report:
        Optional :class:`PerformanceReport` to collect profiling data.
    name:
        Label recorded for the measurement. Defaults to the function's
        ``__qualname__``, which for a nested function reads like
        ``main.<locals>.run_train``. Pass an explicit name when the record is
        published somewhere durable -- benchmark dashboards key their history
        off this string, so a generated one is both unreadable and awkward to
        change later without splitting the series.
    """

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapped(*args, **kwargs):
            start_time = time.perf_counter()
            start_cpu = _current_memory_mb()
            start_gpu = _current_gpu_memory_mb()
            result = fn(*args, **kwargs)
            duration = time.perf_counter() - start_time
            end_cpu = _current_memory_mb()
            end_gpu = _current_gpu_memory_mb()
            cpu_delta = None
            gpu_delta = None
            if start_cpu is not None and end_cpu is not None:
                cpu_delta = max(end_cpu - start_cpu, 0.0)
            if start_gpu is not None and end_gpu is not None:
                gpu_delta = max(end_gpu - start_gpu, 0.0)
            if report is not None:
                report.add(
                    PerformanceRecord(
                        name=name or fn.__qualname__,
                        duration_sec=duration,
                        cpu_mem_mb=cpu_delta,
                        gpu_mem_mb=gpu_delta,
                    )
                )
            return result

        return wrapped

    if func is not None:
        return decorator(func)
    return decorator

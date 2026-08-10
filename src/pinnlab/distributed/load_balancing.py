"""Dynamic workload distribution and fault tolerance helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, MutableMapping, Sequence

import math
import time

__all__ = [
    "WorkerState",
    "DynamicLoadBalancer",
    "FailureTracker",
]


@dataclass
class WorkerState:
    """Runtime statistics for a worker/device."""

    name: str
    throughput: float = 0.0
    processed_batches: int = 0
    failures: int = 0
    last_heartbeat: float = field(default_factory=time.time)
    active: bool = True

    def update(self, batch_size: int, duration: float, smoothing: float) -> None:
        if duration <= 0:
            return
        samples_per_sec = batch_size / duration
        if self.throughput == 0.0:
            self.throughput = samples_per_sec
        else:
            self.throughput = (
                1 - smoothing
            ) * self.throughput + smoothing * samples_per_sec
        self.processed_batches += 1
        self.last_heartbeat = time.time()

    def mark_failure(self) -> None:
        self.failures += 1
        self.active = False
        self.last_heartbeat = time.time()

    def revive(self) -> None:
        self.active = True
        self.last_heartbeat = time.time()


class DynamicLoadBalancer:
    """Distribute work proportionally to worker throughput."""

    def __init__(self, workers: Sequence[str], smoothing: float = 0.2) -> None:
        if not workers:
            raise ValueError("At least one worker must be provided")
        self.smoothing = smoothing
        self.workers: MutableMapping[str, WorkerState] = {
            w: WorkerState(w) for w in workers
        }

    def record_batch(self, worker: str, batch_size: int, duration: float) -> None:
        if worker not in self.workers:
            self.workers[worker] = WorkerState(worker)
        self.workers[worker].update(batch_size, duration, self.smoothing)

    def deactivate(self, worker: str) -> None:
        if worker in self.workers:
            self.workers[worker].active = False

    def activate(self, worker: str) -> None:
        if worker in self.workers:
            self.workers[worker].revive()

    def allocate(self, total_batches: int) -> Dict[str, int]:
        if total_batches <= 0:
            return {w: 0 for w in self.workers}
        active_workers = [w for w in self.workers.values() if w.active]
        if not active_workers:
            raise RuntimeError("No active workers available")
        speeds = [w.throughput or 1.0 for w in active_workers]
        total_speed = sum(speeds)
        allocation: Dict[str, int] = {}
        for worker, speed in zip(active_workers, speeds):
            fraction = speed / total_speed if total_speed else 1.0 / len(active_workers)
            allocation[worker.name] = int(math.floor(total_batches * fraction))
        assigned = sum(allocation.values())
        for worker in active_workers:
            if assigned >= total_batches:
                break
            allocation[worker.name] += 1
            assigned += 1
        return allocation

    def scale_workers(self, new_workers: Iterable[str]) -> None:
        for worker in new_workers:
            if worker not in self.workers:
                self.workers[worker] = WorkerState(worker)

    def active_workers(self) -> List[str]:
        return [w.name for w in self.workers.values() if w.active]


class FailureTracker:
    """Track failures and trigger recovery hooks."""

    def __init__(self, threshold: int = 3) -> None:
        self.threshold = threshold
        self.records: Dict[str, int] = {}

    def record_failure(self, worker: str) -> bool:
        count = self.records.get(worker, 0) + 1
        self.records[worker] = count
        return count >= self.threshold

    def reset(self, worker: str) -> None:
        self.records.pop(worker, None)

"""Memory-efficient training utilities for Physics-Informed Neural Networks."""

from __future__ import annotations

import contextlib
import statistics
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

try:  # pragma: no cover - optional dependency
    import psutil  # type: ignore
except Exception:  # pragma: no cover - psutil is optional
    psutil = None  # type: ignore

from ..solvers.raissi_improved import ContinuousPINN, TrainConfig
from ..utils.checkpointing import CheckpointManager
from ..utils.logging import get_logger

logger = get_logger(__name__)

_MB = 1024**2


@dataclass
class MemoryUsageRecord:
    """Snapshot of memory usage before and after a profiled block."""

    step: Optional[int]
    tag: str
    before_mb: float
    after_mb: float
    peak_mb: float
    device: str
    timestamp: float = field(default_factory=time.time)


class MemoryProfiler:
    """Collect lightweight memory statistics during training."""

    def __init__(self, device: Optional[torch.device] = None) -> None:
        resolved = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.device = torch.device(resolved)
        self.history: List[MemoryUsageRecord] = []

    def _memory_allocated(self) -> float:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            return float(torch.cuda.memory_allocated(self.device))
        if psutil is not None:
            return float(psutil.Process().memory_info().rss)
        return 0.0

    def _peak_memory(self) -> float:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            return float(torch.cuda.max_memory_allocated(self.device))
        return self._memory_allocated()

    @contextlib.contextmanager
    def track(self, tag: str, step: Optional[int] = None):
        """Context manager capturing memory usage around a code block."""

        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        before = self._memory_allocated()
        try:
            yield
        finally:
            after = self._memory_allocated()
            peak = max(self._peak_memory(), after)
            record = MemoryUsageRecord(
                step=step,
                tag=tag,
                before_mb=before / _MB,
                after_mb=after / _MB,
                peak_mb=peak / _MB,
                device=self.device.type,
            )
            self.history.append(record)

    def profile_training_step(self, tag: str = "train_step") -> Callable:
        """Decorator that profiles memory for the wrapped callable."""

        def decorator(fn: Callable):
            def wrapper(*args, **kwargs):
                with self.track(tag):
                    return fn(*args, **kwargs)

            return wrapper

        return decorator

    def summary(self) -> Dict[str, float]:
        """Return aggregated statistics from recorded history."""

        if not self.history:
            return {}
        avg_delta = statistics.mean(
            rec.after_mb - rec.before_mb for rec in self.history
        )
        peak = max(rec.peak_mb for rec in self.history)
        return {
            "records": float(len(self.history)),
            "avg_delta_mb": avg_delta,
            "peak_mb": peak,
        }

    def recommendations(self, budget_bytes: Optional[int]) -> List[str]:
        """Generate human-readable optimisation suggestions."""

        if not self.history:
            return []

        recs: List[str] = []
        peak = max(rec.peak_mb for rec in self.history)
        if budget_bytes:
            budget_mb = budget_bytes / _MB
            if peak > 0.9 * budget_mb:
                recs.append(
                    "Peak memory usage approaches the configured budget; "
                    "consider reducing the collocation batch size or enabling "
                    "gradient checkpointing."
                )
        avg_growth = statistics.mean(
            max(0.0, rec.after_mb - rec.before_mb) for rec in self.history
        )
        if avg_growth > 50.0:
            recs.append(
                "Per-step memory growth is high; investigate caching of tensors "
                "between iterations or enable periodic data regeneration."
            )
        if not recs:
            recs.append("Memory usage is stable across training steps.")
        return recs


class MemoryEfficientPINN(ContinuousPINN):
    """Continuous PINN variant that configures memory-aware training."""

    def __init__(
        self,
        model: nn.Module,
        device: Union[str, torch.device],
        pde_residual_fn: Callable[[nn.Module, Tensor, Tensor, float], Tensor],
        cfg_burgers: Any,
        *,
        memory_budget_mb: Optional[int] = None,
        profiler: Optional[MemoryProfiler] = None,
    ) -> None:
        super().__init__(model, device, pde_residual_fn, cfg_burgers)
        self.default_memory_budget_mb = memory_budget_mb
        self.default_profiler = profiler
        self._prefetched_data: Optional[Dict[str, Tensor]] = None

    def _make_training_points(self, tcfg: TrainConfig) -> Dict[str, Tensor]:
        if self._prefetched_data is not None:
            data = self._prefetched_data
            self._prefetched_data = None
            return data
        return super()._make_training_points(tcfg)

    def _estimate_batch_size(self, tcfg: TrainConfig, data: Dict[str, Tensor]) -> int:
        total_points = data["tf"].shape[0]
        if total_points == 0:
            return 1
        if tcfg.collocation_batch_size is not None:
            return max(1, min(total_points, tcfg.collocation_batch_size))
        if self.device.type != "cuda" or not torch.cuda.is_available():
            heuristic = max(32, total_points // 4)
            return max(1, min(total_points, heuristic))

        sample_size = min(512, total_points)
        tf_sample = data["tf"][0:sample_size]
        xf_sample = data["xf"][0:sample_size]

        try:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            self.model.zero_grad(set_to_none=True)
            if tcfg.use_mixed_precision:
                dtype = self._resolve_amp_dtype(tcfg) or torch.float16
                context = torch.cuda.amp.autocast(dtype=dtype)
            else:
                context = contextlib.nullcontext()
            with context:
                residual = self.pde_residual_fn(
                    lambda tx: self._model_forward(tx, tcfg, requires_grad=True),
                    tf_sample,
                    xf_sample,
                    self.cfg.nu,
                )
                loss = torch.mean(residual**2)
            loss.backward()
            torch.cuda.synchronize(self.device)
            peak_bytes = torch.cuda.max_memory_allocated(self.device)
        except RuntimeError as exc:  # pragma: no cover - hardware specific
            logger.debug("Batch-size estimation failed", exc_info=exc)
            return max(1, min(total_points, 256))
        finally:
            self.model.zero_grad(set_to_none=True)

        per_sample = max(peak_bytes / max(sample_size, 1), 1.0)
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        except RuntimeError:  # pragma: no cover - CUDA < 11
            total_bytes = torch.cuda.get_device_properties(self.device).total_memory
            free_bytes = total_bytes - torch.cuda.memory_allocated(self.device)

        budget_bytes = self._memory_budget_bytes(tcfg)
        if budget_bytes is None and self.default_memory_budget_mb is not None:
            budget_bytes = int(self.default_memory_budget_mb * _MB)
        if budget_bytes is None:
            budget_bytes = total_bytes
        available_bytes = max(
            min(budget_bytes, total_bytes) - torch.cuda.memory_allocated(self.device),
            0,
        )
        available_bytes = max(available_bytes, free_bytes)

        optimal = int(max(1, (available_bytes * 0.8) / per_sample))
        return max(1, min(total_points, optimal))

    def train(
        self,
        tcfg: TrainConfig,
        weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        checkpoint_manager: Optional[CheckpointManager] = None,
        resume: bool = False,
        *,
        callbacks: Optional[Iterable[Callable[[int, Dict[str, float]], None]]] = None,
        distributed: Optional[Any] = None,
    ) -> Dict[str, List[float]]:
        config = replace(tcfg)
        data = super()._make_training_points(config)
        batch_size = self._estimate_batch_size(config, data)
        if config.collocation_batch_size is None:
            config.collocation_batch_size = batch_size
        if config.effective_batch_size is None:
            config.effective_batch_size = data["tf"].shape[0]
        if (
            config.memory_budget_mb is None
            and self.default_memory_budget_mb is not None
        ):
            config.memory_budget_mb = self.default_memory_budget_mb
        if config.memory_profiler is None:
            config.memory_profiler = self.default_profiler or MemoryProfiler(
                self.device
            )

        logger.info(
            "Configured memory-efficient training",
            extra={
                "collocation_batch_size": config.collocation_batch_size,
                "effective_batch_size": config.effective_batch_size,
                "memory_budget_mb": config.memory_budget_mb,
            },
        )

        self._prefetched_data = data
        return super().train(
            config,
            weights=weights,
            checkpoint_manager=checkpoint_manager,
            resume=resume,
            callbacks=callbacks,
            distributed=distributed,
        )


__all__ = ["MemoryProfiler", "MemoryEfficientPINN", "MemoryUsageRecord"]

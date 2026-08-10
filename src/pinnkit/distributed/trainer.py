"""Utilities for multi-GPU and distributed training.

This module provides a lightweight :class:`DistributedTrainer` that wraps a
PINN model for data, model or pipeline parallel execution.  The implementation
favours clarity over raw performance so that it can run in unit tests without
multiple GPUs.  When GPUs are available the trainer automatically detects them
and creates the appropriate parallel wrappers.  On CPU only setups it silently
falls back to single device execution.

The trainer exposes a :py:meth:`prepare_model` method used by the solvers to
initialise the underlying ``nn.Module`` for distributed training and a
:py:meth:`fit` helper that delegates to the solver's ``train`` method.  It also
provides gradient synchronisation helpers and performance accounting that are
reused by higher level orchestration utilities in :mod:`pinnkit.distributed`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import time

import torch
from torch import nn

from .communication import (
    GradientCompression,
    HierarchicalAverager,
    synchronise_gradients,
)
from .parallel import (
    DataParallelEngine,
    HybridParallelEngine,
    ModelParallelEngine,
    PipelineParallelEngine,
)


@dataclass
class _EnvInfo:
    """Runtime information about the available devices."""

    devices: Sequence[int | str]
    primary_device: torch.device


def _auto_devices() -> _EnvInfo:
    """Return available device identifiers and the primary device.

    CPU is returned when no GPU is available.  ``torch.cuda`` calls are wrapped
    so that unit tests can patch them easily.
    """

    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        ids = list(range(count))
        device = torch.device(f"cuda:{ids[0]}")
        return _EnvInfo(ids, device)
    return _EnvInfo(["cpu"], torch.device("cpu"))


class DistributedTrainer:
    """Simple distributed training orchestrator.

    Parameters
    ----------
    model:
        Solver instance or ``nn.Module`` to be wrapped.
    strategy:
        One of ``"data_parallel"``, ``"model_parallel"``, ``"pipeline"`` or
        ``"hybrid"`` to control how the model is partitioned across devices.
    devices:
        Sequence of device identifiers or ``"auto"`` to use all available GPUs.
    mixed_precision:
        Whether to enable ``torch.cuda.amp`` autocasting.
    """

    def __init__(
        self,
        model: Any,
        strategy: str = "data_parallel",
        devices: Sequence[int | str] | str = "auto",
        mixed_precision: bool = False,
        pipeline_chunks: int = 4,
        compression: GradientCompression | None = None,
        averager: HierarchicalAverager | None = None,
    ) -> None:
        if devices == "auto":
            env = _auto_devices()
        else:
            if isinstance(devices, Sequence) and not isinstance(devices, (str, bytes)):
                device_list = list(devices)
            else:
                device_list = [devices]  # type: ignore[arg-type]
            if not device_list:
                device_list = ["cpu"]
            first = device_list[0]
            if isinstance(first, int):
                primary = (
                    torch.device(f"cuda:{first}")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
            else:
                primary = torch.device(first)
            env = _EnvInfo(device_list, primary)
        self.devices: Sequence[int | str] = env.devices
        self.primary_device = env.primary_device
        self.strategy = strategy
        self.mixed_precision = mixed_precision
        self.pipeline_chunks = max(1, pipeline_chunks)
        self.compression = compression
        self.averager = averager or HierarchicalAverager()
        self._engine: (
            DataParallelEngine
            | ModelParallelEngine
            | PipelineParallelEngine
            | HybridParallelEngine
            | None
        ) = None
        self._current_model: nn.Module | None = None
        self._step_durations: list[float] = []

        # The trainer may receive either a solver containing ``model`` or the
        # model itself.  ``prepare_model`` returns the wrapped model and updates
        # the solver if required.
        self.model_ref = model
        if hasattr(model, "model"):
            wrapped = self.prepare_model(model.model)
            model.model = wrapped
            model.device = self.primary_device
        elif isinstance(model, nn.Module):
            self.model_ref = self.prepare_model(model)
        else:
            raise TypeError(
                "model must be an nn.Module or solver with a 'model' attribute"
            )

    # ------------------------------------------------------------------ utils
    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Wrap ``model`` for the requested distributed strategy."""

        engine = self._make_engine(model)
        self._engine = engine
        wrapped = engine.prepare()
        self._current_model = wrapped
        return wrapped

    def _make_engine(self, model: nn.Module):
        devices = self.devices
        if self.strategy == "model_parallel":
            return ModelParallelEngine(model, devices)
        if self.strategy == "pipeline":
            return PipelineParallelEngine(
                model, devices, microbatches=self.pipeline_chunks
            )
        if self.strategy == "hybrid":
            return HybridParallelEngine(
                model, devices, microbatches=self.pipeline_chunks
            )
        return DataParallelEngine(model, devices)

    # ---------------------------------------------------------------- fit
    def fit(self, *args, **kwargs):
        """Delegate training to the underlying solver's ``train`` method.

        Any positional or keyword arguments are forwarded to ``train``.  The
        method wraps the call in ``torch.cuda.amp.autocast`` when mixed
        precision is requested.
        """

        solver = self.model_ref
        if not hasattr(solver, "train"):
            raise AttributeError("Wrapped object has no 'train' method")

        train_fn: Callable[..., Any] = solver.train
        start = time.perf_counter()
        if self.mixed_precision and torch.cuda.is_available():

            def _train(*a, **k):
                with torch.cuda.amp.autocast():
                    return train_fn(*a, distributed=self, **k)

            result = _train(*args, **kwargs)
            self.record_step_duration(time.perf_counter() - start)
            return result
        result = train_fn(*args, distributed=self, **kwargs)
        self.record_step_duration(time.perf_counter() - start)
        return result

    # --------------------------------------------------------------- utilities
    def barrier(self) -> None:
        """Synchronise all processes if ``torch.distributed`` is initialised."""

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    # ------------------------------------------------------------ synchronise
    def synchronize_gradients(self, model: nn.Module | None = None) -> None:
        target = model or self._current_model
        if target is None and hasattr(self.model_ref, "model"):
            target = getattr(self.model_ref, "model")
        if target is None and isinstance(self.model_ref, nn.Module):
            target = self.model_ref
        if target is None:
            return
        synchronise_gradients(target, self.compression, self.averager)

    # ---------------------------------------------------------- performance
    def record_step_duration(self, duration: float) -> None:
        if duration > 0:
            self._step_durations.append(duration)

    def average_step_time(self) -> float:
        if not self._step_durations:
            return 0.0
        return sum(self._step_durations) / len(self._step_durations)

    def world_size(self) -> int:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size()
        if not self.devices:
            return 1
        gpu_count = len([d for d in self.devices if d != "cpu"])
        return gpu_count or 1

    def scaling_efficiency(self, serial_time: float) -> float:
        avg = self.average_step_time()
        if avg <= 0 or serial_time <= 0:
            return 0.0
        return serial_time / (avg * self.world_size())

    # --------------------------------------------------------------- helpers
    def current_model(self) -> nn.Module | None:
        if self._current_model is not None:
            return self._current_model
        if hasattr(self.model_ref, "model"):
            return getattr(self.model_ref, "model")
        if isinstance(self.model_ref, nn.Module):
            return self.model_ref
        return None

"""High level distributed orchestration for PINN training."""

from __future__ import annotations

import os
import time
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    TYPE_CHECKING,
)

import torch
from torch import nn

from .communication import (
    AsyncGradientBuffer,
    GradientCompression,
    HierarchicalAverager,
)
from .load_balancing import DynamicLoadBalancer, FailureTracker
from .trainer import DistributedTrainer

if TYPE_CHECKING:  # pragma: no cover - optional dependency for typing only
    from pinnkit.utils.checkpointing import CheckpointManager

__all__ = [
    "DistributedEnvironment",
    "DistributedPINNTrainer",
]


class DistributedEnvironment:
    """Utility class to manage torch.distributed initialisation."""

    def __init__(
        self,
        backend: str = "nccl",
        init_method: str = "env://",
        auto_init: bool = True,
    ) -> None:
        self.backend = backend
        self.init_method = init_method
        self.auto_init = auto_init
        self.initialised = False
        self.rank = 0
        self.world_size = 1

    def setup(self) -> None:
        if not torch.distributed.is_available():
            return
        if torch.distributed.is_initialized():
            self.initialised = True
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
            return
        if not self.auto_init:
            return
        env_ready = {"RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"} <= set(
            os.environ
        )
        if self.init_method == "env://" and not env_ready:
            return
        try:
            torch.distributed.init_process_group(
                backend=self.backend, init_method=self.init_method
            )
        except (RuntimeError, ValueError):
            return
        self.initialised = True
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()

    def barrier(self) -> None:
        if self.initialised and torch.distributed.is_available():
            torch.distributed.barrier()

    def cleanup(self) -> None:
        if (
            self.initialised
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            torch.distributed.destroy_process_group()
            self.initialised = False
            self.world_size = 1
            self.rank = 0


class DistributedPINNTrainer(DistributedTrainer):
    """Advanced trainer supporting multi-node and hybrid parallelism."""

    def __init__(
        self,
        model: Any,
        strategy: str = "data_parallel",
        backend: str = "nccl",
        init_method: str = "env://",
        devices: Sequence[int | str] | str = "auto",
        mixed_precision: bool = False,
        pipeline_chunks: int = 4,
        compression: GradientCompression | None = None,
        load_balancer: DynamicLoadBalancer | None = None,
    ) -> None:
        self.env = DistributedEnvironment(backend=backend, init_method=init_method)
        self.env.setup()
        self.async_buffer = AsyncGradientBuffer()
        super().__init__(
            model,
            strategy=strategy,
            devices=devices,
            mixed_precision=mixed_precision,
            pipeline_chunks=pipeline_chunks,
            compression=compression,
            averager=HierarchicalAverager(),
        )
        worker_count = max(self.world_size(), 1)
        worker_names = [f"worker-{idx}" for idx in range(worker_count)]
        self.load_balancer = load_balancer or DynamicLoadBalancer(worker_names)
        self.failure_tracker = FailureTracker()
        self.scaling_history: List[float] = []

    # ------------------------------------------------------------------ helpers
    @property
    def rank(self) -> int:
        return self.env.rank

    @property
    def world(self) -> int:
        return self.env.world_size

    def plan_batches(self, total_batches: int) -> Dict[str, int]:
        return self.load_balancer.allocate(total_batches)

    def record_batch(self, batch_size: int, duration: float) -> None:
        self.load_balancer.record_batch(self.worker_id, batch_size, duration)

    @property
    def worker_id(self) -> str:
        return f"worker-{self.rank}"

    # ----------------------------------------------------------------- training
    def train_distributed(
        self,
        dataloader: Iterable,
        step_fn: Callable[[Any], torch.Tensor],
        optimizer: torch.optim.Optimizer,
        epochs: int = 1,
        gradient_clip: Optional[float] = None,
    ) -> None:
        model = self.current_model()
        if model is None:
            raise RuntimeError("No model associated with trainer")
        for epoch in range(epochs):
            for batch in dataloader:
                start = time.perf_counter()
                loss = step_fn(batch)
                if not torch.is_tensor(loss):
                    raise TypeError("step_fn must return a torch.Tensor loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.synchronize_gradients(model)
                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
                duration = time.perf_counter() - start
                self.record_step_duration(duration)
                self.record_batch(self._infer_batch_size(batch), duration)
            if self.env.initialised:
                self.env.barrier()

    # ------------------------------------------------------------- async utils
    def queue_gradients(self, model: nn.Module | None = None) -> None:
        target = model or self.current_model()
        if target is None:
            return
        payload = {
            name: param.grad.detach().cpu()
            for name, param in target.named_parameters()
            if param.grad is not None
        }
        if payload:
            self.async_buffer.push(payload)

    def flush_async_gradients(self, model: nn.Module | None = None) -> None:
        target = model or self.current_model()
        if target is None:
            return
        params = dict(target.named_parameters())
        for gradients in self.async_buffer.pop_all():
            for name, grad in gradients.items():
                if name in params and params[name].grad is not None:
                    params[name].grad.add_(grad.to(params[name].grad.device))
        self.synchronize_gradients(target)

    # ----------------------------------------------------------- fault tolerance
    def recover_from_failure(
        self,
        worker: str,
        checkpoint_manager: Optional["CheckpointManager"] = None,
    ) -> Optional[int]:
        self.load_balancer.deactivate(worker)
        exceeded = self.failure_tracker.record_failure(worker)
        if not exceeded or checkpoint_manager is None:
            return None
        model = self.current_model()
        if model is None:
            return None
        try:
            model, _, step, _ = checkpoint_manager.load_latest(
                model=model, optimizer=None
            )
        except FileNotFoundError:
            return None
        self.failure_tracker.reset(worker)
        return step

    def scale_workers(self, new_workers: Sequence[str]) -> None:
        self.load_balancer.scale_workers(new_workers)

    def synchronise_checkpoints(self) -> None:
        self.env.barrier()

    # --------------------------------------------------------------- statistics
    def scaling_summary(self, serial_time: float) -> Dict[str, float]:
        avg_time = self.average_step_time()
        speedup = serial_time / avg_time if avg_time > 0 else 0.0
        efficiency = self.scaling_efficiency(serial_time)
        summary = {
            "speedup": speedup,
            "efficiency": efficiency,
            "world_size": float(self.world_size()),
        }
        self.scaling_history.append(efficiency)
        return summary

    # ---------------------------------------------------------------- internal
    @staticmethod
    def _infer_batch_size(batch: Any) -> int:
        if hasattr(batch, "size") and callable(getattr(batch, "size")):
            try:
                return int(batch.size(0))
            except Exception:  # pragma: no cover - defensive
                return 1
        if isinstance(batch, (list, tuple)) and batch:
            return DistributedPINNTrainer._infer_batch_size(batch[0])
        return 1

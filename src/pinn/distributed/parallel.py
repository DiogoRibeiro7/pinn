"""Parallel execution strategies for PINN models."""
from __future__ import annotations

from typing import List, Sequence

import torch
from torch import nn

__all__ = [
    "ParallelEngine",
    "DataParallelEngine",
    "ModelParallelEngine",
    "PipelineParallelEngine",
    "HybridParallelEngine",
]


def _normalise_devices(devices: Sequence[int | str] | None) -> List[str]:
    if not devices:
        return ["cpu"]
    norm: List[str] = []
    for dev in devices:
        if isinstance(dev, int):
            norm.append(f"cuda:{dev}")
        else:
            norm.append(str(dev))
    return norm


class ParallelEngine:
    """Base class for parallel strategies."""

    def __init__(self, model: nn.Module, devices: Sequence[int | str] | None = None) -> None:
        self.model = model
        self.devices = _normalise_devices(devices)
        self.primary_device = torch.device(self.devices[0]) if self.devices else torch.device("cpu")

    def prepare(self) -> nn.Module:
        return self.model.to(self.primary_device)


class DataParallelEngine(ParallelEngine):
    """Wrapper around :class:`torch.nn.DataParallel` with graceful CPU fallback."""

    def prepare(self) -> nn.Module:
        model = super().prepare()
        cuda_devices = [int(dev.split(":")[1]) for dev in self.devices if dev.startswith("cuda")]
        if cuda_devices and torch.cuda.is_available() and len(cuda_devices) > 1:
            return nn.DataParallel(model, device_ids=cuda_devices)
        return model


class ModelParallelEngine(ParallelEngine):
    """Split sequential models across multiple devices."""

    def prepare(self) -> nn.Module:
        model = super().prepare()
        cuda_devices = [torch.device(dev) for dev in self.devices if dev.startswith("cuda")]
        if not cuda_devices or len(cuda_devices) == 1:
            return model
        modules = list(model.children())
        if not modules:
            return model
        stride = max(1, len(modules) // len(cuda_devices))
        for idx, module in enumerate(modules):
            device = cuda_devices[min(idx // stride, len(cuda_devices) - 1)]
            module.to(device)
        return model


class PipelineParallelEngine(ParallelEngine):
    """Sequential micro-batch execution across devices."""

    def __init__(self, model: nn.Module, devices: Sequence[int | str] | None = None, microbatches: int = 4) -> None:
        super().__init__(model, devices)
        self.microbatches = max(1, microbatches)

    def prepare(self) -> nn.Module:
        model = super().prepare()
        devices = [torch.device(dev) for dev in self.devices]
        if len(devices) == 1:
            return model
        modules = list(model.children())
        if not modules:
            return model
        stages = max(1, len(modules) // len(devices))
        stage_devices = [devices[min(i // stages, len(devices) - 1)] for i in range(len(modules))]

        class _PipelineWrapper(nn.Module):
            def __init__(self, modules: List[nn.Module], stage_devices: List[torch.device], microbatches: int) -> None:
                super().__init__()
                self.stages = nn.ModuleList(modules)
                self.stage_devices = stage_devices
                self.microbatches = microbatches

            def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                splits = torch.chunk(x, chunks=self.microbatches)
                outputs = []
                for split in splits:
                    activations = split
                    for module, device in zip(self.stages, self.stage_devices):
                        activations = module.to(device)(activations.to(device))
                    outputs.append(activations.to(split.device))
                return torch.cat(outputs, dim=0)

        return _PipelineWrapper(modules, stage_devices, self.microbatches)


class HybridParallelEngine(ParallelEngine):
    """Combine data and pipeline parallelism."""

    def __init__(
        self,
        model: nn.Module,
        devices: Sequence[int | str] | None = None,
        microbatches: int = 4,
    ) -> None:
        super().__init__(model, devices)
        self.microbatches = microbatches

    def prepare(self) -> nn.Module:
        base = PipelineParallelEngine(self.model, self.devices, self.microbatches).prepare()
        if isinstance(base, nn.DataParallel):
            return base
        return DataParallelEngine(base, self.devices).prepare()

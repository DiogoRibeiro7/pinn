"""Communication utilities for distributed PINN training.

This module contains gradient synchronisation helpers, compression
strategies, asynchronous buffers and hierarchical averaging utilities that
work both with and without an initialised ``torch.distributed`` process
group.  The implementations are intentionally lightweight so that they can
be exercised from CPU-only unit tests while still providing the hooks needed
for multi-GPU deployments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import queue

import torch

__all__ = [
    "TensorDict",
    "CompressedGradient",
    "GradientCompression",
    "AsyncGradientBuffer",
    "HierarchicalAverager",
    "synchronise_gradients",
]

TensorDict = Dict[str, torch.Tensor]


@dataclass
class CompressedGradient:
    """Container holding compressed gradient information."""

    indices: Optional[torch.Tensor]
    values: torch.Tensor
    shape: torch.Size
    scale: Optional[torch.Tensor] = None
    bits: Optional[int] = None

    def to(self, device: torch.device) -> "CompressedGradient":
        return CompressedGradient(
            indices=self.indices.to(device) if self.indices is not None else None,
            values=self.values.to(device),
            shape=self.shape,
            scale=self.scale.to(device) if self.scale is not None else None,
            bits=self.bits,
        )


class GradientCompression:
    """Top-k sparsification with optional uniform quantisation."""

    def __init__(self, compression_ratio: float = 0.0, quantization_bits: Optional[int] = None) -> None:
        if compression_ratio < 0 or compression_ratio >= 1:
            raise ValueError("compression_ratio must be in [0, 1)")
        if quantization_bits is not None and quantization_bits <= 1:
            raise ValueError("quantization_bits must be greater than 1")
        self.compression_ratio = float(compression_ratio)
        self.quantization_bits = quantization_bits

    @property
    def enabled(self) -> bool:
        return self.compression_ratio > 0.0 or self.quantization_bits is not None

    def compress(self, gradients: TensorDict) -> Dict[str, CompressedGradient]:
        compressed: Dict[str, CompressedGradient] = {}
        for name, grad in gradients.items():
            flat = grad.detach().flatten()
            if not self.enabled:
                compressed[name] = CompressedGradient(None, flat.cpu(), grad.shape, None, None)
                continue

            k = int(max(1, flat.numel() * max(self.compression_ratio, 1e-12)))
            if k >= flat.numel():
                indices = None
                values = flat.cpu()
            else:
                _, topk_idx = torch.topk(flat.abs(), k)
                values = flat[topk_idx].cpu()
                indices = topk_idx.cpu()

            scale = None
            bits = self.quantization_bits
            if bits is not None:
                scale = values.abs().max().clamp(min=1e-12)
                levels = 2 ** bits - 1
                values = torch.round(values / scale * (levels / 2)).to(torch.int16)
            compressed[name] = CompressedGradient(indices, values, grad.shape, scale, bits)
        return compressed

    def decompress(self, compressed: Dict[str, CompressedGradient]) -> TensorDict:
        gradients: TensorDict = {}
        for name, comp in compressed.items():
            if comp.indices is None:
                values = comp.values
                if comp.bits is not None:
                    levels = 2 ** comp.bits - 1
                    assert comp.scale is not None
                    values = values.to(torch.float32) * comp.scale / (levels / 2)
                grad = values.reshape(comp.shape)
                gradients[name] = grad
                continue

            total = int(torch.tensor(comp.shape).prod())
            tensor = torch.zeros(total, dtype=torch.float32)
            values = comp.values
            if comp.bits is not None:
                assert comp.scale is not None
                levels = 2 ** comp.bits - 1
                values = values.to(torch.float32) * comp.scale / (levels / 2)
            tensor[comp.indices.long()] = values.to(torch.float32)
            gradients[name] = tensor.reshape(comp.shape)
        return gradients


class AsyncGradientBuffer:
    """Thread-safe queue for asynchronous gradient updates."""

    def __init__(self) -> None:
        self._queue: "queue.Queue[Dict[str, torch.Tensor]]" = queue.Queue()

    def push(self, gradients: TensorDict) -> None:
        cpu_gradients = {k: v.detach().cpu().clone() for k, v in gradients.items()}
        self._queue.put(cpu_gradients)

    def pop_all(self) -> List[TensorDict]:
        items: List[TensorDict] = []
        while not self._queue.empty():
            items.append(self._queue.get())
        return items


class HierarchicalAverager:
    """Hierarchical averaging helper supporting ring/tree reductions."""

    def __init__(self, group_size: int = 0) -> None:
        self.group_size = max(0, group_size)

    def average(self, gradients: TensorDict, world_size: int = 1) -> TensorDict:
        if world_size <= 1:
            return gradients
        reduced: TensorDict = {}
        for name, grad in gradients.items():
            tensor = grad.clone()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
                tensor /= float(world_size)
            else:
                tensor /= float(world_size)
            reduced[name] = tensor
        return reduced


def synchronise_gradients(
    model: torch.nn.Module,
    compression: Optional[GradientCompression] = None,
    averager: Optional[HierarchicalAverager] = None,
) -> None:
    """Synchronise gradients across workers with optional compression."""

    grads: TensorDict = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.detach()
    if not grads:
        return

    if compression is not None:
        decompressed = compression.decompress(compression.compress(grads))
        grads = {name: decompressed[name].to(param.grad.device) for name, param in model.named_parameters() if param.grad is not None}

    world_size = 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    if averager is None:
        averager = HierarchicalAverager()
    averaged = averager.average(grads, world_size)
    for name, param in model.named_parameters():
        if param.grad is not None and name in averaged:
            param.grad.copy_(averaged[name].to(param.grad.device))

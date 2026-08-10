"""Characteristic physical scales for nondimensional PINN training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch
from torch import Tensor


def _to_1d_tensor(
    values: Sequence[float] | Tensor,
    *,
    name: str,
    dtype: torch.dtype | None = None,
) -> Tensor:
    tensor = torch.as_tensor(values, dtype=dtype or torch.float64)
    if tensor.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional, got shape {tuple(tensor.shape)}"
        )
    if tensor.numel() < 1:
        raise ValueError(f"{name} must contain at least one value")
    return tensor


@dataclass(frozen=True)
class CharacteristicScales:
    """Physical scales and offsets for nondimensionalization.

    The mapping is

    ``x* = (x - x0) / L`` and ``u* = (u - u0) / U``.

    Args:
        input_scales: Positive coordinate scales such as ``(T, L)``.
        output_scales: Positive state/output scales such as ``(U,)``.
        input_offsets: Coordinate offsets. Defaults to zeros.
        output_offsets: Output offsets. Defaults to zeros.
        parameter_scales: Optional named positive parameter scales.
    """

    input_scales: Sequence[float] | Tensor
    output_scales: Sequence[float] | Tensor
    input_offsets: Sequence[float] | Tensor | None = None
    output_offsets: Sequence[float] | Tensor | None = None
    parameter_scales: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate dimensions and positivity."""
        input_scales = _to_1d_tensor(self.input_scales, name="input_scales")
        output_scales = _to_1d_tensor(self.output_scales, name="output_scales")
        if torch.any(input_scales <= 0):
            raise ValueError("input_scales must be positive")
        if torch.any(output_scales <= 0):
            raise ValueError("output_scales must be positive")

        if self.input_offsets is None:
            input_offsets = torch.zeros_like(input_scales)
        else:
            input_offsets = _to_1d_tensor(self.input_offsets, name="input_offsets")
        if self.output_offsets is None:
            output_offsets = torch.zeros_like(output_scales)
        else:
            output_offsets = _to_1d_tensor(self.output_offsets, name="output_offsets")

        if input_offsets.shape != input_scales.shape:
            raise ValueError("input_offsets must match input_scales")
        if output_offsets.shape != output_scales.shape:
            raise ValueError("output_offsets must match output_scales")
        for name, scale in self.parameter_scales.items():
            if scale <= 0:
                raise ValueError(f"parameter scale {name!r} must be positive")

        object.__setattr__(self, "input_scales", input_scales)
        object.__setattr__(self, "output_scales", output_scales)
        object.__setattr__(self, "input_offsets", input_offsets)
        object.__setattr__(self, "output_offsets", output_offsets)

    @property
    def input_dim(self) -> int:
        """Return the number of coordinate dimensions."""
        return int(self.input_tensor.numel())

    @property
    def output_dim(self) -> int:
        """Return the number of output dimensions."""
        return int(self.output_tensor.numel())

    @property
    def input_tensor(self) -> Tensor:
        """Return input scales as a tensor."""
        return (
            self.input_scales
            if isinstance(self.input_scales, Tensor)
            else torch.as_tensor(self.input_scales)
        )

    @property
    def output_tensor(self) -> Tensor:
        """Return output scales as a tensor."""
        return (
            self.output_scales
            if isinstance(self.output_scales, Tensor)
            else torch.as_tensor(self.output_scales)
        )

    @property
    def input_offset_tensor(self) -> Tensor:
        """Return input offsets as a tensor."""
        if isinstance(self.input_offsets, Tensor):
            return self.input_offsets
        return torch.as_tensor(self.input_offsets)

    @property
    def output_offset_tensor(self) -> Tensor:
        """Return output offsets as a tensor."""
        if isinstance(self.output_offsets, Tensor):
            return self.output_offsets
        return torch.as_tensor(self.output_offsets)

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "CharacteristicScales":
        """Return a copy with tensor fields moved to ``device`` and ``dtype``."""
        resolved_dtype = dtype or self.input_tensor.dtype
        return CharacteristicScales(
            input_scales=self.input_tensor.to(device=device, dtype=resolved_dtype),
            output_scales=self.output_tensor.to(device=device, dtype=resolved_dtype),
            input_offsets=self.input_offset_tensor.to(
                device=device, dtype=resolved_dtype
            ),
            output_offsets=self.output_offset_tensor.to(
                device=device, dtype=resolved_dtype
            ),
            parameter_scales=dict(self.parameter_scales),
        )

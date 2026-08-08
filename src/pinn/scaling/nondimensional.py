"""Physical nondimensionalization transforms."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .characteristic import CharacteristicScales


@dataclass(frozen=True)
class Nondimensionalizer:
    """Convert coordinates and fields between dimensional and dimensionless forms."""

    scales: CharacteristicScales

    def _input_scales(self, like: Tensor) -> Tensor:
        return self.scales.input_tensor.to(device=like.device, dtype=like.dtype)

    def _output_scales(self, like: Tensor) -> Tensor:
        return self.scales.output_tensor.to(device=like.device, dtype=like.dtype)

    def _input_offsets(self, like: Tensor) -> Tensor:
        return self.scales.input_offset_tensor.to(device=like.device, dtype=like.dtype)

    def _output_offsets(self, like: Tensor) -> Tensor:
        return self.scales.output_offset_tensor.to(device=like.device, dtype=like.dtype)

    def validate_input(self, coordinates: Tensor) -> None:
        """Validate coordinate tensor shape."""
        if coordinates.ndim != 2 or coordinates.shape[1] != self.scales.input_dim:
            raise ValueError(
                "coordinates must have shape "
                f"(N, {self.scales.input_dim}), got {tuple(coordinates.shape)}"
            )

    def validate_output(self, values: Tensor) -> None:
        """Validate output tensor shape."""
        if values.ndim != 2 or values.shape[1] != self.scales.output_dim:
            raise ValueError(
                f"values must have shape (N, {self.scales.output_dim}), "
                f"got {tuple(values.shape)}"
            )

    def to_dimensionless_inputs(self, coordinates: Tensor) -> Tensor:
        """Map dimensional coordinates ``x`` to ``x*``."""
        self.validate_input(coordinates)
        return (coordinates - self._input_offsets(coordinates)) / self._input_scales(
            coordinates
        )

    def to_dimensional_inputs(self, coordinates_star: Tensor) -> Tensor:
        """Map dimensionless coordinates ``x*`` to dimensional ``x``."""
        self.validate_input(coordinates_star)
        return coordinates_star * self._input_scales(
            coordinates_star
        ) + self._input_offsets(coordinates_star)

    def to_dimensionless_outputs(self, values: Tensor) -> Tensor:
        """Map dimensional outputs ``u`` to ``u*``."""
        self.validate_output(values)
        return (values - self._output_offsets(values)) / self._output_scales(values)

    def to_dimensional_outputs(self, values_star: Tensor) -> Tensor:
        """Map dimensionless outputs ``u*`` to dimensional ``u``."""
        self.validate_output(values_star)
        return values_star * self._output_scales(values_star) + self._output_offsets(
            values_star
        )

    def to_dimensionless(self, values: Tensor, *, kind: str = "input") -> Tensor:
        """Map dimensional ``input`` or ``output`` tensors to dimensionless form."""
        if kind == "input":
            return self.to_dimensionless_inputs(values)
        if kind == "output":
            return self.to_dimensionless_outputs(values)
        raise ValueError("kind must be 'input' or 'output'")

    def to_dimensional(self, values: Tensor, *, kind: str = "input") -> Tensor:
        """Map dimensionless ``input`` or ``output`` tensors to dimensional form."""
        if kind == "input":
            return self.to_dimensional_inputs(values)
        if kind == "output":
            return self.to_dimensional_outputs(values)
        raise ValueError("kind must be 'input' or 'output'")

    def derivative_scale_factor(
        self,
        *,
        output_index: int,
        coordinate_index: int,
        order: int = 1,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Return factor converting dimensionless derivatives to dimensional ones.

        For ``u = U u*`` and ``x = L x*`` this returns ``U / L**order``.
        """
        if output_index < 0 or output_index >= self.scales.output_dim:
            raise ValueError("output_index is out of range")
        if coordinate_index < 0 or coordinate_index >= self.scales.input_dim:
            raise ValueError("coordinate_index is out of range")
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")
        output_scale = self.scales.output_tensor.to(device=device, dtype=dtype)[
            output_index
        ]
        input_scale = self.scales.input_tensor.to(device=device, dtype=dtype)[
            coordinate_index
        ]
        return output_scale / input_scale.pow(order)

    def parameter_to_dimensionless(self, name: str, value: Tensor | float) -> Tensor:
        """Scale a named dimensional parameter by its characteristic value."""
        if name not in self.scales.parameter_scales:
            raise KeyError(f"unknown parameter scale {name!r}")
        tensor = torch.as_tensor(value)
        scale = torch.as_tensor(
            self.scales.parameter_scales[name],
            device=tensor.device,
            dtype=tensor.dtype,
        )
        return tensor / scale

    def parameter_to_dimensional(self, name: str, value_star: Tensor | float) -> Tensor:
        """Map a dimensionless named parameter back to dimensional units."""
        if name not in self.scales.parameter_scales:
            raise KeyError(f"unknown parameter scale {name!r}")
        tensor = torch.as_tensor(value_star)
        scale = torch.as_tensor(
            self.scales.parameter_scales[name],
            device=tensor.device,
            dtype=tensor.dtype,
        )
        return tensor * scale

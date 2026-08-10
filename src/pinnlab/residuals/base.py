"""Residual and derivative contracts for PDE training."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor, nn


class DerivativeBackend(Protocol):
    """Protocol for coordinate derivatives used by PDE residuals."""

    def partial_derivative(
        self,
        outputs: Tensor,
        coordinates: Tensor,
        *,
        output_index: int = 0,
        coordinate_index: int = 0,
        order: int = 1,
    ) -> Tensor:
        """Return a selected partial derivative."""


class ResidualForm(Protocol):
    """Protocol for evaluating a PDE residual form."""

    def evaluate(
        self,
        problem: object,
        model: nn.Module,
        coordinates: Tensor,
        derivative_backend: DerivativeBackend,
    ) -> Tensor:
        """Evaluate residual values at coordinates."""


class AutogradDerivativeBackend:
    """Derivative backend implemented with ``torch.autograd.grad``."""

    def partial_derivative(
        self,
        outputs: Tensor,
        coordinates: Tensor,
        *,
        output_index: int = 0,
        coordinate_index: int = 0,
        order: int = 1,
    ) -> Tensor:
        """Return ``d^order outputs[..., output_index] / dx_i^order``.

        Args:
            outputs: Model outputs with shape ``(N, output_dim)``.
            coordinates: Input coordinates with shape ``(N, input_dim)``.
            output_index: Component of ``outputs`` to differentiate.
            coordinate_index: Coordinate dimension.
            order: Derivative order, currently one or greater by repeated
                first derivatives.
        """
        if outputs.ndim != 2:
            raise ValueError("outputs must have shape (N, output_dim)")
        if coordinates.ndim != 2:
            raise ValueError("coordinates must have shape (N, input_dim)")
        if output_index < 0 or output_index >= outputs.shape[1]:
            raise ValueError("output_index is out of range")
        if coordinate_index < 0 or coordinate_index >= coordinates.shape[1]:
            raise ValueError("coordinate_index is out of range")
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")

        value = outputs[:, [output_index]]
        for _ in range(order):
            if not value.requires_grad:
                return torch.zeros(
                    (coordinates.shape[0], 1),
                    device=coordinates.device,
                    dtype=coordinates.dtype,
                )
            grads = torch.autograd.grad(
                value,
                coordinates,
                grad_outputs=torch.ones_like(value),
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if grads is None:
                return torch.zeros(
                    (coordinates.shape[0], 1),
                    device=coordinates.device,
                    dtype=coordinates.dtype,
                )
            value = grads[:, [coordinate_index]]
        return value

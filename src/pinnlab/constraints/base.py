"""Constraint contracts for PINN training."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import torch
from torch import Tensor, nn

from ..geometry import Geometry
from ..residuals import DerivativeBackend


class ConstraintKind(str, Enum):
    """Supported semantic kinds of constraints."""

    INITIAL = "initial"
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    PERIODIC = "periodic"


@dataclass(frozen=True)
class ConstraintLoss:
    """Named loss value returned by a constraint."""

    name: str
    value: Tensor


class Constraint(Protocol):
    """Protocol for constraints that can evaluate their own training loss."""

    @property
    def name(self) -> str:
        """Return the stable loss-component name."""

    @property
    def kind(self) -> ConstraintKind:
        """Return the semantic kind of the constraint."""

    @property
    def default_sample_count(self) -> int:
        """Return the sample count used when the trainer has no override."""

    def loss(
        self,
        model: nn.Module,
        geometry: Geometry,
        derivative_backend: DerivativeBackend,
        *,
        n_samples: int | None = None,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> ConstraintLoss:
        """Evaluate this constraint as a scalar loss."""

"""Training state containers for the generic PINN trainer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class TrainingState:
    """Loss and diagnostic values recorded at one optimization step."""

    step: int
    total_loss: float
    named_losses: Mapping[str, float]
    weights: Mapping[str, float]
    elapsed_time: float
    diagnostics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingResult:
    """Result returned by the generic trainer."""

    history: tuple[TrainingState, ...]
    final_state: TrainingState

    @property
    def final_loss(self) -> float:
        """Return the final recorded total loss."""
        return self.final_state.total_loss

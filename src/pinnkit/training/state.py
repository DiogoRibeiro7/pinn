"""Training state containers for the generic PINN trainer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class TrainingState:
    """Loss and diagnostic values recorded at one optimization step.

    Attributes:
        step: Optimizer step the record belongs to. L-BFGS contributes a single
            record whose step is the total budget, since its internal line
            search does not expose per-iteration boundaries.
        total_loss: Weighted sum of the named losses.
        named_losses: Unweighted loss per component, keyed by ``"pde"`` or a
            constraint name.
        weights: Weight applied to each named loss.
        elapsed_time: Seconds since training began.
        diagnostics: Values that describe the run rather than drive it, such as
            the smallest causal weight or refinement counters. These are
            excluded from ``total_loss``.
        phase: Which optimizer produced the record -- ``"adam"``, ``"lbfgs"``,
            or ``"initial"`` for the single record emitted when no optimizer
            steps were requested at all.
    """

    step: int
    total_loss: float
    named_losses: Mapping[str, float]
    weights: Mapping[str, float]
    elapsed_time: float
    diagnostics: Mapping[str, float] = field(default_factory=dict)
    phase: str = "adam"


@dataclass(frozen=True)
class TrainingResult:
    """Result returned by the generic trainer.

    Attributes:
        history: Every recorded state, in order. Adam contributes one record
            per ``log_every`` steps plus the first and last; L-BFGS contributes
            one final record.
        final_state: The last recorded state, repeated here for convenience.
    """

    history: tuple[TrainingState, ...]
    final_state: TrainingState

    @property
    def final_loss(self) -> float:
        """Return the final recorded total loss."""
        return self.final_state.total_loss

    def states_for_phase(self, phase: str) -> tuple[TrainingState, ...]:
        """Return the recorded states produced by one optimizer.

        Args:
            phase: ``"adam"``, ``"lbfgs"`` or ``"initial"``.

        Returns:
            The matching states, in order.
        """
        return tuple(state for state in self.history if state.phase == phase)

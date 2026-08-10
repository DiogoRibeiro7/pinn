"""Callback hooks for the generic PINN trainer.

Callbacks are the supported way to observe a training run without editing the
trainer or post-processing its history. They receive the same immutable
:class:`~pinnlab.training.state.TrainingState` records the trainer stores, so a
logger, a checkpointer and a custom diagnostic can coexist without competing
for the training loop.

Subclass :class:`TrainerCallback` and override only the hooks you need; the
base implementations do nothing. Callbacks are invoked in the order given, and
exceptions are allowed to propagate -- a checkpoint that silently fails to
write is worse than a run that stops.

    class StopWhenConverged(TrainerCallback):
        def on_state_recorded(self, state):
            if state.total_loss < 1e-6:
                raise RuntimeError(f"converged at step {state.step}")

    trainer.train(problem, model, callbacks=[StopWhenConverged()])
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from torch import nn

from ..utils.logging import get_logger
from .state import TrainingResult, TrainingState

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type checking only
    from ..problems import PDEProblem

logger = get_logger(__name__)

__all__ = ["TrainerCallback", "LoggingCallback", "HistoryCollector"]


class TrainerCallback:
    """Base class for trainer callbacks. Every hook defaults to doing nothing."""

    def on_train_begin(self, problem: "PDEProblem", model: nn.Module) -> None:
        """Called once before the first optimizer step."""

    def on_state_recorded(self, state: TrainingState) -> None:
        """Called each time the trainer records a state.

        This fires on the trainer's own logging schedule, not on every step, so
        a callback sees exactly what lands in the history.
        """

    def on_phase_end(self, phase: str, state: TrainingState) -> None:
        """Called when an optimizer finishes, with its last recorded state.

        Args:
            phase: ``"adam"`` or ``"lbfgs"``.
            state: The final state recorded during that phase.
        """

    def on_train_end(self, result: TrainingResult) -> None:
        """Called once after training, with the assembled result."""


class LoggingCallback(TrainerCallback):
    """Log each recorded state through the package logger."""

    def __init__(self, level: str = "info") -> None:
        """Create the callback.

        Args:
            level: Logger method to call, such as ``"info"`` or ``"debug"``.

        Raises:
            ValueError: If ``level`` is not a logger method.
        """
        if not hasattr(logger, level):
            raise ValueError(f"unknown log level: {level}")
        self.level = level

    def on_state_recorded(self, state: TrainingState) -> None:
        """Emit one log line per recorded state."""
        getattr(logger, self.level)(
            "Training step",
            extra={
                "step": state.step,
                "phase": state.phase,
                "total_loss": state.total_loss,
                **state.named_losses,
            },
        )

    def on_phase_end(self, phase: str, state: TrainingState) -> None:
        """Note the transition between optimizers."""
        getattr(logger, self.level)(
            "Phase complete",
            extra={"phase": phase, "step": state.step, "loss": state.total_loss},
        )


class HistoryCollector(TrainerCallback):
    """Accumulate states as they are recorded.

    The trainer already returns the full history, so this is mainly useful for
    watching a run in progress, or in tests that need to assert on the order
    hooks fired rather than on the final result.
    """

    def __init__(self) -> None:
        """Create an empty collector."""
        self.states: list[TrainingState] = []
        self.phases: list[str] = []
        self.began = False
        self.ended = False

    def on_train_begin(self, problem: "PDEProblem", model: nn.Module) -> None:
        """Record that training started."""
        self.began = True

    def on_state_recorded(self, state: TrainingState) -> None:
        """Store the state."""
        self.states.append(state)

    def on_phase_end(self, phase: str, state: TrainingState) -> None:
        """Store the completed phase name."""
        self.phases.append(phase)

    def on_train_end(self, result: TrainingResult) -> None:
        """Record that training finished."""
        self.ended = True


def _notify(callbacks: Sequence[TrainerCallback], hook: str, *args: object) -> None:
    """Invoke ``hook`` on every callback in order.

    Args:
        callbacks: Callbacks to notify.
        hook: Method name to call.
        *args: Positional arguments forwarded to the hook.
    """
    for callback in callbacks:
        getattr(callback, hook)(*args)

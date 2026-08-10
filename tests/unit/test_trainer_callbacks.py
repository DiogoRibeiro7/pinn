"""Tests for trainer callbacks and optimizer-phase reporting.

These cover the observation surface of the generic trainer: the callback hooks
and the ``phase`` marker that says which optimizer produced a recorded state.
They deliberately use tiny budgets -- what matters is the order and content of
the notifications, not whether the network learned anything.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from pinnlab.models import MLP
from pinnlab.problems import HeatProblem
from pinnlab.training import (
    HistoryCollector,
    LoggingCallback,
    OptimizerConfig,
    Trainer,
    TrainerCallback,
    TrainerConfig,
)
from pinnlab.training.state import TrainingState


@pytest.fixture
def problem() -> HeatProblem:
    return HeatProblem()


@pytest.fixture
def model() -> nn.Module:
    return MLP(in_dim=2, hidden_layers=2, width=8, out_dim=1)


def _config(adam: int = 4, lbfgs: int = 0, log_every: int = 1) -> TrainerConfig:
    """Return a trainer config with a budget small enough for a unit test."""
    return TrainerConfig(
        collocation_count=32,
        log_every=log_every,
        optimizer=OptimizerConfig(adam_steps=adam, lbfgs_max_iter=lbfgs),
    )


class TestPhaseReporting:
    def test_adam_states_are_labelled(self, problem, model):
        result = Trainer(_config(adam=3)).train(problem, model)
        assert result.history
        assert {state.phase for state in result.history} == {"adam"}

    def test_lbfgs_contributes_a_labelled_final_state(self, problem, model):
        result = Trainer(_config(adam=2, lbfgs=2)).train(problem, model)
        assert result.final_state.phase == "lbfgs"
        assert result.states_for_phase("adam")
        assert len(result.states_for_phase("lbfgs")) == 1

    def test_phases_appear_in_optimizer_order(self, problem, model):
        result = Trainer(_config(adam=2, lbfgs=2)).train(problem, model)
        phases = [state.phase for state in result.history]
        assert phases == sorted(phases, key=lambda p: 0 if p == "adam" else 1)

    def test_no_steps_still_reports_a_starting_state(self, problem, model):
        """With no optimizer budget the trainer reports where the model began,
        rather than returning an empty history."""
        result = Trainer(_config(adam=0, lbfgs=0)).train(problem, model)
        assert len(result.history) == 1
        assert result.final_state.phase == "initial"
        assert result.final_state.step == 0

    def test_states_for_phase_filters(self, problem, model):
        result = Trainer(_config(adam=2, lbfgs=1)).train(problem, model)
        assert all(s.phase == "adam" for s in result.states_for_phase("adam"))
        assert result.states_for_phase("nonexistent") == ()

    def test_lbfgs_step_counts_the_whole_budget(self, problem, model):
        result = Trainer(_config(adam=3, lbfgs=5)).train(problem, model)
        assert result.final_state.step == 3 + 5


class TestCallbackHooks:
    def test_hooks_fire_in_order(self, problem, model):
        collector = HistoryCollector()
        result = Trainer(_config(adam=3)).train(problem, model, callbacks=[collector])

        assert collector.began
        assert collector.ended
        assert collector.phases == ["adam"]
        # Every recorded state reached the callback, in the same order.
        assert [s.step for s in collector.states] == [s.step for s in result.history]

    def test_callback_sees_both_phase_ends(self, problem, model):
        collector = HistoryCollector()
        Trainer(_config(adam=2, lbfgs=2)).train(problem, model, callbacks=[collector])
        assert collector.phases == ["adam", "lbfgs"]

    def test_states_match_the_returned_history(self, problem, model):
        collector = HistoryCollector()
        result = Trainer(_config(adam=4, lbfgs=1)).train(
            problem, model, callbacks=[collector]
        )
        assert collector.states == list(result.history)

    def test_multiple_callbacks_are_all_notified(self, problem, model):
        first, second = HistoryCollector(), HistoryCollector()
        Trainer(_config(adam=2)).train(problem, model, callbacks=[first, second])
        assert first.states and second.states
        assert [s.step for s in first.states] == [s.step for s in second.states]

    def test_callbacks_are_invoked_in_the_given_order(self, problem, model):
        calls: list[str] = []

        class Recorder(TrainerCallback):
            def __init__(self, label: str) -> None:
                self.label = label

            def on_state_recorded(self, state: TrainingState) -> None:
                calls.append(self.label)

        Trainer(_config(adam=1)).train(
            problem, model, callbacks=[Recorder("first"), Recorder("second")]
        )
        assert calls == ["first", "second"]

    def test_exceptions_propagate(self, problem, model):
        """A callback that fails stops the run. A checkpoint that silently
        failed to write would be worse than a visible error."""

        class Boom(TrainerCallback):
            def on_state_recorded(self, state: TrainingState) -> None:
                raise RuntimeError("callback failed")

        with pytest.raises(RuntimeError, match="callback failed"):
            Trainer(_config(adam=2)).train(problem, model, callbacks=[Boom()])

    def test_training_works_without_callbacks(self, problem, model):
        result = Trainer(_config(adam=2)).train(problem, model)
        assert result.history

    def test_base_callback_hooks_are_no_ops(self, problem, model):
        """Subclasses override only what they need, so the base must tolerate
        being used unchanged."""
        result = Trainer(_config(adam=2)).train(
            problem, model, callbacks=[TrainerCallback()]
        )
        assert result.history

    def test_on_train_begin_receives_the_problem_and_model(self, problem, model):
        seen: dict[str, object] = {}

        class Capture(TrainerCallback):
            def on_train_begin(self, p, m) -> None:
                seen["problem"] = p
                seen["model"] = m

        Trainer(_config(adam=1)).train(problem, model, callbacks=[Capture()])
        assert seen["problem"] is problem
        assert isinstance(seen["model"], nn.Module)

    def test_on_train_end_receives_the_final_result(self, problem, model):
        seen: dict[str, object] = {}

        class Capture(TrainerCallback):
            def on_train_end(self, result) -> None:
                seen["result"] = result

        result = Trainer(_config(adam=2)).train(problem, model, callbacks=[Capture()])
        assert seen["result"] is result


class TestLoggingCallback:
    def test_logs_without_error(self, problem, model):
        result = Trainer(_config(adam=2, lbfgs=1)).train(
            problem, model, callbacks=[LoggingCallback()]
        )
        assert result.history

    def test_rejects_an_unknown_level(self):
        with pytest.raises(ValueError, match="unknown log level"):
            LoggingCallback(level="not_a_level")

    def test_accepts_a_valid_level(self, problem, model):
        Trainer(_config(adam=1)).train(
            problem, model, callbacks=[LoggingCallback(level="debug")]
        )


class TestRecordedStateContent:
    def test_state_carries_losses_and_weights(self, problem, model):
        config = TrainerConfig(
            collocation_count=32,
            log_every=1,
            loss_weights={"pde": 2.0},
            optimizer=OptimizerConfig(adam_steps=2),
        )
        state = Trainer(config).train(problem, model).final_state
        assert "pde" in state.named_losses
        assert state.weights["pde"] == 2.0
        assert state.elapsed_time >= 0.0

    def test_total_loss_is_the_weighted_sum_of_its_parts(self, problem, model):
        """The documented contract: total_loss = sum(weight * named_loss).

        Checked within a single run. Comparing two runs would not work: the
        trainer seeds the RNG inside ``train``, after the model was built, so
        two freshly constructed models start from different parameters and
        their losses are not comparable.
        """
        config = TrainerConfig(
            collocation_count=64,
            log_every=1,
            loss_weights={"pde": 3.0},
            optimizer=OptimizerConfig(adam_steps=0),
        )
        state = Trainer(config).train(problem, model).final_state

        expected = sum(
            state.weights[name] * value for name, value in state.named_losses.items()
        )
        assert state.total_loss == pytest.approx(expected, rel=1e-6)
        assert state.weights["pde"] == 3.0
        # Components with no configured weight default to one.
        assert all(w == 1.0 for n, w in state.weights.items() if n != "pde")

    def test_diagnostics_stay_out_of_named_losses(self, problem, model):
        from pinnlab.training import CausalWeightConfig

        config = TrainerConfig(
            collocation_count=64,
            log_every=1,
            causal=CausalWeightConfig(n_chunks=4, eps=1.0),
            optimizer=OptimizerConfig(adam_steps=2),
        )
        state = Trainer(config).train(problem, model).final_state
        assert "min_causal_weight" in state.diagnostics
        assert "min_causal_weight" not in state.named_losses

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_runs_in_either_supported_dtype(self, problem, dtype):
        config = TrainerConfig(
            collocation_count=32,
            log_every=1,
            dtype=dtype,
            optimizer=OptimizerConfig(adam_steps=2),
        )
        fresh = MLP(in_dim=2, hidden_layers=2, width=8, out_dim=1)
        result = Trainer(config).train(problem, fresh)
        assert result.history

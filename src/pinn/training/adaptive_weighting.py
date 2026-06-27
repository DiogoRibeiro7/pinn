"""Adaptive loss weighting strategies for PINN training.

This module implements several state-of-the-art techniques for balancing
multiple loss components during physics-informed neural network (PINN)
training.  The goal is to automatically adjust the contribution of each
objective (initial conditions, boundary conditions, PDE residual, etc.) so
that no single term dominates optimisation progress.

The implementation provides:

* Gradient-based weighting (GradNorm) and loss-magnitude balancing.
* Dynamic Weight Averaging (DWA) to react to changing loss decrease rates.
* Uncertainty-driven weighting using recent loss statistics.
* Multi-objective optimisation helpers (Pareto MTL and MGDA inspired).
* Advanced strategies powered by lightweight meta-learning, reinforcement
  learning style updates, and a simple Bayesian optimisation heuristic.
* Utility functions for computing gradient norms per loss component.
* Visualisation helpers for analysing weight evolution over time.

All strategies are designed to operate online during training and can be
configured through :class:`AdaptiveWeightingConfig`.  The configuration is
lightweight and validated to prevent misconfiguration during experiments.
"""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from ..utils.logging import get_logger
from ..utils.validation import ValidationError, check_range

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveWeightingConfig:
    """Configuration options for adaptive loss weighting.

    Parameters
    ----------
    enabled:
        Toggle for enabling adaptive weighting.
    method:
        Weighting strategy identifier (``"gradnorm"``, ``"dwa"``,
        ``"uncertainty"``, ``"pareto"``, ``"mgda"``, ``"meta"``,
        ``"reinforce"``, ``"bayes"`` or ``"magnitude"``).
    initial_weights:
        Optional initial weights per loss component.  When ``None`` the
        initial weights are uniform.
    adaptation_rate:
        Step size used by strategies that rely on gradient-based updates.
    adaptation_frequency:
        Number of optimisation steps between weight updates.
    history_window:
        Number of past steps retained for history-based strategies.
    uncertainty_window:
        Window size for variance-based (uncertainty) weighting.
    preview_collocation_points:
        Number of collocation points used to estimate PDE gradients for
        gradient-based strategies.  Set to ``0`` to disable previews.
    temperature:
        Softmax temperature parameter used by DWA and Bayesian updates.
    normalize:
        If ``True`` the weights are normalised to sum to one after each
        update.  Disable to keep absolute scaling but still clamp.
    clip_min, clip_max:
        Lower and upper bounds applied to weights to avoid extreme values.
    gradnorm_alpha:
        Exponent for the GradNorm relative loss term.  ``1.0`` corresponds to
        the formulation in the original paper.
    dwa_momentum:
        Momentum term used to smooth DWA updates.
    rl_discount, rl_learning_rate, rl_epsilon:
        Hyper-parameters controlling the reinforcement-learning-inspired
        weighting heuristic.
    bo_exploration:
        Exploration factor for the Bayesian weighting heuristic.  Higher
        values encourage sampling more diverse weights.
    visualize:
        Enable weight-history tracking suitable for visualisation.
    strict_validation:
        When ``True`` invalid configuration raises :class:`ValidationError`.
    log_history:
        Toggle for recording per-update history for later inspection.
    """

    enabled: bool = False
    method: str = "gradnorm"
    initial_weights: Optional[Sequence[float]] = None
    adaptation_rate: float = 0.1
    adaptation_frequency: int = 50
    history_window: int = 20
    uncertainty_window: int = 10
    preview_collocation_points: int = 256
    temperature: float = 2.0
    normalize: bool = True
    clip_min: float = 1e-3
    clip_max: float = 10.0
    gradnorm_alpha: float = 1.0
    dwa_momentum: float = 0.9
    rl_discount: float = 0.95
    rl_learning_rate: float = 0.2
    rl_epsilon: float = 0.1
    bo_exploration: float = 0.25
    visualize: bool = True
    strict_validation: bool = True
    log_history: bool = True

    def validate(self, num_losses: int) -> None:
        """Validate configuration values.

        Parameters
        ----------
        num_losses:
            Number of loss terms expected by the adaptive strategy.
        """

        check_range("adaptation_frequency", self.adaptation_frequency, min=1)
        check_range("adaptation_rate", self.adaptation_rate, min=0.0)
        check_range("history_window", self.history_window, min=1)
        check_range("uncertainty_window", self.uncertainty_window, min=1)
        check_range("temperature", self.temperature, min=1e-6)
        check_range("clip_min", self.clip_min, min=0.0)
        check_range("clip_max", self.clip_max, min=self.clip_min)
        check_range("gradnorm_alpha", self.gradnorm_alpha, min=0.0)
        check_range("rl_learning_rate", self.rl_learning_rate, min=0.0)
        check_range("rl_discount", self.rl_discount, min=0.0)
        check_range("bo_exploration", self.bo_exploration, min=1e-6)
        if self.initial_weights is not None and len(self.initial_weights) != num_losses:
            msg = (
                "initial_weights must match the number of losses; "
                f"expected {num_losses} values, received {len(self.initial_weights)}"
            )
            if self.strict_validation:
                raise ValidationError(msg)
            logger.warning(msg)


# ---------------------------------------------------------------------------
# Gradient utilities
# ---------------------------------------------------------------------------


def _parameters_list(parameters: Iterable[torch.nn.Parameter]) -> List[torch.nn.Parameter]:
    return [p for p in parameters if p.requires_grad]


def compute_component_gradients(
    losses: MutableMapping[str, Tensor],
    parameters: Iterable[torch.nn.Parameter],
    *,
    retain_graph: bool = False,
) -> Dict[str, List[Optional[Tensor]]]:
    """Return gradients of each loss component with respect to ``parameters``.

    The function uses :func:`torch.autograd.grad` and therefore does not mutate
    ``param.grad`` fields.  ``retain_graph`` should be ``True`` when the caller
    intends to perform a subsequent backward pass using the same computation
    graph.
    """

    params = _parameters_list(parameters)
    grads: Dict[str, List[Optional[Tensor]]] = {}
    loss_items = list(losses.items())
    for idx, (name, loss) in enumerate(loss_items):
        if not torch.is_tensor(loss):
            loss = torch.as_tensor(float(loss))
        grads[name] = list(
            torch.autograd.grad(
                loss,
                params,
                retain_graph=retain_graph or idx < len(loss_items) - 1,
                create_graph=False,
                allow_unused=True,
            )
        )
    return grads


def compute_gradient_norms(
    losses: MutableMapping[str, Tensor],
    parameters: Iterable[torch.nn.Parameter],
    *,
    retain_graph: bool = False,
) -> Dict[str, Tensor]:
    """Compute L2 gradient norms for each loss term."""

    grads = compute_component_gradients(losses, parameters, retain_graph=retain_graph)
    norms: Dict[str, Tensor] = {}
    for name, grad_list in grads.items():
        sq_norm = torch.zeros((), dtype=torch.float32, device=losses[name].device)
        for grad in grad_list:
            if grad is None:
                continue
            sq_norm = sq_norm + grad.float().pow(2).sum()
        norms[name] = torch.sqrt(sq_norm + 1e-12)
    return norms


# ---------------------------------------------------------------------------
# Adaptive weighting implementation
# ---------------------------------------------------------------------------


class AdaptiveLossWeighting:
    """Adaptive weighting engine for balancing PINN loss components."""

    def __init__(
        self,
        loss_names: Sequence[str],
        config: AdaptiveWeightingConfig,
        *,
        device: Optional[torch.device] = None,
        initial_weights: Optional[Sequence[float]] = None,
    ) -> None:
        if not loss_names:
            raise ValueError("loss_names must contain at least one entry")
        self.loss_names = tuple(loss_names)
        self.config = config
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.config.validate(len(self.loss_names))

        init = (
            list(initial_weights)
            if initial_weights is not None
            else (list(config.initial_weights) if config.initial_weights is not None else None)
        )
        if init is None:
            init = [1.0 / len(self.loss_names)] * len(self.loss_names)
        self.weights = torch.tensor(init, dtype=torch.float32, device=self.device)
        self.weights = self._post_process(self.weights)

        self.loss_history: Dict[str, List[float]] = defaultdict(list)
        self.weight_history: List[np.ndarray] = []
        self._last_update_step: int = -1
        self._initial_losses: Optional[Tensor] = None
        self._meta_network: Optional[nn.Module] = None
        self._meta_optim: Optional[torch.optim.Optimizer] = None
        self._rl_values = torch.zeros(len(self.loss_names), dtype=torch.float32, device=self.device)
        self._prev_total_loss: Optional[float] = None
        self._bo_precision = torch.full((len(self.loss_names),), 1e-3, dtype=torch.float32, device=self.device)
        self._bo_mean = torch.zeros(len(self.loss_names), dtype=torch.float32, device=self.device)
        self._multi_objective: Optional[MultiObjectiveOptimizer] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def requires_gradients(self) -> bool:
        method = self.config.method.lower()
        return method in {"gradnorm", "pareto", "pareto_mtl", "mgda", "meta"}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Return serialisable state for checkpointing."""

        return {
            "weights": self.weights.detach().cpu(),
            "loss_history": {k: list(v) for k, v in self.loss_history.items()},
            "weight_history": [torch.as_tensor(v) for v in self.weight_history],
            "prev_total": torch.tensor(self._prev_total_loss or 0.0),
            "bo_precision": self._bo_precision.detach().cpu(),
            "bo_mean": self._bo_mean.detach().cpu(),
            "rl_values": self._rl_values.detach().cpu(),
        }

    def load_state_dict(self, state: Dict[str, Tensor]) -> None:
        """Restore state saved via :meth:`state_dict`."""

        self.weights = state.get("weights", self.weights).to(self.device).float()
        history = state.get("loss_history", {})
        self.loss_history = defaultdict(list, {k: list(v) for k, v in history.items()})
        stored = state.get("weight_history")
        if stored is not None:
            self.weight_history = [np.asarray(t.cpu().numpy()) for t in stored]
        prev = state.get("prev_total")
        self._prev_total_loss = float(prev.item()) if prev is not None else None
        self._bo_precision = state.get("bo_precision", self._bo_precision).to(self.device)
        self._bo_mean = state.get("bo_mean", self._bo_mean).to(self.device)
        self._rl_values = state.get("rl_values", self._rl_values).to(self.device)

    def should_update(self, step: int) -> bool:
        return (step - self._last_update_step) >= self.config.adaptation_frequency

    def update_weights(
        self,
        losses: MutableMapping[str, Tensor],
        *,
        step: int,
        gradient_norms: Optional[MutableMapping[str, Tensor]] = None,
    ) -> Tensor:
        """Update loss weights using the configured strategy.

        Parameters
        ----------
        losses:
            Mapping of loss name to tensor (or scalar).  The dictionary should
            contain entries for each configured loss name.  A ``"total`` entry
            is optional but enables history-based heuristics.
        step:
            Current optimisation step.
        gradient_norms:
            Optional mapping of per-loss gradient norms.  Required for some
            strategies (GradNorm, Pareto/MGDA, meta-learning).
        """

        if not self.config.enabled:
            return self.weights

        self._record_losses(losses)
        if not self.should_update(step):
            return self.weights

        method = self.config.method.lower()
        formatted_grads = (
            {k: torch.as_tensor(v, device=self.device, dtype=torch.float32) for k, v in (gradient_norms or {}).items()}
            if gradient_norms is not None
            else None
        )

        if method in {"gradnorm", "grad_norm"}:
            updated = self._gradnorm(losses, formatted_grads)
        elif method == "dwa":
            updated = self._dwa()
        elif method == "uncertainty":
            updated = self._uncertainty()
        elif method in {"pareto", "pareto_mtl", "mgda"}:
            updated = self._multi_objective_update(method, losses, formatted_grads)
        elif method in {"meta", "meta_learning"}:
            updated = self._meta_learning_update(losses, formatted_grads)
        elif method in {"reinforce", "rl", "reinforcement"}:
            updated = self._reinforcement_update(losses)
        elif method in {"bayes", "bayesian"}:
            updated = self._bayesian_update(losses)
        else:  # magnitude / residual based fallback
            updated = self._magnitude_update(losses)

        self.weights = self._post_process(updated)
        if self.config.log_history and self.config.visualize:
            self.weight_history.append(self.weights.detach().cpu().numpy())
        self._last_update_step = step
        return self.weights

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _post_process(self, weights: Tensor) -> Tensor:
        weights = weights.to(self.device).float()
        weights = torch.clamp(weights, self.config.clip_min, self.config.clip_max)
        if self.config.normalize:
            weights = weights / (weights.sum() + 1e-8)
        return weights

    def _loss_tensor(self, losses: MutableMapping[str, Tensor], name: str) -> Tensor:
        value = losses.get(name)
        if value is None:
            return torch.tensor(0.0, device=self.device)
        if torch.is_tensor(value):
            return value.detach().to(self.device).float()
        return torch.tensor(float(value), device=self.device)

    def _record_losses(self, losses: MutableMapping[str, Tensor]) -> None:
        for name in list(losses.keys()) + ["total"]:
            if name not in losses:
                continue
            value = losses[name]
            scalar = float(value.detach().cpu().item()) if torch.is_tensor(value) else float(value)
            self.loss_history[name].append(scalar)
            window = self.loss_history[name]
            if len(window) > self.config.history_window:
                del window[0]

    def _current_losses_tensor(self, losses: MutableMapping[str, Tensor]) -> Tensor:
        return torch.stack([self._loss_tensor(losses, n) for n in self.loss_names])

    def _gradnorm(self, losses: MutableMapping[str, Tensor], gradient_norms: Optional[Dict[str, Tensor]]) -> Tensor:
        if gradient_norms is None:
            return self._magnitude_update(losses)

        grads = torch.stack([
            gradient_norms.get(name, torch.tensor(0.0, device=self.device)) for name in self.loss_names
        ])
        current_losses = self._current_losses_tensor(losses)
        if self._initial_losses is None:
            self._initial_losses = current_losses.clone()

        relative = current_losses / (self._initial_losses + 1e-8)
        mean_grad = grads.mean()
        target = mean_grad * torch.pow(relative, self.config.gradnorm_alpha)
        weighted = self.weights * grads
        diff = weighted - target
        updated = self.weights - self.config.adaptation_rate * diff
        return updated

    def _dwa(self) -> Tensor:
        if any(len(self.loss_history[name]) < 2 for name in self.loss_names):
            return self.weights
        prev = torch.tensor(
            [self.loss_history[name][-2] for name in self.loss_names], device=self.device, dtype=torch.float32
        )
        current = torch.tensor(
            [self.loss_history[name][-1] for name in self.loss_names], device=self.device, dtype=torch.float32
        )
        ratios = prev / (current + 1e-8)
        raw = torch.softmax(ratios / self.config.temperature, dim=0)
        return self.config.dwa_momentum * self.weights + (1 - self.config.dwa_momentum) * raw

    def _uncertainty(self) -> Tensor:
        variances = []
        for name in self.loss_names:
            history = self.loss_history[name]
            if len(history) < min(self.config.uncertainty_window, 2):
                variances.append(1.0)
            else:
                window = history[-self.config.uncertainty_window :]
                variances.append(float(np.var(window)))
        vals = torch.tensor(variances, device=self.device, dtype=torch.float32)
        weights = vals / (vals.sum() + 1e-8)
        return 0.5 * self.weights + 0.5 * weights

    def _multi_objective_update(
        self,
        method: str,
        losses: MutableMapping[str, Tensor],
        gradient_norms: Optional[Dict[str, Tensor]],
    ) -> Tensor:
        if gradient_norms is None:
            return self._magnitude_update(losses)
        if self._multi_objective is None:
            self._multi_objective = MultiObjectiveOptimizer(list(self.loss_names), method=method)
        grads = torch.stack([
            gradient_norms.get(name, torch.tensor(0.0, device=self.device)) for name in self.loss_names
        ])
        losses_tensor = self._current_losses_tensor(losses)
        return self._multi_objective.compute_weights(losses_tensor, grads).to(self.device)

    def _meta_learning_update(
        self,
        losses: MutableMapping[str, Tensor],
        gradient_norms: Optional[Dict[str, Tensor]],
    ) -> Tensor:
        features: List[Tensor] = []
        current_losses = self._current_losses_tensor(losses)
        features.append(current_losses)
        if gradient_norms is not None:
            grads = torch.stack([
                gradient_norms.get(name, torch.tensor(0.0, device=self.device)) for name in self.loss_names
            ])
            features.append(grads)
        feature_vec = torch.cat(features)
        if self._meta_network is None:
            hidden = max(self.config.meta_hidden, len(feature_vec))
            self._meta_network = nn.Sequential(
                nn.Linear(len(feature_vec), hidden),
                nn.ReLU(),
                nn.Linear(hidden, len(self.loss_names)),
            ).to(self.device)
            self._meta_optim = torch.optim.Adam(self._meta_network.parameters(), lr=1e-3)

        assert self._meta_network is not None and self._meta_optim is not None
        logits = self._meta_network(feature_vec)
        weights = torch.softmax(logits, dim=0)

        if gradient_norms is not None:
            grads = torch.stack([
                gradient_norms.get(name, torch.tensor(0.0, device=self.device)) for name in self.loss_names
            ])
            target = grads.mean()
            meta_loss = torch.mean((weights * grads - target).pow(2))
        else:
            target = current_losses.mean()
            meta_loss = torch.mean((weights * current_losses - target).pow(2))

        self._meta_optim.zero_grad()
        meta_loss.backward()
        self._meta_optim.step()
        return weights.detach()

    def _reinforcement_update(self, losses: MutableMapping[str, Tensor]) -> Tensor:
        total = float(self._loss_tensor(losses, "total").item())
        reward = 0.0
        if self._prev_total_loss is not None:
            reward = self._prev_total_loss - total
        self._prev_total_loss = total
        self._rl_values = (1 - self.config.rl_learning_rate) * self._rl_values + self.config.rl_learning_rate * (
            reward + self.config.rl_discount * self._rl_values
        )
        if np.random.rand() < self.config.rl_epsilon:
            noise = torch.rand_like(self._rl_values)
            return torch.softmax(noise, dim=0)
        return torch.softmax(self._rl_values, dim=0)

    def _bayesian_update(self, losses: MutableMapping[str, Tensor]) -> Tensor:
        total = self._loss_tensor(losses, "total")
        reward = 0.0
        if self._prev_total_loss is not None:
            reward = self._prev_total_loss - float(total.item())
        self._prev_total_loss = float(total.item())

        self._bo_precision = self._bo_precision + self.weights.pow(2)
        self._bo_mean = (
            self._bo_mean * (self._bo_precision - self.weights.pow(2)) + reward * self.weights
        ) / (self._bo_precision + 1e-8)

        exploration = torch.normal(
            self._bo_mean,
            torch.sqrt(1.0 / (self._bo_precision + 1e-8)),
        )
        exploration = exploration / self.config.bo_exploration
        return torch.softmax(exploration, dim=0)

    def _magnitude_update(self, losses: MutableMapping[str, Tensor]) -> Tensor:
        losses_tensor = self._current_losses_tensor(losses)
        inv = 1.0 / (losses_tensor + 1e-8)
        return inv / inv.sum()


# ---------------------------------------------------------------------------
# Multi-objective helper
# ---------------------------------------------------------------------------


class MultiObjectiveOptimizer:
    """Simplified helper for Pareto-based multi-objective weighting."""

    def __init__(self, objectives: Sequence[str], method: str = "pareto") -> None:
        self.objectives = tuple(objectives)
        self.method = method

    def compute_weights(self, losses: Tensor, gradient_norms: Tensor) -> Tensor:
        if losses.numel() == 0:
            raise ValueError("losses tensor cannot be empty")
        if self.method in {"pareto", "pareto_mtl"}:
            return self._pareto_weights(gradient_norms)
        if self.method == "mgda":
            return self._mgda_weights(gradient_norms)
        return torch.ones_like(losses) / float(losses.numel())

    def _pareto_weights(self, gradient_norms: Tensor) -> Tensor:
        if gradient_norms.sum() == 0:
            return torch.ones_like(gradient_norms) / float(gradient_norms.numel())
        inv = 1.0 / (gradient_norms + 1e-8)
        return inv / inv.sum()

    def _mgda_weights(self, gradient_norms: Tensor) -> Tensor:
        weights = torch.ones_like(gradient_norms) / float(gradient_norms.numel())
        for _ in range(5):
            grad = gradient_norms - gradient_norms.mean()
            weights = weights - 0.5 * grad
            weights = torch.clamp(weights, min=0.0)
            if weights.sum() <= 0:
                weights = torch.ones_like(weights)
            weights = weights / weights.sum()
        return weights


# ---------------------------------------------------------------------------
# Visualisation & analysis utilities
# ---------------------------------------------------------------------------


class WeightEvolutionVisualizer:
    """Create visual summaries of adaptive weight evolution."""

    def __init__(self, loss_names: Sequence[str]) -> None:
        self.loss_names = tuple(loss_names)

    def create_payload(
        self,
        weight_history: Sequence[Sequence[float]],
        total_history: Optional[Sequence[float]] = None,
    ) -> Optional[Dict[str, List[float]]]:
        if not weight_history:
            return None
        steps = list(range(len(weight_history)))
        weights = {
            name: [float(history[idx]) for history in weight_history]
            for idx, name in enumerate(self.loss_names)
        }
        payload = {"steps": steps, "weights": weights}
        if total_history is not None:
            payload["total_loss"] = list(map(float, total_history))
        return payload

    def create_plot(
        self,
        weight_history: Sequence[Sequence[float]],
        total_history: Optional[Sequence[float]] = None,
    ) -> Optional[object]:
        payload = self.create_payload(weight_history, total_history)
        if payload is None:
            return None
        try:
            import plotly.graph_objects as go  # type: ignore

            fig = go.Figure()
            for name in self.loss_names:
                fig.add_trace(
                    go.Scatter(
                        x=payload["steps"],
                        y=payload["weights"][name],
                        mode="lines",
                        name=f"w_{name}",
                    )
                )
            if total_history is not None:
                fig.add_trace(
                    go.Scatter(
                        x=payload["steps"],
                        y=payload["total_loss"],
                        mode="lines",
                        name="total_loss",
                        yaxis="y2",
                    )
                )
                fig.update_layout(
                    yaxis2=dict(title="Total Loss", overlaying="y", side="right"),
                )
            fig.update_layout(
                title="Adaptive Loss Weights",
                xaxis_title="Update",
                yaxis_title="Weight",
            )
            return fig
        except Exception:  # pragma: no cover - optional dependency branch
            return payload


class LossBalancingAnalyzer:
    """Compute statistics describing adaptive loss balancing."""

    def __init__(self, engine: AdaptiveLossWeighting) -> None:
        self.engine = engine

    def summary(self) -> Dict[str, float]:
        if not self.engine.weight_history:
            return {"updates": 0.0}
        weights = np.asarray(self.engine.weight_history)
        imbalance = float(np.mean(weights.max(axis=1) / (weights.min(axis=1) + 1e-8)))
        variance = float(np.mean(np.var(weights, axis=0)))
        totals = self.engine.loss_history.get("total", [])
        total_delta = 0.0
        if len(totals) >= 2:
            total_delta = totals[-1] - totals[0]
        return {
            "updates": float(len(weights)),
            "mean_weight": float(np.mean(weights)),
            "weight_std": float(np.std(weights)),
            "imbalance": imbalance,
            "weight_variance": variance,
            "total_delta": float(total_delta),
        }


# ---------------------------------------------------------------------------
# Convenience trainer
# ---------------------------------------------------------------------------


class AdaptivePINNTrainer:
    """Minimal trainer showcasing adaptive weighting with generic models."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        weighting: AdaptiveLossWeighting,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.weighting = weighting
        self.history: List[Dict[str, float]] = []

    def train_step(
        self,
        losses: MutableMapping[str, Tensor],
        *,
        step: int,
        parameters: Optional[Iterable[torch.nn.Parameter]] = None,
    ) -> Tensor:
        if parameters is None:
            parameters = self.model.parameters()
        grads = None
        if self.weighting.requires_gradients:
            grads = compute_gradient_norms(losses, parameters, retain_graph=True)
        weights = self.weighting.update_weights(losses, step=step, gradient_norms=grads)
        total = sum(weights[idx] * losses[name] for idx, name in enumerate(self.weighting.loss_names))
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        self.optimizer.step()
        record = {name: float(losses[name].detach()) for name in losses}
        record["total"] = float(total.detach())
        record.update({f"w_{n}": float(weights[idx]) for idx, n in enumerate(self.weighting.loss_names)})
        self.history.append(record)
        return total


__all__ = [
    "AdaptiveWeightingConfig",
    "AdaptiveLossWeighting",
    "AdaptivePINNTrainer",
    "compute_component_gradients",
    "compute_gradient_norms",
    "WeightEvolutionVisualizer",
    "LossBalancingAnalyzer",
    "MultiObjectiveOptimizer",
]


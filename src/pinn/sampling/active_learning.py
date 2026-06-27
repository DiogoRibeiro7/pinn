"""Active learning strategies for adaptive PINN sampling.

This module provides a lightweight yet expressive toolkit for performing
active learning in physics-informed neural networks.  The implementation is
designed to integrate with the rest of the repository while remaining fast
enough for the unit-test environment.  The key components are:

* :class:`ActiveLearningStrategy` – generic strategy that ranks candidate
  points using an acquisition function and optionally enforces diversity when
  selecting batches of points.
* Acquisition functions covering common criteria such as uncertainty sampling,
  expected improvement, upper confidence bounds, information gain, and
  query-by-committee disagreement.
* High level helpers (:class:`ActivePINNTrainer` and
  :class:`ActiveLearningConfig`) that wire the strategies into the existing
  PINN training loops.

Although the algorithms are simplified, they implement the essential
behaviour required for practical active learning workflows: online addition of
informative collocation points, compatibility with the uncertainty
quantification helpers, and support for different data acquisition modes (pool
based, stream based, and multi-objective optimisation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Optional, Sequence

import torch
from torch import Tensor, nn

from pinn.uncertainty import deep_ensemble_predict, mc_dropout_predict
from pinn.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _as_tensor(points: Tensor | torch.Tensor | Iterable[float]) -> Tensor:
    """Convert arbitrary input into a floating point tensor."""

    if torch.is_tensor(points):
        return points.float()
    return torch.as_tensor(points, dtype=torch.float32)


def _flatten_scores(scores: Tensor) -> Tensor:
    """Reduce multi-dimensional scores to a single dimension."""

    if scores.ndim == 1:
        return scores
    return scores.view(scores.shape[0], -1).mean(dim=1)


def _normal_cdf(x: Tensor) -> Tensor:
    """Standard normal CDF using ``torch.erf`` for numerical stability."""

    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _ensure_std(std: Tensor, eps: float = 1e-12) -> Tensor:
    """Clamp standard deviations away from zero."""

    return torch.clamp(std, min=eps)


def _predict_mean_std(
    model: nn.Module | Sequence[nn.Module] | Callable[[Tensor], Tensor],
    points: Tensor,
    *,
    ensemble: Optional[Sequence[nn.Module]] = None,
    custom: Optional[Callable[[nn.Module, Tensor], tuple[Tensor, Tensor]]] = None,
    mc_passes: int = 16,
) -> tuple[Tensor, Tensor]:
    """Obtain predictive mean and standard deviation for ``points``."""

    if custom is not None:
        mean, std = custom(model, points)
        return mean, std

    if isinstance(model, Sequence):
        mean, std = deep_ensemble_predict(model, points)
        return mean, std

    if ensemble is not None:
        mean, std = deep_ensemble_predict(ensemble, points)
        return mean, std

    if hasattr(model, "predict_with_uncertainty"):
        mean, std = model.predict_with_uncertainty(points)
        return mean, std

    mean, std = mc_dropout_predict(model, points, passes=mc_passes)
    return mean, std


# ---------------------------------------------------------------------------
# Acquisition functions
# ---------------------------------------------------------------------------


class UncertaintyAcquisition:
    """Score candidates by their predictive uncertainty."""

    def __init__(
        self,
        *,
        mode: str = "variance",
        mc_passes: int = 16,
        ensemble: Optional[Sequence[nn.Module]] = None,
        custom_predict: Optional[Callable[[nn.Module, Tensor], tuple[Tensor, Tensor]]] = None,
    ) -> None:
        self.mode = mode
        self.mc_passes = mc_passes
        self.ensemble = ensemble
        self.custom_predict = custom_predict

    def __call__(self, points: Tensor, *, model: nn.Module, **_: dict) -> Tensor:
        mean, std = _predict_mean_std(
            model,
            points,
            ensemble=self.ensemble,
            custom=self.custom_predict,
            mc_passes=self.mc_passes,
        )
        std = std.abs()
        if self.mode == "std":
            score = std
        elif self.mode == "variance":
            score = std ** 2
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported uncertainty mode '{self.mode}'")
        return _flatten_scores(score)


class ExpectedImprovementAcquisition:
    """Classical expected improvement for Bayesian optimisation."""

    def __init__(
        self,
        best_value: float,
        *,
        xi: float = 0.01,
        maximize: bool = False,
        mc_passes: int = 16,
        custom_predict: Optional[Callable[[nn.Module, Tensor], tuple[Tensor, Tensor]]] = None,
    ) -> None:
        self.best_value = best_value
        self.xi = xi
        self.maximize = maximize
        self.mc_passes = mc_passes
        self.custom_predict = custom_predict

    def __call__(self, points: Tensor, *, model: nn.Module, **_: dict) -> Tensor:
        mean, std = _predict_mean_std(model, points, custom=self.custom_predict, mc_passes=self.mc_passes)
        std = _ensure_std(std)
        if self.maximize:
            improvement = mean - self.best_value - self.xi
        else:
            improvement = self.best_value - mean - self.xi
        z = improvement / std
        pdf = torch.exp(-0.5 * z ** 2) / math.sqrt(2.0 * math.pi)
        cdf = _normal_cdf(z)
        ei = torch.clamp(improvement, min=0.0) * cdf + std * pdf
        return _flatten_scores(ei)


class UpperConfidenceBoundAcquisition:
    """Upper confidence bound balancing exploration and exploitation."""

    def __init__(
        self,
        *,
        beta: float = 2.0,
        maximize: bool = False,
        mc_passes: int = 16,
        custom_predict: Optional[Callable[[nn.Module, Tensor], tuple[Tensor, Tensor]]] = None,
    ) -> None:
        self.beta = beta
        self.maximize = maximize
        self.mc_passes = mc_passes
        self.custom_predict = custom_predict

    def __call__(self, points: Tensor, *, model: nn.Module, **_: dict) -> Tensor:
        mean, std = _predict_mean_std(model, points, custom=self.custom_predict, mc_passes=self.mc_passes)
        if self.maximize:
            score = mean + self.beta * std
        else:
            score = -(mean - self.beta * std)
        return _flatten_scores(score)


class InformationGainAcquisition:
    """Approximate mutual information using predictive variance."""

    def __init__(self, *, noise: float = 1e-6, mc_passes: int = 16) -> None:
        self.noise = noise
        self.mc_passes = mc_passes

    def __call__(self, points: Tensor, *, model: nn.Module, **_: dict) -> Tensor:
        _, std = _predict_mean_std(model, points, mc_passes=self.mc_passes)
        var = std ** 2
        score = 0.5 * torch.log1p(var / (self.noise ** 2))
        return _flatten_scores(score)


class QueryByCommitteeAcquisition:
    """Use an ensemble of models to measure prediction disagreement."""

    def __init__(self, committee: Optional[Sequence[nn.Module]] = None) -> None:
        self.committee = committee

    def __call__(self, points: Tensor, *, model: nn.Module | Sequence[nn.Module], **_: dict) -> Tensor:
        committee = self.committee or (model if isinstance(model, Sequence) else (model,))
        _, std = deep_ensemble_predict(committee, points)
        return _flatten_scores(std)


# ---------------------------------------------------------------------------
# Active learning strategies
# ---------------------------------------------------------------------------


class ActiveLearningStrategy:
    """Base strategy for selecting informative points."""

    def __init__(
        self,
        acquisition_function: Callable[..., Tensor],
        *,
        batch_size: int = 1,
        diversity_strength: float = 0.2,
    ) -> None:
        self.acquisition_function = acquisition_function
        self.batch_size = max(1, int(batch_size))
        self.diversity_strength = float(diversity_strength)
        self.last_scores: Optional[Tensor] = None

    # ------------------------------------------------------------------
    def score_candidates(self, candidate_points: Tensor, *, model: nn.Module, **kwargs: dict) -> Tensor:
        points = _as_tensor(candidate_points)
        with torch.no_grad():
            scores = self.acquisition_function(points, model=model, **kwargs)
        scores = _flatten_scores(scores)
        self.last_scores = scores.detach()
        return self.last_scores

    # ------------------------------------------------------------------
    def _diverse_batch_selection(self, points: Tensor, scores: Tensor, n_select: int) -> Tensor:
        n_select = min(n_select, points.shape[0])
        if n_select <= 0:
            return points.new_empty((0, points.shape[1]))

        remaining_scores = scores.clone()
        selected_indices: list[int] = []
        for _ in range(n_select):
            idx = torch.argmax(remaining_scores)
            idx_int = int(idx.item())
            if remaining_scores[idx_int] == -float("inf"):
                break
            selected_indices.append(idx_int)
            remaining_scores[idx_int] = -float("inf")

            if self.diversity_strength <= 0:
                continue

            anchor = points[idx_int : idx_int + 1]
            distances = torch.linalg.norm(points - anchor, dim=1)
            length_scale = torch.clamp(distances.std(), min=1e-6)
            penalty = torch.exp(-distances / length_scale)
            remaining_scores -= self.diversity_strength * penalty

        return points[selected_indices]

    # ------------------------------------------------------------------
    def select_points(
        self,
        candidate_points: Tensor,
        model: nn.Module,
        n_select: int,
        **kwargs: dict,
    ) -> Tensor:
        points = _as_tensor(candidate_points)
        scores = self.score_candidates(points, model=model, **kwargs)
        k = min(n_select, points.shape[0])
        if self.batch_size == 1 or k <= 1:
            topk = torch.topk(scores, k=k).indices
            return points[topk]
        return self._diverse_batch_selection(points, scores, k)


class PoolBasedActiveLearning(ActiveLearningStrategy):
    """Strategy that selects from a static candidate pool."""

    def refresh_pool(self, new_pool: Tensor) -> None:
        self.last_pool = _as_tensor(new_pool)


class StreamActiveLearning(ActiveLearningStrategy):
    """Stream-based active learning for sequential data sources."""

    def __init__(
        self,
        acquisition_function: Callable[..., Tensor],
        *,
        threshold: float = 0.0,
        batch_size: int = 1,
    ) -> None:
        super().__init__(acquisition_function, batch_size=batch_size)
        self.threshold = threshold

    def select_from_stream(self, stream: Iterable[Tensor], model: nn.Module, n_select: int) -> Tensor:
        selected: list[Tensor] = []
        for chunk in stream:
            points = _as_tensor(chunk)
            scores = self.score_candidates(points, model=model)
            mask = scores > self.threshold
            if mask.any():
                selected.append(points[mask])
            if sum(item.shape[0] for item in selected) >= n_select:
                break
        if not selected:
            return torch.empty((0, 0))
        concatenated = torch.cat(selected, dim=0)
        return self.select_points(concatenated, model, n_select)


class MultiObjectiveActiveLearning(ActiveLearningStrategy):
    """Combine multiple acquisition functions with cost-aware weighting."""

    def __init__(
        self,
        objectives: Sequence[Callable[..., Tensor]],
        *,
        weights: Optional[Sequence[float]] = None,
        cost_fn: Optional[Callable[[Tensor], Tensor]] = None,
        cost_tradeoff: float = 0.0,
        batch_size: int = 1,
    ) -> None:
        if not objectives:
            raise ValueError("At least one acquisition objective is required")
        self.objectives = list(objectives)
        self.weights = list(weights) if weights is not None else [1.0] * len(objectives)
        self.cost_fn = cost_fn
        self.cost_tradeoff = cost_tradeoff

        def combined(points: Tensor, *, model: nn.Module, **kwargs: dict) -> Tensor:
            scores = []
            for w, obj in zip(self.weights, self.objectives):
                scores.append(w * obj(points, model=model, **kwargs))
            total = sum(scores)
            if self.cost_fn is not None:
                total = total - cost_tradeoff * _flatten_scores(self.cost_fn(points))
            return total

        super().__init__(combined, batch_size=batch_size)


# ---------------------------------------------------------------------------
# Integration helpers for PINN training
# ---------------------------------------------------------------------------


@dataclass
class ActiveLearningConfig:
    """Configuration for :class:`ActivePINNTrainer`."""

    max_active_iterations: int = 5
    points_per_iteration: int = 64
    candidate_pool_size: int = 512
    update_frequency: int = 500
    convergence_tol: float = 1e-4
    patience: int = 3
    max_points: Optional[int] = None
    candidate_sampler: Optional[Callable[[int], Tensor]] = None


class ActivePINNTrainer:
    """Wrap a PINN solver to run with active learning configuration."""

    def __init__(
        self,
        pinn: nn.Module,
        active_strategy: ActiveLearningStrategy,
    ) -> None:
        self.pinn = pinn
        self.active_strategy = active_strategy

    def train_with_active_learning(
        self,
        train_config,
        config: ActiveLearningConfig,
    ):
        """Invoke the underlying ``train`` method with active learning enabled."""

        if not hasattr(train_config, "__dataclass_fields__"):
            raise TypeError("train_config must be a dataclass-based configuration")

        updated = replace(
            train_config,
            active_strategy=self.active_strategy,
            active_points_per_iteration=config.points_per_iteration,
            active_candidate_pool=config.candidate_pool_size,
            active_update_interval=config.update_frequency,
            active_convergence_tol=config.convergence_tol,
            active_patience=config.patience,
            active_max_points=config.max_points,
            active_max_iterations=config.max_active_iterations,
        )
        if config.candidate_sampler is not None:
            updated = replace(updated, active_candidate_sampler=config.candidate_sampler)

        logger.info(
            "Starting active learning training",
            extra={
                "iterations": config.max_active_iterations,
                "points_per_iteration": config.points_per_iteration,
            },
        )
        return self.pinn.train(updated)


__all__ = [
    "ActiveLearningStrategy",
    "PoolBasedActiveLearning",
    "StreamActiveLearning",
    "MultiObjectiveActiveLearning",
    "UncertaintyAcquisition",
    "ExpectedImprovementAcquisition",
    "UpperConfidenceBoundAcquisition",
    "InformationGainAcquisition",
    "QueryByCommitteeAcquisition",
    "ActivePINNTrainer",
    "ActiveLearningConfig",
]


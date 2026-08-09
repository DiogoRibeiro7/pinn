"""Residual-based adaptive refinement for composable PINN training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping

import torch
from torch import Tensor, nn

from ..residuals import DerivativeBackend, StrongFormResidual

if TYPE_CHECKING:
    from ..problems import PDEProblem


@dataclass(frozen=True)
class ResidualAdaptiveConfig:
    """Configuration for residual-based adaptive refinement.

    Args:
        candidate_count: Number of fresh interior candidates scored during each
            refinement pass.
        points_per_refinement: Number of high-residual candidates retained per
            refinement pass.
        refresh_every: Optimizer-step interval between refinement passes.
        start_step: First Adam step at which refinement may run.
        max_refined_points: Maximum retained adaptive collocation points.
        diversity_weight: Non-negative weight for normalized coordinate
            separation during candidate selection. ``0.0`` preserves strict
            residual top-k selection.
    """

    candidate_count: int = 512
    points_per_refinement: int = 64
    refresh_every: int = 100
    start_step: int = 1
    max_refined_points: int = 512
    diversity_weight: float = 0.0

    def validate(self) -> None:
        """Raise ``ValueError`` when refinement settings are invalid."""
        if self.candidate_count < 1:
            raise ValueError("candidate_count must be >= 1")
        if self.points_per_refinement < 1:
            raise ValueError("points_per_refinement must be >= 1")
        if self.points_per_refinement > self.candidate_count:
            raise ValueError("points_per_refinement cannot exceed candidate_count")
        if self.refresh_every < 1:
            raise ValueError("refresh_every must be >= 1")
        if self.start_step < 1:
            raise ValueError("start_step must be >= 1")
        if self.max_refined_points < self.points_per_refinement:
            raise ValueError("max_refined_points must be >= points_per_refinement")
        if self.diversity_weight < 0.0:
            raise ValueError("diversity_weight must be non-negative")


@dataclass
class ResidualAdaptiveRefiner:
    """Maintain high-residual collocation points for a trainer instance."""

    config: ResidualAdaptiveConfig
    _points: Tensor | None = field(default=None, init=False, repr=False)
    _scores: Tensor | None = field(default=None, init=False, repr=False)
    _last_diagnostics: dict[str, float] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Validate the refinement configuration."""
        self.config.validate()

    @property
    def num_points(self) -> int:
        """Return the number of retained adaptive points."""
        return 0 if self._points is None else int(self._points.shape[0])

    def should_refine(self, step: int) -> bool:
        """Return whether a refinement pass should run at ``step``."""
        return (
            step >= self.config.start_step
            and (step - self.config.start_step) % self.config.refresh_every == 0
        )

    def current_points(
        self, *, device: torch.device | str, dtype: torch.dtype
    ) -> Tensor | None:
        """Return retained points moved to ``device`` and ``dtype``."""
        if self._points is None:
            return None
        return self._points.to(device=device, dtype=dtype)

    def diagnostics(self) -> Mapping[str, float]:
        """Return diagnostics for the most recent refinement state."""
        diagnostics = dict(self._last_diagnostics)
        diagnostics["rar_points"] = float(self.num_points)
        return diagnostics

    def refine(
        self,
        problem: PDEProblem,
        model: nn.Module,
        residual_form: StrongFormResidual,
        derivative_backend: DerivativeBackend,
        *,
        generator: torch.Generator,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Mapping[str, float]:
        """Score candidates and retain high-residual, optionally diverse points."""
        candidates = problem.geometry.sample_interior(
            self.config.candidate_count,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        candidates = candidates.detach().requires_grad_(True)
        residual = residual_form.evaluate(
            problem, model, candidates, derivative_backend
        )
        scores = residual.detach().abs().reshape(candidates.shape[0], -1).mean(dim=1)
        k = min(self.config.points_per_refinement, candidates.shape[0])
        selected_indices = self._select_candidates(
            problem, candidates.detach(), scores, k
        )
        selected_points = candidates.detach()[selected_indices]
        selected_scores = scores[selected_indices].detach()
        self._retain(selected_points, selected_scores)
        self._last_diagnostics = {
            "rar_new_points": float(k),
            "rar_candidate_mean_residual": float(scores.mean()),
            "rar_candidate_max_residual": float(scores.max()),
            "rar_selection_diversity_weight": float(self.config.diversity_weight),
            "rar_selected_min_distance": _minimum_pairwise_distance(selected_points),
        }
        return self.diagnostics()

    def _select_candidates(
        self, problem: PDEProblem, candidates: Tensor, scores: Tensor, k: int
    ) -> Tensor:
        """Select candidates by residual score plus optional coordinate spread."""
        if self.config.diversity_weight == 0.0:
            return torch.topk(scores, k=k).indices

        normalized = _normalize_points(problem, candidates)
        residual_score = _minmax_normalize(scores)
        selected = torch.empty(k, device=candidates.device, dtype=torch.long)
        is_selected = torch.zeros(candidates.shape[0], device=candidates.device).bool()

        nearest_distance = torch.zeros(
            candidates.shape[0], device=candidates.device, dtype=candidates.dtype
        )
        if self._points is not None and self._points.shape[0] > 0:
            retained = _normalize_points(
                problem,
                self._points.to(device=candidates.device, dtype=candidates.dtype),
            )
            nearest_distance = torch.cdist(normalized, retained).min(dim=1).values

        for slot in range(k):
            if slot == 0 and self._points is None:
                composite = residual_score
            else:
                composite = (
                    residual_score + self.config.diversity_weight * nearest_distance
                )
            composite = composite.masked_fill(is_selected, -torch.inf)
            chosen = torch.argmax(composite)
            selected[slot] = chosen
            is_selected[chosen] = True
            distances = torch.linalg.norm(normalized - normalized[chosen], dim=1)
            if slot == 0 and self._points is None:
                nearest_distance = distances
            else:
                nearest_distance = torch.minimum(nearest_distance, distances)

        return selected

    def _retain(self, points: Tensor, scores: Tensor) -> None:
        """Keep the highest-scoring retained adaptive points."""
        if self._points is not None and self._scores is not None:
            points = torch.cat([self._points.to(points), points], dim=0)
            scores = torch.cat([self._scores.to(scores), scores], dim=0)

        limit = min(self.config.max_refined_points, points.shape[0])
        keep = torch.topk(scores, k=limit).indices
        self._points = points.detach()[keep]
        self._scores = scores.detach()[keep]


def _normalize_points(problem: PDEProblem, points: Tensor) -> Tensor:
    """Scale coordinates to unit-box geometry coordinates."""
    lower = problem.geometry.lower_bounds.to(device=points.device, dtype=points.dtype)
    upper = problem.geometry.upper_bounds.to(device=points.device, dtype=points.dtype)
    return (points - lower) / (upper - lower)


def _minmax_normalize(values: Tensor) -> Tensor:
    """Normalize scores to ``[0, 1]`` while handling constant vectors."""
    span = values.max() - values.min()
    if span <= torch.finfo(values.dtype).eps:
        return torch.zeros_like(values)
    return (values - values.min()) / span


def _minimum_pairwise_distance(points: Tensor) -> float:
    """Return the minimum selected point distance, or zero for one point."""
    if points.shape[0] < 2:
        return 0.0
    distances = torch.pdist(points.detach())
    if distances.numel() == 0:
        return 0.0
    return float(distances.min())

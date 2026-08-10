"""Independent benchmark evaluators for composable PINN problems."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..problems import PDEProblem
from ..residuals import AutogradDerivativeBackend, StrongFormResidual


@dataclass(frozen=True)
class IndependentResidualMetrics:
    """Summary statistics for a residual evaluated on fresh collocation points."""

    mean_squared_residual: float
    max_absolute_residual: float
    mean_absolute_residual: float
    sample_count: int
    seed: int | None

    def to_dict(self) -> dict[str, float | int | None]:
        """Return a JSON-compatible representation."""
        return {
            "mean_squared_residual": self.mean_squared_residual,
            "max_absolute_residual": self.max_absolute_residual,
            "mean_absolute_residual": self.mean_absolute_residual,
            "sample_count": self.sample_count,
            "seed": self.seed,
        }


def _resolve_model_device(model: nn.Module) -> torch.device:
    try:
        parameter = next(model.parameters())
    except StopIteration:
        return torch.device("cpu")
    return parameter.device


def evaluate_independent_residual(
    problem: PDEProblem,
    model: nn.Module,
    *,
    sample_count: int = 4096,
    seed: int | None = 0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> IndependentResidualMetrics:
    """Evaluate a problem residual on fresh points not owned by the trainer.

    The function samples uniformly from ``problem.geometry`` and evaluates the
    strong-form residual with a new derivative backend. It intentionally does
    not reuse trainer state, collocation caches or adaptive-refinement points.
    """
    if sample_count < 1:
        raise ValueError(f"sample_count must be >= 1, got {sample_count}")

    resolved_device = (
        torch.device(device) if device is not None else _resolve_model_device(model)
    )
    generator = torch.Generator(device=resolved_device)
    if seed is not None:
        generator.manual_seed(seed)

    coordinates = problem.geometry.sample_interior(
        sample_count,
        generator=generator,
        device=resolved_device,
        dtype=dtype,
    )
    residual = StrongFormResidual().evaluate(
        problem, model, coordinates, AutogradDerivativeBackend()
    )
    detached = residual.detach().abs()
    return IndependentResidualMetrics(
        mean_squared_residual=float(torch.mean(residual.detach() ** 2).cpu().item()),
        max_absolute_residual=float(torch.max(detached).cpu().item()),
        mean_absolute_residual=float(torch.mean(detached).cpu().item()),
        sample_count=sample_count,
        seed=seed,
    )

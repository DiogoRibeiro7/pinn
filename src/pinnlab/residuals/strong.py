"""Strong-form PDE residual evaluation."""

from __future__ import annotations

from torch import Tensor, nn

from .base import DerivativeBackend


class StrongFormResidual:
    """Evaluate a problem's pointwise strong-form PDE residual."""

    def evaluate(
        self,
        problem: object,
        model: nn.Module,
        coordinates: Tensor,
        derivative_backend: DerivativeBackend,
    ) -> Tensor:
        """Delegate residual evaluation to ``problem.strong_residual``."""
        residual_fn = getattr(problem, "strong_residual", None)
        if residual_fn is None:
            raise TypeError("problem must define strong_residual")
        residual = residual_fn(model, coordinates, derivative_backend)
        if residual.ndim != 2:
            raise ValueError("strong residual must have shape (N, residual_dim)")
        return residual

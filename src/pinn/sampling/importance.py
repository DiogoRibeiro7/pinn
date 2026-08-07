"""Importance sampling strategies for PINN training.

This module implements adaptive importance samplers that bias the
collocation point distribution towards regions with large loss gradients or
residuals.  The main entry point is :class:`GradientBasedImportanceSampler`
which wraps an existing sampler (e.g. Latin Hypercube) and updates an
importance map during training.

The sampler computes gradients of a user provided loss function with respect
to the input coordinates and converts their magnitude to sampling weights.
Sampling then proceeds from the non-uniform distribution defined by those
weights.  The implementation is designed to be lightweight so that it can be
used inside unit tests and small examples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
from torch import Tensor

from pinn.utils.sampling import BaseSampler, Domain


@dataclass
class ImportanceState:
    """Container for the current importance map."""

    points: Tensor
    weights: Tensor


class GradientBasedImportanceSampler:
    """Adaptive sampler that biases points based on loss gradients.

    Parameters
    ----------
    domain:
        Sampling domain describing the bounds of each dimension.
    base_sampler:
        Sampler used to draw candidate points (e.g. ``LatinHypercubeSampler``).
    update_frequency:
        Number of ``sample`` calls between importance weight updates.
    n_candidates:
        Number of candidate points used to estimate the importance map.  The
        larger this value the smoother the distribution but also the more
        expensive the update.
    strategy:
        Importance metric to use.  ``"gradient"`` uses the gradient magnitude of
        the provided loss function w.r.t. the inputs.  ``"residual"`` expects the
        loss function to return residual values directly.  ``"error"`` uses the
        absolute error returned by the loss function.
    device:
        Optional torch device for gradient computations.  If ``None`` the device
        is inferred from the model when ``sample`` is called.
    """

    def __init__(
        self,
        domain: Domain,
        base_sampler: BaseSampler,
        *,
        update_frequency: int = 100,
        n_candidates: int = 1024,
        strategy: str = "gradient",
        device: Optional[torch.device | str] = None,
    ) -> None:
        self.domain = domain
        self.base_sampler = base_sampler
        self.update_frequency = int(update_frequency)
        self.n_candidates = int(n_candidates)
        self.strategy = strategy
        self.device = torch.device(device) if device is not None else None

        self._state: Optional[ImportanceState] = None
        self._steps: int = 0

    # ------------------------------------------------------------------
    # Importance weight computation
    # ------------------------------------------------------------------
    def _compute_importance(
        self,
        model: torch.nn.Module,
        loss_fn: Callable[[torch.nn.Module, Tensor], Tensor],
        points: Tensor,
    ) -> Tensor:
        """Compute unnormalised importance weights for ``points``."""
        if self.strategy == "gradient":
            points.requires_grad_(True)
            loss = loss_fn(model, points)
            grad = torch.autograd.grad(
                loss, points, create_graph=False, allow_unused=True
            )[0]
            if grad is None:
                grad = torch.zeros_like(points)
            metric = torch.linalg.norm(grad, dim=1)
        elif self.strategy == "residual":
            residual = loss_fn(model, points)
            if residual.ndim > 1:
                residual = residual.view(residual.size(0), -1).norm(dim=1)
            metric = residual.abs()
        elif self.strategy == "error":
            err = loss_fn(model, points)
            if err.ndim > 1:
                err = err.view(err.size(0), -1).norm(dim=1)
            metric = err.abs()
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unknown strategy '{self.strategy}'")

        # Normalise to obtain a probability distribution
        metric = metric + 1e-12  # avoid zero division
        weights = metric / metric.sum()
        return weights.detach()

    def update_importance_weights(
        self,
        model: torch.nn.Module,
        loss_fn: Callable[[torch.nn.Module, Tensor], Tensor],
    ) -> None:
        """Refresh the importance map using the current model parameters."""
        device = self.device or next(model.parameters()).device
        candidates = torch.from_numpy(self.base_sampler.sample(self.n_candidates)).to(
            device
        )
        weights = self._compute_importance(model, loss_fn, candidates.clone())
        self._state = ImportanceState(points=candidates.detach(), weights=weights)
        self._steps = 0

    # ------------------------------------------------------------------
    # Sampling interface
    # ------------------------------------------------------------------
    def sample(
        self,
        n_points: int,
        *,
        model: Optional[torch.nn.Module] = None,
        loss_fn: Optional[Callable[[torch.nn.Module, Tensor], Tensor]] = None,
    ) -> Tensor:
        """Draw ``n_points`` samples from the current importance distribution.

        If called before any importance map has been computed or if the update
        frequency has been reached, the importance weights are recomputed using
        ``model`` and ``loss_fn``.
        """
        if self._state is None or self._steps >= self.update_frequency:
            if model is None or loss_fn is None:
                raise ValueError("Model and loss_fn must be provided on first call")
            self.update_importance_weights(model, loss_fn)
        assert self._state is not None
        idx = torch.multinomial(self._state.weights, n_points, replacement=True)
        self._steps += 1
        return self._state.points[idx]

    # ------------------------------------------------------------------
    # Convenience utilities
    # ------------------------------------------------------------------
    @property
    def importance_weights(self) -> Optional[Tensor]:
        return None if self._state is None else self._state.weights

    @property
    def last_points(self) -> Optional[Tensor]:
        return None if self._state is None else self._state.points

    def importance_map(self) -> Optional[tuple[Tensor, Tensor]]:
        """Return the last points and their associated weights."""
        if self._state is None:
            return None
        return self._state.points, self._state.weights

    def to(self, device: torch.device | str) -> "GradientBasedImportanceSampler":
        """Move internal tensors to ``device``."""
        self.device = torch.device(device)
        if self._state is not None:
            self._state = ImportanceState(
                points=self._state.points.to(self.device),
                weights=self._state.weights.to(self.device),
            )
        return self

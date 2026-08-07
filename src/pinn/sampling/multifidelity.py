"""Multi-fidelity sampling and hierarchical training utilities.

This module provides tools for sampling collocation points at multiple
fidelities and for training PINNs in a hierarchical fashion from coarse to
fine resolutions.  The implementation is intentionally lightweight so that it
can be executed inside the unit-test environment used in this kata.  The core
classes are:

* :class:`MultiFidelitySampler` – draws samples at user defined fidelity levels
  while accounting for their relative computational cost.
* :class:`HierarchicalPINNTrainer` – trains a list of models progressively
  across fidelities and transfers learned weights to finer levels.
* :class:`MultiFidelityGaussianProcess` – simple multi-fidelity GP surrogate
  used for uncertainty estimation and cokriging style information fusion.
* :class:`CoKrigingModel` – wrapper around ``MultiFidelityGaussianProcess`` for
  terminology clarity.

The algorithms here are simplified variants of their research-grade
counterparts but capture the essential ideas of hierarchical sampling and
knowledge transfer between fidelity levels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from pinn.utils.sampling import BaseSampler, Domain, LatinHypercubeSampler

# ---------------------------------------------------------------------------
# Multi-fidelity sampler
# ---------------------------------------------------------------------------


class MultiFidelitySampler:
    """Sampler capable of generating points at multiple fidelities.

    Parameters
    ----------
    domain:
        Sampling domain describing the bounds of each dimension.
    fidelity_levels:
        Sequence describing the resolution at each fidelity level.  Larger
        numbers correspond to finer resolutions.
    cost_ratios:
        Relative computational cost of drawing a single sample at each
        fidelity level.  Used by :meth:`hierarchical_sampling_schedule`.
    base_sampler:
        Sampler used to generate candidate points in the unit cube.  By default
        a :class:`LatinHypercubeSampler` is used.
    """

    def __init__(
        self,
        domain: Domain,
        fidelity_levels: Sequence[int],
        cost_ratios: Sequence[float],
        *,
        base_sampler: BaseSampler | None = None,
    ) -> None:
        self.domain = domain
        self.fidelity_levels = list(fidelity_levels)
        self.cost_ratios = list(cost_ratios)
        if len(self.fidelity_levels) != len(self.cost_ratios):
            raise ValueError("fidelity_levels and cost_ratios must have same length")
        self.base_sampler = base_sampler or LatinHypercubeSampler(domain)
        self.current_fidelity = 0

    # ------------------------------------------------------------------
    def _generate_samples(self, n_points: int, resolution: int) -> np.ndarray:
        """Generate ``n_points`` using a grid with the given ``resolution``.

        The underlying sampler draws random points which are then snapped to a
        grid of size ``resolution`` to emulate different fidelity levels.
        """

        raw = self.base_sampler.sample(n_points)
        snapped = np.zeros_like(raw)
        for i, (low, high) in enumerate(self.domain.bounds):
            span = high - low
            grid = span / resolution
            snapped[:, i] = np.round((raw[:, i] - low) / grid) * grid + low
            snapped[:, i] = np.clip(snapped[:, i], low, high)
        return snapped

    def sample_at_fidelity(self, n_points: int, fidelity_level: int) -> np.ndarray:
        """Sample ``n_points`` at the specified fidelity level."""

        resolution = self.fidelity_levels[fidelity_level]
        return self._generate_samples(n_points, resolution)

    # ------------------------------------------------------------------
    def hierarchical_sampling_schedule(
        self, total_budget: int
    ) -> List[Tuple[int, int]]:
        """Allocate a sampling budget across fidelity levels.

        The strategy follows a coarse-to-fine allocation where cheaper
        fidelities receive a larger fraction of the budget.
        """

        schedule: List[Tuple[int, int]] = []
        remaining_budget = float(total_budget)
        n_levels = len(self.fidelity_levels)
        for i, cost in enumerate(self.cost_ratios):
            if i < n_levels - 1:
                budget_fraction = 0.3 * (n_levels - i) / n_levels
            else:
                budget_fraction = remaining_budget / total_budget
            n_samples = int(total_budget * budget_fraction / cost)
            n_samples = max(n_samples, 1)
            schedule.append((i, n_samples))
            remaining_budget -= n_samples * cost
        return schedule

    # ------------------------------------------------------------------
    def adaptive_next_fidelity(
        self, metrics: Sequence[float], threshold: float = 0.05
    ) -> int:
        """Select the next fidelity level based on validation metrics.

        Parameters
        ----------
        metrics:
            Sequence of validation errors for each fidelity processed so far.
        threshold:
            If the latest metric is below this value, advance to the next
            fidelity; otherwise repeat current level.
        """

        if not metrics:
            return 0
        if metrics[-1] < threshold and len(metrics) < len(self.fidelity_levels) - 1:
            return len(metrics)
        return len(metrics) - 1


# ---------------------------------------------------------------------------
# Hierarchical training utilities
# ---------------------------------------------------------------------------


@dataclass
class HierarchicalConfig:
    total_budget: int
    epochs: int = 100
    validate_points: int = 100


class HierarchicalPINNTrainer:
    """Train multiple models in a coarse-to-fine hierarchy."""

    def __init__(
        self,
        models: Sequence[torch.nn.Module],
        multifidelity_sampler: MultiFidelitySampler,
        analytic_fn: Callable[[Tensor], Tensor],
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        self.models = list(models)
        self.sampler = multifidelity_sampler
        self.analytic_fn = analytic_fn
        self.device = torch.device(device)

    # ------------------------------------------------------------------
    def _train_at_fidelity(self, level: int, samples: np.ndarray, epochs: int) -> None:
        model = self.models[level].to(self.device)
        x = torch.from_numpy(samples).float().to(self.device)
        y = self.analytic_fn(x)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(epochs):
            opt.zero_grad()
            pred = model(x)
            loss = torch.mean((pred - y) ** 2)
            loss.backward()
            opt.step()

    def _transfer_knowledge(self, from_level: int, to_level: int) -> None:
        src = self.models[from_level].state_dict()
        self.models[to_level].load_state_dict(src, strict=False)

    def train_hierarchical(self, config: HierarchicalConfig) -> None:
        metrics: List[float] = []
        schedule = self.sampler.hierarchical_sampling_schedule(config.total_budget)
        for level, n_samples in schedule:
            samples = self.sampler.sample_at_fidelity(n_samples, level)
            self._train_at_fidelity(level, samples, config.epochs)
            metrics.append(self.validate(level, config.validate_points))
            next_level = self.sampler.adaptive_next_fidelity(metrics)
            if next_level > level and level < len(self.models) - 1:
                self._transfer_knowledge(level, level + 1)

    def validate(self, fidelity_level: int, n_points: int) -> float:
        samples = self.sampler.sample_at_fidelity(n_points, fidelity_level)
        x = torch.from_numpy(samples).float().to(self.device)
        y_true = self.analytic_fn(x)
        with torch.no_grad():
            y_pred = self.models[fidelity_level](x)
        return torch.mean((y_pred - y_true) ** 2).item()


# ---------------------------------------------------------------------------
# Simple multi-fidelity Gaussian process / Cokriging
# ---------------------------------------------------------------------------


class MultiFidelityGaussianProcess:
    """Very small multi-fidelity GP surrogate.

    This implementation captures the core idea of modelling the high-fidelity
    output ``y_H`` as ``rho * y_L + delta`` where ``delta`` is a Gaussian
    process fitted to the discrepancy.  For simplicity we model ``delta`` as a
    constant offset estimated from the training data.
    """

    def __init__(self) -> None:
        self.rho: float | None = None
        self.delta: float | None = None

    def fit(self, y_low: np.ndarray, y_high: np.ndarray) -> None:
        y_low = np.asarray(y_low).ravel()
        y_high = np.asarray(y_high).ravel()
        A = np.vstack([y_low, np.ones_like(y_low)]).T
        self.rho, self.delta = np.linalg.lstsq(A, y_high, rcond=None)[0]

    def predict(self, y_low_pred: np.ndarray) -> np.ndarray:
        if self.rho is None or self.delta is None:
            raise RuntimeError("Model must be fitted before prediction")
        return self.rho * np.asarray(y_low_pred) + self.delta


class CoKrigingModel(MultiFidelityGaussianProcess):
    """Alias of :class:`MultiFidelityGaussianProcess` for terminology clarity."""

    pass


__all__ = [
    "MultiFidelitySampler",
    "HierarchicalConfig",
    "HierarchicalPINNTrainer",
    "MultiFidelityGaussianProcess",
    "CoKrigingModel",
]

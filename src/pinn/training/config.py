"""Configuration objects for the generic PINN trainer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch

from .causal_weighting import CausalWeightConfig
from .adaptive_refinement import ResidualAdaptiveConfig
from ..scaling import Nondimensionalizer


def resolve_dtype(dtype: torch.dtype | str) -> torch.dtype:
    """Resolve a dtype string or ``torch.dtype`` into a supported dtype."""
    if isinstance(dtype, torch.dtype):
        resolved = dtype
    elif dtype == "float32":
        resolved = torch.float32
    elif dtype == "float64":
        resolved = torch.float64
    else:
        raise ValueError(
            "dtype must be torch.float32, torch.float64, 'float32' or 'float64'"
        )
    if resolved not in {torch.float32, torch.float64}:
        raise ValueError("generic Trainer supports torch.float32 and torch.float64")
    return resolved


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer settings for the generic trainer."""

    lr: float = 1e-3
    adam_steps: int = 1_000
    lbfgs_max_iter: int = 0

    def validate(self) -> None:
        """Raise ``ValueError`` when optimizer settings are invalid."""
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.adam_steps < 0:
            raise ValueError(f"adam_steps must be >= 0, got {self.adam_steps}")
        if self.lbfgs_max_iter < 0:
            raise ValueError(f"lbfgs_max_iter must be >= 0, got {self.lbfgs_max_iter}")


@dataclass(frozen=True)
class TrainerConfig:
    """Configuration for equation-agnostic PINN training.

    Args:
        seed: Random seed propagated to Python, NumPy and PyTorch.
        device: Torch device used for model and sampled tensors.
        dtype: Floating-point dtype for the new core path.
        collocation_count: Number of interior residual points.
        constraint_sample_counts: Optional per-constraint sample-count overrides.
        optimizer: Adam and optional L-BFGS settings.
        log_every: Record loss diagnostics every this many Adam steps.
        loss_weights: Optional per-loss weights keyed by ``"pde"`` or a
            constraint name.
        causal: Optional causal residual weighting configuration.
        scaling: Optional physical nondimensionalization transform. When set,
            the raw model is trained in dimensionless input/output variables
            while problems and constraints still see dimensional values.
        adaptive_refinement: Optional residual-based adaptive refinement
            configuration for adding high-residual collocation points.
    """

    seed: int = 0
    device: str | torch.device = "cpu"
    dtype: torch.dtype | str = torch.float32
    collocation_count: int = 1_000
    constraint_sample_counts: Mapping[str, int] = field(default_factory=dict)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    log_every: int = 100
    loss_weights: Mapping[str, float] = field(default_factory=dict)
    causal: CausalWeightConfig | None = None
    scaling: Nondimensionalizer | None = None
    adaptive_refinement: ResidualAdaptiveConfig | None = None

    def validate(self) -> None:
        """Raise ``ValueError`` when the configuration is invalid."""
        resolve_dtype(self.dtype)
        if self.collocation_count < 1:
            raise ValueError("collocation_count must be >= 1")
        if self.log_every < 1:
            raise ValueError("log_every must be >= 1")
        for name, count in self.constraint_sample_counts.items():
            if count < 1:
                raise ValueError(f"constraint sample count for {name} must be >= 1")
        for name, weight in self.loss_weights.items():
            if weight < 0:
                raise ValueError(f"loss weight for {name} must be non-negative")
        self.optimizer.validate()
        if self.causal is not None:
            self.causal.validate()
        if self.adaptive_refinement is not None:
            self.adaptive_refinement.validate()

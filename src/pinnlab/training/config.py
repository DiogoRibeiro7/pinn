"""Configuration objects for the generic PINN trainer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch

from .causal_weighting import CausalWeightConfig
from .adaptive_refinement import ResidualAdaptiveConfig
from ..sampling.core import InteriorSampler, UniformInteriorSampler
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

    This object owns *how* to train and nothing about *what* is being solved.
    The split across the three core types is:

    - ``TrainerConfig`` -- budget, precision, sampling policy, loss weights and
      optional strategies. Reusable across equations; changing it should never
      change the problem's meaning, only how well or how quickly it is solved.
    - :class:`~pinnlab.problems.PDEProblem` -- geometry, residual and
      constraints. Everything equation-specific lives there, so the trainer can
      stay ignorant of which PDE it is optimising.
    - :class:`~pinnlab.training.state.TrainingState` /
      :class:`~pinnlab.training.state.TrainingResult` -- what happened, as
      immutable records. Nothing here feeds back into the configuration.

    Two consequences worth knowing. Entries in ``loss_weights`` are matched by
    name against the components a problem produces (``"pde"`` plus one per
    constraint), so a weight for a name the problem never emits is silently
    unused rather than an error. And ``scaling`` wraps the model rather than the
    problem: the network trains in dimensionless variables while geometry,
    constraints and residuals continue to work in physical ones.

    Args:
        seed: Random seed propagated to Python, NumPy and PyTorch.
        device: Torch device used for model and sampled tensors.
        dtype: Floating-point dtype for the new core path.
        collocation_count: Number of interior residual points.
        sampler: Interior sampler used for trainer-owned residual points.
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
            configuration for adding high-residual collocation points. The
            refiner can optionally weight coordinate diversity during candidate
            selection.
    """

    seed: int = 0
    device: str | torch.device = "cpu"
    dtype: torch.dtype | str = torch.float32
    collocation_count: int = 1_000
    sampler: InteriorSampler = field(default_factory=UniformInteriorSampler)
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
        if not hasattr(self.sampler, "sample"):
            raise ValueError("sampler must define a sample method")
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

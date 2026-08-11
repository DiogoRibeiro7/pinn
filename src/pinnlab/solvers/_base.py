"""Shared machinery for solvers that have a closed-form solution to check against.

The solvers built on this base differ from the rest of the package in one
respect that matters: their accuracy is a measurable number rather than a
plot. Each subclass supplies an analytic solution, so
:meth:`ExactlySolvablePINN.relative_l2_error` reports how far the trained
network actually is from the truth, and the tests can assert on it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn

from ..models.mlp import MLP
from ..training.causal_weighting import CausalWeightConfig
from ..utils.logging import get_logger
from .raissi_generic import Domain1D, set_seed

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from ..training.config import TrainerConfig
    from ..training.state import TrainingState

logger = get_logger(__name__)

__all__ = ["Domain1D", "TrainConfig", "ExactlySolvablePINN", "gradient"]


def gradient(outputs: Tensor, inputs: Tensor) -> Tensor:
    """Differentiate ``outputs`` with respect to ``inputs``, keeping the graph.

    Retained for callers who imported it, though nothing in the package uses
    it now: the equations are differentiated through
    :class:`~pinnlab.residuals.AutogradDerivativeBackend`, which the problems
    use, so derivative handling lives in one place.

    Args:
        outputs: Tensor produced from ``inputs``.
        inputs: Tensor with ``requires_grad=True``.

    Returns:
        The derivative, shaped like ``inputs``.
    """
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
    )[0]


@dataclass(frozen=True)
class TrainConfig:
    """Training settings shared by the exactly-solvable solvers.

    Attributes:
        n_ic: Initial-condition points.
        n_bc: Boundary points per boundary.
        n_f: Interior collocation points.
        adam_steps: Adam iterations.
        lr: Adam learning rate.
        lbfgs_max_iter: L-BFGS iterations run after Adam. Zero disables it.
        weights: Multipliers for the (initial, boundary, residual) loss terms.
        seed: Seed for sampling and initialisation.
        log_every: Emit a log line every this many Adam steps.
        causal: Causal weighting settings for the residual term. ``None`` uses
            the plain mean squared residual.
    """

    n_ic: int = 256
    n_bc: int = 256
    n_f: int = 8_000
    adam_steps: int = 5_000
    lr: float = 1e-3
    lbfgs_max_iter: int = 0
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    seed: int = 0
    log_every: int = 1_000
    causal: Optional[CausalWeightConfig] = None

    def validate(self) -> None:
        """Raise :class:`ValueError` if the configuration is unusable."""
        for name in ("n_ic", "n_bc", "n_f"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.adam_steps < 0:
            raise ValueError(f"adam_steps must be >= 0, got {self.adam_steps}")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.lbfgs_max_iter < 0:
            raise ValueError(f"lbfgs_max_iter must be >= 0, got {self.lbfgs_max_iter}")
        if len(self.weights) != 3 or any(w < 0 for w in self.weights):
            raise ValueError(
                f"weights must be three non-negative numbers, got {self.weights}"
            )


class ExactlySolvablePINN(ABC):
    """Base for 1-D time-dependent PINNs with homogeneous Dirichlet boundaries.

    Subclasses supply the PDE residual, the initial condition and the analytic
    solution. Problems that are second order in time additionally supply an
    initial velocity, and the base adds the matching loss term.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        domain: Optional[Domain1D] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Initialise the solver.

        Args:
            model: Network mapping ``(t, x) -> u``. Defaults to a 4x64 tanh MLP.
            domain: Space-time rectangle. Defaults to the subclass's domain.
            device: Torch device. Defaults to CUDA when available.
        """
        self.domain = domain if domain is not None else self.default_domain()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if model is None:
            model = MLP(in_dim=2, hidden_layers=4, width=64, out_dim=1)
        self.model = model.to(self.device)
        self.loss_history: List[Dict[str, float]] = []

    # -- to be provided by subclasses -------------------------------------

    @staticmethod
    def default_domain() -> Domain1D:
        """Return the domain used when the caller does not supply one."""
        return Domain1D(tmin=0.0, tmax=1.0, xmin=0.0, xmax=1.0)

    @abstractmethod
    def residual(self, t: Tensor, x: Tensor) -> Tensor:
        """Return the pointwise PDE residual at ``(t, x)``."""

    @abstractmethod
    def initial_condition(self, x: np.ndarray) -> np.ndarray:
        """Return ``u(0, x)``."""

    @abstractmethod
    def exact(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Return the analytic solution at ``(t, x)``."""

    def initial_velocity(self, x: np.ndarray) -> Optional[np.ndarray]:
        """Return ``u_t(0, x)``, or ``None`` for first-order problems."""
        return None

    def composable_problem(self):
        """Return the composable problem this wrapper trains through.

        Subclasses build the equivalent :mod:`pinnlab.problems` object. The
        wrapper keeps its own :meth:`residual` for direct use and testing, but
        training runs through the problem so the optimisation logic lives in
        one place.
        """
        raise NotImplementedError

    def _problem_residual(self, t: Tensor, x: Tensor) -> Tensor:
        """Evaluate the composable problem's residual on column coordinates.

        Keeps a single definition of each equation. The wrapper exposes
        ``residual(t, x)`` for direct use and testing, while the problem holds
        the physics that training actually optimises; without this adapter the
        two could drift apart and only one of them would be under test.
        """
        from ..residuals import AutogradDerivativeBackend

        coordinates = torch.cat([t, x], dim=1).detach().requires_grad_(True)
        return self.composable_problem().strong_residual(
            self.model, coordinates, AutogradDerivativeBackend()
        )

    def _require_composable_domain(self) -> None:
        """Reject a domain the composable problem cannot represent.

        The problems in :mod:`pinnlab.problems` build their geometry from the
        physical config, spanning ``[0, t_final] x [0, length]``. A wrapper
        constructed on a shifted or rescaled domain would otherwise be trained
        against a different region than it reports, silently.
        """
        config = getattr(self, "config", None)
        length = getattr(config, "length", None)
        if (
            self.domain.tmin != 0.0
            or self.domain.xmin != 0.0
            or (length is not None and self.domain.xmax != length)
        ):
            raise ValueError(
                "this solver trains through the composable problem, whose "
                "geometry spans [0, t_final] x [0, length]; got "
                f"t in [{self.domain.tmin}, {self.domain.tmax}], "
                f"x in [{self.domain.xmin}, {self.domain.xmax}] "
                f"with length={length}"
            )

    def _trainer_config(self, cfg: TrainConfig) -> "TrainerConfig":
        """Translate the wrapper's TrainConfig into a TrainerConfig.

        The two describe the same run with different vocabulary. The mapping
        that is not one-to-one is the loss weights: this wrapper exposes a
        single boundary weight, while the composable problem carries separate
        ``bc_left`` and ``bc_right`` constraints, so the weight applies to each.
        A second-order problem's initial velocity is a constraint of its own and
        takes the initial-condition weight.
        """
        # Imported here rather than at module scope: pinnlab.problems imports
        # from pinnlab.solvers, so a top-level import of the trainer closes a
        # cycle. The lazy exports in pinnlab.training exist for the same reason.
        from ..training.config import OptimizerConfig, TrainerConfig

        w_ic, w_bc, w_pde = cfg.weights
        return TrainerConfig(
            seed=cfg.seed,
            device=self.device,
            collocation_count=cfg.n_f,
            log_every=cfg.log_every,
            causal=cfg.causal,
            loss_weights={
                "pde": w_pde,
                "ic": w_ic,
                "initial_velocity": w_ic,
                "bc_left": w_bc,
                "bc_right": w_bc,
            },
            constraint_sample_counts={
                "ic": cfg.n_ic,
                "initial_velocity": cfg.n_ic,
                "bc_left": cfg.n_bc,
                "bc_right": cfg.n_bc,
            },
            optimizer=OptimizerConfig(
                lr=cfg.lr,
                adam_steps=cfg.adam_steps,
                lbfgs_max_iter=cfg.lbfgs_max_iter,
            ),
        )

    @staticmethod
    def _legacy_record(state: "TrainingState") -> Dict[str, float]:
        """Render a TrainingState in this wrapper's historical shape.

        Callers depend on the ``total``/``ic``/``bc``/``pde`` keys, so the
        problem's separate boundary constraints are summed back into one ``bc``
        entry and a second-order problem's initial-velocity term is folded into
        ``ic``, matching how this wrapper has always reported them.
        """
        losses = state.named_losses
        record = {
            "step": float(state.step),
            "total": state.total_loss,
            "ic": losses.get("ic", 0.0) + losses.get("initial_velocity", 0.0),
            "bc": losses.get("bc_left", 0.0) + losses.get("bc_right", 0.0),
            "pde": losses.get("pde", 0.0),
        }
        record.update(state.diagnostics)
        record.setdefault("min_causal_weight", 1.0)
        return record

    def train(self, cfg: Optional[TrainConfig] = None) -> List[Dict[str, float]]:
        """Fit the network through the generic trainer.

        Args:
            cfg: Training settings. Defaults to :class:`TrainConfig`.

        Returns:
            The loss history: one dict of components per recorded step.

        Raises:
            ValueError: If the configuration is invalid.
        """
        cfg = cfg or TrainConfig()
        cfg.validate()
        if cfg.causal is not None:
            cfg.causal.validate()

        from ..training.trainer import Trainer

        set_seed(cfg.seed)
        self.model.train()
        result = Trainer(self._trainer_config(cfg)).train(
            self.composable_problem(), self.model
        )
        self.loss_history = [self._legacy_record(s) for s in result.history]
        return self.loss_history

    # -- evaluation --------------------------------------------------------

    def predict(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Evaluate the network at matching ``t`` and ``x`` arrays.

        Args:
            t: Times, any shape.
            x: Positions, broadcastable to the shape of ``t``.

        Returns:
            Predictions with the shape of the broadcast inputs.
        """
        t_arr, x_arr = np.broadcast_arrays(np.asarray(t), np.asarray(x))
        flat = np.stack(
            [
                t_arr.reshape(-1).astype(np.float32),
                x_arr.reshape(-1).astype(np.float32),
            ],
            axis=1,
        )
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.from_numpy(flat).to(self.device))
        return out.cpu().numpy().reshape(t_arr.shape)

    def evaluation_grid(
        self, n_t: int = 101, n_x: int = 101
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return a ``(n_t, n_x)`` mesh spanning the domain."""
        d = self.domain
        t = np.linspace(d.tmin, d.tmax, n_t)
        x = np.linspace(d.xmin, d.xmax, n_x)
        return np.meshgrid(t, x, indexing="ij")

    def relative_l2_error(self, n_t: int = 101, n_x: int = 101) -> float:
        """Return the relative L2 error against the analytic solution.

        This is the number the accuracy tests assert on, and the one worth
        quoting: ``||u_pred - u_exact||_2 / ||u_exact||_2`` over a uniform grid.

        Args:
            n_t: Grid points in time.
            n_x: Grid points in space.

        Returns:
            The relative error. Values below ``1e-2`` indicate a solution that
            tracks the analytic one closely.
        """
        tt, xx = self.evaluation_grid(n_t, n_x)
        pred = self.predict(tt, xx)
        truth = np.asarray(self.exact(tt, xx), dtype=float)
        denominator = np.linalg.norm(truth)
        if denominator == 0.0:
            return float(np.linalg.norm(pred - truth))
        return float(np.linalg.norm(pred - truth) / denominator)

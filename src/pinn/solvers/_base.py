"""Shared machinery for solvers that have a closed-form solution to check against.

The solvers built on this base differ from the rest of the package in one
respect that matters: their accuracy is a measurable number rather than a
plot. Each subclass supplies an analytic solution, so
:meth:`ExactlySolvablePINN.relative_l2_error` reports how far the trained
network actually is from the truth, and the tests can assert on it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn

from ..models.mlp import MLP
from ..training.causal_weighting import CausalWeightConfig, causal_residual_loss
from ..utils.logging import get_logger
from .raissi_generic import Domain1D, latin_hypercube, set_seed

logger = get_logger(__name__)

__all__ = ["Domain1D", "TrainConfig", "ExactlySolvablePINN", "gradient"]


def gradient(outputs: Tensor, inputs: Tensor) -> Tensor:
    """Differentiate ``outputs`` with respect to ``inputs``, keeping the graph.

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


@dataclass
class _Samples:
    """Collocation, initial and boundary points for one training run."""

    t_ic: Tensor
    x_ic: Tensor
    u_ic: Tensor
    v_ic: Optional[Tensor]
    t_bc: Tensor
    x_bc: Tensor
    t_f: Tensor
    x_f: Tensor
    history: List[Dict[str, float]] = field(default_factory=list)


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

    # -- sampling ----------------------------------------------------------

    def _sample(self, cfg: TrainConfig) -> _Samples:
        """Draw the initial, boundary and collocation points for a run."""
        d = self.domain

        x_ic = np.random.uniform(d.xmin, d.xmax, size=(cfg.n_ic, 1)).astype(np.float32)
        t_ic = np.zeros_like(x_ic)
        u_ic = np.asarray(self.initial_condition(x_ic), dtype=np.float32)
        v_ic_np = self.initial_velocity(x_ic)

        t_bc = np.random.uniform(d.tmin, d.tmax, size=(cfg.n_bc, 1)).astype(np.float32)
        # Both boundaries in one batch: the first half is x = xmin.
        t_bc = np.vstack([t_bc, t_bc])
        x_bc = np.vstack(
            [
                np.full((cfg.n_bc, 1), d.xmin, dtype=np.float32),
                np.full((cfg.n_bc, 1), d.xmax, dtype=np.float32),
            ]
        )

        unit = latin_hypercube(cfg.n_f, 2, 0.0, 1.0)
        t_f = (d.tmin + (d.tmax - d.tmin) * unit[:, [0]]).astype(np.float32)
        x_f = (d.xmin + (d.xmax - d.xmin) * unit[:, [1]]).astype(np.float32)

        def to_tensor(a: np.ndarray) -> Tensor:
            return torch.from_numpy(a).to(self.device)

        return _Samples(
            t_ic=to_tensor(t_ic),
            x_ic=to_tensor(x_ic),
            u_ic=to_tensor(u_ic),
            v_ic=(
                to_tensor(np.asarray(v_ic_np, dtype=np.float32))
                if v_ic_np is not None
                else None
            ),
            t_bc=to_tensor(t_bc),
            x_bc=to_tensor(x_bc),
            t_f=to_tensor(t_f),
            x_f=to_tensor(x_f),
        )

    # -- losses ------------------------------------------------------------

    def _loss(self, s: _Samples, cfg: TrainConfig) -> Tuple[Tensor, Dict[str, float]]:
        """Return the total loss and a breakdown of its components."""
        w_ic, w_bc, w_pde = cfg.weights

        # Initial condition. Second-order problems also constrain u_t(0, x),
        # which needs a graph through t.
        needs_velocity = s.v_ic is not None
        t_ic = s.t_ic.clone().requires_grad_(needs_velocity)
        u0 = self.model(torch.cat([t_ic, s.x_ic], dim=1))
        ic = torch.mean((u0 - s.u_ic) ** 2)
        if needs_velocity:
            u0_t = gradient(u0, t_ic)
            ic = ic + torch.mean((u0_t - s.v_ic) ** 2)

        # Homogeneous Dirichlet boundaries.
        u_b = self.model(torch.cat([s.t_bc, s.x_bc], dim=1))
        bc = torch.mean(u_b**2)

        # PDE residual, optionally weighted to respect causality in time.
        t_f = s.t_f.clone().requires_grad_(True)
        x_f = s.x_f.clone().requires_grad_(True)
        res = self.residual(t_f, x_f)
        if cfg.causal is not None and cfg.causal.enabled:
            pde, weights = causal_residual_loss(
                res, t_f, cfg.causal, self.domain.tmin, self.domain.tmax
            )
            min_weight = float(weights.min())
        else:
            pde = torch.mean(res**2)
            min_weight = 1.0

        total = w_ic * ic + w_bc * bc + w_pde * pde
        breakdown = {
            "total": float(total.detach()),
            "ic": float(ic.detach()),
            "bc": float(bc.detach()),
            "pde": float(pde.detach()),
            "min_causal_weight": min_weight,
        }
        return total, breakdown

    # -- training ----------------------------------------------------------

    def train(self, cfg: Optional[TrainConfig] = None) -> List[Dict[str, float]]:
        """Fit the network.

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

        set_seed(cfg.seed)
        samples = self._sample(cfg)
        self.model.train()
        history: List[Dict[str, float]] = []

        optimiser = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        for step in range(1, cfg.adam_steps + 1):
            optimiser.zero_grad(set_to_none=True)
            total, breakdown = self._loss(samples, cfg)
            total.backward()
            optimiser.step()

            if step == 1 or step % cfg.log_every == 0 or step == cfg.adam_steps:
                breakdown["step"] = float(step)
                history.append(breakdown)
                logger.info(
                    "Adam step",
                    extra={"step": step, "loss": breakdown["total"]},
                )

        if cfg.lbfgs_max_iter > 0:
            lbfgs = torch.optim.LBFGS(
                self.model.parameters(),
                max_iter=cfg.lbfgs_max_iter,
                history_size=50,
                line_search_fn="strong_wolfe",
            )

            def closure() -> Tensor:
                lbfgs.zero_grad(set_to_none=True)
                total, _ = self._loss(samples, cfg)
                total.backward()
                return total

            lbfgs.step(closure)
            _, breakdown = self._loss(samples, cfg)
            breakdown["step"] = float(cfg.adam_steps + cfg.lbfgs_max_iter)
            history.append(breakdown)
            logger.info("L-BFGS finished", extra={"loss": breakdown["total"]})

        self.loss_history = history
        return history

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

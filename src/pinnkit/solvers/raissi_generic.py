"""
Extensions for Physics-Informed Neural Networks (PINNs):
- Generic continuous-time PINN that works with scalar or vector-valued fields.
- Residuals for Allen–Cahn (scalar) and Nonlinear Schrödinger (complex → 2 channels).
- Minimal demos showing how to plug these residuals and train.

Usage
-----
1) Drop this module next to your existing `pinnkit.solvers.raissi` module (or use standalone).
2) Run the `demo_allen_cahn()` or `demo_schrodinger()` functions at the bottom.

Other PDEs: Replace `*_residual` with your PDE residual f(t,x) using autograd
(e.g., Schrödinger or Allen–Cahn). Then reuse `ContinuousPINNGeneric`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional, Iterable, List

import numpy as np
import torch
from torch import nn, Tensor
from ..models.mlp import MLP
from ..sampling.core import latin_hypercube

from ..utils.logging import (
    get_logger,
    log_performance,
    log_memory_usage,
    log_gpu_usage,
)
from ..utils.checkpointing import CheckpointManager
from ..utils.profiling import profile_performance
from ..training.adaptive_weighting import (
    AdaptiveLossWeighting,
    AdaptiveWeightingConfig,
    LossBalancingAnalyzer,
    WeightEvolutionVisualizer,
    compute_gradient_norms,
)

logger = get_logger(__name__)


# ============================================================
# Small Utils
# ============================================================


def set_seed(seed: int = 123) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# latin_hypercube is imported from pinnkit.sampling.core. It used to be defined
# independently in solver modules, and one copy was broken for d > 1. Keeping a
# single implementation avoids shape bugs and drift while preserving this
# module-level compatibility name.


# ============================================================
# Domain/PDE config
# ============================================================


@dataclass(frozen=True)
class Domain1D:
    tmin: float = 0.0
    tmax: float = 1.0
    xmin: float = -1.0
    xmax: float = 1.0


# ============================================================
# MLP (supports multi-output)
# ============================================================
# ============================================================


class ContinuousPINNGeneric:
    """
    Generic PINN that:
      - enforces IC/BC via user-provided functions (matching output dim)
      - enforces PDE via a residual function returning shape (N, u_dim)
    """

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device,
        domain: Domain1D,
        u_dim: int,
        u0_fn: Callable[[np.ndarray], np.ndarray],  # x -> u0 (N, u_dim)
        bc_left_fn: Callable[[np.ndarray], np.ndarray],  # t -> u(t, xmin) (N, u_dim)
        bc_right_fn: Callable[[np.ndarray], np.ndarray],  # t -> u(t, xmax) (N, u_dim)
        residual_fn: Callable[
            [nn.Module, Tensor, Tensor], Tensor
        ],  # returns (N, u_dim)
    ) -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self.domain = domain
        self.u_dim = u_dim
        self.u0_fn = u0_fn
        self.bc_left_fn = bc_left_fn
        self.bc_right_fn = bc_right_fn
        self.residual_fn = residual_fn
        self.last_loss_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
        self.adaptive_weight_history: List[List[float]] = []

    def _make_data(self, n_u0: int, n_bc: int, n_f: int) -> Dict[str, Tensor]:
        # IC points (t=0)
        x0 = np.random.uniform(
            self.domain.xmin, self.domain.xmax, size=(n_u0, 1)
        ).astype(np.float32)
        t0 = np.zeros_like(x0, dtype=np.float32)
        u0 = self.u0_fn(x0.astype(np.float32)).astype(np.float32)  # (n_u0, u_dim)

        # BC points (left/right boundaries)
        tl = np.random.uniform(
            self.domain.tmin, self.domain.tmax, size=(n_bc, 1)
        ).astype(np.float32)
        tr = np.random.uniform(
            self.domain.tmin, self.domain.tmax, size=(n_bc, 1)
        ).astype(np.float32)
        xl = np.full_like(tl, self.domain.xmin, dtype=np.float32)
        xr = np.full_like(tr, self.domain.xmax, dtype=np.float32)
        ul = self.bc_left_fn(tl).astype(np.float32)  # (n_bc, u_dim)
        ur = self.bc_right_fn(tr).astype(np.float32)  # (n_bc, u_dim)

        # Collocation points (t, x)
        H = latin_hypercube(n_f, 2, 0.0, 1.0)
        tf = (
            self.domain.tmin + (self.domain.tmax - self.domain.tmin) * H[:, [0]]
        ).astype(np.float32)
        xf = (
            self.domain.xmin + (self.domain.xmax - self.domain.xmin) * H[:, [1]]
        ).astype(np.float32)

        def to_t(a: np.ndarray) -> Tensor:
            return torch.from_numpy(a).to(self.device)

        return {
            "t0": to_t(t0),
            "x0": to_t(x0),
            "u0": to_t(u0),
            "tl": to_t(tl),
            "xl": to_t(xl),
            "ul": to_t(ul),
            "tr": to_t(tr),
            "xr": to_t(xr),
            "ur": to_t(ur),
            "tf": to_t(tf),
            "xf": to_t(xf),
        }

    @profile_performance
    @log_performance
    def train(
        self,
        n_u0: int = 200,
        n_bc: int = 200,
        n_f: int = 20_000,
        lr: float = 1e-3,
        steps: int = 10_000,
        lbfgs_max_iter: int = 0,
        seed: int = 123,
        loss_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        checkpoint_manager: Optional[CheckpointManager] = None,
        resume: bool = False,
        adaptive_weighting: Optional[AdaptiveWeightingConfig] = None,
        *,
        callbacks: Optional[Iterable[Callable[[int, Dict[str, float]], None]]] = None,
    ) -> None:
        set_seed(seed)
        data = self._make_data(n_u0, n_bc, n_f)
        if adaptive_weighting is not None:
            adaptive_weighting.validate(3)

        adaptive_manager: Optional[AdaptiveLossWeighting] = None
        weight_visualizer: Optional[WeightEvolutionVisualizer] = None
        if adaptive_weighting is not None and adaptive_weighting.enabled:
            adaptive_manager = AdaptiveLossWeighting(
                ("ic", "bc", "pde"),
                adaptive_weighting,
                device=self.device,
                initial_weights=loss_weights,
            )
            weights_vector = adaptive_manager.weights.to(self.device)
            if adaptive_weighting.visualize:
                weight_visualizer = WeightEvolutionVisualizer(("ic", "bc", "pde"))
        else:
            weights_vector = torch.tensor(
                loss_weights, dtype=torch.float32, device=self.device
            )

        def compute_components() -> Dict[str, Tensor]:
            u_pred0 = self.model(torch.cat([data["t0"], data["x0"]], dim=1))
            ic = torch.mean((u_pred0 - data["u0"]) ** 2)

            u_left = self.model(torch.cat([data["tl"], data["xl"]], dim=1))
            u_right = self.model(torch.cat([data["tr"], data["xr"]], dim=1))
            bc = torch.mean((u_left - data["ul"]) ** 2) + torch.mean(
                (u_right - data["ur"]) ** 2
            )

            f = self.residual_fn(self.model, data["tf"], data["xf"])  # (n_f, u_dim)
            pde = torch.mean(f**2)
            return {"ic": ic, "bc": bc, "pde": pde}

        def loss_fn() -> Tuple[Tensor, Dict[str, Tensor]]:
            components = compute_components()
            total = (
                weights_vector[0] * components["ic"]
                + weights_vector[1] * components["bc"]
                + weights_vector[2] * components["pde"]
            )
            return total, components

        # Adam
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        start_step = 1
        callbacks = list(callbacks or [])
        if checkpoint_manager and resume:
            try:
                _, opt, start_step, _ = checkpoint_manager.load_latest(self.model, opt)
                start_step += 1
            except FileNotFoundError:
                start_step = 1
        logger.info("Starting Adam optimization", extra={"steps": steps})
        stopped_early = False
        final_loss_value = float("nan")
        for step in range(start_step, steps + 1):
            opt.zero_grad(set_to_none=True)
            total_loss, components = loss_fn()
            grad_info = None
            if (
                adaptive_manager is not None
                and adaptive_manager.requires_gradients
                and adaptive_manager.should_update(step)
            ):
                grad_info = compute_gradient_norms(
                    components,
                    self.model.parameters(),
                    retain_graph=True,
                )
            total_loss.backward()
            opt.step()
            loss_value = float(total_loss.detach())
            final_loss_value = loss_value
            if adaptive_manager is not None:
                adaptive_losses = {
                    name: tensor.detach() for name, tensor in components.items()
                }
                adaptive_losses["total"] = total_loss.detach()
                weights_vector = adaptive_manager.update_weights(
                    adaptive_losses,
                    step=step,
                    gradient_norms=grad_info,
                ).to(self.device)
            if step % 1000 == 0 or step == 1:
                logger.info(
                    "Training progress",
                    extra={"step": step, "loss": loss_value},
                )
                log_memory_usage(logger)
                log_gpu_usage(logger)
            if checkpoint_manager:
                checkpoint_manager.maybe_save(
                    self.model, opt, step, {"loss": loss_value}
                )
                if checkpoint_manager.should_stop():
                    checkpoint_manager.restore_best(self.model, opt)
                    logger.info("Early stopping triggered", extra={"step": step})
                    stopped_early = True
                    break
            for cb in callbacks:
                cb(step, {"loss": loss_value})

        # Optional L-BFGS
        self.last_loss_weights = tuple(float(v) for v in weights_vector)
        if adaptive_manager is not None:
            self.adaptive_weight_history = [
                list(w) for w in adaptive_manager.weight_history
            ]
            analyzer = LossBalancingAnalyzer(adaptive_manager)
            logger.info("Adaptive weighting summary", extra=analyzer.summary())
            if weight_visualizer is not None:
                payload = weight_visualizer.create_payload(
                    adaptive_manager.weight_history,
                    adaptive_manager.loss_history.get("total"),
                )
                if isinstance(payload, dict):
                    logger.debug(
                        "Adaptive weighting evolution captured",
                        extra={"updates": len(payload["steps"])},
                    )

        if not stopped_early and lbfgs_max_iter > 0:
            lb = torch.optim.LBFGS(
                self.model.parameters(),
                max_iter=lbfgs_max_iter,
                tolerance_grad=1e-8,
                tolerance_change=1e-9,
                history_size=50,
                line_search_fn="strong_wolfe",
            )

            def closure() -> Tensor:
                lb.zero_grad(set_to_none=True)
                total, _ = loss_fn()
                total.backward()
                return total

            logger.info("Starting L-BFGS optimization")
            lb.step(closure)
            logger.info("L-BFGS optimization completed")
            with torch.no_grad():
                total, _ = loss_fn()
                final_loss_value = float(total.detach())

        logger.info("Training finished", extra={"final_loss": final_loss_value})

    @torch.no_grad()
    def predict(self, t: np.ndarray, x: np.ndarray, batch: int = 4096) -> np.ndarray:
        if t.shape != x.shape:
            raise ValueError("t and x must have the same shape.")
        T = t.reshape(-1, 1).astype(np.float32)
        X = x.reshape(-1, 1).astype(np.float32)
        self.model.eval()
        out = []
        for i in range(0, T.shape[0], batch):
            tb = torch.from_numpy(T[i : i + batch]).to(self.device)
            xb = torch.from_numpy(X[i : i + batch]).to(self.device)
            ub = self.model(torch.cat([tb, xb], dim=1)).cpu().numpy()
            out.append(ub)
        return np.vstack(out).reshape(t.shape + (self.u_dim,))


# ============================================================
# Residuals
# ============================================================

# 1) Allen–Cahn: u_t - nu u_xx + (u^3 - u) = 0  (scalar)


def allen_cahn_residual_factory(
    nu: float,
) -> Callable[[nn.Module, Tensor, Tensor], Tensor]:
    def residual(model: nn.Module, t: Tensor, x: Tensor) -> Tensor:
        if t.shape != x.shape or t.ndim != 2 or t.shape[1] != 1:
            raise ValueError("t and x must be (N,1) with same shape.")
        t = t.clone().detach().requires_grad_(True)
        x = x.clone().detach().requires_grad_(True)
        u = model(torch.cat([t, x], dim=1))  # (N,1)
        ones = torch.ones_like(u)
        u_t = torch.autograd.grad(u, t, ones, retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, ones, retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(
            u_x, x, torch.ones_like(u_x), retain_graph=True, create_graph=True
        )[0]
        f = u_t - nu * u_xx + (u**3 - u)
        return f  # (N,1)

    return residual


# 2) Nonlinear Schrödinger (cubic focusing): i u_t + 0.5 u_xx + |u|^2 u = 0
#    Represent u = a + i b (two outputs). Residual returns (N,2): [Re, Im].


def schrodinger_residual(model: nn.Module, t: Tensor, x: Tensor) -> Tensor:
    if t.shape != x.shape or t.ndim != 2 or t.shape[1] != 1:
        raise ValueError("t and x must be (N,1) with same shape.")
    t = t.clone().detach().requires_grad_(True)
    x = x.clone().detach().requires_grad_(True)

    ab = model(torch.cat([t, x], dim=1))  # (N,2)
    if ab.shape[1] != 2:
        raise ValueError("Schrödinger model must output 2 channels: [Re, Im].")
    a = ab[:, [0]]
    b = ab[:, [1]]

    ones_a = torch.ones_like(a)
    ones_b = torch.ones_like(b)
    a_t = torch.autograd.grad(a, t, ones_a, retain_graph=True, create_graph=True)[0]
    b_t = torch.autograd.grad(b, t, ones_b, retain_graph=True, create_graph=True)[0]
    a_x = torch.autograd.grad(a, x, ones_a, retain_graph=True, create_graph=True)[0]
    b_x = torch.autograd.grad(b, x, ones_b, retain_graph=True, create_graph=True)[0]
    a_xx = torch.autograd.grad(
        a_x, x, torch.ones_like(a_x), retain_graph=True, create_graph=True
    )[0]
    b_xx = torch.autograd.grad(
        b_x, x, torch.ones_like(b_x), retain_graph=True, create_graph=True
    )[0]

    mod2 = a**2 + b**2
    # Re: -b_t + 0.5 a_xx + |u|^2 a = 0
    # Im:  a_t + 0.5 b_xx + |u|^2 b = 0
    res_re = -b_t + 0.5 * a_xx + mod2 * a
    res_im = a_t + 0.5 * b_xx + mod2 * b
    return torch.cat([res_re, res_im], dim=1)  # (N,2)


# ============================================================
# Demos
# ============================================================


def demo_allen_cahn() -> None:
    """Quick demo for Allen–Cahn on [-1,1]×[0,1] with zero Dirichlet BCs."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    domain = Domain1D(tmin=0.0, tmax=1.0, xmin=-1.0, xmax=1.0)

    # IC/BCs (customize as needed)
    def u0(x: np.ndarray) -> np.ndarray:
        return np.cos(np.pi * x).astype(np.float32)  # (N,1)

    def bc_zero(t: np.ndarray) -> np.ndarray:
        return np.zeros((t.shape[0], 1), dtype=np.float32)

    model = MLP(in_dim=2, hidden_layers=8, width=64, out_dim=1)
    pinn = ContinuousPINNGeneric(
        model=model,
        device=device,
        domain=domain,
        u_dim=1,
        u0_fn=u0,
        bc_left_fn=bc_zero,
        bc_right_fn=bc_zero,
        residual_fn=allen_cahn_residual_factory(nu=1e-2),  # diffusion coeff
    )

    pinn.train(
        n_u0=200,
        n_bc=200,
        n_f=20_000,
        lr=1e-3,
        steps=6000,
        lbfgs_max_iter=200,
        seed=123,
    )

    # Predict on grid
    t = np.linspace(domain.tmin, domain.tmax, 41, dtype=np.float32)
    x = np.linspace(domain.xmin, domain.xmax, 101, dtype=np.float32)
    TT, XX = np.meshgrid(t, x, indexing="ij")
    U = pinn.predict(TT, XX)  # shape (T,X,1)
    print("Allen–Cahn U(t=0.5, x)[:5] =", U[20, :5, 0])


def demo_schrodinger() -> None:
    """Quick demo for cubic NLS with periodic-like zero BCs (for illustration)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    domain = Domain1D(tmin=0.0, tmax=1.0, xmin=-5.0, xmax=5.0)

    # IC: Gaussian wave packet (real), zero imaginary part
    def u0(x: np.ndarray) -> np.ndarray:
        a0 = np.exp(-(x**2)).astype(np.float32)
        b0 = np.zeros_like(a0, dtype=np.float32)
        return np.concatenate([a0, b0], axis=1)  # (N,2)

    # Simple BCs: zero at boundaries (replace with periodic enforcement if needed)
    def bc_zero(t: np.ndarray) -> np.ndarray:
        return np.zeros((t.shape[0], 2), dtype=np.float32)

    model = MLP(in_dim=2, hidden_layers=8, width=96, out_dim=2)  # 2 outputs: Re, Im
    pinn = ContinuousPINNGeneric(
        model=model,
        device=device,
        domain=domain,
        u_dim=2,
        u0_fn=u0,
        bc_left_fn=bc_zero,
        bc_right_fn=bc_zero,
        residual_fn=schrodinger_residual,
    )

    pinn.train(
        n_u0=200,
        n_bc=200,
        n_f=30_000,
        lr=1e-3,
        steps=8000,
        lbfgs_max_iter=200,
        seed=123,
    )

    # Predict on grid
    t = np.linspace(domain.tmin, domain.tmax, 21, dtype=np.float32)
    x = np.linspace(domain.xmin, domain.xmax, 201, dtype=np.float32)
    TT, XX = np.meshgrid(t, x, indexing="ij")
    U = pinn.predict(TT, XX)  # (T,X,2)
    print("Schrödinger Re(u)(t=0.5)[:5] =", U[10, :5, 0])
    print("Schrödinger Im(u)(t=0.5)[:5] =", U[10, :5, 1])


if __name__ == "__main__":
    # Choose which demo to run
    demo_allen_cahn()
    # demo_schrodinger()

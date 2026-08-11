"""
Physics-Informed Neural Networks (PINNs) – continuous-time and discrete RK variants.

Other PDEs: Replace burgers_residual with your PDE residual f(t,x) using autograd
(e.g., Schrödinger or Allen–Cahn). Then reuse ContinuousPINN.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional, Iterable, List

import numpy as np
import torch
from torch import Tensor, nn
from ..models.mlp import MLP
from ..distributed import DistributedTrainer

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
from ..sampling.core import latin_hypercube

logger = get_logger(__name__)


# ============================================================
# Utils
# ============================================================


def set_seed(seed: int = 123) -> None:
    """Set PRNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Problem setup (1D Burgers)
# ============================================================


@dataclass(frozen=True)
class BurgersConfig:
    """Domain and PDE parameters for 1D Burgers."""

    tmin: float = 0.0
    tmax: float = 1.0
    xmin: float = -1.0
    xmax: float = 1.0
    nu: float = 0.01 / math.pi  # viscosity


@dataclass(frozen=True)
class Domain1D:
    tmin: float = 0.0
    tmax: float = 1.0
    xmin: float = -1.0
    xmax: float = 1.0


def u0_true(x: np.ndarray) -> np.ndarray:
    """Initial condition: u(0, x) = -sin(pi x)."""
    return -np.sin(np.pi * x)


def u_left_bc(t: np.ndarray) -> np.ndarray:
    """Dirichlet boundary at x=-1: u(t, -1) = 0."""
    return np.zeros_like(t)


def u_right_bc(t: np.ndarray) -> np.ndarray:
    """Dirichlet boundary at x=+1: u(t, +1) = 0."""
    return np.zeros_like(t)


# ============================================================
# Model
# ============================================================


# ============================================================
# Continuous-time PINN (collocation)
# ============================================================


@dataclass
class TrainConfig:
    n_u0: int = 100  # IC points
    n_bc: int = 100  # boundary points PER boundary
    n_f: int = 10_000  # collocation points
    lr: float = 1e-3  # Adam LR
    adam_steps: int = 10_000  # Adam steps before (optional) LBFGS
    lbfgs_max_iter: int = 0  # 0 disables LBFGS
    seed: int = 123
    adaptive_weighting: Optional[AdaptiveWeightingConfig] = None

    def __post_init__(self) -> None:
        if self.adaptive_weighting is not None:
            if not isinstance(self.adaptive_weighting, AdaptiveWeightingConfig):
                raise ValueError(
                    "adaptive_weighting must be an instance of AdaptiveWeightingConfig"
                )
            self.adaptive_weighting.validate(3)


class ContinuousPINN:
    """
    Physics-Informed Neural Network with PDE residual enforced at collocation points.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device,
        pde_residual_fn: Callable[[nn.Module, Tensor, Tensor, float], Tensor],
        cfg_burgers: BurgersConfig,
    ) -> None:
        """
        Parameters
        ----------
        model : nn.Module
            Neural network u_theta(t, x).
        device : 'cpu' or 'cuda'
        pde_residual_fn : Callable
            Function f_theta(t, x, nu) computing PDE residual.
        cfg_burgers : BurgersConfig
            Domain and PDE constants (nu).
        """
        self.model = model.to(device)
        self.device = torch.device(device)
        self.cfg = cfg_burgers
        self.pde_residual_fn = pde_residual_fn
        self.last_loss_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
        self.adaptive_weight_history: List[List[float]] = []

    def _make_training_points(self, tcfg: TrainConfig) -> Dict[str, Tensor]:
        """Draw IC, BC, and collocation points."""
        # IC: t=0, x in [xmin, xmax]
        x0 = np.random.uniform(
            self.cfg.xmin, self.cfg.xmax, size=(tcfg.n_u0, 1)
        ).astype(np.float32)
        t0 = np.zeros_like(x0, dtype=np.float32)
        u0 = u0_true(x0).astype(np.float32)

        # BC left: x=-1, t in [tmin, tmax]
        tl = np.random.uniform(
            self.cfg.tmin, self.cfg.tmax, size=(tcfg.n_bc, 1)
        ).astype(np.float32)
        xl = np.full_like(tl, self.cfg.xmin, dtype=np.float32)
        ul = u_left_bc(tl).astype(np.float32)

        # BC right: x=+1
        tr = np.random.uniform(
            self.cfg.tmin, self.cfg.tmax, size=(tcfg.n_bc, 1)
        ).astype(np.float32)
        xr = np.full_like(tr, self.cfg.xmax, dtype=np.float32)
        ur = u_right_bc(tr).astype(np.float32)

        # Collocation points in (t, x) using LHS in [0,1]^2 then mapped to domain
        H = latin_hypercube(tcfg.n_f, 2, 0.0, 1.0)
        t_f = (self.cfg.tmin + (self.cfg.tmax - self.cfg.tmin) * H[:, [0]]).astype(
            np.float32
        )
        x_f = (self.cfg.xmin + (self.cfg.xmax - self.cfg.xmin) * H[:, [1]]).astype(
            np.float32
        )

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
            "tf": to_t(t_f),
            "xf": to_t(x_f),
        }

    @profile_performance
    @log_performance
    def train(
        self,
        tcfg: TrainConfig,
        weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        checkpoint_manager: Optional[CheckpointManager] = None,
        resume: bool = False,
        distributed: Optional[DistributedTrainer] = None,
        *,
        callbacks: Optional[Iterable[Callable[[int, Dict[str, float]], None]]] = None,
    ) -> None:
        """Train with Adam, then optional L-BFGS."""
        if distributed is not None:
            # ``DistributedTrainer`` will wrap the model and update device.
            self.model = distributed.prepare_model(self.model)
            self.device = distributed.primary_device
            distributed.barrier()

        set_seed(tcfg.seed)
        data = self._make_training_points(tcfg)
        adaptive_cfg = tcfg.adaptive_weighting
        adaptive_manager: Optional[AdaptiveLossWeighting] = None
        weight_visualizer: Optional[WeightEvolutionVisualizer] = None
        if adaptive_cfg is not None and adaptive_cfg.enabled:
            adaptive_manager = AdaptiveLossWeighting(
                ("ic", "bc", "pde"),
                adaptive_cfg,
                device=self.device,
                initial_weights=weights,
            )
            weights_vector = adaptive_manager.weights.to(self.device)
            if adaptive_cfg.visualize:
                weight_visualizer = WeightEvolutionVisualizer(("ic", "bc", "pde"))
        else:
            weights_vector = torch.tensor(
                weights, dtype=torch.float32, device=self.device
            )

        def compute_components() -> Dict[str, Tensor]:
            u0_pred = self.model(torch.cat([data["t0"], data["x0"]], dim=1))
            ic = torch.mean((u0_pred - data["u0"]) ** 2)

            u_left = self.model(torch.cat([data["tl"], data["xl"]], dim=1))
            u_right = self.model(torch.cat([data["tr"], data["xr"]], dim=1))
            bc = torch.mean((u_left - data["ul"]) ** 2) + torch.mean(
                (u_right - data["ur"]) ** 2
            )

            f = self.pde_residual_fn(self.model, data["tf"], data["xf"], self.cfg.nu)
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

        # Adam warmup
        opt = torch.optim.Adam(self.model.parameters(), lr=tcfg.lr)
        self.model.train()
        start_step = 1
        callbacks = list(callbacks or [])
        if checkpoint_manager and resume:
            try:
                # load_latest populates the optimizer in place and hands
                # back the same object, so there is nothing to reassign.
                _, _, start_step, _ = checkpoint_manager.load_latest(self.model, opt)
                start_step += 1
            except FileNotFoundError:
                start_step = 1
        logger.info("Starting Adam optimization", extra={"steps": tcfg.adam_steps})
        stopped_early = False
        final_loss_value = float("nan")
        for step in range(start_step, tcfg.adam_steps + 1):
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

        # Optional L-BFGS polish
        self.last_loss_weights = (
            float(weights_vector[0]),
            float(weights_vector[1]),
            float(weights_vector[2]),
        )
        if adaptive_manager is not None:
            self.adaptive_weight_history = [
                list(w) for w in adaptive_manager.weight_history
            ]
            analyzer = LossBalancingAnalyzer(adaptive_manager)
            logger.info("Adaptive weighting summary", extra=analyzer.summary())
            if weight_visualizer is not None:
                payload = weight_visualizer.create_payload(
                    [list(map(float, w)) for w in adaptive_manager.weight_history],
                    adaptive_manager.loss_history.get("total"),
                )
                if isinstance(payload, dict):
                    logger.debug(
                        "Adaptive weighting evolution captured",
                        extra={"updates": len(payload["steps"])},
                    )

        if not stopped_early and tcfg.lbfgs_max_iter > 0:
            lb = torch.optim.LBFGS(
                self.model.parameters(),
                max_iter=tcfg.lbfgs_max_iter,
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
        """Predict u(t, x) on numpy grids. Shapes broadcast to output."""
        if t.shape != x.shape:
            raise ValueError("t and x must have the same shape (meshgrid style).")
        flat_t = t.reshape(-1, 1).astype(np.float32)
        flat_x = x.reshape(-1, 1).astype(np.float32)
        out = []
        self.model.eval()
        for i in range(0, flat_t.shape[0], batch):
            tb = torch.from_numpy(flat_t[i : i + batch]).to(self.device)
            xb = torch.from_numpy(flat_x[i : i + batch]).to(self.device)
            ub = self.model(torch.cat([tb, xb], dim=1)).cpu().numpy()
            out.append(ub)
        u = np.vstack(out).reshape(t.shape)
        return u


# ============================================================
# PDE residual for Burgers (continuous)
# ============================================================
# NOTE:
# Other PDEs: Replace burgers_residual with your PDE residual f(t,x) using autograd
# (e.g., Schrödinger or Allen–Cahn). Then reuse ContinuousPINN.


def burgers_residual(model: nn.Module, t: Tensor, x: Tensor, nu: float) -> Tensor:
    """
    f = u_t + u u_x - nu u_xx
    Uses autograd to compute derivatives.
    """
    if t.shape != x.shape or t.ndim != 2 or t.shape[1] != 1:
        raise ValueError("t and x must be (N,1) and have identical shapes.")
    t = t.clone().detach().requires_grad_(True)
    x = x.clone().detach().requires_grad_(True)

    tx = torch.cat([t, x], dim=1)
    u = model(tx)
    ones = torch.ones_like(u)

    u_t = torch.autograd.grad(
        u, t, grad_outputs=ones, retain_graph=True, create_graph=True
    )[0]
    u_x = torch.autograd.grad(
        u, x, grad_outputs=ones, retain_graph=True, create_graph=True
    )[0]
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True
    )[0]
    f = u_t + u * u_x - nu * u_xx
    return f


# ============================================================
# Discrete-time RK-PINN (single big step)
# ============================================================


@dataclass(frozen=True)
class RKTableau:
    """Classic explicit RK Butcher tableau (here: RK4)."""

    A: np.ndarray  # (s, s) strictly lower triangular for explicit RK
    b: np.ndarray  # (s,)
    c: np.ndarray  # (s,)


def rk4_tableau() -> RKTableau:
    A = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    b = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6], dtype=np.float32)
    c = np.array([0.0, 0.5, 0.5, 1.0], dtype=np.float32)
    return RKTableau(A=A, b=b, c=c)


def burgers_rhs(u: Tensor, u_x: Tensor, u_xx: Tensor, nu: float) -> Tensor:
    """Spatial operator N[u] = -u u_x + nu u_xx so that u_t = N[u]."""
    return -u * u_x + nu * u_xx


class DiscreteRKPINN:
    """
    Enforce a Runge–Kutta time-step for Burgers from t0 -> t1 at collocation x.
    We use a network U_theta(x, tau) that predicts u at intermediate stage times tau = t0 + c_i h.
    Loss enforces RK relations between stages and endpoints.
    """

    def __init__(
        self,
        stage_net: nn.Module,
        device: str | torch.device,
        cfg: BurgersConfig,
        tableau: RKTableau,
    ) -> None:
        self.net = stage_net.to(device)
        self.device = torch.device(device)
        self.cfg = cfg
        self.tab = tableau

    def _derivs(self, u: Tensor, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute u_x and u_xx of u(x) via autograd (t is implicit in the stage input)."""
        x = x.clone().detach().requires_grad_(True)
        u_x = torch.autograd.grad(
            u, x, torch.ones_like(u), retain_graph=True, create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x, torch.ones_like(u_x), retain_graph=True, create_graph=True
        )[0]
        return u_x, u_xx

    def _u_stage(self, x: Tensor, t_stage: Tensor) -> Tensor:
        """Network prediction of u at stage time t_stage and coordinates x."""
        return self.net(torch.cat([t_stage, x], dim=1))

    @profile_performance
    @log_performance
    def train_one_step(
        self,
        x_colloc: np.ndarray,
        t0: float,
        t1: float,
        u0_fn: Callable[[np.ndarray], np.ndarray],
        lr: float = 1e-3,
        steps: int = 5000,
        seed: int = 123,
        checkpoint_manager: Optional[CheckpointManager] = None,
        resume: bool = False,
    ) -> None:
        """
        Train RK-PINN to satisfy RK4 from t0->t1 at spatial collocation points x_colloc.
        """
        set_seed(seed)
        x_np = x_colloc.astype(np.float32).reshape(-1, 1)
        xb = torch.from_numpy(x_np).to(self.device)

        h = float(t1 - t0)
        if h <= 0:
            raise ValueError("Require t1 > t0 for forward step.")
        s = len(self.tab.b)

        # Stage times t_i = t0 + c_i h
        t_stage_np = (t0 + self.tab.c * h).astype(np.float32).reshape(1, -1)  # (1, s)
        t_stage = torch.from_numpy(np.repeat(t_stage_np, xb.shape[0], axis=0)).to(
            self.device
        )  # (N, s)

        # u^n (IC at t0) on collocation points
        u0 = torch.from_numpy(u0_fn(x_np)).to(self.device)

        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.net.train()
        start_step = 1
        if checkpoint_manager and resume:
            try:
                # load_latest populates the optimizer in place and hands
                # back the same object, so there is nothing to reassign.
                _, _, start_step, _ = checkpoint_manager.load_latest(self.net, opt)
                start_step += 1
            except FileNotFoundError:
                start_step = 1

        for step in range(start_step, steps + 1):
            opt.zero_grad(set_to_none=True)

            # Stage predictions U_i(x) = U_theta(x, t0 + c_i h)
            Ui: list[Tensor] = []
            for i in range(s):
                Ui.append(self._u_stage(xb, t_stage[:, [i]]))  # shape (N,1)
            # Compute k_i(x) = N[U_i]
            ki: list[Tensor] = []
            for i in range(s):
                u_i = Ui[i]
                u_x, u_xx = self._derivs(u_i, xb)
                ki.append(burgers_rhs(u_i, u_x, u_xx, self.cfg.nu))  # (N,1)

            # Enforce RK stage relations: for explicit RK4 stages, the first stage equals u^n.
            # For a strict PINN-RK (Raissi 2017), you can also enforce implicit stages; here we follow explicit RK4 constraints:
            # U_1 == u^n
            loss_stages = torch.mean((Ui[0] - u0) ** 2)

            # Reconstruct u^{n+1} by RK4 formula: u1 = u0 + h * sum_j b_j k_j
            u1_pred = u0 + h * sum(self.tab.b[j] * ki[j] for j in range(s))

            # Soft continuity: final stage equals u1_pred (helps stabilize)
            loss_end = torch.mean((Ui[-1] - u1_pred) ** 2)

            # PDE compliance of stages (stationary residual consistency): u_t ≈ N[u], but we don't have u_t at stage times.
            # We add a mild smoothness regularizer across stages at each x to couple Ui's:
            # A tensor rather than 0.0 so a single-stage tableau, which
            # skips the loop below, still yields a tensor. Benign here --
            # smooth only ever feeds a tensor expression -- but the same
            # shape in raissi_improved reached .item() and crashed.
            smooth = torch.zeros((), device=Ui[0].device, dtype=Ui[0].dtype)
            for i in range(1, s):
                smooth = smooth + torch.mean((Ui[i] - Ui[i - 1]) ** 2)

            loss = loss_stages + loss_end + 1e-3 * smooth
            loss.backward()
            opt.step()

            if checkpoint_manager:
                checkpoint_manager.maybe_save(
                    self.net, opt, step, {"loss": loss.item()}
                )
                if checkpoint_manager.should_stop():
                    checkpoint_manager.restore_best(self.net, opt)
                    break

            if step % 500 == 0 or step == 1:
                print(f"[RK-PINN] step={step:5d}  loss={loss.item():.6e}")

    @torch.no_grad()
    def predict(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Evaluate stage network at arbitrary (t, x). Useful for inspecting Ui profiles.
        """
        if t.shape != x.shape:
            raise ValueError("t and x must have the same shape.")
        flat_t = t.reshape(-1, 1).astype(np.float32)
        flat_x = x.reshape(-1, 1).astype(np.float32)
        self.net.eval()
        out = []
        for i in range(0, flat_t.shape[0], 4096):
            tb = torch.from_numpy(flat_t[i : i + 4096]).to(self.device)
            xb = torch.from_numpy(flat_x[i : i + 4096]).to(self.device)
            ub = self.net(torch.cat([tb, xb], dim=1)).cpu().numpy()
            out.append(ub)
        return np.vstack(out).reshape(t.shape)


# ============================================================
# Demo: train both continuous and discrete PINNs
# ============================================================


def demo() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = BurgersConfig()

    # ---------- Continuous-time PINN ----------
    model_cont = MLP(in_dim=2, hidden_layers=9, width=20, out_dim=1)
    pinn = ContinuousPINN(
        model=model_cont,
        device=device,
        pde_residual_fn=burgers_residual,
        cfg_burgers=cfg,
    )
    tcfg = TrainConfig(
        n_u0=100,
        n_bc=100,
        n_f=10_000,
        lr=1e-3,
        adam_steps=8_000,  # 8-12k is typical before LBFGS
        lbfgs_max_iter=200,  # set 0 to skip LBFGS
        seed=123,
    )
    pinn.train(tcfg, weights=(1.0, 1.0, 1.0))

    # Predict on a grid
    t = np.linspace(cfg.tmin, cfg.tmax, 51, dtype=np.float32)
    x = np.linspace(cfg.xmin, cfg.xmax, 101, dtype=np.float32)
    TT, XX = np.meshgrid(t, x, indexing="ij")
    U_cont = pinn.predict(TT, XX)
    print("Continuous PINN sample U(t=0.5, x)[:5]:", U_cont[25, :5])

    # ---------- Discrete-time RK-PINN (single large step) ----------
    # Network takes (t, x) and outputs u(t, x); we train it to satisfy RK4 from t0->t1
    stage_net = MLP(in_dim=2, hidden_layers=6, width=32, out_dim=1)
    rk = DiscreteRKPINN(
        stage_net=stage_net,
        device=device,
        cfg=cfg,
        tableau=rk4_tableau(),
    )

    # Collocation points in x for RK step
    x_colloc = np.linspace(cfg.xmin, cfg.xmax, 256, dtype=np.float32)
    t0, t1 = 0.10, 0.90  # a large step (like the paper's discrete-time setting)
    rk.train_one_step(
        x_colloc=x_colloc,
        t0=t0,
        t1=t1,
        u0_fn=u0_true,
        lr=1e-3,
        steps=4000,
        seed=123,
    )

    # Inspect discrete model at t1
    T1 = np.full_like(x_colloc, t1)
    U_rk_t1 = rk.predict(T1, x_colloc)
    print("RK-PINN U(t=0.9, x)[:5]:", U_rk_t1[:5].ravel())


if __name__ == "__main__":
    demo()

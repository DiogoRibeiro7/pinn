"""Navier-Stokes PINN solver."""

# file: pinn_navier_stokes.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Iterable, Callable

import numpy as np
import torch
from torch import nn, Tensor
from ..models.mlp import MLP

from ..utils.logging import (
    get_logger,
    log_performance,
    log_memory_usage,
    log_gpu_usage,
)
from ..utils.checkpointing import CheckpointManager
from ..utils.profiling import profile_performance
from ..sampling.core import latin_hypercube

logger = get_logger(__name__)


# ============================================================
# Utilities
# ============================================================


def set_seed(seed: int = 123) -> None:
    """Set PRNG seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Problem setup
# ============================================================


@dataclass(frozen=True)
class NSConfig:
    """
    Domain and physical parameters for 2D incompressible Navier–Stokes (velocity–pressure form).
    """

    tmin: float = 0.0
    tmax: float = 1.0
    xmin: float = 0.0
    xmax: float = 2.0 * math.pi
    ymin: float = 0.0
    ymax: float = 2.0 * math.pi
    nu: float = 0.01  # kinematic viscosity (dimensionless here)


# Taylor–Green vortex (periodic) analytic solution -----------------------------


def tgv_u(t: np.ndarray, x: np.ndarray, y: np.ndarray, nu: float) -> np.ndarray:
    return (np.sin(x) * np.cos(y) * np.exp(-2.0 * nu * t)).astype(np.float32)


def tgv_v(t: np.ndarray, x: np.ndarray, y: np.ndarray, nu: float) -> np.ndarray:
    return (-np.cos(x) * np.sin(y) * np.exp(-2.0 * nu * t)).astype(np.float32)


def tgv_p(t: np.ndarray, x: np.ndarray, y: np.ndarray, nu: float) -> np.ndarray:
    return (-(np.cos(2.0 * x) + np.cos(2.0 * y)) * 0.25 * np.exp(-4.0 * nu * t)).astype(
        np.float32
    )


# ============================================================
# Network
# ============================================================
# ============================================================
# PINN for incompressible NS with periodic BCs
# ============================================================


@dataclass
class TrainConfig:
    n_ic: int = 2_000  # initial condition points (t=0)
    n_f: int = 40_000  # interior collocation points for PDE residuals
    n_per: int = 4_000  # boundary "periodicity" matching pairs (x/y periodic)
    lr: float = 1e-3  # Adam learning rate
    adam_steps: int = 20_000  # Adam steps
    lbfgs_max_iter: int = 0  # set >0 to polish with LBFGS
    seed: int = 123


class NavierStokesPINN:
    """
    Physics-Informed Neural Network for 2D incompressible Navier–Stokes (u,v,p).
    Enforces:
      - Momentum x/y residuals
      - Continuity (divergence-free)
      - Initial condition (from Taylor–Green vortex by default)
      - Periodic boundary conditions (loss on paired samples at opposite faces)
    """

    def __init__(
        self, model: nn.Module, cfg: NSConfig, device: str | torch.device = "cpu"
    ) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = torch.device(device)

    # ---------- Residuals (autograd) ----------

    def _autograd_grads(
        self, u: Tensor, vars: Tuple[Tensor, ...]
    ) -> Tuple[Tensor, ...]:
        """
        Compute first derivatives of u wrt each variable in `vars`.
        """
        ones = torch.ones_like(u)
        return tuple(
            torch.autograd.grad(u, v, ones, retain_graph=True, create_graph=True)[0]
            for v in vars
        )

    def pde_residuals(
        self, t: Tensor, x: Tensor, y: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute NS residuals at collocation points.

        Returns
        -------
        (r_mom_x, r_mom_y, r_cont) each shape (N, 1)
        """
        assert t.shape == x.shape == y.shape and t.ndim == 2 and t.shape[1] == 1

        # require gradients
        t = t.clone().detach().requires_grad_(True)
        x = x.clone().detach().requires_grad_(True)
        y = y.clone().detach().requires_grad_(True)

        # network prediction
        out = self.model(torch.cat([t, x, y], dim=1))
        u, v, p = out[:, [0]], out[:, [1]], out[:, [2]]

        # 1st derivatives
        u_t, u_x, u_y = self._autograd_grads(u, (t, x, y))
        v_t, v_x, v_y = self._autograd_grads(v, (t, x, y))
        p_t, p_x, p_y = self._autograd_grads(
            p, (t, x, y)
        )  # p_t unused, but computed for completeness

        # 2nd derivatives (Laplacian)
        u_xx = torch.autograd.grad(
            u_x, x, torch.ones_like(u_x), retain_graph=True, create_graph=True
        )[0]
        u_yy = torch.autograd.grad(
            u_y, y, torch.ones_like(u_y), retain_graph=True, create_graph=True
        )[0]
        v_xx = torch.autograd.grad(
            v_x, x, torch.ones_like(v_x), retain_graph=True, create_graph=True
        )[0]
        v_yy = torch.autograd.grad(
            v_y, y, torch.ones_like(v_y), retain_graph=True, create_graph=True
        )[0]

        lap_u = u_xx + u_yy
        lap_v = v_xx + v_yy

        # Momentum residuals
        nu = self.cfg.nu
        r_mom_x = u_t + u * u_x + v * u_y + p_x - nu * lap_u
        r_mom_y = v_t + u * v_x + v * v_y + p_y - nu * lap_v

        # Continuity residual
        r_cont = u_x + v_y

        return r_mom_x, r_mom_y, r_cont

    # ---------- Data generation ----------

    def _make_training_points(self, tcfg: TrainConfig) -> Dict[str, Tensor]:
        """
        Build:
        - IC points at t=0 (supervised by TGV)
        - Interior collocation points (t,x,y)
        - Periodicity pairs on each face (x=0~L, y=0~L)
        """
        Lx = self.cfg.xmax - self.cfg.xmin
        Ly = self.cfg.ymax - self.cfg.ymin

        # IC: t = 0, random (x,y)
        x0 = np.random.uniform(
            self.cfg.xmin, self.cfg.xmax, size=(tcfg.n_ic, 1)
        ).astype(np.float32)
        y0 = np.random.uniform(
            self.cfg.ymin, self.cfg.ymax, size=(tcfg.n_ic, 1)
        ).astype(np.float32)
        t0 = np.zeros_like(x0, dtype=np.float32)
        u0 = tgv_u(t0, x0, y0, self.cfg.nu)
        v0 = tgv_v(t0, x0, y0, self.cfg.nu)
        p0 = tgv_p(
            t0, x0, y0, self.cfg.nu
        )  # pressure defined up to constant; consistent here

        # Interior collocation via LHS in unit cube -> map to domain
        H = latin_hypercube(tcfg.n_f, 3, 0.0, 1.0)
        tf = (self.cfg.tmin + (self.cfg.tmax - self.cfg.tmin) * H[:, [0]]).astype(
            np.float32
        )
        xf = (self.cfg.xmin + (self.cfg.xmax - self.cfg.xmin) * H[:, [1]]).astype(
            np.float32
        )
        yf = (self.cfg.ymin + (self.cfg.ymax - self.cfg.ymin) * H[:, [2]]).astype(
            np.float32
        )

        # Periodicity samples: pair (x=0,y) with (x=L,y), and (y=0) with (y=L)
        # Sample y for x-faces; sample x for y-faces
        y_per = np.random.uniform(
            self.cfg.ymin, self.cfg.ymax, size=(tcfg.n_per, 1)
        ).astype(np.float32)
        x_per = np.random.uniform(
            self.cfg.xmin, self.cfg.xmax, size=(tcfg.n_per, 1)
        ).astype(np.float32)
        t_per = np.random.uniform(
            self.cfg.tmin, self.cfg.tmax, size=(tcfg.n_per, 1)
        ).astype(np.float32)

        # Build tensors
        def to_t(a: np.ndarray) -> Tensor:
            return torch.from_numpy(a).to(self.device)

        return {
            # IC
            "t0": to_t(t0),
            "x0": to_t(x0),
            "y0": to_t(y0),
            "u0": to_t(u0),
            "v0": to_t(v0),
            "p0": to_t(p0),
            # Interior
            "tf": to_t(tf),
            "xf": to_t(xf),
            "yf": to_t(yf),
            # Periodic faces
            "tper": to_t(t_per),
            "yper": to_t(y_per),
            "xper": to_t(x_per),
            # Convenience: constants
            "Lx": torch.tensor(Lx, device=self.device, dtype=torch.float32).view(1, 1),
            "Ly": torch.tensor(Ly, device=self.device, dtype=torch.float32).view(1, 1),
        }

    # ---------- Training ----------

    @profile_performance
    @log_performance
    def train(
        self,
        tcfg: TrainConfig,
        w_ic: float = 1.0,
        w_pde: float = 1.0,
        w_per: float = 1.0,
        seed: int = 123,
        checkpoint_manager: Optional[CheckpointManager] = None,
        resume: bool = False,
        *,
        callbacks: Optional[Iterable[Callable[[int, Dict[str, float]], None]]] = None,
    ) -> None:
        """
        Train with Adam, then optional LBFGS.

        Parameters
        ----------
        w_ic : float
            Weight for initial-condition loss.
        w_pde : float
            Weight for PDE residual loss (momentum + continuity).
        w_per : float
            Weight for periodic boundary loss.
        """
        set_seed(seed)
        data = self._make_training_points(tcfg)
        model = self.model

        def loss_fn() -> Tensor:
            # --- Initial condition (supervise u,v,p at t=0) ---
            out0 = model(torch.cat([data["t0"], data["x0"], data["y0"]], dim=1))
            u0_pred, v0_pred, p0_pred = out0[:, [0]], out0[:, [1]], out0[:, [2]]
            # pressure defined up to a constant; subtract batch mean before MSE
            p0_pred_centered = p0_pred - torch.mean(p0_pred)
            p0_centered = data["p0"] - torch.mean(data["p0"])
            lic = (
                torch.mean((u0_pred - data["u0"]) ** 2)
                + torch.mean((v0_pred - data["v0"]) ** 2)
                + torch.mean((p0_pred_centered - p0_centered) ** 2)
            )

            # --- PDE residuals in the interior ---
            r_mx, r_my, r_c = self.pde_residuals(data["tf"], data["xf"], data["yf"])
            lpde = torch.mean(r_mx**2) + torch.mean(r_my**2) + torch.mean(r_c**2)

            # --- Periodic BCs: match (x=0,y) ~ (x=L,y), and (x,y=0) ~ (x,y=L) ---
            tper, yper, xper = data["tper"], data["yper"], data["xper"]
            zero = torch.zeros_like(tper)
            # x-faces
            out_left = model(torch.cat([tper, zero + self.cfg.xmin, yper], dim=1))
            out_right = model(torch.cat([tper, zero + self.cfg.xmax, yper], dim=1))
            lper_x = torch.mean((out_left - out_right) ** 2)
            # y-faces
            out_bottom = model(torch.cat([tper, xper, zero + self.cfg.ymin], dim=1))
            out_top = model(torch.cat([tper, xper, zero + self.cfg.ymax], dim=1))
            lper_y = torch.mean((out_bottom - out_top) ** 2)

            return w_ic * lic + w_pde * lpde + w_per * (lper_x + lper_y)

        # Adam warmup
        opt = torch.optim.Adam(model.parameters(), lr=tcfg.lr)
        model.train()
        start_step = 1
        callbacks = list(callbacks or [])
        if checkpoint_manager and resume:
            try:
                _, opt, start_step, _ = checkpoint_manager.load_latest(model, opt)
                start_step += 1
            except FileNotFoundError:
                start_step = 1
        logger.info("Starting Adam optimization", extra={"steps": tcfg.adam_steps})
        stopped_early = False
        for step in range(start_step, tcfg.adam_steps + 1):
            opt.zero_grad(set_to_none=True)
            L = loss_fn()
            L.backward()
            opt.step()
            if step % 1000 == 0 or step == 1:
                logger.info(
                    "Training progress",
                    extra={"step": step, "loss": L.item()},
                )
                log_memory_usage(logger)
                log_gpu_usage(logger)
            if checkpoint_manager:
                checkpoint_manager.maybe_save(model, opt, step, {"loss": L.item()})
                if checkpoint_manager.should_stop():
                    checkpoint_manager.restore_best(model, opt)
                    logger.info("Early stopping triggered", extra={"step": step})
                    stopped_early = True
                    break
            for cb in callbacks:
                cb(step, {"loss": L.item()})

        # Optional L-BFGS polish
        if not stopped_early and tcfg.lbfgs_max_iter > 0:
            lb = torch.optim.LBFGS(
                model.parameters(),
                max_iter=tcfg.lbfgs_max_iter,
                tolerance_grad=1e-8,
                tolerance_change=1e-9,
                history_size=50,
                line_search_fn="strong_wolfe",
            )

            def closure() -> Tensor:
                lb.zero_grad(set_to_none=True)
                loss = loss_fn()
                loss.backward()
                return loss

            logger.info("Starting L-BFGS optimization")
            lb.step(closure)
            logger.info("L-BFGS optimization completed")

        logger.info("Training finished", extra={"final_loss": L.item()})

    # ---------- Inference ----------

    @torch.no_grad()
    def predict(
        self, t: np.ndarray, x: np.ndarray, y: np.ndarray, batch: int = 4096
    ) -> np.ndarray:
        """
        Predict [u,v,p] on numpy grids. Shapes must match and will be preserved.

        Returns
        -------
        np.ndarray, shape (*grid_shape, 3)
        """
        if not (t.shape == x.shape == y.shape):
            raise ValueError("t, x, y must have the same shape.")
        flat = np.stack([t.ravel(), x.ravel(), y.ravel()], axis=1).astype(np.float32)
        self.model.eval()
        outs = []
        for i in range(0, flat.shape[0], batch):
            tb = torch.from_numpy(flat[i : i + batch]).to(self.device)
            out = self.model(tb).cpu().numpy()
            outs.append(out)
        out = np.vstack(outs).reshape(t.shape + (3,))
        return out


# ============================================================
# Demo: Taylor–Green vortex on periodic box
# ============================================================


def demo_tgv() -> None:
    """
    Train a NS-PINN on Taylor–Green vortex with periodic BCs.
    Keep training short for demo. Increase steps and collocation for accuracy.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = NSConfig(
        tmin=0.0,
        tmax=1.0,
        xmin=0.0,
        xmax=2.0 * math.pi,
        ymin=0.0,
        ymax=2.0 * math.pi,
        nu=0.01,
    )

    model = MLP(in_dim=3, hidden_layers=8, width=64, out_dim=3)
    pinn = NavierStokesPINN(model, cfg, device=device)

    tcfg = TrainConfig(
        n_ic=2000,
        n_f=40000,
        n_per=4000,
        lr=1e-3,
        adam_steps=15000,  # try 30k–80k for better accuracy
        lbfgs_max_iter=0,  # set to 200–500 to polish
        seed=123,
    )

    pinn.train(tcfg, w_ic=1.0, w_pde=1.0, w_per=1.0, seed=tcfg.seed)

    # Evaluate at t=0.5 on a coarse grid
    nx, ny = 81, 81
    tmid = 0.5
    x = np.linspace(cfg.xmin, cfg.xmax, nx, dtype=np.float32)
    y = np.linspace(cfg.ymin, cfg.ymax, ny, dtype=np.float32)
    XX, YY = np.meshgrid(x, y, indexing="ij")
    TT = np.full_like(XX, tmid, dtype=np.float32)

    pred = pinn.predict(TT, XX, YY)  # (..., 3)
    u_pred, v_pred, p_pred = pred[..., 0], pred[..., 1], pred[..., 2]

    # Quick sanity: compare RMSE to analytic solution on the grid (not strict validation)
    u_true = tgv_u(TT, XX, YY, cfg.nu)
    v_true = tgv_v(TT, XX, YY, cfg.nu)
    p_true = tgv_p(TT, XX, YY, cfg.nu)
    rmse_u = float(np.sqrt(np.mean((u_pred - u_true) ** 2)))
    rmse_v = float(np.sqrt(np.mean((v_pred - v_true) ** 2)))
    rmse_p = float(np.sqrt(np.mean((p_pred - p_true) ** 2)))
    print(f"RMSE at t={tmid:.2f} | u: {rmse_u:.3e}, v: {rmse_v:.3e}, p: {rmse_p:.3e}")


if __name__ == "__main__":
    demo_tgv()

#!/usr/bin/env python3
"""Example: Solving the Allen–Cahn equation with a custom PINN."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from pinnlab.models import MLP
from pinnlab.utils.checkpointing import CheckpointManager
from pinnlab.utils.logging import get_logger
from pinnlab.utils.visualization import PINNVisualizer


def allen_cahn_residual(model: MLP, t: Tensor, x: Tensor, eps: float) -> Tensor:
    """Compute Allen–Cahn PDE residual."""
    t = t.clone().detach().requires_grad_(True)
    x = x.clone().detach().requires_grad_(True)
    u = model(torch.cat([t, x], dim=1))
    u_t = torch.autograd.grad(
        u, t, torch.ones_like(u), retain_graph=True, create_graph=True
    )[0]
    u_x = torch.autograd.grad(
        u, x, torch.ones_like(u), retain_graph=True, create_graph=True
    )[0]
    u_xx = torch.autograd.grad(
        u_x, x, torch.ones_like(u_x), retain_graph=True, create_graph=True
    )[0]
    return u_t - eps * u_xx + (u**3 - u) / eps


@dataclass
class ACConfig:
    tmin: float = 0.0
    tmax: float = 1.0
    xmin: float = -1.0
    xmax: float = 1.0
    eps: float = 0.01


def main() -> None:
    parser = argparse.ArgumentParser(description="Allen-Cahn PINN example")
    parser.add_argument("--output-dir", type=str, default="./results")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(__name__)
    logger.info("Starting Allen-Cahn example")

    cfg = ACConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP(in_dim=2, hidden_layers=3, width=32, out_dim=1).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    checkpoint_manager = CheckpointManager(
        out_dir / "checkpoints", save_frequency=500, keep_best_n=2
    )

    for step in range(1, 3001):
        # sample collocation points
        t = torch.rand(256, 1, device=device) * (cfg.tmax - cfg.tmin) + cfg.tmin
        x = torch.rand(256, 1, device=device) * (cfg.xmax - cfg.xmin) + cfg.xmin
        r = allen_cahn_residual(model, t, x, cfg.eps)
        loss_pde = torch.mean(r**2)

        # initial condition u(0,x)=tanh(x / (sqrt(2)*eps))
        x0 = torch.rand(128, 1, device=device) * (cfg.xmax - cfg.xmin) + cfg.xmin
        u0 = torch.tanh(x0 / (np.sqrt(2) * cfg.eps))
        u0_pred = model(torch.cat([torch.zeros_like(x0), x0], dim=1))
        loss_ic = torch.mean((u0_pred - u0) ** 2)

        loss = loss_pde + loss_ic
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 500 == 0:
            logger.info("Training step", extra={"step": step, "loss": loss.item()})
            checkpoint_manager.maybe_save(model, opt, step, {"loss": loss.item()})

    # evaluation
    t = torch.linspace(cfg.tmin, cfg.tmax, 50)
    x = torch.linspace(cfg.xmin, cfg.xmax, 100)
    TT, XX = torch.meshgrid(t, x, indexing="ij")
    with torch.no_grad():
        UU = (
            model(torch.stack([TT.flatten(), XX.flatten()], dim=1).to(device))
            .cpu()
            .numpy()
        )
    viz = PINNVisualizer()
    viz.plot_2d_field(
        XX.numpy(),
        TT.numpy(),
        UU.reshape(TT.shape),
        title="Allen-Cahn PINN",
        xlabel="x",
        ylabel="t",
        save_path=str(out_dir / "allen_cahn_solution.png"),
    )


if __name__ == "__main__":
    main()

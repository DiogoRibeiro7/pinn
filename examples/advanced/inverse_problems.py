#!/usr/bin/env python3
"""Example: Parameter estimation (inverse problem) for Burgers equation."""

import argparse
from pathlib import Path

import numpy as np
import torch

from pinnlab.models import MLP
from pinnlab.solvers.raissi_improved import BurgersConfig, burgers_residual
from pinnlab.utils.logging import get_logger
from pinnlab.utils.metrics import compute_error_metrics


def analytical_solution(t: np.ndarray, x: np.ndarray, nu: float) -> np.ndarray:
    return -np.sin(np.pi * x) * np.exp(-nu * np.pi**2 * t)


def main() -> None:
    parser = argparse.ArgumentParser(description="Burgers inverse problem")
    parser.add_argument("--config", type=str, help="Configuration file")
    parser.add_argument("--output-dir", type=str, default="./results")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(__name__)
    logger.info("Estimating viscosity parameter nu from data")

    cfg = (
        BurgersConfig()
    )  # ConfigFactory returns a generic PINNConfig without .nu/.tmin

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP(in_dim=2, hidden_layers=4, width=32, out_dim=1).to(device)
    nu = torch.nn.Parameter(torch.tensor(0.05, device=device))  # initial guess
    opt = torch.optim.Adam(list(model.parameters()) + [nu], lr=1e-3)

    # synthetic measurement data generated with true nu
    true_nu = cfg.nu
    t_m = np.linspace(cfg.tmin, cfg.tmax, 20, dtype=np.float32)
    x_m = np.linspace(cfg.xmin, cfg.xmax, 40, dtype=np.float32)
    TT, XX = np.meshgrid(t_m, x_m, indexing="ij")
    U_true = analytical_solution(TT, XX, true_nu)
    meas_t = torch.from_numpy(TT.reshape(-1, 1)).to(device)
    meas_x = torch.from_numpy(XX.reshape(-1, 1)).to(device)
    meas_u = torch.from_numpy(U_true.reshape(-1, 1)).to(device)

    for step in range(1, 3001):
        # collocation points
        t = torch.rand(256, 1, device=device)
        x = torch.rand(256, 1, device=device) * 2 - 1
        r = burgers_residual(model, t, x, nu)
        loss_pde = torch.mean(r**2)

        u_pred = model(torch.cat([meas_t, meas_x], dim=1))
        loss_data = torch.mean((u_pred - meas_u) ** 2)

        loss = loss_pde + loss_data
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 500 == 0:
            logger.info(
                "Training step",
                extra={"step": step, "loss": loss.item(), "nu": nu.item()},
            )

    logger.info("Estimated nu", extra={"nu": nu.item()})
    pred = model(torch.cat([meas_t, meas_x], dim=1)).detach().cpu().numpy()
    metrics = compute_error_metrics(pred, U_true.reshape(-1, 1))
    logger.info("MAE on measurements: %f", metrics.mae)


if __name__ == "__main__":
    main()

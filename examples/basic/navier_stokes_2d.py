#!/usr/bin/env python3
"""Example: 2D Navier–Stokes PINN with visualization."""

import argparse
from pathlib import Path

import numpy as np
import torch

from pinn.models import MLP
from pinn.solvers.navier_stokes import (
    NSConfig,
    NavierStokesPINN,
    TrainConfig,
    tgv_u,
    tgv_v,
)
from pinn.utils.checkpointing import CheckpointManager
from pinn.utils.logging import get_logger
from pinn.utils.metrics import compute_error_metrics
from pinn.utils.visualization import PINNVisualizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve 2D Navier-Stokes with PINN")
    parser.add_argument("--config", type=str, help="Configuration file")
    parser.add_argument("--output-dir", type=str, default="./results")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = NSConfig()  # ConfigFactory returns a generic PINNConfig without .xmin/.nu

    logger = get_logger(__name__)
    logger.info("Starting Navier-Stokes solution")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP(in_dim=3, hidden_layers=4, width=64, out_dim=3)
    pinn = NavierStokesPINN(model=model, cfg=cfg, device=device)

    tcfg = TrainConfig(n_ic=200, n_f=2_000, n_per=400, lr=1e-3, adam_steps=2_000)
    checkpoint_manager = CheckpointManager(
        out_dir / "checkpoints", save_frequency=1_000, keep_best_n=2
    )
    pinn.train(tcfg, checkpoint_manager=checkpoint_manager)

    # Evaluation grid for Taylor-Green vortex
    t = np.array([[0.5]], dtype=np.float32)
    x = np.linspace(cfg.xmin, cfg.xmax, 50, dtype=np.float32)
    y = np.linspace(cfg.ymin, cfg.ymax, 50, dtype=np.float32)
    TT, XX, YY = np.meshgrid(t, x, y, indexing="ij")
    with torch.no_grad():
        pred = (
            pinn.model(
                torch.from_numpy(
                    np.stack([TT.ravel(), XX.ravel(), YY.ravel()], axis=1)
                ).to(device)
            )
            .cpu()
            .numpy()
        )
    pred_u = pred[:, 0].reshape(TT.shape)
    pred_v = pred[:, 1].reshape(TT.shape)

    true_u = tgv_u(TT, XX, YY, cfg.nu)
    true_v = tgv_v(TT, XX, YY, cfg.nu)

    metrics = compute_error_metrics(pred_u, true_u)
    logger.info("Velocity-u MAE: %f", metrics.mae)

    viz = PINNVisualizer()
    mag = np.sqrt(pred_u**2 + pred_v**2)
    viz.plot_2d_field(
        XX[0],
        YY[0],
        mag[0],
        title="Velocity magnitude",
        save_path=str(out_dir / "ns_velocity_magnitude.png"),
    )


if __name__ == "__main__":
    main()

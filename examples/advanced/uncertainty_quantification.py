#!/usr/bin/env python3
"""Example: Uncertainty quantification via PINN ensemble."""

import argparse
from pathlib import Path

import numpy as np
import torch

from pinn.models import MLP
from pinn.solvers.raissi_improved import (
    BurgersConfig,
    TrainConfig,
    ContinuousPINN,
    burgers_residual,
)
from pinn.utils.logging import get_logger
from pinn.utils.visualization import PINNVisualizer


def main() -> None:
    parser = argparse.ArgumentParser(description="PINN ensemble for uncertainty estimation")
    parser.add_argument("--config", type=str, help="Configuration file")
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--ensemble", type=int, default=3, help="Number of ensemble members")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(__name__)
    logger.info("Starting PINN ensemble", extra={"ensemble": args.ensemble})

    cfg = BurgersConfig()  # ConfigFactory returns a generic PINNConfig without .nu/.tmin
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tcfg = TrainConfig(n_u0=32, n_bc=32, n_f=1_000, adam_steps=1_000, lbfgs_max_iter=0)

    predictions = []
    t = np.linspace(cfg.tmin, cfg.tmax, 40, dtype=np.float32)
    x = np.linspace(cfg.xmin, cfg.xmax, 80, dtype=np.float32)
    TT, XX = np.meshgrid(t, x, indexing="ij")

    for i in range(args.ensemble):
        model = MLP(in_dim=2, hidden_layers=4, width=32, out_dim=1)
        pinn = ContinuousPINN(model, device, burgers_residual, cfg)
        pinn.train(tcfg)
        pred = pinn.predict(TT, XX)
        predictions.append(pred)
        logger.info("Finished model", extra={"member": i})

    preds = np.stack(predictions)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)

    viz = PINNVisualizer()
    viz.plot_2d_field(XX, TT, mean, title="Mean solution", xlabel="x", ylabel="t",
                      save_path=str(out_dir / "uq_mean.png"))
    viz.plot_2d_field(XX, TT, std, title="Std deviation", xlabel="x", ylabel="t",
                      save_path=str(out_dir / "uq_std.png"))

    logger.info("Ensemble completed")


if __name__ == "__main__":
    main()

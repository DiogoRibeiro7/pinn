#!/usr/bin/env python3
"""Example: Multi-scale progressive training for Burgers equation."""

import argparse
from pathlib import Path

import torch

from pinn.models import MLP
from pinn.solvers.raissi_improved import (
    BurgersConfig,
    TrainConfig,
    ContinuousPINN,
    burgers_residual,
)
from pinn.utils.logging import get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-scale PINN training")
    parser.add_argument("--config", type=str, help="Configuration file")
    parser.add_argument("--output-dir", type=str, default="./results")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(__name__)
    logger.info("Starting multi-scale training example")

    cfg = (
        BurgersConfig()
    )  # ConfigFactory returns a generic PINNConfig without .nu/.tmin
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP(in_dim=2, hidden_layers=4, width=32, out_dim=1)
    pinn = ContinuousPINN(model, device, burgers_residual, cfg)

    # Phase 1: coarse training
    tcfg1 = TrainConfig(n_u0=32, n_bc=32, n_f=1_000, adam_steps=1_000, lbfgs_max_iter=0)
    logger.info("Phase 1: coarse training")
    pinn.train(tcfg1)

    # Phase 2: fine training with more collocation points
    tcfg2 = TrainConfig(
        n_u0=64, n_bc=64, n_f=5_000, adam_steps=1_000, lbfgs_max_iter=50
    )
    logger.info("Phase 2: fine training")
    pinn.train(tcfg2)

    logger.info("Multi-scale training completed")


if __name__ == "__main__":
    main()

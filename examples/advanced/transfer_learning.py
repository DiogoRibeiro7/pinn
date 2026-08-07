#!/usr/bin/env python3
"""Example: Transfer learning for Burgers PINN."""

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
    parser = argparse.ArgumentParser(description="PINN transfer learning example")
    parser.add_argument(
        "--config", type=str, help="Configuration file for pre-training"
    )
    parser.add_argument("--output-dir", type=str, default="./results")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(__name__)
    logger.info("Starting transfer learning example")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP(in_dim=2, hidden_layers=4, width=32, out_dim=1)

    cfg_pre = (
        BurgersConfig()
    )  # ConfigFactory returns a generic PINNConfig without .nu/.tmin
    pinn = ContinuousPINN(model, device, burgers_residual, cfg_pre)
    tcfg = TrainConfig(n_u0=32, n_bc=32, n_f=1_000, adam_steps=1_000, lbfgs_max_iter=0)
    logger.info("Pre-training", extra={"nu": cfg_pre.nu})
    pinn.train(tcfg)

    cfg_fine = BurgersConfig(nu=cfg_pre.nu * 2)
    pinn.cfg = cfg_fine
    logger.info("Fine-tuning", extra={"nu": cfg_fine.nu})
    pinn.train(tcfg)

    logger.info("Transfer learning completed")


if __name__ == "__main__":
    main()

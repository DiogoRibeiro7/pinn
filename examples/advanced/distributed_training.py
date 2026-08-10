#!/usr/bin/env python3
"""Example: training with the :mod:`pinnkit.distributed` helpers."""

import argparse
from pathlib import Path

import torch

from pinnkit.models import MLP
from pinnkit.solvers.raissi_improved import (
    BurgersConfig,
    TrainConfig,
    ContinuousPINN,
    burgers_residual,
)
from pinnkit.distributed import DistributedTrainer
from pinnkit.utils.logging import get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed PINN training")
    parser.add_argument("--config", type=str, help="Configuration file")
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--strategy", type=str, default="data_parallel")
    parser.add_argument(
        "--devices", type=str, default="auto", help="e.g. '0,1' or 'auto'"
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(__name__)
    logger.info("Starting distributed training example")

    cfg = (
        BurgersConfig()
    )  # ConfigFactory returns a generic PINNConfig without .nu/.tmin
    model = MLP(in_dim=2, hidden_layers=2, width=20, out_dim=1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pinn = ContinuousPINN(model, device, burgers_residual, cfg)

    trainer = DistributedTrainer(pinn, strategy=args.strategy, devices=args.devices)
    tcfg = TrainConfig(n_u0=32, n_bc=32, n_f=1_000, adam_steps=100, lbfgs_max_iter=0)
    trainer.fit(tcfg)

    logger.info("Distributed training completed")


if __name__ == "__main__":
    main()

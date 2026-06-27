#!/usr/bin/env python3
"""Benchmark: convergence study varying network width."""

import argparse
from pathlib import Path

import torch

from pinn.models import MLP
from pinn.solvers.raissi_improved import BurgersConfig, TrainConfig, ContinuousPINN, burgers_residual
from pinn.utils.logging import get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Convergence study for PINN width")
    parser.add_argument("--config", type=str, help="Configuration file")
    parser.add_argument("--output-dir", type=str, default="./results")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(__name__)
    cfg = BurgersConfig()  # ConfigFactory returns a generic PINNConfig without .nu/.tmin
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tcfg = TrainConfig(n_u0=32, n_bc=32, n_f=1_000, adam_steps=1_000, lbfgs_max_iter=0)

    results = {}
    for width in [16, 32, 64]:
        model = MLP(in_dim=2, hidden_layers=4, width=width, out_dim=1)
        pinn = ContinuousPINN(model, device, burgers_residual, cfg)
        history = pinn.train(tcfg)
        results[width] = history["total"][-1]
        logger.info("Finished training", extra={"width": width, "loss": history["total"][-1]})

    logger.info("Convergence results", extra=results)


if __name__ == "__main__":
    main()

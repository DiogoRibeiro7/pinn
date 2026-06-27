#!/usr/bin/env python3
"""Benchmark: compare continuous and discrete PINN methods."""

import argparse
from pathlib import Path

import torch

from pinn.config.management import ConfigFactory, ConfigLoader
from pinn.models import MLP
from pinn.solvers.raissi_improved import (
    BurgersConfig,
    ContinuousPINN,
    DiscreteRKPINN,
    TrainConfig,
    burgers_residual,
    rk4_tableau,
)
from pinn.utils.logging import get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Method comparison benchmark")
    parser.add_argument("--config", type=str, help="Configuration file")
    parser.add_argument("--output-dir", type=str, default="./results")
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logger = get_logger(__name__)
    cfg = ConfigLoader.from_yaml(args.config) if args.config else ConfigFactory.create_burgers_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tcfg = TrainConfig(n_u0=32, n_bc=32, n_f=1_000, adam_steps=1_000, lbfgs_max_iter=0)

    model_cont = MLP(in_dim=2, hidden_layers=4, width=32, out_dim=1)
    pinn_cont = ContinuousPINN(model_cont, device, burgers_residual, cfg)
    logger.info("Training continuous PINN")
    history_cont = pinn_cont.train(tcfg)

    stage_net = MLP(in_dim=2, hidden_layers=3, width=32, out_dim=1)
    pinn_rk = DiscreteRKPINN(stage_net, device, burgers_residual, cfg, tableau=rk4_tableau())
    logger.info("Training RK PINN")
    history_rk = pinn_rk.train(tcfg)

    logger.info(
        "Final losses", extra={"continuous": history_cont["total"][-1], "rk": history_rk["total"][-1]}
    )


if __name__ == "__main__":
    main()

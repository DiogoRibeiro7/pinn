#!/usr/bin/env python3
"""Benchmark: measure training performance for Burgers PINN."""

import argparse
from pathlib import Path

from pinn.config.management import ConfigFactory, ConfigLoader
from pinn.models import MLP
from pinn.solvers.raissi_improved import (
    ContinuousPINN,
    BurgersConfig,
    TrainConfig,
    burgers_residual,
)
from pinn.utils.profiling import PerformanceReport, profile_performance
from pinn.utils.logging import get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="PINN performance benchmark")
    parser.add_argument("--config", type=str, help="Configuration file")
    parser.add_argument("--output-dir", type=str, default="./results")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(__name__)
    cfg = ConfigLoader.from_yaml(args.config) if args.config else ConfigFactory.create_burgers_config()
    model = MLP(in_dim=2, hidden_layers=2, width=20, out_dim=1)
    pinn = ContinuousPINN(model=model, device="cpu", pde_residual_fn=burgers_residual, cfg_burgers=cfg)

    report = PerformanceReport()

    @profile_performance(report=report)
    def run_train() -> None:
        tcfg = TrainConfig(adam_steps=5, lbfgs_max_iter=0, track_losses=False, n_u0=10, n_bc=10, n_f=100)
        pinn.train(tcfg)

    run_train()
    report_path = out_dir / "benchmark_report.json"
    report.to_json(report_path)
    logger.info("Benchmark completed", extra={"report": str(report_path)})


if __name__ == "__main__":
    main()

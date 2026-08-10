#!/usr/bin/env python
"""Benchmark training speed for the Burgers PINN solver.

This script executes a short training run and records performance metrics using
:mod:`pinnkit.utils.profiling`. It can be used in CI to detect gross performance
regressions."""

from pathlib import Path

from pinnkit.models.mlp import MLP
from pinnkit.solvers.raissi_improved import (
    ContinuousPINN,
    BurgersConfig,
    TrainConfig,
    burgers_residual,
)
from pinnkit.utils.profiling import PerformanceReport, profile_performance


def main() -> None:
    cfg = BurgersConfig()
    model = MLP(in_dim=2, hidden_layers=2, width=20, out_dim=1)
    pinn = ContinuousPINN(
        model=model, device="cpu", pde_residual_fn=burgers_residual, cfg_burgers=cfg
    )
    report = PerformanceReport()

    # Named explicitly: the benchmark dashboard keys its history off this
    # string, and the default would be "main.<locals>.run_train".
    @profile_performance(report=report, name="burgers_continuous_pinn_train")
    def run_train() -> None:
        tcfg = TrainConfig(
            adam_steps=5,
            lbfgs_max_iter=0,
            track_losses=False,
            n_u0=10,
            n_bc=10,
            n_f=100,
        )
        pinn.train(tcfg)

    run_train()
    report_path = Path("benchmark_report.json")
    report.to_json(report_path)
    print(f"Benchmark metrics written to {report_path}")


if __name__ == "__main__":
    main()

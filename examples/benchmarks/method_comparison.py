#!/usr/bin/env python3
"""Benchmark: compare loss-weighting strategies for a continuous PINN.

The continuous PINN minimises a sum of three losses (IC, BC, PDE).  How those
terms are weighted strongly affects the result -- in particular, up-weighting the
initial condition keeps the network anchored to ``u(0, x)``.  This script trains
two otherwise-identical models and reports their final losses.
"""

import argparse
from pathlib import Path

from pinn.models import MLP
from pinn.solvers.raissi_improved import (
    BurgersConfig,
    ContinuousPINN,
    TrainConfig,
    burgers_residual,
)
from pinn.utils.logging import get_logger


def build_pinn(cfg: BurgersConfig) -> ContinuousPINN:
    model = MLP(in_dim=2, hidden_layers=4, width=32, out_dim=1)
    return ContinuousPINN(model=model, device="cpu", pde_residual_fn=burgers_residual,
                          cfg_burgers=cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Loss-weighting comparison benchmark")
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--adam-steps", type=int, default=1500)
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logger = get_logger(__name__)
    cfg = BurgersConfig()
    tcfg = TrainConfig(n_u0=150, n_bc=100, n_f=1_500, adam_steps=args.adam_steps,
                       lbfgs_max_iter=0, seed=123, track_losses=True)

    logger.info("Training with equal weights (1, 1, 1)")
    hist_equal = build_pinn(cfg).train(tcfg, weights=(1.0, 1.0, 1.0))

    logger.info("Training with IC-weighted loss (10, 1, 1)")
    hist_weighted = build_pinn(cfg).train(tcfg, weights=(10.0, 1.0, 1.0))

    logger.info(
        "Final losses",
        extra={
            "equal_total": hist_equal["total"][-1],
            "equal_ic": hist_equal["ic"][-1],
            "weighted_total": hist_weighted["total"][-1],
            "weighted_ic": hist_weighted["ic"][-1],
        },
    )
    print(f"Equal weights   : total {hist_equal['total'][-1]:.3e} | IC {hist_equal['ic'][-1]:.3e}")
    print(f"IC up-weighted  : total {hist_weighted['total'][-1]:.3e} | IC {hist_weighted['ic'][-1]:.3e}")


if __name__ == "__main__":
    main()

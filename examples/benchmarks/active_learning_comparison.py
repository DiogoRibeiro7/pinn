#!/usr/bin/env python3
"""Compare active learning against random sampling for Burgers PINN."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from pinnkit.models import MLP
from pinnkit.sampling import (
    ActiveLearningConfig,
    ActiveLearningStrategy,
    ActivePINNTrainer,
    UncertaintyAcquisition,
)
from pinnkit.solvers.raissi_improved import (
    BurgersConfig,
    ContinuousPINN,
    TrainConfig,
    burgers_residual,
)
from pinnkit.utils.logging import get_logger


def run_baseline(pinn: ContinuousPINN, config: TrainConfig) -> float:
    """Train without active learning and return the final loss."""

    history = pinn.train(config)
    total = history.get("total") or history.get("loss")
    return float(total[-1]) if total else float("nan")


def build_pinn(cfg: BurgersConfig) -> ContinuousPINN:
    model = MLP(in_dim=2, hidden_layers=2, width=20, out_dim=1)
    return ContinuousPINN(
        model=model, device="cpu", pde_residual_fn=burgers_residual, cfg_burgers=cfg
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Active learning benchmark for PINNs")
    parser.add_argument("--config", type=str, help="Burgers configuration file")
    parser.add_argument(
        "--output", type=str, default="./results", help="Directory to store logs"
    )
    parser.add_argument(
        "--adam-steps", type=int, default=500, help="Number of Adam steps for each run"
    )
    parser.add_argument(
        "--active-interval",
        type=int,
        default=100,
        help="Active learning update interval",
    )
    parser.add_argument(
        "--active-points",
        type=int,
        default=64,
        help="Points added per active iteration",
    )
    parser.add_argument(
        "--pool-size", type=int, default=512, help="Candidate pool size"
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(__name__)

    cfg = (
        BurgersConfig()
    )  # ConfigFactory returns a generic PINNConfig without .nu/.tmin
    baseline_pinn = build_pinn(cfg)
    active_pinn = build_pinn(cfg)

    base_config = TrainConfig(
        n_u0=32,
        n_bc=32,
        n_f=512,
        adam_steps=args.adam_steps,
        lbfgs_max_iter=0,
        track_losses=True,
    )

    baseline_loss = run_baseline(baseline_pinn, base_config)

    strategy = ActiveLearningStrategy(
        UncertaintyAcquisition(mode="variance"),
        batch_size=4,
        diversity_strength=0.3,
    )
    trainer = ActivePINNTrainer(active_pinn, strategy)
    active_cfg = ActiveLearningConfig(
        max_active_iterations=5,
        points_per_iteration=args.active_points,
        candidate_pool_size=args.pool_size,
        update_frequency=args.active_interval,
        convergence_tol=1e-4,
        patience=2,
    )
    active_train = replace(
        base_config,
        track_losses=True,
    )
    active_history = trainer.train_with_active_learning(active_train, active_cfg)
    active_loss = active_history.get("total")
    active_final = float(active_loss[-1]) if active_loss else float("nan")

    logger.info(
        "Active learning comparison",
        extra={
            "baseline_loss": baseline_loss,
            "active_loss": active_final,
            "improvement": baseline_loss - active_final,
        },
    )


if __name__ == "__main__":
    main()

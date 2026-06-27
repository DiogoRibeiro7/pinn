#!/usr/bin/env python3
"""Example: Solving Burgers Equation with PINN."""

import argparse
from pathlib import Path

import numpy as np
import torch

from pinn.config.management import ConfigFactory, ConfigLoader
from pinn.models import MLP
from pinn.solvers.raissi_improved import (
    BurgersConfig,
    TrainConfig,
    ContinuousPINN,
    burgers_residual,
)
from pinn.utils.checkpointing import CheckpointManager
from pinn.utils.logging import get_logger
from pinn.utils.metrics import compute_error_metrics
from pinn.visualization import PINNVisualizer, TrainingDashboard
from pinn.distributed import DistributedTrainer


def analytical_solution(t: np.ndarray, x: np.ndarray, nu: float) -> np.ndarray:
    """Approximate analytical solution for demonstration."""
    return -np.sin(np.pi * x) * np.exp(-nu * np.pi ** 2 * t)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve Burgers equation with a physics-informed neural network"
    )
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--output-dir", type=str, default="./results", help="Directory for results"
    )
    parser.add_argument("--distributed", action="store_true", help="Enable multi-GPU training")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    if args.config:
        cfg = ConfigLoader.from_yaml(args.config)
    else:
        cfg = ConfigFactory.create_burgers_config()

    logger = get_logger(__name__)
    logger.info("Starting Burgers equation solution")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP(in_dim=2, hidden_layers=4, width=32, out_dim=1)
    pinn = ContinuousPINN(
        model=model,
        device=device,
        pde_residual_fn=burgers_residual,
        cfg_burgers=cfg,
    )

    tcfg = TrainConfig(
        n_u0=64,
        n_bc=64,
        n_f=2_000,
        lr=1e-3,
        adam_steps=2_000,
        lbfgs_max_iter=50,
        seed=123,
        track_losses=True,
    )

    checkpoint_manager = CheckpointManager(
        checkpoint_dir=output_dir / "checkpoints",
        save_frequency=1_000,
        keep_best_n=3,
    )
    dashboard = TrainingDashboard(tcfg, update_frequency=200)
    callbacks = [dashboard.update_callback]
    if args.distributed:
        trainer = DistributedTrainer(pinn, devices="auto")
        trainer.fit(
            tcfg,
            checkpoint_manager=checkpoint_manager,
            resume=False,
            callbacks=callbacks,
        )
    else:
        pinn.train(
            tcfg,
            checkpoint_manager=checkpoint_manager,
            resume=False,
            callbacks=callbacks,
        )
    dashboard.serve()

    # Evaluation grid
    t = np.linspace(cfg.tmin, cfg.tmax, 51, dtype=np.float32)
    x = np.linspace(cfg.xmin, cfg.xmax, 101, dtype=np.float32)
    TT, XX = np.meshgrid(t, x, indexing="ij")
    pred = pinn.predict(TT, XX)
    true = analytical_solution(TT, XX, cfg.nu)

    metrics = compute_error_metrics(pred, true)
    logger.info("MAE: %f MSE: %f", metrics["mae"], metrics["mse"])

    viz = PINNVisualizer()
    viz.plot_spacetime_1d(
        TT,
        XX,
        pred,
        title="Burgers PINN Solution",
        save_path=str(output_dir / "burgers_solution.png"),
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Example: Solving Burgers Equation with PINN."""

import argparse
from pathlib import Path

import numpy as np
import torch

from pinnkit.models import MLP
from pinnkit.solvers.reference import burgers_cole_hopf, relative_l2_error
from pinnkit.solvers.raissi_improved import (
    BurgersConfig,
    TrainConfig,
    ContinuousPINN,
    burgers_residual,
)
from pinnkit.utils.checkpointing import CheckpointManager
from pinnkit.utils.logging import get_logger
from pinnkit.utils.metrics import compute_error_metrics
from pinnkit.visualization import PINNVisualizer, TrainingDashboard
from pinnkit.distributed import DistributedTrainer


def analytical_solution(t: np.ndarray, x: np.ndarray, nu: float) -> np.ndarray:
    """Return the exact Burgers solution via the Cole-Hopf transform.

    This used to return ``-sin(pi x) exp(-nu pi^2 t)``, which is the solution of
    the *heat* equation: it drops the nonlinear advection term entirely and so
    never steepens into a shock. Every error metric measured against it was
    therefore meaningless. :func:`burgers_cole_hopf` is the genuine solution of
    the equation this example solves.
    """
    return burgers_cole_hopf(t, x, nu)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve Burgers equation with a physics-informed neural network"
    )
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--output-dir", type=str, default="./results", help="Directory for results"
    )
    parser.add_argument(
        "--distributed", action="store_true", help="Enable multi-GPU training"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Problem configuration (ConfigFactory returns a generic PINNConfig without .nu/.tmin)
    cfg = BurgersConfig()

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
            weights=(10.0, 1.0, 1.0),
            checkpoint_manager=checkpoint_manager,
            resume=False,
            callbacks=callbacks,
        )

    # Evaluation grid
    t = np.linspace(cfg.tmin, cfg.tmax, 51, dtype=np.float32)
    x = np.linspace(cfg.xmin, cfg.xmax, 101, dtype=np.float32)
    TT, XX = np.meshgrid(t, x, indexing="ij")
    pred = pinn.predict(TT, XX)
    true = analytical_solution(TT, XX, cfg.nu)

    metrics = compute_error_metrics(pred, true)
    rel_l2 = relative_l2_error(pred, true)
    logger.info(
        "Accuracy against the Cole-Hopf solution",
        extra={"mae": metrics.mae, "mse": metrics.mse, "relative_l2": rel_l2},
    )
    print(f"Relative L2 error against the exact solution: {rel_l2:.3e}")

    viz = PINNVisualizer()
    viz.plot_2d_field(
        XX,
        TT,
        pred,
        title="Burgers PINN Solution",
        xlabel="x",
        ylabel="t",
        save_path=str(output_dir / "burgers_solution.png"),
    )


if __name__ == "__main__":
    main()

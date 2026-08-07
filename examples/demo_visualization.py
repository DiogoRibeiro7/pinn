#!/usr/bin/env python3
"""Demonstration of the PINN visualization toolkit.

Trains a short Burgers PINN and then exercises the ``PINNVisualizer`` plotting
API -- loss curves, 2-D fields, vector fields, and error analysis -- saving each
figure to the output directory.  Runs headless (no GUI) and non-interactively.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; no display required

import numpy as np

from pinn.models import MLP
from pinn.solvers.raissi_improved import (
    BurgersConfig,
    ContinuousPINN,
    TrainConfig,
    burgers_residual,
)
from pinn.utils.logging import get_logger
from pinn.visualization import PINNVisualizer


def main() -> None:
    parser = argparse.ArgumentParser(description="PINN visualization demo")
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--adam-steps", type=int, default=1200)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(__name__)

    # --- Train a quick model to visualize ---------------------------------
    cfg = BurgersConfig()
    pinn = ContinuousPINN(MLP(2, 4, 32, 1), "cpu", burgers_residual, cfg)
    logger.info("Training a short Burgers PINN for the demo")
    history = pinn.train(
        TrainConfig(
            n_u0=150,
            n_bc=100,
            n_f=1_200,
            adam_steps=args.adam_steps,
            lbfgs_max_iter=0,
            seed=123,
            track_losses=True,
        ),
        weights=(10.0, 1.0, 1.0),
    )

    viz = PINNVisualizer()

    # 1. Loss curves -------------------------------------------------------
    viz.plot_loss_curves(
        history,
        title="Burgers PINN - Training Loss",
        log_scale=True,
        save_path=str(out_dir / "loss_curves.png"),
    )

    # 2. Space-time solution field ----------------------------------------
    nt, nx = 100, 200
    t_grid = np.linspace(cfg.tmin, cfg.tmax, nt, dtype=np.float32)
    x_grid = np.linspace(cfg.xmin, cfg.xmax, nx, dtype=np.float32)
    TT, XX = np.meshgrid(t_grid, x_grid, indexing="ij")
    U = pinn.predict(TT, XX)
    viz.plot_2d_field(
        XX,
        TT,
        U,
        title="u(t, x)",
        xlabel="x",
        ylabel="t",
        colorbar_label="u",
        save_path=str(out_dir / "solution_field.png"),
    )

    # 3. Vector field (analytic Taylor-Green vortex) ----------------------
    gx = np.linspace(0, 2 * np.pi, 40)
    X, Y = np.meshgrid(gx, gx)
    nu = 0.01
    Uf = np.sin(X) * np.cos(Y) * np.exp(-2 * nu * 0.0)
    Vf = -np.cos(X) * np.sin(Y) * np.exp(-2 * nu * 0.0)
    viz.plot_vector_field_2d(
        X,
        Y,
        Uf,
        Vf,
        title="Taylor-Green Vortex",
        skip=2,
        save_path=str(out_dir / "vector_field.png"),
    )

    # 4. Error analysis at the initial condition --------------------------
    x = np.linspace(cfg.xmin, cfg.xmax, 300, dtype=np.float32)
    u_pred = pinn.predict(np.zeros_like(x), x)
    u_true = -np.sin(np.pi * x)
    viz.plot_error_analysis(
        x,
        u_pred,
        u_true,
        title="Error vs. Initial Condition",
        save_path=str(out_dir / "error_analysis.png"),
    )

    logger.info("Saved figures to %s", out_dir)
    print(
        "Saved: loss_curves.png, solution_field.png, vector_field.png, error_analysis.png"
    )


if __name__ == "__main__":
    main()

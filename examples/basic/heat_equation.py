#!/usr/bin/env python3
"""Example: the 1-D heat equation, scored against its exact solution.

Unlike the other examples in this directory, this one ends with a number. The
heat equation with sine initial data has a closed-form Fourier solution, so the
script reports the relative L2 error of the trained network rather than leaving
you to judge a plot.

Run the multi-mode variant to see the harder case: mode ``n`` decays like
``exp(-alpha n^2 pi^2 t)``, so a field built from modes 1 and 4 forces the
network to resolve two very different time scales at once.

    python examples/basic/heat_equation.py
    python examples/basic/heat_equation.py --multi-mode --adam-steps 4000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; no display required

import numpy as np  # noqa: E402

from pinnlab.solvers._base import TrainConfig  # noqa: E402
from pinnlab.solvers.heat import HeatConfig, HeatPINN, heat_exact  # noqa: E402
from pinnlab.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.1, help="thermal diffusivity")
    parser.add_argument("--adam-steps", type=int, default=2000)
    parser.add_argument("--lbfgs-iter", type=int, default=150)
    parser.add_argument("--collocation", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--multi-mode",
        action="store_true",
        help="use a two-mode initial condition, a stiffer problem",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    """Train a heat-equation PINN and report its accuracy."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    modes = ((1, 1.0), (4, 0.5)) if args.multi_mode else ((1, 1.0),)
    config = HeatConfig(alpha=args.alpha, length=1.0, modes=modes)
    solver = HeatPINN(config)

    logger.info(
        "Training heat PINN",
        extra={"alpha": args.alpha, "modes": modes, "steps": args.adam_steps},
    )
    solver.train(
        TrainConfig(
            adam_steps=args.adam_steps,
            lbfgs_max_iter=args.lbfgs_iter,
            n_f=args.collocation,
            seed=args.seed,
        )
    )

    error = solver.relative_l2_error()
    logger.info("Training complete", extra={"relative_l2_error": error})
    print(f"Relative L2 error against the exact solution: {error:.3e}")

    _plot(solver, config, args.out_dir)
    print(f"Figures written to {args.out_dir}")


def _plot(solver: HeatPINN, config: HeatConfig, out_dir: Path) -> None:
    """Write prediction, exact solution and error figures."""
    import matplotlib.pyplot as plt

    tt, xx = solver.evaluation_grid(201, 201)
    pred = solver.predict(tt, xx)
    exact = heat_exact(tt, xx, config.alpha, config.length, config.modes)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, field, title in zip(
        axes,
        (pred, exact, np.abs(pred - exact)),
        ("PINN prediction", "Exact solution", "Absolute error"),
    ):
        mesh = ax.pcolormesh(xx, tt, field, shading="auto", cmap="RdBu_r")
        fig.colorbar(mesh, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
    fig.tight_layout()
    fig.savefig(out_dir / "heat_fields.png", dpi=150)
    plt.close(fig)

    # Slices through time show the decay directly.
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.linspace(0.0, config.length, 201)
    for t_slice in (0.0, 0.25, 0.5, 1.0):
        t = np.full_like(x, t_slice)
        ax.plot(
            x,
            heat_exact(t, x, config.alpha, config.length, config.modes),
            "k-",
            alpha=0.4,
        )
        ax.plot(x, solver.predict(t, x), "--", label=f"t = {t_slice:g}")
    ax.set_xlabel("x")
    ax.set_ylabel("u(t, x)")
    ax.set_title("Solid: exact.  Dashed: PINN.")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "heat_slices.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

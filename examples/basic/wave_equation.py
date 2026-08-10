#!/usr/bin/env python3
"""Example: the 1-D wave equation, scored against its exact solution.

The wave equation is a sterner test of a PINN than the heat equation. It is
second order in time, so the initial data constrains both the displacement and
the velocity; nothing decays, so an error made early is carried forward rather
than damped away; and the solution is periodic, so the optimiser has a family of
plausible wrong answers sitting at the wrong phase.

The script reports the relative L2 error against the exact standing-wave
solution, and can extend the horizon to several periods to show how the error
grows with the length of the interval.

    python examples/basic/wave_equation.py
    python examples/basic/wave_equation.py --periods 3 --adam-steps 6000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; no display required

import numpy as np  # noqa: E402

from pinnlab.solvers._base import TrainConfig  # noqa: E402
from pinnlab.solvers.wave import WaveConfig, WavePINN, wave_exact  # noqa: E402
from pinnlab.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed", type=float, default=1.0, help="wave speed c")
    parser.add_argument(
        "--periods", type=float, default=1.0, help="how many periods to solve over"
    )
    parser.add_argument("--adam-steps", type=int, default=3000)
    parser.add_argument("--lbfgs-iter", type=int, default=200)
    parser.add_argument("--collocation", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    """Train a wave-equation PINN and report its accuracy."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    config = WaveConfig(c=args.speed, length=1.0, modes=((1, 1.0),))
    t_final = args.periods * config.period
    solver = WavePINN(config, t_final=t_final)

    logger.info(
        "Training wave PINN",
        extra={"c": args.speed, "t_final": t_final, "steps": args.adam_steps},
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
    print(f"Horizon: {args.periods:g} period(s), t in [0, {t_final:.3f}]")
    print(f"Relative L2 error against the exact solution: {error:.3e}")
    if args.periods > 1 and error > 0.1:
        print(
            "Note: accuracy degrades over long horizons. This is the regime that "
            "causal weighting targets -- see examples/advanced/causal_weighting.py."
        )

    _plot(solver, config, args.out_dir)
    print(f"Figures written to {args.out_dir}")


def _plot(solver: WavePINN, config: WaveConfig, out_dir: Path) -> None:
    """Write field comparison and error-in-time figures."""
    import matplotlib.pyplot as plt

    tt, xx = solver.evaluation_grid(201, 201)
    pred = solver.predict(tt, xx)
    exact = wave_exact(tt, xx, config.c, config.length, config.modes)

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
    fig.savefig(out_dir / "wave_fields.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(tt[:, 0], np.abs(pred - exact).mean(axis=1))
    ax.set_xlabel("t")
    ax.set_ylabel("mean absolute error")
    ax.set_title("Error growth in time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "wave_error_in_time.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

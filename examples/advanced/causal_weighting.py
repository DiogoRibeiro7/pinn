#!/usr/bin/env python3
"""Example: causal weighting on the 1-D wave equation.

The wave equation is where the ordinary PINN loss is at its weakest. Nothing
decays, so an error made at early times is carried forward rather than damped
away, and the solution is periodic in time, which gives the optimiser a family
of nearby wrong answers at the wrong phase. Minimising the residual everywhere
at once lets the network fit late times before it has fitted early ones.

Causal weighting (Wang, Sankaran and Perdikaris, 2022) splits the interval into
windows and weights window ``i`` by ``exp(-eps * sum of earlier window losses)``,
so a window only starts to matter once everything before it is already
satisfied. This script trains the same problem twice, with and without it, and
reports the relative L2 error of each against the exact standing-wave solution.

    python examples/advanced/causal_weighting.py
    python examples/advanced/causal_weighting.py --eps 5.0 --chunks 32
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")  # headless backend; no display required

import numpy as np  # noqa: E402

from pinn.solvers._base import TrainConfig  # noqa: E402
from pinn.solvers.wave import WaveConfig, WavePINN, wave_exact  # noqa: E402
from pinn.training.causal_weighting import CausalWeightConfig  # noqa: E402
from pinn.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed", type=float, default=1.0, help="wave speed c")
    parser.add_argument("--adam-steps", type=int, default=3000)
    parser.add_argument("--lbfgs-iter", type=int, default=200)
    parser.add_argument("--collocation", type=int, default=4000)
    parser.add_argument("--chunks", type=int, default=16, help="time windows")
    parser.add_argument("--eps", type=float, default=1.0, help="causality strength")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def _train(
    args: argparse.Namespace, causal: CausalWeightConfig | None
) -> Tuple[WavePINN, float]:
    """Train one wave solver and return it with its relative L2 error."""
    solver = WavePINN(WaveConfig(c=args.speed, length=1.0, modes=((1, 1.0),)))
    solver.train(
        TrainConfig(
            adam_steps=args.adam_steps,
            lbfgs_max_iter=args.lbfgs_iter,
            n_f=args.collocation,
            seed=args.seed,
            causal=causal,
        )
    )
    return solver, solver.relative_l2_error()


def main() -> None:
    """Train with and without causal weighting and compare the errors."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Training without causal weighting")
    plain, plain_error = _train(args, None)

    logger.info("Training with causal weighting")
    causal_cfg = CausalWeightConfig(n_chunks=args.chunks, eps=args.eps)
    causal, causal_error = _train(args, causal_cfg)

    print(f"relative L2 error, uniform residual weighting : {plain_error:.3e}")
    print(f"relative L2 error, causal residual weighting  : {causal_error:.3e}")
    if causal_error < plain_error:
        factor = plain_error / max(causal_error, 1e-30)
        print(f"causal weighting improved the error by {factor:.1f}x")
    else:
        # Worth reporting honestly: the benefit depends on eps, the window
        # count and how long the interval is relative to the period.
        print("causal weighting did not help at this setting; try a larger --eps")

    _plot(plain, causal, args.out_dir)
    print(f"Figures written to {args.out_dir}")


def _plot(plain: WavePINN, causal: WavePINN, out_dir: Path) -> None:
    """Plot the error of both runs as a function of time."""
    import matplotlib.pyplot as plt

    cfg = causal.config
    tt, xx = causal.evaluation_grid(201, 201)
    exact = wave_exact(tt, xx, cfg.c, cfg.length, cfg.modes)

    fig, ax = plt.subplots(figsize=(9, 5))
    for solver, label in ((plain, "uniform weighting"), (causal, "causal weighting")):
        err = np.abs(solver.predict(tt, xx) - exact).mean(axis=1)
        ax.semilogy(tt[:, 0], err, label=label)
    ax.set_xlabel("t")
    ax.set_ylabel("mean absolute error")
    ax.set_title("Error growth in time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "causal_weighting_error.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

"""Compare FP32 and FP64 heat PINN smoke runs with benchmark metadata."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pinn.models import MLP
from pinn.problems import HeatProblem
from pinn.residuals import AutogradDerivativeBackend, StrongFormResidual
from pinn.scaling import CharacteristicScales, Nondimensionalizer, ScaledModel
from pinn.solvers.heat import HeatConfig
from pinn.training import OptimizerConfig, Trainer, TrainerConfig


def _git_commit() -> str | None:
    """Return the current git commit, if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _relative_l2(pred: np.ndarray, truth: np.ndarray) -> float:
    """Return relative L2 error, falling back to absolute error for zero truth."""
    denominator = np.linalg.norm(truth)
    error = np.linalg.norm(pred - truth)
    return float(error if denominator == 0.0 else error / denominator)


def _evaluate_solution(
    problem: HeatProblem,
    model: torch.nn.Module,
    *,
    dtype: torch.dtype,
    n_grid: int,
) -> tuple[float, float]:
    """Return solution relative L2 error and independent residual MSE."""
    t = np.linspace(0.0, problem.t_final, n_grid)
    x = np.linspace(0.0, problem.config.length, n_grid)
    tt, xx = np.meshgrid(t, x, indexing="ij")
    flat = torch.tensor(
        np.stack([tt.reshape(-1), xx.reshape(-1)], axis=1),
        dtype=dtype,
    )
    model.eval()
    with torch.no_grad():
        pred = model(flat).cpu().numpy().reshape(tt.shape)
    truth = problem.exact(tt, xx)
    rel_l2 = _relative_l2(pred, truth)

    residual_points = problem.geometry.sample_interior(512, dtype=dtype)
    residual_points.requires_grad_(True)
    residual = StrongFormResidual().evaluate(
        problem,
        model,
        residual_points,
        AutogradDerivativeBackend(),
    )
    residual_mse = float(torch.mean(residual.detach() ** 2))
    return rel_l2, residual_mse


def run_case(
    dtype: torch.dtype, *, use_scaling: bool, args: argparse.Namespace
) -> dict[str, Any]:
    """Run one benchmark case and return serializable metrics."""
    problem = HeatProblem(
        HeatConfig(alpha=args.alpha, length=args.length),
        t_final=args.t_final,
        n_constraint_samples=args.constraint_points,
    )
    scaling = None
    if use_scaling:
        scaling = Nondimensionalizer(
            CharacteristicScales(
                input_scales=(args.t_final, args.length),
                output_scales=(1.0,),
                parameter_scales={"alpha": args.length**2 / args.t_final},
            )
        )
    model = MLP(2, args.hidden_layers, args.width, 1)
    config = TrainerConfig(
        seed=args.seed,
        dtype=dtype,
        scaling=scaling,
        collocation_count=args.collocation_points,
        constraint_sample_counts={
            "ic": args.constraint_points,
            "bc_left": args.constraint_points,
            "bc_right": args.constraint_points,
        },
        optimizer=OptimizerConfig(adam_steps=args.steps, lr=args.lr),
        log_every=max(1, args.steps),
    )
    start = time.perf_counter()
    result = Trainer(config).train(problem, model)
    runtime = time.perf_counter() - start
    eval_model = ScaledModel(model, scaling) if scaling is not None else model
    rel_l2, residual_mse = _evaluate_solution(
        problem, eval_model, dtype=dtype, n_grid=args.eval_grid
    )
    return {
        "dtype": str(dtype).replace("torch.", ""),
        "scaling_enabled": use_scaling,
        "relative_l2_error": rel_l2,
        "residual_mse": residual_mse,
        "runtime_seconds": runtime,
        "final_training_loss": result.final_loss,
        "metadata": {
            "seed": args.seed,
            "device": "cpu",
            "model": {
                "architecture": "MLP",
                "hidden_layers": args.hidden_layers,
                "width": args.width,
            },
            "training": {
                "collocation_points": args.collocation_points,
                "constraint_points": args.constraint_points,
                "adam_steps": args.steps,
                "lr": args.lr,
            },
            "problem": {
                "name": "heat",
                "alpha": args.alpha,
                "length": args.length,
                "t_final": args.t_final,
            },
            "scaling": (
                None
                if scaling is None
                else {
                    "input_scales": [args.t_final, args.length],
                    "output_scales": [1.0],
                    "parameter_scales": {"alpha": args.length**2 / args.t_final},
                }
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("precision_benchmark.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--collocation-points", type=int, default=256)
    parser.add_argument("--constraint-points", type=int, default=32)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--length", type=float, default=1.0)
    parser.add_argument("--t-final", type=float, default=1.0)
    parser.add_argument("--eval-grid", type=int, default=31)
    parser.add_argument("--no-scaling", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the benchmark and write JSON results."""
    args = parse_args()
    cases = []
    for dtype in (torch.float32, torch.float64):
        cases.append(run_case(dtype, use_scaling=not args.no_scaling, args=args))
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "git_commit": _git_commit(),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

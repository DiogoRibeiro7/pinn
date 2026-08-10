"""Compare FP32 and FP64 heat PINN smoke runs with benchmark metadata."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from pinnkit.benchmarks import (
    BenchmarkCase,
    BenchmarkMetric,
    BenchmarkReference,
    BenchmarkReport,
    evaluate_independent_residual,
)
from pinnkit.models import MLP
from pinnkit.problems import HeatProblem
from pinnkit.scaling import CharacteristicScales, Nondimensionalizer, ScaledModel
from pinnkit.solvers.heat import HeatConfig
from pinnkit.training import OptimizerConfig, Trainer, TrainerConfig


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
) -> float:
    """Return solution relative L2 error on a regular grid."""
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
    return _relative_l2(pred, truth)


def run_case(
    dtype: torch.dtype, *, use_scaling: bool, args: argparse.Namespace
) -> BenchmarkCase:
    """Run one benchmark case and return a schema-compatible report case."""
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
    rel_l2 = _evaluate_solution(problem, eval_model, dtype=dtype, n_grid=args.eval_grid)
    residual = evaluate_independent_residual(
        problem,
        eval_model,
        sample_count=args.residual_points,
        seed=args.seed + 1,
        dtype=dtype,
        device="cpu",
    )
    return BenchmarkCase(
        name=f"heat-{str(dtype).replace('torch.', '')}",
        reference=BenchmarkReference(
            kind="analytic",
            name="Fourier heat solution",
            details={"modes": [list(mode) for mode in problem.config.modes]},
        ),
        metrics=(
            BenchmarkMetric(
                name="relative_l2_error",
                value=rel_l2,
                higher_is_better=False,
            ),
            BenchmarkMetric(
                name="residual_mse",
                value=residual.mean_squared_residual,
                higher_is_better=False,
            ),
            BenchmarkMetric(
                name="residual_linf",
                value=residual.max_absolute_residual,
                higher_is_better=False,
            ),
            BenchmarkMetric(
                name="runtime_seconds",
                value=runtime,
                units="s",
                higher_is_better=False,
            ),
            BenchmarkMetric(
                name="final_training_loss",
                value=result.final_loss,
                higher_is_better=False,
            ),
        ),
        metadata={
            "seed": args.seed,
            "residual_seed": residual.seed,
            "device": "cpu",
            "dtype": str(dtype).replace("torch.", ""),
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
            "evaluation": {
                "grid_size": args.eval_grid,
                "residual_points": residual.sample_count,
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
    )


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
    parser.add_argument("--residual-points", type=int, default=512)
    parser.add_argument("--no-scaling", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the benchmark and write JSON results."""
    args = parse_args()
    cases = []
    for dtype in (torch.float32, torch.float64):
        cases.append(run_case(dtype, use_scaling=not args.no_scaling, args=args))
    report = BenchmarkReport(
        cases=tuple(cases),
        metadata={
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "script": Path(__file__).name,
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_json(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

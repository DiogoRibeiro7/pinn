"""Compare uniform collocation against residual-adaptive refinement on heat."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from pinn.benchmarks import (
    BenchmarkCase,
    BenchmarkMetric,
    BenchmarkReference,
    BenchmarkReport,
    evaluate_independent_residual,
)
from pinn.models import MLP
from pinn.problems import HeatProblem
from pinn.solvers.heat import HeatConfig
from pinn.training import (
    OptimizerConfig,
    ResidualAdaptiveConfig,
    Trainer,
    TrainerConfig,
)


def _relative_l2(prediction: np.ndarray, truth: np.ndarray) -> float:
    """Return relative L2 error with a zero-reference fallback."""
    denominator = np.linalg.norm(truth)
    error = np.linalg.norm(prediction - truth)
    return float(error if denominator == 0.0 else error / denominator)


def _evaluate_solution(
    problem: HeatProblem,
    model: torch.nn.Module,
    *,
    dtype: torch.dtype,
    n_grid: int,
) -> float:
    """Return solution relative L2 error on a regular space-time grid."""
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
    return _relative_l2(pred, problem.exact(tt, xx))


def _rar_config(args: argparse.Namespace, mode: str) -> ResidualAdaptiveConfig | None:
    """Return a RAR configuration for a benchmark mode."""
    if mode == "uniform":
        return None
    diversity_weight = args.diversity_weight if mode == "rar_diverse" else 0.0
    return ResidualAdaptiveConfig(
        candidate_count=args.candidate_points,
        points_per_refinement=args.refinement_points,
        refresh_every=args.refresh_every,
        start_step=args.start_step,
        max_refined_points=args.max_refined_points,
        diversity_weight=diversity_weight,
    )


def run_case(mode: str, args: argparse.Namespace) -> BenchmarkCase:
    """Train one benchmark mode and return a schema-compatible case."""
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    problem = HeatProblem(
        HeatConfig(alpha=args.alpha, length=args.length),
        t_final=args.t_final,
        n_constraint_samples=args.constraint_points,
    )
    model = MLP(2, args.hidden_layers, args.width, 1)
    adaptive_refinement = _rar_config(args, mode)
    trainer = Trainer(
        TrainerConfig(
            seed=args.seed,
            dtype=dtype,
            collocation_count=args.collocation_points,
            constraint_sample_counts={
                "ic": args.constraint_points,
                "bc_left": args.constraint_points,
                "bc_right": args.constraint_points,
            },
            adaptive_refinement=adaptive_refinement,
            optimizer=OptimizerConfig(adam_steps=args.steps, lr=args.lr),
            log_every=max(1, args.steps),
        )
    )

    start = time.perf_counter()
    result = trainer.train(problem, model)
    runtime = time.perf_counter() - start
    rel_l2 = _evaluate_solution(problem, model, dtype=dtype, n_grid=args.eval_grid)
    residual = evaluate_independent_residual(
        problem,
        model,
        sample_count=args.residual_points,
        seed=args.seed + 10,
        dtype=dtype,
        device="cpu",
    )
    diagnostics = result.final_state.diagnostics

    metrics = [
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
    ]
    if "rar_points" in diagnostics:
        metrics.append(
            BenchmarkMetric(
                name="rar_points",
                value=diagnostics["rar_points"],
                higher_is_better=None,
            )
        )

    return BenchmarkCase(
        name=f"heat-{mode}",
        reference=BenchmarkReference(
            kind="analytic",
            name="Fourier heat solution",
            details={"modes": [list(mode_) for mode_ in problem.config.modes]},
        ),
        metrics=tuple(metrics),
        metadata={
            "mode": mode,
            "seed": args.seed,
            "residual_seed": residual.seed,
            "dtype": args.dtype,
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
            "adaptive_refinement": (
                None
                if adaptive_refinement is None
                else {
                    "candidate_count": adaptive_refinement.candidate_count,
                    "points_per_refinement": (
                        adaptive_refinement.points_per_refinement
                    ),
                    "refresh_every": adaptive_refinement.refresh_every,
                    "start_step": adaptive_refinement.start_step,
                    "max_refined_points": adaptive_refinement.max_refined_points,
                    "diversity_weight": adaptive_refinement.diversity_weight,
                }
            ),
            "diagnostics": diagnostics,
        },
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("rar_benchmark.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--collocation-points", type=int, default=256)
    parser.add_argument("--constraint-points", type=int, default=32)
    parser.add_argument("--candidate-points", type=int, default=1024)
    parser.add_argument("--refinement-points", type=int, default=64)
    parser.add_argument("--refresh-every", type=int, default=25)
    parser.add_argument("--start-step", type=int, default=25)
    parser.add_argument("--max-refined-points", type=int, default=256)
    parser.add_argument("--diversity-weight", type=float, default=0.25)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--length", type=float, default=1.0)
    parser.add_argument("--t-final", type=float, default=1.0)
    parser.add_argument("--eval-grid", type=int, default=31)
    parser.add_argument("--residual-points", type=int, default=512)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("uniform", "rar", "rar_diverse"),
        default=("uniform", "rar", "rar_diverse"),
    )
    return parser.parse_args()


def main() -> None:
    """Run the benchmark and write a JSON report."""
    args = parse_args()
    cases = tuple(run_case(mode, args) for mode in args.modes)
    report = BenchmarkReport(
        cases=cases,
        metadata={
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "script": Path(__file__).name,
            "description": "Uniform collocation vs residual-adaptive refinement",
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_json(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

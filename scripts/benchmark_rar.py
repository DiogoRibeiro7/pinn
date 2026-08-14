"""Compare uniform collocation against RAR on composable PDE problems."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pinnlab.benchmarks import (
    BenchmarkCase,
    BenchmarkMetric,
    BenchmarkReference,
    BenchmarkReport,
    evaluate_independent_residual,
)
from pinnlab.models import MLP
from pinnlab.problems import (
    AllenCahnProblem,
    BurgersProblem,
    HeatProblem,
    NavierStokesProblem,
    SchrodingerProblem,
    WaveProblem,
)
from pinnlab.problems.allen_cahn import AllenCahnConfig
from pinnlab.solvers.heat import HeatConfig
from pinnlab.solvers.raissi_improved import BurgersConfig
from pinnlab.solvers.wave import WaveConfig
from pinnlab.training import (
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


def _reference_fields(problem: Any, n_grid: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return stacked grid coordinates and one reference array per output.

    Handles the three shapes the composable problems now come in: an analytic
    ``exact`` returning one array, an analytic ``exact`` returning a tuple of
    components, and a problem whose reference is numerical and therefore
    published as ``reference_grid`` rather than a closed form.
    """
    lower = problem.geometry.lower_bounds.numpy()
    upper = problem.geometry.upper_bounds.numpy()
    axes = [
        np.linspace(float(lower[i]), float(upper[i]), n_grid) for i in range(len(lower))
    ]
    if not hasattr(problem, "exact"):
        # A numerical reference is published as a grid rather than a closed
        # form, and it chooses its own resolution: Allen-Cahn needs enough
        # points in x to resolve its phase interfaces, and re-sampling that onto
        # an arbitrary grid would discard the accuracy it was computed at.
        grid = problem.reference_grid()
        axis_names = [name for name in ("t", "x", "y") if name in grid]
        coordinates = np.stack([grid[name].reshape(-1) for name in axis_names], axis=1)
        return coordinates, [grid["reference"]]

    mesh = np.meshgrid(*axes, indexing="ij")
    coordinates = np.stack([m.reshape(-1) for m in mesh], axis=1)

    truth = problem.exact(*mesh)
    fields = list(truth) if isinstance(truth, tuple) else [truth]
    return coordinates, fields


def _evaluate_solution(
    problem: Any,
    model: torch.nn.Module,
    *,
    dtype: torch.dtype,
    n_grid: int,
) -> float:
    """Return relative L2 error over every output on a regular grid.

    For a system the components are compared together rather than one at a
    time: the error is the norm of the stacked residual over the norm of the
    stacked reference, so a problem is not scored well merely because one of its
    fields is easy.
    """
    coordinates, fields = _reference_fields(problem, n_grid)
    flat = torch.tensor(coordinates, dtype=dtype)
    model.eval()
    with torch.no_grad():
        pred = model(flat).cpu().numpy()
    stacked_truth = np.concatenate([f.reshape(-1) for f in fields])
    stacked_pred = np.concatenate([pred[:, i] for i in range(len(fields))])
    return _relative_l2(stacked_pred, stacked_truth)


def _make_problem(
    name: str, args: argparse.Namespace
) -> BurgersProblem | HeatProblem | WaveProblem:
    """Create a composable benchmark problem."""
    if name == "burgers":
        return BurgersProblem(
            BurgersConfig(tmax=args.t_final, nu=args.nu),
            n_constraint_samples=args.constraint_points,
        )
    if name == "heat":
        return HeatProblem(
            HeatConfig(alpha=args.alpha, length=args.length),
            t_final=args.t_final,
            n_constraint_samples=args.constraint_points,
        )
    if name == "wave":
        return WaveProblem(
            WaveConfig(c=args.wave_speed, length=args.length),
            t_final=args.t_final,
            n_constraint_samples=args.constraint_points,
        )
    if name == "allen_cahn":
        # nu = 1e-2 rather than the 1e-4 benchmark: at the stiff value nothing
        # at this budget escapes the trivial solution, so every mode would score
        # the same and the comparison would measure nothing.
        return AllenCahnProblem(
            AllenCahnConfig(nu=1e-2),
            n_constraint_samples=args.constraint_points,
        )
    if name == "schrodinger":
        return SchrodingerProblem(n_constraint_samples=args.constraint_points)
    if name == "navier_stokes":
        return NavierStokesProblem(n_constraint_samples=args.constraint_points)
    raise ValueError(f"unknown benchmark problem: {name}")


def _reference_name(problem_name: str) -> str:
    """Return the analytic reference label for a problem."""
    if problem_name == "heat":
        return "Fourier heat solution"
    if problem_name == "wave":
        return "Standing-wave solution"
    if problem_name == "burgers":
        return "Cole-Hopf Burgers solution"
    if problem_name == "allen_cahn":
        return "Fourier spectral Allen-Cahn solution"
    if problem_name == "schrodinger":
        return "Nonlinear Schrodinger soliton"
    if problem_name == "navier_stokes":
        return "Taylor-Green vortex"
    raise ValueError(f"unknown benchmark problem: {problem_name}")


def _reference_details(problem: Any) -> dict[str, Any]:
    """Return serializable reference details for a problem.

    Dispatches on type rather than assuming every non-Burgers problem carries
    ``config.modes``, which was true only while heat and wave were the others.
    """
    if isinstance(problem, BurgersProblem):
        return {
            "initial_condition": "-sin(pi x)",
            "boundary_conditions": "homogeneous_dirichlet",
            "nu": problem.config.nu,
        }
    if isinstance(problem, AllenCahnProblem):
        return {
            "initial_condition": "x^2 cos(pi x)",
            "boundary_conditions": "periodic",
            "nu": problem.config.nu,
            "gamma": problem.config.gamma,
        }
    if isinstance(problem, SchrodingerProblem):
        return {
            "initial_condition": f"{problem.config.amplitude} sech(x)",
            "boundary_conditions": "periodic",
            "amplitude": problem.config.amplitude,
        }
    if isinstance(problem, NavierStokesProblem):
        return {
            "initial_condition": "Taylor-Green vortex",
            "boundary_conditions": "doubly_periodic",
            "nu": problem.config.nu,
        }
    return {"modes": [list(mode_) for mode_ in problem.config.modes]}


def _problem_metadata(problem_name: str, problem: Any) -> dict[str, Any]:
    """Return serializable problem metadata."""
    geometry = problem.geometry
    metadata: dict[str, Any] = {
        "name": problem_name,
        "t_final": float(geometry.upper_bounds[0]),
        "x_min": float(geometry.lower_bounds[1]),
        "x_max": float(geometry.upper_bounds[1]),
        "input_dim": problem.input_dim,
        "output_dim": problem.output_dim,
        "reference_kind": getattr(problem, "reference_kind", "analytic"),
    }
    if isinstance(problem, BurgersProblem):
        metadata["nu"] = problem.config.nu
    elif isinstance(problem, HeatProblem):
        metadata["alpha"] = problem.config.alpha
    elif isinstance(problem, WaveProblem):
        metadata["wave_speed"] = problem.config.c
    elif isinstance(problem, AllenCahnProblem):
        metadata["nu"] = problem.config.nu
        metadata["gamma"] = problem.config.gamma
    elif isinstance(problem, SchrodingerProblem):
        metadata["amplitude"] = problem.config.amplitude
    elif isinstance(problem, NavierStokesProblem):
        metadata["nu"] = problem.config.nu
    return metadata


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


def run_case(problem_name: str, mode: str, args: argparse.Namespace) -> BenchmarkCase:
    """Train one benchmark mode and return a schema-compatible case."""
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    problem = _make_problem(problem_name, args)
    model = MLP(problem.input_dim, args.hidden_layers, args.width, problem.output_dim)
    adaptive_refinement = _rar_config(args, mode)
    constraint_sample_counts = {
        constraint.name: args.constraint_points for constraint in problem.constraints()
    }
    trainer = Trainer(
        TrainerConfig(
            seed=args.seed,
            dtype=dtype,
            collocation_count=args.collocation_points,
            constraint_sample_counts=constraint_sample_counts,
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
        name=f"{problem_name}-{mode}",
        reference=BenchmarkReference(
            kind="analytic",
            name=_reference_name(problem_name),
            details=_reference_details(problem),
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
            "problem": _problem_metadata(problem_name, problem),
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
    parser.add_argument("--nu", type=float, default=0.01 / np.pi)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--wave-speed", type=float, default=1.0)
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
    # The default set is the three problems with closed-form scalar solutions,
    # which is what this benchmark was built around. The others are opt-in
    # because they are slower and, for Allen-Cahn, scored against a numerical
    # reference rather than an exact one.
    parser.add_argument(
        "--problems",
        nargs="+",
        choices=(
            "burgers",
            "heat",
            "wave",
            "allen_cahn",
            "schrodinger",
            "navier_stokes",
        ),
        default=("burgers", "heat", "wave"),
    )
    return parser.parse_args()


def main() -> None:
    """Run the benchmark and write a JSON report."""
    args = parse_args()
    cases = tuple(
        run_case(problem_name, mode, args)
        for problem_name in args.problems
        for mode in args.modes
    )
    report = BenchmarkReport(
        cases=cases,
        metadata={
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "script": Path(__file__).name,
            "description": "Uniform collocation vs residual-adaptive refinement",
            "problems": list(args.problems),
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_json(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

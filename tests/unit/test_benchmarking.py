"""Tests for benchmark reports and numerical reference helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from pinnlab.benchmarks import (
    BenchmarkCase,
    BenchmarkMetric,
    BenchmarkReference,
    BenchmarkReport,
    evaluate_independent_residual,
)
from pinnlab.numerics import solve_heat_explicit
from pinnlab.problems import HeatProblem
from pinnlab.solvers.heat import HeatConfig, heat_exact


class AnalyticHeatModel(nn.Module):
    """Torch module for the single-mode analytic heat solution."""

    def __init__(self, alpha: float, length: float) -> None:
        """Store physical coefficients for the analytic solution."""
        super().__init__()
        self.alpha = alpha
        self.length = length

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Evaluate the analytic single-mode heat solution."""
        t = coordinates[:, [0]]
        x = coordinates[:, [1]]
        k = torch.as_tensor(np.pi / self.length, dtype=x.dtype, device=x.device)
        decay = torch.exp(-self.alpha * k**2 * t)
        return decay * torch.sin(k * x)


def test_benchmark_report_round_trips_json(tmp_path) -> None:
    """Benchmark reports preserve schema fields through JSON."""
    report = BenchmarkReport(
        generated_at="2026-08-09T00:00:00+00:00",
        git_commit="abc123",
        metadata={"runner": "unit-test"},
        cases=(
            BenchmarkCase(
                name="heat-single-mode",
                reference=BenchmarkReference(
                    kind="analytic",
                    name="Fourier heat solution",
                    details={"modes": [[1, 1.0]]},
                ),
                metrics=(
                    BenchmarkMetric(
                        name="relative_l2",
                        value=1.2e-3,
                        higher_is_better=False,
                    ),
                ),
                metadata={"dtype": "float64"},
            ),
        ),
    )

    path = tmp_path / "benchmark_report.json"
    report.to_json(path)

    loaded = BenchmarkReport.from_json(path)
    assert loaded == report
    assert loaded.to_dict()["cases"][0]["reference"]["kind"] == "analytic"


def test_report_rejects_unknown_reference_kind() -> None:
    """Reference provenance labels are restricted to supported categories."""
    with pytest.raises(ValueError, match="unknown reference kind"):
        BenchmarkReference(kind="empirical", name="bad")  # type: ignore[arg-type]


def test_independent_residual_evaluator_scores_fresh_points() -> None:
    """The independent evaluator scores an analytic model near zero residual."""
    config = HeatConfig(alpha=0.1, length=1.0, modes=((1, 1.0),))
    problem = HeatProblem(config=config, t_final=0.25)
    model = AnalyticHeatModel(alpha=config.alpha, length=config.length).double()

    metrics = evaluate_independent_residual(
        problem,
        model,
        sample_count=128,
        seed=123,
        dtype=torch.float64,
    )

    assert metrics.sample_count == 128
    assert metrics.seed == 123
    assert metrics.mean_squared_residual < 1e-24
    assert metrics.max_absolute_residual < 1e-11


def test_explicit_heat_reference_matches_analytic_solution() -> None:
    """The explicit finite-difference reference matches the closed form."""
    config = HeatConfig(alpha=0.1, length=1.0, modes=((1, 1.0),))
    t_final = 0.02

    solution = solve_heat_explicit(
        alpha=config.alpha,
        length=config.length,
        t_final=t_final,
        n_x=51,
        n_t=101,
        initial_condition=lambda x: heat_exact(
            np.zeros_like(x),
            x,
            config.alpha,
            config.length,
            config.modes,
        ),
    )

    t_grid, x_grid = np.meshgrid(solution.t, solution.x, indexing="ij")
    exact = heat_exact(
        t_grid,
        x_grid,
        config.alpha,
        config.length,
        config.modes,
    )
    rel_l2 = np.linalg.norm(solution.u - exact) / np.linalg.norm(exact)

    assert solution.reference_kind == "numerical"
    assert solution.metadata["method"] == "explicit_ftcs"
    assert rel_l2 < 1e-3


def test_explicit_heat_reference_rejects_unstable_grid() -> None:
    """The explicit heat solver rejects CFL-unstable configurations."""
    with pytest.raises(ValueError, match="unstable"):
        solve_heat_explicit(
            alpha=0.1,
            length=1.0,
            t_final=1.0,
            n_x=11,
            n_t=10,
            initial_condition=np.sin,
        )

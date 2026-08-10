"""Tests for the RAR benchmark script."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest

from pinnkit.benchmarks import BenchmarkCase


@pytest.fixture(scope="module")
def benchmark_rar_module():
    """Load the RAR benchmark script as a module."""
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_rar.py"
    spec = importlib.util.spec_from_file_location("benchmark_rar", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load benchmark_rar.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args() -> argparse.Namespace:
    """Return a tiny benchmark configuration for unit tests."""
    return argparse.Namespace(
        seed=0,
        steps=1,
        collocation_points=8,
        constraint_points=4,
        candidate_points=8,
        refinement_points=2,
        refresh_every=1,
        start_step=1,
        max_refined_points=4,
        diversity_weight=0.5,
        hidden_layers=1,
        width=8,
        lr=1e-3,
        nu=0.01,
        alpha=0.1,
        wave_speed=1.0,
        length=1.0,
        t_final=0.02,
        eval_grid=5,
        residual_points=8,
        dtype="float32",
    )


def test_rar_benchmark_script_emits_uniform_case(benchmark_rar_module) -> None:
    """The uniform benchmark mode produces a schema-compatible case."""
    case = benchmark_rar_module.run_case("heat", "uniform", _args())

    assert isinstance(case, BenchmarkCase)
    assert case.name == "heat-uniform"
    assert case.reference.kind == "analytic"
    assert case.metadata["adaptive_refinement"] is None
    assert {metric.name for metric in case.metrics} >= {
        "relative_l2_error",
        "residual_mse",
        "runtime_seconds",
    }


def test_rar_benchmark_script_emits_diverse_rar_case(benchmark_rar_module) -> None:
    """The diverse RAR mode records adaptive-refinement metadata."""
    case = benchmark_rar_module.run_case("heat", "rar_diverse", _args())

    assert case.name == "heat-rar_diverse"
    assert case.metadata["adaptive_refinement"]["diversity_weight"] == 0.5
    assert "rar_points" in {metric.name for metric in case.metrics}


def test_rar_benchmark_script_emits_wave_case(benchmark_rar_module) -> None:
    """The benchmark can run the composable wave problem."""
    case = benchmark_rar_module.run_case("wave", "uniform", _args())

    assert case.name == "wave-uniform"
    assert case.reference.name == "Standing-wave solution"
    assert case.metadata["problem"]["name"] == "wave"
    assert case.metadata["problem"]["wave_speed"] == 1.0


def test_rar_benchmark_script_emits_burgers_case(benchmark_rar_module) -> None:
    """The benchmark can run the composable Burgers problem."""
    case = benchmark_rar_module.run_case("burgers", "uniform", _args())

    assert case.name == "burgers-uniform"
    assert case.reference.name == "Cole-Hopf Burgers solution"
    assert case.metadata["problem"]["name"] == "burgers"
    assert case.metadata["problem"]["nu"] == 0.01

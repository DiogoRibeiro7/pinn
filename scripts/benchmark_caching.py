#!/usr/bin/env python3
"""Benchmark adaptive caching strategies for PINN training."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from statistics import mean

import torch

from pinn.models.mlp import MLP
from pinn.optimization.caching import CacheConfig
from pinn.solvers.raissi_improved import (
    BurgersConfig,
    ContinuousPINN,
    TrainConfig,
    burgers_residual,
)


def run_training(
    use_cache: bool, repeats: int, steps: int, cache_dir: Path | None
) -> list[float]:
    durations: list[float] = []
    for _ in range(repeats):
        model = MLP(in_dim=2, hidden_layers=3, width=32, out_dim=1)
        pinn = ContinuousPINN(
            model=model,
            device="cpu",
            pde_residual_fn=burgers_residual,
            cfg_burgers=BurgersConfig(),
        )

        cache_config = None
        if use_cache:
            cache_config = CacheConfig(
                enabled=True,
                identifier="benchmark",
                persistent_path=cache_dir,
                store_gradients=True,
                store_losses=False,
                store_activations=True,
                async_precompute=True,
                precompute_workers=2,
            )

        config = TrainConfig(
            n_u0=16,
            n_bc=16,
            n_f=256,
            lr=5e-4,
            adam_steps=steps,
            lbfgs_max_iter=0,
            print_frequency=max(steps, 1),
            track_losses=False,
            cache_config=cache_config,
        )

        start = time.perf_counter()
        pinn.train(config, weights=(1.0, 1.0, 1.0))
        durations.append(time.perf_counter() - start)
    return durations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps", type=int, default=25, help="Training steps per trial"
    )
    parser.add_argument(
        "--repeats", type=int, default=3, help="Number of trials to average"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional directory used for persistent cache storage",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    uncached = run_training(False, args.repeats, args.steps, cache_dir)
    cached = run_training(True, args.repeats, args.steps, cache_dir)

    baseline = mean(uncached)
    improved = mean(cached)
    speedup = baseline / improved if improved else float("inf")

    print("CACHING BENCHMARK")
    print("=================")
    print(f"Steps per run : {args.steps}")
    print(f"Repeats       : {args.repeats}")
    print(f"No cache time : {baseline:.3f}s (avg)")
    print(f"Cache time    : {improved:.3f}s (avg)")
    print(f"Speed-up      : {speedup:.2f}x")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()

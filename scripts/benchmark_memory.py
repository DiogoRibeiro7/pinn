#!/usr/bin/env python
"""Benchmark memory usage for the Burgers PINN with memory-efficient training."""

import json
from pathlib import Path

from pinnlab.models.mlp import MLP
from pinnlab.solvers.raissi_improved import BurgersConfig, TrainConfig, burgers_residual
from pinnlab.training.memory_efficient import MemoryEfficientPINN, MemoryProfiler


def main() -> None:
    cfg = BurgersConfig()
    model = MLP(in_dim=2, hidden_layers=2, width=20, out_dim=1)
    profiler = MemoryProfiler()
    pinn = MemoryEfficientPINN(
        model=model,
        device="cpu",
        pde_residual_fn=burgers_residual,
        cfg_burgers=cfg,
        profiler=profiler,
        memory_budget_mb=512,
    )

    config = TrainConfig(
        adam_steps=5, lbfgs_max_iter=0, track_losses=False, n_u0=10, n_bc=10, n_f=200
    )
    pinn.train(config)

    summary = profiler.summary()
    recommendations = profiler.recommendations(512 * 1024**2)

    payload = {"summary": summary, "recommendations": recommendations}
    path = Path("memory_benchmark.json")
    path.write_text(json.dumps(payload, indent=2))
    print(f"Memory benchmark written to {path}")


if __name__ == "__main__":
    main()

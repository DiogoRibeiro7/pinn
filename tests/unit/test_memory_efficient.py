"""Tests for memory-aware training utilities."""

import torch

from pinnkit.solvers.raissi_improved import BurgersConfig, TrainConfig, burgers_residual
from pinnkit.training.memory_efficient import MemoryEfficientPINN, MemoryProfiler
from tests.fixtures.models import simple_mlp


def test_memory_profiler_records_cpu_history():
    profiler = MemoryProfiler(torch.device("cpu"))
    with profiler.track("unit", step=0):
        torch.zeros(4)
    summary = profiler.summary()
    assert summary["records"] == 1.0
    recommendations = profiler.recommendations(None)
    assert recommendations  # should offer at least one message


def test_memory_efficient_pinn_estimates_batch_and_profiles():
    model = simple_mlp()
    profiler = MemoryProfiler(torch.device("cpu"))
    pinn = MemoryEfficientPINN(
        model,
        "cpu",
        burgers_residual,
        BurgersConfig(),
        profiler=profiler,
    )
    config = TrainConfig(
        n_u0=4,
        n_bc=4,
        n_f=64,
        adam_steps=2,
        lbfgs_max_iter=0,
        track_losses=False,
    )

    data = pinn._make_training_points(config)
    batch_size = pinn._estimate_batch_size(config, data)
    assert 1 <= batch_size <= data["tf"].shape[0]

    history = pinn.train(config)
    assert isinstance(history, dict)
    assert profiler.history  # profiling should record at least one step

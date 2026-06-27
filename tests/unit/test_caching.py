"""Tests for caching and precomputation helpers."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import torch

from pinn.optimization.caching import AdaptiveCache, CacheConfig, TrainingCache
from pinn.solvers.raissi_improved import (
    BurgersConfig,
    ContinuousPINN,
    TrainConfig,
    burgers_residual,
)
from pinn.models.mlp import MLP


def test_adaptive_cache_eviction_and_persistence(tmp_path: Path) -> None:
    cache = AdaptiveCache(max_memory_mb=1, compression=False)
    cache.put("small", torch.zeros(32, dtype=torch.float32))
    large_tensor = torch.zeros(512, 512, dtype=torch.float32)
    cache.put("large", large_tensor)

    assert "large" in cache
    assert "small" not in cache

    cache_path = tmp_path / "cache.pkl"
    cache.save(cache_path)

    restored = AdaptiveCache(max_memory_mb=1, compression=False)
    restored.load(cache_path)
    loaded = restored.get("large")
    assert loaded is not None
    assert torch.equal(loaded, large_tensor)


def test_training_cache_prepare_points(tmp_path: Path) -> None:
    config = CacheConfig(enabled=True, identifier="unit", persistent_path=tmp_path)
    cache = TrainingCache(config)
    calls = []
    context = {"solver": "test", "cfg": {}}
    tcfg = SimpleNamespace(n_u0=1, n_bc=1, n_f=2, seed=42)

    def factory() -> dict[str, torch.Tensor]:
        calls.append(1)
        return {
            "tf": torch.zeros(2, 1),
            "xf": torch.ones(2, 1),
        }

    first = cache.prepare_training_points(tcfg, factory, device=torch.device("cpu"), context=context)
    second = cache.prepare_training_points(tcfg, factory, device=torch.device("cpu"), context=context)

    assert len(calls) == 1
    assert torch.equal(first["tf"], second["tf"])

    cached_files = list((tmp_path / "unit").glob("*_samples.pt"))
    assert cached_files, "expected cached samples to be persisted"


def test_training_cache_gradient_versioning() -> None:
    config = CacheConfig(enabled=True, store_gradients=True)
    cache = TrainingCache(config)
    t = torch.linspace(0, 1, steps=4).reshape(-1, 1)
    x = torch.linspace(-1, 1, steps=4).reshape(-1, 1)
    key = cache.coordinates_key(t, x)

    value = torch.arange(4, dtype=torch.float32).reshape(-1, 1)
    cache.store_gradient(key, value)

    cached = cache.get_gradient(key)
    assert cached is not None
    assert torch.equal(cached, value)

    cache.advance_model_version()
    assert cache.get_gradient(key) is None


def test_continuous_pinn_training_with_caching(tmp_path: Path) -> None:
    model = MLP(in_dim=2, hidden_layers=2, width=8, out_dim=1)
    pinn = ContinuousPINN(
        model=model,
        device="cpu",
        pde_residual_fn=burgers_residual,
        cfg_burgers=BurgersConfig(),
    )

    cache_config = CacheConfig(
        enabled=True,
        identifier="integration",
        persistent_path=tmp_path,
        store_gradients=True,
        store_losses=True,
        store_activations=True,
        async_precompute=False,
    )

    tcfg = TrainConfig(
        n_u0=4,
        n_bc=4,
        n_f=16,
        lr=1e-3,
        adam_steps=2,
        lbfgs_max_iter=0,
        print_frequency=10,
        track_losses=False,
        cache_config=cache_config,
    )

    history = pinn.train(tcfg, weights=(1.0, 1.0, 1.0))
    assert isinstance(history, dict)

    cache_dir = tmp_path / "integration"
    assert cache_dir.exists()
    assert list(cache_dir.glob("*_samples.pt")), "expected cached samples on disk"

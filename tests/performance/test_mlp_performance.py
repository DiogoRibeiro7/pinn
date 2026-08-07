"""Performance benchmarks for key components."""

from __future__ import annotations

import time

import pytest
import torch

from tests.fixtures import simple_mlp


@pytest.mark.performance
def test_mlp_forward_speed(device):
    """Ensure that a forward pass runs within an acceptable time budget."""

    model = simple_mlp().to(device)
    x = torch.randn(1000, 2, device=device)

    start = time.perf_counter()
    for _ in range(100):
        model(x)
    duration = time.perf_counter() - start

    # Threshold generous for heterogeneous environments
    assert duration < 1.0

"""Global pytest fixtures for the PINN test suite."""

from __future__ import annotations

import pytest
import torch
from hypothesis import HealthCheck, settings

# Hypothesis' ``too_slow`` health check and per-example deadline both measure
# wall-clock time while inputs are generated, so they report on the machine
# rather than on the code under test. A shared CI runner, a cold import of
# torch, or Hypothesis' own constant-harvesting cache can stall an otherwise
# trivial draw for tens of seconds and fail the run. Drop the timing-based
# checks and keep every correctness-based one.
settings.register_profile(
    "pinn",
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("pinn")


@pytest.fixture(params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def device(request: pytest.FixtureRequest) -> torch.device:
    """Return a torch device for running tests.

    Tests depending on this fixture will run on both CPU and GPU (if
    available), ensuring device-agnostic behavior across the codebase.
    """

    return torch.device(request.param)


__all__ = ["device"]

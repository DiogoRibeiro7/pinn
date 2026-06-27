"""Global pytest fixtures for the PINN test suite."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def device(request: pytest.FixtureRequest) -> torch.device:
    """Return a torch device for running tests.

    Tests depending on this fixture will run on both CPU and GPU (if
    available), ensuring device-agnostic behavior across the codebase.
    """

    return torch.device(request.param)


__all__ = ["device"]


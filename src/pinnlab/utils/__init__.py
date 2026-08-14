"""Utility modules for PINNs.

``visualization`` is imported lazily. It pulls in ``matplotlib.pyplot``, which
costs about 1.1s of a 5.5s ``import pinnlab`` and is not needed by anyone who
only wants to solve a PDE. Importing it here eagerly was also what forced
matplotlib to be a required dependency rather than part of the ``viz`` extra.

Access is unchanged: ``pinnlab.utils.visualization`` and
``from pinnlab.utils import visualization`` both work and import the module on
first use. Only the timing moves.
"""

import importlib
from typing import Any

from . import (
    checkpointing,
    error_handling,
    logging,
    metrics,
    profiling,
    sampling,
    validation,
)

from .checkpointing import CheckpointManager
from .profiling import MemoryTracker, PerformanceReport, profile_performance

#: Submodules imported on first access rather than at package import.
_LAZY_SUBMODULES = frozenset({"visualization"})

__all__ = [
    "sampling",
    "metrics",
    "visualization",
    "error_handling",
    "logging",
    "validation",
    "checkpointing",
    "profiling",
    "CheckpointManager",
    "profile_performance",
    "MemoryTracker",
    "PerformanceReport",
]


def __getattr__(name: str) -> Any:
    """Import a deferred submodule on first attribute access (PEP 562)."""
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f".{name}", __name__)
        # Cache it so later lookups skip this function entirely.
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    """Include the deferred submodules in ``dir()`` without importing them."""
    return sorted(set(globals()) | _LAZY_SUBMODULES)

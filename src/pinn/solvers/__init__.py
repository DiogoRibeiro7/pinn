"""PDE solver implementations for PINNs."""

from . import raissi, raissi_improved, raissi_generic, navier_stokes

__all__ = [
    "raissi",
    "raissi_improved",
    "raissi_generic",
    "navier_stokes",
]

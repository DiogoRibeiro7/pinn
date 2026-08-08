"""PDE solver implementations for PINNs."""

from . import (
    heat,
    navier_stokes,
    raissi,
    raissi_generic,
    raissi_improved,
    reference,
    wave,
)
from ._base import Domain1D, ExactlySolvablePINN, TrainConfig
from .heat import HeatConfig, HeatPINN, heat_exact
from .reference import burgers_cole_hopf, relative_l2_error
from .wave import WaveConfig, WavePINN, wave_exact

__all__ = [
    "raissi",
    "raissi_improved",
    "raissi_generic",
    "navier_stokes",
    "heat",
    "wave",
    "reference",
    "Domain1D",
    "ExactlySolvablePINN",
    "TrainConfig",
    "HeatConfig",
    "HeatPINN",
    "heat_exact",
    "WaveConfig",
    "WavePINN",
    "wave_exact",
    "burgers_cole_hopf",
    "relative_l2_error",
]

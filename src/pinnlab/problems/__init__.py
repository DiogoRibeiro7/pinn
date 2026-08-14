"""PDE problem definitions for composable PINN training."""

from .allen_cahn import AllenCahnConfig, AllenCahnProblem, benchmark_initial_profile
from .base import PDEProblem
from .burgers import BurgersProblem
from .heat import HeatProblem
from .navier_stokes import (
    NavierStokesConfig,
    NavierStokesProblem,
    taylor_green,
)
from .wave import WaveProblem

__all__ = [
    "AllenCahnConfig",
    "AllenCahnProblem",
    "BurgersProblem",
    "HeatProblem",
    "NavierStokesConfig",
    "NavierStokesProblem",
    "taylor_green",
    "PDEProblem",
    "WaveProblem",
    "benchmark_initial_profile",
]

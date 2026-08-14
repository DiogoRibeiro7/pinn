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
from .schrodinger import (
    SOLITON_AMPLITUDE,
    SchrodingerConfig,
    SchrodingerProblem,
    soliton,
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
    "SOLITON_AMPLITUDE",
    "SchrodingerConfig",
    "SchrodingerProblem",
    "soliton",
    "WaveProblem",
    "benchmark_initial_profile",
]

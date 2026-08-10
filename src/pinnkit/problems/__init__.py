"""PDE problem definitions for composable PINN training."""

from .base import PDEProblem
from .burgers import BurgersProblem
from .heat import HeatProblem
from .wave import WaveProblem

__all__ = ["PDEProblem", "BurgersProblem", "HeatProblem", "WaveProblem"]

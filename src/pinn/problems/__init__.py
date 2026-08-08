"""PDE problem definitions for composable PINN training."""

from .base import PDEProblem
from .heat import HeatProblem
from .wave import WaveProblem

__all__ = ["PDEProblem", "HeatProblem", "WaveProblem"]

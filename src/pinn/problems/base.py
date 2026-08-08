"""Problem contracts for composable PINN training."""

from __future__ import annotations

from typing import Protocol, Sequence

from torch import Tensor, nn

from ..constraints import Constraint
from ..geometry import Geometry
from ..residuals import DerivativeBackend


class PDEProblem(Protocol):
    """Protocol for a PDE problem solved by a generic trainer."""

    @property
    def input_dim(self) -> int:
        """Return the number of input coordinate dimensions."""

    @property
    def output_dim(self) -> int:
        """Return the number of model output components."""

    @property
    def geometry(self) -> Geometry:
        """Return the problem geometry."""

    def strong_residual(
        self,
        model: nn.Module,
        coordinates: Tensor,
        derivative_backend: DerivativeBackend,
    ) -> Tensor:
        """Evaluate the strong-form PDE residual at coordinates."""

    def constraints(self) -> Sequence[Constraint]:
        """Return the constraints used by generic training."""

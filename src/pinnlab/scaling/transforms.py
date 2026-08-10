"""Model transforms for nondimensional PINN training."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from .nondimensional import Nondimensionalizer


@dataclass(frozen=True)
class ModelInputTransform:
    """Transform dimensional model coordinates into dimensionless inputs."""

    nondimensionalizer: Nondimensionalizer

    def __call__(self, coordinates: Tensor) -> Tensor:
        """Return dimensionless model inputs."""
        return self.nondimensionalizer.to_dimensionless_inputs(coordinates)


@dataclass(frozen=True)
class ModelOutputTransform:
    """Transform dimensionless model outputs into dimensional values."""

    nondimensionalizer: Nondimensionalizer

    def __call__(self, values_star: Tensor) -> Tensor:
        """Return dimensional model outputs."""
        return self.nondimensionalizer.to_dimensional_outputs(values_star)


class ScaledModel(nn.Module):
    """Wrap a model that operates in dimensionless coordinates and outputs.

    The wrapper presents a dimensional interface: callers pass dimensional
    coordinates and receive dimensional outputs. Autograd through this wrapper
    applies the coordinate and output scale factors automatically.
    """

    def __init__(
        self, model: nn.Module, nondimensionalizer: Nondimensionalizer
    ) -> None:
        """Store the raw dimensionless model and transforms."""
        super().__init__()
        self.model = model
        self.input_transform = ModelInputTransform(nondimensionalizer)
        self.output_transform = ModelOutputTransform(nondimensionalizer)

    def forward(self, coordinates: Tensor) -> Tensor:
        """Evaluate the wrapped model with dimensional coordinates."""
        values_star = self.model(self.input_transform(coordinates))
        return self.output_transform(values_star)

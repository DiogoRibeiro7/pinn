"""One-dimensional interval geometry."""

from __future__ import annotations

from dataclasses import dataclass

from .rectangle import Rectangle


@dataclass(frozen=True)
class Interval(Rectangle):
    """Closed interval ``[lower, upper]``."""

    def __init__(self, lower: float, upper: float, name: str = "x") -> None:
        """Create a one-dimensional interval."""
        super().__init__((lower,), (upper,), (name,))

    @property
    def lower_value(self) -> float:
        """Return the scalar lower bound."""
        return self.lower[0]

    @property
    def upper_value(self) -> float:
        """Return the scalar upper bound."""
        return self.upper[0]

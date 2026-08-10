"""Geometry primitives for PINN domains."""

from .base import Geometry
from .interval import Interval
from .rectangle import Rectangle

__all__ = ["Geometry", "Interval", "Rectangle"]

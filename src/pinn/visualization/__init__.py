"""Visualization utilities and dashboards for PINNs."""

from .interactive import TrainingDashboard, export_animation
from ..utils.visualization import PINNVisualizer, quick_plot_1d, quick_plot_loss

__all__ = [
    "TrainingDashboard",
    "export_animation",
    "PINNVisualizer",
    "quick_plot_1d",
    "quick_plot_loss",
]

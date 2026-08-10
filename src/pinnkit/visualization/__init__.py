"""Visualization utilities and dashboards for PINNs."""

try:
    from .interactive import TrainingDashboard, export_animation
except Exception:  # pragma: no cover - optional dependency may be missing
    TrainingDashboard = None  # type: ignore[assignment]
    export_animation = None  # type: ignore[assignment]

try:
    from ..utils.visualization import PINNVisualizer, quick_plot_1d, quick_plot_loss
except Exception:  # pragma: no cover - optional dependency may be missing
    PINNVisualizer = None  # type: ignore[assignment]
    quick_plot_1d = None  # type: ignore[assignment]
    quick_plot_loss = None  # type: ignore[assignment]

__all__ = [
    "TrainingDashboard",
    "export_animation",
    "PINNVisualizer",
    "quick_plot_1d",
    "quick_plot_loss",
]

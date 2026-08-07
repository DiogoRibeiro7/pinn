"""Interactive visualization and real-time monitoring tools.

This module provides an optional :class:`TrainingDashboard` based on Plotly that
can be attached to PINN training loops.  The dashboard collects loss values and
resource usage statistics and can be served as a small web application using
Dash when the dependency is available.  On environments without Dash the class
falls back to ``plotly.graph_objects.Figure`` widgets that work in Jupyter
notebooks or create static HTML files.

The implementation keeps dependencies optional so that unit tests can exercise
basic behaviour without requiring a running event loop or a graphical backend.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Any

import plotly.graph_objects as go

try:  # Optional dependencies used when available
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore


@dataclass
class _MetricHistory:
    steps: Deque[int]
    losses: Deque[float]

    def append(self, step: int, loss: float, maxlen: int = 1000) -> None:
        if len(self.steps) >= maxlen:
            self.steps.popleft()
            self.losses.popleft()
        self.steps.append(step)
        self.losses.append(loss)


class TrainingDashboard:
    """Simple real-time training dashboard.

    Parameters
    ----------
    config:
        Arbitrary configuration object.  Stored for display and export.
    update_frequency:
        Number of training steps between UI updates.
    max_points:
        Maximum number of data points kept in history (to avoid memory leaks
        during long runs).
    """

    def __init__(
        self,
        config: Any,
        update_frequency: int = 10,
        max_points: int = 1000,
    ) -> None:
        self.config = config
        self.update_frequency = update_frequency
        self.history = _MetricHistory(deque(), deque())
        self.max_points = max_points

        # Pre-create figure for loss curve. ``FigureWidget`` offers richer
        # interactivity but requires the optional ``anywidget`` dependency.  To
        # keep the runtime lightweight we fall back to a regular ``Figure``
        # which still supports interactive rendering in notebooks and Dash.
        self.loss_fig = go.Figure(
            data=[go.Scatter(x=[], y=[], mode="lines", name="loss")],
            layout={
                "title": "Training loss",
                "xaxis": {"title": "step"},
                "yaxis": {"title": "loss"},
            },
        )

    # ------------------------------------------------------------------ update
    def update(self, step: int, metrics: Dict[str, float]) -> None:
        """Record metrics and update figures."""

        loss = float(metrics.get("loss") or metrics.get("total") or 0.0)
        self.history.append(step, loss, maxlen=self.max_points)
        self.loss_fig.data[0].x = list(self.history.steps)
        self.loss_fig.data[0].y = list(self.history.losses)

        # Update resource information if psutil is available
        if psutil is not None:
            self.cpu_percent = psutil.cpu_percent()  # type: ignore[attr-defined]
            mem = psutil.virtual_memory()  # type: ignore[attr-defined]
            self.mem_percent = getattr(mem, "percent", 0.0)

    def update_callback(self, step: int, metrics: Dict[str, float]) -> None:
        """Callback compatible with solver training loops."""
        if step % self.update_frequency == 0:
            self.update(step, metrics)

    # -------------------------------------------------------------------- serve
    def serve(self, port: int = 8050) -> None:  # pragma: no cover - requires UI
        """Serve dashboard using Dash when available.

        When Dash is not installed the loss curve is simply displayed using the
        built-in Plotly renderer.  The method is purposely light-weight so that
        it can be invoked from notebooks without spawning background threads.
        """

        try:
            import dash
            from dash import dcc, html
        except Exception:  # Dash not available, fall back to showing figure
            self.loss_fig.show()
            return

        app = dash.Dash(__name__)
        app.layout = html.Div(
            [
                dcc.Graph(id="loss", figure=self.loss_fig),
                dcc.Interval(id="interval", interval=1000, n_intervals=0),
            ]
        )

        # Dash callbacks update the graph from stored history
        @app.callback(
            dash.dependencies.Output("loss", "figure"),
            [dash.dependencies.Input("interval", "n_intervals")],
        )
        def _refresh(_: int) -> go.Figure:
            return self.loss_fig

        app.run_server(port=port, debug=False)


def export_animation(fig: go.Figure, path: str) -> None:
    """Export ``fig`` with animation frames to ``path``.

    The function relies on Plotly's built-in ``write_html`` method so that no
    additional video encoder dependencies are required.
    """

    fig.write_html(path)

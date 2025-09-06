"""
Visualization utilities for Physics-Informed Neural Networks (PINNs).

This module provides comprehensive plotting functions for:
- Solution fields (1D/2D)
- Training loss curves
- PDE residuals
- Comparison with analytical solutions
- Error analysis
"""

from __future__ import annotations

from typing import Optional, Tuple, List, Dict, Any, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns

# Set up plotting style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class PINNVisualizer:
    """
    Comprehensive visualization toolkit for PINN results.

    Provides methods for plotting solution fields, loss curves, residuals,
    and performing error analysis for Physics-Informed Neural Networks.
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100) -> None:
        """
        Initialize the visualizer with default figure settings.

        Args:
            figsize: Default figure size (width, height) in inches
            dpi: Figure resolution in dots per inch
        """
        self.figsize = figsize
        self.dpi = dpi

    def plot_1d_solution(
        self,
        x: np.ndarray,
        u_pred: np.ndarray,
        u_true: Optional[np.ndarray] = None,
        title: str = "1D Solution",
        xlabel: str = "x",
        ylabel: str = "u(x)",
        save_path: Optional[str] = None,
        show_grid: bool = True,
    ) -> Figure:
        """
        Plot 1D solution field with optional comparison to analytical solution.

        Args:
            x: Spatial coordinates, shape (N,)
            u_pred: Predicted solution, shape (N,)
            u_true: True/analytical solution, shape (N,), optional
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Path to save figure, optional
            show_grid: Whether to show grid

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Plot predicted solution
        ax.plot(x, u_pred, "b-", linewidth=2, label="PINN Prediction", alpha=0.8)

        # Plot true solution if available
        if u_true is not None:
            ax.plot(
                x, u_true, "r--", linewidth=2, label="Analytical Solution", alpha=0.8
            )

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)

        if show_grid:
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_2d_field(
        self,
        x: np.ndarray,
        y: np.ndarray,
        field: np.ndarray,
        title: str = "2D Field",
        xlabel: str = "x",
        ylabel: str = "y",
        colorbar_label: str = "Field Value",
        cmap: str = "RdBu_r",
        save_path: Optional[str] = None,
        levels: int = 20,
    ) -> Figure:
        """
        Plot 2D scalar field using contour plot.

        Args:
            x: X coordinates, shape (Nx, Ny)
            y: Y coordinates, shape (Nx, Ny)
            field: Field values, shape (Nx, Ny)
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            colorbar_label: Colorbar label
            cmap: Colormap name
            save_path: Path to save figure, optional
            levels: Number of contour levels

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Create contour plot
        cs = ax.contourf(x, y, field, levels=levels, cmap=cmap, extend="both")

        # Add contour lines
        ax.contour(x, y, field, levels=levels, colors="k", alpha=0.3, linewidths=0.5)

        # Colorbar
        cbar = plt.colorbar(cs, ax=ax, shrink=0.8)
        cbar.set_label(colorbar_label, fontsize=11)

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_aspect("equal", adjustable="box")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_vector_field_2d(
        self,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        title: str = "2D Vector Field",
        xlabel: str = "x",
        ylabel: str = "y",
        save_path: Optional[str] = None,
        skip: int = 4,
        scale: float = 1.0,
    ) -> Figure:
        """
        Plot 2D vector field (e.g., velocity field for Navier-Stokes).

        Args:
            x: X coordinates, shape (Nx, Ny)
            y: Y coordinates, shape (Nx, Ny)
            u: X-component of vector field, shape (Nx, Ny)
            v: Y-component of vector field, shape (Nx, Ny)
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Path to save figure, optional
            skip: Skip every nth point for cleaner visualization
            scale: Arrow scaling factor

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Subsample for cleaner visualization
        x_sub = x[::skip, ::skip]
        y_sub = y[::skip, ::skip]
        u_sub = u[::skip, ::skip]
        v_sub = v[::skip, ::skip]

        # Vector magnitude for coloring
        magnitude = np.sqrt(u_sub**2 + v_sub**2)

        # Create quiver plot
        q = ax.quiver(
            x_sub,
            y_sub,
            u_sub,
            v_sub,
            magnitude,
            scale=scale,
            cmap="viridis",
            alpha=0.8,
        )

        # Colorbar
        cbar = plt.colorbar(q, ax=ax, shrink=0.8)
        cbar.set_label("Magnitude", fontsize=11)

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_aspect("equal", adjustable="box")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_spacetime_1d(
        self,
        t: np.ndarray,
        x: np.ndarray,
        u: np.ndarray,
        title: str = "Spacetime Evolution",
        xlabel: str = "x",
        ylabel: str = "t",
        colorbar_label: str = "u(t,x)",
        cmap: str = "RdBu_r",
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Plot spacetime evolution of 1D solution.

        Args:
            t: Time coordinates, shape (Nt, Nx)
            x: Spatial coordinates, shape (Nt, Nx)
            u: Solution values, shape (Nt, Nx)
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            colorbar_label: Colorbar label
            cmap: Colormap name
            save_path: Path to save figure, optional

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Create heatmap
        im = ax.imshow(
            u,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            extent=[x.min(), x.max(), t.min(), t.max()],
        )

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(colorbar_label, fontsize=11)

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_loss_curves(
        self,
        loss_history: Dict[str, List[float]],
        title: str = "Training Loss History",
        log_scale: bool = True,
        save_path: Optional[str] = None,
        smoothing_window: int = 50,
    ) -> Figure:
        """
        Plot training loss curves with optional smoothing.

        Args:
            loss_history: Dictionary with loss component names as keys and
                         loss values as lists
            title: Plot title
            log_scale: Whether to use log scale for y-axis
            save_path: Path to save figure, optional
            smoothing_window: Window size for moving average smoothing

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        colors = plt.cm.tab10(np.linspace(0, 1, len(loss_history)))

        for i, (loss_name, loss_values) in enumerate(loss_history.items()):
            steps = np.arange(len(loss_values))

            # Plot raw data with transparency
            ax.plot(steps, loss_values, alpha=0.3, color=colors[i])

            # Plot smoothed curve if enough data points
            if len(loss_values) > smoothing_window:
                smoothed = self._moving_average(loss_values, smoothing_window)
                smooth_steps = steps[smoothing_window - 1 :]
                ax.plot(
                    smooth_steps,
                    smoothed,
                    linewidth=2,
                    label=f"{loss_name} (smoothed)",
                    color=colors[i],
                )
            else:
                ax.plot(
                    steps, loss_values, linewidth=2, label=loss_name, color=colors[i]
                )

        if log_scale:
            ax.set_yscale("log")

        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("Loss Value", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_residuals(
        self,
        x: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
        residuals: np.ndarray,
        residual_names: Optional[List[str]] = None,
        title: str = "PDE Residuals",
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Plot PDE residuals at collocation points.

        Args:
            x: Coordinates - for 1D: shape (N,), for 2D: tuple of (X, Y) arrays
            residuals: Residual values, shape (N, n_residuals)
            residual_names: Names of residual components
            title: Plot title
            save_path: Path to save figure, optional

        Returns:
            matplotlib Figure object
        """
        if residual_names is None:
            residual_names = [f"Residual {i + 1}" for i in range(residuals.shape[1])]

        n_residuals = residuals.shape[1]

        # Determine if 1D or 2D
        is_2d = isinstance(x, tuple) and len(x) == 2

        if is_2d:
            # 2D residuals - use scatter plots
            fig, axes = plt.subplots(
                1, n_residuals, figsize=(5 * n_residuals, 5), dpi=self.dpi
            )
            if n_residuals == 1:
                axes = [axes]

            X, Y = x
            for i, (ax, name) in enumerate(zip(axes, residual_names)):
                scatter = ax.scatter(
                    X.ravel(),
                    Y.ravel(),
                    c=residuals[:, i],
                    cmap="RdBu_r",
                    s=10,
                    alpha=0.7,
                )
                ax.set_xlabel("x", fontsize=11)
                ax.set_ylabel("y", fontsize=11)
                ax.set_title(name, fontsize=12)
                plt.colorbar(scatter, ax=ax, shrink=0.8)
                ax.set_aspect("equal", adjustable="box")
        else:
            # 1D residuals - use line plots
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

            for i, name in enumerate(residual_names):
                ax.plot(
                    x,
                    residuals[:, i],
                    "o-",
                    alpha=0.7,
                    markersize=3,
                    linewidth=1,
                    label=name,
                )

            ax.set_xlabel("x", fontsize=12)
            ax.set_ylabel("Residual Value", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_error_analysis(
        self,
        x: np.ndarray,
        u_pred: np.ndarray,
        u_true: np.ndarray,
        title: str = "Error Analysis",
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Comprehensive error analysis comparing predicted vs true solutions.

        Args:
            x: Spatial coordinates
            u_pred: Predicted solution
            u_true: True/analytical solution
            title: Plot title
            save_path: Path to save figure, optional

        Returns:
            matplotlib Figure object
        """
        # Calculate errors
        abs_error = np.abs(u_pred - u_true)
        rel_error = abs_error / (np.abs(u_true) + 1e-10)

        # Statistics
        mae = np.mean(abs_error)
        rmse = np.sqrt(np.mean((u_pred - u_true) ** 2))
        max_error = np.max(abs_error)

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=self.dpi)

        # Solution comparison
        axes[0, 0].plot(x, u_true, "r-", linewidth=2, label="True", alpha=0.8)
        axes[0, 0].plot(x, u_pred, "b--", linewidth=2, label="Predicted", alpha=0.8)
        axes[0, 0].set_xlabel("x")
        axes[0, 0].set_ylabel("u(x)")
        axes[0, 0].set_title("Solution Comparison")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Absolute error
        axes[0, 1].plot(x, abs_error, "g-", linewidth=2)
        axes[0, 1].set_xlabel("x")
        axes[0, 1].set_ylabel("|u_pred - u_true|")
        axes[0, 1].set_title("Absolute Error")
        axes[0, 1].grid(True, alpha=0.3)

        # Relative error
        axes[1, 0].plot(x, rel_error, "m-", linewidth=2)
        axes[1, 0].set_xlabel("x")
        axes[1, 0].set_ylabel("Relative Error")
        axes[1, 0].set_title("Relative Error")
        axes[1, 0].grid(True, alpha=0.3)

        # Error histogram
        axes[1, 1].hist(abs_error, bins=30, alpha=0.7, density=True, color="skyblue")
        axes[1, 1].set_xlabel("Absolute Error")
        axes[1, 1].set_ylabel("Density")
        axes[1, 1].set_title("Error Distribution")
        axes[1, 1].grid(True, alpha=0.3)

        # Add error statistics as text
        stats_text = f"MAE: {mae:.2e}\nRMSE: {rmse:.2e}\nMax Error: {max_error:.2e}"
        axes[1, 1].text(
            0.05,
            0.95,
            stats_text,
            transform=axes[1, 1].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        fig.suptitle(title, fontsize=16, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def create_comparison_plot(
        self,
        results: Dict[str, Dict[str, np.ndarray]],
        x: np.ndarray,
        title: str = "Method Comparison",
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Compare multiple PINN results or methods.

        Args:
            results: Dictionary with method names as keys and solution data as values
                    Each value should be a dict with 'solution' key
            x: Spatial coordinates
            title: Plot title
            save_path: Path to save figure, optional

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

        for i, (method_name, data) in enumerate(results.items()):
            solution = data["solution"]
            ax.plot(
                x, solution, linewidth=2, label=method_name, color=colors[i], alpha=0.8
            )

        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("u(x)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    @staticmethod
    def _moving_average(data: List[float], window: int) -> np.ndarray:
        """Calculate moving average for smoothing loss curves."""
        return np.convolve(data, np.ones(window) / window, mode="valid")


# Convenience functions for quick plotting
def quick_plot_1d(
    x: np.ndarray,
    u_pred: np.ndarray,
    u_true: Optional[np.ndarray] = None,
    title: str = "1D Solution",
    save_path: Optional[str] = None,
) -> Figure:
    """Quick 1D plotting function."""
    viz = PINNVisualizer()
    return viz.plot_1d_solution(x, u_pred, u_true, title, save_path=save_path)


def quick_plot_2d(
    x: np.ndarray,
    y: np.ndarray,
    field: np.ndarray,
    title: str = "2D Field",
    save_path: Optional[str] = None,
) -> Figure:
    """Quick 2D plotting function."""
    viz = PINNVisualizer()
    return viz.plot_2d_field(x, y, field, title, save_path=save_path)


def quick_plot_loss(
    loss_history: Dict[str, List[float]],
    title: str = "Loss History",
    save_path: Optional[str] = None,
) -> Figure:
    """Quick loss plotting function."""
    viz = PINNVisualizer()
    return viz.plot_loss_curves(loss_history, title, save_path=save_path)

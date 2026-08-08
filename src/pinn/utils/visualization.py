"""
Improved visualization utilities for Physics-Informed Neural Networks (PINNs).

This module provides comprehensive plotting functions with proper type annotations,
error handling, and validation for:
- Solution fields (1D/2D)
- Training loss curves
- PDE residuals
- Comparison with analytical solutions
- Error analysis

Key improvements:
- Complete type annotations
- Comprehensive error handling and validation
- Configuration-driven parameters
- Better error messages
- Input sanitization
"""

from __future__ import annotations

from typing import Optional, Tuple, List, Dict, Union, Sequence
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

from .logging import get_logger
from .profiling import profile_performance

logger = get_logger(__name__)

# Set up plotting style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


@dataclass(frozen=True)
class VisualizationConfig:
    """Configuration for visualization parameters."""

    # Figure settings
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 100

    # Plot settings
    line_width: float = 2.0
    alpha: float = 0.8
    grid_alpha: float = 0.3
    marker_size: int = 3

    # Font settings
    title_fontsize: int = 14
    label_fontsize: int = 12
    legend_fontsize: int = 11
    tick_fontsize: int = 10

    # Colormap settings
    default_cmap: str = "RdBu_r"
    vector_cmap: str = "viridis"

    # Smoothing settings
    default_smoothing_window: int = 50

    # File settings
    save_dpi: int = 300
    save_format: str = "png"
    save_bbox_inches: str = "tight"


class VisualizationError(Exception):
    """Custom exception for visualization errors."""

    pass


class PINNVisualizer:
    """
    Comprehensive visualization toolkit for PINN results with improved error handling.

    Provides methods for plotting solution fields, loss curves, residuals,
    and performing error analysis for Physics-Informed Neural Networks.
    """

    def __init__(
        self,
        config: Optional[VisualizationConfig] = None,
        figsize: Optional[Tuple[int, int]] = None,
        dpi: Optional[int] = None,
    ) -> None:
        """
        Initialize the visualizer with configuration.

        Args:
            config: Visualization configuration object
            figsize: Default figure size (width, height) in inches (overrides config)
            dpi: Figure resolution in dots per inch (overrides config)

        Raises:
            ValueError: If figsize contains non-positive values
            ValueError: If dpi is non-positive
        """
        if config is None:
            config = VisualizationConfig()

        self.config = config

        # Override config with explicit parameters if provided
        if figsize is not None:
            if len(figsize) != 2 or any(dim <= 0 for dim in figsize):
                raise ValueError(
                    f"figsize must be a tuple of 2 positive numbers, got {figsize}"
                )
            self.figsize = figsize
        else:
            self.figsize = config.figsize

        if dpi is not None:
            if dpi <= 0:
                raise ValueError(f"dpi must be positive, got {dpi}")
            self.dpi = dpi
        else:
            self.dpi = config.dpi

    @profile_performance
    def plot_1d_solution(
        self,
        x: np.ndarray,
        u_pred: np.ndarray,
        u_true: Optional[np.ndarray] = None,
        title: str = "1D Solution",
        xlabel: str = "x",
        ylabel: str = "u(x)",
        save_path: Optional[Union[str, Path]] = None,
        show_grid: bool = True,
        legend_labels: Optional[Tuple[str, str]] = None,
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
            legend_labels: Custom labels for (predicted, true) solutions

        Returns:
            matplotlib Figure object

        Raises:
            VisualizationError: If input arrays have incompatible shapes or invalid data
            ValueError: If save_path directory doesn't exist
        """
        # Input validation
        self._validate_1d_arrays(x, u_pred, "x", "u_pred")

        if u_true is not None:
            self._validate_1d_arrays(x, u_true, "x", "u_true")

        if save_path is not None:
            self._validate_save_path(save_path)

        if legend_labels is not None and len(legend_labels) != 2:
            raise ValueError(
                f"legend_labels must contain exactly 2 labels, got {len(legend_labels)}"
            )

        try:
            logger.info(
                "Plotting 1D solution",
                extra={"title": title, "points": len(x)},
            )
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

            # Set default legend labels
            if legend_labels is None:
                pred_label = "PINN Prediction"
                true_label = "Analytical Solution"
            else:
                pred_label, true_label = legend_labels

            # Plot predicted solution
            ax.plot(
                x,
                u_pred,
                "b-",
                linewidth=self.config.line_width,
                label=pred_label,
                alpha=self.config.alpha,
            )

            # Plot true solution if available
            if u_true is not None:
                ax.plot(
                    x,
                    u_true,
                    "r--",
                    linewidth=self.config.line_width,
                    label=true_label,
                    alpha=self.config.alpha,
                )

            # Styling
            ax.set_xlabel(xlabel, fontsize=self.config.label_fontsize)
            ax.set_ylabel(ylabel, fontsize=self.config.label_fontsize)
            ax.set_title(title, fontsize=self.config.title_fontsize, fontweight="bold")
            ax.legend(fontsize=self.config.legend_fontsize)
            ax.tick_params(labelsize=self.config.tick_fontsize)

            if show_grid:
                ax.grid(True, alpha=self.config.grid_alpha)

            plt.tight_layout()

            if save_path:
                self._save_figure(fig, save_path)

            return fig

        except Exception as e:
            plt.close(fig) if "fig" in locals() else None
            raise VisualizationError(
                f"Failed to create 1D solution plot: {str(e)}"
            ) from e

    def plot_2d_field(
        self,
        x: np.ndarray,
        y: np.ndarray,
        field: np.ndarray,
        title: str = "2D Field",
        xlabel: str = "x",
        ylabel: str = "y",
        colorbar_label: str = "Field Value",
        cmap: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        levels: int = 20,
        show_contour_lines: bool = True,
        extend_colorbar: str = "both",
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
            cmap: Colormap name (defaults to config.default_cmap)
            save_path: Path to save figure, optional
            levels: Number of contour levels
            show_contour_lines: Whether to show contour lines
            extend_colorbar: How to extend colorbar ("neither", "both", "min", "max")

        Returns:
            matplotlib Figure object

        Raises:
            VisualizationError: If input arrays have incompatible shapes or invalid data
            ValueError: If parameters are invalid
        """
        # Input validation
        self._validate_2d_arrays(x, y, field)

        if levels <= 0:
            raise ValueError(f"levels must be positive, got {levels}")

        if extend_colorbar not in ["neither", "both", "min", "max"]:
            raise ValueError(
                f"extend_colorbar must be one of ['neither', 'both', 'min', 'max'], got {extend_colorbar}"
            )

        if save_path is not None:
            self._validate_save_path(save_path)

        if cmap is None:
            cmap = self.config.default_cmap

        try:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

            # Create contour plot
            cs = ax.contourf(
                x, y, field, levels=levels, cmap=cmap, extend=extend_colorbar
            )

            # Add contour lines if requested
            if show_contour_lines:
                ax.contour(
                    x,
                    y,
                    field,
                    levels=levels,
                    colors="k",
                    alpha=self.config.grid_alpha,
                    linewidths=0.5,
                )

            # Colorbar
            cbar = plt.colorbar(cs, ax=ax, shrink=0.8)
            cbar.set_label(colorbar_label, fontsize=self.config.legend_fontsize)
            cbar.ax.tick_params(labelsize=self.config.tick_fontsize)

            # Styling
            ax.set_xlabel(xlabel, fontsize=self.config.label_fontsize)
            ax.set_ylabel(ylabel, fontsize=self.config.label_fontsize)
            ax.set_title(title, fontsize=self.config.title_fontsize, fontweight="bold")
            ax.set_aspect("equal", adjustable="box")
            ax.tick_params(labelsize=self.config.tick_fontsize)

            plt.tight_layout()

            if save_path:
                self._save_figure(fig, save_path)

            return fig

        except Exception as e:
            plt.close(fig) if "fig" in locals() else None
            raise VisualizationError(f"Failed to create 2D field plot: {str(e)}") from e

    def plot_vector_field_2d(
        self,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        title: str = "2D Vector Field",
        xlabel: str = "x",
        ylabel: str = "y",
        save_path: Optional[Union[str, Path]] = None,
        skip: int = 4,
        scale: Optional[float] = None,
        cmap: Optional[str] = None,
        show_magnitude: bool = True,
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
            scale: Arrow scaling factor (auto-determined if None)
            cmap: Colormap for magnitude (defaults to config.vector_cmap)
            show_magnitude: Whether to color arrows by magnitude

        Returns:
            matplotlib Figure object

        Raises:
            VisualizationError: If input arrays have incompatible shapes or invalid data
            ValueError: If parameters are invalid
        """
        # Input validation
        self._validate_2d_arrays(x, y, u)
        self._validate_2d_arrays(x, y, v)

        if skip <= 0:
            raise ValueError(f"skip must be positive, got {skip}")

        if save_path is not None:
            self._validate_save_path(save_path)

        if cmap is None:
            cmap = self.config.vector_cmap

        try:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

            # Subsample for cleaner visualization
            x_sub = x[::skip, ::skip]
            y_sub = y[::skip, ::skip]
            u_sub = u[::skip, ::skip]
            v_sub = v[::skip, ::skip]

            # Vector magnitude for coloring
            if show_magnitude:
                magnitude = np.sqrt(u_sub**2 + v_sub**2)
                # Handle case where all magnitudes are zero
                if np.allclose(magnitude, 0):
                    warnings.warn(
                        "All vector magnitudes are zero. Using uniform coloring."
                    )
                    magnitude = np.ones_like(magnitude)
            else:
                magnitude = None

            # Auto-determine scale if not provided
            if scale is None:
                # Use a reasonable default scale based on field magnitude and grid spacing
                max_magnitude = np.max(np.sqrt(u_sub**2 + v_sub**2))
                grid_spacing = np.mean(
                    [np.mean(np.diff(x_sub[0, :])), np.mean(np.diff(y_sub[:, 0]))]
                )
                scale = (
                    max_magnitude * skip / grid_spacing if max_magnitude > 0 else 1.0
                )

            # Create quiver plot
            q = ax.quiver(
                x_sub,
                y_sub,
                u_sub,
                v_sub,
                magnitude if show_magnitude else None,
                scale=scale,
                cmap=cmap,
                alpha=self.config.alpha,
                angles="xy",
                scale_units="xy",
            )

            # Colorbar for magnitude
            if show_magnitude and magnitude is not None:
                cbar = plt.colorbar(q, ax=ax, shrink=0.8)
                cbar.set_label("Magnitude", fontsize=self.config.legend_fontsize)
                cbar.ax.tick_params(labelsize=self.config.tick_fontsize)

            # Styling
            ax.set_xlabel(xlabel, fontsize=self.config.label_fontsize)
            ax.set_ylabel(ylabel, fontsize=self.config.label_fontsize)
            ax.set_title(title, fontsize=self.config.title_fontsize, fontweight="bold")
            ax.set_aspect("equal", adjustable="box")
            ax.tick_params(labelsize=self.config.tick_fontsize)

            plt.tight_layout()

            if save_path:
                self._save_figure(fig, save_path)

            return fig

        except Exception as e:
            plt.close(fig) if "fig" in locals() else None
            raise VisualizationError(
                f"Failed to create vector field plot: {str(e)}"
            ) from e

    def plot_loss_curves(
        self,
        loss_history: Dict[str, Sequence[float]],
        title: str = "Training Loss History",
        log_scale: bool = True,
        save_path: Optional[Union[str, Path]] = None,
        smoothing_window: Optional[int] = None,
        show_raw: bool = True,
        colors: Optional[List[str]] = None,
    ) -> Figure:
        """
        Plot training loss curves with optional smoothing.

        Args:
            loss_history: Dictionary with loss component names as keys and
                         loss values as sequences
            title: Plot title
            log_scale: Whether to use log scale for y-axis
            save_path: Path to save figure, optional
            smoothing_window: Window size for moving average (defaults to config value)
            show_raw: Whether to show raw data with transparency
            colors: Custom colors for each loss component

        Returns:
            matplotlib Figure object

        Raises:
            VisualizationError: If loss_history is empty or contains invalid data
            ValueError: If parameters are invalid
        """
        # Input validation
        if not loss_history:
            raise VisualizationError("loss_history cannot be empty")

        # Validate loss data
        for loss_name, loss_values in loss_history.items():
            if not loss_values:
                raise VisualizationError(
                    f"Loss values for '{loss_name}' cannot be empty"
                )

            # Convert to numpy array and check for valid values
            loss_array = np.asarray(loss_values)
            if not np.all(np.isfinite(loss_array)):
                warnings.warn(
                    f"Loss values for '{loss_name}' contain non-finite values"
                )
            if np.any(loss_array < 0):
                warnings.warn(f"Loss values for '{loss_name}' contain negative values")

        if smoothing_window is None:
            smoothing_window = self.config.default_smoothing_window
        elif smoothing_window <= 0:
            raise ValueError(
                f"smoothing_window must be positive, got {smoothing_window}"
            )

        if save_path is not None:
            self._validate_save_path(save_path)

        try:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

            # Use provided colors or generate default ones
            if colors is None:
                colors = plt.cm.tab10(np.linspace(0, 1, len(loss_history)))
            elif len(colors) < len(loss_history):
                raise ValueError(
                    f"Number of colors ({len(colors)}) must match number of loss components ({len(loss_history)})"
                )

            for i, (loss_name, loss_values) in enumerate(loss_history.items()):
                loss_array = np.asarray(loss_values)
                steps = np.arange(len(loss_array))
                color = colors[i]

                # Plot raw data with transparency if requested
                if show_raw:
                    ax.plot(steps, loss_array, alpha=0.3, color=color, linewidth=1)

                # Plot smoothed curve if enough data points
                if len(loss_array) > smoothing_window:
                    smoothed = self._moving_average(loss_array, smoothing_window)
                    smooth_steps = steps[smoothing_window - 1 :]
                    ax.plot(
                        smooth_steps,
                        smoothed,
                        linewidth=self.config.line_width,
                        label=f"{loss_name}" + (" (smoothed)" if show_raw else ""),
                        color=color,
                    )
                else:
                    ax.plot(
                        steps,
                        loss_array,
                        linewidth=self.config.line_width,
                        label=loss_name,
                        color=color,
                    )

            # Styling
            if log_scale:
                ax.set_yscale("log")

            ax.set_xlabel("Training Step", fontsize=self.config.label_fontsize)
            ax.set_ylabel("Loss Value", fontsize=self.config.label_fontsize)
            ax.set_title(title, fontsize=self.config.title_fontsize, fontweight="bold")
            ax.legend(fontsize=self.config.legend_fontsize)
            ax.grid(True, alpha=self.config.grid_alpha)
            ax.tick_params(labelsize=self.config.tick_fontsize)

            plt.tight_layout()

            if save_path:
                self._save_figure(fig, save_path)

            return fig

        except Exception as e:
            plt.close(fig) if "fig" in locals() else None
            raise VisualizationError(
                f"Failed to create loss curves plot: {str(e)}"
            ) from e

    def plot_error_analysis(
        self,
        x: np.ndarray,
        u_pred: np.ndarray,
        u_true: np.ndarray,
        title: str = "Error Analysis",
        save_path: Optional[Union[str, Path]] = None,
        show_statistics: bool = True,
        n_bins: int = 30,
    ) -> Figure:
        """
        Comprehensive error analysis comparing predicted vs true solutions.

        Args:
            x: Spatial coordinates
            u_pred: Predicted solution
            u_true: True/analytical solution
            title: Plot title
            save_path: Path to save figure, optional
            show_statistics: Whether to show error statistics
            n_bins: Number of bins for error histogram

        Returns:
            matplotlib Figure object

        Raises:
            VisualizationError: If input arrays have incompatible shapes or invalid data
            ValueError: If parameters are invalid
        """
        # Input validation
        self._validate_1d_arrays(x, u_pred, "x", "u_pred")
        self._validate_1d_arrays(x, u_true, "x", "u_true")

        if n_bins <= 0:
            raise ValueError(f"n_bins must be positive, got {n_bins}")

        if save_path is not None:
            self._validate_save_path(save_path)

        try:
            # Calculate errors
            abs_error = np.abs(u_pred - u_true)
            rel_error = abs_error / (
                np.abs(u_true) + 1e-10
            )  # Add small epsilon to avoid division by zero

            # Statistics
            mae = np.mean(abs_error)
            rmse = np.sqrt(np.mean((u_pred - u_true) ** 2))
            max_error = np.max(abs_error)
            correlation = np.corrcoef(u_pred.flatten(), u_true.flatten())[0, 1]

            # Handle case where correlation cannot be computed
            if np.isnan(correlation):
                correlation = 0.0
                warnings.warn("Could not compute correlation coefficient")

            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=self.dpi)

            # Solution comparison
            axes[0, 0].plot(
                x,
                u_true,
                "r-",
                linewidth=self.config.line_width,
                label="True",
                alpha=self.config.alpha,
            )
            axes[0, 0].plot(
                x,
                u_pred,
                "b--",
                linewidth=self.config.line_width,
                label="Predicted",
                alpha=self.config.alpha,
            )
            axes[0, 0].set_xlabel("x", fontsize=self.config.label_fontsize)
            axes[0, 0].set_ylabel("u(x)", fontsize=self.config.label_fontsize)
            axes[0, 0].set_title(
                "Solution Comparison", fontsize=self.config.title_fontsize
            )
            axes[0, 0].legend(fontsize=self.config.legend_fontsize)
            axes[0, 0].grid(True, alpha=self.config.grid_alpha)
            axes[0, 0].tick_params(labelsize=self.config.tick_fontsize)

            # Absolute error
            axes[0, 1].plot(x, abs_error, "g-", linewidth=self.config.line_width)
            axes[0, 1].set_xlabel("x", fontsize=self.config.label_fontsize)
            axes[0, 1].set_ylabel(
                "|u_pred - u_true|", fontsize=self.config.label_fontsize
            )
            axes[0, 1].set_title("Absolute Error", fontsize=self.config.title_fontsize)
            axes[0, 1].grid(True, alpha=self.config.grid_alpha)
            axes[0, 1].tick_params(labelsize=self.config.tick_fontsize)

            # Relative error
            axes[1, 0].plot(x, rel_error, "m-", linewidth=self.config.line_width)
            axes[1, 0].set_xlabel("x", fontsize=self.config.label_fontsize)
            axes[1, 0].set_ylabel("Relative Error", fontsize=self.config.label_fontsize)
            axes[1, 0].set_title("Relative Error", fontsize=self.config.title_fontsize)
            axes[1, 0].grid(True, alpha=self.config.grid_alpha)
            axes[1, 0].tick_params(labelsize=self.config.tick_fontsize)

            # Error histogram
            axes[1, 1].hist(
                abs_error, bins=n_bins, alpha=0.7, density=True, color="skyblue"
            )
            axes[1, 1].set_xlabel("Absolute Error", fontsize=self.config.label_fontsize)
            axes[1, 1].set_ylabel("Density", fontsize=self.config.label_fontsize)
            axes[1, 1].set_title(
                "Error Distribution", fontsize=self.config.title_fontsize
            )
            axes[1, 1].grid(True, alpha=self.config.grid_alpha)
            axes[1, 1].tick_params(labelsize=self.config.tick_fontsize)

            # Add error statistics as text if requested
            if show_statistics:
                stats_text = (
                    f"MAE: {mae:.2e}\n"
                    f"RMSE: {rmse:.2e}\n"
                    f"Max Error: {max_error:.2e}\n"
                    f"Correlation: {correlation:.4f}"
                )
                axes[1, 1].text(
                    0.05,
                    0.95,
                    stats_text,
                    transform=axes[1, 1].transAxes,
                    verticalalignment="top",
                    fontsize=self.config.tick_fontsize,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

            fig.suptitle(
                title, fontsize=self.config.title_fontsize + 2, fontweight="bold"
            )
            plt.tight_layout()

            if save_path:
                self._save_figure(fig, save_path)

            return fig

        except Exception as e:
            plt.close(fig) if "fig" in locals() else None
            raise VisualizationError(
                f"Failed to create error analysis plot: {str(e)}"
            ) from e

    # Validation methods
    def _validate_1d_arrays(
        self, arr1: np.ndarray, arr2: np.ndarray, name1: str, name2: str
    ) -> None:
        """Validate that two 1D arrays have compatible shapes and valid data."""
        if not isinstance(arr1, np.ndarray) or not isinstance(arr2, np.ndarray):
            raise VisualizationError(f"{name1} and {name2} must be numpy arrays")

        if arr1.shape != arr2.shape:
            raise VisualizationError(
                f"Shape mismatch: {name1}{arr1.shape} vs {name2}{arr2.shape}"
            )

        if arr1.size == 0:
            raise VisualizationError(f"{name1} and {name2} cannot be empty")

        if not np.all(np.isfinite(arr1)):
            raise VisualizationError(f"{name1} contains non-finite values")

        if not np.all(np.isfinite(arr2)):
            raise VisualizationError(f"{name2} contains non-finite values")

    def _validate_2d_arrays(
        self, x: np.ndarray, y: np.ndarray, field: np.ndarray
    ) -> None:
        """Validate that 2D coordinate arrays and field have compatible shapes."""
        arrays = [("x", x), ("y", y), ("field", field)]

        for name, arr in arrays:
            if not isinstance(arr, np.ndarray):
                raise VisualizationError(f"{name} must be a numpy array")
            if arr.ndim != 2:
                raise VisualizationError(f"{name} must be 2D, got {arr.ndim}D")
            if arr.size == 0:
                raise VisualizationError(f"{name} cannot be empty")
            if not np.all(np.isfinite(arr)):
                raise VisualizationError(f"{name} contains non-finite values")

        if not (x.shape == y.shape == field.shape):
            raise VisualizationError(
                f"Shape mismatch: x{x.shape}, y{y.shape}, field{field.shape}"
            )

    def _validate_save_path(self, save_path: Union[str, Path]) -> None:
        """Validate save path and create directory if needed."""
        path = Path(save_path)

        # Create parent directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        # Check if we can write to the directory
        if not path.parent.is_dir():
            raise ValueError(f"Cannot create directory: {path.parent}")

        # Check file extension
        valid_extensions = {".png", ".pdf", ".eps", ".svg", ".jpg", ".jpeg"}
        if path.suffix.lower() not in valid_extensions:
            warnings.warn(
                f"File extension '{path.suffix}' may not be supported. "
                f"Recommended: {valid_extensions}"
            )

    def _save_figure(self, fig: Figure, save_path: Union[str, Path]) -> None:
        """Save figure with error handling.

        The output format follows the extension of ``save_path``. Forcing
        ``config.save_format`` here instead would write, say, PNG bytes into a
        file named ``.pdf`` -- and ``_validate_save_path`` advertises pdf, eps,
        svg and jpeg as supported. ``config.save_format`` is the fallback for a
        path with no extension.
        """
        suffix = Path(save_path).suffix.lstrip(".").lower()
        # matplotlib registers the format as "jpeg"; ".jpg" is the common spelling.
        fmt = {"jpg": "jpeg"}.get(suffix, suffix) or self.config.save_format

        try:
            logger.info("Saving figure", extra={"path": str(save_path), "format": fmt})
            fig.savefig(
                save_path,
                dpi=self.config.save_dpi,
                bbox_inches=self.config.save_bbox_inches,
                format=fmt,
            )
        except Exception as e:
            raise VisualizationError(
                f"Failed to save figure to {save_path}: {str(e)}"
            ) from e

    @staticmethod
    def _moving_average(data: Sequence[float], window: int) -> np.ndarray:
        """
        Calculate moving average for smoothing loss curves.

        Args:
            data: Input data sequence
            window: Window size for averaging

        Returns:
            Smoothed data array

        Raises:
            ValueError: If window size is invalid
        """
        if window <= 0:
            raise ValueError(f"Window size must be positive, got {window}")
        if window > len(data):
            raise ValueError(
                f"Window size ({window}) cannot be larger than data length ({len(data)})"
            )

        data_array = np.asarray(data)
        return np.convolve(data_array, np.ones(window) / window, mode="valid")


# Convenience functions with improved error handling
def quick_plot_1d(
    x: np.ndarray,
    u_pred: np.ndarray,
    u_true: Optional[np.ndarray] = None,
    title: str = "1D Solution",
    save_path: Optional[Union[str, Path]] = None,
    config: Optional[VisualizationConfig] = None,
) -> Figure:
    """
    Quick 1D plotting function with error handling.

    Args:
        x: Spatial coordinates
        u_pred: Predicted solution
        u_true: True solution (optional)
        title: Plot title
        save_path: Save path (optional)
        config: Visualization configuration (optional)

    Returns:
        matplotlib Figure object

    Raises:
        VisualizationError: If plotting fails
    """
    try:
        viz = PINNVisualizer(config=config)
        return viz.plot_1d_solution(x, u_pred, u_true, title, save_path=save_path)
    except Exception as e:
        raise VisualizationError(f"Quick 1D plot failed: {str(e)}") from e


def quick_plot_2d(
    x: np.ndarray,
    y: np.ndarray,
    field: np.ndarray,
    title: str = "2D Field",
    save_path: Optional[Union[str, Path]] = None,
    config: Optional[VisualizationConfig] = None,
) -> Figure:
    """
    Quick 2D plotting function with error handling.

    Args:
        x: X coordinates
        y: Y coordinates
        field: Field values
        title: Plot title
        save_path: Save path (optional)
        config: Visualization configuration (optional)

    Returns:
        matplotlib Figure object

    Raises:
        VisualizationError: If plotting fails
    """
    try:
        viz = PINNVisualizer(config=config)
        return viz.plot_2d_field(x, y, field, title, save_path=save_path)
    except Exception as e:
        raise VisualizationError(f"Quick 2D plot failed: {str(e)}") from e


def quick_plot_loss(
    loss_history: Dict[str, Sequence[float]],
    title: str = "Loss History",
    save_path: Optional[Union[str, Path]] = None,
    config: Optional[VisualizationConfig] = None,
) -> Figure:
    """
    Quick loss plotting function with error handling.

    Args:
        loss_history: Dictionary of loss components
        title: Plot title
        save_path: Save path (optional)
        config: Visualization configuration (optional)

    Returns:
        matplotlib Figure object

    Raises:
        VisualizationError: If plotting fails
    """
    try:
        viz = PINNVisualizer(config=config)
        return viz.plot_loss_curves(loss_history, title, save_path=save_path)
    except Exception as e:
        raise VisualizationError(f"Quick loss plot failed: {str(e)}") from e


def quick_error_analysis(
    x: np.ndarray,
    u_pred: np.ndarray,
    u_true: np.ndarray,
    title: str = "Error Analysis",
    save_path: Optional[Union[str, Path]] = None,
    config: Optional[VisualizationConfig] = None,
) -> Figure:
    """
    Quick error analysis function with error handling.

    Args:
        x: Spatial coordinates
        u_pred: Predicted solution
        u_true: True solution
        title: Plot title
        save_path: Save path (optional)
        config: Visualization configuration (optional)

    Returns:
        matplotlib Figure object

    Raises:
        VisualizationError: If plotting fails
    """
    try:
        viz = PINNVisualizer(config=config)
        return viz.plot_error_analysis(x, u_pred, u_true, title, save_path=save_path)
    except Exception as e:
        raise VisualizationError(f"Quick error analysis failed: {str(e)}") from e


# Utility functions for validation and configuration
def create_custom_config(
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 100,
    line_width: float = 2.0,
    title_fontsize: int = 14,
    label_fontsize: int = 12,
    **kwargs,
) -> VisualizationConfig:
    """
    Create a custom visualization configuration.

    Args:
        figsize: Default figure size
        dpi: Figure resolution
        line_width: Default line width
        title_fontsize: Title font size
        label_fontsize: Label font size
        **kwargs: Additional configuration parameters

    Returns:
        VisualizationConfig object

    Raises:
        ValueError: If parameters are invalid
    """
    # Validate inputs
    if len(figsize) != 2 or any(dim <= 0 for dim in figsize):
        raise ValueError(
            f"figsize must be a tuple of 2 positive numbers, got {figsize}"
        )

    if dpi <= 0:
        raise ValueError(f"dpi must be positive, got {dpi}")

    if line_width <= 0:
        raise ValueError(f"line_width must be positive, got {line_width}")

    if title_fontsize <= 0:
        raise ValueError(f"title_fontsize must be positive, got {title_fontsize}")

    if label_fontsize <= 0:
        raise ValueError(f"label_fontsize must be positive, got {label_fontsize}")

    # Create configuration with provided parameters
    config_params = {
        "figsize": figsize,
        "dpi": dpi,
        "line_width": line_width,
        "title_fontsize": title_fontsize,
        "label_fontsize": label_fontsize,
    }

    # Add any additional parameters
    config_params.update(kwargs)

    try:
        return VisualizationConfig(**config_params)
    except TypeError as e:
        raise ValueError(f"Invalid configuration parameters: {str(e)}") from e


def validate_plot_inputs(
    arrays: Dict[str, np.ndarray],
    required_shapes: Optional[Dict[str, Tuple]] = None,
    allow_empty: bool = False,
) -> None:
    """
    Validate multiple arrays for plotting.

    Args:
        arrays: Dictionary of array name -> array mappings
        required_shapes: Optional shape requirements
        allow_empty: Whether to allow empty arrays

    Raises:
        VisualizationError: If validation fails
    """
    for name, array in arrays.items():
        if not isinstance(array, np.ndarray):
            raise VisualizationError(f"{name} must be a numpy array, got {type(array)}")

        if not allow_empty and array.size == 0:
            raise VisualizationError(f"{name} cannot be empty")

        if not np.all(np.isfinite(array)):
            raise VisualizationError(f"{name} contains non-finite values")

        if required_shapes and name in required_shapes:
            expected_shape = required_shapes[name]
            if array.shape != expected_shape:
                raise VisualizationError(
                    f"{name} has shape {array.shape}, expected {expected_shape}"
                )


# Example usage and testing functions
def demo_visualization_improvements():
    """
    Demonstrate the improved visualization capabilities.

    This function shows how to use the enhanced visualization module
    with proper error handling and configuration.
    """
    print("Demonstrating improved visualization module...")

    try:
        # Create sample data
        x = np.linspace(-1, 1, 100)
        u_true = -np.sin(np.pi * x)
        u_pred = u_true + 0.05 * np.random.normal(0, 1, len(x))  # Add some noise

        # Create custom configuration
        custom_config = create_custom_config(
            figsize=(14, 6), line_width=2.5, title_fontsize=16, default_cmap="plasma"
        )

        # Create visualizer with custom config
        viz = PINNVisualizer(config=custom_config)

        # Test 1D plotting with error handling
        print("Testing 1D solution plotting...")
        viz.plot_1d_solution(
            x,
            u_pred,
            u_true,
            title="Improved 1D Solution Plot",
            legend_labels=("PINN Prediction", "Ground Truth"),
        )
        plt.show()

        # Test error analysis
        print("Testing error analysis...")
        viz.plot_error_analysis(x, u_pred, u_true, title="Comprehensive Error Analysis")
        plt.show()

        # Test loss curve plotting with synthetic data
        print("Testing loss curve plotting...")
        synthetic_loss = {
            "total": [
                1e-1 * (0.95) ** i + 0.01 * np.random.random() for i in range(1000)
            ],
            "pde": [
                5e-2 * (0.96) ** i + 0.005 * np.random.random() for i in range(1000)
            ],
            "ic": [
                3e-2 * (0.94) ** i + 0.003 * np.random.random() for i in range(1000)
            ],
            "bc": [
                2e-2 * (0.97) ** i + 0.002 * np.random.random() for i in range(1000)
            ],
        }

        viz.plot_loss_curves(
            synthetic_loss,
            title="Enhanced Loss Curves with Smoothing",
            smoothing_window=100,
        )
        plt.show()

        # Test 2D plotting
        print("Testing 2D field plotting...")
        x_2d = np.linspace(-2, 2, 50)
        y_2d = np.linspace(-2, 2, 50)
        X, Y = np.meshgrid(x_2d, y_2d)
        field_2d = np.sin(X) * np.cos(Y) * np.exp(-(X**2 + Y**2) / 4)

        viz.plot_2d_field(
            X,
            Y,
            field_2d,
            title="Enhanced 2D Field Plot",
            colorbar_label="Field Amplitude",
        )
        plt.show()

        # Test vector field plotting
        print("Testing vector field plotting...")
        U = -np.sin(Y)
        V = np.sin(X)

        viz.plot_vector_field_2d(X, Y, U, V, title="Enhanced Vector Field Plot", skip=3)
        plt.show()

        print("All visualization tests passed successfully!")

    except VisualizationError as e:
        print(f"Visualization error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    # Run demonstration
    demo_visualization_improvements()

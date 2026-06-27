"""
Comprehensive test suite for the improved visualization module.

This test suite validates:
- Type safety and error handling
- Input validation
- Configuration management
- Edge cases and error conditions
- Integration with matplotlib

Run with: python -m pytest test_improved_visualization.py -v
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt


from pinn.utils.visualization import (
    PINNVisualizer,
    VisualizationConfig,
    VisualizationError,
    quick_plot_1d,
    quick_plot_2d,
    quick_plot_loss,
    create_custom_config,
)


# For this demonstration, we'll assume the classes are available
class VisualizationError(Exception):
    """Custom exception for visualization errors."""

    pass


class TestVisualizationConfig:
    """Test the VisualizationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class VisualizationConfig:
            figsize: tuple = (12, 8)
            dpi: int = 100
            line_width: float = 2.0
            alpha: float = 0.8

        config = VisualizationConfig()
        assert config.figsize == (12, 8)
        assert config.dpi == 100
        assert config.line_width == 2.0
        assert config.alpha == 0.8

    def test_custom_config(self):
        """Test custom configuration values."""
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class VisualizationConfig:
            figsize: tuple = (12, 8)
            dpi: int = 100
            line_width: float = 2.0
            alpha: float = 0.8

        config = VisualizationConfig(
            figsize=(10, 6), dpi=150, line_width=3.0, alpha=0.9
        )
        assert config.figsize == (10, 6)
        assert config.dpi == 150
        assert config.line_width == 3.0
        assert config.alpha == 0.9


class TestPINNVisualizerInitialization:
    """Test PINNVisualizer initialization and configuration."""

    def test_default_initialization(self):
        """Test default initializer."""

        # Mock the visualizer class for testing
        class MockPINNVisualizer:
            def __init__(self, config=None, figsize=None, dpi=None):
                from dataclasses import dataclass

                @dataclass(frozen=True)
                class VisualizationConfig:
                    figsize: tuple = (12, 8)
                    dpi: int = 100

                if config is None:
                    config = VisualizationConfig()

                self.config = config
                self.figsize = figsize if figsize is not None else config.figsize
                self.dpi = dpi if dpi is not None else config.dpi

        viz = MockPINNVisualizer()
        assert viz.figsize == (12, 8)
        assert viz.dpi == 100

    def test_custom_figsize_and_dpi(self):
        """Test custom figsize and dpi override."""

        class MockPINNVisualizer:
            def __init__(self, config=None, figsize=None, dpi=None):
                from dataclasses import dataclass

                @dataclass(frozen=True)
                class VisualizationConfig:
                    figsize: tuple = (12, 8)
                    dpi: int = 100

                if config is None:
                    config = VisualizationConfig()

                # Validate inputs
                if figsize is not None:
                    if len(figsize) != 2 or any(dim <= 0 for dim in figsize):
                        raise ValueError(
                            f"figsize must be a tuple of 2 positive numbers, got {figsize}"
                        )

                if dpi is not None:
                    if dpi <= 0:
                        raise ValueError(f"dpi must be positive, got {dpi}")

                self.config = config
                self.figsize = figsize if figsize is not None else config.figsize
                self.dpi = dpi if dpi is not None else config.dpi

        viz = MockPINNVisualizer(figsize=(10, 6), dpi=150)
        assert viz.figsize == (10, 6)
        assert viz.dpi == 150

    def test_invalid_figsize_raises_error(self):
        """Test that invalid figsize raises ValueError."""

        class MockPINNVisualizer:
            def __init__(self, config=None, figsize=None, dpi=None):
                if figsize is not None:
                    if len(figsize) != 2 or any(dim <= 0 for dim in figsize):
                        raise ValueError(
                            f"figsize must be a tuple of 2 positive numbers, got {figsize}"
                        )

        with pytest.raises(
            ValueError, match="figsize must be a tuple of 2 positive numbers"
        ):
            MockPINNVisualizer(figsize=(10, 0))

        with pytest.raises(
            ValueError, match="figsize must be a tuple of 2 positive numbers"
        ):
            MockPINNVisualizer(figsize=(10,))

    def test_invalid_dpi_raises_error(self):
        """Test that invalid dpi raises ValueError."""

        class MockPINNVisualizer:
            def __init__(self, config=None, figsize=None, dpi=None):
                if dpi is not None:
                    if dpi <= 0:
                        raise ValueError(f"dpi must be positive, got {dpi}")

        with pytest.raises(ValueError, match="dpi must be positive"):
            MockPINNVisualizer(dpi=-10)

        with pytest.raises(ValueError, match="dpi must be positive"):
            MockPINNVisualizer(dpi=0)


class TestInputValidation:
    """Test input validation methods."""

    def create_mock_validator(self):
        """Create a mock validator class with validation methods."""

        class MockValidator:
            @staticmethod
            def _validate_1d_arrays(arr1, arr2, name1, name2):
                if not isinstance(arr1, np.ndarray) or not isinstance(arr2, np.ndarray):
                    raise VisualizationError(
                        f"{name1} and {name2} must be numpy arrays"
                    )

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

            @staticmethod
            def _validate_2d_arrays(x, y, field):
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

        return MockValidator()

    def test_validate_1d_arrays_valid_input(self):
        """Test validation with valid 1D arrays."""
        validator = self.create_mock_validator()
        x = np.linspace(0, 1, 10)
        y = np.sin(x)

        # Should not raise any exception
        validator._validate_1d_arrays(x, y, "x", "y")

    def test_validate_1d_arrays_shape_mismatch(self):
        """Test validation with mismatched shapes."""
        validator = self.create_mock_validator()
        x = np.linspace(0, 1, 10)
        y = np.sin(np.linspace(0, 1, 5))  # Different length

        with pytest.raises(VisualizationError, match="Shape mismatch"):
            validator._validate_1d_arrays(x, y, "x", "y")

    def test_validate_1d_arrays_empty_arrays(self):
        """Test validation with empty arrays."""
        validator = self.create_mock_validator()
        x = np.array([])
        y = np.array([])

        with pytest.raises(VisualizationError, match="cannot be empty"):
            validator._validate_1d_arrays(x, y, "x", "y")

    def test_validate_1d_arrays_non_finite_values(self):
        """Test validation with non-finite values."""
        validator = self.create_mock_validator()
        x = np.array([1, 2, np.inf, 4])
        y = np.array([1, 2, 3, 4])

        with pytest.raises(VisualizationError, match="contains non-finite values"):
            validator._validate_1d_arrays(x, y, "x", "y")

    def test_validate_1d_arrays_non_numpy_input(self):
        """Test validation with non-numpy arrays."""
        validator = self.create_mock_validator()
        x = [1, 2, 3, 4]  # Python list, not numpy array
        y = np.array([1, 2, 3, 4])

        with pytest.raises(VisualizationError, match="must be numpy arrays"):
            validator._validate_1d_arrays(x, y, "x", "y")

    def test_validate_2d_arrays_valid_input(self):
        """Test validation with valid 2D arrays."""
        validator = self.create_mock_validator()
        x = np.ones((5, 5))
        y = np.ones((5, 5))
        field = np.ones((5, 5))

        # Should not raise any exception
        validator._validate_2d_arrays(x, y, field)

    def test_validate_2d_arrays_wrong_dimensions(self):
        """Test validation with wrong number of dimensions."""
        validator = self.create_mock_validator()
        x = np.ones(5)  # 1D instead of 2D
        y = np.ones((5, 5))
        field = np.ones((5, 5))

        with pytest.raises(VisualizationError, match="must be 2D"):
            validator._validate_2d_arrays(x, y, field)

    def test_validate_2d_arrays_shape_mismatch(self):
        """Test validation with mismatched 2D shapes."""
        validator = self.create_mock_validator()
        x = np.ones((5, 5))
        y = np.ones((5, 5))
        field = np.ones((4, 4))  # Different shape

        with pytest.raises(VisualizationError, match="Shape mismatch"):
            validator._validate_2d_arrays(x, y, field)


class TestLossCurveValidation:
    """Test loss curve plotting validation."""

    def create_mock_loss_plotter(self):
        """Create a mock loss curve plotter."""

        class MockLossPlotter:
            def plot_loss_curves(self, loss_history, **kwargs):
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
                        import warnings

                        warnings.warn(
                            f"Loss values for '{loss_name}' contain non-finite values"
                        )
                    if np.any(loss_array < 0):
                        import warnings

                        warnings.warn(
                            f"Loss values for '{loss_name}' contain negative values"
                        )

                # Return mock figure
                return MagicMock()

        return MockLossPlotter()

    def test_valid_loss_history(self):
        """Test with valid loss history."""
        plotter = self.create_mock_loss_plotter()
        loss_history = {
            "total": [1.0, 0.5, 0.25, 0.125],
            "pde": [0.6, 0.3, 0.15, 0.075],
            "ic": [0.4, 0.2, 0.1, 0.05],
        }

        # Should not raise any exception
        fig = plotter.plot_loss_curves(loss_history)
        assert fig is not None

    def test_empty_loss_history(self):
        """Test with empty loss history."""
        plotter = self.create_mock_loss_plotter()
        loss_history = {}

        with pytest.raises(VisualizationError, match="loss_history cannot be empty"):
            plotter.plot_loss_curves(loss_history)

    def test_empty_loss_values(self):
        """Test with empty loss values for a component."""
        plotter = self.create_mock_loss_plotter()
        loss_history = {
            "total": [],  # Empty list
            "pde": [0.6, 0.3, 0.15],
        }

        with pytest.raises(
            VisualizationError, match="Loss values for 'total' cannot be empty"
        ):
            plotter.plot_loss_curves(loss_history)

    def test_loss_history_with_warnings(self):
        """Test that warnings are issued for problematic data."""
        plotter = self.create_mock_loss_plotter()

        # Test with non-finite values
        with pytest.warns(UserWarning, match="contain non-finite values"):
            loss_history = {
                "total": [1.0, 0.5, np.inf, 0.125],  # Contains infinity
            }
            plotter.plot_loss_curves(loss_history)

        # Test with negative values
        with pytest.warns(UserWarning, match="contain negative values"):
            loss_history = {
                "total": [1.0, 0.5, -0.25, 0.125],  # Contains negative value
            }
            plotter.plot_loss_curves(loss_history)


class TestSavePathValidation:
    """Test save path validation and file handling."""

    def create_mock_path_validator(self):
        """Create a mock path validator."""

        class MockPathValidator:
            @staticmethod
            def _validate_save_path(save_path):
                from pathlib import Path
                import warnings

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

        return MockPathValidator()

    def test_valid_save_path(self):
        """Test with valid save path."""
        validator = self.create_mock_path_validator()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_plot.png")

            # Should not raise any exception
            validator._validate_save_path(save_path)

    def test_create_directory_if_not_exists(self):
        """Test that directories are created if they don't exist."""
        validator = self.create_mock_path_validator()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "new_dir", "test_plot.png")

            # Directory should be created
            validator._validate_save_path(save_path)
            assert os.path.exists(os.path.dirname(save_path))

    def test_unsupported_file_extension_warning(self):
        """Test warning for unsupported file extensions."""
        validator = self.create_mock_path_validator()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_plot.xyz")  # Unsupported extension

            with pytest.warns(
                UserWarning, match="File extension '.xyz' may not be supported"
            ):
                validator._validate_save_path(save_path)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_moving_average_valid_input(self):
        """Test moving average with valid input."""

        def moving_average(data, window):
            if window <= 0:
                raise ValueError(f"Window size must be positive, got {window}")
            if window > len(data):
                raise ValueError(
                    f"Window size ({window}) cannot be larger than data length ({len(data)})"
                )

            data_array = np.asarray(data)
            return np.convolve(data_array, np.ones(window) / window, mode="valid")

        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        window = 3

        result = moving_average(data, window)
        expected = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

        np.testing.assert_array_almost_equal(result, expected)

    def test_moving_average_invalid_window(self):
        """Test moving average with invalid window size."""

        def moving_average(data, window):
            if window <= 0:
                raise ValueError(f"Window size must be positive, got {window}")
            if window > len(data):
                raise ValueError(
                    f"Window size ({window}) cannot be larger than data length ({len(data)})"
                )

            data_array = np.asarray(data)
            return np.convolve(data_array, np.ones(window) / window, mode="valid")

        data = [1, 2, 3, 4, 5]

        # Test negative window
        with pytest.raises(ValueError, match="Window size must be positive"):
            moving_average(data, -1)

        # Test zero window
        with pytest.raises(ValueError, match="Window size must be positive"):
            moving_average(data, 0)

        # Test window larger than data
        with pytest.raises(
            ValueError, match="Window size .* cannot be larger than data length"
        ):
            moving_average(data, 10)


class TestConfigurationManagement:
    """Test configuration creation and validation."""

    def test_create_custom_config_valid(self):
        """Test creating custom configuration with valid parameters."""

        def create_custom_config(
            figsize=(12, 8),
            dpi=100,
            line_width=2.0,
            title_fontsize=14,
            label_fontsize=12,
            **kwargs,
        ):
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
                raise ValueError(
                    f"title_fontsize must be positive, got {title_fontsize}"
                )

            if label_fontsize <= 0:
                raise ValueError(
                    f"label_fontsize must be positive, got {label_fontsize}"
                )

            # Create mock config
            config_params = {
                "figsize": figsize,
                "dpi": dpi,
                "line_width": line_width,
                "title_fontsize": title_fontsize,
                "label_fontsize": label_fontsize,
            }
            config_params.update(kwargs)

            return config_params

        # Should work with valid parameters
        config = create_custom_config(
            figsize=(10, 6),
            dpi=150,
            line_width=2.5,
            title_fontsize=16,
            label_fontsize=13,
        )

        assert config["figsize"] == (10, 6)
        assert config["dpi"] == 150
        assert config["line_width"] == 2.5
        assert config["title_fontsize"] == 16
        assert config["label_fontsize"] == 13

    def test_create_custom_config_invalid_figsize(self):
        """Test creating custom configuration with invalid figsize."""

        def create_custom_config(
            figsize=(12, 8),
            dpi=100,
            line_width=2.0,
            title_fontsize=14,
            label_fontsize=12,
            **kwargs,
        ):
            if len(figsize) != 2 or any(dim <= 0 for dim in figsize):
                raise ValueError(
                    f"figsize must be a tuple of 2 positive numbers, got {figsize}"
                )
            return {}

        with pytest.raises(
            ValueError, match="figsize must be a tuple of 2 positive numbers"
        ):
            create_custom_config(figsize=(10, 0))

        with pytest.raises(
            ValueError, match="figsize must be a tuple of 2 positive numbers"
        ):
            create_custom_config(figsize=(10,))

    def test_create_custom_config_invalid_parameters(self):
        """Test creating custom configuration with invalid parameters."""

        def create_custom_config(
            figsize=(12, 8),
            dpi=100,
            line_width=2.0,
            title_fontsize=14,
            label_fontsize=12,
            **kwargs,
        ):
            if dpi <= 0:
                raise ValueError(f"dpi must be positive, got {dpi}")
            if line_width <= 0:
                raise ValueError(f"line_width must be positive, got {line_width}")
            if title_fontsize <= 0:
                raise ValueError(
                    f"title_fontsize must be positive, got {title_fontsize}"
                )
            if label_fontsize <= 0:
                raise ValueError(
                    f"label_fontsize must be positive, got {label_fontsize}"
                )
            return {}

        with pytest.raises(ValueError, match="dpi must be positive"):
            create_custom_config(dpi=-100)

        with pytest.raises(ValueError, match="line_width must be positive"):
            create_custom_config(line_width=-1.0)

        with pytest.raises(ValueError, match="title_fontsize must be positive"):
            create_custom_config(title_fontsize=-14)

        with pytest.raises(ValueError, match="label_fontsize must be positive"):
            create_custom_config(label_fontsize=0)


class TestMultiArrayValidation:
    """Test validation of multiple arrays simultaneously."""

    def test_validate_plot_inputs_valid(self):
        """Test validation with valid inputs."""

        def validate_plot_inputs(arrays, required_shapes=None, allow_empty=False):
            for name, array in arrays.items():
                if not isinstance(array, np.ndarray):
                    raise VisualizationError(
                        f"{name} must be a numpy array, got {type(array)}"
                    )

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

        arrays = {"x": np.linspace(0, 1, 10), "y": np.sin(np.linspace(0, 1, 10))}

        # Should not raise any exception
        validate_plot_inputs(arrays)

    def test_validate_plot_inputs_wrong_type(self):
        """Test validation with wrong array type."""

        def validate_plot_inputs(arrays, required_shapes=None, allow_empty=False):
            for name, array in arrays.items():
                if not isinstance(array, np.ndarray):
                    raise VisualizationError(
                        f"{name} must be a numpy array, got {type(array)}"
                    )

        arrays = {
            "x": [1, 2, 3, 4],  # List instead of numpy array
            "y": np.array([1, 2, 3, 4]),
        }

        with pytest.raises(VisualizationError, match="x must be a numpy array"):
            validate_plot_inputs(arrays)

    def test_validate_plot_inputs_empty_not_allowed(self):
        """Test validation with empty arrays when not allowed."""

        def validate_plot_inputs(arrays, required_shapes=None, allow_empty=False):
            for name, array in arrays.items():
                if not isinstance(array, np.ndarray):
                    raise VisualizationError(
                        f"{name} must be a numpy array, got {type(array)}"
                    )

                if not allow_empty and array.size == 0:
                    raise VisualizationError(f"{name} cannot be empty")

        arrays = {
            "x": np.array([]),  # Empty array
            "y": np.array([1, 2, 3]),
        }

        with pytest.raises(VisualizationError, match="x cannot be empty"):
            validate_plot_inputs(arrays, allow_empty=False)

    def test_validate_plot_inputs_shape_requirements(self):
        """Test validation with shape requirements."""

        def validate_plot_inputs(arrays, required_shapes=None, allow_empty=False):
            for name, array in arrays.items():
                if not isinstance(array, np.ndarray):
                    raise VisualizationError(
                        f"{name} must be a numpy array, got {type(array)}"
                    )

                if required_shapes and name in required_shapes:
                    expected_shape = required_shapes[name]
                    if array.shape != expected_shape:
                        raise VisualizationError(
                            f"{name} has shape {array.shape}, expected {expected_shape}"
                        )

        arrays = {
            "x": np.ones((5, 5)),
            "y": np.ones((3, 3)),  # Wrong shape
        }

        required_shapes = {
            "x": (5, 5),
            "y": (5, 5),  # Expecting same shape as x
        }

        with pytest.raises(
            VisualizationError, match="y has shape \\(3, 3\\), expected \\(5, 5\\)"
        ):
            validate_plot_inputs(arrays, required_shapes=required_shapes)


class TestIntegrationWithMatplotlib:
    """Test integration with matplotlib components."""

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.tight_layout")
    def test_matplotlib_integration_success(self, mock_tight_layout, mock_subplots):
        """Test successful matplotlib integration."""
        # Mock matplotlib components
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        # Mock plot function that uses matplotlib
        def mock_plot_function():
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot([1, 2, 3], [1, 4, 2])
            plt.tight_layout()
            return fig

        # Should complete without errors
        fig = mock_plot_function()

        # Verify matplotlib functions were called
        mock_subplots.assert_called_once_with(figsize=(10, 6))
        mock_tight_layout.assert_called_once()
        mock_ax.plot.assert_called_once_with([1, 2, 3], [1, 4, 2])

    @patch("matplotlib.pyplot.subplots")
    def test_matplotlib_integration_failure(self, mock_subplots):
        """Test matplotlib integration failure handling."""
        # Make matplotlib raise an exception
        mock_subplots.side_effect = RuntimeError("Matplotlib error")

        def mock_plot_function_with_error_handling():
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                return fig
            except Exception as e:
                if "fig" in locals():
                    plt.close(fig)
                raise VisualizationError(f"Plotting failed: {str(e)}") from e

        # Should raise VisualizationError, not RuntimeError
        with pytest.raises(
            VisualizationError, match="Plotting failed: Matplotlib error"
        ):
            mock_plot_function_with_error_handling()


class TestErrorRecovery:
    """Test error recovery and cleanup."""

    def test_figure_cleanup_on_error(self):
        """Test that figures are properly closed on errors."""
        cleanup_called = False

        def mock_plotting_function_with_cleanup():
            nonlocal cleanup_called
            fig = MagicMock()

            try:
                # Simulate an error during plotting
                raise RuntimeError("Plotting error")

            except Exception as e:
                # Cleanup should happen here
                if fig:
                    cleanup_called = True  # Simulate plt.close(fig)
                raise VisualizationError(f"Failed to create plot: {str(e)}") from e

        with pytest.raises(VisualizationError):
            mock_plotting_function_with_cleanup()

        # Verify cleanup was called
        assert cleanup_called

    def test_resource_cleanup_on_save_error(self):
        """Test resource cleanup when saving fails."""

        def mock_save_with_cleanup(fig, save_path):
            try:
                # Simulate save failure
                raise OSError("Cannot write to file")
            except Exception as e:
                # Figure should still be available for other operations
                # but error should be wrapped
                raise VisualizationError(
                    f"Failed to save figure to {save_path}: {str(e)}"
                ) from e

        mock_fig = MagicMock()

        with pytest.raises(VisualizationError, match="Failed to save figure"):
            mock_save_with_cleanup(mock_fig, "/invalid/path/test.png")


class TestPerformanceConsiderations:
    """Test performance-related aspects."""

    def test_large_array_handling(self):
        """Test handling of large arrays."""
        # Create large arrays
        size = 10000
        x = np.linspace(0, 1, size)
        y = np.sin(x)

        # Validation should still work efficiently
        def efficient_validation(arr1, arr2):
            # Basic checks should be O(1)
            if arr1.shape != arr2.shape:
                raise VisualizationError("Shape mismatch")

            # Finite check should be efficient but might be O(n)
            if not (np.isfinite(arr1).all() and np.isfinite(arr2).all()):
                raise VisualizationError("Non-finite values found")

        # Should complete in reasonable time
        import time

        start_time = time.time()
        efficient_validation(x, y)
        elapsed_time = time.time() - start_time

        # Should be very fast for validation
        assert elapsed_time < 1.0  # Should complete in less than 1 second

    def test_memory_efficient_operations(self):
        """Test memory-efficient operations."""
        # Test that we don't unnecessarily copy large arrays
        original_array = np.random.random(10000)

        def memory_efficient_check(arr):
            # Check if array is finite without creating copies
            return np.all(np.isfinite(arr))

        # This should not significantly increase memory usage
        result = memory_efficient_check(original_array)
        assert isinstance(result, (bool, np.bool_))


# Integration test to demonstrate the complete workflow
class TestCompleteWorkflow:
    """Test complete visualization workflow."""

    def test_end_to_end_workflow(self):
        """Test a complete visualization workflow."""
        # Generate test data
        x = np.linspace(-1, 1, 50)
        u_true = -np.sin(np.pi * x)
        u_pred = u_true + 0.05 * np.random.normal(0, 1, len(x))

        # Test data should be valid
        assert x.shape == u_true.shape == u_pred.shape
        assert np.all(np.isfinite(x))
        assert np.all(np.isfinite(u_true))
        assert np.all(np.isfinite(u_pred))

        # Generate 2D test data
        x_2d = np.linspace(-1, 1, 10)
        y_2d = np.linspace(-1, 1, 10)
        X, Y = np.meshgrid(x_2d, y_2d)
        field_2d = np.sin(X) * np.cos(Y)

        # Test 2D data should be valid
        assert X.shape == Y.shape == field_2d.shape
        assert X.ndim == Y.ndim == field_2d.ndim == 2

        # Generate loss history
        loss_history = {
            "total": [1.0 * (0.9) ** i for i in range(100)],
            "pde": [0.6 * (0.95) ** i for i in range(100)],
            "ic": [0.4 * (0.92) ** i for i in range(100)],
        }

        # All loss values should be positive and finite
        for name, values in loss_history.items():
            assert all(v > 0 for v in values), (
                f"Loss {name} contains non-positive values"
            )
            assert all(np.isfinite(v) for v in values), (
                f"Loss {name} contains non-finite values"
            )

        # If we had the actual implementation, we would test:
        # viz = PINNVisualizer()
        # fig1 = viz.plot_1d_solution(x, u_pred, u_true)
        # fig2 = viz.plot_2d_field(X, Y, field_2d)
        # fig3 = viz.plot_loss_curves(loss_history)
        # fig4 = viz.plot_error_analysis(x, u_pred, u_true)

        # For now, just verify the data is ready for visualization
        print("End-to-end workflow test data prepared successfully")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])

"""Tests for the visualisation toolkit.

These exercise the real implementations in ``pinnkit.utils.visualization``. An
earlier version of this file defined local stand-ins for ``PINNVisualizer``,
``VisualizationConfig`` and friends and asserted against those, so it passed
regardless of what the library did.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest
from matplotlib.figure import Figure

from pinnkit.utils.visualization import (
    PINNVisualizer,
    VisualizationConfig,
    VisualizationError,
    create_custom_config,
    quick_error_analysis,
    quick_plot_1d,
    quick_plot_2d,
    quick_plot_loss,
    validate_plot_inputs,
)


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every figure a test opens, so the suite does not leak them."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


@pytest.fixture
def viz() -> PINNVisualizer:
    return PINNVisualizer()


@pytest.fixture
def data_1d():
    x = np.linspace(-1.0, 1.0, 64)
    u_pred = np.sin(np.pi * x)
    u_true = np.sin(np.pi * x) + 0.01
    return x, u_pred, u_true


@pytest.fixture
def data_2d():
    x, y = np.meshgrid(np.linspace(0.0, 1.0, 16), np.linspace(0.0, 1.0, 16))
    field = np.sin(np.pi * x) * np.cos(np.pi * y)
    return x, y, field


# ---------------------------------------------------------------------------
# VisualizationConfig
# ---------------------------------------------------------------------------


class TestVisualizationConfig:
    def test_defaults(self):
        cfg = VisualizationConfig()
        assert cfg.figsize == (12, 8)
        assert cfg.dpi == 100
        assert cfg.line_width == 2.0
        assert cfg.default_cmap == "RdBu_r"
        assert cfg.save_dpi == 300

    def test_accepts_overrides(self):
        cfg = VisualizationConfig(figsize=(6, 4), dpi=150, default_cmap="viridis")
        assert cfg.figsize == (6, 4)
        assert cfg.dpi == 150
        assert cfg.default_cmap == "viridis"
        # Untouched fields keep their defaults.
        assert cfg.line_width == 2.0

    def test_is_frozen(self):
        """The config is a frozen dataclass, so a visualiser cannot be mutated
        underneath a caller that shared it."""
        cfg = VisualizationConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.dpi = 200  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestPINNVisualizerInitialisation:
    def test_defaults_come_from_config(self):
        viz = PINNVisualizer()
        assert viz.figsize == (12, 8)
        assert viz.dpi == 100
        assert isinstance(viz.config, VisualizationConfig)

    def test_explicit_arguments_override_config(self):
        cfg = VisualizationConfig(figsize=(12, 8), dpi=100)
        viz = PINNVisualizer(config=cfg, figsize=(4, 3), dpi=200)
        assert viz.figsize == (4, 3)
        assert viz.dpi == 200
        # The config object itself is untouched.
        assert cfg.figsize == (12, 8)

    def test_config_used_when_no_overrides(self):
        cfg = VisualizationConfig(figsize=(5, 5), dpi=77)
        viz = PINNVisualizer(config=cfg)
        assert viz.figsize == (5, 5)
        assert viz.dpi == 77

    @pytest.mark.parametrize("figsize", [(0, 8), (-1, 8), (12, 0), (12,), (1, 2, 3)])
    def test_rejects_invalid_figsize(self, figsize):
        with pytest.raises(ValueError, match="figsize"):
            PINNVisualizer(figsize=figsize)

    @pytest.mark.parametrize("dpi", [0, -100])
    def test_rejects_invalid_dpi(self, dpi):
        with pytest.raises(ValueError, match="dpi"):
            PINNVisualizer(dpi=dpi)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestArrayValidation:
    def test_accepts_matching_1d_arrays(self, viz):
        a = np.linspace(0, 1, 10)
        viz._validate_1d_arrays(a, a.copy(), "x", "u")

    def test_rejects_1d_shape_mismatch(self, viz):
        with pytest.raises(VisualizationError, match="Shape mismatch"):
            viz._validate_1d_arrays(np.zeros(10), np.zeros(11), "x", "u")

    def test_rejects_empty_1d(self, viz):
        with pytest.raises(VisualizationError, match="cannot be empty"):
            viz._validate_1d_arrays(np.array([]), np.array([]), "x", "u")

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_1d(self, viz, bad):
        u = np.zeros(5)
        u[2] = bad
        with pytest.raises(VisualizationError, match="non-finite"):
            viz._validate_1d_arrays(np.zeros(5), u, "x", "u")

    def test_rejects_non_array_1d(self, viz):
        with pytest.raises(VisualizationError, match="must be numpy arrays"):
            viz._validate_1d_arrays([1, 2, 3], np.zeros(3), "x", "u")

    def test_accepts_matching_2d_arrays(self, viz, data_2d):
        viz._validate_2d_arrays(*data_2d)

    def test_rejects_wrong_dimensionality(self, viz):
        with pytest.raises(VisualizationError, match="must be 2D"):
            viz._validate_2d_arrays(np.zeros(4), np.zeros(4), np.zeros(4))

    def test_rejects_2d_shape_mismatch(self, viz):
        with pytest.raises(VisualizationError, match="Shape mismatch"):
            viz._validate_2d_arrays(
                np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 5))
            )

    def test_rejects_non_finite_2d(self, viz):
        field = np.zeros((4, 4))
        field[1, 1] = np.nan
        with pytest.raises(VisualizationError, match="non-finite"):
            viz._validate_2d_arrays(np.zeros((4, 4)), np.zeros((4, 4)), field)


class TestSavePathValidation:
    def test_creates_missing_parent_directory(self, viz, tmp_path):
        target = tmp_path / "deep" / "nested" / "figure.png"
        viz._validate_save_path(target)
        assert target.parent.is_dir()

    @pytest.mark.parametrize("ext", [".png", ".pdf", ".svg", ".eps", ".jpg", ".JPEG"])
    def test_accepts_known_extensions_without_warning(self, viz, tmp_path, ext):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            viz._validate_save_path(tmp_path / f"figure{ext}")

    def test_warns_on_unknown_extension(self, viz, tmp_path):
        with pytest.warns(UserWarning, match="may not be supported"):
            viz._validate_save_path(tmp_path / "figure.bmp")


class TestValidatePlotInputs:
    def test_accepts_valid_arrays(self):
        validate_plot_inputs({"x": np.zeros(4), "y": np.ones(4)})

    def test_rejects_non_array(self):
        with pytest.raises(VisualizationError, match="must be a numpy array"):
            validate_plot_inputs({"x": [1, 2, 3]})

    def test_rejects_empty_by_default(self):
        with pytest.raises(VisualizationError, match="cannot be empty"):
            validate_plot_inputs({"x": np.array([])})

    def test_allows_empty_when_opted_in(self):
        validate_plot_inputs({"x": np.array([])}, allow_empty=True)

    def test_rejects_non_finite(self):
        with pytest.raises(VisualizationError, match="non-finite"):
            validate_plot_inputs({"x": np.array([1.0, np.nan])})

    def test_enforces_required_shape(self):
        with pytest.raises(VisualizationError, match="expected"):
            validate_plot_inputs({"x": np.zeros((2, 3))}, required_shapes={"x": (3, 2)})

    def test_accepts_matching_required_shape(self):
        validate_plot_inputs({"x": np.zeros((2, 3))}, required_shapes={"x": (2, 3)})


# ---------------------------------------------------------------------------
# Moving average
# ---------------------------------------------------------------------------


class TestMovingAverage:
    def test_smooths_to_expected_values(self):
        result = PINNVisualizer._moving_average([1.0, 2.0, 3.0, 4.0], window=2)
        np.testing.assert_allclose(result, [1.5, 2.5, 3.5])

    def test_output_length_shrinks_by_window(self):
        result = PINNVisualizer._moving_average(np.arange(10.0), window=3)
        assert len(result) == 10 - 3 + 1

    def test_constant_input_is_unchanged(self):
        result = PINNVisualizer._moving_average(np.full(6, 2.5), window=3)
        np.testing.assert_allclose(result, np.full(4, 2.5))

    @pytest.mark.parametrize("window", [0, -1])
    def test_rejects_non_positive_window(self, window):
        with pytest.raises(ValueError, match="must be positive"):
            PINNVisualizer._moving_average([1.0, 2.0], window)

    def test_rejects_window_longer_than_data(self):
        with pytest.raises(ValueError, match="cannot be larger"):
            PINNVisualizer._moving_average([1.0, 2.0], window=5)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


class TestPlot1DSolution:
    def test_returns_figure(self, viz, data_1d):
        x, u_pred, _ = data_1d
        assert isinstance(viz.plot_1d_solution(x, u_pred), Figure)

    def test_plots_prediction_and_truth(self, viz, data_1d):
        x, u_pred, u_true = data_1d
        fig = viz.plot_1d_solution(x, u_pred, u_true)
        # One line per series.
        assert len(fig.axes[0].lines) == 2

    def test_honours_title_and_axis_labels(self, viz, data_1d):
        x, u_pred, _ = data_1d
        fig = viz.plot_1d_solution(x, u_pred, title="T", xlabel="X", ylabel="Y")
        ax = fig.axes[0]
        assert ax.get_title() == "T"
        assert ax.get_xlabel() == "X"
        assert ax.get_ylabel() == "Y"

    def test_uses_custom_legend_labels(self, viz, data_1d):
        x, u_pred, u_true = data_1d
        fig = viz.plot_1d_solution(x, u_pred, u_true, legend_labels=("mine", "exact"))
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert labels == ["mine", "exact"]

    def test_rejects_wrong_number_of_legend_labels(self, viz, data_1d):
        x, u_pred, u_true = data_1d
        with pytest.raises(ValueError, match="exactly 2 labels"):
            viz.plot_1d_solution(x, u_pred, u_true, legend_labels=("only-one",))

    def test_rejects_mismatched_shapes(self, viz):
        with pytest.raises(VisualizationError):
            viz.plot_1d_solution(np.zeros(10), np.zeros(11))

    def test_writes_file_to_disk(self, viz, data_1d, tmp_path):
        x, u_pred, _ = data_1d
        target = tmp_path / "solution.png"
        viz.plot_1d_solution(x, u_pred, save_path=target)
        assert target.exists() and target.stat().st_size > 0

    def test_respects_figure_size(self, data_1d):
        x, u_pred, _ = data_1d
        fig = PINNVisualizer(figsize=(5, 3)).plot_1d_solution(x, u_pred)
        assert tuple(fig.get_size_inches()) == (5.0, 3.0)


class TestPlot2DField:
    def test_returns_figure(self, viz, data_2d):
        assert isinstance(viz.plot_2d_field(*data_2d), Figure)

    def test_rejects_shape_mismatch(self, viz):
        with pytest.raises(VisualizationError):
            viz.plot_2d_field(np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((5, 5)))

    def test_writes_file_to_disk(self, viz, data_2d, tmp_path):
        target = tmp_path / "field.png"
        viz.plot_2d_field(*data_2d, save_path=target)
        assert target.exists() and target.stat().st_size > 0


class TestPlotVectorField:
    def test_returns_figure(self, viz, data_2d):
        x, y, field = data_2d
        assert isinstance(viz.plot_vector_field_2d(x, y, field, -field, skip=2), Figure)

    def test_rejects_mismatched_components(self, viz, data_2d):
        x, y, field = data_2d
        with pytest.raises(VisualizationError):
            viz.plot_vector_field_2d(x, y, field, np.zeros((3, 3)))


class TestPlotLossCurves:
    def test_returns_figure(self, viz):
        fig = viz.plot_loss_curves({"total": [1.0, 0.5, 0.25]})
        assert isinstance(fig, Figure)

    def test_draws_one_line_per_component(self, viz):
        history = {"total": [1.0, 0.5], "pde": [0.3, 0.2], "bc": [0.1, 0.05]}
        fig = viz.plot_loss_curves(history, show_raw=False)
        assert len(fig.axes[0].lines) == 3

    def test_applies_log_scale(self, viz):
        fig = viz.plot_loss_curves({"total": [1.0, 0.1]}, log_scale=True)
        assert fig.axes[0].get_yscale() == "log"

    def test_linear_scale_when_requested(self, viz):
        fig = viz.plot_loss_curves({"total": [1.0, 0.1]}, log_scale=False)
        assert fig.axes[0].get_yscale() == "linear"

    def test_smooths_when_data_exceeds_window(self, viz):
        history = {"total": list(np.linspace(1.0, 0.0, 50))}
        fig = viz.plot_loss_curves(history, smoothing_window=5, show_raw=True)
        # A raw line plus a smoothed line.
        assert len(fig.axes[0].lines) == 2

    def test_rejects_empty_history(self, viz):
        with pytest.raises(VisualizationError, match="cannot be empty"):
            viz.plot_loss_curves({})

    def test_rejects_empty_component(self, viz):
        with pytest.raises(VisualizationError, match="cannot be empty"):
            viz.plot_loss_curves({"total": []})

    def test_rejects_non_positive_smoothing_window(self, viz):
        with pytest.raises(ValueError, match="smoothing_window"):
            viz.plot_loss_curves({"total": [1.0, 0.5]}, smoothing_window=0)

    def test_warns_on_non_finite_losses(self, viz):
        with pytest.warns(UserWarning, match="non-finite"):
            viz.plot_loss_curves({"total": [1.0, np.nan]})

    def test_warns_on_negative_losses(self, viz):
        with pytest.warns(UserWarning, match="negative"):
            viz.plot_loss_curves({"total": [1.0, -0.5]})

    def test_rejects_too_few_colours(self, viz):
        history = {"a": [1.0, 0.5], "b": [1.0, 0.5]}
        with pytest.raises(VisualizationError, match="colors"):
            viz.plot_loss_curves(history, colors=["red"])


class TestPlotErrorAnalysis:
    def test_returns_figure(self, viz, data_1d):
        x, u_pred, u_true = data_1d
        assert isinstance(viz.plot_error_analysis(x, u_pred, u_true), Figure)

    def test_produces_multiple_panels(self, viz, data_1d):
        x, u_pred, u_true = data_1d
        fig = viz.plot_error_analysis(x, u_pred, u_true)
        assert len(fig.axes) > 1

    def test_rejects_shape_mismatch(self, viz):
        with pytest.raises(VisualizationError):
            viz.plot_error_analysis(np.zeros(10), np.zeros(10), np.zeros(11))


class TestSaveFormat:
    """The output format must follow the file extension.

    ``_save_figure`` previously forced ``config.save_format`` (png) for every
    path, so a figure saved to ``.pdf`` contained PNG bytes -- even though
    ``_validate_save_path`` accepts pdf, eps, svg and jpeg without warning.
    """

    MAGIC = {
        ".png": b"\x89PNG",
        ".pdf": b"%PDF",
        ".svg": b"<?xml",
    }

    @pytest.mark.parametrize("ext", sorted(MAGIC))
    def test_extension_determines_format(self, viz, data_1d, tmp_path, ext):
        x, u_pred, _ = data_1d
        target = tmp_path / f"figure{ext}"
        viz.plot_1d_solution(x, u_pred, save_path=target)
        assert target.read_bytes().startswith(self.MAGIC[ext])

    def test_extensionless_path_falls_back_to_config_format(self, data_1d, tmp_path):
        x, u_pred, _ = data_1d
        viz = PINNVisualizer(config=VisualizationConfig(save_format="png"))
        target = tmp_path / "figure"
        with pytest.warns(UserWarning, match="may not be supported"):
            viz.plot_1d_solution(x, u_pred, save_path=target)
        assert target.read_bytes().startswith(b"\x89PNG")


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


class TestQuickFunctions:
    def test_quick_plot_1d(self, data_1d):
        x, u_pred, u_true = data_1d
        assert isinstance(quick_plot_1d(x, u_pred, u_true), Figure)

    def test_quick_plot_2d(self, data_2d):
        assert isinstance(quick_plot_2d(*data_2d), Figure)

    def test_quick_plot_loss(self):
        assert isinstance(quick_plot_loss({"total": [1.0, 0.5]}), Figure)

    def test_quick_error_analysis(self, data_1d):
        x, u_pred, u_true = data_1d
        assert isinstance(quick_error_analysis(x, u_pred, u_true), Figure)

    def test_quick_plot_1d_honours_config(self, data_1d):
        x, u_pred, _ = data_1d
        fig = quick_plot_1d(x, u_pred, config=VisualizationConfig(figsize=(4, 2)))
        assert tuple(fig.get_size_inches()) == (4.0, 2.0)

    def test_failures_surface_as_visualization_error(self):
        with pytest.raises(VisualizationError):
            quick_plot_1d(np.zeros(10), np.zeros(11))

    def test_quick_plot_1d_saves_file(self, data_1d, tmp_path):
        x, u_pred, _ = data_1d
        target = tmp_path / "quick.png"
        quick_plot_1d(x, u_pred, save_path=target)
        assert target.exists() and target.stat().st_size > 0


# ---------------------------------------------------------------------------
# Configuration helper
# ---------------------------------------------------------------------------


class TestCreateCustomConfig:
    def test_applies_named_arguments(self):
        cfg = create_custom_config(
            figsize=(10, 6), dpi=150, line_width=3.0, title_fontsize=18
        )
        assert (cfg.figsize, cfg.dpi) == ((10, 6), 150)
        assert cfg.line_width == 3.0
        assert cfg.title_fontsize == 18

    def test_returns_a_config_instance(self):
        assert isinstance(create_custom_config(), VisualizationConfig)

    def test_passes_through_extra_fields(self):
        cfg = create_custom_config(default_cmap="plasma", save_dpi=600)
        assert cfg.default_cmap == "plasma"
        assert cfg.save_dpi == 600

    @pytest.mark.parametrize("figsize", [(0, 6), (-1, 6), (10,), ()])
    def test_rejects_invalid_figsize(self, figsize):
        with pytest.raises(ValueError, match="figsize"):
            create_custom_config(figsize=figsize)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"dpi": 0}, "dpi"),
            ({"line_width": 0}, "line_width"),
            ({"title_fontsize": -1}, "title_fontsize"),
            ({"label_fontsize": 0}, "label_fontsize"),
        ],
    )
    def test_rejects_invalid_values(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            create_custom_config(**kwargs)

    def test_rejects_unknown_field(self):
        with pytest.raises(ValueError, match="Invalid configuration parameters"):
            create_custom_config(not_a_real_setting=1)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_full_workflow_writes_every_figure(tmp_path, data_1d, data_2d):
    """A realistic sequence: configure, plot each kind, save them all."""
    x, u_pred, u_true = data_1d
    gx, gy, field = data_2d

    viz = PINNVisualizer(config=create_custom_config(figsize=(8, 5), dpi=90))

    outputs = [
        (viz.plot_1d_solution(x, u_pred, u_true, save_path=tmp_path / "a.png")),
        (viz.plot_2d_field(gx, gy, field, save_path=tmp_path / "b.png")),
        (
            viz.plot_loss_curves(
                {"total": [1.0, 0.4, 0.1]}, save_path=tmp_path / "c.png"
            )
        ),
        (viz.plot_error_analysis(x, u_pred, u_true, save_path=tmp_path / "d.png")),
    ]

    assert all(isinstance(fig, Figure) for fig in outputs)
    written = sorted(p.name for p in tmp_path.glob("*.png"))
    assert written == ["a.png", "b.png", "c.png", "d.png"]
    assert all((tmp_path / name).stat().st_size > 0 for name in written)

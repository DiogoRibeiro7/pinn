"""
Comprehensive metrics and error analysis utilities for Physics-Informed Neural Networks.

This module provides various metrics for evaluating PINN performance:
- Point-wise error metrics (MAE, MSE, RMSE, relative errors)
- Field-based metrics (energy norms, conservation laws)
- Physics consistency metrics (PDE residual analysis)
- Statistical analysis tools (distribution comparisons, correlation analysis)
- Convergence analysis utilities
- Benchmark comparison tools

These metrics are essential for validating PINN solutions and comparing
different architectures, training strategies, and hyperparameters.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
import warnings

try:
    # integrate is imported to probe SciPy's availability, not used directly.
    from scipy import stats, integrate  # noqa: F401
    from scipy.spatial.distance import wasserstein_distance

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Some advanced metrics will be unavailable.")


# ============================================================
# Core Error Metrics
# ============================================================


@dataclass
class ErrorMetrics:
    """
    Container for various error metrics.

    Attributes:
        mae: Mean Absolute Error
        mse: Mean Squared Error
        rmse: Root Mean Squared Error
        max_error: Maximum absolute error
        relative_mae: Mean Absolute Relative Error
        relative_rmse: Root Mean Squared Relative Error
        l2_norm: L2 norm of error
        l_inf_norm: L-infinity norm of error
        correlation: Pearson correlation coefficient
        r2_score: R-squared coefficient of determination
    """

    mae: float
    mse: float
    rmse: float
    max_error: float
    relative_mae: float
    relative_rmse: float
    l2_norm: float
    l_inf_norm: float
    correlation: float
    r2_score: float

    def __str__(self) -> str:
        """String representation of error metrics."""
        return f"""Error Metrics:
  MAE:           {self.mae:.6e}
  RMSE:          {self.rmse:.6e}
  Max Error:     {self.max_error:.6e}
  Relative MAE:  {self.relative_mae:.6e}
  Relative RMSE: {self.relative_rmse:.6e}
  L2 Norm:       {self.l2_norm:.6e}
  L∞ Norm:       {self.l_inf_norm:.6e}
  Correlation:   {self.correlation:.6f}
  R² Score:      {self.r2_score:.6f}"""


def compute_error_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: Optional[np.ndarray] = None,
    eps: float = 1e-10,
) -> ErrorMetrics:
    """
    Compute comprehensive error metrics between true and predicted values.

    Args:
        y_true: True values, shape (N,) or (N, d)
        y_pred: Predicted values, same shape as y_true
        weights: Optional weights for weighted metrics, shape (N,)
        eps: Small value to avoid division by zero

    Returns:
        ErrorMetrics object containing all computed metrics

    Raises:
        ValueError: If arrays have incompatible shapes
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    # Flatten arrays for easier computation
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()

    if weights is not None:
        if weights.shape[0] != y_true.shape[0]:
            raise ValueError(
                "Weights must have same length as first dimension of y_true"
            )
        weights_flat = np.repeat(weights, y_true.shape[1] if y_true.ndim > 1 else 1)
    else:
        weights_flat = None

    # Absolute errors
    abs_errors = np.abs(y_pred_flat - y_true_flat)
    squared_errors = (y_pred_flat - y_true_flat) ** 2

    # Basic metrics
    if weights_flat is not None:
        mae = np.average(abs_errors, weights=weights_flat)
        mse = np.average(squared_errors, weights=weights_flat)
    else:
        mae = np.mean(abs_errors)
        mse = np.mean(squared_errors)

    rmse = np.sqrt(mse)
    max_error = np.max(abs_errors)

    # Relative errors (avoid division by zero)
    y_true_safe = np.where(np.abs(y_true_flat) < eps, eps, y_true_flat)
    relative_errors = abs_errors / np.abs(y_true_safe)
    relative_squared_errors = squared_errors / (y_true_safe**2)

    if weights_flat is not None:
        relative_mae = np.average(relative_errors, weights=weights_flat)
        relative_mse = np.average(relative_squared_errors, weights=weights_flat)
    else:
        relative_mae = np.mean(relative_errors)
        relative_mse = np.mean(relative_squared_errors)

    relative_rmse = np.sqrt(relative_mse)

    # Norms
    l2_norm = np.linalg.norm(y_pred_flat - y_true_flat)
    l_inf_norm = np.max(abs_errors)  # Same as max_error

    # Statistical metrics
    if len(y_true_flat) > 1:
        correlation = np.corrcoef(y_true_flat, y_pred_flat)[0, 1]

        # R² score
        ss_res = np.sum(squared_errors)
        ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
        r2_score = 1 - (ss_res / (ss_tot + eps))
    else:
        correlation = 1.0 if abs_errors[0] < eps else 0.0
        r2_score = 1.0 if abs_errors[0] < eps else -np.inf

    return ErrorMetrics(
        mae=mae,
        mse=mse,
        rmse=rmse,
        max_error=max_error,
        relative_mae=relative_mae,
        relative_rmse=relative_rmse,
        l2_norm=l2_norm,
        l_inf_norm=l_inf_norm,
        correlation=correlation if not np.isnan(correlation) else 0.0,
        r2_score=r2_score,
    )


# ============================================================
# Physics-Based Metrics
# ============================================================


@dataclass
class PhysicsMetrics:
    """
    Container for physics-based validation metrics.

    Attributes:
        mean_residual: Mean PDE residual magnitude
        max_residual: Maximum PDE residual magnitude
        residual_std: Standard deviation of residuals
        conservation_error: Error in conservation laws
        energy_error: Error in energy conservation
        mass_error: Error in mass conservation
        momentum_error: Error in momentum conservation
    """

    mean_residual: float
    max_residual: float
    residual_std: float
    conservation_error: Optional[float] = None
    energy_error: Optional[float] = None
    mass_error: Optional[float] = None
    momentum_error: Optional[float] = None

    def __str__(self) -> str:
        """String representation of physics metrics."""
        result = f"""Physics Metrics:
  Mean Residual: {self.mean_residual:.6e}
  Max Residual:  {self.max_residual:.6e}
  Residual Std:  {self.residual_std:.6e}"""

        if self.conservation_error is not None:
            result += f"\n  Conservation:  {self.conservation_error:.6e}"
        if self.energy_error is not None:
            result += f"\n  Energy Error:  {self.energy_error:.6e}"
        if self.mass_error is not None:
            result += f"\n  Mass Error:    {self.mass_error:.6e}"
        if self.momentum_error is not None:
            result += f"\n  Momentum Error: {self.momentum_error:.6e}"

        return result


def compute_residual_metrics(residuals: np.ndarray) -> PhysicsMetrics:
    """
    Compute metrics for PDE residuals.

    Args:
        residuals: PDE residual values, shape (N,) or (N, n_equations)

    Returns:
        PhysicsMetrics object with residual statistics
    """
    if residuals.ndim > 1:
        # Multiple equations - compute magnitude
        residual_magnitudes = np.linalg.norm(residuals, axis=1)
    else:
        residual_magnitudes = np.abs(residuals)

    return PhysicsMetrics(
        mean_residual=np.mean(residual_magnitudes),
        max_residual=np.max(residual_magnitudes),
        residual_std=np.std(residual_magnitudes),
    )


def compute_conservation_metrics(
    solution: np.ndarray,
    domain_coords: np.ndarray,
    conservation_laws: List[Callable[[np.ndarray, np.ndarray], float]],
) -> Dict[str, float]:
    """
    Compute conservation law violations.

    Args:
        solution: Solution field values
        domain_coords: Coordinate arrays for integration
        conservation_laws: List of functions computing conserved quantities

    Returns:
        Dictionary of conservation errors
    """
    conservation_errors = {}

    for i, law in enumerate(conservation_laws):
        try:
            # Compute initial and current conserved quantity
            # This is a simplified interface - real implementation would need
            # access to initial conditions and proper integration
            conserved_quantity = law(solution, domain_coords)
            conservation_errors[f"conservation_law_{i}"] = conserved_quantity
        except Exception as e:
            warnings.warn(f"Could not compute conservation law {i}: {e}")
            conservation_errors[f"conservation_law_{i}"] = np.nan

    return conservation_errors


# ============================================================
# Field-Based Metrics
# ============================================================


class FieldMetrics:
    """
    Metrics for vector/tensor fields and spatial analysis.
    """

    @staticmethod
    def compute_energy_norm(
        u_true: np.ndarray,
        u_pred: np.ndarray,
        domain_volume: float,
        weights: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute L2 energy norm error.

        Args:
            u_true: True field values
            u_pred: Predicted field values
            domain_volume: Volume of computational domain
            weights: Integration weights

        Returns:
            Energy norm error
        """
        error = u_pred - u_true

        if weights is not None:
            energy_error = (
                np.sum(weights * (error**2)) * domain_volume / np.sum(weights)
            )
        else:
            energy_error = np.mean(error**2) * domain_volume

        return np.sqrt(energy_error)

    @staticmethod
    def compute_sobolev_norm(
        u_true: np.ndarray,
        u_pred: np.ndarray,
        gradients_true: np.ndarray,
        gradients_pred: np.ndarray,
        domain_volume: float,
    ) -> float:
        """
        Compute H¹ Sobolev norm error.

        Args:
            u_true: True field values
            u_pred: Predicted field values
            gradients_true: True gradient values
            gradients_pred: Predicted gradient values
            domain_volume: Domain volume

        Returns:
            H¹ norm error
        """
        # L² norm of function error
        l2_error = np.mean((u_pred - u_true) ** 2) * domain_volume

        # L² norm of gradient error
        grad_error = gradients_pred - gradients_true
        h1_semi_error = np.mean(np.sum(grad_error**2, axis=-1)) * domain_volume

        return np.sqrt(l2_error + h1_semi_error)

    @staticmethod
    def compute_divergence_error(
        velocity_field: np.ndarray, coordinates: List[np.ndarray]
    ) -> float:
        """
        Compute divergence error for incompressible flow fields.

        Args:
            velocity_field: Velocity components, shape (..., ndim)
            coordinates: List of coordinate arrays for each dimension

        Returns:
            RMS divergence error
        """
        if not SCIPY_AVAILABLE:
            warnings.warn("SciPy required for gradient computation")
            return np.nan

        ndim = velocity_field.shape[-1]
        divergence = np.zeros(velocity_field.shape[:-1])

        for i in range(ndim):
            # Compute gradient of i-th velocity component in i-th direction
            grad = np.gradient(velocity_field[..., i], coordinates[i], axis=i)
            divergence += grad

        return np.sqrt(np.mean(divergence**2))


# ============================================================
# Statistical Analysis
# ============================================================


class StatisticalAnalysis:
    """
    Statistical analysis tools for PINN validation.
    """

    @staticmethod
    def compute_distribution_distance(
        y_true: np.ndarray, y_pred: np.ndarray, method: str = "wasserstein"
    ) -> float:
        """
        Compute distance between true and predicted distributions.

        Args:
            y_true: True values
            y_pred: Predicted values
            method: Distance metric ("wasserstein", "ks", "anderson")

        Returns:
            Distribution distance
        """
        if not SCIPY_AVAILABLE:
            warnings.warn("SciPy required for distribution analysis")
            return np.nan

        y_true_flat = y_true.ravel()
        y_pred_flat = y_pred.ravel()

        if method == "wasserstein":
            return wasserstein_distance(y_true_flat, y_pred_flat)
        elif method == "ks":
            # Kolmogorov-Smirnov test
            statistic, _ = stats.ks_2samp(y_true_flat, y_pred_flat)
            return statistic
        elif method == "anderson":
            # Anderson-Darling test (requires same distribution)
            try:
                statistic, _, _ = stats.anderson_ksamp([y_true_flat, y_pred_flat])
                return statistic
            except Exception:
                return np.nan
        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def compute_moment_errors(
        y_true: np.ndarray, y_pred: np.ndarray, moments: List[int] = [1, 2, 3, 4]
    ) -> Dict[int, float]:
        """
        Compute errors in statistical moments.

        Args:
            y_true: True values
            y_pred: Predicted values
            moments: List of moment orders to compute

        Returns:
            Dictionary mapping moment order to relative error
        """
        y_true_flat = y_true.ravel()
        y_pred_flat = y_pred.ravel()

        moment_errors = {}

        for k in moments:
            true_moment = np.mean(y_true_flat**k)
            pred_moment = np.mean(y_pred_flat**k)

            if np.abs(true_moment) > 1e-10:
                relative_error = np.abs(pred_moment - true_moment) / np.abs(true_moment)
            else:
                relative_error = np.abs(pred_moment - true_moment)

            moment_errors[k] = relative_error

        return moment_errors

    @staticmethod
    def compute_quantile_errors(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
    ) -> Dict[float, float]:
        """
        Compute errors in distribution quantiles.

        Args:
            y_true: True values
            y_pred: Predicted values
            quantiles: List of quantiles to analyze

        Returns:
            Dictionary mapping quantile to absolute error
        """
        y_true_flat = y_true.ravel()
        y_pred_flat = y_pred.ravel()

        quantile_errors = {}

        for q in quantiles:
            true_quantile = np.quantile(y_true_flat, q)
            pred_quantile = np.quantile(y_pred_flat, q)
            quantile_errors[q] = np.abs(pred_quantile - true_quantile)

        return quantile_errors


# ============================================================
# Convergence Analysis
# ============================================================


@dataclass
class ConvergenceRate:
    """
    Container for convergence analysis results.

    Attributes:
        rates: Convergence rates for different metrics
        r_squared: R² values for convergence fits
        extrapolated_error: Extrapolated error at infinite resolution
    """

    rates: Dict[str, float]
    r_squared: Dict[str, float]
    extrapolated_error: Dict[str, float]


class ConvergenceAnalysis:
    """
    Tools for analyzing convergence rates and extrapolation.
    """

    @staticmethod
    def analyze_convergence(
        resolutions: List[int],
        errors: Dict[str, List[float]],
        fit_method: str = "log_linear",
    ) -> ConvergenceRate:
        """
        Analyze convergence rates from resolution study.

        Args:
            resolutions: List of resolution values (e.g., number of points)
            errors: Dictionary mapping error metric names to error lists
            fit_method: Fitting method ("log_linear", "power_law")

        Returns:
            ConvergenceRate object with analysis results
        """
        if len(resolutions) != len(next(iter(errors.values()))):
            raise ValueError("Resolutions and error lists must have same length")

        rates = {}
        r_squared = {}
        extrapolated_error = {}

        for metric_name, error_list in errors.items():
            if fit_method == "log_linear":
                # Fit log(error) = a*log(resolution) + b
                log_res = np.log(resolutions)
                log_err = np.log(error_list)

                # Remove any infinite or NaN values
                valid_mask = np.isfinite(log_err) & np.isfinite(log_res)
                if np.sum(valid_mask) < 2:
                    rates[metric_name] = np.nan
                    r_squared[metric_name] = np.nan
                    extrapolated_error[metric_name] = np.nan
                    continue

                coeffs = np.polyfit(log_res[valid_mask], log_err[valid_mask], 1)
                slope, intercept = coeffs

                # Convergence rate is negative of slope
                rates[metric_name] = -slope

                # R² value
                y_pred = slope * log_res[valid_mask] + intercept
                ss_res = np.sum((log_err[valid_mask] - y_pred) ** 2)
                ss_tot = np.sum(
                    (log_err[valid_mask] - np.mean(log_err[valid_mask])) ** 2
                )
                r_squared[metric_name] = 1 - ss_res / (ss_tot + 1e-10)

                # Extrapolated error (at infinite resolution is zero for convergent methods)
                extrapolated_error[metric_name] = 0.0

            elif fit_method == "power_law":
                # Fit error = a * resolution^(-b)
                # This requires non-linear fitting - simplified here
                rates[metric_name] = np.nan
                r_squared[metric_name] = np.nan
                extrapolated_error[metric_name] = np.nan

        return ConvergenceRate(rates, r_squared, extrapolated_error)

    @staticmethod
    def richardson_extrapolation(
        coarse_solution: np.ndarray,
        fine_solution: np.ndarray,
        refinement_ratio: float = 2.0,
        order: float = 2.0,
    ) -> np.ndarray:
        """
        Apply Richardson extrapolation for improved accuracy.

        Args:
            coarse_solution: Solution on coarse grid
            fine_solution: Solution on fine grid (same spatial locations as coarse)
            refinement_ratio: Ratio of fine to coarse resolution
            order: Expected order of convergence

        Returns:
            Extrapolated solution with higher accuracy
        """
        factor = refinement_ratio**order
        extrapolated = fine_solution + (fine_solution - coarse_solution) / (factor - 1)
        return extrapolated


# ============================================================
# Benchmark Comparison
# ============================================================


class BenchmarkComparison:
    """
    Tools for comparing PINN results against benchmarks.
    """

    def __init__(self):
        """Initialize benchmark comparison."""
        self.benchmarks: Dict[str, Dict[str, Any]] = {}

    def add_benchmark(
        self,
        name: str,
        solution: np.ndarray,
        coordinates: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a benchmark solution.

        Args:
            name: Benchmark name
            solution: Benchmark solution values
            coordinates: Coordinate points
            metadata: Optional metadata (method, parameters, etc.)
        """
        self.benchmarks[name] = {
            "solution": solution,
            "coordinates": coordinates,
            "metadata": metadata or {},
        }

    def compare_against_benchmark(
        self,
        pinn_solution: np.ndarray,
        benchmark_name: str,
        metrics: List[str] = ["mae", "rmse", "max_error"],
    ) -> Dict[str, float]:
        """
        Compare PINN solution against a benchmark.

        Args:
            pinn_solution: PINN solution values
            benchmark_name: Name of benchmark to compare against
            metrics: List of metrics to compute

        Returns:
            Dictionary of computed metrics
        """
        if benchmark_name not in self.benchmarks:
            raise ValueError(f"Benchmark {benchmark_name} not found")

        benchmark_solution = self.benchmarks[benchmark_name]["solution"]

        # Ensure compatible shapes
        if pinn_solution.shape != benchmark_solution.shape:
            raise ValueError("PINN and benchmark solutions must have same shape")

        # Compute requested metrics
        error_metrics = compute_error_metrics(benchmark_solution, pinn_solution)

        result = {}
        for metric in metrics:
            if hasattr(error_metrics, metric):
                result[metric] = getattr(error_metrics, metric)
            else:
                warnings.warn(f"Unknown metric: {metric}")

        return result

    def compare_multiple_methods(
        self,
        solutions: Dict[str, np.ndarray],
        reference_name: str,
        metrics: List[str] = ["mae", "rmse", "correlation"],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple methods against a reference solution.

        Args:
            solutions: Dictionary mapping method names to solution arrays
            reference_name: Name of reference solution in solutions dict
            metrics: List of metrics to compute

        Returns:
            Nested dictionary: {method_name: {metric: value}}
        """
        if reference_name not in solutions:
            raise ValueError(f"Reference solution {reference_name} not found")

        reference_solution = solutions[reference_name]
        comparison_results = {}

        for method_name, solution in solutions.items():
            if method_name == reference_name:
                continue

            error_metrics = compute_error_metrics(reference_solution, solution)

            method_results = {}
            for metric in metrics:
                if hasattr(error_metrics, metric):
                    method_results[metric] = getattr(error_metrics, metric)

            comparison_results[method_name] = method_results

        return comparison_results


# ============================================================
# Comprehensive Analysis Functions
# ============================================================


def comprehensive_error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    coordinates: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    residuals: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Perform comprehensive error analysis.

    Args:
        y_true: True solution values
        y_pred: Predicted solution values
        coordinates: Coordinate arrays for field analysis
        weights: Optional weights for integration
        residuals: Optional PDE residuals

    Returns:
        Dictionary containing all analysis results
    """
    results = {}

    # Basic error metrics
    results["error_metrics"] = compute_error_metrics(y_true, y_pred, weights)

    # Statistical analysis
    stat_analysis = StatisticalAnalysis()
    results["distribution_distance"] = stat_analysis.compute_distribution_distance(
        y_true, y_pred
    )
    results["moment_errors"] = stat_analysis.compute_moment_errors(y_true, y_pred)
    results["quantile_errors"] = stat_analysis.compute_quantile_errors(y_true, y_pred)

    # Physics metrics (if residuals provided)
    if residuals is not None:
        results["physics_metrics"] = compute_residual_metrics(residuals)

    # Field metrics (if coordinates provided)
    if coordinates is not None and len(coordinates.shape) > 1:
        try:
            domain_volume = 1.0  # Simplified - should compute actual volume
            results["energy_norm"] = FieldMetrics.compute_energy_norm(
                y_true, y_pred, domain_volume, weights
            )
        except Exception as e:
            warnings.warn(f"Could not compute field metrics: {e}")

    return results


def generate_metrics_report(
    analysis_results: Dict[str, Any], title: str = "PINN Analysis Report"
) -> str:
    """
    Generate a formatted text report from analysis results.

    Args:
        analysis_results: Results from comprehensive_error_analysis
        title: Report title

    Returns:
        Formatted text report
    """
    report = [f"\n{'=' * 60}", f"{title:^60}", f"{'=' * 60}\n"]

    # Error metrics
    if "error_metrics" in analysis_results:
        report.append(str(analysis_results["error_metrics"]))
        report.append("")

    # Physics metrics
    if "physics_metrics" in analysis_results:
        report.append(str(analysis_results["physics_metrics"]))
        report.append("")

    # Statistical analysis
    if "distribution_distance" in analysis_results:
        report.append(
            f"Distribution Distance: {analysis_results['distribution_distance']:.6e}"
        )

    if "moment_errors" in analysis_results:
        report.append("Moment Errors:")
        for order, error in analysis_results["moment_errors"].items():
            report.append(f"  Moment {order}: {error:.6e}")
        report.append("")

    # Field metrics
    if "energy_norm" in analysis_results:
        report.append(f"Energy Norm Error: {analysis_results['energy_norm']:.6e}")
        report.append("")

    return "\n".join(report)


# ============================================================
# Convenience Functions
# ============================================================


def quick_error_summary(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """
    Quick error summary for basic validation.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Formatted error summary string
    """
    metrics = compute_error_metrics(y_true, y_pred)

    return f"""Quick Error Summary:
MAE: {metrics.mae:.3e} | RMSE: {metrics.rmse:.3e} | Max: {metrics.max_error:.3e}
Correlation: {metrics.correlation:.4f} | R²: {metrics.r2_score:.4f}"""


def validate_pinn_solution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    residuals: np.ndarray,
    tolerance: Dict[str, float] = None,
) -> Tuple[bool, Dict[str, bool], str]:
    """
    Validate PINN solution against tolerance criteria.

    Args:
        y_true: True solution
        y_pred: PINN prediction
        residuals: PDE residuals
        tolerance: Dictionary of tolerance values for different metrics

    Returns:
        Tuple of (overall_pass, individual_results, summary_message)
    """
    if tolerance is None:
        tolerance = {
            "rmse": 1e-2,
            "max_error": 1e-1,
            "mean_residual": 1e-3,
            "correlation": 0.95,
        }

    # Compute metrics
    error_metrics = compute_error_metrics(y_true, y_pred)
    physics_metrics = compute_residual_metrics(residuals)

    # Check each tolerance
    results = {}
    results["rmse"] = error_metrics.rmse <= tolerance.get("rmse", np.inf)
    results["max_error"] = error_metrics.max_error <= tolerance.get("max_error", np.inf)
    results["mean_residual"] = physics_metrics.mean_residual <= tolerance.get(
        "mean_residual", np.inf
    )
    results["correlation"] = error_metrics.correlation >= tolerance.get(
        "correlation", -np.inf
    )

    # Overall pass/fail
    overall_pass = all(results.values())

    # Summary message
    passed = sum(results.values())
    total = len(results)
    summary = f"Validation: {passed}/{total} criteria passed. Overall: {'PASS' if overall_pass else 'FAIL'}"

    return overall_pass, results, summary

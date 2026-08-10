"""
Comprehensive error handling system for PINN repository.

This module provides:
- Custom exception hierarchy
- Error context management
- Automatic error recovery strategies
- Detailed error reporting and logging
- User-friendly error messages
- Debug information collection

Features:
- Structured exception types for different error categories
- Error context tracking for better debugging
- Automatic retry mechanisms
- Resource cleanup on errors
- Integration with logging system
- Performance impact monitoring
"""

from __future__ import annotations

import sys
import traceback
import functools
import logging
import time

try:  # Optional dependency, matching logging.py, profiling.py and the rest
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional
    psutil = None  # type: ignore
from typing import Dict, Any, Optional, Callable, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager
from enum import Enum
import numpy as np
import torch


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""

    CONFIGURATION = "configuration"
    DATA_VALIDATION = "data_validation"
    NETWORK_ARCHITECTURE = "network_architecture"
    TRAINING = "training"
    VISUALIZATION = "visualization"
    IO = "io"
    COMPUTATION = "computation"
    MEMORY = "memory"
    HARDWARE = "hardware"
    UNKNOWN = "unknown"


# ============================================================
# Custom Exception Hierarchy
# ============================================================


@dataclass
class ErrorContext:
    """Context information for errors."""

    # Basic information
    timestamp: float = field(default_factory=time.time)
    function_name: Optional[str] = None
    module_name: Optional[str] = None
    line_number: Optional[int] = None

    # System information
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    gpu_memory: Optional[float] = None

    # PINN-specific context
    training_step: Optional[int] = None
    epoch: Optional[int] = None
    loss_values: Optional[Dict[str, float]] = None
    model_state: Optional[Dict[str, Any]] = None

    # Additional context
    user_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "timestamp": self.timestamp,
            "function_name": self.function_name,
            "module_name": self.module_name,
            "line_number": self.line_number,
            "memory_usage": self.memory_usage,
            "cpu_usage": self.cpu_usage,
            "gpu_memory": self.gpu_memory,
            "training_step": self.training_step,
            "epoch": self.epoch,
            "loss_values": self.loss_values,
            "model_state": self.model_state,
            "user_data": self.user_data,
        }


class PINNError(Exception):
    """
    Base exception class for all PINN-related errors.

    Provides structured error information including context,
    severity, category, and potential recovery suggestions.
    """

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        suggestions: Optional[List[str]] = None,
        original_exception: Optional[Exception] = None,
    ):
        """
        Initialize PINN error.

        Args:
            message: Human-readable error message
            category: Error category for classification
            severity: Error severity level
            context: Additional context information
            suggestions: List of suggested solutions
            original_exception: Original exception if this is a wrapper
        """
        super().__init__(message)

        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or ErrorContext()
        self.suggestions = suggestions or []
        self.original_exception = original_exception

        # Automatically capture system context
        self._capture_system_context()

        # Automatically capture stack trace
        self.stack_trace = traceback.format_exc()

        # Capture function context
        frame = sys._getframe(1)
        self.context.function_name = frame.f_code.co_name
        self.context.module_name = frame.f_globals.get("__name__")
        self.context.line_number = frame.f_lineno

    def _capture_system_context(self) -> None:
        """Capture current system state."""
        try:
            if psutil is not None:
                # Memory usage
                process = psutil.Process()
                self.context.memory_usage = (
                    process.memory_info().rss / 1024 / 1024
                )  # MB

                # CPU usage
                self.context.cpu_usage = psutil.cpu_percent()

            # GPU memory if available
            if torch.cuda.is_available():
                self.context.gpu_memory = (
                    torch.cuda.memory_allocated() / 1024 / 1024
                )  # MB

        except Exception:
            # Don't let context capture fail the original error
            pass

    def add_suggestion(self, suggestion: str) -> None:
        """Add a recovery suggestion."""
        if suggestion not in self.suggestions:
            self.suggestions.append(suggestion)

    def add_context(self, key: str, value: Any) -> None:
        """Add additional context information."""
        self.context.user_data[key] = value

    def get_formatted_message(self) -> str:
        """Get formatted error message with context and suggestions."""
        lines = [
            f"❌ {self.category.value.title()} Error ({self.severity.value}): {self.message}",
            f"📍 Location: {self.context.module_name}.{self.context.function_name}:{self.context.line_number}",
        ]

        # Add system information if available
        if self.context.memory_usage:
            lines.append(f"💾 Memory: {self.context.memory_usage:.1f} MB")
        if self.context.cpu_usage:
            lines.append(f"🖥️  CPU: {self.context.cpu_usage:.1f}%")
        if self.context.gpu_memory:
            lines.append(f"🎮 GPU Memory: {self.context.gpu_memory:.1f} MB")

        # Add training context if available
        if self.context.training_step:
            lines.append(f"🏃 Training Step: {self.context.training_step}")
        if self.context.loss_values:
            loss_str = ", ".join(
                f"{k}: {v:.2e}" for k, v in self.context.loss_values.items()
            )
            lines.append(f"📉 Loss Values: {loss_str}")

        # Add suggestions
        if self.suggestions:
            lines.append("💡 Suggestions:")
            for suggestion in self.suggestions:
                lines.append(f"   • {suggestion}")

        return "\n".join(lines)

    def __str__(self) -> str:
        """String representation of the error."""
        return self.get_formatted_message()


# ============================================================
# Specific Exception Types
# ============================================================


class ConfigurationError(PINNError):
    """Errors related to configuration issues."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )


class DataValidationError(PINNError):
    """Errors related to data validation."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.DATA_VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            **kwargs,
        )


class NetworkArchitectureError(PINNError):
    """Errors related to neural network architecture."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.NETWORK_ARCHITECTURE,
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )


class TrainingError(PINNError):
    """Errors related to training process."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.TRAINING,
            severity=ErrorSeverity.MEDIUM,
            **kwargs,
        )


class VisualizationError(PINNError):
    """Errors related to visualization."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.VISUALIZATION,
            severity=ErrorSeverity.LOW,
            **kwargs,
        )


class ComputationError(PINNError):
    """Errors related to numerical computations."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.COMPUTATION,
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )


class MemoryError(PINNError):
    """Errors related to memory issues."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.MEMORY,
            severity=ErrorSeverity.CRITICAL,
            **kwargs,
        )


class HardwareError(PINNError):
    """Errors related to hardware issues."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.HARDWARE,
            severity=ErrorSeverity.CRITICAL,
            **kwargs,
        )


# ============================================================
# Error Handler Registry
# ============================================================


class ErrorHandler:
    """Base class for error handlers."""

    def can_handle(self, error: Exception) -> bool:
        """Check if this handler can handle the given error."""
        return False

    def handle(self, error: Exception) -> Tuple[bool, Optional[Any]]:
        """
        Handle the error.

        Returns:
            Tuple of (recovery_successful, recovery_result)
        """
        return False, None


class OutOfMemoryHandler(ErrorHandler):
    """Handler for out-of-memory errors."""

    def can_handle(self, error: Exception) -> bool:
        """Check if this is an OOM error."""
        error_msg = str(error).lower()
        return (
            "out of memory" in error_msg
            or "cuda out of memory" in error_msg
            or isinstance(error, torch.cuda.OutOfMemoryError)
            if hasattr(torch.cuda, "OutOfMemoryError")
            else False
        )

    def handle(self, error: Exception) -> Tuple[bool, Optional[Any]]:
        """Handle OOM by clearing cache and suggesting smaller batch size."""
        try:
            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Create helpful error with suggestions
            recovery_error = MemoryError(
                "GPU/CPU out of memory during computation",
                suggestions=[
                    "Reduce batch size or number of collocation points",
                    "Use gradient accumulation to simulate larger batches",
                    "Enable mixed precision training (if using GPU)",
                    "Use a smaller network architecture",
                    "Process data in smaller chunks",
                ],
            )

            return True, recovery_error

        except Exception:
            return False, None


class NumericalInstabilityHandler(ErrorHandler):
    """Handler for numerical instability issues."""

    def can_handle(self, error: Exception) -> bool:
        """Check if this is a numerical stability error."""
        error_msg = str(error).lower()
        numerical_keywords = [
            "nan",
            "inf",
            "overflow",
            "underflow",
            "singular matrix",
            "not converge",
            "diverged",
        ]
        return any(keyword in error_msg for keyword in numerical_keywords)

    def handle(self, error: Exception) -> Tuple[bool, Optional[Any]]:
        """Handle numerical instability."""
        recovery_error = ComputationError(
            "Numerical instability detected during computation",
            suggestions=[
                "Reduce learning rate to improve training stability",
                "Use gradient clipping to prevent exploding gradients",
                "Check for inf/nan values in input data",
                "Increase numerical precision (float64 instead of float32)",
                "Add regularization to the loss function",
                "Use a more stable optimizer (L-BFGS instead of Adam)",
                "Initialize network weights more carefully",
            ],
        )

        return True, recovery_error


class ConfigurationErrorHandler(ErrorHandler):
    """Handler for configuration-related errors."""

    def can_handle(self, error: Exception) -> bool:
        """Check if this is a configuration error."""
        return isinstance(error, (ConfigurationError, ValueError, TypeError))

    def handle(self, error: Exception) -> Tuple[bool, Optional[Any]]:
        """Handle configuration errors."""
        if isinstance(error, ConfigurationError):
            return True, error

        # Wrap other errors as configuration errors
        recovery_error = ConfigurationError(
            f"Configuration issue: {str(error)}",
            suggestions=[
                "Check configuration file syntax and values",
                "Validate all parameter ranges and types",
                "Ensure all required parameters are specified",
                "Review configuration documentation",
                "Use configuration validation before training",
            ],
            original_exception=error,
        )

        return True, recovery_error


class ErrorHandlerRegistry:
    """Registry for managing error handlers."""

    def __init__(self):
        """Initialize with default handlers."""
        self.handlers: List[ErrorHandler] = [
            OutOfMemoryHandler(),
            NumericalInstabilityHandler(),
            ConfigurationErrorHandler(),
        ]

    def register_handler(self, handler: ErrorHandler) -> None:
        """Register a new error handler."""
        self.handlers.append(handler)

    def handle_error(self, error: Exception) -> Tuple[bool, Optional[Exception]]:
        """
        Try to handle an error using registered handlers.

        Returns:
            Tuple of (handled, processed_error)
        """
        for handler in self.handlers:
            if handler.can_handle(error):
                success, result = handler.handle(error)
                if success:
                    return True, result

        # No handler could process the error
        return False, None


# ============================================================
# Error Context Manager
# ============================================================


@contextmanager
def error_context(
    operation: str,
    error_handler_registry: Optional[ErrorHandlerRegistry] = None,
    training_step: Optional[int] = None,
    additional_context: Optional[Dict[str, Any]] = None,
):
    """
    Context manager for enhanced error handling.

    Args:
        operation: Description of the operation being performed
        error_handler_registry: Registry for error handlers
        training_step: Current training step (if applicable)
        additional_context: Additional context information
    """
    registry = error_handler_registry or ErrorHandlerRegistry()
    start_time = time.time()

    try:
        yield

    except Exception as e:
        # Try to handle the error
        handled, processed_error = registry.handle_error(e)

        if handled and processed_error:
            # Add operation context
            if isinstance(processed_error, PINNError):
                processed_error.add_context("operation", operation)
                processed_error.add_context(
                    "operation_duration", time.time() - start_time
                )

                if training_step is not None:
                    processed_error.context.training_step = training_step

                if additional_context:
                    for key, value in additional_context.items():
                        processed_error.add_context(key, value)

            raise processed_error from e
        else:
            # Create a generic PINN error
            context = ErrorContext()
            if training_step is not None:
                context.training_step = training_step

            generic_error = PINNError(
                f"Error during {operation}: {str(e)}",
                context=context,
                original_exception=e,
                suggestions=[
                    f"Check the implementation of {operation}",
                    "Enable debug logging for more information",
                    "Verify input data and parameters",
                    "Consider reducing problem complexity",
                ],
            )

            if additional_context:
                for key, value in additional_context.items():
                    generic_error.add_context(key, value)

            raise generic_error from e


# ============================================================
# Decorators for Error Handling
# ============================================================


def handle_errors(
    operation: Optional[str] = None,
    retry_attempts: int = 0,
    retry_delay: float = 1.0,
    log_errors: bool = True,
):
    """
    Decorator for automatic error handling and retry logic.

    Args:
        operation: Description of the operation (defaults to function name)
        retry_attempts: Number of retry attempts
        retry_delay: Delay between retries in seconds
        log_errors: Whether to log errors
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            operation_name = operation or func.__name__
            logger = logging.getLogger(func.__module__)

            last_exception = None

            for attempt in range(retry_attempts + 1):
                try:
                    with error_context(
                        operation=operation_name,
                        additional_context={
                            "attempt": attempt + 1,
                            "max_attempts": retry_attempts + 1,
                        },
                    ):
                        return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    if log_errors:
                        if attempt < retry_attempts:
                            logger.warning(
                                f"Attempt {attempt + 1} failed for {operation_name}: {e}"
                            )
                            logger.info(f"Retrying in {retry_delay} seconds...")
                        else:
                            logger.error(f"All attempts failed for {operation_name}")

                    # Wait before retry (except on last attempt)
                    if attempt < retry_attempts:
                        time.sleep(retry_delay)
                    else:
                        raise last_exception

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception

        return wrapper

    return decorator


def validate_inputs(*validators):
    """
    Decorator for input validation.

    Args:
        *validators: Validation functions that raise exceptions on invalid input
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Apply all validators
            for validator in validators:
                try:
                    validator(*args, **kwargs)
                except Exception as e:
                    raise DataValidationError(
                        f"Input validation failed for {func.__name__}: {str(e)}",
                        suggestions=[
                            "Check input data types and shapes",
                            "Verify input ranges and constraints",
                            "Review function documentation for requirements",
                        ],
                        original_exception=e,
                    ) from e

            return func(*args, **kwargs)

        return wrapper

    return decorator


# ============================================================
# Input Validators
# ============================================================


def validate_numpy_array(
    name: str, array: Any, shape: Optional[Tuple] = None, dtype: Optional[type] = None
):
    """Validate numpy array input."""
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{name} must be a numpy array, got {type(array)}")

    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values (inf/nan)")

    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}, expected {shape}")

    if dtype is not None and array.dtype != dtype:
        raise ValueError(f"{name} has dtype {array.dtype}, expected {dtype}")


def validate_tensor(
    name: str, tensor: Any, shape: Optional[Tuple] = None, device: Optional[str] = None
):
    """Validate PyTorch tensor input."""
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor, got {type(tensor)}")

    if tensor.numel() == 0:
        raise ValueError(f"{name} cannot be empty")

    if not torch.all(torch.isfinite(tensor)):
        raise ValueError(f"{name} contains non-finite values (inf/nan)")

    if shape is not None and tensor.shape != shape:
        raise ValueError(f"{name} has shape {tensor.shape}, expected {shape}")

    if device is not None and str(tensor.device) != device:
        raise ValueError(f"{name} is on device {tensor.device}, expected {device}")


def validate_positive_number(name: str, value: Any):
    """Validate that a value is a positive number."""
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {type(value)}")

    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_non_negative_number(name: str, value: Any):
    """Validate that a value is a non-negative number."""
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {type(value)}")

    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


# ============================================================
# Resource Management and Cleanup
# ============================================================


class ResourceManager:
    """Manages resources and ensures proper cleanup on errors."""

    def __init__(self):
        """Initialize resource manager."""
        self.resources: List[Callable[[], None]] = []
        self.cleanup_functions: List[Callable[[], None]] = []

    def register_cleanup(self, cleanup_func: Callable[[], None]) -> None:
        """Register a cleanup function."""
        self.cleanup_functions.append(cleanup_func)

    def cleanup_all(self) -> None:
        """Execute all cleanup functions."""
        for cleanup_func in reversed(self.cleanup_functions):  # LIFO order
            try:
                cleanup_func()
            except Exception as e:
                # Log cleanup errors but don't raise them
                logging.warning(f"Error during cleanup: {e}")

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and perform cleanup."""
        self.cleanup_all()


@contextmanager
def managed_resources():
    """Context manager for automatic resource cleanup."""
    manager = ResourceManager()
    try:
        yield manager
    finally:
        manager.cleanup_all()


# ============================================================
# Error Recovery Strategies
# ============================================================


class ErrorRecoveryStrategy:
    """Base class for error recovery strategies."""

    def can_recover(self, error: Exception) -> bool:
        """Check if recovery is possible for this error."""
        return False

    def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Attempt to recover from the error."""
        raise NotImplementedError


class CheckpointRecoveryStrategy(ErrorRecoveryStrategy):
    """Recovery strategy using model checkpoints."""

    def __init__(self, checkpoint_dir: str):
        """Initialize with checkpoint directory."""
        self.checkpoint_dir = Path(checkpoint_dir)

    def can_recover(self, error: Exception) -> bool:
        """Check if we can recover using checkpoints."""
        return (
            isinstance(error, (TrainingError, ComputationError))
            and self._has_valid_checkpoint()
        )

    def _has_valid_checkpoint(self) -> bool:
        """Check if valid checkpoints exist."""
        if not self.checkpoint_dir.exists():
            return False

        checkpoint_files = list(self.checkpoint_dir.glob("*.pth"))
        return len(checkpoint_files) > 0

    def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Recover by loading the latest checkpoint."""
        latest_checkpoint = self._get_latest_checkpoint()

        if latest_checkpoint is None:
            raise TrainingError(
                "No valid checkpoint found for recovery",
                suggestions=[
                    "Enable checkpoint saving during training",
                    "Check checkpoint directory permissions",
                    "Restart training from the beginning",
                ],
            )

        return {
            "checkpoint_path": latest_checkpoint,
            "recovery_action": "load_checkpoint",
            "message": f"Recovered using checkpoint: {latest_checkpoint}",
        }

    def _get_latest_checkpoint(self) -> Optional[Path]:
        """Get the path to the latest checkpoint."""
        checkpoint_files = list(self.checkpoint_dir.glob("*.pth"))

        if not checkpoint_files:
            return None

        # Sort by modification time and return the latest
        return max(checkpoint_files, key=lambda p: p.stat().st_mtime)


# ============================================================
# Demonstration and Testing
# ============================================================


def demonstrate_error_handling():
    """Demonstrate the error handling system capabilities."""

    print("=== PINN Error Handling System Demo ===\n")

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # 1. Demonstrate basic error creation and formatting
    print("1. Creating and formatting errors...")

    try:
        raise ConfigurationError(
            "Invalid learning rate specified",
            suggestions=[
                "Use a positive learning rate between 1e-5 and 1e-1",
                "Consider using learning rate scheduling",
                "Check the configuration file syntax",
            ],
        )
    except ConfigurationError as e:
        print("Configuration Error Example:")
        print(e.get_formatted_message())
        print()

    # 2. Demonstrate error context manager
    print("2. Using error context manager...")

    try:
        with error_context("data validation", training_step=100):
            # Simulate an error
            invalid_array = np.array([1, 2, np.inf, 4])
            validate_numpy_array("test_array", invalid_array)

    except PINNError as e:
        print("Context Manager Error:")
        print(e.get_formatted_message())
        print()

    # 3. Demonstrate decorator usage
    print("3. Using error handling decorators...")

    @handle_errors(retry_attempts=2, retry_delay=0.1)
    @validate_inputs(
        lambda *args, **kwargs: validate_positive_number(
            "learning_rate", kwargs.get("lr", 0.01)
        )
    )
    def mock_training_step(lr: float = 0.01):
        """Mock training step that might fail."""
        if lr < 0:
            raise ValueError("Learning rate must be positive")
        return f"Training step completed with lr={lr}"

    # This should work
    try:
        result = mock_training_step(lr=0.001)
        print(f"✓ {result}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

    # This should fail with validation
    try:
        result = mock_training_step(lr=-0.001)
        print(f"✗ This should have failed: {result}")
    except DataValidationError as e:
        print("✓ Input validation worked:")
        print(f"  {e.message}")

    # 4. Demonstrate resource management
    print("\n4. Using resource management...")

    def simulate_resource_operation():
        with managed_resources() as manager:
            # Register cleanup functions
            manager.register_cleanup(lambda: print("  Cleaned up temporary files"))
            manager.register_cleanup(lambda: print("  Released GPU memory"))
            manager.register_cleanup(lambda: print("  Closed network connections"))

            # Simulate an error
            raise RuntimeError("Simulated operation failure")

    try:
        simulate_resource_operation()
    except Exception:
        print("Resource cleanup on error:")
        # Cleanup functions were automatically called

    # 5. Demonstrate error recovery
    print("\n5. Error recovery strategies...")

    # Create a mock checkpoint directory
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        checkpoint_dir.mkdir()

        # Create a fake checkpoint
        fake_checkpoint = checkpoint_dir / "model_epoch_100.pth"
        fake_checkpoint.write_text("fake checkpoint data")

        # Test recovery strategy
        recovery_strategy = CheckpointRecoveryStrategy(checkpoint_dir)

        mock_error = TrainingError("Training diverged")

        if recovery_strategy.can_recover(mock_error):
            recovery_info = recovery_strategy.recover(mock_error, {})
            print("✓ Recovery successful:")
            print(f"  Action: {recovery_info['recovery_action']}")
            print(f"  Message: {recovery_info['message']}")
        else:
            print("✗ Recovery not possible")

    print("\n=== Error Handling System Demo Complete ===")


# Example usage with actual PINN components
def example_pinn_with_error_handling():
    """Example of how to integrate error handling with PINN components."""

    @handle_errors(operation="PINN training", retry_attempts=1)
    @validate_inputs(
        lambda model, *args, **kwargs: validate_tensor(
            "model_output", model(torch.randn(1, 2)), device="cpu"
        )
    )
    def train_pinn_step(model, data, optimizer, step: int):
        """Example PINN training step with error handling."""

        with error_context(
            "forward pass",
            training_step=step,
            additional_context={
                "model_params": sum(p.numel() for p in model.parameters())
            },
        ):
            # Forward pass
            output = model(data)

            # Check for numerical issues
            if not torch.all(torch.isfinite(output)):
                raise ComputationError(
                    "Non-finite values in model output",
                    suggestions=[
                        "Check input data for inf/nan values",
                        "Reduce learning rate",
                        "Use gradient clipping",
                        "Check network initialization",
                    ],
                )

            return output

    # This demonstrates how the system would be used in practice
    print("\nExample PINN training with error handling:")
    print("✓ Error handling decorators and context managers integrated")
    print("✓ Automatic validation and recovery strategies available")
    print("✓ Comprehensive error reporting and suggestions provided")


if __name__ == "__main__":
    demonstrate_error_handling()
    example_pinn_with_error_handling()

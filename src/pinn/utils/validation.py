"""Input validation utilities for PINN modules."""

from __future__ import annotations

import inspect
from functools import lru_cache, wraps
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = [
    "ValidationError",
    "check_array",
    "check_range",
    "check_config",
    "check_device",
    "validate_arrays",
    "validate_config",
    "validate_device",
]


class ValidationError(ValueError):
    """Raised when validation fails."""


# ---------------------------------------------------------------------------
# Primitive check functions
# ---------------------------------------------------------------------------


def _match_shape(actual: Sequence[int], expected: Sequence[int]) -> bool:
    if len(actual) != len(expected):
        return False
    for a, e in zip(actual, expected):
        if e != -1 and a != e:
            return False
    return True


def check_array(
    name: str,
    array: Any,
    *,
    shape: Optional[Sequence[int]] = None,
    dtype: Optional[Any] = None,
    strict: bool = True,
) -> None:
    """Validate numpy arrays or torch tensors."""
    if not isinstance(array, (np.ndarray, torch.Tensor)):
        raise ValidationError(
            f"{name} must be a numpy array or torch Tensor; got {type(array)}"
        )

    actual_shape = tuple(array.shape)
    if shape is not None and not _match_shape(actual_shape, shape):
        raise ValidationError(f"{name} has shape {actual_shape}, expected {shape}")

    if dtype is not None:
        actual_dtype = array.dtype
        if actual_dtype != dtype and strict:
            raise ValidationError(f"{name} has dtype {actual_dtype}, expected {dtype}")


def check_range(
    name: str,
    value: float,
    *,
    min: Optional[float] = None,
    max: Optional[float] = None,
) -> None:
    """Validate numeric ranges.

    Examples
    --------
    >>> from pinn.utils.validation import check_range
    >>> check_range("lr", 0.1, min=0.0, max=1.0)
    >>> check_range("lr", -1.0, min=0.0)
    Traceback (most recent call last):
    ...
    ValidationError: lr must be >= 0.0, got -1.0
    """
    if min is not None and value < min:
        raise ValidationError(f"{name} must be >= {min}, got {value}")
    if max is not None and value > max:
        raise ValidationError(f"{name} must be <= {max}, got {value}")


@lru_cache(None)
def _config_fields(config_cls: type) -> Tuple[str, ...]:
    return tuple(getattr(config_cls, "__annotations__", {}).keys())


def check_config(
    config: Any,
    config_cls: type,
    *,
    strict: bool = True,
) -> None:
    """Validate configuration objects against a dataclass type."""
    if not isinstance(config, config_cls):
        raise ValidationError(f"Expected {config_cls.__name__}, got {type(config)}")

    if strict:
        for field in _config_fields(config_cls):
            if not hasattr(config, field):
                raise ValidationError(f"Config missing required field '{field}'")


def check_device(name: str, tensor: Any, device: str) -> None:
    """Ensure tensor resides on the specified device."""
    if torch.is_tensor(tensor) and tensor.device.type != device:
        raise ValidationError(
            f"{name} is on device {tensor.device.type}, expected {device}"
        )


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------


def validate_arrays(
    params: Sequence[str],
    *,
    shapes: Optional[Mapping[str, Sequence[int]]] = None,
    dtypes: Optional[Mapping[str, Any]] = None,
    strict: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator validating array arguments by name."""

    shapes = shapes or {}
    dtypes = dtypes or {}

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            for name in params:
                if name in bound.arguments:
                    arr = bound.arguments[name]
                    check_array(
                        name,
                        arr,
                        shape=shapes.get(name),
                        dtype=dtypes.get(name),
                        strict=strict,
                    )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def validate_config(
    config_cls: type,
    *,
    param: str = "config",
    strict: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator validating configuration objects."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            if param not in bound.arguments:
                raise ValidationError(f"Missing config parameter '{param}'")
            check_config(bound.arguments[param], config_cls, strict=strict)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def validate_device(
    params: Sequence[str],
    *,
    device_arg: str = "device",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator ensuring tensors are on a compatible device."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            device = bound.arguments.get(device_arg)
            if device is None:
                for name in params:
                    t = bound.arguments.get(name)
                    if torch.is_tensor(t):
                        device = t.device.type
                        break
            if device is not None:
                for name in params:
                    t = bound.arguments.get(name)
                    if t is not None:
                        check_device(name, t, device)
            return func(*args, **kwargs)

        return wrapper

    return decorator

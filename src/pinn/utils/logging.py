"""Comprehensive logging utilities for the PINN repository.

This module provides a flexible logging framework with support for:
- Structured (JSON) and plain-text formatting
- Per-module log level configuration
- Console, file, and remote output destinations
- Log rotation for file handlers
- Performance timing decorators
- Memory and GPU utilisation tracking

Example:
    from pinn.utils.logging import get_logger, log_performance

    logger = get_logger(__name__)

    @log_performance
    def train_step(model, data):
        logger.info("Starting training step", extra={"step": 1})
        # ... training code ...
        logger.info("Training step completed", extra={"loss": 0.1})
"""

from __future__ import annotations

import json
import logging
import logging.config
import time
from functools import wraps
from logging.handlers import HTTPHandler, RotatingFileHandler
from typing import Any, Dict, Optional

try:  # Optional dependency
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional
    psutil = None

try:  # Optional GPU support
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional
    torch = None

__all__ = [
    "configure_logging",
    "get_logger",
    "log_performance",
    "log_memory_usage",
    "log_gpu_usage",
]

_LOGGING_CONFIGURED = False
_STANDARD_ATTRS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
}


class JsonFormatter(logging.Formatter):
    """Minimal JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - formatting
        data = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                data[key] = value
        if record.exc_info:
            data["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(data)


def configure_logging(
    level: str = "INFO",
    *,
    console: bool = True,
    filename: Optional[str] = None,
    json_format: bool = False,
    max_bytes: int = 10_000_000,
    backup_count: int = 5,
    module_levels: Optional[Dict[str, str]] = None,
    remote_url: Optional[str] = None,
) -> None:
    """Configure the global logging system.

    Args:
        level: Default logging level.
        console: Enable console output.
        filename: Optional log file path.
        json_format: Emit logs as JSON when True.
        max_bytes: Maximum size of a log file before rotation.
        backup_count: Number of rotated log files to keep.
        module_levels: Optional mapping of module names to log levels.
        remote_url: Optional HTTP endpoint to receive logs.
    """
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    formatter = "json" if json_format else "plain"
    formatters: Dict[str, Any] = {
        "plain": {
            "format": "%(asctime)s %(levelname)s %(name)s: %(message)s"
        },
        "json": {"()": JsonFormatter},
    }

    handlers: Dict[str, Dict[str, Any]] = {}
    handler_names = []

    if console:
        handlers["console"] = {
            "class": "logging.StreamHandler",
            "formatter": formatter,
            "level": level,
        }
        handler_names.append("console")

    if filename:
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": filename,
            "maxBytes": max_bytes,
            "backupCount": backup_count,
            "formatter": formatter,
            "level": level,
        }
        handler_names.append("file")

    if remote_url:
        try:
            host, path = remote_url.split("/", 1)
        except ValueError:
            host, path = remote_url, ""
        handlers["remote"] = {
            "class": "logging.handlers.HTTPHandler",
            "host": host,
            "url": "/" + path,
            "method": "POST",
            "formatter": formatter,
            "level": level,
        }
        handler_names.append("remote")

    config: Dict[str, Any] = {
        "version": 1,
        "formatters": formatters,
        "handlers": handlers,
        "root": {
            "level": level,
            "handlers": handler_names,
        },
        "loggers": {},
        "disable_existing_loggers": False,
    }

    if module_levels:
        for mod, lvl in module_levels.items():
            config["loggers"][mod] = {"level": lvl}

    logging.config.dictConfig(config)
    _LOGGING_CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance, configuring logging on first use."""
    if not _LOGGING_CONFIGURED:
        configure_logging()
    return logging.getLogger(name)


def log_memory_usage(logger: logging.Logger) -> None:
    """Log current process memory usage in MB."""
    if psutil is None:  # pragma: no cover - optional dependency
        return
    process = psutil.Process()
    mem_mb = process.memory_info().rss / (1024 ** 2)
    logger.debug("Memory usage", extra={"memory_mb": round(mem_mb, 2)})


def log_gpu_usage(logger: logging.Logger) -> None:
    """Log current GPU utilisation if available."""
    if torch is None or not torch.cuda.is_available():  # pragma: no cover - optional
        return
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    extra = {
        "gpu_mem_allocated_mb": round(allocated, 2),
        "gpu_mem_reserved_mb": round(reserved, 2),
    }
    try:  # pragma: no cover - device properties
        prop = torch.cuda.get_device_properties(0)
        extra["gpu_name"] = prop.name
        extra["gpu_total_mem_mb"] = round(prop.total_memory / (1024 ** 2), 2)
    except Exception:
        pass
    logger.debug("GPU usage", extra=extra)


def log_performance(func):
    """Decorator to measure and log function execution time and resources."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        except Exception:
            logger.exception("Error in %s", func.__qualname__)
            raise
        finally:
            duration = time.perf_counter() - start
            logger.info(
                "Performance",
                extra={"function": func.__qualname__, "duration_sec": round(duration, 4)},
            )
            log_memory_usage(logger)
            log_gpu_usage(logger)

    return wrapper

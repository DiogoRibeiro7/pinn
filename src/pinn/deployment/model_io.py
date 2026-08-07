"""Utilities for serializing and loading trained PINN models."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Type

import torch


def _version_path(base: Path, version: Optional[str]) -> Path:
    """Return a path with an optional version subdirectory."""
    if version:
        base = base / version
    base.parent.mkdir(parents=True, exist_ok=True)
    return base


def save_model(
    model: torch.nn.Module, path: str, version: Optional[str] = None
) -> Path:
    """Save ``model`` to ``path`` with optional ``version`` tagging.

    Parameters
    ----------
    model:
        The ``torch.nn.Module`` to persist.
    path:
        Destination directory or file stem.
    version:
        Optional version identifier used to namespace the save location.

    Returns
    -------
    Path
        Location of the saved model state.
    """
    base = _version_path(Path(path), version)
    file_path = base.with_suffix(".pt")
    torch.save(model.state_dict(), file_path)
    return file_path


def load_model(
    model_cls: Type[torch.nn.Module], path: str, version: Optional[str] = None
) -> torch.nn.Module:
    """Load a model instance of ``model_cls`` from ``path``.

    Parameters
    ----------
    model_cls:
        Class of the model to instantiate before loading weights.
    path:
        File stem or directory containing the saved model.
    version:
        Optional version identifier selecting a subdirectory.
    """
    base = _version_path(Path(path), version)
    file_path = base.with_suffix(".pt")
    model = model_cls()
    state = torch.load(file_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model

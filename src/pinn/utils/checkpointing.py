"""Checkpointing utilities for PINN training."""

from __future__ import annotations

import gzip
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn, optim


@dataclass
class CheckpointMeta:
    """Metadata stored alongside checkpoint state."""

    metrics: Dict[str, float]
    config: Optional[Dict[str, Any]]
    timestamp: float


class CheckpointManager:
    """Manage saving and loading of model/optimizer checkpoints.

    Parameters
    ----------
    checkpoint_dir:
        Directory where checkpoints are stored.
    save_frequency:
        Save every ``save_frequency`` training steps.
    keep_best_n:
        Keep only the ``keep_best_n`` checkpoints with lowest metric.
    metric_key:
        Metric used to determine best checkpoints.
    patience:
        Early stopping patience (in number of saves). ``None`` disables.
    compress:
        Whether to gzip-compress checkpoint files.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        save_frequency: int = 1000,
        keep_best_n: int = 5,
        metric_key: str = "loss",
        patience: Optional[int] = None,
        compress: bool = False,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_frequency = save_frequency
        self.keep_best_n = keep_best_n
        self.metric_key = metric_key
        self.patience = patience
        self.compress = compress

        self.best_metric = float("inf")
        self.best_path: Optional[Path] = None
        self._no_improve = 0

    # ------------------------------------------------------------------
    # Paths and helpers
    # ------------------------------------------------------------------
    def _checkpoint_path(self, step: int) -> Path:
        ext = ".pt.gz" if self.compress else ".pt"
        return self.checkpoint_dir / f"checkpoint_{step}{ext}"

    def _metadata_path(self, step: int) -> Path:
        return self.checkpoint_dir / f"checkpoint_{step}.json"

    def _open(self, path: Path, mode: str):
        if path.suffix == ".gz" or (self.compress and mode.startswith("w")):
            return gzip.open(path, mode)
        return open(path, mode)

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    def save(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        step: int,
        metrics: Dict[str, float],
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save model/optimizer state and metadata."""
        state = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
        }
        cpath = self._checkpoint_path(step)
        with self._open(cpath, "wb") as f:
            # Default pickle protocol: the weights_only unpickler used on load
            # cannot read protocol-5 frames, and tensor storage is written to
            # the archive separately, so protocol 5 bought nothing here.
            torch.save(state, f)

        meta = CheckpointMeta(metrics=metrics, config=config, timestamp=time.time())
        with open(self._metadata_path(step), "w", encoding="utf-8") as f:
            json.dump(meta.__dict__, f)

        metric_val = metrics.get(self.metric_key)
        if metric_val is not None:
            if metric_val < self.best_metric:
                self.best_metric = metric_val
                self.best_path = cpath
                self._no_improve = 0
            else:
                self._no_improve += 1
        self._prune_best()

    def maybe_save(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        step: int,
        metrics: Dict[str, float],
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if step % self.save_frequency == 0:
            self.save(model, optimizer, step, metrics, config)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_latest(
        self,
        model: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
    ) -> Tuple[Optional[nn.Module], Optional[optim.Optimizer], int, CheckpointMeta]:
        """Load the most recent checkpoint.

        Returns the provided ``model`` and ``optimizer`` populated with loaded
        state, the training step, and metadata.
        """
        files = sorted(
            self.checkpoint_dir.glob("checkpoint_*.pt*"), key=self._step_from_name
        )
        if not files:
            raise FileNotFoundError("No checkpoints found")
        latest = files[-1]
        with self._open(latest, "rb") as f:
            state = torch.load(f, map_location="cpu", weights_only=True)
        step = int(state.get("step", 0))
        if model is not None:
            model.load_state_dict(state["model_state"])
        if optimizer is not None:
            optimizer.load_state_dict(state["optimizer_state"])
        with open(self._metadata_path(step), "r", encoding="utf-8") as f:
            meta_dict = json.load(f)
        return model, optimizer, step, CheckpointMeta(**meta_dict)

    def restore_best(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
    ) -> Tuple[nn.Module, optim.Optimizer, int]:
        """Restore model/optimizer to the best checkpoint seen so far."""
        if self.best_path is None:
            raise FileNotFoundError("No best checkpoint available")
        with self._open(self.best_path, "rb") as f:
            state = torch.load(f, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        return model, optimizer, int(state.get("step", 0))

    # ------------------------------------------------------------------
    # Book-keeping
    # ------------------------------------------------------------------
    def _step_from_name(self, path: Path) -> int:
        stem = path.stem
        if stem.endswith(".pt"):
            stem = stem[:-3]
        return int(stem.split("_")[1])

    def _prune_best(self) -> None:
        metas = []
        for meta_file in self.checkpoint_dir.glob("checkpoint_*.json"):
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            step = int(meta_file.stem.split("_")[1])
            val = data.get("metrics", {}).get(self.metric_key, float("inf"))
            metas.append((val, step))
        metas.sort()
        keep_steps = {step for _, step in metas[: self.keep_best_n]}
        for _, step in metas:
            if step not in keep_steps:
                self._remove(step)

    def _remove(self, step: int) -> None:
        for p in [self._checkpoint_path(step), self._metadata_path(step)]:
            if p.exists():
                p.unlink()

    # ------------------------------------------------------------------
    # Early stopping
    # ------------------------------------------------------------------
    def should_stop(self) -> bool:
        return self.patience is not None and self._no_improve >= self.patience


__all__ = ["CheckpointManager"]

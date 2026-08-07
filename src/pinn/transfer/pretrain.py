"""Pre-training strategies for transfer learning across PDE problems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    MutableMapping,
    Optional,
    Sequence,
)

import torch
from torch.utils.data import DataLoader, TensorDataset

from ..utils.logging import get_logger

BatchType = MutableMapping[str, torch.Tensor]
LossFn = Callable[[torch.nn.Module, BatchType], torch.Tensor]


@dataclass
class PretrainingTask:
    """Container describing a single pre-training task.

    Attributes
    ----------
    name:
        Human readable identifier for the task.
    data_loader:
        Iterable yielding mini-batches that contain at least an ``"input"`` tensor.
    loss_fn:
        Callable receiving the model and a batch dictionary returning a scalar loss.
    epochs:
        Number of epochs to train on this task.
    weight:
        Optional weight used when constructing progressive curricula; higher values are
        interpreted as more complex tasks.
    description:
        Free-form text describing the task; used in logs and experiment summaries.
    adaptation_fn:
        Optional callback invoked before training starts to adapt the model to the task
        (for example, by adjusting output heads or boundary conditions).
    """

    name: str
    data_loader: Iterable[BatchType] | DataLoader
    loss_fn: LossFn
    epochs: int = 100
    weight: float = 1.0
    description: Optional[str] = None
    adaptation_fn: Optional[Callable[[torch.nn.Module], None]] = None

    def iter_batches(self) -> Iterator[BatchType]:
        """Yield batches as dictionaries with the required keys."""

        for batch in self.data_loader:
            yield ensure_batch_dict(batch)


@dataclass
class PretrainingConfig:
    """Configuration for :class:`PretrainingManager`."""

    lr: float = 1e-3
    device: str = "cpu"
    gradient_clip: Optional[float] = None
    progressive: bool = True
    log_frequency: int = 10
    reset_optimizer_each_task: bool = False
    max_steps_per_task: Optional[int] = None


class PretrainingManager:
    """Orchestrates pre-training across multiple tasks and fidelities.

    The manager supports progressive curricula where the model is first trained on
    simplified problems (for instance linearised PDEs or coarse discretisations) before
    moving on to high-fidelity simulations.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: PretrainingConfig | None = None,
        optimizer_factory: (
            Callable[[Iterable[torch.nn.Parameter], float], torch.optim.Optimizer]
            | None
        ) = None,
    ) -> None:
        self.model = model
        self.config = config or PretrainingConfig()
        self.optimizer_factory = (
            optimizer_factory
            if optimizer_factory is not None
            else lambda params, lr: torch.optim.Adam(params, lr=lr)
        )
        self.logger = get_logger(__name__)
        self._tasks: List[PretrainingTask] = []
        self._optimizer: Optional[torch.optim.Optimizer] = None

    def register_task(self, task: PretrainingTask) -> None:
        """Register a new pre-training task."""

        self._tasks.append(task)

    def run(
        self, tasks: Optional[Sequence[PretrainingTask]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Execute the pre-training schedule and return metrics per task."""

        task_sequence = list(tasks or self._tasks)
        if self.config.progressive:
            task_sequence.sort(key=lambda t: t.weight)
        results: Dict[str, Dict[str, float]] = {}
        device = torch.device(self.config.device)
        self.model.to(device)

        for index, task in enumerate(task_sequence):
            if self.config.reset_optimizer_each_task or self._optimizer is None:
                self._optimizer = self.optimizer_factory(
                    self.model.parameters(), self.config.lr
                )
            self.logger.info(
                "Starting pre-training task %s (%d/%d)",
                task.name,
                index + 1,
                len(task_sequence),
            )
            if task.description:
                self.logger.debug("Task description: %s", task.description)
            if task.adaptation_fn is not None:
                task.adaptation_fn(self.model)

            metrics = self._run_single_task(task, device)
            results[task.name] = metrics

        return results

    def _run_single_task(
        self, task: PretrainingTask, device: torch.device
    ) -> Dict[str, float]:
        assert (
            self._optimizer is not None
        ), "Optimizer must be created before running a task"
        running_loss = 0.0
        total_steps = 0
        best_loss = float("inf")

        for epoch in range(task.epochs):
            for step, batch in enumerate(task.iter_batches()):
                batch = move_batch_to_device(batch, device)
                self._optimizer.zero_grad()
                loss = task.loss_fn(self.model, batch)
                loss.backward()
                if self.config.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip
                    )
                self._optimizer.step()

                running_loss += loss.item()
                total_steps += 1
                best_loss = min(best_loss, loss.item())

                if (
                    self.config.max_steps_per_task is not None
                    and total_steps >= self.config.max_steps_per_task
                ):
                    break
            if (
                epoch % self.config.log_frequency == 0
                or epoch == task.epochs - 1
                or self.config.max_steps_per_task is not None
            ):
                self.logger.debug(
                    "Task %s epoch %d/%d - loss %.4e",
                    task.name,
                    epoch + 1,
                    task.epochs,
                    running_loss / max(total_steps, 1),
                )
            if (
                self.config.max_steps_per_task is not None
                and total_steps >= self.config.max_steps_per_task
            ):
                break

        avg_loss = running_loss / max(total_steps, 1)
        return {
            "avg_loss": avg_loss,
            "best_loss": best_loss,
            "steps": float(total_steps),
        }


def ensure_batch_dict(
    batch: BatchType | tuple[torch.Tensor, ...] | torch.Tensor
) -> BatchType:
    """Convert batches returned by :class:`~torch.utils.data.DataLoader` to dictionaries."""

    if isinstance(batch, dict):
        return batch
    if isinstance(batch, torch.Tensor):
        return {"input": batch}
    if isinstance(batch, (list, tuple)):
        if len(batch) == 1:
            return {"input": batch[0]}
        if len(batch) >= 2:
            return {"input": batch[0], "target": batch[1]}
    raise TypeError(f"Unsupported batch type: {type(batch)!r}")


def move_batch_to_device(batch: BatchType, device: torch.device) -> BatchType:
    """Move all tensor values in ``batch`` to ``device``."""

    for key, value in batch.items():
        if torch.is_tensor(value):
            batch[key] = value.to(device)
    return batch


def build_simplified_pretraining_task(
    name: str,
    solution_fn: Callable[[torch.Tensor], torch.Tensor],
    num_points: int = 256,
    noise_std: float = 0.0,
    batch_size: Optional[int] = None,
    description: Optional[str] = None,
) -> PretrainingTask:
    """Create a :class:`PretrainingTask` from an analytic solution."""

    x = torch.linspace(0.0, 1.0, num_points).unsqueeze(-1)
    y = solution_fn(x)
    if noise_std > 0:
        y = y + noise_std * torch.randn_like(y)
    dataset = TensorDataset(x, y)
    loader = DataLoader(
        dataset, batch_size=batch_size or min(64, num_points), shuffle=True
    )

    def loss_fn(model: torch.nn.Module, batch: BatchType) -> torch.Tensor:
        preds = model(batch["input"])
        return torch.nn.functional.mse_loss(preds, batch["target"])

    return PretrainingTask(
        name=name, data_loader=loader, loss_fn=loss_fn, description=description
    )


def progressive_complexity_schedule(
    tasks: Sequence[PretrainingTask],
) -> List[PretrainingTask]:
    """Return tasks sorted by increasing ``weight`` to form a curriculum."""

    return sorted(tasks, key=lambda task: task.weight)


def create_cross_pde_pretraining_tasks(
    base_model: torch.nn.Module,
    equations: Sequence[str],
    dataset_factory: Callable[[str, torch.nn.Module], PretrainingTask],
) -> List[PretrainingTask]:
    """Generate tasks for cross-PDE pre-training."""

    tasks: List[PretrainingTask] = []
    for index, equation in enumerate(equations):
        task = dataset_factory(equation, base_model)
        task.weight = index + 1
        tasks.append(task)
    return tasks


def create_linear_reference_task(
    slope: float = 1.0,
    intercept: float = 0.0,
    **kwargs: object,
) -> PretrainingTask:
    """Convenience helper returning a linear reference task."""

    def solution_fn(x: torch.Tensor) -> torch.Tensor:
        return slope * x + intercept

    return build_simplified_pretraining_task(
        name="linear_reference",
        solution_fn=solution_fn,
        description="Linearised PDE",
        **kwargs,
    )


__all__ = [
    "PretrainingTask",
    "PretrainingConfig",
    "PretrainingManager",
    "build_simplified_pretraining_task",
    "progressive_complexity_schedule",
    "create_cross_pde_pretraining_tasks",
    "create_linear_reference_task",
]

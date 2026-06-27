"""Knowledge distillation utilities for PINN transfer learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, MutableMapping, Optional

import torch

from ..utils.logging import get_logger
from .pretrain import ensure_batch_dict, move_batch_to_device

BatchType = MutableMapping[str, torch.Tensor]


@dataclass
class DistillationConfig:
    """Configuration for :class:`KnowledgeDistillationTrainer`."""

    temperature: float = 3.0
    alpha: float = 0.5
    epochs: int = 200
    lr: float = 1e-3
    device: str = "cpu"
    loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None


class KnowledgeDistillationTrainer:
    """Train a student PINN using soft targets from a teacher network."""

    def __init__(
        self,
        teacher_model: torch.nn.Module,
        student_model: torch.nn.Module,
        config: Optional[DistillationConfig] = None,
    ) -> None:
        self.teacher_model = teacher_model.eval()
        self.student_model = student_model
        self.config = config or DistillationConfig()
        self.logger = get_logger(__name__)
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def train(self, data_loader: Iterable[BatchType]) -> None:
        optimizer = torch.optim.Adam(self.student_model.parameters(), lr=self.config.lr)
        device = torch.device(self.config.device)
        self.student_model.to(device)
        self.teacher_model.to(device)

        for epoch in range(self.config.epochs):
            total_loss = 0.0
            steps = 0
            for batch in _iterate_batches(data_loader):
                batch = move_batch_to_device(batch, device)
                optimizer.zero_grad()
                with torch.no_grad():
                    teacher_logits = self.teacher_model(batch["input"])
                student_logits = self.student_model(batch["input"])
                distill_loss = self._distillation_loss(student_logits, teacher_logits)
                task_loss = torch.tensor(0.0, device=device)
                if "target" in batch and batch["target"] is not None:
                    loss_fn = self.config.loss_fn or torch.nn.functional.mse_loss
                    task_loss = loss_fn(student_logits, batch["target"])
                loss = self.config.alpha * distill_loss + (1 - self.config.alpha) * task_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                steps += 1
            avg_loss = total_loss / max(steps, 1)
            if epoch % max(self.config.epochs // 10, 1) == 0 or epoch == self.config.epochs - 1:
                self.logger.debug(
                    "Distillation epoch %d/%d loss %.4e", epoch + 1, self.config.epochs, avg_loss
                )

    def _distillation_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        temperature = self.config.temperature
        student_soft = torch.nn.functional.log_softmax(student_logits / temperature, dim=-1)
        teacher_soft = torch.nn.functional.softmax(teacher_logits / temperature, dim=-1)
        loss = torch.nn.functional.kl_div(student_soft, teacher_soft, reduction="batchmean")
        return loss * (temperature**2)


def _iterate_batches(loader: Iterable[BatchType]) -> Iterable[BatchType]:
    for batch in loader:
        yield ensure_batch_dict(batch)


__all__ = ["KnowledgeDistillationTrainer", "DistillationConfig"]

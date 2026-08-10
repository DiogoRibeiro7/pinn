"""Meta-learning utilities."""

from __future__ import annotations

from typing import Iterable

import torch
import copy


def meta_learning_step(
    model: torch.nn.Module,
    task_batch: Iterable[tuple[torch.Tensor, torch.Tensor]],
    inner_steps: int = 1,
    lr: float = 1e-2,
) -> None:
    """Perform a simple MAML-style meta-learning update.

    Parameters
    ----------
    model:
        The model to meta-train.
    task_batch:
        Iterable of ``(x, y)`` pairs representing different tasks.
    inner_steps:
        Number of gradient steps per task.
    lr:
        Learning rate for the inner adaptation.
    """

    meta_grads = [torch.zeros_like(p) for p in model.parameters()]
    for x, y in task_batch:
        temp_model = copy.deepcopy(model)
        opt = torch.optim.SGD(temp_model.parameters(), lr=lr)
        for _ in range(inner_steps):
            loss = torch.nn.functional.mse_loss(temp_model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        for mg, p_t, p in zip(meta_grads, temp_model.parameters(), model.parameters()):
            mg += p_t.data - p.data
    for p, mg in zip(model.parameters(), meta_grads):
        p.data += mg / len(task_batch)

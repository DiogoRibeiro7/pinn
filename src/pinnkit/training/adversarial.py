"""Adversarial training helpers."""

from __future__ import annotations

from typing import Callable

import torch


def adversarial_training_step(
    model: torch.nn.Module,
    loss_fn: Callable,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 1e-2,
) -> torch.Tensor:
    """Perform a single adversarial training step using FGSM.

    Parameters
    ----------
    model: nn.Module
        The model being trained.
    loss_fn: Callable
        Loss function that accepts ``(pred, y)``.
    x, y: torch.Tensor
        Input and target tensors.
    epsilon: float
        Perturbation size for the FGSM attack.
    """

    x_adv = x.clone().detach().requires_grad_(True)
    pred = model(x_adv)
    loss = loss_fn(pred, y)
    loss.backward()
    x_adv = x_adv + epsilon * x_adv.grad.sign()
    adv_pred = model(x_adv.detach())
    return adv_pred

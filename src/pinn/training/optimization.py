"""Optimisation utilities for PINN training."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List

import torch


def adaptive_loss_weights(losses: Dict[str, torch.Tensor], temperature: float = 1.0) -> Dict[str, float]:
    """Return normalised weights for each loss term based on their magnitudes."""

    loss_vals = torch.tensor([v.detach() for v in losses.values()])
    weights = torch.softmax(loss_vals / temperature, dim=0)
    return {k: float(w) for k, w in zip(losses.keys(), weights)}


def second_order_step(model: torch.nn.Module, loss_fn: Callable, x: torch.Tensor, y: torch.Tensor) -> float:
    """Perform a single step of second-order optimisation using LBFGS."""

    opt = torch.optim.LBFGS(model.parameters(), max_iter=5)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        return loss

    loss = opt.step(closure)
    return float(loss)


def gradient_hyperparam_search(
    param: torch.nn.Parameter,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    lr: float = 1e-2,
) -> None:
    """Take a gradient step on a hyperparameter parameterising ``loss_fn``."""

    loss = loss_fn(param)
    loss.backward()
    with torch.no_grad():
        param -= lr * param.grad
    param.grad.zero_()


def architecture_search(
    architectures: Iterable[Callable[[], torch.nn.Module]],
    eval_fn: Callable[[torch.nn.Module], float],
) -> torch.nn.Module:
    """Evaluate a list of architecture factories and return the best model."""

    scores: List[float] = []
    models: List[torch.nn.Module] = []
    for factory in architectures:
        model = factory()
        models.append(model)
        scores.append(eval_fn(model))
    best = models[int(torch.tensor(scores).argmin())]
    return best

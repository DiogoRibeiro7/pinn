"""Multi-fidelity training strategies."""

from __future__ import annotations

from typing import Callable

import torch


def multi_fidelity_train(
    model: torch.nn.Module,
    high_fidelity: torch.Tensor,
    low_fidelity: torch.Tensor,
    loss_fn: Callable,
) -> torch.Tensor:
    """Combine high and low fidelity data during training.

    This simplistic implementation mixes both datasets with equal weight and
    returns the resulting loss.
    """

    pred_high = model(high_fidelity[:, :-1])
    pred_low = model(low_fidelity[:, :-1])
    loss = loss_fn(pred_high, high_fidelity[:, -1:]) + loss_fn(pred_low, low_fidelity[:, -1:])
    loss.backward()
    return loss

"""Uncertainty quantification helpers for PINNs."""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

import torch
from torch import nn


def mc_dropout_predict(
    model: nn.Module, x: torch.Tensor, passes: int = 20
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estimate prediction mean and std using Monte Carlo dropout."""

    model.train()  # enable dropout
    preds = []
    for _ in range(passes):
        preds.append(model(x))
    stacked = torch.stack(preds)
    return stacked.mean(dim=0), stacked.std(dim=0)


def deep_ensemble_predict(
    models: Sequence[nn.Module], x: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Aggregate predictions from an ensemble of models."""

    preds = torch.stack([m(x) for m in models])
    return preds.mean(dim=0), preds.std(dim=0)


def gp_prior_predict(
    kernel_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    noise: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the GP posterior mean and covariance for a test set."""

    y = y_train.squeeze(-1)
    k_tt = kernel_fn(x_train, x_train) + noise * torch.eye(len(x_train))
    k_ts = kernel_fn(x_train, x_test)
    k_ss = kernel_fn(x_test, x_test)
    alpha = torch.linalg.solve(k_tt, y)
    mean = k_ts.transpose(0, 1) @ alpha
    v = torch.linalg.solve(k_tt, k_ts)
    cov = k_ss - k_ts.transpose(0, 1) @ v
    return mean.unsqueeze(-1), cov


class BayesianLinear(nn.Module):
    """A linear layer with Gaussian weight uncertainty."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.logvar = nn.Parameter(torch.full((out_features, in_features), -5.0))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple
        std = torch.exp(0.5 * self.logvar)
        w = self.mu + std * torch.randn_like(std)
        return x @ w.T + self.bias


class BayesianPINN(nn.Module):
    """Simple Bayesian PINN using variational inference with ``BayesianLinear`` layers."""

    def __init__(self, in_features: int = 1, hidden: int = 20, out_features: int = 1):
        super().__init__()
        self.layer1 = BayesianLinear(in_features, hidden)
        self.layer2 = BayesianLinear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple
        x = torch.tanh(self.layer1(x))
        return self.layer2(x)


__all__ = [
    "mc_dropout_predict",
    "deep_ensemble_predict",
    "gp_prior_predict",
    "BayesianPINN",
    "BayesianLinear",
]

#!/usr/bin/env python3
"""Benchmark script comparing transfer learning strategies for PINNs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import torch
from torch.utils.data import DataLoader, TensorDataset

from pinnkit.models.mlp import MLP
from pinnkit.transfer import (
    FineTuningConfig,
    PretrainingConfig,
    PretrainingManager,
    build_simplified_pretraining_task,
    create_linear_reference_task,
    progressive_complexity_schedule,
    TransferLearningPINN,
)


@dataclass
class BenchmarkConfig:
    device: str = "cpu"
    pretrain_epochs: int = 200
    finetune_epochs: int = 200
    batch_size: int = 64
    lr_pretrain: float = 5e-4
    lr_finetune: float = 1e-3
    noise: float = 0.01


def analytic_heat_solution(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(torch.pi * x)


def analytic_diffusion_solution(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-torch.pi**2 * x) * torch.sin(torch.pi * x)


def analytic_burgers_solution(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(torch.pi * x) + 0.1 * torch.sin(2 * torch.pi * x)


def make_loader(
    func: Callable[[torch.Tensor], torch.Tensor],
    num_points: int,
    batch_size: int,
    noise: float,
) -> DataLoader:
    x = torch.linspace(0.0, 1.0, num_points).unsqueeze(-1)
    y = func(x)
    if noise:
        y = y + noise * torch.randn_like(y)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_baseline(
    model: torch.nn.Module, loader: DataLoader, epochs: int, lr: float, device: str
) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = torch.nn.functional.mse_loss(pred, yb)
            loss.backward()
            optimizer.step()
    return evaluate(model, loader, device)


def evaluate(model: torch.nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            total_loss += torch.nn.functional.mse_loss(pred, yb).item() * xb.shape[0]
            count += xb.shape[0]
    model.train()
    return total_loss / max(count, 1)


def run_benchmark(cfg: BenchmarkConfig) -> None:
    device = cfg.device
    base_model = MLP(in_dim=1, hidden_layers=4, width=32, out_dim=1)
    transfer = TransferLearningPINN(base_model, device=device)

    linear_task = create_linear_reference_task(
        num_points=256, batch_size=cfg.batch_size
    )
    linear_task.epochs = cfg.pretrain_epochs
    heat_task = build_simplified_pretraining_task(
        "heat_equation",
        analytic_heat_solution,
        num_points=256,
        batch_size=cfg.batch_size,
        noise_std=cfg.noise,
    )
    heat_task.epochs = cfg.pretrain_epochs
    diffusion_task = build_simplified_pretraining_task(
        "diffusion_equation",
        analytic_diffusion_solution,
        num_points=256,
        batch_size=cfg.batch_size,
        noise_std=cfg.noise,
    )
    diffusion_task.epochs = cfg.pretrain_epochs

    tasks = progressive_complexity_schedule([linear_task, heat_task, diffusion_task])
    pretrain_cfg = PretrainingConfig(
        lr=cfg.lr_pretrain, device=device, log_frequency=cfg.pretrain_epochs
    )
    manager = PretrainingManager(base_model, pretrain_cfg)
    for task in tasks:
        manager.register_task(task)
    manager.run()

    burgers_loader = make_loader(
        analytic_burgers_solution, 512, cfg.batch_size, cfg.noise
    )
    finetune_cfg = FineTuningConfig(
        epochs=cfg.finetune_epochs, lr=cfg.lr_finetune, device=device
    )
    transfer.fine_tune(burgers_loader, finetune_cfg)
    transfer_error = evaluate(base_model, burgers_loader, device)

    baseline_model = MLP(in_dim=1, hidden_layers=4, width=32, out_dim=1)
    baseline_error = train_baseline(
        baseline_model, burgers_loader, cfg.finetune_epochs, cfg.lr_finetune, device
    )

    print("Transfer learning error:", transfer_error)
    print("Training from scratch error:", baseline_error)


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(
        description="Transfer learning benchmark for PINNs"
    )
    parser.add_argument("--device", default="cpu", help="Device to run on")
    parser.add_argument("--pretrain-epochs", type=int, default=200)
    parser.add_argument("--finetune-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr-pretrain", type=float, default=5e-4)
    parser.add_argument("--lr-finetune", type=float, default=1e-3)
    parser.add_argument("--noise", type=float, default=0.01)
    args = parser.parse_args()
    return BenchmarkConfig(
        device=args.device,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        lr_pretrain=args.lr_pretrain,
        lr_finetune=args.lr_finetune,
        noise=args.noise,
    )


def main() -> None:
    cfg = parse_args()
    run_benchmark(cfg)


if __name__ == "__main__":
    main()

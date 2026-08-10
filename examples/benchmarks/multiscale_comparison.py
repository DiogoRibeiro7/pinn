#!/usr/bin/env python3
"""Benchmark comparing multi-scale and single-scale PINN models."""

import argparse
import time
from pathlib import Path

import torch

from pinnkit.models import (
    MLP,
    MultiScalePINN,
    MultiScaleTrainer,
    MultiScaleTrainingConfig,
)
from pinnkit.utils.logging import get_logger


def create_dataset(
    num_points: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.linspace(0.0, 1.0, num_points, device=device).unsqueeze(1)
    target = torch.sin(2 * torch.pi * x) + 0.5 * torch.sin(10 * torch.pi * x)
    return x, target


def evaluate(
    model: torch.nn.Module, coords: torch.Tensor, targets: torch.Tensor
) -> float:
    with torch.no_grad():
        preds = model(coords)
        return torch.mean((preds - targets) ** 2).item()


def train_single_scale(
    coords: torch.Tensor,
    targets: torch.Tensor,
    *,
    epochs: int,
    lr: float,
) -> tuple[torch.nn.Module, list[float], float]:
    model = MLP(in_dim=1, hidden_layers=4, width=64, out_dim=1).to(coords.device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    start = time.perf_counter()
    for _ in range(epochs):
        optim.zero_grad()
        preds = model(coords)
        loss = torch.mean((preds - targets) ** 2)
        loss.backward()
        optim.step()
        losses.append(loss.item())
    duration = time.perf_counter() - start
    return model, losses, duration


def train_multi_scale(
    coords: torch.Tensor,
    targets: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    scales: tuple[float, ...],
) -> tuple[torch.nn.Module, list[float], float]:
    model = MultiScalePINN(
        in_features=1,
        scales=scales,
        combination_method="learned",
        hidden_layers=3,
        width=64,
    ).to(coords.device)
    epochs_per_scale = max(1, epochs // len(scales))
    trainer = MultiScaleTrainer(
        model,
        MultiScaleTrainingConfig(
            lr=lr,
            epochs_per_scale=epochs_per_scale,
            consistency_weight=0.05,
            device=str(coords.device),
        ),
    )
    start = time.perf_counter()
    state = trainer.train_progressive({"coordinates": coords, "targets": targets})
    duration = time.perf_counter() - start
    return model, state.losses, duration


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare multi-scale and baseline PINNs"
    )
    parser.add_argument(
        "--epochs", type=int, default=300, help="Training epochs for baseline model"
    )
    parser.add_argument(
        "--num-points", type=int, default=512, help="Number of collocation points"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--scales",
        type=float,
        nargs="*",
        default=(1.0, 2.0, 4.0),
        help="Scales for multi-scale model",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    logger = get_logger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coords, targets = create_dataset(args.num_points, device)

    logger.info("Training single-scale baseline", extra={"epochs": args.epochs})
    baseline_model, baseline_losses, baseline_time = train_single_scale(
        coords, targets, epochs=args.epochs, lr=args.lr
    )
    baseline_mse = evaluate(baseline_model, coords, targets)

    logger.info(
        "Training multi-scale model",
        extra={"epochs": args.epochs, "scales": list(args.scales)},
    )
    multiscale_model, multiscale_losses, multiscale_time = train_multi_scale(
        coords,
        targets,
        epochs=args.epochs,
        lr=args.lr,
        scales=tuple(args.scales),
    )
    multiscale_mse = evaluate(multiscale_model, coords, targets)

    logger.info(
        "Benchmark results",
        extra={
            "baseline_mse": baseline_mse,
            "baseline_time": baseline_time,
            "multiscale_mse": multiscale_mse,
            "multiscale_time": multiscale_time,
        },
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "baseline_losses": baseline_losses,
            "baseline_mse": baseline_mse,
            "baseline_time": baseline_time,
            "multiscale_losses": multiscale_losses,
            "multiscale_mse": multiscale_mse,
            "multiscale_time": multiscale_time,
            "scales": list(args.scales),
        },
        args.output_dir / "multiscale_comparison.pt",
    )


if __name__ == "__main__":
    main()

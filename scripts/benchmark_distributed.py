#!/usr/bin/env python3
"""Simple benchmark comparing single vs multi GPU training.

The benchmark runs a small number of optimisation steps with and without the
:class:`pinnlab.distributed.DistributedTrainer`.  It reports the elapsed time and
can be used in CI to ensure that the distributed utilities do not regress.
"""

import time

import torch

from pinnlab.models import MLP
from pinnlab.solvers.raissi_improved import (
    BurgersConfig,
    TrainConfig,
    ContinuousPINN,
    burgers_residual,
)
from pinnlab.distributed import DistributedPINNTrainer, DistributedTrainer


def run_single() -> tuple[float, DistributedTrainer]:
    cfg = BurgersConfig()
    model = MLP(in_dim=2, hidden_layers=2, width=20, out_dim=1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pinn = ContinuousPINN(model, device, burgers_residual, cfg)
    dev = "cuda:0" if device != "cpu" else "cpu"
    trainer = DistributedTrainer(pinn, devices=[dev])
    tcfg = TrainConfig(n_u0=16, n_bc=16, n_f=500, adam_steps=50, lbfgs_max_iter=0)
    start = time.time()
    trainer.fit(tcfg)
    duration = time.time() - start
    return duration, trainer


def run_with_devices(num_devices: int) -> tuple[float, DistributedTrainer]:
    cfg = BurgersConfig()
    model = MLP(in_dim=2, hidden_layers=2, width=20, out_dim=1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pinn = ContinuousPINN(model, device, burgers_residual, cfg)
    if num_devices <= 1:
        dev = "cuda:0" if device != "cpu" else "cpu"
        trainer = DistributedTrainer(pinn, devices=[dev])
    else:
        devices = list(range(num_devices))
        trainer = DistributedPINNTrainer(
            pinn, devices=devices, strategy="data_parallel"
        )
    tcfg = TrainConfig(n_u0=16, n_bc=16, n_f=500, adam_steps=50, lbfgs_max_iter=0)
    start = time.time()
    trainer.fit(tcfg)
    duration = time.time() - start
    return duration, trainer


def main() -> None:
    single_time, single_trainer = run_single()
    print(f"1 device time   : {single_time:.2f}s")
    results: dict[int, tuple[float, float]] = {1: (single_time, 1.0)}

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        base_time = single_time
        max_devices = torch.cuda.device_count()
        for count in range(2, max_devices + 1):
            duration, trainer = run_with_devices(count)
            efficiency = trainer.scaling_efficiency(base_time)
            results[count] = (duration, efficiency)
            speedup = base_time / duration if duration > 0 else 0.0
            print(
                f"{count} devices time: {duration:.2f}s | "
                f"speedup {speedup:.2f}x | efficiency {efficiency:.2%}"
            )
    else:
        print(
            "GPU not available or only one GPU detected; skipping multi-GPU benchmark."
        )

    if len(results) > 1:
        best = max(results.items(), key=lambda kv: kv[1][1])
        print(f"Best scaling: {best[0]} devices with efficiency {best[1][1]:.2%}")


if __name__ == "__main__":
    main()

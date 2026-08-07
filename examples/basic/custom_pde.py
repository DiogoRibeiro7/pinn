#!/usr/bin/env python3
"""Template example for implementing a custom PDE with PINNs."""

import argparse
from pathlib import Path

from pinn.config.management import ConfigLoader
from pinn.models import MLP
from pinn.utils.logging import get_logger


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Template script for custom PDE PINN implementation"
    )
    parser.add_argument("--config", type=str, help="Configuration file")
    parser.add_argument("--output-dir", type=str, default="./results")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(__name__)
    logger.info("Custom PDE example started")

    if args.config:
        cfg = ConfigLoader.from_yaml(args.config)
        logger.info("Loaded configuration", extra={"config": args.config})
    else:
        cfg = None
        logger.warning("No configuration supplied; using defaults")

    # User would define model, residual function and training loop here
    model = MLP(in_dim=2, hidden_layers=2, width=16, out_dim=1)
    logger.info(
        "Created placeholder model",
        extra={"parameters": sum(p.numel() for p in model.parameters())},
    )

    logger.info(
        "This template is intended for customization. Define your PDE residual, "
        "data sampling, training loop and evaluation in this file."
    )


if __name__ == "__main__":
    main()

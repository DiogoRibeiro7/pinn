"""Command-line interface for the pinn package."""
import argparse

from . import __version__
from .solvers.raissi_improved import demo as burgers_demo


def main() -> None:
    """Entry point for basic command-line utilities."""
    parser = argparse.ArgumentParser(description="PINN command-line tools")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run Burgers equation demonstration",
    )
    args = parser.parse_args()

    if args.demo:
        burgers_demo()
    else:
        print(f"pinn version {__version__}")
        parser.print_help()


if __name__ == "__main__":
    main()

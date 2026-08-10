"""Test helper models."""

from pinnkit.models.mlp import MLP


def simple_mlp() -> MLP:
    """Return a very small MLP suitable for quick tests."""

    return MLP(in_dim=2, hidden_layers=1, width=8, out_dim=1)


__all__ = ["simple_mlp"]

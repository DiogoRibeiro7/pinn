"""Neural network architectures for Physics-Informed Neural Networks."""

from __future__ import annotations

from typing import List, cast

from torch import nn, Tensor

from ..utils.validation import check_array, check_range


class MLP(nn.Module):
    """Multi-layer perceptron with Tanh activations used in PINN models."""

    def __init__(
        self,
        in_dim: int = 2,
        hidden_layers: int = 8,
        width: int = 64,
        out_dim: int = 1,
    ) -> None:
        """Initialize network layers.

        Args:
            in_dim: Input dimension.
            hidden_layers: Number of hidden layers.
            width: Hidden layer width.
            out_dim: Output dimension.
        """
        super().__init__()
        check_range("in_dim", in_dim, min=1)
        check_range("hidden_layers", hidden_layers, min=0)
        check_range("width", width, min=1)
        check_range("out_dim", out_dim, min=1)

        layers: List[nn.Module] = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)
        self.apply(self._xavier_init)

    @staticmethod
    def _xavier_init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, tx: Tensor) -> Tensor:
        first_layer = cast(nn.Linear, self.net[0])
        check_array("tx", tx, shape=(-1, first_layer.in_features), strict=False)
        return self.net(tx)

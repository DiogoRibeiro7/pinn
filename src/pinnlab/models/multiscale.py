"""Multi-scale neural architectures and training helpers for PINNs.

This module collects reusable components for constructing physics-informed
neural networks that can reason over multiple characteristic length or time
scales.  The implementations aim to be expressive yet light-weight so they can
be composed with the rest of the training utilities in this repository.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from .mlp import MLP

NetworkFactory = Callable[[], nn.Module]


def _resolve_factory(
    factory: Optional[NetworkFactory],
    *,
    in_dim: int,
    out_dim: int,
    hidden_layers: int,
    width: int,
) -> NetworkFactory:
    """Return a network factory that produces ``MLP``-like modules."""

    if factory is not None:
        return factory

    def _default_factory() -> nn.Module:
        return MLP(
            in_dim=in_dim, hidden_layers=hidden_layers, width=width, out_dim=out_dim
        )

    return _default_factory


class MultiScaleAttention(nn.Module):
    """Attention mechanism that scores features from multiple scales."""

    def __init__(
        self, num_scales: int, feature_dim: int = 16, hidden_dim: int = 32
    ) -> None:
        super().__init__()
        self.num_scales = num_scales
        self.feature_dim = feature_dim
        self.query_network = nn.Linear(feature_dim, hidden_dim)
        self.key_networks = nn.ModuleList(
            nn.Linear(feature_dim, hidden_dim) for _ in range(num_scales)
        )
        self.value_networks = nn.ModuleList(
            nn.Linear(feature_dim, hidden_dim) for _ in range(num_scales)
        )
        self.projection = nn.Linear(hidden_dim, 1)

    def forward(self, scale_features: Sequence[Tensor]) -> Tensor:
        """Return attention weights of shape ``(batch, num_scales)``."""

        if not scale_features:
            raise ValueError("scale_features must not be empty")

        stacked = torch.stack(list(scale_features), dim=1)  # (batch, scales, dim)
        query = self.query_network(stacked.mean(dim=1))  # (batch, hidden_dim)
        keys = torch.stack(
            [layer(feat) for layer, feat in zip(self.key_networks, scale_features)],
            dim=1,
        )
        values = torch.stack(
            [layer(feat) for layer, feat in zip(self.value_networks, scale_features)],
            dim=1,
        )
        scores = torch.einsum("bd,bkd->bk", query, keys)
        attention = torch.softmax(scores, dim=1)
        attended = torch.einsum("bk,bkd->bd", attention, values)
        _ = self.projection(
            attended
        )  # included for completeness, output not used directly
        return attention


class MultiScalePINN(nn.Module):
    """Physics-informed network that processes inputs at multiple scales."""

    # Declared because nn.Module.__getattr__ is typed Tensor | Module, so a
    # registered buffer otherwise reads back as that union everywhere.
    scales_tensor: Tensor
    _scale_usage: Tensor

    def __init__(
        self,
        *,
        in_features: int = 2,
        out_features: int = 1,
        scales: Iterable[float] = (1.0, 2.0, 4.0),
        base_network_factory: Optional[NetworkFactory] = None,
        hidden_layers: int = 4,
        width: int = 64,
        combination_method: str = "learned",
        feature_dim: int = 16,
        learnable_scales: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.learnable_scales = learnable_scales
        self._combination_method = combination_method
        self._feature_dim = feature_dim

        scale_values = torch.tensor(tuple(scales), dtype=torch.float32)
        if scale_values.ndim != 1 or scale_values.numel() == 0:
            raise ValueError("scales must be a non-empty 1-D iterable")

        if learnable_scales:
            self.log_scales = nn.Parameter(scale_values.log())
        else:
            self.register_buffer("scales_tensor", scale_values, persistent=False)

        factory = _resolve_factory(
            base_network_factory,
            in_dim=in_features,
            out_dim=out_features,
            hidden_layers=hidden_layers,
            width=width,
        )
        self.scale_networks = nn.ModuleList(
            factory() for _ in range(scale_values.numel())
        )
        self.scale_feature_extractors = nn.ModuleList(
            nn.Sequential(
                nn.Linear(in_features, feature_dim),
                nn.Tanh(),
                nn.Dropout(p=dropout),
                nn.Linear(feature_dim, feature_dim),
                nn.Tanh(),
            )
            for _ in range(scale_values.numel())
        )

        if combination_method not in {
            "learned",
            "attention",
            "weighted_average",
            "uniform",
        }:
            raise ValueError("Unknown combination method")

        if combination_method == "learned":
            self.combination_network = nn.Sequential(
                nn.Linear(feature_dim * scale_values.numel(), max(feature_dim, 32)),
                nn.Tanh(),
                nn.Linear(max(feature_dim, 32), scale_values.numel()),
            )
        elif combination_method == "attention":
            self.attention = MultiScaleAttention(
                scale_values.numel(), feature_dim=feature_dim
            )
        elif combination_method == "weighted_average":
            self.combination_logits = nn.Parameter(torch.zeros(scale_values.numel()))

        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer(
            "_scale_usage", torch.zeros(scale_values.numel()), persistent=False
        )

    def extra_repr(self) -> str:  # pragma: no cover - representation helper
        return f"scales={self.get_scales().tolist()}, combination='{self._combination_method}'"

    def get_scales(self) -> Tensor:
        return (
            torch.exp(self.log_scales) if self.learnable_scales else self.scales_tensor
        )

    def forward(self, x: Tensor) -> Tensor:
        combined, _, _ = self.forward_with_details(x)
        return combined

    def forward_with_details(self, x: Tensor) -> Tuple[Tensor, List[Tensor], Tensor]:
        """Return combined prediction, per-scale outputs and weights."""

        scales = self.get_scales().to(x.device)
        outputs: List[Tensor] = []
        features: List[Tensor] = []
        for scale, network, extractor in zip(
            scales, self.scale_networks, self.scale_feature_extractors
        ):
            scaled_input = x * scale
            feat = extractor(scaled_input)
            features.append(feat)
            outputs.append(network(scaled_input))

        weights = self._combine(outputs, features, x)
        stacked = torch.stack(outputs, dim=-1)  # (batch, out_features, num_scales)
        combined = torch.einsum("bon,bn->bo", stacked, weights)
        self._scale_usage = (
            0.95 * self._scale_usage + 0.05 * weights.mean(dim=0).detach()
        )
        return combined, outputs, weights

    def _combine(
        self, outputs: Sequence[Tensor], features: Sequence[Tensor], x: Tensor
    ) -> Tensor:
        num_scales = len(outputs)
        if self._combination_method == "learned":
            combined_features = torch.cat(list(features), dim=-1)
            logits = self.combination_network(combined_features)
            return torch.softmax(logits, dim=-1)
        if self._combination_method == "attention":
            return self.attention(features)
        if self._combination_method == "weighted_average":
            return torch.softmax(self.combination_logits, dim=0).expand(
                x.shape[0], num_scales
            )
        return torch.full(
            (x.shape[0], num_scales), 1.0 / num_scales, device=x.device, dtype=x.dtype
        )

    def predict_with_uncertainty(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        combined, outputs, weights = self.forward_with_details(x)
        stacked = torch.stack(outputs, dim=-1)
        variance = torch.var(stacked, dim=-1, unbiased=False)
        return combined, variance.sqrt()

    def scale_statistics(self) -> Dict[str, Tensor]:
        scales = self.get_scales().detach()
        return {
            "scales": scales,
            "usage": self._scale_usage.detach(),
        }

    def adapt_scales(self, factors: Tensor, momentum: float = 0.2) -> None:
        """Update learnable scales with simple exponential smoothing."""

        if not self.learnable_scales:
            raise RuntimeError("Scales are not learnable in this instance")
        if factors.shape != self.log_scales.shape:
            raise ValueError("factors must match number of scales")
        with torch.no_grad():
            log_update = torch.log(factors.clamp(min=1e-3))
            self.log_scales.mul_(1 - momentum).add_(log_update * momentum)


class HierarchicalPINN(nn.Module):
    """Coarse-to-fine hierarchy where each level refines the previous output."""

    def __init__(
        self,
        *,
        in_features: int = 2,
        levels: int = 3,
        base_width: int = 32,
        growth: float = 2.0,
    ) -> None:
        super().__init__()
        if levels < 1:
            raise ValueError("levels must be >= 1")
        self.levels = levels
        self.level_networks = nn.ModuleList()
        width: float = float(base_width)
        for level in range(levels):
            net = MLP(
                in_dim=in_features, hidden_layers=3 + level, width=int(width), out_dim=1
            )
            self.level_networks.append(net)
            width *= growth
        self.correction_layers = nn.ModuleList(
            nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 1))
            for _ in range(levels - 1)
        )

    def forward(
        self, x: Tensor, *, return_all: bool = False
    ) -> Tensor | Tuple[List[Tensor], Tensor]:
        outputs: List[Tensor] = []
        for idx, net in enumerate(self.level_networks):
            prediction = net(x)
            if idx > 0:
                correction = self.correction_layers[idx - 1](prediction)
                prediction = outputs[-1] + correction
            outputs.append(prediction)
        return (outputs, outputs[-1]) if return_all else outputs[-1]


class WaveletPINN(nn.Module):
    """Multi-resolution network that augments inputs with wavelet responses."""

    def __init__(
        self,
        *,
        in_features: int = 2,
        wavelet_levels: int = 3,
        wavelet_type: str = "mexican_hat",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.wavelet_levels = wavelet_levels
        if wavelet_levels < 1:
            raise ValueError("wavelet_levels must be >= 1")
        if wavelet_type not in {"mexican_hat", "morlet"}:
            raise ValueError("Unsupported wavelet type")
        self.wavelet_type = wavelet_type
        self.feature_extractors = nn.ModuleList(
            nn.Sequential(
                nn.Linear(in_features + 1, 32),
                nn.Tanh(),
                nn.Linear(32, 16),
                nn.Tanh(),
                nn.Linear(16, 8),
                nn.Tanh(),
            )
            for _ in range(wavelet_levels)
        )
        total_features = in_features + wavelet_levels * 8
        self.main_network = MLP(
            in_dim=total_features, hidden_layers=6, width=64, out_dim=1
        )

    def forward(self, x: Tensor) -> Tensor:
        features = [x]
        for level in range(self.wavelet_levels):
            scale = 2**level
            scaled = x * scale
            norm = scaled.pow(2).sum(dim=1, keepdim=True)
            if self.wavelet_type == "mexican_hat":
                wavelet = (1 - norm) * torch.exp(-0.5 * norm)
            else:
                wavelet = torch.cos(5 * scaled).prod(dim=1, keepdim=True) * torch.exp(
                    -0.5 * norm
                )
            augmented = torch.cat([x, wavelet], dim=1)
            features.append(self.feature_extractors[level](augmented))
        return self.main_network(torch.cat(features, dim=1))


class FourierMultiScalePINN(MultiScalePINN):
    """Specialised multi-scale model using Fourier feature encodings."""

    def __init__(
        self,
        *,
        in_features: int = 1,
        mapping_size: int = 64,
        sigma: float = 1.0,
        **kwargs,
    ) -> None:
        def factory() -> nn.Module:
            B = sigma * torch.randn(in_features, mapping_size)

            class _FourierModule(nn.Module):
                B: Tensor

                def __init__(self) -> None:
                    super().__init__()
                    self.register_buffer("B", B, persistent=False)
                    self.mlp = MLP(
                        in_dim=2 * mapping_size, hidden_layers=4, width=64, out_dim=1
                    )

                def forward(self, x: Tensor) -> Tensor:
                    proj = 2 * torch.pi * x @ self.B
                    emb = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
                    return self.mlp(emb)

            return _FourierModule()

        super().__init__(
            in_features=in_features,
            base_network_factory=factory,
            **kwargs,
        )


@dataclass
class MultiScaleTrainingConfig:
    """Configuration for :class:`MultiScaleTrainer`."""

    lr: float = 1e-3
    epochs_per_scale: int = 100
    consistency_weight: float = 0.1
    freeze_lower_scales: bool = False
    device: str = "cpu"
    gradient_clip: Optional[float] = 1.0


@dataclass
class MultiScaleTrainingState:
    losses: List[float] = field(default_factory=list)
    consistency: List[float] = field(default_factory=list)


class MultiScaleTrainer:
    """Utility class implementing progressive training across scales."""

    def __init__(self, model: MultiScalePINN, config: MultiScaleTrainingConfig) -> None:
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.state = MultiScaleTrainingState()
        self.device = torch.device(config.device)
        self.model.to(self.device)

    def _consistency_loss(self, per_scale: Sequence[Tensor]) -> Tensor:
        if not per_scale:
            return torch.tensor(0.0, device=self.device)
        stacked = torch.stack(list(per_scale), dim=-1)
        variance = torch.var(stacked, dim=-1, unbiased=False)
        return variance.mean()

    def train_progressive(self, data: Dict[str, Tensor]) -> MultiScaleTrainingState:
        coordinates = data["coordinates"].to(self.device)
        targets = data["targets"].to(self.device)
        num_scales = len(self.model.get_scales())

        for scale_idx in range(num_scales):
            if self.config.freeze_lower_scales and scale_idx > 0:
                for param in self.model.scale_networks[scale_idx - 1].parameters():
                    param.requires_grad = False
            for epoch in range(self.config.epochs_per_scale):
                self.optimizer.zero_grad()
                prediction, per_scale, _ = self.model.forward_with_details(coordinates)
                mse = torch.mean((prediction - targets) ** 2)
                consistency = self._consistency_loss(per_scale)
                loss = mse + self.config.consistency_weight * consistency
                loss.backward()
                if self.config.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip
                    )
                self.optimizer.step()
                self.state.losses.append(loss.item())
                self.state.consistency.append(consistency.item())
        return self.state


def multiscale_loss(
    prediction: Tensor,
    target: Tensor,
    per_scale_outputs: Sequence[Tensor],
    *,
    consistency_weight: float = 0.1,
) -> Tuple[Tensor, Dict[str, float]]:
    """Compute a loss that balances accuracy and cross-scale consistency."""

    mse = torch.mean((prediction - target) ** 2)
    if per_scale_outputs:
        stacked = torch.stack(list(per_scale_outputs), dim=-1)
        consistency = torch.var(stacked, dim=-1, unbiased=False).mean()
    else:
        consistency = torch.tensor(0.0, device=prediction.device)
    total = mse + consistency_weight * consistency
    return total, {
        "mse": float(mse.detach()),
        "consistency": float(consistency.detach()),
    }


def scale_adaptive_sampling(
    sampler: Callable[[int], Tensor],
    importance: Tensor,
    n_points: int,
) -> Tensor:
    """Sample more points from regions with higher multi-scale importance."""

    if importance.numel() == 0:
        raise ValueError("importance tensor must not be empty")
    weights = torch.softmax(importance.view(-1), dim=0)
    indices = torch.multinomial(weights, n_points, replacement=True)
    base_points = sampler(n_points)
    return base_points[indices % base_points.shape[0]]


__all__ = [
    "MultiScalePINN",
    "MultiScaleAttention",
    "HierarchicalPINN",
    "WaveletPINN",
    "FourierMultiScalePINN",
    "MultiScaleTrainingConfig",
    "MultiScaleTrainingState",
    "MultiScaleTrainer",
    "multiscale_loss",
    "scale_adaptive_sampling",
]

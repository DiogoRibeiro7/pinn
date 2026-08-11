"""Domain adaptation utilities for transfer learning between PDE domains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, MutableMapping, Optional, Sequence

import torch
from torch import nn

from ..utils.logging import get_logger
from .pretrain import ensure_batch_dict, move_batch_to_device

BatchType = MutableMapping[str, torch.Tensor]


@dataclass
class DomainAdaptationConfig:
    """Configuration for :class:`DomainAdaptationPINN`."""

    method: str = "adversarial"
    lr: float = 1e-3
    epochs: int = 200
    device: str = "cpu"
    discriminator_hidden: Sequence[int] = (64, 32)
    coral_lambda: float = 0.1
    mmd_bandwidth: float = 1.0


class DomainAdaptationPINN:
    """Adapt a source PINN to a target domain using different strategies."""

    def __init__(
        self,
        source_model: torch.nn.Module,
        feature_extractor: Optional[
            Callable[[torch.nn.Module, torch.Tensor], torch.Tensor]
        ] = None,
        config: Optional[DomainAdaptationConfig] = None,
    ) -> None:
        self.source_model = source_model
        self.config = config or DomainAdaptationConfig()
        self.logger = get_logger(__name__)
        self.feature_extractor = feature_extractor or self._default_feature_extractor
        self.domain_discriminator: Optional[torch.nn.Module]
        if self.config.method == "adversarial":
            self.domain_discriminator = self._create_discriminator()
        else:
            self.domain_discriminator = None

    def adapt(
        self,
        source_loader: Iterable[BatchType],
        target_loader: Iterable[BatchType],
    ) -> Dict[str, float]:
        """Adapt the source model to the target domain and return diagnostics."""

        method = self.config.method
        if method == "adversarial":
            return self._adversarial_adaptation(source_loader, target_loader)
        if method == "coral":
            return self._coral_adaptation(source_loader, target_loader)
        if method == "mmd":
            return self._mmd_adaptation(source_loader, target_loader)
        raise ValueError(f"Unknown adaptation method: {method}")

    # ------------------------------------------------------------------
    # Adaptation strategies
    # ------------------------------------------------------------------
    def _adversarial_adaptation(
        self,
        source_loader: Iterable[BatchType],
        target_loader: Iterable[BatchType],
    ) -> Dict[str, float]:
        assert self.domain_discriminator is not None
        device = torch.device(self.config.device)
        self.source_model.to(device)
        self.domain_discriminator.to(device)
        feature_optimizer = torch.optim.Adam(
            self.source_model.parameters(), lr=self.config.lr
        )
        discriminator_optimizer = torch.optim.Adam(
            self.domain_discriminator.parameters(), lr=self.config.lr
        )
        history = {"discriminator_loss": 0.0, "adversarial_loss": 0.0}

        target_iter = iter(target_loader)
        for epoch in range(self.config.epochs):
            for source_batch in _iterate_batches(source_loader):
                try:
                    target_batch = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    target_batch = next(target_iter)
                source_batch = move_batch_to_device(source_batch, device)
                target_batch = move_batch_to_device(
                    ensure_batch_dict(target_batch), device
                )

                # Train discriminator
                discriminator_optimizer.zero_grad()
                source_features = self.feature_extractor(
                    self.source_model, source_batch["input"]
                )
                target_features = self.feature_extractor(
                    self.source_model, target_batch["input"]
                )
                src_logits = self.domain_discriminator(source_features.detach())
                tgt_logits = self.domain_discriminator(target_features.detach())
                loss_src = torch.nn.functional.binary_cross_entropy_with_logits(
                    src_logits, torch.ones_like(src_logits)
                )
                loss_tgt = torch.nn.functional.binary_cross_entropy_with_logits(
                    tgt_logits, torch.zeros_like(tgt_logits)
                )
                disc_loss = loss_src + loss_tgt
                disc_loss.backward()
                discriminator_optimizer.step()

                # Train feature extractor to fool discriminator
                feature_optimizer.zero_grad()
                target_features = self.feature_extractor(
                    self.source_model, target_batch["input"]
                )
                fool_logits = self.domain_discriminator(target_features)
                adversarial_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    fool_logits, torch.ones_like(fool_logits)
                )
                adversarial_loss.backward()
                feature_optimizer.step()

                history["discriminator_loss"] += disc_loss.item()
                history["adversarial_loss"] += adversarial_loss.item()
            if epoch % max(self.config.epochs // 10, 1) == 0:
                self.logger.debug(
                    "Adversarial adaptation epoch %d/%d - D %.4e, A %.4e",
                    epoch + 1,
                    self.config.epochs,
                    history["discriminator_loss"],
                    history["adversarial_loss"],
                )
        return history

    def _coral_adaptation(
        self,
        source_loader: Iterable[BatchType],
        target_loader: Iterable[BatchType],
    ) -> Dict[str, float]:
        device = torch.device(self.config.device)
        optimizer = torch.optim.Adam(self.source_model.parameters(), lr=self.config.lr)
        history = {"coral_loss": 0.0, "task_loss": 0.0}
        for epoch in range(self.config.epochs):
            for source_batch, target_batch in zip(
                _iterate_batches(source_loader), _iterate_batches(target_loader)
            ):
                source_batch = move_batch_to_device(source_batch, device)
                target_batch = move_batch_to_device(target_batch, device)
                optimizer.zero_grad()
                source_features = self.feature_extractor(
                    self.source_model, source_batch["input"]
                )
                target_features = self.feature_extractor(
                    self.source_model, target_batch["input"]
                )
                coral_loss = self._coral_loss(source_features, target_features)
                preds = self.source_model(source_batch["input"])
                task_loss = torch.tensor(0.0, device=device)
                if "target" in source_batch:
                    task_loss = torch.nn.functional.mse_loss(
                        preds, source_batch["target"]
                    )
                loss = task_loss + self.config.coral_lambda * coral_loss
                loss.backward()
                optimizer.step()
                history["coral_loss"] += coral_loss.item()
                history["task_loss"] += task_loss.item()
        return history

    def _mmd_adaptation(
        self,
        source_loader: Iterable[BatchType],
        target_loader: Iterable[BatchType],
    ) -> Dict[str, float]:
        device = torch.device(self.config.device)
        optimizer = torch.optim.Adam(self.source_model.parameters(), lr=self.config.lr)
        history = {"mmd_loss": 0.0}
        for epoch in range(self.config.epochs):
            for source_batch, target_batch in zip(
                _iterate_batches(source_loader), _iterate_batches(target_loader)
            ):
                source_batch = move_batch_to_device(source_batch, device)
                target_batch = move_batch_to_device(target_batch, device)
                optimizer.zero_grad()
                source_features = self.feature_extractor(
                    self.source_model, source_batch["input"]
                )
                target_features = self.feature_extractor(
                    self.source_model, target_batch["input"]
                )
                mmd_loss = self._maximum_mean_discrepancy(
                    source_features, target_features
                )
                mmd_loss.backward()
                optimizer.step()
                history["mmd_loss"] += mmd_loss.item()
        return history

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _create_discriminator(self) -> nn.Module:
        layers: List[nn.Module] = []
        for hidden in self.config.discriminator_hidden:
            layers.append(nn.LazyLinear(hidden))
            layers.append(nn.ReLU())
        layers.append(nn.LazyLinear(1))
        return nn.Sequential(*layers)

    def _default_feature_extractor(
        self, model: torch.nn.Module, inputs: torch.Tensor
    ) -> torch.Tensor:
        extractor = getattr(model, "feature_extractor", None)
        if extractor is not None:
            return extractor(inputs)
        outputs = model(inputs)
        if outputs.dim() == 1:
            return outputs.unsqueeze(-1)
        return outputs

    @staticmethod
    def _coral_loss(
        source_features: torch.Tensor, target_features: torch.Tensor
    ) -> torch.Tensor:
        source = source_features - source_features.mean(dim=0, keepdim=True)
        target = target_features - target_features.mean(dim=0, keepdim=True)
        cov_source = source.t() @ source / max(source.shape[0] - 1, 1)
        cov_target = target.t() @ target / max(target.shape[0] - 1, 1)
        return torch.norm(cov_source - cov_target, p="fro") ** 2

    def _maximum_mean_discrepancy(
        self, source_features: torch.Tensor, target_features: torch.Tensor
    ) -> torch.Tensor:
        bandwidth = self.config.mmd_bandwidth
        kernel = self._gaussian_kernel(source_features, source_features, bandwidth)
        kernel += self._gaussian_kernel(target_features, target_features, bandwidth)
        kernel -= 2 * self._gaussian_kernel(source_features, target_features, bandwidth)
        return kernel.mean()

    @staticmethod
    def _gaussian_kernel(
        x: torch.Tensor, y: torch.Tensor, bandwidth: float
    ) -> torch.Tensor:
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        return torch.exp(-diff.pow(2).sum(-1) / (2 * bandwidth**2))


def _iterate_batches(loader: Iterable[BatchType]) -> Iterable[BatchType]:
    for batch in loader:
        yield ensure_batch_dict(batch)


__all__ = ["DomainAdaptationPINN", "DomainAdaptationConfig"]

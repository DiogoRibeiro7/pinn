"""Fine-tuning utilities and high level transfer-learning workflows."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, MutableMapping, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from ..utils.logging import get_logger
from .pretrain import PretrainingConfig, PretrainingManager, PretrainingTask, ensure_batch_dict, move_batch_to_device

BatchType = MutableMapping[str, torch.Tensor]


@dataclass
class FineTuningConfig:
    """Configuration describing a fine-tuning experiment."""

    epochs: int = 500
    lr: float = 1e-4
    device: str = "cpu"
    gradient_clip: Optional[float] = None
    freeze_layers: Optional[Sequence[str]] = None
    unfreeze_interval: int = 200
    patience: Optional[int] = None
    use_amp: bool = False
    loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None


class ContinualLearningBuffer:
    """Simple replay buffer used for continual learning scenarios."""

    def __init__(self, memory_size: int = 1024) -> None:
        self.memory_size = memory_size
        self._storage: List[BatchType] = []

    def add_batch(self, batch: BatchType) -> None:
        clone = {key: value.detach().cpu() for key, value in batch.items() if torch.is_tensor(value)}
        self._storage.append(clone)
        if len(self._storage) > self.memory_size:
            self._storage = self._storage[-self.memory_size :]

    def sample(self, batch_size: int) -> Optional[BatchType]:
        if not self._storage:
            return None
        batch = self._storage[0]
        self._storage = self._storage[1:] + [batch]
        return {key: value[:batch_size] for key, value in batch.items()}


class TransferLearningPINN:
    """High-level utility orchestrating transfer-learning workflows for PINNs."""

    def __init__(
        self,
        base_model: torch.nn.Module,
        device: str = "cpu",
        transfer_strategy: str = "parameter_transfer",
    ) -> None:
        self.base_model = base_model
        self.device = torch.device(device)
        self.transfer_strategy = transfer_strategy
        self.logger = get_logger(__name__)
        self.source_models: Dict[str, torch.nn.Module] = {}
        self.adaptation_layers: Dict[str, torch.nn.Module] = {}
        self.replay_buffer = ContinualLearningBuffer()
        self._fisher_information: Dict[str, torch.Tensor] = {}
        self._reference_params: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Pre-training and registration helpers
    # ------------------------------------------------------------------
    def pretrain(self, tasks: Sequence[PretrainingTask], config: Optional[PretrainingConfig] = None) -> Dict[str, Dict[str, float]]:
        """Run pre-training using :class:`PretrainingManager`."""

        manager = PretrainingManager(self.base_model, config)
        for task in tasks:
            manager.register_task(task)
        return manager.run()

    def register_source_model(self, name: str, model: torch.nn.Module) -> None:
        self.source_models[name] = model

    # ------------------------------------------------------------------
    # Knowledge transfer primitives
    # ------------------------------------------------------------------
    def transfer_parameters(self, source: torch.nn.Module) -> int:
        """Copy matching parameters from ``source`` to the base model."""

        src_dict = source.state_dict()
        tgt_dict = self.base_model.state_dict()
        transferred = 0
        for key, value in src_dict.items():
            if key in tgt_dict and tgt_dict[key].shape == value.shape:
                tgt_dict[key] = value.detach().clone()
                transferred += 1
        self.base_model.load_state_dict(tgt_dict)
        self.logger.info("Transferred %d parameters from source model", transferred)
        return transferred

    def transfer_features(self, source: torch.nn.Module, freeze: bool = True, depth: int = 2) -> List[str]:
        """Transfer the first ``depth`` layers and optionally freeze them."""

        transferred_layers: List[str] = []
        src_modules = list(source.named_modules())
        tgt_modules = dict(self.base_model.named_modules())
        for name, module in src_modules:
            if name == "":
                continue
            if len(transferred_layers) >= depth:
                break
            if name in tgt_modules:
                src_state = dict(module.state_dict())
                tgt_modules[name].load_state_dict(src_state)
                transferred_layers.append(name)
                if freeze:
                    for param in tgt_modules[name].parameters():
                        param.requires_grad = False
        if freeze:
            self.logger.info("Frozen layers: %s", transferred_layers)
        return transferred_layers

    def progressive_transfer(self, sources: Sequence[torch.nn.Module]) -> int:
        """Apply progressive layer-wise transfer from multiple sources."""

        transferred = 0
        target_layers = list(self.base_model.children())
        for src_model, tgt_layer in zip(sources, target_layers):
            src_layers = list(src_model.children())
            if not src_layers:
                continue
            src_state = src_layers[-1].state_dict()
            tgt_layer.load_state_dict(src_state, strict=False)
            transferred += 1
        return transferred

    def distill_from(
        self,
        teacher: torch.nn.Module,
        data_loader: Iterable[BatchType],
        temperature: float = 3.0,
        alpha: float = 0.5,
        epochs: int = 200,
    ) -> None:
        """Perform knowledge distillation from ``teacher`` using the provided data."""

        from .knowledge_distillation import DistillationConfig, KnowledgeDistillationTrainer

        config = DistillationConfig(temperature=temperature, alpha=alpha, epochs=epochs)
        trainer = KnowledgeDistillationTrainer(teacher, self.base_model, config)
        trainer.train(data_loader)

    # ------------------------------------------------------------------
    # Fine-tuning
    # ------------------------------------------------------------------
    def fine_tune(
        self,
        data_loader: Iterable[BatchType] | DataLoader,
        config: Optional[FineTuningConfig] = None,
        optimizer_factory: Callable[[Iterable[torch.nn.Parameter], float], torch.optim.Optimizer]
        | None = None,
    ) -> Dict[str, float]:
        """Fine-tune the base model on ``data_loader``."""

        cfg = config or FineTuningConfig()
        self.base_model.to(self.device)
        if cfg.freeze_layers:
            self._set_frozen(cfg.freeze_layers, True)
        optimizer_factory = optimizer_factory or (lambda params, lr: torch.optim.Adam(params, lr=lr))
        params = [p for p in self.base_model.parameters() if p.requires_grad]
        optimizer = optimizer_factory(params, cfg.lr)
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
        best_loss = float("inf")
        epochs_without_improve = 0

        for epoch in range(cfg.epochs):
            running = 0.0
            steps = 0
            for batch in _iterate_batches(data_loader):
                batch = move_batch_to_device(batch, self.device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                    preds = self.base_model(batch["input"])
                    target = batch.get("target")
                    loss_fn = cfg.loss_fn or torch.nn.functional.mse_loss
                    if target is None:
                        raise ValueError("Fine-tuning requires batches to provide a 'target' tensor")
                    loss = loss_fn(preds, target)
                scaler.scale(loss).backward()
                if cfg.gradient_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, cfg.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                running += loss.item()
                steps += 1
            epoch_loss = running / max(steps, 1)
            self.logger.debug("Fine-tune epoch %d/%d loss %.4e", epoch + 1, cfg.epochs, epoch_loss)
            if cfg.unfreeze_interval and cfg.freeze_layers and (epoch + 1) % cfg.unfreeze_interval == 0:
                self._gradually_unfreeze()
            if epoch_loss + 1e-8 < best_loss:
                best_loss = epoch_loss
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1
                if cfg.patience is not None and epochs_without_improve >= cfg.patience:
                    self.logger.info("Early stopping fine-tuning after %d epochs", epoch + 1)
                    break
        return {"loss": best_loss}

    # ------------------------------------------------------------------
    # Continual, multi-task and few-shot learning utilities
    # ------------------------------------------------------------------
    def continual_update(
        self,
        data_loader: Iterable[BatchType],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        regularization_strength: float = 1e-2,
    ) -> float:
        """Perform a continual-learning update with experience replay and EWC-like penalty."""

        loss_fn = loss_fn or torch.nn.functional.mse_loss
        optimizer = torch.optim.Adam(self.base_model.parameters(), lr=1e-4)
        self.base_model.to(self.device)
        self._capture_reference_parameters()
        running_loss = 0.0
        steps = 0

        for batch in _iterate_batches(data_loader):
            batch = move_batch_to_device(batch, self.device)
            optimizer.zero_grad()
            preds = self.base_model(batch["input"])
            target = batch.get("target")
            if target is None:
                raise ValueError("Continual learning expects a 'target' tensor")
            loss = loss_fn(preds, target)
            if self._reference_params:
                penalty = 0.0
                for name, param in self.base_model.named_parameters():
                    ref = self._reference_params.get(name)
                    fisher = self._fisher_information.get(name)
                    if ref is None or fisher is None:
                        continue
                    penalty = penalty + (fisher * (param - ref) ** 2).sum()
                loss = loss + regularization_strength * penalty
            loss.backward()
            optimizer.step()
            self._update_fisher_information()
            self.replay_buffer.add_batch(batch)
            running_loss += loss.item()
            steps += 1

            replay = self.replay_buffer.sample(batch["input"].shape[0])
            if replay is not None:
                replay = move_batch_to_device(replay, self.device)
                optimizer.zero_grad()
                preds = self.base_model(replay["input"])
                replay_loss = loss_fn(preds, replay["target"])
                replay_loss.backward()
                optimizer.step()
                running_loss += replay_loss.item()
                steps += 1
        return running_loss / max(steps, 1)

    def train_multi_task(
        self,
        task_loaders: Sequence[Iterable[BatchType]],
        epochs: int = 200,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> float:
        """Jointly train on several tasks for multi-task learning."""

        loss_fn = loss_fn or torch.nn.functional.mse_loss
        optimizer = torch.optim.Adam(self.base_model.parameters(), lr=1e-3)
        self.base_model.to(self.device)
        total_loss = 0.0
        for epoch in range(epochs):
            for loader in task_loaders:
                for batch in _iterate_batches(loader):
                    batch = move_batch_to_device(batch, self.device)
                    optimizer.zero_grad()
                    preds = self.base_model(batch["input"])
                    loss = loss_fn(preds, batch["target"])
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
        return total_loss / max(epochs, 1)

    def few_shot_adapt(
        self,
        support_batch: BatchType,
        query_batch: BatchType,
        inner_steps: int = 5,
        lr: float = 1e-2,
    ) -> float:
        """Perform few-shot adaptation using a lightweight inner loop."""

        support = move_batch_to_device(support_batch, self.device)
        query = move_batch_to_device(query_batch, self.device)
        cloned = copy.deepcopy(self.base_model)
        optimizer = torch.optim.SGD(cloned.parameters(), lr=lr)
        for _ in range(inner_steps):
            optimizer.zero_grad()
            preds = cloned(support["input"])
            loss = torch.nn.functional.mse_loss(preds, support["target"])
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            preds = cloned(query["input"])
            query_loss = torch.nn.functional.mse_loss(preds, query["target"])
        return float(query_loss.item())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _set_frozen(self, layer_names: Sequence[str], frozen: bool) -> None:
        for name, param in self.base_model.named_parameters():
            if any(layer in name for layer in layer_names):
                param.requires_grad = not frozen

    def _gradually_unfreeze(self) -> None:
        for param in self.base_model.parameters():
            if not param.requires_grad:
                param.requires_grad = True
                self.logger.info("Unfroze parameter with %d elements", param.numel())
                break

    def _capture_reference_parameters(self) -> None:
        self._reference_params = {name: param.detach().clone() for name, param in self.base_model.named_parameters()}

    def _update_fisher_information(self) -> None:
        for name, param in self.base_model.named_parameters():
            if param.grad is None:
                continue
            grad_sq = param.grad.detach() ** 2
            if name not in self._fisher_information:
                self._fisher_information[name] = grad_sq
            else:
                self._fisher_information[name] = 0.5 * (self._fisher_information[name] + grad_sq)


def _iterate_batches(loader: Iterable[BatchType] | DataLoader) -> Iterable[BatchType]:
    for batch in loader:
        yield ensure_batch_dict(batch)


__all__ = [
    "FineTuningConfig",
    "TransferLearningPINN",
    "ContinualLearningBuffer",
]

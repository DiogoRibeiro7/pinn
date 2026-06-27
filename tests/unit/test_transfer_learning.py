"""Tests for transfer learning utilities."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from pinn.models.mlp import MLP
from pinn.transfer import (
    DomainAdaptationConfig,
    DomainAdaptationPINN,
    FineTuningConfig,
    KnowledgeDistillationTrainer,
    DistillationConfig,
    PretrainingConfig,
    PretrainingManager,
    create_linear_reference_task,
    TransferLearningPINN,
)


def _make_linear_loader(slope: float, intercept: float, num: int = 64) -> DataLoader:
    x = torch.linspace(0.0, 1.0, num).unsqueeze(-1)
    y = slope * x + intercept
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=16, shuffle=True)


def test_pretraining_manager_runs() -> None:
    model = MLP(in_dim=1, hidden_layers=2, width=16, out_dim=1)
    task = create_linear_reference_task(num_points=64, batch_size=16)
    task.epochs = 5
    manager = PretrainingManager(model, PretrainingConfig(lr=1e-3, device="cpu"))
    manager.register_task(task)
    results = manager.run()
    assert "linear_reference" in results
    assert results["linear_reference"]["avg_loss"] >= 0.0


def test_transfer_learning_parameter_copy() -> None:
    source = MLP(in_dim=1, hidden_layers=2, width=16, out_dim=1)
    target = MLP(in_dim=1, hidden_layers=2, width=16, out_dim=1)
    transfer = TransferLearningPINN(target)
    copied = transfer.transfer_parameters(source)
    assert copied > 0
    for p_src, p_tgt in zip(source.parameters(), target.parameters()):
        assert torch.allclose(p_src, p_tgt)


def test_fine_tuning_reduces_loss() -> None:
    torch.manual_seed(0)
    model = MLP(in_dim=1, hidden_layers=2, width=16, out_dim=1)
    transfer = TransferLearningPINN(model)
    loader = _make_linear_loader(1.5, -0.2)
    before = evaluate_model(model, loader)
    transfer.fine_tune(loader, FineTuningConfig(epochs=20, lr=1e-2, device="cpu"))
    after = evaluate_model(model, loader)
    assert after <= before


def evaluate_model(model: torch.nn.Module, loader: DataLoader) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb)
            total += torch.nn.functional.mse_loss(pred, yb).item() * xb.shape[0]
            count += xb.shape[0]
    model.train()
    return total / max(count, 1)


def test_knowledge_distillation_improves_student() -> None:
    torch.manual_seed(0)
    teacher = MLP(in_dim=1, hidden_layers=2, width=16, out_dim=1)
    student = MLP(in_dim=1, hidden_layers=2, width=16, out_dim=1)
    loader = _make_linear_loader(0.8, 0.1)
    before = evaluate_model(student, loader)
    trainer = KnowledgeDistillationTrainer(
        teacher,
        student,
        DistillationConfig(epochs=10, temperature=2.0, alpha=0.7, lr=1e-2, device="cpu"),
    )
    trainer.train(loader)
    after = evaluate_model(student, loader)
    assert after <= before


def test_domain_adaptation_returns_metrics() -> None:
    model = MLP(in_dim=1, hidden_layers=2, width=16, out_dim=1)
    source_loader = _make_linear_loader(1.0, 0.0)
    target_loader = _make_linear_loader(1.5, 0.5)
    adapter = DomainAdaptationPINN(model, config=DomainAdaptationConfig(method="coral", epochs=5, lr=1e-2))
    history = adapter.adapt(source_loader, target_loader)
    assert "coral_loss" in history



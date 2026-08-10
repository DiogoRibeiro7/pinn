"""Transfer learning utilities for Physics-Informed Neural Networks."""

from .pretrain import (
    PretrainingTask,
    PretrainingConfig,
    PretrainingManager,
    build_simplified_pretraining_task,
    progressive_complexity_schedule,
    create_cross_pde_pretraining_tasks,
    create_linear_reference_task,
)
from .finetune import FineTuningConfig, TransferLearningPINN, ContinualLearningBuffer
from .knowledge_distillation import KnowledgeDistillationTrainer, DistillationConfig
from .domain_adaptation import DomainAdaptationPINN, DomainAdaptationConfig

__all__ = [
    "PretrainingTask",
    "PretrainingConfig",
    "PretrainingManager",
    "build_simplified_pretraining_task",
    "progressive_complexity_schedule",
    "create_cross_pde_pretraining_tasks",
    "create_linear_reference_task",
    "FineTuningConfig",
    "TransferLearningPINN",
    "ContinualLearningBuffer",
    "KnowledgeDistillationTrainer",
    "DistillationConfig",
    "DomainAdaptationPINN",
    "DomainAdaptationConfig",
]

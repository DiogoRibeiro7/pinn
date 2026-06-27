"""Training strategies for advanced PINN workflows."""

from .adaptive_weighting import (
    AdaptiveLossWeighting,
    AdaptivePINNTrainer,
    AdaptiveWeightingConfig,
    LossBalancingAnalyzer,
    MultiObjectiveOptimizer,
    WeightEvolutionVisualizer,
    compute_component_gradients,
    compute_gradient_norms,
)
from .adversarial import adversarial_training_step
from .curriculum import curriculum_train
from .memory_efficient import MemoryEfficientPINN, MemoryProfiler, MemoryUsageRecord
from .meta_learning import meta_learning_step
from .multi_fidelity import multi_fidelity_train
from .optimization import (
    adaptive_loss_weights,
    architecture_search,
    gradient_hyperparam_search,
    second_order_step,
)

__all__ = [
    "curriculum_train",
    "adversarial_training_step",
    "meta_learning_step",
    "multi_fidelity_train",
    "adaptive_loss_weights",
    "second_order_step",
    "gradient_hyperparam_search",
    "architecture_search",
    "MemoryEfficientPINN",
    "MemoryProfiler",
    "MemoryUsageRecord",
    "AdaptiveLossWeighting",
    "AdaptiveWeightingConfig",
    "AdaptivePINNTrainer",
    "compute_component_gradients",
    "compute_gradient_norms",
    "WeightEvolutionVisualizer",
    "LossBalancingAnalyzer",
    "MultiObjectiveOptimizer",
]

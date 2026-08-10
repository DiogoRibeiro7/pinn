"""Neural network model architectures for PINN."""

from .mlp import MLP
from .advanced import (
    AttentionPINN,
    EnsemblePINN,
    FourierFeaturePINN,
    MultiScalePINN,
    SineLayer,
    SirenPINN,
)
from .multiscale import (
    FourierMultiScalePINN,
    HierarchicalPINN,
    MultiScaleTrainer,
    MultiScaleTrainingConfig,
    MultiScaleTrainingState,
    WaveletPINN,
    multiscale_loss,
    scale_adaptive_sampling,
)

__all__ = [
    "MLP",
    "MultiScalePINN",
    "FourierMultiScalePINN",
    "HierarchicalPINN",
    "WaveletPINN",
    "MultiScaleTrainer",
    "MultiScaleTrainingConfig",
    "MultiScaleTrainingState",
    "multiscale_loss",
    "scale_adaptive_sampling",
    "FourierFeaturePINN",
    "SineLayer",
    "SirenPINN",
    "AttentionPINN",
    "EnsemblePINN",
]

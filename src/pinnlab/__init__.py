"""Core package for Physics-Informed Neural Networks."""

from importlib.metadata import PackageNotFoundError, version

from . import (
    config,
    deployment,
    models,
    optimization,
    solvers,
    transfer,
    utils,
)
from .models import (
    MLP,
    AttentionPINN,
    EnsemblePINN,
    FourierFeaturePINN,
    SineLayer,
    SirenPINN,
    MultiScalePINN,
    FourierMultiScalePINN,
    HierarchicalPINN,
    WaveletPINN,
    MultiScaleTrainer,
    MultiScaleTrainingConfig,
    MultiScaleTrainingState,
    multiscale_loss,
    scale_adaptive_sampling,
)
from .distributed import (
    DistributedTrainer,
    DistributedPINNTrainer,
    DistributedEnvironment,
    GradientCompression,
    HierarchicalAverager,
    AsyncGradientBuffer,
    synchronise_gradients,
    DynamicLoadBalancer,
)
from .solvers.raissi_improved import (
    BurgersConfig,
    ContinuousPINN,
    DiscreteRKPINN,
    TrainConfig,
    demo as burgers_demo,
)
from .solvers.navier_stokes import (
    NSConfig,
    NavierStokesPINN,
    TrainConfig as NSTrainConfig,
)
from .utils import error_handling, logging, metrics, sampling, checkpointing, profiling
from .utils.logging import get_logger, log_performance
from .utils.checkpointing import CheckpointManager
from .utils.profiling import profile_performance, MemoryTracker, PerformanceReport
from .utils.validation import (
    ValidationError,
    validate_arrays,
    validate_config,
    validate_device,
    check_array,
    check_range,
    check_config,
    check_device,
)
from .utils.metrics import compute_error_metrics
from .utils.sampling import create_lhs_sampler

from .sampling import (
    GradientBasedImportanceSampler,
    ActiveLearningStrategy,
    PoolBasedActiveLearning,
    StreamActiveLearning,
    MultiObjectiveActiveLearning,
    UncertaintyAcquisition,
    ExpectedImprovementAcquisition,
    UpperConfidenceBoundAcquisition,
    InformationGainAcquisition,
    QueryByCommitteeAcquisition,
    ActivePINNTrainer,
    ActiveLearningConfig,
)
from .training import (
    AdaptiveLossWeighting,
    AdaptivePINNTrainer,
    AdaptiveWeightingConfig,
    LossBalancingAnalyzer,
    MultiObjectiveOptimizer,
    WeightEvolutionVisualizer,
    adversarial_training_step,
    adaptive_loss_weights,
    architecture_search,
    compute_component_gradients,
    compute_gradient_norms,
    curriculum_train,
    gradient_hyperparam_search,
    meta_learning_step,
    multi_fidelity_train,
    second_order_step,
    OptimizerConfig,
    Trainer,
    TrainerConfig,
    TrainingResult,
    TrainingState,
)
from .geometry import Geometry, Interval, Rectangle
from .constraints import (
    Constraint,
    ConstraintKind,
    DirichletBoundary,
    InitialCondition,
    NeumannBoundary,
    PeriodicBoundary,
)
from .problems import (
    AllenCahnConfig,
    AllenCahnProblem,
    BurgersProblem,
    HeatProblem,
    PDEProblem,
    WaveProblem,
)
from .residuals import AutogradDerivativeBackend, DerivativeBackend, StrongFormResidual
from .scaling import (
    CharacteristicScales,
    ModelInputTransform,
    ModelOutputTransform,
    Nondimensionalizer,
    ScaledModel,
)
from .uncertainty import (
    BayesianPINN,
    deep_ensemble_predict,
    gp_prior_predict,
    mc_dropout_predict,
)
from .deployment import (
    PINNModelServer,
    create_app,
    load_model,
    save_model,
    serve_grpc,
)
from .optimization import (
    AdaptiveCache,
    CacheConfig,
    CacheEntry,
    CachedComputation,
    CachedSampler,
    PrecomputationManager,
    TrainingCache,
    cached_computation,
)
from .transfer import (
    PretrainingTask,
    PretrainingConfig,
    PretrainingManager,
    build_simplified_pretraining_task,
    progressive_complexity_schedule,
    create_cross_pde_pretraining_tasks,
    create_linear_reference_task,
    FineTuningConfig,
    TransferLearningPINN,
    ContinualLearningBuffer,
    KnowledgeDistillationTrainer,
    DistillationConfig,
    DomainAdaptationPINN,
    DomainAdaptationConfig,
)

try:
    __version__ = version("pinnlab")
except PackageNotFoundError:  # pragma: no cover - during local development
    __version__ = "0.5.1"

# Backward compatibility aliases
from .solvers import navier_stokes as pinn_navier_stokes
from .solvers import raissi as pinn_raissi
from .solvers import raissi_generic as pinn_raissi_generic
from .solvers import raissi_improved as pinn_raissi_improved

__all__ = [
    "config",
    "deployment",
    "models",
    "solvers",
    "optimization",
    "transfer",
    "utils",
    "visualization",
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
    "DistributedTrainer",
    "DistributedPINNTrainer",
    "DistributedEnvironment",
    "ContinuousPINN",
    "DiscreteRKPINN",
    "NavierStokesPINN",
    "BurgersConfig",
    "TrainConfig",
    "NSConfig",
    "NSTrainConfig",
    "sampling",
    "metrics",
    "error_handling",
    "logging",
    "checkpointing",
    "profiling",
    "ValidationError",
    "validate_arrays",
    "validate_config",
    "validate_device",
    "check_array",
    "check_range",
    "check_config",
    "check_device",
    "compute_error_metrics",
    "create_lhs_sampler",
    "PINNVisualizer",
    "TrainingDashboard",
    "export_animation",
    "quick_plot_1d",
    "quick_plot_loss",
    "GradientBasedImportanceSampler",
    "ActiveLearningStrategy",
    "PoolBasedActiveLearning",
    "StreamActiveLearning",
    "MultiObjectiveActiveLearning",
    "UncertaintyAcquisition",
    "ExpectedImprovementAcquisition",
    "UpperConfidenceBoundAcquisition",
    "InformationGainAcquisition",
    "QueryByCommitteeAcquisition",
    "ActivePINNTrainer",
    "ActiveLearningConfig",
    "GradientCompression",
    "HierarchicalAverager",
    "AsyncGradientBuffer",
    "synchronise_gradients",
    "DynamicLoadBalancer",
    "burgers_demo",
    "curriculum_train",
    "adversarial_training_step",
    "meta_learning_step",
    "multi_fidelity_train",
    "adaptive_loss_weights",
    "second_order_step",
    "gradient_hyperparam_search",
    "architecture_search",
    "AdaptiveLossWeighting",
    "AdaptiveWeightingConfig",
    "AdaptivePINNTrainer",
    "compute_component_gradients",
    "compute_gradient_norms",
    "WeightEvolutionVisualizer",
    "LossBalancingAnalyzer",
    "MultiObjectiveOptimizer",
    "OptimizerConfig",
    "Trainer",
    "TrainerConfig",
    "TrainingResult",
    "TrainingState",
    "Geometry",
    "Interval",
    "Rectangle",
    "Constraint",
    "ConstraintKind",
    "DirichletBoundary",
    "InitialCondition",
    "NeumannBoundary",
    "PeriodicBoundary",
    "PDEProblem",
    "AllenCahnConfig",
    "AllenCahnProblem",
    "BurgersProblem",
    "HeatProblem",
    "WaveProblem",
    "AutogradDerivativeBackend",
    "DerivativeBackend",
    "StrongFormResidual",
    "CharacteristicScales",
    "ModelInputTransform",
    "ModelOutputTransform",
    "Nondimensionalizer",
    "ScaledModel",
    "mc_dropout_predict",
    "deep_ensemble_predict",
    "gp_prior_predict",
    "BayesianPINN",
    "PINNModelServer",
    "create_app",
    "load_model",
    "save_model",
    "serve_grpc",
    "__version__",
    "pinn_raissi",
    "pinn_raissi_improved",
    "pinn_raissi_generic",
    "pinn_navier_stokes",
    "get_logger",
    "log_performance",
    "profile_performance",
    "MemoryTracker",
    "PerformanceReport",
    "CheckpointManager",
    "AdaptiveCache",
    "CacheConfig",
    "CacheEntry",
    "CachedSampler",
    "TrainingCache",
    "PrecomputationManager",
    "cached_computation",
    "CachedComputation",
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


# Plotting names resolve on first access rather than at import. Importing them
# eagerly pulled in matplotlib.pyplot, about 1.1s of a 5.5s `import pinnlab`,
# and forced matplotlib to be a required dependency rather than part of the
# `viz` extra. They used to be set to None when the import failed, which turned
# a missing dependency into a TypeError somewhere further along; now the
# AttributeError names what to install.
_LAZY_ATTRIBUTES = {
    "PINNVisualizer": ".utils.visualization",
    "quick_plot_1d": ".utils.visualization",
    "quick_plot_loss": ".utils.visualization",
    "TrainingDashboard": ".visualization",
    "export_animation": ".visualization",
    "visualization": ".visualization",
}


def __getattr__(name: str):
    """Resolve deferred plotting names on first access (PEP 562).

    Raises:
        AttributeError: If the name is unknown, or if it needs an optional
            dependency that is not installed. The message names the extra to
            install rather than leaving a ``None`` to fail later.
    """
    target = _LAZY_ATTRIBUTES.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    try:
        module = importlib.import_module(target, __name__)
    except ImportError as exc:  # pragma: no cover - needs matplotlib absent
        raise AttributeError(
            f"{name} needs the optional plotting dependencies: "
            f"pip install 'pinnlab[viz]' ({exc})"
        ) from exc
    value = module if name == "visualization" else getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list:
    """List deferred names alongside the eagerly imported ones."""
    return sorted(set(globals()) | set(_LAZY_ATTRIBUTES))

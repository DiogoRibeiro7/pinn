"""
Configuration management system for PINN repository.

This module provides a centralized configuration system to manage:
- Training parameters
- Network architectures
- Visualization settings
- Sampling strategies
- File paths and logging

Features:
- Type-safe configuration classes
- Environment-specific configs (dev/prod/test)
- Configuration validation
- YAML/JSON serialization support
- Dynamic configuration updates
- Configuration inheritance and composition
"""

from __future__ import annotations

import os
import json
import yaml
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import warnings
from abc import ABC, abstractmethod

from ..utils.logging import get_logger

logger = get_logger(__name__)


class ConfigError(Exception):
    """Custom exception for configuration errors."""

    pass


class Environment(Enum):
    """Environment types for configuration."""

    DEVELOPMENT = "dev"
    PRODUCTION = "prod"
    TESTING = "test"


# ============================================================
# Base Configuration Classes
# ============================================================


@dataclass(frozen=True)
class BaseConfig(ABC):
    """
    Abstract base class for all configuration objects.

    Provides common functionality for validation, serialization,
    and configuration management.
    """

    def validate(self) -> None:
        """
        Validate configuration parameters.

        Raises:
            ConfigError: If validation fails
        """
        try:
            self._validate_implementation()
            logger.debug(
                "Configuration validated",
                extra={"config": self.__class__.__name__},
            )
        except Exception as exc:
            logger.exception(
                "Configuration validation failed",
                extra={"config": self.__class__.__name__},
            )
            raise exc

    @abstractmethod
    def _validate_implementation(self) -> None:
        """Implementation-specific validation logic."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def to_json(self, filepath: Optional[Union[str, Path]] = None) -> str:
        """
        Convert configuration to JSON.

        Args:
            filepath: Optional path to save JSON file

        Returns:
            JSON string representation
        """
        json_str = json.dumps(self.to_dict(), indent=2, default=str)

        if filepath:
            Path(filepath).write_text(json_str)

        return json_str

    def to_yaml(self, filepath: Optional[Union[str, Path]] = None) -> str:
        """
        Convert configuration to YAML.

        Args:
            filepath: Optional path to save YAML file

        Returns:
            YAML string representation
        """
        yaml_str = yaml.dump(self.to_dict(), default_flow_style=False, indent=2)

        if filepath:
            Path(filepath).write_text(yaml_str)

        return yaml_str


# ============================================================
# Training Configuration
# ============================================================


@dataclass(frozen=True)
class OptimizerConfig:
    """Configuration for optimizer settings."""

    # Adam optimizer settings
    adam_lr: float = 1e-3
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    adam_weight_decay: float = 0.0

    # L-BFGS optimizer settings
    lbfgs_lr: float = 1.0
    lbfgs_max_iter: int = 300
    lbfgs_tolerance_grad: float = 1e-8
    lbfgs_tolerance_change: float = 1e-9
    lbfgs_history_size: int = 50
    lbfgs_line_search_fn: Optional[str] = "strong_wolfe"

    def _validate_implementation(self) -> None:
        """Validate optimizer configuration."""
        if self.adam_lr <= 0:
            raise ConfigError(f"adam_lr must be positive, got {self.adam_lr}")

        if not (0 <= self.adam_betas[0] < 1 and 0 <= self.adam_betas[1] < 1):
            raise ConfigError(f"adam_betas must be in [0, 1), got {self.adam_betas}")

        if self.adam_eps <= 0:
            raise ConfigError(f"adam_eps must be positive, got {self.adam_eps}")

        if self.adam_weight_decay < 0:
            raise ConfigError(
                f"adam_weight_decay must be non-negative, got {self.adam_weight_decay}"
            )

        if self.lbfgs_max_iter < 0:
            raise ConfigError(
                f"lbfgs_max_iter must be non-negative, got {self.lbfgs_max_iter}"
            )

        if self.lbfgs_tolerance_grad <= 0:
            raise ConfigError(
                f"lbfgs_tolerance_grad must be positive, got {self.lbfgs_tolerance_grad}"
            )

        if self.lbfgs_tolerance_change <= 0:
            raise ConfigError(
                f"lbfgs_tolerance_change must be positive, got {self.lbfgs_tolerance_change}"
            )


@dataclass(frozen=True)
class SamplingConfig:
    """Configuration for sampling strategies."""

    # Collocation points
    n_collocation: int = 20_000
    n_initial_condition: int = 200
    n_boundary_condition: int = 200

    # Sampling methods
    collocation_method: str = "latin_hypercube"  # "random", "latin_hypercube", "sobol"
    boundary_method: str = "uniform"  # "uniform", "random", "adaptive"

    # Adaptive sampling
    adaptive_refinement: bool = False
    refinement_iterations: int = 3
    refinement_factor: float = 0.2

    # Random seeds
    seed: int = 123

    def _validate_implementation(self) -> None:
        """Validate sampling configuration."""
        if self.n_collocation <= 0:
            raise ConfigError(
                f"n_collocation must be positive, got {self.n_collocation}"
            )

        if self.n_initial_condition <= 0:
            raise ConfigError(
                f"n_initial_condition must be positive, got {self.n_initial_condition}"
            )

        if self.n_boundary_condition <= 0:
            raise ConfigError(
                f"n_boundary_condition must be positive, got {self.n_boundary_condition}"
            )

        valid_collocation_methods = {"random", "latin_hypercube", "sobol"}
        if self.collocation_method not in valid_collocation_methods:
            raise ConfigError(
                f"collocation_method must be one of {valid_collocation_methods}"
            )

        valid_boundary_methods = {"uniform", "random", "adaptive"}
        if self.boundary_method not in valid_boundary_methods:
            raise ConfigError(
                f"boundary_method must be one of {valid_boundary_methods}"
            )

        if self.refinement_iterations < 0:
            raise ConfigError(
                f"refinement_iterations must be non-negative, got {self.refinement_iterations}"
            )

        if not (0 < self.refinement_factor < 1):
            raise ConfigError(
                f"refinement_factor must be in (0, 1), got {self.refinement_factor}"
            )


@dataclass(frozen=True)
class LossConfig:
    """Configuration for loss function weights and settings."""

    # Loss weights
    weight_initial_condition: float = 1.0
    weight_boundary_condition: float = 1.0
    weight_pde_residual: float = 1.0

    # Adaptive weighting
    adaptive_weighting: bool = False
    weight_adaptation_frequency: int = 1000
    weight_adaptation_rate: float = 0.1

    # Loss clipping and scaling
    gradient_clipping: bool = False
    gradient_clip_value: float = 1.0
    loss_scaling: bool = False

    def _validate_implementation(self) -> None:
        """Validate loss configuration."""
        weights = [
            ("weight_initial_condition", self.weight_initial_condition),
            ("weight_boundary_condition", self.weight_boundary_condition),
            ("weight_pde_residual", self.weight_pde_residual),
        ]

        for name, weight in weights:
            if weight < 0:
                raise ConfigError(f"{name} must be non-negative, got {weight}")

        if self.weight_adaptation_frequency <= 0:
            raise ConfigError(
                f"weight_adaptation_frequency must be positive, got {self.weight_adaptation_frequency}"
            )

        if not (0 < self.weight_adaptation_rate <= 1):
            raise ConfigError(
                f"weight_adaptation_rate must be in (0, 1], got {self.weight_adaptation_rate}"
            )

        if self.gradient_clip_value <= 0:
            raise ConfigError(
                f"gradient_clip_value must be positive, got {self.gradient_clip_value}"
            )


@dataclass(frozen=True)
class TrainingConfig(BaseConfig):
    """Comprehensive training configuration."""

    # Training steps
    adam_steps: int = 15_000
    warmup_steps: int = 1_000
    total_steps: Optional[int] = None

    # Checkpointing
    checkpoint_frequency: int = 1_000
    save_best_model: bool = True
    early_stopping: bool = False
    patience: int = 5_000

    # Logging and monitoring
    log_frequency: int = 100
    validate_frequency: int = 1_000
    plot_frequency: int = 5_000

    # Batch processing
    batch_size: int = 4_096
    gradient_accumulation_steps: int = 1

    # Mixed precision training
    use_mixed_precision: bool = False

    # Component configurations
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    def _validate_implementation(self) -> None:
        """Validate training configuration."""
        if self.adam_steps <= 0:
            raise ConfigError(f"adam_steps must be positive, got {self.adam_steps}")

        if self.warmup_steps < 0:
            raise ConfigError(
                f"warmup_steps must be non-negative, got {self.warmup_steps}"
            )

        if self.total_steps is not None and self.total_steps <= 0:
            raise ConfigError(f"total_steps must be positive, got {self.total_steps}")

        if self.checkpoint_frequency <= 0:
            raise ConfigError(
                f"checkpoint_frequency must be positive, got {self.checkpoint_frequency}"
            )

        if self.patience <= 0:
            raise ConfigError(f"patience must be positive, got {self.patience}")

        if self.log_frequency <= 0:
            raise ConfigError(
                f"log_frequency must be positive, got {self.log_frequency}"
            )

        if self.batch_size <= 0:
            raise ConfigError(f"batch_size must be positive, got {self.batch_size}")

        if self.gradient_accumulation_steps <= 0:
            raise ConfigError(
                f"gradient_accumulation_steps must be positive, got {self.gradient_accumulation_steps}"
            )

        # Validate component configurations
        self.optimizer.validate()
        self.sampling.validate()
        self.loss.validate()


# ============================================================
# Network Architecture Configuration
# ============================================================


@dataclass(frozen=True)
class NetworkConfig(BaseConfig):
    """Configuration for neural network architecture."""

    # Architecture
    input_dim: int = 2
    output_dim: int = 1
    hidden_layers: int = 8
    hidden_width: int = 64

    # Activation function
    activation: str = "tanh"  # "tanh", "relu", "swish", "sine"

    # Initialization
    initialization: str = (
        "xavier_uniform"  # "xavier_uniform", "xavier_normal", "he_uniform", "he_normal"
    )

    # Regularization
    dropout_rate: float = 0.0
    batch_normalization: bool = False
    layer_normalization: bool = False

    # Skip connections
    skip_connections: bool = False
    skip_every_n_layers: int = 2

    # Custom architecture features
    fourier_features: bool = False
    fourier_scale: float = 1.0

    def _validate_implementation(self) -> None:
        """Validate network configuration."""
        if self.input_dim <= 0:
            raise ConfigError(f"input_dim must be positive, got {self.input_dim}")

        if self.output_dim <= 0:
            raise ConfigError(f"output_dim must be positive, got {self.output_dim}")

        if self.hidden_layers < 0:
            raise ConfigError(
                f"hidden_layers must be non-negative, got {self.hidden_layers}"
            )

        if self.hidden_width <= 0:
            raise ConfigError(f"hidden_width must be positive, got {self.hidden_width}")

        valid_activations = {"tanh", "relu", "swish", "sine", "gelu", "elu"}
        if self.activation not in valid_activations:
            raise ConfigError(
                f"activation must be one of {valid_activations}, got {self.activation}"
            )

        valid_initializations = {
            "xavier_uniform",
            "xavier_normal",
            "he_uniform",
            "he_normal",
            "orthogonal",
        }
        if self.initialization not in valid_initializations:
            raise ConfigError(
                f"initialization must be one of {valid_initializations}, got {self.initialization}"
            )

        if not (0 <= self.dropout_rate < 1):
            raise ConfigError(
                f"dropout_rate must be in [0, 1), got {self.dropout_rate}"
            )

        if self.skip_every_n_layers <= 0:
            raise ConfigError(
                f"skip_every_n_layers must be positive, got {self.skip_every_n_layers}"
            )

        if self.fourier_scale <= 0:
            raise ConfigError(
                f"fourier_scale must be positive, got {self.fourier_scale}"
            )


# ============================================================
# Problem-Specific Configurations
# ============================================================


@dataclass(frozen=True)
class DomainConfig(BaseConfig):
    """Configuration for computational domain."""

    # Spatial domain
    x_min: float = -1.0
    x_max: float = 1.0
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    z_min: Optional[float] = None
    z_max: Optional[float] = None

    # Temporal domain
    t_min: float = 0.0
    t_max: float = 1.0

    # Boundary conditions
    boundary_conditions: Dict[str, str] = field(
        default_factory=lambda: {
            "left": "dirichlet",
            "right": "dirichlet",
            "top": "dirichlet",
            "bottom": "dirichlet",
        }
    )

    # Periodic boundaries
    periodic_x: bool = False
    periodic_y: bool = False
    periodic_z: bool = False

    def _validate_implementation(self) -> None:
        """Validate domain configuration."""
        if self.x_max <= self.x_min:
            raise ConfigError(
                f"x_max ({self.x_max}) must be greater than x_min ({self.x_min})"
            )

        if self.y_min is not None and self.y_max is not None:
            if self.y_max <= self.y_min:
                raise ConfigError(
                    f"y_max ({self.y_max}) must be greater than y_min ({self.y_min})"
                )

        if self.z_min is not None and self.z_max is not None:
            if self.z_max <= self.z_min:
                raise ConfigError(
                    f"z_max ({self.z_max}) must be greater than z_min ({self.z_min})"
                )

        if self.t_max <= self.t_min:
            raise ConfigError(
                f"t_max ({self.t_max}) must be greater than t_min ({self.t_min})"
            )

        valid_bc_types = {"dirichlet", "neumann", "robin", "periodic"}
        for boundary, bc_type in self.boundary_conditions.items():
            if bc_type not in valid_bc_types:
                raise ConfigError(
                    f"Invalid boundary condition '{bc_type}' for {boundary}. Must be one of {valid_bc_types}"
                )

    @property
    def spatial_dims(self) -> int:
        """Get number of spatial dimensions."""
        dims = 1  # x is always present
        if self.y_min is not None and self.y_max is not None:
            dims += 1
        if self.z_min is not None and self.z_max is not None:
            dims += 1
        return dims

    @property
    def domain_volume(self) -> float:
        """Calculate domain volume/area/length."""
        volume = self.x_max - self.x_min

        if self.y_min is not None and self.y_max is not None:
            volume *= self.y_max - self.y_min

        if self.z_min is not None and self.z_max is not None:
            volume *= self.z_max - self.z_min

        return volume


@dataclass(frozen=True)
class PDEConfig(BaseConfig):
    """Configuration for PDE-specific parameters."""

    # PDE type
    pde_type: str = "burgers"  # "burgers", "navier_stokes", "allen_cahn", "schrodinger"

    # Physical parameters
    viscosity: float = 0.01
    diffusion_coefficient: float = 1.0
    reaction_coefficient: float = 1.0

    # Burgers equation specific
    burgers_nu: float = 0.01 / 3.14159265359

    # Navier-Stokes specific
    reynolds_number: Optional[float] = None

    # Allen-Cahn specific
    interface_width: float = 0.1

    # Schrödinger specific
    nonlinearity_strength: float = 1.0

    def _validate_implementation(self) -> None:
        """Validate PDE configuration."""
        valid_pde_types = {
            "burgers",
            "navier_stokes",
            "allen_cahn",
            "schrodinger",
            "heat",
            "wave",
            "poisson",
        }
        if self.pde_type not in valid_pde_types:
            raise ConfigError(
                f"pde_type must be one of {valid_pde_types}, got {self.pde_type}"
            )

        if self.viscosity < 0:
            raise ConfigError(f"viscosity must be non-negative, got {self.viscosity}")

        if self.diffusion_coefficient < 0:
            raise ConfigError(
                f"diffusion_coefficient must be non-negative, got {self.diffusion_coefficient}"
            )

        if self.burgers_nu <= 0:
            raise ConfigError(f"burgers_nu must be positive, got {self.burgers_nu}")

        if self.reynolds_number is not None and self.reynolds_number <= 0:
            raise ConfigError(
                f"reynolds_number must be positive, got {self.reynolds_number}"
            )

        if self.interface_width <= 0:
            raise ConfigError(
                f"interface_width must be positive, got {self.interface_width}"
            )


# ============================================================
# Visualization Configuration
# ============================================================


@dataclass(frozen=True)
class VisualizationConfig(BaseConfig):
    """Configuration for visualization settings."""

    # Figure settings
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 100
    save_dpi: int = 300

    # Style settings
    style: str = "seaborn-v0_8"
    color_palette: str = "husl"
    line_width: float = 2.0
    alpha: float = 0.8
    grid_alpha: float = 0.3
    marker_size: int = 3

    # Font settings
    title_fontsize: int = 14
    label_fontsize: int = 12
    legend_fontsize: int = 11
    tick_fontsize: int = 10

    # Colormap settings
    default_cmap: str = "RdBu_r"
    vector_cmap: str = "viridis"

    # Animation settings
    animation_interval: int = 100
    animation_fps: int = 10

    # Output settings
    save_format: str = "png"
    save_bbox_inches: str = "tight"
    show_plots: bool = True

    # Smoothing settings
    default_smoothing_window: int = 50

    def _validate_implementation(self) -> None:
        """Validate visualization configuration."""
        if len(self.figsize) != 2 or any(dim <= 0 for dim in self.figsize):
            raise ConfigError(
                f"figsize must be a tuple of 2 positive numbers, got {self.figsize}"
            )

        if self.dpi <= 0:
            raise ConfigError(f"dpi must be positive, got {self.dpi}")

        if self.save_dpi <= 0:
            raise ConfigError(f"save_dpi must be positive, got {self.save_dpi}")

        if self.line_width <= 0:
            raise ConfigError(f"line_width must be positive, got {self.line_width}")

        if not (0 <= self.alpha <= 1):
            raise ConfigError(f"alpha must be in [0, 1], got {self.alpha}")

        if not (0 <= self.grid_alpha <= 1):
            raise ConfigError(f"grid_alpha must be in [0, 1], got {self.grid_alpha}")

        font_sizes = [
            ("title_fontsize", self.title_fontsize),
            ("label_fontsize", self.label_fontsize),
            ("legend_fontsize", self.legend_fontsize),
            ("tick_fontsize", self.tick_fontsize),
        ]

        for name, size in font_sizes:
            if size <= 0:
                raise ConfigError(f"{name} must be positive, got {size}")

        if self.animation_interval <= 0:
            raise ConfigError(
                f"animation_interval must be positive, got {self.animation_interval}"
            )

        if self.animation_fps <= 0:
            raise ConfigError(
                f"animation_fps must be positive, got {self.animation_fps}"
            )

        if self.default_smoothing_window <= 0:
            raise ConfigError(
                f"default_smoothing_window must be positive, got {self.default_smoothing_window}"
            )

        valid_formats = {"png", "pdf", "eps", "svg", "jpg", "jpeg"}
        if self.save_format not in valid_formats:
            raise ConfigError(
                f"save_format must be one of {valid_formats}, got {self.save_format}"
            )


# ============================================================
# Logging and Monitoring Configuration
# ============================================================


@dataclass(frozen=True)
class LoggingConfig(BaseConfig):
    """Configuration for logging and monitoring."""

    # Basic logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # File settings
    log_dir: str = "logs"
    log_filename: str = "pinn_training.log"
    max_log_size: int = 10 * 1024 * 1024  # 10 MB
    backup_count: int = 5

    # Experiment tracking
    use_wandb: bool = False
    wandb_project: str = "pinn-experiments"
    wandb_entity: Optional[str] = None

    use_tensorboard: bool = False
    tensorboard_log_dir: str = "runs"

    # Metrics tracking
    track_gradients: bool = False
    track_weights: bool = False
    track_memory_usage: bool = False
    track_computational_time: bool = True

    def _validate_implementation(self) -> None:
        """Validate logging configuration."""
        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level not in valid_log_levels:
            raise ConfigError(
                f"log_level must be one of {valid_log_levels}, got {self.log_level}"
            )

        if self.max_log_size <= 0:
            raise ConfigError(f"max_log_size must be positive, got {self.max_log_size}")

        if self.backup_count < 0:
            raise ConfigError(
                f"backup_count must be non-negative, got {self.backup_count}"
            )


# ============================================================
# Master Configuration Class
# ============================================================


@dataclass(frozen=True)
class PINNConfig(BaseConfig):
    """Master configuration class combining all components."""

    # Meta information
    experiment_name: str = "pinn_experiment"
    description: str = "PINN training experiment"
    environment: Environment = Environment.DEVELOPMENT

    # Component configurations
    training: TrainingConfig = field(default_factory=TrainingConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    domain: DomainConfig = field(default_factory=DomainConfig)
    pde: PDEConfig = field(default_factory=PDEConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Paths
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    plot_dir: str = "plots"

    # Hardware settings
    device: str = "auto"  # "auto", "cpu", "cuda", "cuda:0", etc.
    num_workers: int = 4
    pin_memory: bool = True

    def _validate_implementation(self) -> None:
        """Validate master configuration."""
        if not self.experiment_name:
            raise ConfigError("experiment_name cannot be empty")

        if self.num_workers < 0:
            raise ConfigError(
                f"num_workers must be non-negative, got {self.num_workers}"
            )

        # Validate all component configurations
        self.training.validate()
        self.network.validate()
        self.domain.validate()
        self.pde.validate()
        self.visualization.validate()
        self.logging.validate()

    def create_directories(self) -> None:
        """Create necessary directories for the experiment."""
        directories = [
            self.output_dir,
            self.checkpoint_dir,
            self.plot_dir,
            self.logging.log_dir,
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


# ============================================================
# Configuration Factory and Utilities
# ============================================================


class ConfigFactory:
    """Factory class for creating environment-specific configurations."""

    @staticmethod
    def create_development_config() -> PINNConfig:
        """Create configuration optimized for development."""
        return PINNConfig(
            experiment_name="dev_experiment",
            environment=Environment.DEVELOPMENT,
            training=TrainingConfig(
                adam_steps=1_000,  # Shorter training for development
                checkpoint_frequency=100,
                log_frequency=10,
            ),
            visualization=VisualizationConfig(
                show_plots=True,
                save_dpi=100,  # Lower quality for faster development
            ),
            logging=LoggingConfig(log_level="DEBUG", log_to_console=True),
        )

    @staticmethod
    def create_production_config() -> PINNConfig:
        """Create configuration optimized for production."""
        return PINNConfig(
            experiment_name="prod_experiment",
            environment=Environment.PRODUCTION,
            training=TrainingConfig(
                adam_steps=50_000,  # Full training
                checkpoint_frequency=5_000,
                log_frequency=1_000,
                use_mixed_precision=True,
            ),
            visualization=VisualizationConfig(
                show_plots=False,  # Don't display plots in production
                save_dpi=300,  # High quality for production
            ),
            logging=LoggingConfig(
                log_level="INFO", log_to_console=False, use_wandb=True
            ),
        )

    @staticmethod
    def create_testing_config() -> PINNConfig:
        """Create configuration optimized for testing."""
        return PINNConfig(
            experiment_name="test_experiment",
            environment=Environment.TESTING,
            training=TrainingConfig(
                adam_steps=100,  # Very short training for tests
                checkpoint_frequency=50,
                log_frequency=10,
            ),
            network=NetworkConfig(
                hidden_layers=2,  # Smaller network for tests
                hidden_width=16,
            ),
            visualization=VisualizationConfig(
                show_plots=False,
                save_dpi=72,  # Low quality for tests
            ),
            logging=LoggingConfig(log_level="WARNING", log_to_file=False),
        )

    @staticmethod
    def create_burgers_config() -> PINNConfig:
        """Create configuration for Burgers equation."""
        return PINNConfig(
            experiment_name="burgers_experiment",
            pde=PDEConfig(pde_type="burgers", burgers_nu=0.01 / 3.14159265359),
            domain=DomainConfig(x_min=-1.0, x_max=1.0, t_min=0.0, t_max=1.0),
            network=NetworkConfig(input_dim=2, output_dim=1),
        )

    @staticmethod
    def create_navier_stokes_config() -> PINNConfig:
        """Create configuration for Navier-Stokes equations."""
        return PINNConfig(
            experiment_name="navier_stokes_experiment",
            pde=PDEConfig(
                pde_type="navier_stokes", viscosity=0.01, reynolds_number=100.0
            ),
            domain=DomainConfig(
                x_min=0.0,
                x_max=2 * 3.14159265359,
                y_min=0.0,
                y_max=2 * 3.14159265359,
                t_min=0.0,
                t_max=1.0,
                periodic_x=True,
                periodic_y=True,
            ),
            network=NetworkConfig(
                input_dim=3,  # (t, x, y)
                output_dim=3,  # (u, v, p)
            ),
        )


class ConfigLoader:
    """Utility class for loading configurations from files."""

    @staticmethod
    def from_yaml(filepath: Union[str, Path]) -> PINNConfig:
        """
        Load configuration from YAML file.

        Args:
            filepath: Path to YAML configuration file

        Returns:
            PINNConfig object

        Raises:
            ConfigError: If file cannot be loaded or parsed
        """
        try:
            with open(filepath, "r") as f:
                config_dict = yaml.safe_load(f)

            return ConfigLoader._dict_to_config(config_dict)

        except Exception as e:
            raise ConfigError(
                f"Failed to load configuration from {filepath}: {str(e)}"
            ) from e

    @staticmethod
    def from_json(filepath: Union[str, Path]) -> PINNConfig:
        """
        Load configuration from JSON file.

        Args:
            filepath: Path to JSON configuration file

        Returns:
            PINNConfig object

        Raises:
            ConfigError: If file cannot be loaded or parsed
        """
        try:
            with open(filepath, "r") as f:
                config_dict = json.load(f)

            return ConfigLoader._dict_to_config(config_dict)

        except Exception as e:
            raise ConfigError(
                f"Failed to load configuration from {filepath}: {str(e)}"
            ) from e

    @staticmethod
    def _dict_to_config(config_dict: Dict[str, Any]) -> PINNConfig:
        """Convert dictionary to PINNConfig object."""
        # This is a simplified implementation
        # In practice, you'd need more sophisticated parsing
        try:
            return PINNConfig(**config_dict)
        except Exception as e:
            raise ConfigError(
                f"Failed to create configuration from dictionary: {str(e)}"
            ) from e


class ConfigValidator:
    """Utility class for advanced configuration validation."""

    @staticmethod
    def validate_compatibility(config: PINNConfig) -> List[str]:
        """
        Validate compatibility between different configuration components.

        Args:
            config: Configuration to validate

        Returns:
            List of warning messages
        """
        warnings = []

        # Check network input/output dimensions vs PDE requirements
        if config.pde.pde_type == "burgers":
            if config.network.input_dim != 2:
                warnings.append(
                    "Burgers equation typically requires input_dim=2 (t, x)"
                )
            if config.network.output_dim != 1:
                warnings.append("Burgers equation typically requires output_dim=1 (u)")

        elif config.pde.pde_type == "navier_stokes":
            expected_input = 2 + config.domain.spatial_dims  # time + spatial dims
            if config.network.input_dim != expected_input:
                warnings.append(
                    f"Navier-Stokes requires input_dim={expected_input} for {config.domain.spatial_dims}D"
                )

            expected_output = (
                config.domain.spatial_dims + 1
            )  # velocity components + pressure
            if config.network.output_dim != expected_output:
                warnings.append(
                    f"Navier-Stokes requires output_dim={expected_output} for {config.domain.spatial_dims}D"
                )

        # Check sampling points vs domain size
        domain_volume = config.domain.domain_volume
        collocation_density = config.training.sampling.n_collocation / domain_volume

        if collocation_density < 1000:  # Rule of thumb
            warnings.append(
                f"Low collocation point density: {collocation_density:.1f} points per unit volume"
            )

        # Check optimization settings
        if config.training.adam_steps > 100_000:
            warnings.append(
                "Very long training detected - consider using learning rate scheduling"
            )

        if (
            config.training.optimizer.lbfgs_max_iter > 0
            and config.training.use_mixed_precision
        ):
            warnings.append("L-BFGS may not work well with mixed precision training")

        return warnings

    @staticmethod
    def suggest_improvements(config: PINNConfig) -> List[str]:
        """
        Suggest configuration improvements based on best practices.

        Args:
            config: Configuration to analyze

        Returns:
            List of improvement suggestions
        """
        suggestions = []

        # Network architecture suggestions
        if config.network.hidden_layers < 4:
            suggestions.append(
                "Consider using at least 4-8 hidden layers for better approximation"
            )

        if config.network.hidden_width < 32:
            suggestions.append("Consider using at least 32-64 neurons per layer")

        # Training suggestions
        if not config.training.early_stopping:
            suggestions.append(
                "Consider enabling early stopping to prevent overfitting"
            )

        if config.training.sampling.adaptive_refinement is False:
            suggestions.append(
                "Consider using adaptive sampling for better accuracy in critical regions"
            )

        # Monitoring suggestions
        if not config.logging.use_wandb and not config.logging.use_tensorboard:
            suggestions.append(
                "Consider using experiment tracking (Weights & Biases or TensorBoard)"
            )

        return suggestions


# ============================================================
# Example Usage and Testing
# ============================================================


def demonstrate_config_system():
    """Demonstrate the configuration system capabilities."""

    print("=== PINN Configuration System Demo ===\n")

    # Create different environment configurations
    print("1. Creating environment-specific configurations...")

    dev_config = ConfigFactory.create_development_config()
    prod_config = ConfigFactory.create_production_config()
    test_config = ConfigFactory.create_testing_config()

    print(f"Development config - Training steps: {dev_config.training.adam_steps}")
    print(f"Production config - Training steps: {prod_config.training.adam_steps}")
    print(f"Testing config - Training steps: {test_config.training.adam_steps}")

    # Create problem-specific configurations
    print("\n2. Creating problem-specific configurations...")

    burgers_config = ConfigFactory.create_burgers_config()
    ns_config = ConfigFactory.create_navier_stokes_config()

    print(f"Burgers config - PDE type: {burgers_config.pde.pde_type}")
    print(f"Navier-Stokes config - PDE type: {ns_config.pde.pde_type}")

    # Demonstrate validation
    print("\n3. Testing configuration validation...")

    try:
        # This should work
        valid_config = PINNConfig(
            training=TrainingConfig(adam_steps=1000),
            network=NetworkConfig(hidden_layers=5),
        )
        valid_config.validate()
        print("✓ Valid configuration passed validation")

    except ConfigError as e:
        print(f"✗ Configuration validation failed: {e}")

    # Test invalid configuration
    try:
        invalid_config = PINNConfig(
            training=TrainingConfig(adam_steps=-1000)  # Invalid negative steps
        )
        invalid_config.validate()
        print("✗ Invalid configuration should have failed!")

    except ConfigError as e:
        print(f"✓ Invalid configuration correctly rejected: {e}")

    # Demonstrate serialization
    print("\n4. Testing configuration serialization...")

    config = ConfigFactory.create_burgers_config()

    # Convert to dictionary
    config_dict = config.to_dict()
    print(f"✓ Converted to dictionary with {len(config_dict)} top-level keys")

    # Convert to YAML
    yaml_str = config.to_yaml()
    print(f"✓ Converted to YAML ({len(yaml_str)} characters)")

    # Convert to JSON
    json_str = config.to_json()
    print(f"✓ Converted to JSON ({len(json_str)} characters)")

    # Demonstrate compatibility checking
    print("\n5. Testing compatibility validation...")

    warnings = ConfigValidator.validate_compatibility(config)
    if warnings:
        print("⚠ Compatibility warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("✓ No compatibility issues found")

    # Demonstrate improvement suggestions
    suggestions = ConfigValidator.suggest_improvements(config)
    if suggestions:
        print("💡 Improvement suggestions:")
        for suggestion in suggestions:
            print(f"  - {suggestion}")
    else:
        print("✓ Configuration follows best practices")

    print("\n=== Configuration System Demo Complete ===")


if __name__ == "__main__":
    demonstrate_config_system()

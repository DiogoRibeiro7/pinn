"""
Physics-Informed Neural Networks (PINNs) for Burgers Equation.

This module implements both continuous-time and discrete-time (Runge-Kutta) PINNs
for solving the 1D viscous Burgers equation:
    u_t + u*u_x - ν*u_xx = 0

The implementation includes:
- Continuous-time PINN with collocation method
- Discrete-time PINN with Runge-Kutta integration
- Comprehensive training utilities with loss tracking
- Support for Dirichlet boundary conditions

Example:
    >>> from pinn_raissi_improved import ContinuousPINN, demo
    >>> demo()  # Run demonstration with both PINN variants

References:
    Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). 
    Physics-informed neural networks: A deep learning framework for 
    solving forward and inverse problems involving nonlinear partial 
    differential equations. Journal of Computational Physics, 378, 686-707.
"""

from __future__ import annotations

import contextlib
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
    Union,
)

import numpy as np
import torch
from torch import Tensor, nn
import torch.utils.checkpoint as checkpoint
from ..models.mlp import MLP
from tqdm import tqdm

from ..utils.logging import (
    get_logger,
    log_performance,
    log_memory_usage,
    log_gpu_usage,
)
from ..utils.profiling import profile_performance
from ..optimization.caching import CacheConfig, TrainingCache
from ..training.adaptive_weighting import (
    AdaptiveLossWeighting,
    AdaptiveWeightingConfig,
    LossBalancingAnalyzer,
    WeightEvolutionVisualizer,
    compute_gradient_norms,
)
from ..utils.validation import (
    ValidationError,
    check_array,
    check_range,
    validate_config,
)
from ..utils.checkpointing import CheckpointManager
from ..utils.sampling import BaseSampler, Domain, LatinHypercubeSampler
from ..sampling import GradientBasedImportanceSampler

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from ..sampling.active_learning import ActiveLearningStrategy
    from ..training.memory_efficient import MemoryProfiler

logger = get_logger(__name__)


# ============================================================
# Utility Functions
# ============================================================


def set_seed(seed: int = 123) -> None:
    """
    Set random seeds for reproducible results.

    Args:
        seed: Random seed value for all random number generators

    Note:
        Sets seeds for Python's random module, NumPy, and PyTorch (both CPU and GPU).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def latin_hypercube(n: int, d: int, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    """
    Generate Latin Hypercube samples for efficient space-filling sampling.

    Latin Hypercube Sampling (LHS) ensures good coverage of the parameter space
    by dividing each dimension into n equally probable intervals and sampling
    exactly once from each interval.

    Args:
        n: Number of samples to generate
        d: Number of dimensions
        low: Lower bound for all dimensions
        high: Upper bound for all dimensions

    Returns:
        Array of shape (n, d) containing the samples

    Raises:
        ValueError: If n or d is not positive

    Example:
        >>> samples = latin_hypercube(100, 2, 0.0, 1.0)
        >>> print(samples.shape)
        (100, 2)
    """
    if n <= 0 or d <= 0:
        raise ValueError("Number of samples (n) and dimensions (d) must be positive.")

    # Create equally spaced intervals
    cut = np.linspace(0, 1, n + 1)
    u = np.random.rand(n, d)

    # Sample within each interval
    a = cut[:n][:, None]
    b = cut[1 : n + 1][:, None]
    rdpoints = a + (b - a) * u

    # Apply random permutation to each dimension
    H = np.zeros_like(rdpoints)
    for j in range(d):
        order = np.random.permutation(n)
        H[:, j] = rdpoints[order, j]

    return low + (high - low) * H


# ============================================================
# Problem Configuration
# ============================================================


@dataclass(frozen=True)
class BurgersConfig:
    """
    Configuration for the 1D viscous Burgers equation problem.

    Defines the computational domain and physical parameters for the PDE:
        u_t + u*u_x - ν*u_xx = 0

    Attributes:
        tmin: Minimum time value
        tmax: Maximum time value
        xmin: Minimum spatial coordinate
        xmax: Maximum spatial coordinate
        nu: Kinematic viscosity coefficient
    """

    tmin: float = 0.0
    tmax: float = 1.0
    xmin: float = -1.0
    xmax: float = 1.0
    nu: float = 0.01 / math.pi

    def __post_init__(self) -> None:
        if self.tmax <= self.tmin:
            raise ValidationError("tmax must be greater than tmin")
        if self.xmax <= self.xmin:
            raise ValidationError("xmax must be greater than xmin")
        check_range("nu", self.nu, min=0.0)


@dataclass(frozen=True)
class Domain1D:
    """
    Simple 1D domain specification.

    Attributes:
        tmin: Minimum time value
        tmax: Maximum time value
        xmin: Minimum spatial coordinate
        xmax: Maximum spatial coordinate
    """

    tmin: float = 0.0
    tmax: float = 1.0
    xmin: float = -1.0
    xmax: float = 1.0


# ============================================================
# Analytical Solutions and Boundary Conditions
# ============================================================


def u0_true(x: np.ndarray) -> np.ndarray:
    """
    Initial condition for Burgers equation: u(0, x) = -sin(π*x).

    Args:
        x: Spatial coordinates, shape (N,) or (N, 1)

    Returns:
        Initial condition values, same shape as input
    """
    return -np.sin(np.pi * x)


def u_left_bc(t: np.ndarray) -> np.ndarray:
    """
    Left boundary condition: u(t, -1) = 0.

    Args:
        t: Time coordinates, shape (N,) or (N, 1)

    Returns:
        Boundary condition values (zeros), same shape as input
    """
    return np.zeros_like(t)


def u_right_bc(t: np.ndarray) -> np.ndarray:
    """
    Right boundary condition: u(t, +1) = 0.

    Args:
        t: Time coordinates, shape (N,) or (N, 1)

    Returns:
        Boundary condition values (zeros), same shape as input
    """
    return np.zeros_like(t)


# ============================================================
# Neural Network Architecture
# ============================================================
@dataclass
class TrainConfig:
    """Training configuration for PINN optimization.

    Attributes
    ----------
    n_u0, n_bc, n_f:
        Dataset sizes for initial/boundary/collocation points.
    lr, adam_steps, lbfgs_max_iter:
        Optimiser configuration for the two-stage training routine.
    importance_sampler:
        Optional :class:`~pinn.sampling.GradientBasedImportanceSampler` instance
        used to refresh collocation points.
    active_strategy:
        Optional :class:`~pinn.sampling.ActiveLearningStrategy` enabling
        adaptive addition of collocation points during optimisation.
    active_*:
        Parameters controlling the behaviour of the active learning loop such
        as update interval, candidate pool size, and convergence tolerance.
    collocation_batch_size:
        Optional mini-batch size for collocation residual evaluation.  When
        ``None`` the full set of collocation points is used.
    effective_batch_size:
        Target number of collocation points to simulate via gradient
        accumulation.  Defaults to the collocation batch size.
    gradient_accumulation_steps:
        Explicitly control the number of accumulation steps per optimisation
        update.  When ``None`` the value is derived from the batch sizes.
    use_mixed_precision:
        Enable automatic mixed-precision (AMP) training when a compatible
        device is available.
    amp_dtype:
        Preferred floating-point dtype for AMP.  Supported values are
        ``"float16"`` and ``"bfloat16"``.
    gradient_checkpointing:
        Toggle activation checkpointing for the network forward pass to trade
        compute for memory.
    memory_budget_mb:
        Optional soft memory budget used for automatic batch-size selection
        and warning thresholds.
    memory_alert_fraction:
        Fraction of the memory budget at which warnings are emitted.
    memory_profiler:
        Optional :class:`~pinn.training.memory_efficient.MemoryProfiler`
        instance for collecting per-step statistics.
    cache_config:
        Optional :class:`~pinn.optimization.caching.CacheConfig` describing
        caching and precomputation preferences for the training run.
    """

    n_u0: int = 100
    n_bc: int = 100
    n_f: int = 10_000
    lr: float = 1e-3
    adam_steps: int = 10_000
    lbfgs_max_iter: int = 0
    seed: int = 123
    print_frequency: int = 1000
    track_losses: bool = True
    importance_sampler: Optional[GradientBasedImportanceSampler] = None
    active_strategy: Optional["ActiveLearningStrategy"] = None
    active_points_per_iteration: int = 0
    active_candidate_pool: int = 0
    active_update_interval: int = 0
    active_candidate_sampler: Optional[BaseSampler] = None
    active_convergence_tol: float = 1e-4
    active_patience: int = 3
    active_max_iterations: int = 0
    active_max_points: Optional[int] = None
    active_initial_points: Optional[Tensor] = None
    active_metric: str = "pde"
    collocation_batch_size: Optional[int] = None
    effective_batch_size: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = None
    use_mixed_precision: bool = False
    amp_dtype: Optional[str] = None
    gradient_checkpointing: bool = False
    memory_budget_mb: Optional[int] = None
    memory_alert_fraction: float = 0.9
    memory_profiler: Optional["MemoryProfiler"] = None
    cache_config: Optional[CacheConfig] = None
    adaptive_weighting: Optional[AdaptiveWeightingConfig] = None

    def __post_init__(self) -> None:
        check_range("n_u0", self.n_u0, min=1)
        check_range("n_bc", self.n_bc, min=1)
        check_range("n_f", self.n_f, min=1)
        check_range("lr", self.lr, min=0.0)
        check_range("adam_steps", self.adam_steps, min=0)
        check_range("lbfgs_max_iter", self.lbfgs_max_iter, min=0)
        check_range("print_frequency", self.print_frequency, min=1)
        check_range("active_max_iterations", self.active_max_iterations, min=0)
        if self.active_convergence_tol < 0.0:
            raise ValidationError("active_convergence_tol must be non-negative")
        if self.active_initial_points is not None:
            check_array(
                "active_initial_points", self.active_initial_points, shape=(-1, 2)
            )
        if self.active_strategy is not None:
            check_range(
                "active_points_per_iteration", self.active_points_per_iteration, min=1
            )
            check_range("active_candidate_pool", self.active_candidate_pool, min=1)
            check_range("active_update_interval", self.active_update_interval, min=1)
            check_range("active_patience", self.active_patience, min=1)
        if self.collocation_batch_size is not None:
            check_range("collocation_batch_size", self.collocation_batch_size, min=1)
        if self.effective_batch_size is not None:
            check_range("effective_batch_size", self.effective_batch_size, min=1)
        if self.gradient_accumulation_steps is not None:
            check_range(
                "gradient_accumulation_steps", self.gradient_accumulation_steps, min=1
            )
        if self.memory_budget_mb is not None:
            check_range("memory_budget_mb", self.memory_budget_mb, min=1)
        if not 0.0 < self.memory_alert_fraction <= 1.0:
            raise ValidationError("memory_alert_fraction must be in (0, 1]")
        if self.amp_dtype is not None and self.amp_dtype not in {"float16", "bfloat16"}:
            raise ValidationError("amp_dtype must be 'float16' or 'bfloat16'")
        if self.cache_config is not None:
            if not isinstance(self.cache_config, CacheConfig):
                raise ValidationError("cache_config must be an instance of CacheConfig")
            try:
                self.cache_config.validate()
            except ValueError as exc:
                raise ValidationError(str(exc)) from exc
        if self.adaptive_weighting is not None:
            if not isinstance(self.adaptive_weighting, AdaptiveWeightingConfig):
                raise ValidationError(
                    "adaptive_weighting must be an instance of AdaptiveWeightingConfig"
                )
            self.adaptive_weighting.validate(3)


# ============================================================
# Continuous-Time PINN Implementation
# ============================================================


class ContinuousPINN:
    """
    Physics-Informed Neural Network for continuous-time PDE solving.

    This class implements the standard PINN approach where the PDE residual
    is enforced at collocation points throughout the domain, while initial
    and boundary conditions are enforced through supervised learning.

    Attributes:
        model: Neural network model
        device: Computation device (CPU or CUDA)
        cfg: Problem configuration
        pde_residual_fn: Function to compute PDE residuals
        loss_history: Dictionary tracking loss components during training
    """

    @validate_config(BurgersConfig, param="cfg_burgers")
    def __init__(
        self,
        model: nn.Module,
        device: Union[str, torch.device],
        pde_residual_fn: Callable[[nn.Module, Tensor, Tensor, float], Tensor],
        cfg_burgers: BurgersConfig,
    ) -> None:
        """
        Initialize the continuous PINN.

        Args:
            model: Neural network for u_theta(t, x)
            device: Computation device ('cpu' or 'cuda')
            pde_residual_fn: Function computing PDE residual f_theta(t, x, nu)
            cfg_burgers: Problem configuration with domain and PDE parameters
        """
        self.model = model.to(device)
        self.device = torch.device(device)
        self.cfg = cfg_burgers
        self.pde_residual_fn = pde_residual_fn
        self.loss_history: Dict[str, List[float]] = defaultdict(list)
        self._cache: Optional[TrainingCache] = None
        self.adaptive_weight_history: List[np.ndarray] = []
        self.last_loss_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    # ------------------------------------------------------------------
    # Helper utilities for memory-aware training
    # ------------------------------------------------------------------

    def _resolve_amp_dtype(self, tcfg: TrainConfig) -> Optional[torch.dtype]:
        if not tcfg.use_mixed_precision:
            return None
        if tcfg.amp_dtype == "bfloat16":
            return getattr(torch, "bfloat16")
        return torch.float16

    def _model_forward(
        self,
        inputs: Tensor,
        tcfg: TrainConfig,
        *,
        requires_grad: bool = False,
    ) -> Tensor:
        """Forward pass through the model with optional checkpointing."""

        cache = self._cache
        if cache is not None and cache.config.store_activations and not requires_grad:
            cached = cache.get_activation(inputs)
            if cached is not None:
                return cached
        if requires_grad:
            inputs = inputs.requires_grad_(True)
        if tcfg.gradient_checkpointing:
            output = checkpoint.checkpoint(lambda inp: self.model(inp), inputs)
        else:
            output = self.model(inputs)
        if cache is not None and cache.config.store_activations and not requires_grad:
            cache.store_activation(inputs, output)
        return output

    def _memory_budget_bytes(self, tcfg: TrainConfig) -> Optional[int]:
        if tcfg.memory_budget_mb is not None:
            return int(tcfg.memory_budget_mb * 1024 * 1024)
        if self.device.type == "cuda" and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(self.device)
            return props.total_memory
        return None

    def _collocation_loss(self, model: nn.Module, tx: Tensor) -> Tensor:
        """Compute mean squared residual at collocation points ``tx``.

        This helper is used by importance samplers to evaluate how important a
        region of the domain is.  ``tx`` is expected to have shape ``(N, 2)`` with
        columns corresponding to time and space coordinates.
        """
        t, x = tx[:, :1], tx[:, 1:2]
        residual = self.pde_residual_fn(model, t, x, self.cfg.nu)
        return torch.mean(residual**2)

    @validate_config(TrainConfig, param="tcfg")
    def _make_training_points(self, tcfg: TrainConfig) -> Dict[str, Tensor]:
        """
        Generate training data points for IC, BC, and collocation.

        Args:
            tcfg: Training configuration

        Returns:
            Dictionary containing training points and target values
        """
        # Initial condition points: t=0, x in [xmin, xmax]
        check_range("n_u0", tcfg.n_u0, min=1)
        x0 = np.random.uniform(
            self.cfg.xmin, self.cfg.xmax, size=(tcfg.n_u0, 1)
        ).astype(np.float32)
        t0 = np.zeros_like(x0, dtype=np.float32)
        u0 = u0_true(x0).astype(np.float32)

        # Left boundary condition: x=xmin, t in [tmin, tmax]
        check_range("n_bc", tcfg.n_bc, min=1)
        tl = np.random.uniform(
            self.cfg.tmin, self.cfg.tmax, size=(tcfg.n_bc, 1)
        ).astype(np.float32)
        xl = np.full_like(tl, self.cfg.xmin, dtype=np.float32)
        ul = u_left_bc(tl).astype(np.float32)

        # Right boundary condition: x=xmax, t in [tmin, tmax]
        tr = np.random.uniform(
            self.cfg.tmin, self.cfg.tmax, size=(tcfg.n_bc, 1)
        ).astype(np.float32)
        xr = np.full_like(tr, self.cfg.xmax, dtype=np.float32)
        ur = u_right_bc(tr).astype(np.float32)

        # Collocation points in (t, x) domain
        check_range("n_f", tcfg.n_f, min=1)
        if tcfg.importance_sampler is not None:
            tx = tcfg.importance_sampler.sample(
                tcfg.n_f, model=self.model, loss_fn=self._collocation_loss
            )
            t_f_tensor = tx[:, [0]].to(self.device)
            x_f_tensor = tx[:, [1]].to(self.device)
        else:
            H = latin_hypercube(tcfg.n_f, 2, 0.0, 1.0)
            t_f = (self.cfg.tmin + (self.cfg.tmax - self.cfg.tmin) * H[:, [0]]).astype(
                np.float32
            )
            x_f = (self.cfg.xmin + (self.cfg.xmax - self.cfg.xmin) * H[:, [1]]).astype(
                np.float32
            )
            t_f_tensor = torch.from_numpy(t_f).to(self.device)
            x_f_tensor = torch.from_numpy(x_f).to(self.device)

        # Convert remaining arrays to tensors
        to_tensor = lambda a: torch.from_numpy(a).to(self.device)

        data = {
            "t0": to_tensor(t0),
            "x0": to_tensor(x0),
            "u0": to_tensor(u0),
            "tl": to_tensor(tl),
            "xl": to_tensor(xl),
            "ul": to_tensor(ul),
            "tr": to_tensor(tr),
            "xr": to_tensor(xr),
            "ur": to_tensor(ur),
            "tf": t_f_tensor,
            "xf": x_f_tensor,
        }
        for key, arr in data.items():
            check_array(key, arr, shape=(-1, 1), strict=False)
        return data

    @profile_performance
    @log_performance
    @validate_config(TrainConfig, param="tcfg")
    def train(
        self,
        tcfg: TrainConfig,
        weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        checkpoint_manager: Optional[CheckpointManager] = None,
        resume: bool = False,
        *,
        callbacks: Optional[Iterable[Callable[[int, Dict[str, float]], None]]] = None,
        distributed: Optional[Any] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the PINN using Adam optimizer and optional L-BFGS polish.

        The training loop is augmented with gradient accumulation, mixed
        precision, and memory monitoring utilities so that large collocation
        sets can be handled efficiently without exhausting device memory.
        """

        set_seed(tcfg.seed)
        callbacks = list(callbacks or [])

        training_cache: Optional[TrainingCache] = None
        cache_context = {
            "solver": "ContinuousPINN",
            "cfg": {
                "tmin": float(self.cfg.tmin),
                "tmax": float(self.cfg.tmax),
                "xmin": float(self.cfg.xmin),
                "xmax": float(self.cfg.xmax),
                "nu": float(self.cfg.nu),
            },
        }
        if tcfg.cache_config is not None and tcfg.cache_config.enabled:
            training_cache = TrainingCache(tcfg.cache_config, rng_seed=tcfg.seed)
            self._cache = training_cache
            data = training_cache.prepare_training_points(
                tcfg,
                lambda: self._make_training_points(tcfg),
                device=self.device,
                context=cache_context,
            )
            training_cache.register_collocation_set(data["tf"].shape[0])
        else:
            self._cache = None
            data = self._make_training_points(tcfg)
        if tcfg.active_initial_points is not None:
            extra = torch.as_tensor(
                tcfg.active_initial_points, dtype=torch.float32, device=self.device
            )
            data["tf"] = torch.cat([data["tf"], extra[:, :1]], dim=0)
            data["xf"] = torch.cat([data["xf"], extra[:, 1:2]], dim=0)
        adaptive_cfg = tcfg.adaptive_weighting
        adaptive_manager: Optional[AdaptiveLossWeighting] = None
        weight_visualizer: Optional[WeightEvolutionVisualizer] = None
        if adaptive_cfg is not None and adaptive_cfg.enabled:
            adaptive_manager = AdaptiveLossWeighting(
                ("ic", "bc", "pde"),
                adaptive_cfg,
                device=self.device,
                initial_weights=weights,
            )
            weights_vector = adaptive_manager.weights.to(self.device)
            if adaptive_cfg.visualize:
                weight_visualizer = WeightEvolutionVisualizer(("ic", "bc", "pde"))
        else:
            weights_vector = torch.tensor(
                weights, dtype=torch.float32, device=self.device
            )

        w_ic, w_bc, w_pde = [float(v) for v in weights_vector]

        # Clear previous loss history
        self.loss_history.clear()

        active_enabled = (
            tcfg.active_strategy is not None
            and tcfg.active_points_per_iteration > 0
            and tcfg.active_candidate_pool > 0
            and tcfg.active_update_interval > 0
        )
        active_sampler: Optional[BaseSampler]
        active_sampler = None
        active_metric_history: List[float] = []
        active_updates = 0
        if active_enabled:
            domain = Domain(
                [
                    (self.cfg.tmin, self.cfg.tmax),
                    (self.cfg.xmin, self.cfg.xmax),
                ]
            )
            active_sampler = tcfg.active_candidate_sampler or LatinHypercubeSampler(
                domain, seed=tcfg.seed
            )
            logger.info(
                "Active learning enabled",
                extra={
                    "points_per_iteration": tcfg.active_points_per_iteration,
                    "candidate_pool": tcfg.active_candidate_pool,
                    "update_interval": tcfg.active_update_interval,
                },
            )

        use_amp = (
            tcfg.use_mixed_precision
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        )
        amp_dtype = self._resolve_amp_dtype(tcfg) if use_amp else None
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

        if use_amp:
            logger.info(
                "Using mixed precision training",
                extra={"dtype": str(amp_dtype)},
            )
        if tcfg.gradient_checkpointing:
            logger.info("Gradient checkpointing enabled for memory savings")

        memory_budget_bytes = self._memory_budget_bytes(tcfg)
        memory_warned = False
        memory_profiler = tcfg.memory_profiler

        def make_amp_context() -> contextlib.AbstractContextManager:
            if not use_amp:
                return contextlib.nullcontext()
            return torch.cuda.amp.autocast(dtype=amp_dtype)

        def collocation_params() -> Tuple[int, int, int]:
            total = data["tf"].shape[0]
            batch_size = tcfg.collocation_batch_size or total
            if tcfg.effective_batch_size is not None:
                batch_size = min(batch_size, tcfg.effective_batch_size)
            batch_size = max(1, min(batch_size, total))
            derived = max(1, math.ceil(total / batch_size))
            accum = tcfg.gradient_accumulation_steps or derived
            if accum < derived:
                accum = derived
            return batch_size, accum, total

        def collocation_iterator(batch_size: int) -> Iterable[Tensor]:
            total_points = data["tf"].shape[0]
            if training_cache is not None and training_cache.config.async_precompute:
                order = training_cache.next_permutation(total_points, self.device)
            else:
                order = torch.randperm(total_points, device=self.device)
            for idx in order.split(batch_size):
                yield idx

        def backward(loss_tensor: Tensor) -> None:
            if use_amp and scaler is not None:
                scaler.scale(loss_tensor).backward()
            else:
                loss_tensor.backward()

        def optimizer_step(optimizer: torch.optim.Optimizer) -> None:
            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

        def loss_fn() -> Tuple[Tensor, Dict[str, float]]:
            """Compute total loss with batching for PDE residuals."""

            batch_size, _, total = collocation_params()
            total_points = float(total)

            with make_amp_context():
                u0_pred = self._model_forward(
                    torch.cat([data["t0"], data["x0"]], dim=1), tcfg
                )
                ic_loss = torch.mean((u0_pred - data["u0"]) ** 2)

                u_left = self._model_forward(
                    torch.cat([data["tl"], data["xl"]], dim=1), tcfg
                )
                u_right = self._model_forward(
                    torch.cat([data["tr"], data["xr"]], dim=1), tcfg
                )
                bc_loss = torch.mean((u_left - data["ul"]) ** 2) + torch.mean(
                    (u_right - data["ur"]) ** 2
                )

            total_loss_tensor = w_ic * ic_loss + w_bc * bc_loss
            pde_running = 0.0

            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                tf_batch = data["tf"][start:end]
                xf_batch = data["xf"][start:end]
                residual = None
                grad_key: Optional[str] = None
                if training_cache is not None and training_cache.config.store_gradients:
                    grad_key = training_cache.coordinates_key(tf_batch, xf_batch)
                    cached_residual = training_cache.get_gradient(grad_key)
                    if cached_residual is not None:
                        residual = cached_residual.to(self.device)
                if residual is None:
                    with make_amp_context():
                        residual = self.pde_residual_fn(
                            lambda tx: self._model_forward(
                                tx, tcfg, requires_grad=True
                            ),
                            tf_batch,
                            xf_batch,
                            self.cfg.nu,
                        )
                    if grad_key is not None:
                        training_cache.store_gradient(grad_key, residual)
                pde_batch = torch.mean(residual**2)
                weight = tf_batch.shape[0] / total_points
                total_loss_tensor = total_loss_tensor + w_pde * pde_batch * weight
                pde_running += pde_batch.item() * weight

            loss_components = {
                "total": total_loss_tensor.detach().item(),
                "ic": ic_loss.item(),
                "bc": bc_loss.item(),
                "pde": pde_running,
            }
            return total_loss_tensor, loss_components

        # Adam optimization phase
        optimizer = torch.optim.Adam(self.model.parameters(), lr=tcfg.lr)
        self.model.train()
        start_step = 1
        if checkpoint_manager and resume:
            try:
                _, optimizer, start_step, _ = checkpoint_manager.load_latest(
                    self.model, optimizer
                )
                start_step += 1
            except FileNotFoundError:
                start_step = 1

        logger.info(
            "Starting Adam optimization",
            extra={"steps": tcfg.adam_steps},
        )

        initial_batch_size, initial_accum, initial_total = collocation_params()
        logger.info(
            "Collocation batching configured",
            extra={
                "batch_size": initial_batch_size,
                "accumulation_steps": initial_accum,
                "collocation_points": initial_total,
            },
        )

        pbar = tqdm(range(start_step, tcfg.adam_steps + 1), desc="Adam Training")
        stopped_early = False

        best_metric = float("inf")
        for step in pbar:
            collocation_batch_size, accumulation_steps, collocation_total = (
                collocation_params()
            )
            optimizer.zero_grad(set_to_none=True)

            step_context = (
                memory_profiler.track("train_step", step=step)
                if memory_profiler is not None
                else contextlib.nullcontext()
            )

            gradient_norms: Optional[Dict[str, Tensor]] = None
            with step_context:
                with make_amp_context():
                    u0_pred = self._model_forward(
                        torch.cat([data["t0"], data["x0"]], dim=1), tcfg
                    )
                    ic_loss = torch.mean((u0_pred - data["u0"]) ** 2)

                    u_left = self._model_forward(
                        torch.cat([data["tl"], data["xl"]], dim=1), tcfg
                    )
                    u_right = self._model_forward(
                        torch.cat([data["tr"], data["xr"]], dim=1), tcfg
                    )
                    bc_loss = torch.mean((u_left - data["ul"]) ** 2) + torch.mean(
                        (u_right - data["ur"]) ** 2
                    )

                collect_gradients = (
                    adaptive_manager is not None
                    and adaptive_manager.requires_gradients
                    and adaptive_manager.should_update(step)
                )
                if collect_gradients:
                    preview_points = 0
                    if adaptive_cfg is not None:
                        preview_points = min(
                            adaptive_cfg.preview_collocation_points,
                            data["tf"].shape[0],
                        )
                    preview_loss = torch.zeros(
                        (), device=self.device, dtype=ic_loss.dtype
                    )
                    if preview_points > 0:
                        indices = torch.randperm(
                            data["tf"].shape[0], device=self.device
                        )[:preview_points]
                        with make_amp_context():
                            preview_residual = self.pde_residual_fn(
                                lambda tx: self._model_forward(
                                    tx, tcfg, requires_grad=True
                                ),
                                data["tf"][indices],
                                data["xf"][indices],
                                self.cfg.nu,
                            )
                        preview_loss = torch.mean(preview_residual**2)
                    gradient_norms = compute_gradient_norms(
                        {"ic": ic_loss, "bc": bc_loss, "pde": preview_loss},
                        self.model.parameters(),
                        retain_graph=True,
                    )

                base_loss = w_ic * ic_loss + w_bc * bc_loss
                base_loss_value = base_loss.detach().item()
                backward(base_loss / accumulation_steps)

                processed_points = 0
                pde_running_value = 0.0

                batch_iter = iter(collocation_iterator(collocation_batch_size))
                for _ in range(accumulation_steps):
                    try:
                        indices = next(batch_iter)
                    except StopIteration:
                        batch_iter = iter(collocation_iterator(collocation_batch_size))
                        indices = next(batch_iter)

                    tf_batch = data["tf"][indices]
                    xf_batch = data["xf"][indices]
                    residual_batch = None
                    grad_key = None
                    if (
                        training_cache is not None
                        and training_cache.config.store_gradients
                    ):
                        grad_key = training_cache.coordinates_key(tf_batch, xf_batch)
                        cached_residual = training_cache.get_gradient(grad_key)
                        if cached_residual is not None:
                            residual_batch = cached_residual.to(self.device)
                    if residual_batch is None:
                        with make_amp_context():
                            residual_batch = self.pde_residual_fn(
                                lambda tx: self._model_forward(
                                    tx, tcfg, requires_grad=True
                                ),
                                tf_batch,
                                xf_batch,
                                self.cfg.nu,
                            )
                        if grad_key is not None:
                            training_cache.store_gradient(grad_key, residual_batch)
                    pde_batch = torch.mean(residual_batch**2)
                    backward((w_pde * pde_batch) / accumulation_steps)

                    batch_size_actual = tf_batch.shape[0]
                    processed_points += batch_size_actual
                    pde_running_value += pde_batch.item() * batch_size_actual

                optimizer_step(optimizer)
                if training_cache is not None:
                    training_cache.advance_model_version()

            if processed_points > 0:
                pde_loss_value = pde_running_value / processed_points
            else:
                pde_loss_value = 0.0

            loss_components = {
                "total": (base_loss_value + w_pde * pde_loss_value),
                "ic": ic_loss.item(),
                "bc": bc_loss.item(),
                "pde": pde_loss_value,
            }

            if adaptive_manager is not None:
                adaptive_losses = {
                    "ic": ic_loss.detach(),
                    "bc": bc_loss.detach(),
                    "pde": torch.tensor(
                        pde_loss_value,
                        device=self.device,
                        dtype=ic_loss.dtype,
                    ),
                    "total": torch.tensor(
                        loss_components["total"],
                        device=self.device,
                        dtype=ic_loss.dtype,
                    ),
                }
                weights_vector = adaptive_manager.update_weights(
                    adaptive_losses,
                    step=step,
                    gradient_norms=gradient_norms,
                ).to(self.device)
                w_ic, w_bc, w_pde = [float(v) for v in weights_vector]

            if training_cache is not None and training_cache.config.store_losses:
                training_cache.record_loss(step, loss_components)

            best_metric = min(
                best_metric, loss_components.get("pde", loss_components["total"])
            )

            # Periodically refresh collocation points using importance sampling
            if tcfg.importance_sampler is not None and (
                step % tcfg.importance_sampler.update_frequency == 0
            ):
                tx = tcfg.importance_sampler.sample(
                    tcfg.n_f, model=self.model, loss_fn=self._collocation_loss
                )
                data["tf"], data["xf"] = tx[:, [0]], tx[:, [1]]
                if training_cache is not None:
                    training_cache.register_collocation_set(data["tf"].shape[0])

            # Track losses
            if tcfg.track_losses:
                for key, value in loss_components.items():
                    self.loss_history[key].append(value)

            for cb in callbacks:
                cb(step, {"loss": loss_components["total"], **loss_components})

            if active_enabled and (
                step % tcfg.active_update_interval == 0
                and (
                    tcfg.active_max_iterations <= 0
                    or active_updates < tcfg.active_max_iterations
                )
            ):
                assert active_sampler is not None
                pool = active_sampler.sample(tcfg.active_candidate_pool)
                candidate_tensor = torch.as_tensor(
                    pool, dtype=torch.float32, device=self.device
                )
                selected = tcfg.active_strategy.select_points(
                    candidate_tensor,
                    self.model,
                    tcfg.active_points_per_iteration,
                    best_value=best_metric,
                    current_step=step,
                    current_loss=loss_components.get("pde", loss_components["total"]),
                )
                if selected.numel() > 0:
                    data["tf"] = torch.cat([data["tf"], selected[:, :1]], dim=0)
                    data["xf"] = torch.cat([data["xf"], selected[:, 1:2]], dim=0)
                    if (
                        tcfg.active_max_points is not None
                        and data["tf"].shape[0] > tcfg.active_max_points
                    ):
                        data["tf"] = data["tf"][-tcfg.active_max_points :]
                        data["xf"] = data["xf"][-tcfg.active_max_points :]
                    if training_cache is not None:
                        training_cache.register_collocation_set(data["tf"].shape[0])
                    metric_name = (
                        tcfg.active_metric
                        if tcfg.active_metric in loss_components
                        else "total"
                    )
                    metric_value = loss_components.get(
                        metric_name, loss_components["total"]
                    )
                    active_metric_history.append(metric_value)
                    active_updates += 1
                    logger.info(
                        "Active learning added points",
                        extra={
                            "step": step,
                            "added": int(selected.shape[0]),
                            "metric": metric_value,
                        },
                    )
                    if (
                        tcfg.active_patience > 0
                        and len(active_metric_history) >= tcfg.active_patience
                    ):
                        window = active_metric_history[-tcfg.active_patience :]
                        if max(window) - min(window) < tcfg.active_convergence_tol:
                            logger.info(
                                "Active learning convergence reached",
                                extra={"step": step, "metric": metric_value},
                            )
                            stopped_early = True
                            break

            if self.device.type == "cuda" and torch.cuda.is_available():
                current = torch.cuda.memory_allocated(self.device)
                if (
                    memory_budget_bytes is not None
                    and current > memory_budget_bytes * tcfg.memory_alert_fraction
                    and not memory_warned
                ):
                    logger.warning(
                        "High memory usage detected",
                        extra={
                            "step": step,
                            "allocated_mb": current / 1_048_576,
                            "budget_mb": memory_budget_bytes / 1_048_576,
                        },
                    )
                    memory_warned = True

            # Print progress
            if step % tcfg.print_frequency == 0 or step == start_step:
                pbar.set_postfix(
                    {
                        "Total": f"{loss_components['total']:.2e}",
                        "PDE": f"{loss_components['pde']:.2e}",
                    }
                )
                logger.info(
                    "Training progress",
                    extra={"step": step, **loss_components},
                )
                log_memory_usage(logger)
                log_gpu_usage(logger)
            if checkpoint_manager:
                checkpoint_manager.maybe_save(
                    self.model, optimizer, step, {"loss": loss_components["total"]}
                )
                if checkpoint_manager.should_stop():
                    checkpoint_manager.restore_best(self.model, optimizer)
                    logger.info("Early stopping triggered", extra={"step": step})
                    stopped_early = True
                    break

        self.last_loss_weights = tuple(float(v) for v in weights_vector)
        if adaptive_manager is not None:
            self.adaptive_weight_history = list(adaptive_manager.weight_history)
            analyzer = LossBalancingAnalyzer(adaptive_manager)
            logger.info("Adaptive weighting summary", extra=analyzer.summary())
            if weight_visualizer is not None:
                payload = weight_visualizer.create_payload(
                    adaptive_manager.weight_history,
                    adaptive_manager.loss_history.get("total"),
                )
                if isinstance(payload, dict):
                    logger.debug(
                        "Adaptive weighting evolution captured",
                        extra={"updates": len(payload["steps"])},
                    )

        # Optional L-BFGS refinement
        if not stopped_early and tcfg.lbfgs_max_iter > 0:
            logger.info("Starting L-BFGS optimization")

            lbfgs_optimizer = torch.optim.LBFGS(
                self.model.parameters(),
                max_iter=tcfg.lbfgs_max_iter,
                tolerance_grad=1e-8,
                tolerance_change=1e-9,
                history_size=50,
                line_search_fn="strong_wolfe",
            )

            def closure() -> Tensor:
                lbfgs_optimizer.zero_grad(set_to_none=True)
                loss, _ = loss_fn()
                loss.backward()
                return loss

            lbfgs_optimizer.step(closure)
            if training_cache is not None:
                training_cache.advance_model_version()
            logger.info("L-BFGS optimization completed")

        if memory_profiler is not None:
            summary = memory_profiler.summary()
            if summary:
                logger.info("Memory profiling summary", extra=summary)
            for recommendation in memory_profiler.recommendations(memory_budget_bytes):
                logger.info(
                    "Memory recommendation",
                    extra={"suggestion": recommendation},
                )

        logger.info(
            "Training finished",
            extra={
                "final_loss": (
                    self.loss_history.get("total", [None])[-1]
                    if self.loss_history.get("total")
                    else None
                )
            },
        )
        if training_cache is not None:
            training_cache.close()
        self._cache = None
        return dict(self.loss_history)

    @torch.no_grad()
    def predict(
        self, t: np.ndarray, x: np.ndarray, batch_size: int = 4096
    ) -> np.ndarray:
        """
        Predict solution values at given (t, x) points.

        Args:
            t: Time coordinates, must have same shape as x
            x: Spatial coordinates, must have same shape as t
            batch_size: Batch size for prediction to manage memory

        Returns:
            Predicted solution values with same shape as input arrays

        Raises:
            ValueError: If t and x don't have the same shape
        """
        if t.shape != x.shape:
            raise ValueError("Time and spatial coordinates must have the same shape.")

        # Flatten inputs for batched processing
        original_shape = t.shape
        flat_t = t.reshape(-1, 1).astype(np.float32)
        flat_x = x.reshape(-1, 1).astype(np.float32)

        self.model.eval()
        predictions = []

        # Process in batches to manage memory
        for i in range(0, flat_t.shape[0], batch_size):
            batch_t = torch.from_numpy(flat_t[i : i + batch_size]).to(self.device)
            batch_x = torch.from_numpy(flat_x[i : i + batch_size]).to(self.device)
            batch_input = torch.cat([batch_t, batch_x], dim=1)

            batch_pred = self.model(batch_input).cpu().numpy()
            predictions.append(batch_pred)

        # Concatenate and reshape to original form
        full_prediction = np.vstack(predictions)
        return full_prediction.reshape(original_shape)

    def get_loss_history(self) -> Dict[str, List[float]]:
        """
        Get the training loss history.

        Returns:
            Dictionary with loss component names as keys and loss values as lists
        """
        return dict(self.loss_history)


# ============================================================
# PDE Residual Functions
# ============================================================


def burgers_residual(model: nn.Module, t: Tensor, x: Tensor, nu: float) -> Tensor:
    """
    Compute the Burgers equation residual using automatic differentiation.

    The Burgers equation is:
        u_t + u*u_x - ν*u_xx = 0

    This function computes: f = u_t + u*u_x - ν*u_xx

    Args:
        model: Neural network model
        t: Time coordinates, shape (N, 1)
        x: Spatial coordinates, shape (N, 1)
        nu: Viscosity parameter

    Returns:
        PDE residual values, shape (N, 1)

    Raises:
        ValueError: If t and x don't have compatible shapes
    """
    if t.shape != x.shape or t.ndim != 2 or t.shape[1] != 1:
        raise ValueError("t and x must be (N,1) tensors with identical shapes.")

    # Enable gradient computation
    t = t.clone().detach().requires_grad_(True)
    x = x.clone().detach().requires_grad_(True)

    # Network prediction
    tx = torch.cat([t, x], dim=1)
    u = model(tx)

    # Compute first derivatives using autograd
    ones = torch.ones_like(u)
    u_t = torch.autograd.grad(
        u, t, grad_outputs=ones, retain_graph=True, create_graph=True
    )[0]
    u_x = torch.autograd.grad(
        u, x, grad_outputs=ones, retain_graph=True, create_graph=True
    )[0]

    # Compute second derivative u_xx
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True
    )[0]

    # Burgers equation residual: u_t + u*u_x - ν*u_xx
    f = u_t + u * u_x - nu * u_xx
    return f


# ============================================================
# Discrete-Time RK-PINN Implementation
# ============================================================


@dataclass(frozen=True)
class RKTableau:
    """
    Runge-Kutta Butcher tableau for time integration schemes.

    Attributes:
        A: Coefficient matrix, shape (s, s) - lower triangular for explicit RK
        b: Weights vector, shape (s,)
        c: Nodes vector, shape (s,)
    """

    A: np.ndarray
    b: np.ndarray
    c: np.ndarray


def rk4_tableau() -> RKTableau:
    """
    Create the classical 4th-order Runge-Kutta tableau.

    Returns:
        RK4 Butcher tableau
    """
    A = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    b = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6], dtype=np.float32)
    c = np.array([0.0, 0.5, 0.5, 1.0], dtype=np.float32)

    return RKTableau(A=A, b=b, c=c)


def burgers_rhs(u: Tensor, u_x: Tensor, u_xx: Tensor, nu: float) -> Tensor:
    """
    Spatial operator for Burgers equation: N[u] = -u*u_x + ν*u_xx.

    The time derivative is: u_t = N[u]

    Args:
        u: Solution values
        u_x: First spatial derivative
        u_xx: Second spatial derivative
        nu: Viscosity parameter

    Returns:
        Right-hand side of the ODE system
    """
    return -u * u_x + nu * u_xx


class DiscreteRKPINN:
    """
    Discrete-time PINN using Runge-Kutta time integration.

    This approach enforces the RK time-stepping relations as part of the
    physics constraints, allowing for large time steps while maintaining
    temporal accuracy.

    Attributes:
        net: Neural network model
        device: Computation device
        cfg: Problem configuration
        tableau: RK integration tableau
    """

    def __init__(
        self,
        stage_net: nn.Module,
        device: Union[str, torch.device],
        cfg: BurgersConfig,
        tableau: RKTableau,
    ) -> None:
        """
        Initialize the discrete RK-PINN.

        Args:
            stage_net: Network U_theta(t, x) for stage predictions
            device: Computation device
            cfg: Problem configuration
            tableau: Runge-Kutta tableau
        """
        self.net = stage_net.to(device)
        self.device = torch.device(device)
        self.cfg = cfg
        self.tableau = tableau

    def _compute_derivatives(self, u: Tensor, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute spatial derivatives u_x and u_xx using autograd.

        Args:
            u: Solution values, shape (N, 1)
            x: Spatial coordinates, shape (N, 1)

        Returns:
            Tuple of (u_x, u_xx) with same shape as u
        """
        x = x.clone().detach().requires_grad_(True)

        u_x = torch.autograd.grad(
            u, x, torch.ones_like(u), retain_graph=True, create_graph=True
        )[0]

        u_xx = torch.autograd.grad(
            u_x, x, torch.ones_like(u_x), retain_graph=True, create_graph=True
        )[0]

        return u_x, u_xx

    def _stage_prediction(self, x: Tensor, t_stage: Tensor) -> Tensor:
        """
        Get network prediction at stage time.

        Args:
            x: Spatial coordinates
            t_stage: Stage time coordinates

        Returns:
            Network prediction U_theta(t_stage, x)
        """
        return self.net(torch.cat([t_stage, x], dim=1))

    @profile_performance
    @log_performance
    def train_one_step(
        self,
        x_colloc: np.ndarray,
        t0: float,
        t1: float,
        u0_fn: Callable[[np.ndarray], np.ndarray],
        lr: float = 1e-3,
        steps: int = 5000,
        seed: int = 123,
        checkpoint_manager: Optional[CheckpointManager] = None,
        resume: bool = False,
        cache_config: Optional[CacheConfig] = None,
    ) -> Dict[str, List[float]]:
        """
        Train RK-PINN for a single time step from t0 to t1.

        Args:
            x_colloc: Spatial collocation points
            t0: Initial time
            t1: Final time
            u0_fn: Function providing initial condition
            lr: Learning rate
            steps: Number of training steps
            seed: Random seed
            cache_config: Optional caching configuration controlling reuse of
                intermediate values and persistent diagnostics

        Returns:
            Dictionary containing training loss history
        """
        set_seed(seed)

        training_cache: Optional[TrainingCache] = None
        if cache_config is not None and cache_config.enabled:
            training_cache = TrainingCache(cache_config, rng_seed=seed)

        # Prepare data
        x_np = x_colloc.astype(np.float32).reshape(-1, 1)
        x_tensor = torch.from_numpy(x_np).to(self.device)

        h = float(t1 - t0)
        if h <= 0:
            raise ValueError("Final time must be greater than initial time.")

        s = len(self.tableau.b)  # Number of stages

        # Stage times: t_i = t0 + c_i * h
        t_stages = (t0 + self.tableau.c * h).astype(np.float32).reshape(1, -1)
        t_stage_tensor = torch.from_numpy(
            np.repeat(t_stages, x_tensor.shape[0], axis=0)
        ).to(self.device)

        # Initial condition
        u0 = torch.from_numpy(u0_fn(x_np)).to(self.device)

        # Training loop
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.net.train()

        loss_history = defaultdict(list)

        logger.info(
            "Training RK step",
            extra={"t0": t0, "t1": t1, "steps": steps},
        )
        start_step = 1
        if checkpoint_manager and resume:
            try:
                _, optimizer, start_step, _ = checkpoint_manager.load_latest(
                    self.net, optimizer
                )
                start_step += 1
            except FileNotFoundError:
                start_step = 1
        pbar = tqdm(range(start_step, steps + 1), desc="RK-PINN Training")

        for step in pbar:
            optimizer.zero_grad(set_to_none=True)

            # Stage predictions
            stage_solutions = []
            stage_derivatives = []

            for i in range(s):
                U_i = self._stage_prediction(x_tensor, t_stage_tensor[:, [i]])
                stage_solutions.append(U_i)

                # Compute spatial derivatives for RHS evaluation
                u_x, u_xx = self._compute_derivatives(U_i, x_tensor)
                k_i = burgers_rhs(U_i, u_x, u_xx, self.cfg.nu)
                stage_derivatives.append(k_i)

            # RK constraints
            # First stage should equal initial condition
            stage_loss = torch.mean((stage_solutions[0] - u0) ** 2)

            # Final solution from RK formula
            u1_pred = u0 + h * sum(
                self.tableau.b[j] * stage_derivatives[j] for j in range(s)
            )

            # Consistency: final stage should match RK prediction
            consistency_loss = torch.mean((stage_solutions[-1] - u1_pred) ** 2)

            # Smoothness regularization between stages
            smoothness_loss = 0.0
            for i in range(1, s):
                smoothness_loss += torch.mean(
                    (stage_solutions[i] - stage_solutions[i - 1]) ** 2
                )

            # Total loss
            total_loss = stage_loss + consistency_loss + 1e-3 * smoothness_loss

            total_loss.backward()
            optimizer.step()

            # Track losses
            loss_components = {
                "total": total_loss.item(),
                "stage": stage_loss.item(),
                "consistency": consistency_loss.item(),
                "smoothness": smoothness_loss.item(),
            }
            if training_cache is not None:
                if training_cache.config.store_losses:
                    training_cache.record_loss(step, loss_components)
                training_cache.advance_model_version()

            for key, value in loss_components.items():
                loss_history[key].append(value)

            if checkpoint_manager:
                checkpoint_manager.maybe_save(
                    self.net, optimizer, step, {"loss": loss_components["total"]}
                )
                if checkpoint_manager.should_stop():
                    checkpoint_manager.restore_best(self.net, optimizer)
                    logger.info("Early stopping triggered", extra={"step": step})
                    break

            if step % 500 == 0 or step == start_step:
                pbar.set_postfix(
                    {
                        "Total": f"{loss_components['total']:.2e}",
                        "Stage": f"{loss_components['stage']:.2e}",
                    }
                )
                logger.info(
                    "RK training progress",
                    extra={"step": step, **loss_components},
                )
                log_memory_usage(logger)
                log_gpu_usage(logger)

        if training_cache is not None:
            training_cache.close()
        return dict(loss_history)

    @torch.no_grad()
    def predict(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the trained network at arbitrary (t, x) points.

        Args:
            t: Time coordinates
            x: Spatial coordinates

        Returns:
            Network predictions
        """
        if t.shape != x.shape:
            raise ValueError("Time and spatial coordinates must have the same shape.")

        flat_t = t.reshape(-1, 1).astype(np.float32)
        flat_x = x.reshape(-1, 1).astype(np.float32)

        self.net.eval()
        predictions = []

        for i in range(0, flat_t.shape[0], 4096):
            batch_t = torch.from_numpy(flat_t[i : i + 4096]).to(self.device)
            batch_x = torch.from_numpy(flat_x[i : i + 4096]).to(self.device)
            batch_pred = self.net(torch.cat([batch_t, batch_x], dim=1))
            predictions.append(batch_pred.cpu().numpy())

        return np.vstack(predictions).reshape(t.shape)


# ============================================================
# Demonstration Functions
# ============================================================


def demo() -> None:
    """
    Demonstrate both continuous and discrete PINN implementations.

    This function trains both types of PINNs on the Burgers equation
    and shows basic usage patterns.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    cfg = BurgersConfig()

    # Continuous-time PINN demonstration
    print("\n" + "=" * 50)
    print("CONTINUOUS-TIME PINN DEMONSTRATION")
    print("=" * 50)

    model_cont = MLP(in_dim=2, hidden_layers=9, width=20, out_dim=1)
    pinn = ContinuousPINN(
        model=model_cont,
        device=device,
        pde_residual_fn=burgers_residual,
        cfg_burgers=cfg,
    )

    tcfg = TrainConfig(
        n_u0=100,
        n_bc=100,
        n_f=10_000,
        lr=1e-3,
        adam_steps=8_000,
        lbfgs_max_iter=200,
        seed=123,
        track_losses=True,
    )

    loss_history = pinn.train(tcfg, weights=(1.0, 1.0, 1.0))

    # Test prediction
    t_test = np.linspace(cfg.tmin, cfg.tmax, 51, dtype=np.float32)
    x_test = np.linspace(cfg.xmin, cfg.xmax, 101, dtype=np.float32)
    TT, XX = np.meshgrid(t_test, x_test, indexing="ij")
    U_cont = pinn.predict(TT, XX)

    print(f"Continuous PINN - Final loss: {loss_history['total'][-1]:.2e}")
    print(f"Solution range: [{U_cont.min():.3f}, {U_cont.max():.3f}]")

    # Discrete-time RK-PINN demonstration
    print("\n" + "=" * 50)
    print("DISCRETE-TIME RK-PINN DEMONSTRATION")
    print("=" * 50)

    stage_net = MLP(in_dim=2, hidden_layers=6, width=32, out_dim=1)
    rk_pinn = DiscreteRKPINN(
        stage_net=stage_net,
        device=device,
        cfg=cfg,
        tableau=rk4_tableau(),
    )

    # Train for a large time step
    x_colloc = np.linspace(cfg.xmin, cfg.xmax, 256, dtype=np.float32)
    t0, t1 = 0.10, 0.90

    rk_loss_history = rk_pinn.train_one_step(
        x_colloc=x_colloc,
        t0=t0,
        t1=t1,
        u0_fn=u0_true,
        lr=1e-3,
        steps=4000,
        seed=123,
    )

    # Test RK prediction
    T1 = np.full_like(x_colloc, t1)
    U_rk = rk_pinn.predict(T1, x_colloc)

    print(f"RK-PINN - Final loss: {rk_loss_history['total'][-1]:.2e}")
    print(f"Solution range: [{U_rk.min():.3f}, {U_rk.max():.3f}]")

    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    demo()

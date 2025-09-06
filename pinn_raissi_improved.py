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

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, List, Optional, Union, Any
from collections import defaultdict

import numpy as np
import torch
from torch import Tensor, nn
from tqdm import tqdm


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


def latin_hypercube(
    n: int, 
    d: int, 
    low: float = 0.0, 
    high: float = 1.0
) -> np.ndarray:
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
    a = cut[:n]
    b = cut[1 : n + 1]
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

class MLP(nn.Module):
    """
    Multi-Layer Perceptron with tanh activations for PINN applications.
    
    This network architecture is commonly used in PINNs due to the smooth
    activation functions that allow for higher-order derivatives via autodiff.
    
    Attributes:
        net: Sequential neural network layers
    """

    def __init__(
        self, 
        in_dim: int = 2, 
        hidden_layers: int = 8, 
        width: int = 64, 
        out_dim: int = 1
    ) -> None:
        """
        Initialize the MLP network.
        
        Args:
            in_dim: Input dimension (typically 2 for (t,x))
            hidden_layers: Number of hidden layers
            width: Number of neurons per hidden layer
            out_dim: Output dimension (typically 1 for scalar u)
            
        Raises:
            ValueError: If any dimension parameter is non-positive
        """
        super().__init__()
        
        if in_dim <= 0 or hidden_layers < 0 or width <= 0 or out_dim <= 0:
            raise ValueError("All layer dimensions must be positive.")

        # Build network layers
        layers: List[nn.Module] = [nn.Linear(in_dim, width), nn.Tanh()]
        
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
            
        layers += [nn.Linear(width, out_dim)]
        
        self.net = nn.Sequential(*layers)
        self.apply(self._xavier_init)

    @staticmethod
    def _xavier_init(m: nn.Module) -> None:
        """
        Initialize network weights using Xavier uniform initialization.
        
        Args:
            m: PyTorch module to initialize
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, tx: Tensor) -> Tensor:
        """
        Forward pass through the network.
        
        Args:
            tx: Input tensor of shape (batch_size, in_dim)
            
        Returns:
            Output tensor of shape (batch_size, out_dim)
        """
        return self.net(tx)


# ============================================================
# Training Configuration
# ============================================================

@dataclass
class TrainConfig:
    """
    Training configuration for PINN optimization.
    
    Attributes:
        n_u0: Number of initial condition points
        n_bc: Number of boundary condition points per boundary
        n_f: Number of collocation points for PDE residual
        lr: Learning rate for Adam optimizer
        adam_steps: Number of Adam optimization steps
        lbfgs_max_iter: Maximum L-BFGS iterations (0 to disable)
        seed: Random seed for reproducibility
        print_frequency: How often to print loss values
        track_losses: Whether to track individual loss components
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

    def _make_training_points(self, tcfg: TrainConfig) -> Dict[str, Tensor]:
        """
        Generate training data points for IC, BC, and collocation.
        
        Args:
            tcfg: Training configuration
            
        Returns:
            Dictionary containing training points and target values
        """
        # Initial condition points: t=0, x in [xmin, xmax]
        x0 = np.random.uniform(
            self.cfg.xmin, self.cfg.xmax, size=(tcfg.n_u0, 1)
        ).astype(np.float32)
        t0 = np.zeros_like(x0, dtype=np.float32)
        u0 = u0_true(x0).astype(np.float32)

        # Left boundary condition: x=xmin, t in [tmin, tmax]
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

        # Collocation points in (t, x) domain using Latin Hypercube Sampling
        H = latin_hypercube(tcfg.n_f, 2, 0.0, 1.0)
        t_f = (self.cfg.tmin + (self.cfg.tmax - self.cfg.tmin) * H[:, [0]]).astype(np.float32)
        x_f = (self.cfg.xmin + (self.cfg.xmax - self.cfg.xmin) * H[:, [1]]).astype(np.float32)

        # Convert to tensors
        to_tensor = lambda a: torch.from_numpy(a).to(self.device)
        
        return {
            "t0": to_tensor(t0), "x0": to_tensor(x0), "u0": to_tensor(u0),
            "tl": to_tensor(tl), "xl": to_tensor(xl), "ul": to_tensor(ul),
            "tr": to_tensor(tr), "xr": to_tensor(xr), "ur": to_tensor(ur),
            "tf": to_tensor(t_f), "xf": to_tensor(x_f),
        }

    def train(
        self, 
        tcfg: TrainConfig, 
        weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> Dict[str, List[float]]:
        """
        Train the PINN using Adam optimizer and optional L-BFGS polish.
        
        Args:
            tcfg: Training configuration
            weights: Loss weights for (IC, BC, PDE) components
            
        Returns:
            Dictionary containing loss history for each component
        """
        set_seed(tcfg.seed)
        data = self._make_training_points(tcfg)
        w_ic, w_bc, w_pde = weights
        
        # Clear previous loss history
        self.loss_history.clear()

        def loss_fn() -> Tuple[Tensor, Dict[str, float]]:
            """Compute total loss and individual components."""
            # Initial condition loss
            u0_pred = self.model(torch.cat([data["t0"], data["x0"]], dim=1))
            ic_loss = torch.mean((u0_pred - data["u0"]) ** 2)

            # Boundary condition loss
            u_left = self.model(torch.cat([data["tl"], data["xl"]], dim=1))
            u_right = self.model(torch.cat([data["tr"], data["xr"]], dim=1))
            bc_loss = torch.mean((u_left - data["ul"]) ** 2) + \
                     torch.mean((u_right - data["ur"]) ** 2)

            # PDE residual loss
            f = self.pde_residual_fn(self.model, data["tf"], data["xf"], self.cfg.nu)
            pde_loss = torch.mean(f ** 2)

            # Total weighted loss
            total_loss = w_ic * ic_loss + w_bc * bc_loss + w_pde * pde_loss
            
            # Loss components for tracking
            loss_components = {
                "total": total_loss.item(),
                "ic": ic_loss.item(),
                "bc": bc_loss.item(), 
                "pde": pde_loss.item()
            }
            
            return total_loss, loss_components

        # Adam optimization phase
        optimizer = torch.optim.Adam(self.model.parameters(), lr=tcfg.lr)
        self.model.train()
        
        print("Starting Adam optimization...")
        pbar = tqdm(range(1, tcfg.adam_steps + 1), desc="Adam Training")
        
        for step in pbar:
            optimizer.zero_grad(set_to_none=True)
            loss, loss_components = loss_fn()
            loss.backward()
            optimizer.step()
            
            # Track losses
            if tcfg.track_losses:
                for key, value in loss_components.items():
                    self.loss_history[key].append(value)
            
            # Print progress
            if step % tcfg.print_frequency == 0 or step == 1:
                pbar.set_postfix({
                    "Total": f"{loss_components['total']:.2e}",
                    "PDE": f"{loss_components['pde']:.2e}"
                })

        # Optional L-BFGS refinement
        if tcfg.lbfgs_max_iter > 0:
            print("Starting L-BFGS optimization...")
            
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
            print("L-BFGS optimization completed.")

        return dict(self.loss_history)

    @torch.no_grad()
    def predict(
        self, 
        t: np.ndarray, 
        x: np.ndarray, 
        batch_size: int = 4096
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
            batch_t = torch.from_numpy(flat_t[i:i+batch_size]).to(self.device)
            batch_x = torch.from_numpy(flat_x[i:i+batch_size]).to(self.device)
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

def burgers_residual(
    model: nn.Module, 
    t: Tensor, 
    x: Tensor, 
    nu: float
) -> Tensor:
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
        u_x, x, grad_outputs=torch.ones_like(u_x), 
        retain_graph=True, create_graph=True
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
    A = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=np.float32)
    
    b = np.array([1/6, 1/3, 1/3, 1/6], dtype=np.float32)
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

    def train_one_step(
        self,
        x_colloc: np.ndarray,
        t0: float,
        t1: float,
        u0_fn: Callable[[np.ndarray], np.ndarray],
        lr: float = 1e-3,
        steps: int = 5000,
        seed: int = 123,
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
            
        Returns:
            Dictionary containing training loss history
        """
        set_seed(seed)
        
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
        
        print(f"Training RK step from t={t0:.3f} to t={t1:.3f}")
        pbar = tqdm(range(1, steps + 1), desc="RK-PINN Training")
        
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
                    (stage_solutions[i] - stage_solutions[i-1]) ** 2
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
                "smoothness": smoothness_loss.item()
            }
            
            for key, value in loss_components.items():
                loss_history[key].append(value)
            
            if step % 500 == 0 or step == 1:
                pbar.set_postfix({
                    "Total": f"{loss_components['total']:.2e}",
                    "Stage": f"{loss_components['stage']:.2e}"
                })
        
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
            batch_t = torch.from_numpy(flat_t[i:i+4096]).to(self.device)
            batch_x = torch.from_numpy(flat_x[i:i+4096]).to(self.device)
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
    print("\n" + "="*50)
    print("CONTINUOUS-TIME PINN DEMONSTRATION")
    print("="*50)
    
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
        track_losses=True
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
    print("\n" + "="*50)
    print("DISCRETE-TIME RK-PINN DEMONSTRATION")
    print("="*50)
    
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

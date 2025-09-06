# API Reference Documentation

Comprehensive API documentation for the Physics-Informed Neural Networks (PINNs) repository.

## Table of Contents

- [Core PINN Implementations](#core-pinn-implementations)
- [Visualization Module](#visualization-module)
- [Sampling Module](#sampling-module)
- [Metrics Module](#metrics-module)
- [Network Architectures](#network-architectures)
- [Configuration Classes](#configuration-classes)
- [Utility Functions](#utility-functions)

--------------------------------------------------------------------------------

## Core PINN Implementations

### `ContinuousPINN` Class

The main class for continuous-time Physics-Informed Neural Networks.

#### Constructor

```python
ContinuousPINN(
    model: nn.Module,
    device: Union[str, torch.device],
    pde_residual_fn: Callable[[nn.Module, Tensor, Tensor, float], Tensor],
    cfg_burgers: BurgersConfig
)
```

**Parameters:**

- `model`: Neural network model (typically `MLP`)
- `device`: Computation device (`'cpu'` or `'cuda'`)
- `pde_residual_fn`: Function computing PDE residuals
- `cfg_burgers`: Problem configuration object

#### Methods

##### `train()`

```python
train(
    tcfg: TrainConfig, 
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Dict[str, List[float]]
```

Train the PINN using Adam optimizer and optional L-BFGS polish.

**Parameters:**

- `tcfg`: Training configuration object
- `weights`: Loss weights for (IC, BC, PDE) components

**Returns:**

- Dictionary containing loss history for each component

**Example:**

```python
pinn = ContinuousPINN(model, device, burgers_residual, cfg)
loss_history = pinn.train(tcfg, weights=(1.0, 1.0, 1.0))
```

##### `predict()`

```python
predict(
    t: np.ndarray, 
    x: np.ndarray, 
    batch_size: int = 4096
) -> np.ndarray
```

Predict solution values at given (t, x) points.

**Parameters:**

- `t`: Time coordinates, must have same shape as x
- `x`: Spatial coordinates, must have same shape as t
- `batch_size`: Batch size for prediction to manage memory

**Returns:**

- Predicted solution values with same shape as input arrays

**Raises:**

- `ValueError`: If t and x don't have the same shape

##### `get_loss_history()`

```python
get_loss_history() -> Dict[str, List[float]]
```

Get the training loss history.

**Returns:**

- Dictionary with loss component names as keys and loss values as lists

### `ContinuousPINNGeneric` Class

Generic PINN framework for arbitrary PDEs.

#### Constructor

```python
ContinuousPINNGeneric(
    model: nn.Module,
    device: Union[str, torch.device],
    domain: Domain1D,
    u_dim: int,
    u0_fn: Callable[[np.ndarray], np.ndarray],
    bc_left_fn: Callable[[np.ndarray], np.ndarray],
    bc_right_fn: Callable[[np.ndarray], np.ndarray],
    residual_fn: Callable[[nn.Module, Tensor, Tensor], Tensor]
)
```

**Parameters:**

- `model`: Neural network model
- `device`: Computation device
- `domain`: Domain specification
- `u_dim`: Solution dimension (1 for scalar, 2 for complex, etc.)
- `u0_fn`: Initial condition function: x → u0
- `bc_left_fn`: Left boundary function: t → u(t, x_min)
- `bc_right_fn`: Right boundary function: t → u(t, x_max)
- `residual_fn`: PDE residual function

#### Methods

##### `train()`

```python
train(
    n_u0: int = 200,
    n_bc: int = 200,
    n_f: int = 20_000,
    lr: float = 1e-3,
    steps: int = 10_000,
    lbfgs_max_iter: int = 0,
    seed: int = 123,
    loss_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> None
```

Train the generic PINN.

**Parameters:**

- `n_u0`: Number of initial condition points
- `n_bc`: Number of boundary condition points
- `n_f`: Number of interior collocation points
- `lr`: Adam learning rate
- `steps`: Number of Adam optimization steps
- `lbfgs_max_iter`: L-BFGS iterations (0 to disable)
- `seed`: Random seed for reproducibility
- `loss_weights`: Weights for (IC, BC, PDE) loss components

### `DiscreteRKPINN` Class

Discrete-time PINN using Runge-Kutta integration.

#### Constructor

```python
DiscreteRKPINN(
    stage_net: nn.Module,
    device: Union[str, torch.device],
    cfg: BurgersConfig,
    tableau: RKTableau
)
```

**Parameters:**

- `stage_net`: Network for stage predictions
- `device`: Computation device
- `cfg`: Problem configuration
- `tableau`: Runge-Kutta tableau (e.g., from `rk4_tableau()`)

#### Methods

##### `train_one_step()`

```python
train_one_step(
    x_colloc: np.ndarray,
    t0: float,
    t1: float,
    u0_fn: Callable[[np.ndarray], np.ndarray],
    lr: float = 1e-3,
    steps: int = 5000,
    seed: int = 123
) -> Dict[str, List[float]]
```

Train RK-PINN for a single time step from t0 to t1.

**Parameters:**

- `x_colloc`: Spatial collocation points
- `t0`: Initial time
- `t1`: Final time
- `u0_fn`: Function providing initial condition
- `lr`: Learning rate
- `steps`: Number of training steps
- `seed`: Random seed

**Returns:**

- Dictionary containing training loss history

### `NavierStokesPINN` Class

PINN for 2D incompressible Navier-Stokes equations.

#### Constructor

```python
NavierStokesPINN(
    model: nn.Module, 
    cfg: NSConfig, 
    device: Union[str, torch.device] = "cpu"
)
```

**Parameters:**

- `model`: Neural network (input: (t,x,y), output: (u,v,p))
- `cfg`: Navier-Stokes configuration
- `device`: Computation device

#### Methods

##### `train()`

```python
train(
    tcfg: TrainConfig,
    w_ic: float = 1.0,
    w_pde: float = 1.0,
    w_per: float = 1.0,
    seed: int = 123
) -> None
```

Train with Adam, then optional LBFGS.

**Parameters:**

- `tcfg`: Training configuration
- `w_ic`: Weight for initial-condition loss
- `w_pde`: Weight for PDE residual loss
- `w_per`: Weight for periodic boundary loss
- `seed`: Random seed

##### `predict()`

```python
predict(
    t: np.ndarray, 
    x: np.ndarray, 
    y: np.ndarray, 
    batch: int = 4096
) -> np.ndarray
```

Predict [u,v,p] on numpy grids.

**Parameters:**

- `t`, `x`, `y`: Coordinate grids (must have same shape)
- `batch`: Batch size for memory management

**Returns:**

- Array of shape (*grid_shape, 3) with [u, v, p] values

--------------------------------------------------------------------------------

## Visualization Module

### `PINNVisualizer` Class

Comprehensive visualization toolkit for PINN results.

#### Constructor

```python
PINNVisualizer(figsize: Tuple[int, int] = (12, 8), dpi: int = 100)
```

**Parameters:**

- `figsize`: Default figure size (width, height) in inches
- `dpi`: Figure resolution in dots per inch

#### Methods

##### `plot_1d_solution()`

```python
plot_1d_solution(
    x: np.ndarray,
    u_pred: np.ndarray,
    u_true: Optional[np.ndarray] = None,
    title: str = "1D Solution",
    xlabel: str = "x",
    ylabel: str = "u(x)",
    save_path: Optional[str] = None,
    show_grid: bool = True
) -> Figure
```

Plot 1D solution field with optional comparison to analytical solution.

**Parameters:**

- `x`: Spatial coordinates, shape (N,)
- `u_pred`: Predicted solution, shape (N,)
- `u_true`: True/analytical solution, shape (N,), optional
- `title`: Plot title
- `xlabel`, `ylabel`: Axis labels
- `save_path`: Path to save figure, optional
- `show_grid`: Whether to show grid

**Returns:**

- matplotlib Figure object

##### `plot_2d_field()`

```python
plot_2d_field(
    x: np.ndarray,
    y: np.ndarray,
    field: np.ndarray,
    title: str = "2D Field",
    xlabel: str = "x",
    ylabel: str = "y",
    colorbar_label: str = "Field Value",
    cmap: str = "RdBu_r",
    save_path: Optional[str] = None,
    levels: int = 20
) -> Figure
```

Plot 2D scalar field using contour plot.

**Parameters:**

- `x`, `y`: Coordinate meshgrids, shape (Nx, Ny)
- `field`: Field values, shape (Nx, Ny)
- `title`: Plot title
- `xlabel`, `ylabel`: Axis labels
- `colorbar_label`: Colorbar label
- `cmap`: Colormap name
- `save_path`: Path to save figure, optional
- `levels`: Number of contour levels

**Returns:**

- matplotlib Figure object

##### `plot_vector_field_2d()`

```python
plot_vector_field_2d(
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    title: str = "2D Vector Field",
    xlabel: str = "x",
    ylabel: str = "y",
    save_path: Optional[str] = None,
    skip: int = 4,
    scale: float = 1.0
) -> Figure
```

Plot 2D vector field (e.g., velocity field for Navier-Stokes).

**Parameters:**

- `x`, `y`: Coordinate meshgrids, shape (Nx, Ny)
- `u`, `v`: Vector field components, shape (Nx, Ny)
- `title`: Plot title
- `xlabel`, `ylabel`: Axis labels
- `save_path`: Path to save figure, optional
- `skip`: Skip every nth point for cleaner visualization
- `scale`: Arrow scaling factor

**Returns:**

- matplotlib Figure object

##### `plot_spacetime_1d()`

```python
plot_spacetime_1d(
    t: np.ndarray,
    x: np.ndarray,
    u: np.ndarray,
    title: str = "Spacetime Evolution",
    xlabel: str = "x",
    ylabel: str = "t",
    colorbar_label: str = "u(t,x)",
    cmap: str = "RdBu_r",
    save_path: Optional[str] = None
) -> Figure
```

Plot spacetime evolution of 1D solution.

**Parameters:**

- `t`, `x`: Coordinate meshgrids, shape (Nt, Nx)
- `u`: Solution values, shape (Nt, Nx)
- `title`: Plot title
- `xlabel`, `ylabel`: Axis labels
- `colorbar_label`: Colorbar label
- `cmap`: Colormap name
- `save_path`: Path to save figure, optional

**Returns:**

- matplotlib Figure object

##### `plot_loss_curves()`

```python
plot_loss_curves(
    loss_history: Dict[str, List[float]],
    title: str = "Training Loss History",
    log_scale: bool = True,
    save_path: Optional[str] = None,
    smoothing_window: int = 50
) -> Figure
```

Plot training loss curves with optional smoothing.

**Parameters:**

- `loss_history`: Dictionary with loss component names as keys and loss values as lists
- `title`: Plot title
- `log_scale`: Whether to use log scale for y-axis
- `save_path`: Path to save figure, optional
- `smoothing_window`: Window size for moving average smoothing

**Returns:**

- matplotlib Figure object

##### `plot_error_analysis()`

```python
plot_error_analysis(
    x: np.ndarray,
    u_pred: np.ndarray,
    u_true: np.ndarray,
    title: str = "Error Analysis",
    save_path: Optional[str] = None
) -> Figure
```

Comprehensive error analysis comparing predicted vs true solutions.

**Parameters:**

- `x`: Spatial coordinates
- `u_pred`: Predicted solution
- `u_true`: True/analytical solution
- `title`: Plot title
- `save_path`: Path to save figure, optional

**Returns:**

- matplotlib Figure object

### Convenience Functions

#### `quick_plot_1d()`

```python
quick_plot_1d(
    x: np.ndarray, 
    u_pred: np.ndarray, 
    u_true: Optional[np.ndarray] = None,
    title: str = "1D Solution", 
    save_path: Optional[str] = None
) -> Figure
```

Quick 1D plotting function.

#### `quick_plot_2d()`

```python
quick_plot_2d(
    x: np.ndarray, 
    y: np.ndarray, 
    field: np.ndarray,
    title: str = "2D Field", 
    save_path: Optional[str] = None
) -> Figure
```

Quick 2D plotting function.

#### `quick_plot_loss()`

```python
quick_plot_loss(
    loss_history: Dict[str, List[float]], 
    title: str = "Loss History", 
    save_path: Optional[str] = None
) -> Figure
```

Quick loss plotting function.

--------------------------------------------------------------------------------

## Sampling Module

### `Domain` Class

Define a computational domain for sampling.

#### Constructor

```python
Domain(
    bounds: List[Tuple[float, float]],
    names: Optional[List[str]] = None
)
```

**Parameters:**

- `bounds`: List of (min, max) tuples for each dimension
- `names`: Optional names for each dimension (e.g., ['t', 'x', 'y'])

#### Properties

- `ndim`: Number of dimensions
- `volume`: Domain volume/area/length

### `LatinHypercubeSampler` Class

Latin Hypercube Sampling for efficient space-filling sampling.

#### Constructor

```python
LatinHypercubeSampler(
    domain: Domain, 
    seed: Optional[int] = None,
    criterion: str = "random"
)
```

**Parameters:**

- `domain`: Computational domain
- `seed`: Random seed
- `criterion`: Sampling criterion ("random", "centered", "maximin", "correlation")

#### Methods

##### `sample()`

```python
sample(n_points: int) -> np.ndarray
```

Generate Latin Hypercube samples.

**Parameters:**

- `n_points`: Number of points to generate

**Returns:**

- LHS samples of shape (n_points, ndim)

### `SobolSampler` Class

Sobol sequence generator for low-discrepancy sampling.

#### Constructor

```python
SobolSampler(
    domain: Domain, 
    seed: Optional[int] = None, 
    scramble: bool = True
)
```

**Parameters:**

- `domain`: Computational domain
- `seed`: Random seed for scrambling
- `scramble`: Whether to scramble the sequence

### `AdaptiveSampler` Class

Adaptive sampling based on PDE residual magnitudes.

#### Constructor

```python
AdaptiveSampler(
    domain: Domain,
    residual_fn: Callable[[np.ndarray], np.ndarray],
    base_sampler: BaseSampler,
    refinement_factor: float = 0.2,
    max_iterations: int = 5
)
```

**Parameters:**

- `domain`: Computational domain
- `residual_fn`: Function computing PDE residuals at points
- `base_sampler`: Base sampling strategy
- `refinement_factor`: Fraction of points to refine each iteration
- `max_iterations`: Maximum refinement iterations

### `BoundarySampler` Class

Specialized sampling for domain boundaries.

#### Constructor

```python
BoundarySampler(domain: Domain, seed: Optional[int] = None)
```

#### Methods

##### `sample_boundary()`

```python
sample_boundary(
    n_points_per_face: int,
    boundary_dim: int,
    boundary_value: float
) -> np.ndarray
```

Sample points on a specific boundary face.

##### `sample_all_boundaries()`

```python
sample_all_boundaries(
    n_points_per_face: int,
    exclude_corners: bool = True
) -> Dict[str, np.ndarray]
```

Sample points on all domain boundaries.

### Convenience Functions

#### `create_lhs_sampler()`

```python
create_lhs_sampler(
    bounds: List[Tuple[float, float]], 
    seed: Optional[int] = None
) -> LatinHypercubeSampler
```

Convenience function to create LHS sampler.

#### `create_boundary_points()`

```python
create_boundary_points(
    bounds: List[Tuple[float, float]],
    n_points_per_face: int,
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]
```

Convenience function to generate boundary points.

--------------------------------------------------------------------------------

## Metrics Module

### `ErrorMetrics` Class

Container for various error metrics.

#### Attributes

- `mae`: Mean Absolute Error
- `mse`: Mean Squared Error
- `rmse`: Root Mean Squared Error
- `max_error`: Maximum absolute error
- `relative_mae`: Mean Absolute Relative Error
- `relative_rmse`: Root Mean Squared Relative Error
- `l2_norm`: L2 norm of error
- `l_inf_norm`: L-infinity norm of error
- `correlation`: Pearson correlation coefficient
- `r2_score`: R-squared coefficient of determination

### `PhysicsMetrics` Class

Container for physics-based validation metrics.

#### Attributes

- `mean_residual`: Mean PDE residual magnitude
- `max_residual`: Maximum PDE residual magnitude
- `residual_std`: Standard deviation of residuals
- `conservation_error`: Error in conservation laws (optional)
- `energy_error`: Error in energy conservation (optional)
- `mass_error`: Error in mass conservation (optional)
- `momentum_error`: Error in momentum conservation (optional)

### Functions

#### `compute_error_metrics()`

```python
compute_error_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    weights: Optional[np.ndarray] = None,
    eps: float = 1e-10
) -> ErrorMetrics
```

Compute comprehensive error metrics between true and predicted values.

#### `compute_residual_metrics()`

```python
compute_residual_metrics(residuals: np.ndarray) -> PhysicsMetrics
```

Compute metrics for PDE residuals.

#### `comprehensive_error_analysis()`

```python
comprehensive_error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    coordinates: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    residuals: Optional[np.ndarray] = None
) -> Dict[str, Any]
```

Perform comprehensive error analysis.

#### `quick_error_summary()`

```python
quick_error_summary(y_true: np.ndarray, y_pred: np.ndarray) -> str
```

Quick error summary for basic validation.

#### `validate_pinn_solution()`

```python
validate_pinn_solution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    residuals: np.ndarray,
    tolerance: Dict[str, float] = None
) -> Tuple[bool, Dict[str, bool], str]
```

Validate PINN solution against tolerance criteria.

--------------------------------------------------------------------------------

## Network Architectures

### `MLP` Class

Multi-Layer Perceptron with tanh activations.

#### Constructor

```python
MLP(
    in_dim: int = 2, 
    hidden_layers: int = 8, 
    width: int = 64, 
    out_dim: int = 1
)
```

**Parameters:**

- `in_dim`: Input dimension (typically 2 for (t,x))
- `hidden_layers`: Number of hidden layers
- `width`: Number of neurons per hidden layer
- `out_dim`: Output dimension (typically 1 for scalar u)

#### Methods

##### `forward()`

```python
forward(tx: Tensor) -> Tensor
```

Forward pass through the network.

**Parameters:**

- `tx`: Input tensor of shape (batch_size, in_dim)

**Returns:**

- Output tensor of shape (batch_size, out_dim)

--------------------------------------------------------------------------------

## Configuration Classes

### `BurgersConfig`

Configuration for the 1D viscous Burgers equation problem.

#### Attributes

```python
@dataclass(frozen=True)
class BurgersConfig:
    tmin: float = 0.0
    tmax: float = 1.0
    xmin: float = -1.0
    xmax: float = 1.0
    nu: float = 0.01 / math.pi  # kinematic viscosity
```

### `NSConfig`

Configuration for 2D incompressible Navier-Stokes equations.

#### Attributes

```python
@dataclass(frozen=True)
class NSConfig:
    tmin: float = 0.0
    tmax: float = 1.0
    xmin: float = 0.0
    xmax: float = 2.0 * math.pi
    ymin: float = 0.0
    ymax: float = 2.0 * math.pi
    nu: float = 0.01  # kinematic viscosity
```

### `TrainConfig`

Training configuration for PINN optimization.

#### Attributes

```python
@dataclass
class TrainConfig:
    n_u0: int = 100            # Initial condition points
    n_bc: int = 100            # Boundary condition points per boundary
    n_f: int = 10_000          # Collocation points for PDE residual
    lr: float = 1e-3           # Adam learning rate
    adam_steps: int = 10_000   # Adam optimization steps
    lbfgs_max_iter: int = 0    # L-BFGS iterations (0 to disable)
    seed: int = 123            # Random seed for reproducibility
    print_frequency: int = 1000 # How often to print loss values
    track_losses: bool = True   # Whether to track individual loss components
```

--------------------------------------------------------------------------------

## Utility Functions

### Seed Management

#### `set_seed()`

```python
set_seed(seed: int = 123) -> None
```

Set random seeds for reproducible results.

### Sampling Utilities

#### `latin_hypercube()`

```python
latin_hypercube(
    n: int, 
    d: int, 
    low: float = 0.0, 
    high: float = 1.0
) -> np.ndarray
```

Generate Latin Hypercube samples for efficient space-filling sampling.

### Analytical Solutions

#### `u0_true()`

```python
u0_true(x: np.ndarray) -> np.ndarray
```

Initial condition for Burgers equation: u(0, x) = -sin(π*x).

#### `u_left_bc()`, `u_right_bc()`

```python
u_left_bc(t: np.ndarray) -> np.ndarray
u_right_bc(t: np.ndarray) -> np.ndarray
```

Boundary conditions: u(t, ±1) = 0.

### Taylor-Green Vortex Solutions

#### `tgv_u()`, `tgv_v()`, `tgv_p()`

```python
tgv_u(t: np.ndarray, x: np.ndarray, y: np.ndarray, nu: float) -> np.ndarray
tgv_v(t: np.ndarray, x: np.ndarray, y: np.ndarray, nu: float) -> np.ndarray
tgv_p(t: np.ndarray, x: np.ndarray, y: np.ndarray, nu: float) -> np.ndarray
```

Analytical solutions for Taylor-Green vortex.

### PDE Residual Functions

#### `burgers_residual()`

```python
burgers_residual(
    model: nn.Module, 
    t: Tensor, 
    x: Tensor, 
    nu: float
) -> Tensor
```

Compute the Burgers equation residual using automatic differentiation.

#### `allen_cahn_residual_factory()`

```python
allen_cahn_residual_factory(nu: float) -> Callable[[nn.Module, Tensor, Tensor], Tensor]
```

Factory function for Allen-Cahn equation residual.

#### `schrodinger_residual()`

```python
schrodinger_residual(model: nn.Module, t: Tensor, x: Tensor) -> Tensor
```

Residual function for nonlinear Schrödinger equation.

--------------------------------------------------------------------------------

## Usage Examples

### Basic PINN Training

```python
from pinn_raissi_improved import ContinuousPINN, MLP, BurgersConfig, TrainConfig
from pinn_raissi_improved import burgers_residual

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = BurgersConfig(nu=0.01/np.pi)
model = MLP(in_dim=2, hidden_layers=8, width=64, out_dim=1)

# Create PINN
pinn = ContinuousPINN(model, device, burgers_residual, cfg)

# Train
tcfg = TrainConfig(n_f=20_000, adam_steps=15_000, lbfgs_max_iter=300)
loss_history = pinn.train(tcfg, weights=(1.0, 1.0, 1.0))

# Predict
t_test = np.linspace(0, 1, 51)
x_test = np.linspace(-1, 1, 101)
TT, XX = np.meshgrid(t_test, x_test, indexing="ij")
U_pred = pinn.predict(TT, XX)
```

### Advanced Visualization

```python
from visualization import PINNVisualizer

viz = PINNVisualizer(figsize=(15, 10))

# Spacetime evolution
fig1 = viz.plot_spacetime_1d(TT, XX, U_pred, title="Burgers Evolution")

# Loss tracking
fig2 = viz.plot_loss_curves(loss_history, title="Training Progress")

# Error analysis (if analytical solution available)
fig3 = viz.plot_error_analysis(x_test, U_pred[-1], U_analytical[-1])
```

### Custom PDE Implementation

```python
def my_custom_residual(model, t, x):
    """Custom PDE: ∂u/∂t + α·u + β·∂²u/∂x² = f(t,x)"""
    t = t.clone().detach().requires_grad_(True)
    x = x.clone().detach().requires_grad_(True)

    u = model(torch.cat([t, x], dim=1))
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]

    alpha, beta = 0.1, 0.01
    source = torch.sin(t) * torch.cos(x)
    residual = u_t + alpha * u + beta * u_xx - source

    return residual

# Use with generic framework
from pinn_raissi_generic import ContinuousPINNGeneric

pinn = ContinuousPINNGeneric(
    model=model, device=device, domain=domain, u_dim=1,
    u0_fn=lambda x: np.sin(np.pi * x),
    bc_left_fn=lambda t: np.zeros_like(t),
    bc_right_fn=lambda t: np.zeros_like(t),
    residual_fn=my_custom_residual
)
```

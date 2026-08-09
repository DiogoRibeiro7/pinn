# Physics-Informed Neural Networks (PINNs) 🧠⚡

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21844101.svg)](https://doi.org/10.5281/zenodo.21844101) [![CI](https://github.com/DiogoRibeiro7/pinn/actions/workflows/ci.yml/badge.svg)](https://github.com/DiogoRibeiro7/pinn/actions/workflows/ci.yml) [![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.13%2B-red)](https://pytorch.org/) [![License](https://img.shields.io/badge/License-MIT-green)](LICENSE) [![Documentation](https://img.shields.io/badge/docs-latest-brightgreen)](docs/)

A research-oriented PyTorch framework for implementing and testing Physics-Informed Neural Networks for partial differential equations (PDEs). This repository provides modular implementations with visualization, error analysis, and several sampling strategies.

## 🌟 Overview

Physics-Informed Neural Networks (PINNs) embed physical laws described by PDEs directly into neural network loss functions. This approach enables solving forward and inverse PDE problems without requiring large datasets, making it particularly powerful for:

- **Scientific Computing**: Solve PDEs where traditional methods struggle
- **Parameter Estimation**: Discover unknown parameters from sparse data
- **Multi-Physics Problems**: Handle coupled systems naturally
- **Real-time Applications**: Fast inference once trained

## ✨ Key Features

### 🔬 **Multiple PDE Solvers**

- **Heat Equation**: 1D diffusion, scored against its closed-form Fourier solution
- **Wave Equation**: 1D standing waves, second order in time, also exactly solvable
- **Burgers Equation**: 1D viscous flow with shock formation, validated against the
  exact Cole–Hopf solution
- **Navier-Stokes**: 2D incompressible flow with periodic boundaries
- **Allen-Cahn**: Phase field modeling and interface dynamics
- **Nonlinear Schrödinger**: Quantum mechanics and nonlinear optics

> The heat, wave and Burgers problems have **exact solutions**, so accuracy is a
> number rather than a picture. `relative_l2_error` reports it, and the test suite
> asserts on it — see [`notebooks/basic/05_heat_equation.ipynb`](notebooks/basic/05_heat_equation.ipynb).

### 🏗️ **Advanced Architecture**

- **Continuous-time PINNs**: Standard collocation approach
- **Discrete-time PINNs**: Runge-Kutta integration schemes
- **Composable Core API**: Problem, geometry, constraint, residual and trainer
  abstractions for equation-agnostic Burgers/heat/wave training
- **SIREN Networks**: Sinusoidal representation models for oscillatory fields
  and coordinate-derivative-heavy PDE residuals
- **Nondimensionalization**: Characteristic physical scales, derivative scale
  factors and FP32/FP64-preserving transforms for the composable trainer
- **Residual-Adaptive Refinement**: Generic trainer support for retaining
  high-residual collocation points, optional diversity-aware selection and
  recorded RAR diagnostics
- **Composable Sampling**: Configurable interior collocation samplers for the
  generic trainer, with uniform and Latin-hypercube geometry sampling policies
  plus adapters for legacy sampler interoperability
- **Benchmark Provenance**: JSON benchmark reports, independent residual
  scoring, and analytic/numerical/observational reference labels
- **RAR Benchmarks**: Scripted uniform-vs-adaptive Burgers, heat and wave
  comparisons using the shared benchmark report schema
- **Generic Framework**: Easy extension to new PDEs
- **GPU Acceleration**: CUDA support with automatic detection
- **Distributed Training**: Data-parallel scaling across multiple GPUs

### 📊 **Professional Visualization**

- **Publication-quality plots** with Matplotlib/Seaborn
- **Spacetime evolution** heatmaps and animations
- **Error analysis** with statistical distributions
- **Loss tracking** with component breakdown
- **Vector field visualization** for fluid dynamics

### 🎯 **Advanced Sampling**

- **Latin Hypercube Sampling**: Efficient space-filling
- **Sobol Sequences**: Low-discrepancy sampling
- **Adaptive Refinement**: Residual-based point addition
- **Boundary Sampling**: Specialized BC enforcement
- **Composite Strategies**: Multi-region sampling

### 📈 **Comprehensive Metrics**

- **Error Analysis**: 10+ accuracy metrics (MAE, RMSE, correlation, etc.)
- **Physics Validation**: PDE residual analysis
- **Statistical Tests**: Distribution comparisons, moment analysis
- **Convergence Analysis**: Richardson extrapolation, rate estimation
- **Benchmark Comparison**: Multi-method validation

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/DiogoRibeiro7/pinn.git
cd pinn

# Development installation
pip install -e .

# Regular installation
pip install .

# With optional dependencies
pip install .[viz,dev,gpu,deploy]
```

### Build Documentation

```bash
pip install .[dev]
sphinx-build -b html docs docs/_build/html
```

### Serving Models

Export a trained model to TorchScript and serve it with FastAPI:

```bash
PINN_MODEL_PATH=path/to/model.pt uvicorn pinn.deployment.server:app
```

Docker and helper scripts for AWS, GCP, and Azure are available under the
`deployment/` directory.

### Basic Usage

#### 🔥 Run Complete Demo with Visualization

```python
from examples.demo_visualization import comprehensive_demo_with_visualization

# Run full demonstration with plots and analysis
comprehensive_demo_with_visualization()
```

#### 📐 Burgers Equation (1D)

```python
from pinn.solvers.raissi_improved import ContinuousPINN, demo

# Quick demo
demo()

# Or customize training
from pinn.solvers.raissi_improved import MLP, BurgersConfig, TrainConfig

model = MLP(in_dim=2, hidden_layers=8, width=64, out_dim=1)
cfg = BurgersConfig(nu=0.01/np.pi)
tcfg = TrainConfig(n_f=20_000, adam_steps=15_000, lbfgs_max_iter=300)

# Train and get loss history
loss_history = pinn.train(tcfg, weights=(1.0, 1.0, 1.0))
```

#### 🌊 Navier-Stokes Equations (2D)

```python
from pinn.solvers.navier_stokes import demo_tgv

# Taylor-Green vortex simulation
demo_tgv()

# Custom visualization
from pinn.utils.visualization import PINNVisualizer
viz = PINNVisualizer()
fig = viz.plot_vector_field_2d(X, Y, U, V, title="Velocity Field")
```

#### 🧪 Generic PDE Framework

```python
from pinn.solvers.raissi_generic import ContinuousPINNGeneric

# Allen-Cahn equation
from pinn.solvers.raissi_generic import demo_allen_cahn
demo_allen_cahn()

# Nonlinear Schrödinger equation
from pinn.solvers.raissi_generic import demo_schrodinger
demo_schrodinger()
```

## 📁 Repository Structure

```
pinn/
├── 📋 README.md                  # This guide
├── 🗺️  ROADMAP.md                 # Project status & planned work
├── 📦 pyproject.toml             # Packaging & dependencies
│
├── 🧠 src/pinn/                  # The library
│   ├── models/                   # MLP, SIREN, Fourier-feature, multi-scale, …
│   ├── solvers/                  # Burgers (continuous + RK), Navier–Stokes, generic
│   ├── benchmarks/               # Report schema and independent evaluators
│   ├── numerics/                 # Classical numerical reference solvers
│   ├── sampling/                 # Latin-Hypercube, importance, active learning
│   ├── training/                 # Adaptive weighting, curriculum, meta-learning, …
│   ├── transfer/                 # Pre-training, fine-tuning, distillation, domain adaptation
│   ├── uncertainty/              # Ensembles, MC-dropout, Bayesian, GP priors
│   ├── distributed/              # Data-parallel trainer, gradient compression, load balancing
│   ├── optimization/             # Caching & computation reuse
│   ├── deployment/               # REST/gRPC model serving, model I/O
│   ├── visualization/            # Plotting toolkit & training dashboard
│   ├── utils/                    # Metrics, checkpointing, profiling, validation
│   └── config/                   # Configuration management
│
├── 🎨 examples/                  # Runnable scripts (basic / advanced / benchmarks)
├── 📓 notebooks/                 # Self-contained, executed Kaggle-style tutorials
├── 🧪 tests/                     # Test suite
└── 📚 docs/                      # Sphinx documentation
```

## 🔬 Implemented PDEs

### 1\. 📊 Burgers Equation

```
∂u/∂t + u·∂u/∂x - ν·∂²u/∂x² = 0
```

- **Domain**: 1D spatial [-1,1], temporal [0,1]
- **BC**: Dirichlet (u = 0 at boundaries)
- **IC**: u(0,x) = -sin(πx)
- **Applications**: Shock wave formation, traffic flow

### 2\. 🌊 Navier-Stokes Equations

```
∂u/∂t + (u·∇)u + ∇p - ν∇²u = 0    (momentum)
∇·u = 0                            (continuity)
```

- **Domain**: 2D spatial [0,2π]×[0,2π], temporal
- **BC**: Periodic boundaries
- **Test Case**: Taylor-Green vortex (analytical solution)
- **Applications**: Fluid dynamics, weather prediction

### 3\. 🔄 Allen-Cahn Equation

```
∂u/∂t - ν·∂²u/∂x² + (u³ - u) = 0
```

- **Domain**: 1D spatial, temporal
- **BC**: Dirichlet (customizable)
- **Applications**: Phase transitions, material interfaces

### 4\. 〰️ Nonlinear Schrödinger Equation

```
i·∂u/∂t + ½·∂²u/∂x² + |u|²u = 0
```

- **Domain**: 1D spatial, temporal
- **Representation**: Complex field as [Re(u), Im(u)]
- **Applications**: Quantum mechanics, nonlinear optics

## 🎨 Visualization Gallery

### Loss Tracking with Component Breakdown

```python
from pinn.utils.visualization import quick_plot_loss

loss_history = {
    "total": [...],    # Total weighted loss
    "pde": [...],      # PDE residual loss
    "ic": [...],       # Initial condition loss
    "bc": [...]        # Boundary condition loss
}

fig = quick_plot_loss(loss_history, "Training Progress")
```

### Spacetime Evolution Analysis

```python
from pinn.utils.visualization import PINNVisualizer

viz = PINNVisualizer()

# Spacetime heatmap
fig = viz.plot_spacetime_1d(t_grid, x_grid, u_solution,
                           title="Solution Evolution")

# Error analysis with statistics
fig = viz.plot_error_analysis(x, u_pred, u_true,
                             title="Comprehensive Error Analysis")
```

### Vector Field Visualization

```python
# 2D vector fields (e.g., velocity)
fig = viz.plot_vector_field_2d(X, Y, U, V,
                              title="Navier-Stokes Velocity Field",
                              skip=3, scale=1.0)
```

## 🎯 Advanced Sampling Strategies

### Latin Hypercube Sampling

```python
from pinn.sampling import create_lhs_sampler

# Efficient space-filling sampling
sampler = create_lhs_sampler(bounds=[(-1,1), (0,1)], seed=123)
points = sampler.sample(10000)
```

### Adaptive Refinement

```python
from pinn.sampling import AdaptiveSampler

# Residual-based adaptive sampling
adaptive_sampler = AdaptiveSampler(domain, residual_fn, base_sampler)
refined_points = adaptive_sampler.sample(5000)
```

### Boundary Condition Sampling

```python
from pinn.sampling import create_boundary_points

# Specialized boundary sampling
boundaries = create_boundary_points(
    bounds=[(-1,1), (0,1)],
    n_points_per_face=200
)
```

## 📊 Comprehensive Error Analysis

### Quick Validation

```python
from pinn.utils.metrics import quick_error_summary, validate_pinn_solution

# Fast error overview
print(quick_error_summary(y_true, y_pred))

# Validation against tolerances
pass_fail, results, summary = validate_pinn_solution(
    y_true, y_pred, residuals,
    tolerance={"rmse": 1e-3, "correlation": 0.98}
)
```

### Detailed Analysis

```python
from pinn.utils.metrics import comprehensive_error_analysis, generate_metrics_report

# Full error analysis
results = comprehensive_error_analysis(
    y_true, y_pred,
    coordinates=coords,
    residuals=residuals
)

# Generate formatted report
report = generate_metrics_report(results, "Burgers Equation Analysis")
print(report)
```

### Convergence Analysis

```python
from pinn.utils.metrics import ConvergenceAnalysis

# Analyze convergence rates
analyzer = ConvergenceAnalysis()
convergence_rates = analyzer.analyze_convergence(
    resolutions=[1000, 2000, 4000, 8000],
    errors={"rmse": [1e-2, 5e-3, 2.5e-3, 1.2e-3]}
)
```

## ⚙️ Custom PDE Implementation

### 1\. Define PDE Residual

```python
def my_pde_residual(model, t, x):
    """Custom PDE: ∂u/∂t + α·u + β·∂²u/∂x² = f(t,x)"""

    # Enable gradients
    t = t.clone().detach().requires_grad_(True)
    x = x.clone().detach().requires_grad_(True)

    # Network prediction
    u = model(torch.cat([t, x], dim=1))

    # Compute derivatives using autograd
    u_t = torch.autograd.grad(u, t, torch.ones_like(u),
                             retain_graph=True, create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u),
                             retain_graph=True, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x),
                              retain_graph=True, create_graph=True)[0]

    # PDE residual
    alpha, beta = 0.1, 0.01
    source_term = torch.sin(t) * torch.cos(x)  # Example source
    residual = u_t + alpha * u + beta * u_xx - source_term

    return residual
```

### 2\. Use Generic Framework

```python
from pinn.solvers.raissi_generic import ContinuousPINNGeneric
from pinn.sampling import Domain

# Setup domain and model
domain = Domain(bounds=[(0, 1), (-1, 1)], names=['t', 'x'])
model = MLP(in_dim=2, hidden_layers=8, width=64, out_dim=1)

# Create PINN
pinn = ContinuousPINNGeneric(
    model=model,
    device="cuda",
    domain=domain,
    u_dim=1,
    u0_fn=lambda x: np.sin(np.pi * x),           # Initial condition
    bc_left_fn=lambda t: np.zeros_like(t),       # Left boundary
    bc_right_fn=lambda t: np.zeros_like(t),      # Right boundary
    residual_fn=my_pde_residual                  # Your PDE residual
)

# Train with comprehensive tracking
loss_history = pinn.train(
    n_u0=200, n_bc=200, n_f=20_000,
    lr=1e-3, steps=15_000, lbfgs_max_iter=300
)
```

## 🏎️ Performance Optimization

### Training Configuration

```python
@dataclass
class OptimalTrainConfig:
    # Sampling points
    n_u0: int = 200           # Initial condition points
    n_bc: int = 200           # Boundary points per face
    n_f: int = 20_000         # PDE collocation points (increase for accuracy)

    # Optimization
    lr: float = 1e-3          # Adam learning rate
    adam_steps: int = 15_000  # Adam iterations (8k-20k typical)
    lbfgs_max_iter: int = 300 # L-BFGS polish (0 to disable)

    # Loss weighting (tune for balance)
    w_ic: float = 1.0         # Initial condition weight
    w_bc: float = 1.0         # Boundary condition weight
    w_pde: float = 1.0        # PDE residual weight

    # Reproducibility
    seed: int = 123
```

### Best Practices

1. **🎯 Sampling Strategy**:

  - Use Latin Hypercube for collocation points
  - More boundary points for complex BCs
  - Adaptive refinement in high-gradient regions

2. **🧠 Network Architecture**:

  - Start with 8 layers × 64 neurons
  - Tanh activation for smooth derivatives
  - Xavier initialization for stability

3. **🏋️ Training Strategy**:

  - Adam (15k steps) → L-BFGS (300 steps)
  - Monitor loss components separately
  - Early stopping if residuals plateau

4. **⚖️ Loss Balancing**:

  - Equal weights initially (1:1:1)
  - Increase PDE weight if residuals are high
  - Increase BC weight if boundaries not satisfied

## 🔧 Theory & Mathematical Background

### PINN Loss Function

```
ℒ_total = λ_IC · ℒ_IC + λ_BC · ℒ_BC + λ_PDE · ℒ_PDE
```

Where:

- **ℒ_IC = MSE(u_θ(t₀,x), u₀(x))**: Initial condition loss (supervised)
- **ℒ_BC = MSE(u_θ(t,x_b), g(t))**: Boundary condition loss (supervised)
- **ℒ_PDE = MSE(f_θ(t,x), 0)**: PDE residual loss (unsupervised physics)

### Automatic Differentiation

PINNs leverage automatic differentiation to compute exact derivatives:

```python
# First derivative: ∂u/∂t
u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                         retain_graph=True, create_graph=True)[0]

# Second derivative: ∂²u/∂x²
u_x = torch.autograd.grad(u, x, torch.ones_like(u),
                         retain_graph=True, create_graph=True)[0]
u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x),
                          retain_graph=True, create_graph=True)[0]
```

### Key Advantages

- **Meshfree**: No spatial/temporal discretization
- **Flexible Geometry**: Irregular domains handled naturally
- **Multi-scale**: Captures multiple time/length scales
- **Inverse Problems**: Parameter estimation from sparse data

## 📚 References & Citations

### Citing This Software

If you use this library in academic work, please cite it. Each release is archived on
Zenodo, and the DOI below is the *concept DOI*: it always resolves to the most recent
version, so it is the right one to cite when you mean the software in general.

**DOI: [10.5281/zenodo.21844101](https://doi.org/10.5281/zenodo.21844101)**

```bibtex
@software{ribeiro_pinn,
  author    = {Ribeiro, Diogo},
  title     = {{pinn}: A Modular PyTorch Framework for Physics-Informed Neural Networks},
  year      = {2026},
  version   = {0.3.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21844101},
  url       = {https://doi.org/10.5281/zenodo.21844101}
}
```

To cite the exact version you used rather than the software as a whole, take the
version-specific DOI from that release's
[Zenodo record](https://doi.org/10.5281/zenodo.21844101) and substitute it above.

Machine-readable metadata is provided in [`CITATION.cff`](CITATION.cff), from which GitHub
renders a "Cite this repository" button, and in [`.zenodo.json`](.zenodo.json), which
Zenodo reads when it archives a tagged release. Both are validated in CI.

### Primary References

1. **Raissi, M., Perdikaris, P., & Karniadakis, G. E.** (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. _Journal of Computational Physics_, 378, 686-707.

2. **Raissi, M., Perdikaris, P., & Karniadakis, G. E.** (2018). Numerical Gaussian processes for time-dependent and nonlinear partial differential equations. _SIAM Journal on Scientific Computing_, 40(1), A172-A198.

### Foundational Papers

1. **Raissi, M., Perdikaris, P., & Karniadakis, G. E.** (2017). Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations. _arXiv preprint arXiv:1711.10561_.

2. **Raissi, M., Perdikaris, P., & Karniadakis, G. E.** (2017). Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations. _arXiv preprint arXiv:1711.10566_.

### BibTeX Citations

```bibtex
@article{raissi2019physics,
  title={Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations},
  author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em},
  journal={Journal of Computational Physics},
  volume={378},
  pages={686--707},
  year={2019},
  publisher={Elsevier}
}

@article{raissi2018numerical,
  title={Numerical Gaussian processes for time-dependent and nonlinear partial differential equations},
  author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em},
  journal={SIAM Journal on Scientific Computing},
  volume={40},
  number={1},
  pages={A172--A198},
  year={2018},
  publisher={SIAM}
}

@article{raissi2017physicsI,
  title={Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations},
  author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em},
  journal={arXiv preprint arXiv:1711.10561},
  year={2017}
}

@article{raissi2017physicsII,
  title={Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations},
  author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em},
  journal={arXiv preprint arXiv:1711.10566},
  year={2017}
}
```

## 🗺️ Roadmap

See [**ROADMAP.md**](ROADMAP.md) for the current status of the library and what's planned next —
new PDE solvers and architectures, GPU/multi-node validation, richer uncertainty quantification, and
documentation. Contributions toward any roadmap item are very welcome.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📘 Tutorials (Jupyter)

The [`notebooks/`](notebooks/) folder contains **self-contained, executed** tutorials in a narrative,
Kaggle-style format — theory in LaTeX, runnable code, and baked-in plots and metrics. Each runs
end-to-end on a laptop CPU in a minute or two.

- **Core** — [getting started](notebooks/01_getting_started.ipynb), the
  [visualization toolkit](notebooks/00_demo_visualization.ipynb), and
  [sampling & metrics](notebooks/02_sampling_and_metrics.ipynb)
- **[`basic/`](notebooks/basic/)** — Burgers, Allen–Cahn (custom residual), a reusable PDE template,
  and 2-D Navier–Stokes
- **[`advanced/`](notebooks/advanced/)** — inverse problems, uncertainty quantification,
  multi-scale/curriculum training, and distributed training
- **[`benchmarks/`](notebooks/benchmarks/)** — active learning, convergence, loss weighting,
  spectral bias, performance, and transfer learning

Start with [`notebooks/01_getting_started.ipynb`](notebooks/01_getting_started.ipynb) and follow the
guided path in the [notebooks README](notebooks/README.md). The standalone scripts the notebooks are
based on live in [`examples/`](examples/).

### Development Setup

```bash
# Clone and set up a development environment
git clone https://github.com/DiogoRibeiro7/pinn.git
cd pinn
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install with the development and visualisation extras. Install `viz` too:
# importing pinn pulls in matplotlib and seaborn.
pip install -e ".[dev,viz]"

# Optional: run the formatting and lint checks on every commit
pre-commit install

# Run the checks CI runs
pytest
black --check .
flake8 .
bandit -r src -ll
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow, and
[SECURITY.md](SECURITY.md) before deploying the serving layer or loading a
checkpoint you did not produce yourself.

### Areas for Contribution

- 🔬 New PDE implementations
- 🎨 Visualization enhancements
- 📊 Additional metrics and analysis tools
- 📚 Documentation and tutorials
- 🏃‍♂️ Performance optimizations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/DiogoRibeiro7/pinn/issues)
- **Discussions**: [GitHub Discussions](https://github.com/DiogoRibeiro7/pinn/discussions)
- **Email**: [dfr@esmad.ipp.pt](mailto:dfr@esmad.ipp.pt)
- **Documentation**: [Full Documentation](https://DiogoRibeiro7.github.io/pinn)

## 🙏 Acknowledgments

- Original PINN framework by Raissi, Perdikaris, and Karniadakis
- PyTorch team for automatic differentiation capabilities
- Scientific computing community for continuous inspiration

--------------------------------------------------------------------------------

⭐ **Star this repository** if you find it useful for your research!

📖 **Read the full documentation** at [DiogoRibeiro7.github.io/pinn](https://DiogoRibeiro7.github.io/pinn)

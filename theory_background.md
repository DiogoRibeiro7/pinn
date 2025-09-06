# Mathematical Foundations of Physics-Informed Neural Networks

Comprehensive theoretical background for Physics-Informed Neural Networks (PINNs) and their mathematical foundations.

## Table of Contents

- [Introduction to PINNs](#introduction-to-pinns)
- [Mathematical Framework](#mathematical-framework)
- [Loss Function Formulation](#loss-function-formulation)
- [Automatic Differentiation](#automatic-differentiation)
- [Implemented PDEs](#implemented-pdes)
- [Numerical Methods Integration](#numerical-methods-integration)
- [Convergence Theory](#convergence-theory)
- [Advanced Topics](#advanced-topics)

--------------------------------------------------------------------------------

## Introduction to PINNs

### Motivation

Traditional numerical methods for solving partial differential equations (PDEs) face several challenges:

1. **Mesh Generation**: Complex geometries require sophisticated meshing
2. **Curse of Dimensionality**: High-dimensional problems become computationally intractable
3. **Inverse Problems**: Parameter estimation requires iterative forward solves
4. **Multi-scale Problems**: Different time/length scales are difficult to handle

Physics-Informed Neural Networks (PINNs) address these challenges by:

- **Meshfree approach**: No spatial/temporal discretization required
- **Automatic differentiation**: Exact computation of derivatives
- **Unified framework**: Forward and inverse problems in single formulation
- **Flexible geometry**: Irregular domains handled naturally

### Core Principle

PINNs embed the physics of the problem directly into the neural network loss function. The network learns to satisfy:

1. **Initial conditions** (supervised learning)
2. **Boundary conditions** (supervised learning)
3. **Governing PDEs** (unsupervised, physics-informed learning)

--------------------------------------------------------------------------------

## Mathematical Framework

### Problem Formulation

Consider a general time-dependent PDE in domain $\Omega \subset \mathbb{R}^d$ with time interval $T = [0, T_{final}]$:

$\mathcal{F}[u](t, \mathbf{x}) = f(t, \mathbf{x}), \quad (t, \mathbf{x}) \in T \times \Omega$

Subject to:

- **Initial condition**: $u(0, \mathbf{x}) = u_0(\mathbf{x}), \quad \mathbf{x} \in \Omega$
- **Boundary conditions**: $\mathcal{B}[u](t, \mathbf{x}) = g(t, \mathbf{x}), \quad (t, \mathbf{x}) \in T \times \partial\Omega$

Where:

- $u(t, \mathbf{x})$: Unknown solution field
- $\mathcal{F}[\cdot]$: Differential operator (e.g., $\frac{\partial}{\partial t} + \mathcal{L}$ where $\mathcal{L}$ is spatial operator)
- $\mathcal{B}[\cdot]$: Boundary operator (Dirichlet, Neumann, Robin, periodic)
- $f(t, \mathbf{x})$: Source/forcing term
- $g(t, \mathbf{x})$: Boundary data

### Neural Network Approximation

The PINN approximates the solution using a deep neural network:

$u(t, \mathbf{x}) \approx u_\theta(t, \mathbf{x}) = \mathcal{N}_\theta(t, \mathbf{x})$

Where:

- $\mathcal{N}_\theta$: Neural network with parameters $\theta$
- Typical architecture: Multi-Layer Perceptron (MLP) with tanh activations

$u_\theta(t, \mathbf{x}) = W^{(L)} \circ \sigma \circ W^{(L-1)} \circ \cdots \circ \sigma \circ W^{(1)}(t, \mathbf{x}) + b^{(L)}$

Where $\sigma(z) = \tanh(z)$ is the activation function, chosen for its infinite differentiability.

--------------------------------------------------------------------------------

## Loss Function Formulation

### Composite Loss Function

The PINN loss function combines three components:

$\mathcal{L}_{total}(\theta) = \lambda_{IC} \mathcal{L}_{IC}(\theta) + \lambda_{BC} \mathcal{L}_{BC}(\theta) + \lambda_{PDE} \mathcal{L}_{PDE}(\theta)$

Where $\lambda_{IC}$, $\lambda_{BC}$, $\lambda_{PDE}$ are weighting parameters.

### 1\. Initial Condition Loss

$\mathcal{L}_{IC}(\theta) = \frac{1}{N_{IC}} \sum_{i=1}^{N_{IC}} |u_\theta(0, \mathbf{x}_i^{IC}) - u_0(\mathbf{x}_i^{IC})|^2$

Where ${\mathbf{x}_i^{IC}}_{i=1}^{N_{IC}}$ are sampled initial condition points.

### 2\. Boundary Condition Loss

$\mathcal{L}_{BC}(\theta) = \frac{1}{N_{BC}} \sum_{i=1}^{N_{BC}} |\mathcal{B}[u_\theta](t_i^{BC}, \mathbf{x}_i^{BC}) - g(t_i^{BC}, \mathbf{x}_i^{BC})|^2$

Where ${(t_i^{BC}, \mathbf{x}_i^{BC})}_{i=1}^{N_{BC}}$ are sampled boundary points.

### 3\. PDE Residual Loss

$\mathcal{L}_{PDE}(\theta) = \frac{1}{N_f} \sum_{i=1}^{N_f} |\mathcal{F}[u_\theta](t_i^f, \mathbf{x}_i^f) - f(t_i^f, \mathbf{x}_i^f)|^2$

Where ${(t_i^f, \mathbf{x}_i^f)}_{i=1}^{N_f}$ are interior collocation points.

### Loss Weighting Strategies

The choice of weights $\lambda_{IC}$, $\lambda_{BC}$, $\lambda_{PDE}$ is crucial:

1. **Equal weighting**: $\lambda_{IC} = \lambda_{BC} = \lambda_{PDE} = 1$
2. **Adaptive weighting**: Adjust based on relative loss magnitudes
3. **Annealing**: Time-dependent weights during training
4. **Gradient-based**: Balance gradient magnitudes across loss terms

--------------------------------------------------------------------------------

## Automatic Differentiation

### Forward Mode AD

For computing derivatives $\frac{\partial u}{\partial x_i}$:

1. **Dual numbers**: $u = a + b\epsilon$ where $\epsilon^2 = 0$
2. **Chain rule**: Automatically propagated through network
3. **Computational cost**: $O(n)$ for $n$ input variables

### Reverse Mode AD (Backpropagation)

For computing $\frac{\partial \mathcal{L}}{\partial \theta_i}$:

1. **Forward pass**: Compute loss and store intermediate values
2. **Backward pass**: Apply chain rule in reverse order
3. **Computational cost**: $O(1)$ relative to number of parameters

### Higher-Order Derivatives

PINNs require computing mixed partial derivatives. For Burgers equation:

$\frac{\partial^2 u}{\partial x^2} = \frac{\partial}{\partial x}\left(\frac{\partial u}{\partial x}\right)$

Implementation using PyTorch:

```python
u_x = torch.autograd.grad(u, x, torch.ones_like(u), 
                         retain_graph=True, create_graph=True)[0]
u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), 
                          retain_graph=True, create_graph=True)[0]
```

The `create_graph=True` flag enables higher-order derivatives by building computational graph for gradients.

--------------------------------------------------------------------------------

## Implemented PDEs

### 1\. Burgers Equation

#### Governing Equation

$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2} = 0$

#### Physical Interpretation

- **Convection term**: $u \frac{\partial u}{\partial x}$ (nonlinear transport)
- **Diffusion term**: $\nu \frac{\partial^2 u}{\partial x^2}$ (viscous dissipation)
- **Balance**: Competition between steepening and smoothing

#### Dimensionless Parameters

- **Reynolds number**: $Re = \frac{UL}{\nu}$ where $U$, $L$ are characteristic velocity and length
- **High Re**: Convection-dominated (shock formation)
- **Low Re**: Diffusion-dominated (smooth solutions)

#### PINN Residual

$r_{Burgers} = \frac{\partial u_\theta}{\partial t} + u_\theta \frac{\partial u_\theta}{\partial x} - \nu \frac{\partial^2 u_\theta}{\partial x^2}$

### 2\. Navier-Stokes Equations

#### Governing Equations

**Momentum equations**: $\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} + \nabla p - \nu \nabla^2 \mathbf{u} = \mathbf{f}$

**Continuity equation**: $\nabla \cdot \mathbf{u} = 0$

#### 2D Incompressible Form

$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} + \frac{\partial p}{\partial x} - \nu \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) = 0$

$\frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} + \frac{\partial p}{\partial y} - \nu \left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}\right) = 0$

$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$

#### PINN Residual Vector

$\mathbf{r}_{NS} = \begin{bmatrix} r_u \ r_v \ r_c \end{bmatrix}$

Where $r_u$, $r_v$ are momentum residuals and $r_c$ is continuity residual.

#### Taylor-Green Vortex (Analytical Solution)

For testing, we use the Taylor-Green vortex with periodic boundary conditions:

$u(t,x,y) = \sin(x)\cos(y)e^{-2\nu t}$ $v(t,x,y) = -\cos(x)\sin(y)e^{-2\nu t}$ $p(t,x,y) = -\frac{1}{4}(\cos(2x) + \cos(2y))e^{-4\nu t}$

### 3\. Allen-Cahn Equation

#### Governing Equation

$\frac{\partial u}{\partial t} - \nu \frac{\partial^2 u}{\partial x^2} + u^3 - u = 0$

#### Physical Interpretation

- **Diffusion term**: $\nu \frac{\partial^2 u}{\partial x^2}$ (interface smoothing)
- **Reaction term**: $u^3 - u = u(u^2 - 1)$ (double-well potential)
- **Phase separation**: $u \approx \pm 1$ in bulk, $u \approx 0$ at interfaces

#### Ginzburg-Landau Energy

The Allen-Cahn equation is the $L^2$ gradient flow of: $E[u] = \int_\Omega \left[ \frac{\nu}{2}|\nabla u|^2 + \frac{1}{4}(u^2 - 1)^2 \right] dx$

#### PINN Residual

$r_{AC} = \frac{\partial u_\theta}{\partial t} - \nu \frac{\partial^2 u_\theta}{\partial x^2} + u_\theta^3 - u_\theta$

### 4\. Nonlinear Schrödinger Equation

#### Governing Equation

$i\frac{\partial u}{\partial t} + \frac{1}{2}\frac{\partial^2 u}{\partial x^2} + |u|^2 u = 0$

#### Complex Field Representation

Since $u(t,x) \in \mathbb{C}$, we represent it as: $u = a + ib$ where $a, b \in \mathbb{R}$.

#### Real-Imaginary Decomposition

**Real part**: $\frac{\partial a}{\partial t} = -\frac{1}{2}\frac{\partial^2 b}{\partial x^2} - (a^2 + b^2)b$

**Imaginary part**: $\frac{\partial b}{\partial t} = \frac{1}{2}\frac{\partial^2 a}{\partial x^2} + (a^2 + b^2)a$

#### PINN Residual Vector

$\mathbf{r}_{NLS} = \begin{bmatrix} \frac{\partial a_\theta}{\partial t} + \frac{1}{2}\frac{\partial^2 b_\theta}{\partial x^2} + (a_\theta^2 + b_\theta^2)b_\theta \ \frac{\partial b_\theta}{\partial t} - \frac{1}{2}\frac{\partial^2 a_\theta}{\partial x^2} - (a_\theta^2 + b_\theta^2)a_\theta \end{bmatrix}$

#### Conservation Laws

The NLS equation conserves:

1. **Mass**: $\int |u|^2 dx = \int (a^2 + b^2) dx$
2. **Energy**: $\int \left[\frac{1}{2}|\nabla u|^2 + \frac{1}{2}|u|^4\right] dx$

--------------------------------------------------------------------------------

## Numerical Methods Integration

### Discrete-Time PINNs

#### Runge-Kutta Integration

For time-dependent PDEs, we can discretize in time using RK methods:

$\frac{du}{dt} = N[u]$

where $N[\cdot]$ is the spatial operator.

#### RK4 Scheme

$u^{n+1} = u^n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$

Where:

- $k_1 = N[u^n]$
- $k_2 = N[u^n + \frac{h}{2}k_1]$
- $k_3 = N[u^n + \frac{h}{2}k_2]$
- $k_4 = N[u^n + hk_3]$

#### PINN-RK Loss

The discrete PINN enforces the RK relations: $\mathcal{L}_{RK} = \sum_{i=1}^{4} |U_i^{pred} - U_i^{RK}|^2$

where $U_i^{pred}$ are network predictions and $U_i^{RK}$ are RK stage values.

### Sampling Strategies

#### Latin Hypercube Sampling (LHS)

For $n$ samples in $d$ dimensions:

1. Divide each dimension into $n$ equal intervals
2. Sample once from each interval
3. Apply random permutation to each dimension

**Advantages**:

- Better space-filling than random sampling
- Improved convergence properties
- Efficient for high dimensions

#### Sobol Sequences

Low-discrepancy sequences with:

- **Discrepancy**: $D_N^* = O((\log N)^d/N)$
- **Better than random**: $O(N^{-1/2})$ for Monte Carlo
- **Quasi-random**: Deterministic but uniformly distributed

#### Adaptive Sampling

Based on residual magnitude:

1. Compute $|r(t_i, x_i)|$ at current points
2. Identify high-residual regions
3. Add new points in these regions
4. Iterate until convergence

--------------------------------------------------------------------------------

## Convergence Theory

### Universal Approximation

#### Theoretical Foundation

For MLPs with sufficient width, PINNs can approximate solutions arbitrarily well under regularity assumptions.

**Theorem (Universal Approximation)**: Let $\sigma$ be a non-polynomial activation function. For any continuous function $f: [0,1]^d \to \mathbb{R}$ and $\epsilon > 0$, there exists an MLP $\mathcal{N}$ such that: $\sup_{x \in [0,1]^d} |f(x) - \mathcal{N}(x)| < \epsilon$

#### PDE-Specific Results

For well-posed PDEs with smooth solutions:

1. **Existence**: Solutions exist in appropriate function spaces
2. **Uniqueness**: Solutions are uniquely determined by IC/BC
3. **Regularity**: Solutions have sufficient smoothness for NN approximation

### Error Analysis

#### Sources of Error

1. **Approximation error**: $|u - u_\theta^_|$ where $\theta^_$ is optimal parameters
2. **Optimization error**: $|u_{\theta^*} - u_{\hat{\theta}}|$ where $\hat{\theta}$ are trained parameters
3. **Sampling error**: Due to finite collocation points

#### Convergence Rates

Under regularity assumptions:

- **Network size**: Error decreases as $O(W^{-r/d})$ where $W$ is width, $r$ is solution regularity
- **Sampling**: Error decreases as $O(N^{-s/d})$ where $N$ is number of points
- **Training**: Depends on optimization landscape and algorithm

### Spectral Properties

#### Spectral Bias

Neural networks exhibit spectral bias toward low frequencies:

1. **Low frequencies**: Learned first and accurately
2. **High frequencies**: Learned slowly, may require more points
3. **Implications**: Need sufficient network capacity for multi-scale problems

#### Fourier Analysis

For periodic problems, the network learns Fourier modes: $u_\theta(x) \approx \sum_{k} c_k e^{ikx}$

The convergence rate depends on the decay of Fourier coefficients ${c_k}$.

--------------------------------------------------------------------------------

## Advanced Topics

### Multi-Scale Problems

#### Temporal Multi-Scale

For problems with disparate time scales: $\frac{\partial u}{\partial t} = -\frac{1}{\epsilon}A(u) + B(u)$

where $\epsilon \ll 1$ creates stiff dynamics.

**Strategies**:

1. **Progressive time windows**: Train on increasing time intervals
2. **Adaptive time sampling**: More points during rapid transitions
3. **Time-dependent weights**: Emphasize recent behavior

#### Spatial Multi-Scale

For problems with boundary layers or shock waves: **Strategies**:

1. **Adaptive mesh refinement**: Add points near sharp gradients
2. **Multi-fidelity**: Combine low/high resolution training
3. **Transfer learning**: Pre-train on smooth problems

### Inverse Problems

#### Parameter Estimation

Simultaneously learn solution $u_\theta$ and parameters $\lambda$: $\mathcal{L}_{inverse} = \mathcal{L}_{data} + \mathcal{L}_{PDE}(\lambda) + \mathcal{L}_{reg}(\lambda)$

where $\mathcal{L}_{data}$ is data fitting loss.

#### Data Assimilation

Incorporate sparse measurements: $\mathcal{L}_{data} = \sum_{i} |u_\theta(t_i^{obs}, x_i^{obs}) - u_i^{obs}|^2$

### Conservation Laws

#### Weak Enforcement

For conservative PDEs, enforce conservation in weak form: $\int_\Omega \frac{\partial u}{\partial t} dx + \int_{\partial\Omega} \mathbf{F} \cdot \mathbf{n} ds = 0$

#### Hard Constraints

Modify network architecture to satisfy conservation: $u_\theta = u_0 + \nabla \cdot \phi_\theta$

where $\phi_\theta$ is a vector potential ensuring $\nabla \cdot u_\theta = \nabla \cdot u_0$.

### Uncertainty Quantification

#### Bayesian PINNs

Use Bayesian neural networks: $p(\theta | D) \propto p(D | \theta) p(\theta)$

**Methods**:

1. **Variational inference**: Approximate posterior with simpler distribution
2. **Hamiltonian Monte Carlo**: Sample from posterior
3. **Ensemble methods**: Train multiple networks with different initializations

#### Dropout-Based UQ

Use Monte Carlo dropout during inference: $\text{Var}[u] \approx \frac{1}{T}\sum_{t=1}^T (u_{\theta,t} - \bar{u})^2$

where $u_{\theta,t}$ are predictions with different dropout masks.

### Optimization Challenges

#### Loss Balancing

The multi-objective nature creates optimization challenges:

1. **Gradient conflicts**: Different loss terms may have opposing gradients
2. **Scale differences**: Loss magnitudes can vary by orders of magnitude
3. **Training dynamics**: Some terms may dominate others

**Solutions**:

1. **GradNorm**: Balance gradient magnitudes
2. **Multi-task learning**: Use task-specific learning rates
3. **Curriculum learning**: Gradually introduce complexity

#### Initialization Strategies

1. **Xavier/Glorot**: $W \sim \mathcal{N}(0, \frac{2}{n_{in} + n_{out}})$
2. **He initialization**: $W \sim \mathcal{N}(0, \frac{2}{n_{in}})$
3. **LSUV**: Layer-sequential unit-variance initialization
4. **Pre-training**: Initialize with simpler problems

--------------------------------------------------------------------------------

## Practical Considerations

### Activation Functions

#### Tanh vs. Other Activations

**Tanh advantages**:

- Smooth and infinitely differentiable
- Bounded output $\in (-1, 1)$
- Good gradient properties

**Alternatives**:

- **Swish**: $\sigma(x) = x \cdot \text{sigmoid}(x)$
- **Sine**: $\sigma(x) = \sin(x)$ (periodic problems)
- **Adaptive**: Learn activation parameters

### Network Architecture

#### Depth vs. Width Trade-offs

- **Deep networks**: Better function approximation, harder to train
- **Wide networks**: Easier optimization, may require more parameters
- **Optimal choice**: Problem-dependent, often 8-10 layers with 64-128 units

#### Skip Connections

For very deep networks: $h^{(l+1)} = \sigma(W^{(l)}h^{(l)} + b^{(l)}) + h^{(l)}$

Benefits:

- Mitigate vanishing gradients
- Enable deeper architectures
- Improve training stability

### Computational Efficiency

#### GPU Acceleration

- **Automatic differentiation**: Highly parallelizable
- **Batch processing**: Process multiple collocation points simultaneously
- **Memory management**: Trade-off between memory and computation

#### Adaptive Training

- **Early stopping**: Stop when validation loss plateaus
- **Learning rate scheduling**: Reduce rate during training
- **Gradient clipping**: Prevent exploding gradients

--------------------------------------------------------------------------------

## Summary

Physics-Informed Neural Networks provide a powerful framework for solving PDEs by embedding physical laws directly into the learning process. Key advantages include:

1. **Meshfree**: No spatial discretization required
2. **Flexible**: Handle complex geometries and boundary conditions
3. **Unified**: Forward and inverse problems in single framework
4. **Automatic**: Exact derivatives via automatic differentiation

The mathematical foundation combines:

- Universal approximation theorems
- Variational principles from PDE theory
- Modern optimization techniques
- Automatic differentiation capabilities

While challenges remain in optimization, loss balancing, and multi-scale problems, PINNs represent a significant advance in computational physics and scientific machine learning.

--------------------------------------------------------------------------------

## References

1. **Raissi, M., Perdikaris, P., & Karniadakis, G. E.** (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. _Journal of Computational Physics_, 378, 686-707.

2. **Cybenko, G.** (1989). Approximation by superpositions of a sigmoidal function. _Mathematics of Control, Signals and Systems_, 2(4), 303-314.

3. **Hornik, K., Stinchcombe, M., & White, H.** (1989). Multilayer feedforward networks are universal approximators. _Neural Networks_, 2(5), 359-366.

4. **Lu, L., Meng, X., Mao, Z., & Karniadakis, G. E.** (2021). DeepXDE: A deep learning library for solving differential equations. _SIAM Review_, 63(1), 208-228.

5. **Wang, S., Yu, X., & Perdikaris, P.** (2022). When and why PINNs fail to train: A neural tangent kernel perspective. _Journal of Computational Physics_, 449, 110768.

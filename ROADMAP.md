# 🗺️ Roadmap

This document tracks where the `pinn` library is today and where it's headed. Items under
**Implemented** are available now; **Near-term** and **Future** are aspirational and may change.
Contributions toward any item are welcome — open an issue to coordinate.

> Status legend: ✅ done · 🚧 in progress / partial · 🔭 planned

---

## ✅ Implemented

**Solvers**
- Continuous-time PINN for the viscous Burgers equation (`ContinuousPINN`)
- Discrete-time Runge–Kutta PINN (`DiscreteRKPINN`)
- 2-D incompressible Navier–Stokes with periodic BCs (`NavierStokesPINN`)
- Generic residual framework (Allen–Cahn, nonlinear Schrödinger)
- Exact-solution-validated 1-D heat (`HeatPINN`) and wave (`WavePINN`) solvers, scored by
  `relative_l2_error` against their closed-form solutions
- Cole–Hopf reference solution for viscous Burgers (`pinn.solvers.reference`), cross-checked
  against an independent finite-difference solve
- Composable core API for Burgers, heat and wave problems, with first-class
  geometry, soft constraints, strong residuals and an equation-agnostic trainer
- Physical nondimensionalization for the composable trainer, including
  characteristic scales, derivative scale factors, dimensional model wrappers
  and FP32/FP64 smoke coverage
- Residual-based adaptive refinement (RAR) in the composable trainer, with
  candidate residual scoring, retained high-residual collocation points and
  optional diversity-aware selection and training-state diagnostics
- RAR benchmark script comparing uniform, residual-adaptive and
  diversity-aware residual-adaptive collocation on composable Burgers, heat and
  wave problems
- Configurable composable interior samplers (`pinn.sampling.InteriorSampler`)
  for trainer-owned residual collocation points, including uniform and
  Latin-hypercube policies plus adapters for legacy sampler interoperability
- Shared `pinn.sampling.latin_hypercube` helper re-exported by legacy solver
  modules for backward-compatible LHS sampling
- Benchmark reporting primitives (`pinn.benchmarks`) with JSON serialization,
  independent residual evaluation on fresh points and explicit reference
  provenance labels (`analytic`, `numerical`, `observational`)
- Classical numerical heat-equation reference solver (`pinn.numerics`) using a
  stable explicit finite-difference scheme for benchmark comparisons

**Architectures** (`pinn.models`)
- `MLP`, `SirenPINN`, `FourierFeaturePINN`, `MultiScalePINN`,
  `FourierMultiScalePINN`, `AttentionPINN`, `WaveletPINN`,
  `HierarchicalPINN`, `BayesianPINN`, `EnsemblePINN`

**Training** (`pinn.training`)
- Two-stage Adam → L-BFGS optimisation with loss-component tracking
- Adaptive loss weighting & gradient-balancing analysis
- Curriculum / multi-scale progressive training, meta-learning, multi-fidelity,
  adversarial training, memory-efficient (gradient checkpointing, AMP)

**Sampling** (`pinn.sampling`)
- Latin-Hypercube space-filling sampling
- Uniform and Latin-hypercube interior samplers for composable problem
  geometries
- Adapters between composable `InteriorSampler` policies and legacy
  `BaseSampler` consumers
- Gradient-based importance sampling
- Active learning (uncertainty / query-by-committee / expected-improvement acquisitions)

**Inverse problems & UQ** (`pinn.uncertainty`)
- Learnable PDE parameters via `torch.nn.Parameter`
- Deep ensembles, MC-dropout, Bayesian PINNs, GP-prior prediction

**Transfer learning** (`pinn.transfer`)
- Pre-training pipelines, fine-tuning, knowledge distillation, domain adaptation

**Scaling & ops**
- `DistributedTrainer` (data-parallel), gradient compression, hierarchical averaging, load balancing
- Computation caching (`pinn.optimization`)
- Model serving (REST + gRPC), model I/O, performance profiling

**Docs & examples**
- 16 runnable example scripts (`examples/{basic,advanced,benchmarks}`)
- 19 self-contained, executed Kaggle-style notebooks (`notebooks/`)
- Sphinx documentation scaffold (`docs/`)

---

## 🚧 Near-term

- **Accuracy & rigor**
  - Make the Burgers inverse-problem example identifiable (PDE-consistent measurements)
  - Demonstrate a problem where causal weighting measurably helps. The implementation is
    verified against the published formula, but on the wave equation it made no useful
    difference at any setting tried; the reported gains are on harder problems
    (Allen–Cahn, Kuramoto–Sivashinsky) with larger budgets
- **Architectures in practice**
  - Worked examples for `FourierFeaturePINN` / `MultiScalePINN` on stiff problems
- **Training**
  - Migrate the remaining legacy generic PDE solvers onto the composable
    problem/geometry/constraint/trainer path
  - Move legacy solver scaling and precision behavior onto the shared
    nondimensionalization path; the new composable trainer preserves configured
    FP32/FP64 dtypes, but legacy paths still need the same audit
  - Extend RAR benchmark comparisons beyond Burgers/heat/wave to additional
    representative PDEs
- **Quality**
  - CI that executes notebooks and example scripts to prevent regressions
  - Expand test coverage for solvers, sampling, and transfer modules. Coverage is
    currently ~60%, and `--cov-fail-under` in `pyproject.toml` is a ratchet set just
    below it — raise the gate as tests land. The thinnest modules are
    `utils/metrics.py` (24%), `utils/error_handling.py` (29%) and
    `utils/sampling.py` (41%)
  - Clear the 93 `mypy` errors across 27 modules measured on 2026-08-08. The type-check step in
    `.github/workflows/ci.yml` is `continue-on-error: true` until then; make it
    blocking once the backlog is gone
  - Keep tests pointed at the real implementations. `tests/unit/test_visualization.py`
    once shadowed every name it imported with a local test double, so its assertions
    never reached the library; an AST sweep of the suite found no other module doing
    this, and it is worth re-checking when new tests land

---

## 🔭 Future

- **Solvers**: 3-D Navier–Stokes, hard-constraint boundary conditions, additional PDE families
  (KdV, reaction–diffusion systems, elasticity)
- **Performance**: validated multi-GPU / multi-node runs, `torch.compile` integration,
  GPU benchmark suite
- **Uncertainty**: calibration metrics and conformal prediction for PINN outputs
- **Operator learning**: DeepONet / Fourier Neural Operator baselines for comparison
- **Packaging**: PyPI release and versioned, hosted documentation

---

## How to contribute

Pick an item, open an issue to discuss scope, and send a PR. Good first contributions: a new
exact-solution-validated PDE example, a notebook for an existing-but-undemonstrated architecture, or
test coverage for a module above. See the [README](README.md#-contributing) for the development setup.

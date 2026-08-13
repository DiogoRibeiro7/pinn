# 📓 PINN Notebooks

A gallery of **self-contained, runnable** Jupyter notebooks for the `pinnlab` library — written in a
narrative, Kaggle-style format with theory, code, executed outputs, and visualizations baked in.

Each notebook stands on its own: a styled header, a table of contents, the relevant physics with
LaTeX, step-by-step code, and a takeaways section. They all run end-to-end on a laptop **CPU**.

Runtimes, measured on one machine over a full execution pass rather than estimated: the median
notebook takes about 75 seconds and most finish inside two minutes, but the tail is long —
[`basic/04_navier_stokes_2d`](basic/04_navier_stokes_2d.ipynb) is the slowest at roughly 9 minutes,
and [`basic/05_heat_equation`](basic/05_heat_equation.ipynb) about 3. Budgets are deliberately modest;
see the notes at the bottom.

## 🚀 Quick start

```bash
pip install pinnlab            # or: pip install -e ".[dev,viz]" from a clone
jupyter notebook notebooks/    # or open in VS Code
```

Then open [`01_getting_started.ipynb`](01_getting_started.ipynb) and run all cells.

## 🗺️ Recommended path

| Order | Notebook | What you'll learn |
|---|---|---|
| 1 | [`01_getting_started`](01_getting_started.ipynb) | Orientation + your first solve in 5 minutes |
| 2 | [`basic/01_burgers_equation`](basic/01_burgers_equation.ipynb) | A complete PINN, explained step by step ⭐ |
| 3 | [`basic/02_allen_cahn`](basic/02_allen_cahn.ipynb) | Write your **own** PDE residual |
| 4 | [`basic/03_custom_pde`](basic/03_custom_pde.ipynb) | A reusable template (verified vs. an exact solution) |
| 5 | [`basic/04_navier_stokes_2d`](basic/04_navier_stokes_2d.ipynb) | A coupled PDE **system** (fluid flow) |
| 6 | [`basic/05_heat_equation`](basic/05_heat_equation.ipynb) | Scoring a PINN against a **closed-form** solution 📐 |
| 7 | [`basic/06_composable_api`](basic/06_composable_api.ipynb) | Splitting the **equation** from the **optimisation** 🧩 |
| 8 | [`advanced/…`](advanced/) and [`benchmarks/…`](benchmarks/) | Techniques & studies (below) |

## 📚 Contents

### Core

- [`00_demo_visualization`](00_demo_visualization.ipynb) — a tour of the `PINNVisualizer` plotting toolkit.
- [`01_getting_started`](01_getting_started.ipynb) — what PINNs are, library anatomy, first solve.
- [`02_sampling_and_metrics`](02_sampling_and_metrics.ipynb) — Latin-Hypercube vs. adaptive sampling, and the error-metrics utilities.

### Basic — forward problems

- [`basic/01_burgers_equation`](basic/01_burgers_equation.ipynb) — the flagship tutorial: viscous Burgers, validated via IC/BC fit and PDE residual.
- [`basic/02_allen_cahn`](basic/02_allen_cahn.ipynb) — phase separation with a hand-written residual.
- [`basic/03_custom_pde`](basic/03_custom_pde.ipynb) — a 5-step template, demonstrated on the heat equation with a known exact solution.
- [`basic/04_navier_stokes_2d`](basic/04_navier_stokes_2d.ipynb) — 2-D incompressible flow (Taylor–Green vortex).
- [`basic/05_heat_equation`](basic/05_heat_equation.ipynb) — the heat equation against its Fourier-series
  solution, reporting a relative L2 error rather than a plot, plus a stiffer two-mode variant.
- [`basic/06_composable_api`](basic/06_composable_api.ipynb) — the same physics as `02`, split into a
  problem that knows the equation and a trainer that does not. Uses Allen–Cahn's trivial solution, which
  plain training walks into, to show a strategy being swapped by changing one config field.

### Advanced — techniques

- [`advanced/01_transfer_and_distribution`](advanced/01_transfer_and_distribution.ipynb) — a roadmap of the advanced toolbox.
- [`advanced/02_distributed_training`](advanced/02_distributed_training.ipynb) — data-parallel scaling with `DistributedTrainer`.
- [`advanced/03_inverse_problems`](advanced/03_inverse_problems.ipynb) — recover a hidden PDE parameter from noisy data.
- [`advanced/04_multi_scale_training`](advanced/04_multi_scale_training.ipynb) — progressive coarse→fine (curriculum) training.
- [`advanced/05_uncertainty_quantification`](advanced/05_uncertainty_quantification.ipynb) — deep-ensemble error bars.

### Benchmarks — studies

- [`benchmarks/01_active_learning_comparison`](benchmarks/01_active_learning_comparison.ipynb) — adaptive vs. random collocation.
- [`benchmarks/02_convergence_study`](benchmarks/02_convergence_study.ipynb) — solution error vs. network width (vs. an exact solution).
- [`benchmarks/03_method_comparison`](benchmarks/03_method_comparison.ipynb) — the impact of loss weighting / balancing.
- [`benchmarks/04_multiscale_comparison`](benchmarks/04_multiscale_comparison.ipynb) — spectral bias and Fourier features.
- [`benchmarks/05_performance_benchmark`](benchmarks/05_performance_benchmark.ipynb) — training cost vs. collocation points and width.
- [`benchmarks/06_transfer_learning`](benchmarks/06_transfer_learning.ipynb) — transfer learning vs. training from scratch.

## ℹ️ Notes

- Notebooks are tuned for **fast CPU runs**, so training budgets are modest. For research-grade
  accuracy, increase `adam_steps` / `n_f` and (where available) enable the L-BFGS polish, adaptive
  sampling, and distributed training.
- Outputs are committed so the notebooks render fully on GitHub without re-running.
- The standalone scripts these notebooks are based on live in [`../examples/`](../examples/).

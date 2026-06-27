# Jupyter Notebooks

This folder contains interactive notebooks that mirror key examples from `examples/`.

## Available notebooks

### Core

- `01_getting_started.ipynb`
  - Quick sanity checks and import smoke-test.
  - Points to Burgers demo script.
- `02_sampling_and_metrics.ipynb`
  - Quick examples for `pinn.sampling` and `pinn.utils.metrics`.

### Demo

- `00_demo_visualization.ipynb`
  - Full standalone visualization demo script wrapper.

### Basic

- `basic/01_burgers_equation.ipynb`
  - Maps to `examples/basic/burgers_equation.py`
- `basic/02_allen_cahn.ipynb`
  - Maps to `examples/basic/allen_cahn.py`
- `basic/03_custom_pde.ipynb`
  - Maps to `examples/basic/custom_pde.py`
- `basic/04_navier_stokes_2d.ipynb`
  - Maps to `examples/basic/navier_stokes_2d.py`

### Advanced

- `advanced/01_transfer_and_distribution.ipynb`
  - API entry points for distributed/transfer/uncertainty modules.
- `advanced/02_distributed_training.ipynb`
  - Maps to `examples/advanced/distributed_training.py`
- `advanced/03_inverse_problems.ipynb`
  - Maps to `examples/advanced/inverse_problems.py`
- `advanced/04_multi_scale_training.ipynb`
  - Maps to `examples/advanced/multi_scale_training.py`
- `advanced/05_uncertainty_quantification.ipynb`
  - Maps to `examples/advanced/uncertainty_quantification.py`

### Benchmarks

- `benchmarks/01_active_learning_comparison.ipynb`
  - Maps to `examples/benchmarks/active_learning_comparison.py`
- `benchmarks/02_convergence_study.ipynb`
  - Maps to `examples/benchmarks/convergence_study.py`
- `benchmarks/03_method_comparison.ipynb`
  - Maps to `examples/benchmarks/method_comparison.py`
- `benchmarks/04_multiscale_comparison.ipynb`
  - Maps to `examples/benchmarks/multiscale_comparison.py`
- `benchmarks/05_performance_benchmark.ipynb`
  - Maps to `examples/benchmarks/performance_benchmark.py`
- `benchmarks/06_transfer_learning.ipynb`
  - Maps to `examples/benchmarks/transfer_learning.py`

## Running

From the repository root:

```bash
python -m pip install -e .
# Open notebooks with Jupyter/VS Code
jupyter notebook notebooks/
```

For full experiment runs, uncomment the `%run ...` cells in each notebook and ensure required optional dependencies are installed.

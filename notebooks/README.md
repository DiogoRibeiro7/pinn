# Jupyter Notebooks

This folder contains interactive notebooks that mirror key examples from `examples/`.

## Available notebooks

- `01_getting_started.ipynb`
  - Quick sanity checks and import smoke-test.
  - Points to Burgers demo script.
- `02_sampling_and_metrics.ipynb`
  - Quick examples for `pinn.sampling` and `pinn.utils.metrics`.
- `advanced/01_transfer_and_distribution.ipynb`
  - Advanced API entry points and pointers to distributed/transfer/uncertainty examples.

## Running

From the repository root:

```bash
python -m pip install -e .
# Open notebooks with Jupyter/VS Code
jupyter notebook notebooks/
```

For full experiment runs, uncomment the `%run ...` cells in each notebook and ensure required optional dependencies are installed.

Performance Optimization Guide
==============================

Use the profiling utilities to monitor training throughput and memory usage.

.. code-block:: python

   from pinn.utils.profiling import profile_performance

   @profile_performance
   def step():
       pass

Memory-aware training
---------------------

Large collocation sets can quickly exhaust accelerator memory.  The
:class:`~pinn.training.memory_efficient.MemoryEfficientPINN` wrapper enables
automatic batch-size selection, gradient accumulation and mixed-precision
training:

.. code-block:: python

   from pinn.training.memory_efficient import MemoryEfficientPINN, MemoryProfiler
   from pinn.solvers.raissi_improved import BurgersConfig, TrainConfig, burgers_residual
   from pinn.models.mlp import MLP

   profiler = MemoryProfiler()
   pinn = MemoryEfficientPINN(
       model=MLP(in_dim=2, hidden_layers=3, width=64, out_dim=1),
       device="cuda",
       pde_residual_fn=burgers_residual,
       cfg_burgers=BurgersConfig(),
       profiler=profiler,
       memory_budget_mb=8192,
   )

   config = TrainConfig(adam_steps=1000, lbfgs_max_iter=0, use_mixed_precision=True)
   pinn.train(config)

   print("Peak memory: ", profiler.summary()["peak_mb"], "MB")
   for message in profiler.recommendations(8192 * 1024**2):
       print("-", message)

The trainer honours the new :class:`~pinn.solvers.raissi_improved.TrainConfig`
fields ``collocation_batch_size``, ``effective_batch_size`` and
``gradient_checkpointing``.  When unset, a short warm-up evaluates the memory
footprint of the PDE residual to select a safe batch size, and a
:class:`~pinn.training.memory_efficient.MemoryProfiler` instance records per-step
statistics.

Benchmarking memory usage
-------------------------

The ``scripts/benchmark_memory.py`` helper runs a short Burgers experiment and
emits a JSON report with memory deltas and optimisation tips:

.. code-block:: bash

   python scripts/benchmark_memory.py
   cat memory_benchmark.json

The recommendations highlight when to lower the collocation batch size, enable
gradient checkpointing, or revisit sampling strategies to keep memory growth
under control.

Adaptive caching and precomputation
----------------------------------

The optimisation module now provides caching primitives that reuse training
points, forward activations and PDE residuals across optimisation steps.  This
dramatically reduces duplicate work when using hybrid strategies such as
gradient accumulation or L-BFGS polishing.

Enable caching through :class:`~pinn.solvers.raissi_improved.TrainConfig` by
supplying a :class:`~pinn.optimization.caching.CacheConfig` instance:

.. code-block:: python

   from pathlib import Path
   from pinn.optimization.caching import CacheConfig

   cache_cfg = CacheConfig(
       enabled=True,
       identifier="experiment-42",
       persistent_path=Path("./.pinn-cache"),
       store_gradients=True,
       store_activations=True,
       async_precompute=True,
   )

   config = TrainConfig(
       adam_steps=200,
       cache_config=cache_cfg,
   )

Cached training points are persisted to disk for future sessions, while the
``async_precompute`` flag overlaps permutation generation and gradient
evaluation with the main optimisation loop.

The accompanying ``scripts/benchmark_caching.py`` helper compares cached versus
baseline runs and reports the observed speed-up:

.. code-block:: bash

   python scripts/benchmark_caching.py --steps 30 --repeats 5

The script spins up lightweight Burgers experiments and prints the average
runtime with and without caching so that regressions can be detected in CI.

Multi-scale architecture comparison
-----------------------------------

To evaluate the benefits of the new multi-scale architectures, run the
``examples/benchmarks/multiscale_comparison.py`` script.  It trains both a
single-scale baseline network and a multi-scale counterpart on a synthetic
signal containing multiple characteristic frequencies and reports accuracy as
well as training time:

.. code-block:: bash

   python examples/benchmarks/multiscale_comparison.py --epochs 200 --num-points 256

The resulting ``multiscale_comparison.pt`` file stores the loss histories and
timings so that repeated experiments can be aggregated when tracking
regressions.

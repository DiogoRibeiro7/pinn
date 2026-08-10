Composable Core API
===================

The composable core separates a PINN experiment into problem, geometry,
constraints, residual form, model and trainer objects. The first migrated
problems are the exact-solution-validated Burgers, heat and wave equations.

Minimal Heat Example
--------------------

.. code-block:: python

   import torch

   from pinnlab.models import MLP
   from pinnlab.problems import HeatProblem
   from pinnlab.solvers.heat import HeatConfig
   from pinnlab.training import OptimizerConfig, Trainer, TrainerConfig

   problem = HeatProblem(HeatConfig(alpha=0.1))
   model = MLP(in_dim=problem.input_dim, hidden_layers=2, width=32,
               out_dim=problem.output_dim)

   trainer = Trainer(
       TrainerConfig(
           seed=0,
           device="cpu",
           dtype=torch.float64,
           collocation_count=512,
           optimizer=OptimizerConfig(adam_steps=500, lr=1e-3),
           loss_weights={"pde": 1.0, "ic": 1.0, "bc_left": 1.0, "bc_right": 1.0},
       )
   )
   result = trainer.train(problem, model)
   print(result.final_loss)

Configurable Sampling
---------------------

The trainer uses ``pinnlab.sampling.UniformInteriorSampler`` by default for
residual collocation points. ``LatinHypercubeInteriorSampler`` provides a
stratified space-filling policy over the same geometry bounds. Pass any object
implementing ``InteriorSampler`` to ``TrainerConfig.sampler`` when experiments
need a different interior point policy. Constraint sampling remains owned by
the constraint objects, and RAR adds retained high-residual points on top of the
configured base sampler.

``BaseSamplerInteriorAdapter`` wraps legacy ``pinnlab.utils.sampling.BaseSampler``
instances so they can feed ``TrainerConfig.sampler``. ``InteriorSamplerBaseAdapter``
exposes composable samplers through the legacy NumPy ``BaseSampler`` interface
for older utilities that have not migrated yet.

.. code-block:: python

   from pinnlab.sampling import LatinHypercubeInteriorSampler

   trainer = Trainer(
       TrainerConfig(
           collocation_count=512,
           sampler=LatinHypercubeInteriorSampler(),
           optimizer=OptimizerConfig(adam_steps=500, lr=1e-3),
       )
   )

Nondimensionalized Training
---------------------------

The composable trainer can run a model in dimensionless coordinates while
sampling constraints and residual points in physical coordinates. Pass a
``Nondimensionalizer`` to ``TrainerConfig.scaling`` to scale inputs before the
network sees them and scale outputs back before losses are evaluated.

.. code-block:: python

   import torch

   from pinnlab.models import MLP
   from pinnlab.problems import HeatProblem
   from pinnlab.scaling import CharacteristicScales, Nondimensionalizer, ScaledModel
   from pinnlab.solvers.heat import HeatConfig
   from pinnlab.training import OptimizerConfig, Trainer, TrainerConfig

   problem = HeatProblem(HeatConfig(alpha=0.1, length=2.0), t_final=5.0)
   scales = CharacteristicScales(
       input_scales=(5.0, 2.0),
       output_scales=(1.0,),
       parameter_scales={"alpha": 2.0**2 / 5.0},
   )
   scaling = Nondimensionalizer(scales)

   model = MLP(in_dim=problem.input_dim, hidden_layers=2, width=32,
               out_dim=problem.output_dim)
   trainer = Trainer(
       TrainerConfig(
           dtype=torch.float64,
           scaling=scaling,
           collocation_count=512,
           optimizer=OptimizerConfig(adam_steps=500, lr=1e-3),
       )
   )
   trainer.train(problem, model)

   # The trained model itself accepts dimensionless inputs. Wrap it when
   # evaluating with physical coordinates.
   physical_model = ScaledModel(model, scaling)

SIREN Models
------------

``SirenPINN`` is available from ``pinnlab.models`` for coordinate-based PDE
solutions with oscillatory structure. It uses sinusoidal hidden activations and
SIREN initialization bounds so coordinate derivatives remain useful during
PINN residual evaluation.

.. code-block:: python

   from pinnlab.models import SirenPINN

   model = SirenPINN(
       in_dim=problem.input_dim,
       hidden_layers=3,
       width=64,
       out_dim=problem.output_dim,
       omega_0=30.0,
   )

Residual-Adaptive Refinement
----------------------------

The generic trainer can add high-residual interior collocation points during
Adam training. Configure ``ResidualAdaptiveConfig`` when you want residual-based
adaptive refinement (RAR) on the composable problem path.

.. code-block:: python

   from pinnlab.training import (
       OptimizerConfig,
       ResidualAdaptiveConfig,
       Trainer,
       TrainerConfig,
   )

   trainer = Trainer(
       TrainerConfig(
           collocation_count=512,
           adaptive_refinement=ResidualAdaptiveConfig(
               candidate_count=2048,
               points_per_refinement=128,
               refresh_every=100,
               max_refined_points=512,
               diversity_weight=0.25,
           ),
           optimizer=OptimizerConfig(adam_steps=1000, lr=1e-3),
       )
   )

RAR diagnostics are recorded in ``TrainingState.diagnostics`` with keys such as
``rar_points``, ``rar_new_points``, ``rar_candidate_mean_residual`` and
``rar_candidate_max_residual``. Set ``diversity_weight`` above zero to trade a
portion of residual rank for normalized coordinate spread; diagnostics then
also include ``rar_selection_diversity_weight`` and
``rar_selected_min_distance``.

Benchmark Reports
-----------------

Use ``pinnlab.benchmarks`` when evaluating a trained model outside the trainer.
``evaluate_independent_residual`` samples fresh interior points from the
problem geometry and evaluates the strong-form residual with a new derivative
backend, avoiding accidental reuse of training collocation points.

.. code-block:: python

   from pinnlab.benchmarks import (
       BenchmarkCase,
       BenchmarkMetric,
       BenchmarkReference,
       BenchmarkReport,
       evaluate_independent_residual,
   )

   residual = evaluate_independent_residual(
       problem, physical_model, sample_count=4096, seed=123
   )
   report = BenchmarkReport(
       cases=(
           BenchmarkCase(
               name="heat-residual",
               reference=BenchmarkReference(
                   kind="analytic",
                   name="Fourier heat solution",
               ),
               metrics=(
                   BenchmarkMetric(
                       "residual_mse",
                       residual.mean_squared_residual,
                       higher_is_better=False,
                   ),
               ),
           ),
       ),
   )
   report.to_json("benchmark_report.json")

Classical references live in ``pinnlab.numerics``. For example,
``solve_heat_explicit`` returns a finite-difference heat solution labelled as a
``numerical`` reference, keeping it distinct from closed-form analytic data and
observational measurements in benchmark reports.

Design Responsibilities
-----------------------

``pinnlab.geometry``
    Owns coordinate bounds, containment checks, interior sampling and boundary
    sampling. ``Rectangle`` is used for space-time heat and wave domains;
    ``Interval`` is available for one-dimensional domains.

``pinnlab.sampling``
    Owns sampling policies used by the generic trainer. ``InteriorSampler`` is
    the collocation sampler protocol, ``UniformInteriorSampler`` preserves the
    default geometry-uniform behavior and ``LatinHypercubeInteriorSampler`` adds
    stratified space-filling residual collocation. Compatibility adapters bridge
    this protocol with legacy ``BaseSampler`` consumers during migration.

``pinnlab.constraints``
    Owns soft penalty losses for initial, Dirichlet, Neumann and periodic
    constraints. Constraint objects sample their own points and evaluate their
    own loss, so the trainer does not need equation-specific branches.

``pinnlab.problems``
    Owns equation parameters, dimensions, geometry, constraints and strong-form
    residual definitions. ``BurgersProblem``, ``HeatProblem`` and
    ``WaveProblem`` use coordinate order ``(t, x)`` and provide reference or
    analytic solution helpers.

``pinnlab.residuals``
    Provides ``StrongFormResidual`` and the initial autograd derivative backend.
    A later migration can replace the backend without changing problem or
    trainer contracts.

``pinnlab.scaling``
    Provides physical characteristic scales, nondimensional input/output
    transforms, derivative scale factors and ``ScaledModel``. The transforms
    preserve the configured dtype, so FP32 and FP64 training paths can be tested
    without silent downcasts.

``pinnlab.training.Trainer``
    Combines residual and constraint losses, applies configured loss weights,
    optionally retains high-residual RAR collocation points, and performs Adam
    plus optional L-BFGS optimization. It records typed ``TrainingState`` entries
    with named losses, weights, elapsed time and diagnostics.

``pinnlab.benchmarks`` and ``pinnlab.numerics``
    Own benchmark result serialization, independent residual scoring and
    classical numerical references. Reference descriptors must label data as
    ``analytic``, ``numerical`` or ``observational``.

Compatibility
-------------

Legacy solver imports such as ``pinnlab.solvers.heat.HeatPINN`` and
``pinnlab.solvers.wave.WavePINN`` remain available. The new ``HeatProblem`` and
``WaveProblem`` classes are adapters onto the same mathematical problems and
reference solutions, but the legacy solver classes have not been removed.
``BurgersProblem`` provides the same composable path for the standard viscous
Burgers benchmark while ``ContinuousPINN`` and ``DiscreteRKPINN`` remain
available as legacy solver facades.

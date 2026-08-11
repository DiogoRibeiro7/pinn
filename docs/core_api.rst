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
    residual definitions. ``AllenCahnProblem``, ``BurgersProblem``,
    ``HeatProblem`` and ``WaveProblem`` use coordinate order ``(t, x)`` and
    provide reference or analytic solution helpers.

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

Nonlinear And Periodic Problems
-------------------------------

``AllenCahnProblem`` solves ``u_t - nu u_xx + gamma (u^3 - u) = 0`` and is the
case that tests whether the composable core generalises: heat and wave are
linear with separable exact solutions, and Burgers is nonlinear but Dirichlet.
Allen-Cahn is nonlinear *and* periodic, and it needs no trainer changes.

.. code-block:: python

   from pinnlab.models.mlp import MLP
   from pinnlab.problems import AllenCahnProblem
   from pinnlab.training import OptimizerConfig, Trainer, TrainerConfig

   problem = AllenCahnProblem(n_constraint_samples=256)
   model = MLP(in_dim=2, out_dim=1, hidden_layers=4, width=128)
   trainer = Trainer(
       TrainerConfig(
           collocation_count=4_000,
           optimizer=OptimizerConfig(adam_steps=6_000, lbfgs_max_iter=500),
           loss_weights={"ic": 100.0, "bc_periodic": 100.0,
                         "bc_periodic_flux": 100.0},
       )
   )
   trainer.train(problem, model)

Three things about this problem are worth knowing before using it.

Allen-Cahn has a trivial solution and plain Adam training falls into it.
``u = 0`` is the unstable equilibrium between the two phases: it satisfies the
PDE exactly *and* both periodic constraints exactly, so only the initial
condition objects to it, and only on the ``t = 0`` face. A run that has landed
there fits the initial condition, decays to zero for ``t > 0``, and scores a
relative L2 error near ``1.0`` while three of its four loss components look
healthy. Measured at ``nu = 1e-2`` with 1500 Adam steps: ``ic`` at ``6e-4`` and
both periodic losses at ``2e-5``, with mean ``|u|`` of ``0.07`` against a
reference mean of order one. This is a property of the equation, not of this
implementation, and it is why Allen-Cahn is a standard test for
temporal-causality methods. The fix is not more Adam steps -- reach for
``CausalWeightConfig`` or a time-marching scheme.

Periodicity is imposed as *two* constraints, ``bc_periodic`` on the value and
``bc_periodic_flux`` on ``u_x``. Value matching alone is not periodicity: it
admits a field with a kink at the seam, and no collocation point straddles
``x = xmax`` so the residual never objects. ``PeriodicBoundary`` grew a
``derivative_order`` field for this; ``match_boundary_derivative=False``
reproduces the weaker condition.

The reference is *numerical*, not analytic -- Allen-Cahn has no closed-form
solution. ``problem.reference_kind`` reports ``"numerical"`` and
``problem.reference_grid()`` integrates the equation with a Fourier spectral
method in space and integrating-factor RK4 in time. It carries its own
discretisation error, so a reported ``relative_l2_error`` below the reference's
accuracy is not meaningful. The resolution that matters is spatial: with the
benchmark ``nu = 1e-4, gamma = 5`` the phase interfaces are about
``sqrt(nu / gamma) = 0.0045`` wide, and below roughly ``n_x = 512`` on
``[-1, 1]`` they are unresolved and the solution overshoots ``|u| = 1``.

Compatibility
-------------

Legacy solver imports such as ``pinnlab.solvers.heat.HeatPINN`` and
``pinnlab.solvers.wave.WavePINN`` remain available. The new ``HeatProblem`` and
``WaveProblem`` classes are adapters onto the same mathematical problems and
reference solutions, but the legacy solver classes have not been removed.
``BurgersProblem`` provides the same composable path for the standard viscous
Burgers benchmark while ``ContinuousPINN`` and ``DiscreteRKPINN`` remain
available as legacy solver facades.

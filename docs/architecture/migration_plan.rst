Migration Plan
==============

This plan maps the current solver-centric implementation to the target
scientific architecture required by the modernization prompts. It preserves
legacy imports while moving behavior into composable problem, geometry,
constraint, residual, differentiation and trainer abstractions.

Target Package Shape
--------------------

The long-term target is:

.. code-block:: text

   src/pinn/
   |-- problems/
   |-- geometry/
   |-- constraints/
   |-- residuals/
   |-- differentiation/
   |-- scaling/
   |-- models/
   |-- sampling/
   |-- optimizers/
   |-- training/
   |-- inverse/
   |-- decomposition/
   |-- operators/
   |-- benchmarks/
   `-- numerics/

New modules should be created only when they contain exercised functionality.
The migration should not create empty packages just to match the tree.

Core Abstraction Mapping
------------------------

Problems
    Move equation identity, dimensions, parameters, exact/reference solutions
    and residual definitions into ``pinn.problems``. Initial candidates are
    ``HeatProblem``, ``WaveProblem`` and ``BurgersProblem`` because they already
    have reference solutions and tests.

Geometry
    Replace duplicated ``Domain1D`` and ``pinn.utils.sampling.Domain`` usage in
    solver code with geometry objects such as ``Interval`` and
    ``Rectangle``. Geometry should own bounds, dimensionality, containment
    checks, interior sampling and boundary identification.

Constraints
    Extract initial, Dirichlet, Neumann and periodic conditions from solver
    training methods into concrete ``Constraint`` objects. The first migration
    should support soft constraint loss evaluation. Hard constraints should be
    added later as explicit model transforms, not hidden in problem classes.

Residuals
    Represent strong-form residual evaluation through a residual abstraction
    that calls the problem with a derivative backend. This removes
    equation-specific branches from a generic trainer while preserving
    multi-output residuals.

Differentiation
    Replace repeated ``torch.autograd.grad`` blocks with a derivative backend
    protocol. Start with an autograd backend that supports selected partials,
    gradients and Laplacians, then add a ``torch.func`` backend after analytic
    derivative tests exist.

Training
    Add a generic trainer that accepts a problem, model, residual form,
    sampler, constraints, optimizer configuration and loss weights. The trainer
    should produce a typed state/result object with named loss components,
    weights, step, elapsed time and diagnostics.

Scaling and dtype
    Add physical nondimensionalization and central dtype/device configuration
    only after the generic core path exists. New training code should avoid
    hard-coded float32 assumptions.

Compatibility Strategy
----------------------

No public import should be removed during the migration. Instead:

* Keep ``pinn.solvers.heat.HeatPINN`` and ``pinn.solvers.wave.WavePINN`` as
  compatibility wrappers around the new problem plus generic trainer.
* Keep ``pinn.solvers._base.TrainConfig`` importable until callers have a
  documented replacement.
* Keep ``pinn.solvers.raissi_improved.ContinuousPINN`` and
  ``DiscreteRKPINN`` available while a Burgers problem adapter is introduced.
* Keep top-level ``pinn.ContinuousPINN``, ``pinn.DiscreteRKPINN`` and
  ``pinn.TrainConfig`` bound to the legacy Burgers symbols until a deliberate
  public API migration is documented.
* Re-export moved reference functions from their old locations.

Proposed Milestones
-------------------

1. Core heat/wave path
    Introduce ``problems``, ``geometry``, ``constraints``, ``residuals`` and a
    generic trainer. Migrate heat and wave first because they share
    ``ExactlySolvablePINN`` and have analytic tests.

2. Differentiation backend
    Add a common derivative API and migrate the new heat/wave residuals to it.
    Keep legacy residuals working, then route safe helper calls through the
    backend.

3. Burgers adapter
    Implemented through ``pinn.problems.BurgersProblem`` with initial and
    Dirichlet constraints plus the Cole-Hopf reference evaluator. Keep
    ``raissi_improved`` as the compatibility facade until the new trainer
    supports all of its required features.

4. Sampling unification
    Partially implemented for the composable trainer through
    ``pinn.sampling.InteriorSampler``, ``UniformInteriorSampler`` and
    ``LatinHypercubeInteriorSampler``. Compatibility adapters bridge the
    composable protocol with legacy ``BaseSampler`` consumers. Legacy solver
    ``latin_hypercube`` names now re-export a shared ``pinn.sampling.core``
    helper. Remaining work is to migrate richer adaptive sampling policies onto
    the same interface directly.

5. Optimizer and training state
    Unify Adam, L-BFGS, callbacks, checkpointing, adaptive weighting and causal
    weighting through trainer configuration. Keep solver wrappers thin.

6. Numerical scaling
    Add physical scaling, dtype/device handling and derivative scale factors.
    Validate against heat-equation equivalence tests before expanding further.

7. Benchmarks and classical references
    Implemented for the composable heat path through ``pinn.benchmarks`` report
    serialization, independent residual evaluation and
    ``pinn.numerics.solve_heat_explicit``. Reference data is explicitly labelled
    as analytic, numerical or observational.

8. Advanced methods
    Add hard constraints, modern architectures, VPINNs, domain decomposition,
    inverse problems, conservation, neural operators and unified reporting only
    after the core abstractions are exercised by real tests.

Old-to-New Mapping
------------------

.. list-table::
   :header-rows: 1

   * - Current symbol
     - Future home
     - Migration note
   * - ``HeatConfig`` / ``HeatPINN`` / ``heat_exact``
     - ``pinn.problems`` plus ``pinn.solvers.heat`` wrappers
     - Best first target for generic trainer migration.
   * - ``WaveConfig`` / ``WavePINN`` / ``wave_exact``
     - ``pinn.problems`` plus ``pinn.solvers.wave`` wrappers
     - Exercises second-time-derivative residuals and initial velocity.
   * - ``BurgersConfig`` / ``burgers_residual``
     - ``pinn.problems`` and ``pinn.residuals``
     - ``BurgersProblem`` now covers the composable forward-problem path; legacy
       solver features such as caching and checkpoints remain in
       ``raissi_improved``.
   * - ``Domain1D``
     - ``pinn.geometry.Interval`` or space-time rectangle composition
     - Preserve old dataclass import during transition.
   * - ``ContinuousPINNGeneric``
     - Generic trainer plus problem/constraint instances
     - Its callable-based extension point becomes the compatibility adapter.
   * - ``NavierStokesPINN``
     - ``pinn.problems`` plus multi-output residuals and periodic constraints
     - Migrate after derivative and constraint abstractions support systems.
   * - ``pinn.utils.sampling.Domain``
     - ``pinn.geometry`` for domain semantics; ``pinn.sampling`` for policies
     - Composable trainer now consumes ``InteriorSampler`` policies; keep
       legacy sampling APIs backward compatible while migrating callers.
   * - solver-local ``latin_hypercube``
     - ``pinn.sampling.core.latin_hypercube``
     - Solver modules preserve the old names as compatibility imports.
   * - ``AdaptiveLossWeighting`` and ``causal_residual_loss``
     - ``pinn.training`` strategies used by generic trainer
     - Avoid equation-specific special cases.
   * - ``burgers_cole_hopf`` and ``relative_l2_error``
     - ``pinn.numerics.references`` or ``pinn.problems`` reference helpers
     - Re-export from ``pinn.solvers.reference``.

Validation Requirements for Each Migration Step
-----------------------------------------------

Every step should keep the following checks passing or record a precise
pre-existing failure:

* Old public imports still resolve.
* Existing examples that use retained imports still run or have documented
  replacements.
* Exact/reference solution tests compare residuals against analytic functions
  independently of training.
* New trainer tests exercise real production code, not local test doubles that
  shadow imports.
* Training and evaluation points are distinct in benchmark-style tests.
* No newly added public API lacks a docstring or type annotations.

Open Risks
----------

* Current quality gates are not all green: flake8 and mypy have historical
  backlogs.
* Several existing modules downcast inputs to float32, which conflicts with the
  later precision and nondimensionalization goals.
* Solver-specific training features are uneven: ``raissi_improved`` has richer
  batching, caching, AMP and active-learning paths than heat/wave/generic.
* The top-level package imports many optional subsystems eagerly, which makes
  import time nontrivial and emits optional-dependency warnings.
* Documentation currently builds with warnings, so future ``-W`` enforcement
  needs cleanup before it can gate CI.

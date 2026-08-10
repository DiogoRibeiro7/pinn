Current Architecture
====================

This page records the repository architecture as measured during the Prompt 00
baseline audit on 2026-08-08. It is intentionally descriptive: it does not
define the target design and it does not imply that existing duplicated
responsibilities should remain.

Package Layout
--------------

The package currently lives under ``src/pinnlab`` and is organized around a set of
mostly independent feature areas:

``pinnlab.solvers``
    Equation-specific PINN implementations. Burgers, generic 1-D PDEs,
    Navier-Stokes, heat and wave each own some combination of sampling, residual
    calculation, constraints, training loops and prediction.

``pinnlab.models``
    Neural network architectures, including MLP, SIREN, Fourier-feature,
    multiscale, attention, ensemble and wavelet variants.

``pinnlab.sampling`` and ``pinnlab.utils.sampling``
    Latin-hypercube, Sobol, residual-adaptive, active-learning, importance and
    multi-fidelity sampling utilities. Solver training loops still use local
    sampling helpers in several places.

``pinnlab.training``
    Reusable training utilities such as adaptive loss weighting, causal
    weighting, curriculum hooks, memory-efficient training, adversarial
    training and optimizer helper routines.

``pinnlab.uncertainty``
    MC-dropout, deep ensemble, GP-prior and Bayesian PINN helper APIs.

Other support packages
    Supporting subsystems for parallelism, caching, transfer learning, serving,
    plotting and configuration management.

Public Solver Surface
---------------------

The following symbols are part of the observed public surface because they are
exported from package modules, used in examples/tests, or documented in the
README:

``pinnlab.solvers.raissi``
    ``BurgersConfig``, ``Domain1D``, ``TrainConfig``, ``ContinuousPINN``,
    ``DiscreteRKPINN``, ``RKTableau``, ``rk4_tableau``, ``burgers_residual``,
    ``burgers_rhs``, ``u0_true``, ``u_left_bc``, ``u_right_bc``,
    ``latin_hypercube`` and ``set_seed``.

``pinnlab.solvers.raissi_improved``
    The main legacy Burgers implementation. It exports the same core symbols as
    ``raissi`` with additional adaptive sampling, active learning,
    memory-efficiency, caching, checkpointing and loss-history behavior.
    Top-level ``pinn`` imports ``BurgersConfig``, ``ContinuousPINN``,
    ``DiscreteRKPINN`` and ``TrainConfig`` from here.

``pinnlab.solvers.raissi_generic``
    ``Domain1D``, ``ContinuousPINNGeneric``, ``allen_cahn_residual_factory``,
    ``schrodinger_residual``, ``demo_allen_cahn`` and ``demo_schrodinger``.
    It imports ``latin_hypercube`` from ``pinnlab.sampling.core`` to avoid a
    solver-local drift-prone copy.

``pinnlab.solvers._base``
    ``ExactlySolvablePINN``, shared ``TrainConfig``, ``Domain1D`` re-export and
    ``gradient`` for the exact heat/wave path.

``pinnlab.solvers.heat`` and ``pinnlab.solvers.wave``
    ``HeatConfig``, ``HeatPINN``, ``heat_exact``, ``WaveConfig``, ``WavePINN``
    and ``wave_exact``. These derive from ``ExactlySolvablePINN``.

``pinnlab.solvers.reference``
    ``burgers_cole_hopf``, ``burgers_reference_grid`` and
    ``relative_l2_error``. Tests cross-check the Burgers reference against an
    independent finite-difference solve.

``pinnlab.solvers.navier_stokes``
    ``NSConfig``, ``TrainConfig``, ``NavierStokesPINN``, ``tgv_u``, ``tgv_v``,
    ``tgv_p`` and ``demo_tgv``.

Symbol and Responsibility Map
-----------------------------

Domain objects
    ``BurgersConfig`` in ``raissi`` and ``raissi_improved`` owns both PDE
    parameters and space-time bounds for Burgers. ``NSConfig`` does the same for
    Navier-Stokes. ``Domain1D`` is duplicated across ``raissi``,
    ``raissi_improved`` and ``raissi_generic``; ``_base`` re-exports the
    ``raissi_generic`` version. General rectangular sampling uses
    ``pinnlab.utils.sampling.Domain``.

Solver base classes
    ``ExactlySolvablePINN`` is the only current base with shared production
    training behavior. ``ContinuousPINN``, ``DiscreteRKPINN``,
    ``ContinuousPINNGeneric`` and ``NavierStokesPINN`` are independent concrete
    trainers rather than subclasses of a common solver protocol.

PDE residual definitions
    Burgers residuals live in ``raissi`` and ``raissi_improved`` as
    ``burgers_residual``. Allen-Cahn and Schrodinger residuals live in
    ``raissi_generic``. Heat and wave residuals are instance methods on
    ``HeatPINN`` and ``WavePINN``. Navier-Stokes residuals are computed by
    ``NavierStokesPINN.pde_residuals``.

Derivative helpers
    ``_base.gradient`` wraps one first derivative for heat/wave. Burgers,
    generic residuals and Navier-Stokes repeat direct ``torch.autograd.grad``
    calls locally for first and second derivatives. There is no common
    Jacobian, Hessian, Laplacian or vector-output derivative backend yet.

Constraint and boundary-condition handling
    Initial and boundary conditions are embedded in solver-specific training
    code. Burgers uses ``u0_true``, ``u_left_bc`` and ``u_right_bc``. Heat/wave
    use subclass methods plus homogeneous Dirichlet assumptions in
    ``ExactlySolvablePINN._loss``. ``ContinuousPINNGeneric`` accepts user
    callables for initial, left and right boundary values. Navier-Stokes builds
    Taylor-Green initial data and periodic paired samples inside
    ``_make_training_points`` and ``train``. There is no first-class
    ``Constraint`` abstraction.

Samplers
    ``latin_hypercube`` is owned by ``pinnlab.sampling.core`` and re-exported from
    legacy solver modules for compatibility. Separately, ``pinnlab.utils.sampling``
    defines ``BaseSampler``, ``LatinHypercubeSampler``, ``SobolSampler``,
    ``AdaptiveSampler``, ``BoundarySampler``, ``TimeSteppingSampler`` and
    ``CompositeSampler``. ``pinnlab.sampling`` adds composable interior samplers,
    adapters, importance, active-learning and multi-fidelity samplers.

Loss weighting
    Static weights are passed directly to solver training methods. Adaptive
    weighting is implemented in ``pinnlab.training.adaptive_weighting`` and wired
    into the legacy Burgers and generic paths. Causal residual weighting is
    implemented in ``pinnlab.training.causal_weighting`` and wired into
    ``ExactlySolvablePINN`` through ``TrainConfig.causal``.

Optimizer selection and training loops
    Burgers, generic, exact-solvable and Navier-Stokes solvers each implement
    their own Adam loop and optional L-BFGS polish. Checkpointing, callbacks,
    batching, AMP, active learning and caching are present mainly in
    ``raissi_improved``. No equation-agnostic trainer currently owns the common
    training contract.

Prediction and evaluation
    Each solver has its own ``predict`` implementation converting numpy inputs
    to float32 tensors. Heat and wave add ``evaluation_grid`` and
    ``relative_l2_error`` in ``ExactlySolvablePINN``. Burgers has reference-grid
    utilities in ``pinnlab.solvers.reference``.

Reference and exact solutions
    Heat and wave have analytic closed-form functions. Burgers has a Cole-Hopf
    reference implementation that is treated as exact for the configured
    viscous Burgers problem and tested against an independent finite-difference
    solve. Navier-Stokes uses the analytic Taylor-Green vortex for smoke and
    initial data.

Duplication Points
------------------

The main duplication that a future migration must remove is concentrated in
the solver modules:

* ``Domain1D`` is repeated in ``raissi``, ``raissi_improved`` and
  ``raissi_generic``.
* ``set_seed`` is still owned by solver modules rather than a reproducibility
  subsystem.
* Adam plus optional L-BFGS loops are repeated in ``ContinuousPINN``,
  ``ContinuousPINNGeneric``, ``ExactlySolvablePINN`` and ``NavierStokesPINN``.
* Residual implementations manually repeat autograd derivative patterns.
* Constraint data generation and loss calculation are embedded inside training
  loops.
* ``TrainConfig`` names collide across solver modules while representing
  different fields.
* Prediction paths all downcast numpy arrays to float32.

Baseline Quality Results
------------------------

These results were measured on Windows with Python 3.13.5 in the local working
tree.

.. list-table::
   :header-rows: 1

   * - Command
     - Result
   * - ``black --check .``
     - Passed. 113 files would be left unchanged.
   * - ``flake8 .``
     - Failed with 668 violations. The first failures are pydocstyle docstring
       rules in docs, examples, scripts, source and tests; bugbear warnings are
       also present.
   * - ``bandit -r src -ll``
     - Passed. No medium or high severity issues identified.
   * - ``mypy src``
     - Failed with 93 errors in 27 files.
   * - ``pytest``
     - Passed. 241 tests passed, 4 warnings. Total coverage was 60.45%, above
       the configured 58% gate.
   * - ``sphinx-build -b html docs docs/_build/html``
     - Passed with 58 warnings. Warnings include duplicate autodoc object
       descriptions and a Navier-Stokes docstring parse issue.

Additional Measurements
-----------------------

Package import time
    ``import pinnlab`` took 5.292538 seconds in a fresh Python process. The import
    emitted a warning that SciPy was not available for advanced metrics and
    NumExpr selected 16 threads.

Smoke execution
    Tiny CPU one-step smoke runs succeeded for Burgers, heat and wave. Burgers
    produced a ``(2, 2)`` prediction grid. Heat and wave produced finite
    relative L2 values after one step; these smoke values are not accuracy
    claims.

Current Test Coverage Hot Spots
-------------------------------

The measured coverage table highlights the following low-coverage modules:

* ``pinnlab.solvers.raissi_generic``: 17%
* ``pinnlab.solvers.navier_stokes``: 23%
* ``pinnlab.utils.metrics``: 24%
* ``pinnlab.utils.error_handling``: 29%
* ``pinnlab.utils.sampling``: 41%
* ``pinnlab.config.management``: 43%

Factual Status Notes
--------------------

* Heat and wave exact solutions are implemented and covered by tests.
* Burgers reference evaluation is implemented through Cole-Hopf quadrature and
  is cross-checked against an independent finite-difference solve.
* Causal weighting is implemented and exercised, but the repository does not
  currently contain a benchmark proving broad accuracy improvement.
* Residual-adaptive and active-learning samplers exist, but residual-based
  adaptive refinement is not yet a unified training-loop capability across
  solvers.
* The package remains alpha-stage scientific software, not a production-ready
  framework with a stable modern architecture.

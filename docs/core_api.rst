Composable Core API
===================

The composable core separates a PINN experiment into problem, geometry,
constraints, residual form, model and trainer objects. The first migrated
problems are the exact-solution-validated heat and wave equations.

Minimal Heat Example
--------------------

.. code-block:: python

   import torch

   from pinn.models import MLP
   from pinn.problems import HeatProblem
   from pinn.solvers.heat import HeatConfig
   from pinn.training import OptimizerConfig, Trainer, TrainerConfig

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

Design Responsibilities
-----------------------

``pinn.geometry``
    Owns coordinate bounds, containment checks, interior sampling and boundary
    sampling. ``Rectangle`` is used for space-time heat and wave domains;
    ``Interval`` is available for one-dimensional domains.

``pinn.constraints``
    Owns soft penalty losses for initial, Dirichlet, Neumann and periodic
    constraints. Constraint objects sample their own points and evaluate their
    own loss, so the trainer does not need equation-specific branches.

``pinn.problems``
    Owns equation parameters, dimensions, geometry, constraints and strong-form
    residual definitions. ``HeatProblem`` and ``WaveProblem`` use coordinate
    order ``(t, x)`` and provide analytic solution helpers.

``pinn.residuals``
    Provides ``StrongFormResidual`` and the initial autograd derivative backend.
    A later migration can replace the backend without changing problem or
    trainer contracts.

``pinn.training.Trainer``
    Combines residual and constraint losses, applies configured loss weights,
    and performs Adam plus optional L-BFGS optimization. It records typed
    ``TrainingState`` entries with named losses, weights, elapsed time and
    diagnostics.

Compatibility
-------------

Legacy solver imports such as ``pinn.solvers.heat.HeatPINN`` and
``pinn.solvers.wave.WavePINN`` remain available. The new ``HeatProblem`` and
``WaveProblem`` classes are adapters onto the same mathematical problems and
reference solutions, but the legacy solver classes have not been removed.

# Roadmap

This document is the working plan for `pinn`: what is already available, what
stage the project is in now, what should happen next, and what must be true
before another release is cut.

Status legend: done = available in a release, in progress = partially available
or actively being migrated, planned = not started or intentionally deferred.

---

## Current Snapshot

The current released version is `v0.3.0`, published on 2026-08-09.

Release state:
- Tag: `v0.3.0`
- GitHub Release: published
- Release artifacts: source distribution and wheel attached to the GitHub
  Release
- Validation before release: full test suite passed (`285 passed`), Sphinx HTML
  documentation build succeeded, `python -m build` produced release artifacts,
  and `twine check` passed for both artifacts
- PyPI: not yet uploaded. The distribution is named `pinnlab`, because
  `pinn` on PyPI belongs to an unrelated REST-API client that also occupies
  the `import pinn` name. Publishing is wired up through
  `.github/workflows/publish.yml` using trusted publishing and runs on a
  version tag once the publisher is registered on pypi.org

What `v0.3.0` means architecturally:
- The project is no longer only a solver-centric collection of PINN examples.
- A composable path now exists for problem definitions, geometry, constraints,
  residual evaluation, scaling, sampling and trainer-owned optimization.
- Legacy solver imports remain public and must keep working while behavior
  moves behind them onto the composable architecture.
- The next stage is not another broad feature burst. It is hardening,
  migration, and making the new architecture the reliable default.

---

## Implemented

### Solver Coverage

Available solver families:
- Continuous-time PINN for the viscous Burgers equation through
  `ContinuousPINN`
- Discrete-time Runge-Kutta PINN through `DiscreteRKPINN`
- 2-D incompressible Navier-Stokes with periodic boundary conditions through
  `NavierStokesPINN`
- Generic residual framework for Allen-Cahn and nonlinear Schrodinger examples
- Exact-solution-validated 1-D heat and wave solvers, scored with
  `relative_l2_error` against closed-form solutions
- Composable Burgers, heat and wave problem adapters for the new generic
  trainer path

Current limitation:
- Solver behavior is still split between legacy training loops and the
  composable trainer. Heat, wave and Burgers have composable problem coverage;
  richer legacy features such as caching, AMP, active learning and some batching
  behavior still live mostly in `raissi_improved`.

### References And Benchmarks

Available reference and benchmark infrastructure:
- Cole-Hopf reference solution for viscous Burgers in `pinnlab.solvers.reference`
- Fourier spectral / integrating-factor RK4 reference for periodic Allen-Cahn in
  `pinnlab.solvers.reference`, labelled `numerical` because the equation has no
  closed form. Verified against the two limits where the answer is known: with
  `gamma = 0` it reproduces the heat equation to 1e-15, with `nu = 0` the
  logistic ODE to 1e-11, and halving the step divides the error by ~16
- Classical explicit finite-difference heat-equation reference solver in
  `pinnlab.numerics`
- Benchmark report primitives in `pinnlab.benchmarks`, including JSON
  serialization and explicit reference provenance labels:
  `analytic`, `numerical`, and `observational`
- Benchmark scripts for precision and residual-adaptive refinement workflows
- RAR comparison script covering uniform, residual-adaptive and
  diversity-aware residual-adaptive collocation on composable Burgers, heat and
  wave problems

Current limitation:
- Benchmark comparisons are still concentrated on Burgers, heat and wave.
  Additional PDE families need composable problem definitions before benchmark
  coverage should expand.

### Composable Core API

Implemented composable pieces:
- `pinnlab.geometry`: `Interval`, `Rectangle`, shape validation, containment
  checks, interior sampling and boundary masks
- `pinnlab.constraints`: soft initial, Dirichlet, Neumann and periodic constraints
- `pinnlab.residuals`: strong-form residual evaluation through a derivative
  backend
- `pinnlab.problems`: equation identity, dimensions, geometry, constraints and
  residual definitions for migrated problems
- `pinnlab.training.Trainer`: equation-agnostic training loop with typed config,
  state, result, named losses and diagnostics
- `pinnlab.scaling`: characteristic scales, nondimensionalization transforms,
  derivative scale factors and scaled model wrappers

Current limitation:
- Hard constraints, richer PDE systems, inverse-problem workflows and
  decomposition/operator-learning APIs are not yet first-class composable
  modules.

### Adaptive Refinement And Sampling

Implemented adaptive and sampling features:
- Residual-adaptive refinement (RAR) in the composable trainer
- Candidate residual scoring and retained high-residual collocation points
- Optional diversity-aware RAR selection
- RAR diagnostics recorded in training state
- Configurable trainer-owned interior sampler protocol:
  `pinnlab.sampling.InteriorSampler`
- `UniformInteriorSampler` as the default composable sampler
- `LatinHypercubeInteriorSampler` for stratified composable geometry sampling
- Compatibility adapters between the composable `InteriorSampler` protocol and
  legacy `BaseSampler` consumers
- Shared `pinnlab.sampling.latin_hypercube`, re-exported by legacy solver modules
  for backward-compatible LHS imports

Current limitation:
- Importance sampling, active learning and multi-fidelity sampling still mostly
  use legacy sampler/domain concepts. They need gradual migration onto the
  composable geometry and sampler interfaces.

### Models

Available architectures:
- `MLP`
- `SirenPINN`
- `FourierFeaturePINN`
- `MultiScalePINN`
- `FourierMultiScalePINN`
- `AttentionPINN`
- `WaveletPINN`
- `HierarchicalPINN`
- `BayesianPINN`
- `EnsemblePINN`

Current limitation:
- Several architectures exist as APIs but need stronger examples, benchmarks
  and guidance for when they are appropriate. The next practical examples
  should focus on `FourierFeaturePINN`, `MultiScalePINN` and `SirenPINN` on
  problems where their inductive biases matter.

### Training And Operations

Available training/ops features:
- Adam to L-BFGS optimization patterns
- Loss-component tracking
- Adaptive loss weighting and gradient-balancing analysis
- Curriculum and multi-scale progressive training
- Meta-learning, multi-fidelity and adversarial training utilities
- Memory-efficient training utilities, including gradient checkpointing and AMP
- Distributed/data-parallel training, gradient compression, hierarchical
  averaging and load balancing
- Computation caching
- REST and gRPC model serving, model I/O and performance profiling

Current limitation:
- These capabilities are unevenly integrated. The composable trainer owns the
  new core path, while several advanced strategies still sit beside or inside
  legacy solver-specific training loops.

### Documentation And Examples

Available documentation and examples:
- README with project overview, installation, examples and citation guidance
- Sphinx documentation scaffold
- Core API documentation for composable problems, scaling, RAR and sampling
- Architecture documentation describing the current state and migration plan
- 16 runnable example scripts under `examples/{basic,advanced,benchmarks}`
- 19 self-contained tutorial notebooks under `notebooks/`

Current limitation:
- Documentation builds successfully but still reports existing warnings,
  mainly duplicate object descriptions and legacy docstring formatting issues.
- Example and notebook execution is not yet a blocking CI gate.

---

## Active Stage: Post-0.3 Hardening

The next work cycle should focus on hardening the composable architecture and
reducing migration risk. The goal is to make the new path boring, documented
and hard to regress before adding another large feature family.

Primary goals:
- Keep every legacy public import stable.
- Move behavior behind legacy facades onto composable problem, geometry,
  constraint, residual, scaling, sampling and trainer components.
- Reduce duplicated training-loop logic across solvers.
- Make validation stricter without breaking users abruptly.
- Pay down type, lint and documentation debt where it prevents CI from becoming
  stricter.

Non-goals for the immediate next cycle:
- Do not add empty target packages just to match the long-term architecture.
- Do not introduce hard-constraint, operator-learning or domain-decomposition
  APIs until the existing composable path is exercised by more tests and
  examples.
- Do not remove legacy solver imports during the migration.

---

## Near-Term Milestones

### 1. Stabilize The Composable Trainer

Why it matters:
The generic trainer is the center of the new architecture. It should absorb
shared behavior from legacy solvers instead of becoming another parallel path.

Concrete work:
- ✅ Callback hooks for logging, checkpoints and custom diagnostics.
  `Trainer.train` takes `callbacks=[...]`; `TrainerCallback` provides
  `on_train_begin`, `on_state_recorded`, `on_phase_end` and `on_train_end`.
- ✅ Adam/L-BFGS transitions are inspectable. `TrainingState.phase` names the
  optimizer that produced each record and `TrainingResult.states_for_phase`
  filters on it, instead of leaving the boundary implicit in step numbering.
- ✅ Documented the responsibilities of `TrainerConfig`, `TrainingState` and
  `TrainingResult`, including the boundary between them: config owns how to
  train, the problem owns what is solved, and the state records what happened.
  (An earlier note here claimed `TrainerConfig` was undocumented; it already
  carried per-field docs, and what was missing was the boundary itself.)
- ✅ Tests: callbacks, phase reporting, loss weights, causal diagnostics, dtype
  behaviour, custom samplers and scaling interactions are all covered by
  focused tests. `tests/unit/test_trainer_components.py` asserts the configured
  sampler is genuinely consulted, with the promised geometry, count, dtype and
  seeded generator -- a trainer that ignored `config.sampler` would otherwise
  still produce a falling loss and look correct.
- 🔭 Identify which `raissi_improved` training features should become generic
  trainer features first.

Also done here, since it blocked the above: the sample-evaluate-combine
sequence in `Trainer.train` had been duplicated at four call sites (Adam loop,
L-BFGS closure, post-L-BFGS record, no-step fallback), so any change to
sampling or refinement had to be repeated four times. It now lives in
`Trainer._evaluate`.

Done when:
- Heat, wave and Burgers composable training paths can cover the common
  workflows currently demonstrated by examples.
- New trainer behavior is covered by focused tests rather than only by
  end-to-end training smoke tests.

### 2. Migrate Legacy Solver Behavior Behind Stable Facades

Why it matters:
Users should not have to rewrite imports immediately, but the internals should
stop drifting across solver modules.

Concrete work:
- Keep `HeatPINN`, `WavePINN`, `ContinuousPINN`, `DiscreteRKPINN`,
  `ContinuousPINNGeneric` and `NavierStokesPINN` importable.
- ✅ Heat and wave wrappers now train through `Trainer` and their composable
  problem. `HeatPINN` and `WavePINN` keep their public API -- same
  `TrainConfig` in, same history dicts out -- while `_base.py` lost its own
  sampling and loss machinery, 88 lines. A domain guard rejects wrappers whose
  geometry the composable problem cannot express, rather than training against
  a different region than the object reports.

  On accuracy the honest answer is that the two paths are indistinguishable at
  the sample size measured. Three seeds each, heat problem, 2000 Adam steps
  plus 150 L-BFGS iterations, 3000 collocation points:

  | | legacy | delegated |
  | --- | --- | --- |
  | median | 4.28e-4 | 6.58e-4 |
  | range | 3.94e-4 to 8.70e-4 | 3.73e-4 to 9.12e-4 |

  The within-path spread is 2.2x to 2.4x while the median difference is 1.54x,
  and the delegated path wins on one of the three seeds.

  **Caution for anyone comparing configurations here.** Every "A beats B"
  claim recorded in this file before 2026-08-11 came from single runs, and a
  single run cannot resolve a difference of 2x on this problem. That includes
  the earlier entries reporting that the generic trainer beat the legacy loop,
  and that resampled collocation beat fixed collocation by roughly 2x. Those
  differences sit inside the spread measured above and should be treated as
  unproven rather than as results. The L-BFGS closure fix is the exception: a
  6.3x improvement, reproduced across separate experiments and supported by a
  mechanism, comfortably exceeds the noise.

  The figures the project publishes were single runs, and both were optimistic.
  Measured over five seeds at the notebook's budget:

  | | published | median | range |
  | --- | --- | --- | --- |
  | heat single-mode | 3.5e-4 | 7.3e-4 | 3.6e-4 to 9.6e-4 |
  | heat two-mode | 2.3e-3 | 6.0e-3 | 3.6e-3 to 8.6e-3 |

  Both published values beat the best of five seeds, so a reader who reran the
  notebook would most likely see roughly double the documented error and
  reasonably conclude they had broken something. The notebook now states the
  median and range before printing its own single figure, and its outputs have
  been regenerated against the delegated path.

  Remaining: the v0.2.0 changelog entry still quotes 1.7e-3, 2.3e-3 and 4.0e-2
  as characteristic. It is a historical record of what was measured then and
  has been left as written, but it should not be treated as a benchmark.

Done when:
- ✅ Public import compatibility tests cover the retained legacy symbols.
  `tests/unit/test_public_api.py` pins 346 exported names across 20 modules
  against `public_api_snapshot.json`, and separately checks the legacy solver
  classes, legacy module paths and top-level aliases named above. The snapshot
  is a floor: adding exports needs no change, removing one fails and has to be
  recorded deliberately, which is the moment to decide whether a deprecation
  shim is owed.
- ✅ Legacy wrappers are thin enough that shared behavior is tested in one
  place. `HeatPINN` and `WavePINN` drive `Trainer` rather than carrying their
  own loop, and `_base.py` lost its sampling and loss machinery. Their
  `residual` methods now delegate to the problem's `strong_residual` and adapt
  the calling convention only, so each equation is written down once. That also
  moved the definition training optimises under the residual-vanishing tests,
  which previously exercised only the wrapper's copy while training used the
  problem's -- the two could have drifted with just one of them covered.

### 3. Expand Problem Coverage Carefully

Why it matters:
The composable architecture needs to prove it handles more than scalar
exact-solution examples.

Concrete work:
- ✅ Add a composable Allen-Cahn problem with constraints and
  reference/benchmark strategy. `AllenCahnProblem` solves
  `u_t - nu u_xx + gamma (u^3 - u) = 0` on a periodic strip and runs through the
  generic `Trainer` with no trainer changes, which was the point: heat and wave
  are linear with separable exact solutions and Burgers is nonlinear but
  Dirichlet, so this is the first problem that is nonlinear *and* periodic.
- ✅ Add exact, numerical or observational reference labels for every new
  benchmark. Problems expose `reference_kind`; Allen-Cahn reports `numerical`
  and its reference is `reference_grid()`, deliberately not named `exact`.
- Add a composable nonlinear Schrodinger problem or document the blocking
  decisions around complex-valued residuals.
- Start Navier-Stokes migration only after multi-output residuals and periodic
  constraints are exercised strongly enough. Periodic constraints are now
  exercised by Allen-Cahn; multi-output residuals are still not.

Two things learned while doing this, both worth keeping:

`PeriodicBoundary` matched values only, which is not periodicity. A field can
match at both faces and still have a kink at the seam, and nothing objects: no
collocation point straddles `x = xmax`, so the residual never sees it. The
constraint grew a `derivative_order` field, and Allen-Cahn imposes both. The
test that pins this down uses `u = x^2`, which matches by value at `x = +-1`
while its slope there is `-2` and `+2`.

The benchmark initial condition `x^2 cos(pi x)` is continuous around the circle
but its derivative is not, so the derivative constraint contradicts the initial
condition exactly at `t = 0`. The two constraints sample disjoint faces and meet
only at two corner points, which random sampling does not hit. This is a
property of the standard benchmark, not of the formulation, and it is documented
on the class rather than quietly worked around.

Follow-up recorded while doing this:
- `notebooks/basic/02_allen_cahn.ipynb` predates the composable path. It
  hand-rolls the residual with raw `torch.autograd.grad` calls and uses
  `D = 5e-3` rather than the benchmark parameters, so it now duplicates
  `AllenCahnProblem` in a form that cannot be tested. Migrating it would make
  the notebook a demonstration of the composable path instead of a competitor
  to it, and would let it score against `reference_grid` rather than plotting
  and stopping there.

Done when:
- ✅ At least one additional PDE family runs through the generic trainer
  with meaningful tests and documented reference data.

### 4. Unify Sampling And Adaptive Strategies

Why it matters:
Sampling is now split across `pinnlab.sampling`, `pinnlab.utils.sampling` and solver
modules. `v0.3.0` introduced the bridge; the next step is reducing the need for
bridges.

Concrete work:
- Migrate importance sampling candidates to use composable geometry where
  possible.
- Add adapter examples showing old `BaseSampler` consumers and new
  `InteriorSampler` policies working together.
- Decide whether `pinnlab.utils.sampling.Domain` remains as a compatibility type
  or becomes a wrapper around `pinnlab.geometry`.
- Extend RAR benchmark comparisons beyond Burgers, heat and wave after new
  composable problem adapters exist.

Done when:
- New sampling features target `InteriorSampler` and `Geometry` first.
- Legacy `BaseSampler` compatibility remains available but is no longer the
  default path for new code.

### 5. Improve Accuracy And Scientific Rigor

Why it matters:
PINN examples are easy to make visually plausible but scientifically weak.
Benchmarks should remain explicit about reference data and evaluation points.

Concrete work:
- Make the Burgers inverse-problem example identifiable with PDE-consistent
  measurements.
- ✅ Demonstrate causal weighting on a problem where it measurably helps.
  Allen-Cahn is that problem, because its trivial solution `u = 0` satisfies
  the PDE and both periodic constraints exactly and plain Adam falls into it,
  which is a causality failure in the precise sense the weighting addresses.
  At `nu = 1e-2`, three seeds each, identical budgets: median relative L2 of
  `0.917` unweighted (range `0.913-0.922`) against `0.615` weighted (range
  `0.600-0.688`). The ranges are disjoint, which the earlier wave-equation
  comparison never managed. Mean `|u|` over the domain went from `0.07` to
  `0.32-0.44` against a reference of `0.578`, so the mechanism is visible
  directly and not only in the score. This is not "Allen-Cahn is solved" --
  `0.6` relative error is still far from accurate at that budget.
- Keep training points and evaluation points distinct in benchmarks.
- Add benchmark metadata for seeds, dtype, sampler, optimizer budget and
  reference provenance.
- Add stronger regression tests for relative error and residual evaluation.

Done when:
- Accuracy claims in examples are supported by reference solutions or clearly
  labeled numerical/observational references.

### 6. Make Architectures Practical

Why it matters:
Architecture classes are only useful if users know when they help and if the
tests cover their expected behavior.

Concrete work:
- Add worked examples for `SirenPINN`, `FourierFeaturePINN` and
  `MultiScalePINN` on problems that justify those architectures.
- Compare architecture choices using the benchmark report schema rather than
  one-off print statements.
- Document initialization and dtype caveats for derivative-heavy PDE residuals.

Done when:
- At least one benchmark or example demonstrates a clear architecture tradeoff
  using the shared reporting format.

### 7. Harden Quality Gates

Why it matters:
The project has enough surface area now that soft quality checks will keep
allowing drift.

Concrete work:
- Fix Sphinx warning debt so documentation can eventually build with warnings
  as errors.
- Burn down the mypy backlog. The type check is now **blocking** in CI. The
  modules that still carry errors are listed explicitly, one per line, under
  `[[tool.mypy.overrides]]` in `pyproject.toml`; everything else must stay
  clean, so new code cannot introduce type errors and a cleaned module cannot
  regress. The list only shrinks: removing an entry means fixing that module in
  the same commit. It is deliberately not a wildcard, which would silently
  exempt new files as well.

  Started at 27 modules and 93 errors on 2026-08-11; 21 remain. Three of the
  first five fixes were real defects rather than annotation noise: an
  unguarded `Optional` dereference of `.grad` in adversarial training, a
  `len()` call on a declared `Iterable` in meta-learning that crashes on a
  generator after doing all the inner training work, and a base class reading
  an attribute only its subclasses define.

  `solvers.raissi_improved`, the heaviest at 15 errors, was worth the whole
  exercise on its own. One of its errors was a latent crash — a smoothness
  term initialised to `0.0` and only promoted to a tensor inside a loop that a
  one-stage tableau skips, so `.item()` hit a bare float. Chasing it revealed
  that `DiscreteRKPINN.train_one_step` had never worked at all:
  `_compute_derivatives` detached its coordinate tensor *after* the prediction
  was computed from it, so autograd could not find it in the graph and every
  call raised. The class was in the public API with only an importability
  test, which is how a solver that could not take a step went unnoticed. Both
  are fixed and covered by `tests/unit/test_discrete_rk_solver.py`.

  `training.adaptive_weighting` (9), `models.multiscale` (8) and
  `utils.visualization` (7) followed, and were annotation debt rather than
  defects. Two causes dominated. `nn.Module.__getattr__` is typed
  `Tensor | Module`, so every `register_buffer` attribute reads back as that
  union until it is declared at class level. And several signatures described
  a narrower contract than the function had always honoured -- `**kwargs: dict`
  where callers pass floats, `Sequence[float]` on a helper whose first
  statement is `np.asarray`, `Dict[str, Tensor]` on a `state_dict` that has
  always held a nested dict and a list.

  `utils.metrics` and `sampling.active_learning` followed, and active learning
  turned up three more real problems. Its acquisition kwargs catch-all was
  literally named `_`, which the throwaway in `_, std = ...` immediately
  overwrote. `MultiObjectiveActiveLearning` never checked that `weights`
  matched `objectives`, so a short list silently dropped objectives through
  `zip` and an empty one made the weighted total a bare `int`. And a plain
  callable passed without `custom` or `ensemble` fell through to Monte Carlo
  dropout, which calls `model.train()`, failing as an `AttributeError` inside
  the uncertainty helper rather than saying what was wrong.

  16 modules and 36 errors remain, measured rather than estimated. The heaviest
  is `solvers.raissi` at five, then a long tail of three or fewer. The work is
  spread thin rather than concentrated, so the block can plausibly be emptied
  and deleted outright.
- Decide whether full flake8 should cover legacy solver modules or whether
  per-file ignores should document known debt explicitly.
- Raise `--cov-fail-under` only after meaningful tests land, not as a cosmetic
  change.
- Add CI jobs that execute examples and selected notebooks.
- Keep tests pointed at production implementations, not local test doubles that
  shadow imports.

Done when:
- Type checking can move from `continue-on-error: true` to blocking for at
  least the composable core packages.
- Documentation warning count trends down and is tracked.

---

## Future Work

These are not immediate post-0.3 priorities, but they remain part of the
long-term direction.

Solver and problem families:
- 3-D Navier-Stokes
- Korteweg-de Vries
- Reaction-diffusion systems
- Elasticity
- Hard-constraint boundary conditions through explicit model transforms

Numerical methods and decomposition:
- VPINNs
- Domain decomposition
- Conservation-aware formulations
- Stronger classical numerical baselines

Operator learning:
- DeepONet baselines
- Fourier Neural Operator baselines
- Shared reporting for PINN versus operator-learning comparisons

Performance:
- Validated multi-GPU and multi-node runs
- `torch.compile` integration
- GPU benchmark suite
- Memory and throughput reporting in benchmark artifacts

Uncertainty and inverse problems:
- Calibration metrics
- Conformal prediction
- Stronger inverse-problem examples with identifiable parameters
- Bayesian and ensemble benchmark coverage

Packaging and distribution:
- PyPI publication
- Versioned hosted documentation
- Automated release checklist for metadata, artifacts and archive contents

---

## Release Checklist

Use this checklist for the next release.

Before tagging:
- Update `CHANGELOG.md`.
- Bump `pyproject.toml`.
- Update fallback `__version__` in `src/pinnlab/__init__.py`.
- Update `CITATION.cff`, `.zenodo.json` and README citation metadata.
- Search for stale version strings.
- Run the full test suite.
- Build Sphinx docs and record warning count.
- Build artifacts with `python -m build`.
- Run `twine check dist/*`.
- Inspect source and wheel artifacts for accidental generated files such as
  `__pycache__`, `.pyc`, notebook checkpoints or local benchmark outputs.

After tagging:
- Push the release commit.
- Create an annotated tag.
- Push the tag.
- Create the GitHub Release and attach wheel/source artifacts.
- Decide explicitly whether PyPI upload is in scope for that release.

PyPI publishing:
- The distribution is `pinnlab`. `pinn` is taken by an unrelated REST-API
  client, and `pinnkit` was rejected as too similar to the existing `pinn-kit`,
  because PyPI strips non-alphanumeric characters before comparing names.
- A trusted publisher is registered on pypi.org for project `pinnlab`, owner
  `DiogoRibeiro7`, repository `pinn`, workflow `publish.yml`, with no
  environment set. The environment field is optional; `publish.yml` therefore
  declares no GitHub environment either. Adding one later would allow gating a
  publish behind required reviewers, and would need adding on PyPI to match.
- Publishing runs on a `v*.*.*` tag. A manual `workflow_dispatch` run builds and
  runs every check without uploading, which is how to rehearse. TestPyPI is not
  used.
- Before any upload the workflow runs `twine check --strict`, asserts the tag
  matches the version in `pyproject.toml`, installs the wheel into an isolated
  environment and imports it, and fails on stray files in the artifacts.
- The first upload claims the name permanently.

---

## How To Contribute

Good first contributions:
- Add focused tests for a module with weak coverage.
- Convert a legacy helper into a shared implementation while preserving old
  imports.
- Add an exact-solution-validated PDE example.
- Add a benchmark report for an existing example.
- Improve documentation where the migration boundary is unclear.

For larger work, open an issue first and describe:
- the user-facing API being added or changed
- how legacy imports remain compatible
- what reference data or benchmark validates the behavior
- which tests will prove the change is real

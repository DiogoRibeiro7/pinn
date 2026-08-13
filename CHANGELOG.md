# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/).

Releases are cut manually: tag the commit (`git tag -a vX.Y.Z -m "vX.Y.Z"`), push the
tag, publish the GitHub release, and — if desired — build and upload with
`python -m build && twine upload dist/*`.

## [Unreleased]

### Fixed

- **SciPy was never detected, even when installed.** `utils/metrics.py`
  imported `wasserstein_distance` from `scipy.spatial.distance`, where it has
  never lived -- it is in `scipy.stats`. The import raised `ImportError` on
  every SciPy version, so `SCIPY_AVAILABLE` was permanently `False`,
  `compute_distribution_distance` and `compute_divergence` always returned
  `nan`, and the module warned that SciPy was missing while it was installed
  and importable.

  Nothing caught it because the failure was indistinguishable from the
  supported one: a user genuinely without SciPy sees the same warning and the
  same `nan`. Found by chasing a warning that appeared in a notebook's output
  after SciPy had been installed. `tests/unit/test_metrics_scipy_gate.py` now
  asserts that importable SciPy means detected SciPy, and that each distance
  responds to its data rather than merely returning a finite number.

- `notebooks/basic/05_heat_equation.ipynb` rendered no figures at all. It was
  the only notebook calling `matplotlib.use("Agg")`, which forces a
  non-interactive backend, so every `plt.show()` warned and displayed nothing --
  in a gallery notebook whose section 4 is "Where the error lives". Removed; it
  now renders like the other nineteen.

- The accuracy claim in `notebooks/basic/03_custom_pde.ipynb` was wrong by more
  than a factor of ten. It said "a relative L2 error around 1e-3 confirms the
  template is sound". Measured over five seeds at the notebook's own
  configuration: median 1.3e-2, spanning 4.1e-3 to 2.1e-2, a 5.2x spread. No
  seed reached 1e-3; the best of five was 4x above it. Anyone reproducing the
  notebook would have seen roughly ten times the documented error and
  reasonably suspected their own setup.

### Changed

- All 20 notebook outputs regenerated against 0.5.0. They had last been
  executed under 0.1.0, so the gallery rendered a setup cell printing `pinnlab`
  above saved output reading `pinn version : 0.1.0`.

- Remaining `pinn` references in notebook prose updated to `pinnlab`: 18
  footers, one heading, three body mentions and the gallery README. References
  where `pinn` names a local variable holding a solver are correct and were
  left alone.

- `notebooks/README.md` states measured runtimes instead of claiming every
  notebook finishes "in a minute or two". The median is about 75 seconds, but
  `basic/04_navier_stokes_2d` takes roughly ten minutes.

- `notebooks/basic/05_heat_equation.ipynb` gained the gallery's header, badges,
  table of contents and footer. It was the only notebook missing all four.

### Added

- **`notebooks/basic/06_composable_api.ipynb`.** Only three of nineteen
  notebooks touched the composable API and none used it as their primary path,
  so the architecture 0.5.0 is built around was invisible in the gallery. It
  splits Allen-Cahn into a problem that knows the equation and a trainer that
  does not, then uses the trivial-solution trap to show a strategy being
  swapped by changing one config field with the problem object untouched.

- **`scripts/smoke_notebooks.py` and `.github/workflows/notebooks.yml`.** A full
  pass executes all 20 in 29 minutes, so CI runs it weekly, on demand, and on
  pull requests that touch notebooks, rather than per push. Execution and
  regeneration are separate: `--write` saves outputs, and without it nothing is
  written, because a very large diff should not be a side effect of a check.

## [0.5.0] - 2026-08-11

### Changed

- **`setup.py` is removed.** It duplicated the packaging metadata and had
  drifted badly from it: `torch>=1.9.0` against pyproject's `>=2.13.0`,
  `numpy>=1.20.0` against `>=2.2.6`, and seaborn still listed as required. With
  `build-backend = "setuptools.build_meta"` and full `[project]` metadata,
  setuptools reads pyproject and ignores it, so it built nothing and existed
  only to mislead anyone reading it for the dependency floors. Confirmed by
  inspecting the built wheel's `METADATA`, which carries the pyproject values.
  Nothing in the repository referenced it.

- **`seaborn` moved from the required dependencies to the `viz` extra.**
  `pinnlab.utils.visualization` called `sns.set_palette` at module scope and
  `pinnlab.utils` imports that module eagerly, so `import pinnlab` imported
  seaborn and through it ipywidgets and IPython. Measured with
  `python -X importtime`, that was 2.3s of a 9.0s import for a single palette
  call. The palette is applied on first plot instead, and without seaborn
  installed the colours differ and nothing else does.

  Measured end to end with a warm cache: an example at a token budget averages
  8.17s before against 6.14s after, about 2.0s per process. Raw `import
  pinnlab` timings on the same machine swing between 5.6s and 15.3s from disk
  caching alone, so the structural claim is the reliable one -- those three
  modules are no longer in the import graph, which a subprocess test pins.

  `matplotlib` remains required: it is still imported at module scope by a
  module `pinnlab.utils` imports eagerly.

- The documentation builds with `sphinx-build -W`, at zero warnings, down from
  64. All 64 had one cause, and it was not the one the count suggested: any
  class with a numpydoc `Attributes` section had each attribute described twice
  under the same name, once by napoleon and once by autodoc's
  `:undoc-members:`. `napoleon_use_ivar = True` renders them as `:ivar:` fields
  instead. The API reference was restructured in the same pass and now
  documents all twenty subpackages, each once at its own path, rather than the
  top-level package plus five modules.

- The coverage gate rose from 58% to 64%, after the Allen-Cahn and discrete-RK
  suites added 63 tests. Measured before moving it: 65.93% in CI, 66.50%
  locally, with CI reporting an identical figure on all six matrix cells.


- The CI type check is blocking, with no exemptions. It started as a list of
  27 modules and 93 errors written out explicitly in `pyproject.toml` so a
  wildcard could not silently exempt new files; that list is now empty and the
  block is deleted. Verified by injecting a type error into a
  previously-exempt module and confirming CI fails.

  The exercise was worth more for what it found than for the clean run. Two
  latent crashes, both listed under Fixed below: `DiscreteRKPINN.train_one_step`
  had never worked, and `PINNConfig.validate()` raised on the default config.

- Accuracy figures are reported as a median over seeds rather than a single
  run. Measured over five seeds at the notebook's budget, the single-mode heat
  problem has a median relative L2 error of 7.3e-4 spanning 3.6e-4 to 9.6e-4,
  and the two-mode case a median of 6.0e-3 spanning 3.6e-3 to 8.6e-3. The
  previously published 3.5e-4 and 2.3e-3 were both better than the best of
  those five seeds, so anyone rerunning the notebook would have seen roughly
  double the documented error and reasonably suspected their own setup.
  `notebooks/basic/05_heat_equation.ipynb` now states the median and range
  before printing its own figure, and its outputs were regenerated against the
  delegated training path.
- Causal weighting is now demonstrated on a problem where it measurably helps,
  which had been an open question in its own docstring. The wave-equation
  comparison it used to cite -- 4.3e-2 with weighting against 4.0e-2 without --
  was two single runs differing by 7% on a problem whose run-to-run spread is
  about 2.5x, so it carried no information either way. Heat and wave were the
  wrong problems: the causality violation does not bite there.

  Allen-Cahn does bite, because `u = 0` satisfies its PDE and both periodic
  constraints exactly and plain Adam falls into it. At `nu = 1e-2`, four seeds
  each, identical budgets: median relative L2 of 0.920 unweighted (range
  0.913-0.940) against 0.651 weighted (range 0.600-0.791). The ranges are
  disjoint -- the worst weighted run beats the best unweighted one -- which the
  wave comparison never managed. The fourth seed was added when
  `notebooks/basic/06_composable_api.ipynb` drew it and landed outside the
  original three-seed range, which is what a too-narrow range looks like. Mean `|u|` over the domain
  rose from 0.07 to 0.32-0.44 against a reference of 0.578, so the mechanism is
  visible directly and not only in the score. This is not "Allen-Cahn is
  solved": 0.6 relative error is still far from accurate at that budget.

- `HeatPINN.residual` and `WavePINN.residual` delegate to their composable
  problem instead of carrying a second implementation of the same equation, so
  each PDE is written down once. This also puts the definition training
  optimises under the residual-vanishing tests, which previously exercised only
  the wrapper's copy. Gradients flow to the coordinates the problem
  differentiates rather than to the `t` and `x` passed in, so use the returned
  values rather than backpropagating into the arguments.
- `HeatPINN` and `WavePINN` train through the composable `Trainer` and their
  `pinnlab.problems` equivalent instead of carrying their own optimisation
  loop. Their public API is unchanged: the same `TrainConfig` goes in and the
  same history dictionaries come out, with the problem's separate `bc_left`
  and `bc_right` losses summed back into one `bc` entry and the wave
  equation's `initial_velocity` folded into `ic`. A wrapper built on a domain
  the composable problem cannot express is now rejected rather than trained
  against a different region than it reports.

  Accuracy is unchanged within measurement noise. Over three seeds at the
  budget the notebook uses, the medians are 4.28e-4 for the old loop and
  6.58e-4 for the new path, against a within-path spread of 2.2x to 2.4x; the
  new path is ahead on one of the three seeds. Single-run comparisons on this
  problem cannot resolve a 2x difference, so the accuracy figures quoted in
  the README and notebook, which are single runs, should be read as favourable
  draws rather than typical results.

### Fixed

- **`DiscreteRKPINN.train_one_step` had never worked.** `_compute_derivatives`
  detached its coordinate tensor *after* the stage prediction was computed from
  it, so autograd was asked for a gradient with respect to a tensor its graph
  had never seen and every call raised `RuntimeError`, for every tableau. The
  class is part of the public API and its only test checked that it imported,
  which is how a solver that could not take a single step went unnoticed.
  Fixing it exposed a second crash in the same function: the smoothness term
  was initialised to `0.0` and only became a tensor inside a loop that a
  one-stage tableau -- forward Euler is a legitimate one -- skips entirely, so
  `.item()` met a bare float. Both are covered by
  `tests/unit/test_discrete_rk_solver.py`.

- **`PINNConfig.validate()` raised `AttributeError` on the default config.**
  `OptimizerConfig`, `SamplingConfig` and `LossConfig` never inherited
  `BaseConfig`, the only three config classes in the module that did not, so
  the public `validate()` wrapper did not exist on them. Each already defined
  `_validate_implementation`, meaning all three had complete validation logic
  that nothing could reach. Negative learning rates, out-of-range betas,
  negative collocation counts and negative loss weights are now rejected with
  `ConfigError` instead of accepted in silence.

- `MultiObjectiveActiveLearning` never checked `weights` against `objectives`.
  `zip` truncates, so a short weights list silently dropped objectives, and an
  empty one made the weighted total a bare `int` for a function declared to
  return a `Tensor`. Both are now rejected with a message naming both lengths.

- A plain callable passed to the active-learning uncertainty helper without
  `custom` or `ensemble` fell through to Monte Carlo dropout, which calls
  `model.train()`. That failed as an `AttributeError` inside the uncertainty
  helper; it now says what is actually missing.

- 18 of 19 notebooks raised `NameError` in their setup cell. The `pinn` ->
  `pinnlab` rename updated the import and left the uses behind, so
  `pinn.__version__` was undefined; `logging.getLogger("pinn")` failed more
  quietly, simply ceasing to suppress library logs. Their stored outputs still
  read version 0.1.0, which is the tell -- they were last executed before the
  rename. Notebooks are not in `MANIFEST.in`, so no distribution ever shipped
  them broken.

  All 19 have since been executed end to end and all 19 pass, so the rename was
  the only thing wrong with them. Their *stored* outputs are still the stale
  0.1.0 ones: the source now prints `pinnlab` and the saved output below it
  still says `pinn version : 0.1.0`. Regenerating them is a separate job and is
  recorded in ROADMAP.md.

- Adversarial training no longer crashes when no gradient reaches its input.
  `adversarial_training_step` dereferenced `x_adv.grad.sign()` where `.grad` is
  `Optional`, so an input the loss did not depend on raised `AttributeError`
  mid-training; it now raises a message that says what went wrong.
- `meta_learning_step` accepts a generator. It declared `task_batch` as an
  `Iterable` and then called `len()` on it, which raises `TypeError` for any
  iterator -- after every inner training loop had already run. The batch is now
  counted while iterating, and an empty batch leaves the model unchanged
  instead of dividing by zero.

- L-BFGS is far more effective in the composable `Trainer`. Its closure
  resampled collocation and constraint points on every evaluation, so the
  strong Wolfe line search compared the objective at several trial step sizes
  while that objective moved underneath it. Reseeding the generator inside the
  closure pins the draw for the duration of the search. Measured on the heat
  problem from identical initial weights at 1500 Adam steps plus 100 L-BFGS
  iterations, the relative L2 error improves from 7.39e-3 to 1.17e-3, and the
  benefit L-BFGS delivers rises from 2.3x to 14.4x. Adam still resamples every
  step, which is intended; only the line search needs a fixed objective.

### Added

- **`AllenCahnProblem`**, the first composable problem that is both nonlinear
  and periodic. Heat and wave are linear with separable exact solutions and
  Burgers is nonlinear but Dirichlet, so nothing before it forced the
  composable core to handle this combination. It runs through the existing
  `Trainer` with no trainer changes, and inherits causal weighting,
  residual-adaptive refinement and Latin hypercube sampling untouched.

  Allen-Cahn has a trap worth knowing before using it: `u = 0` satisfies the
  PDE *and* both periodic constraints exactly, so only the initial condition
  objects to it, and only on the `t = 0` face. Plain Adam falls into that basin
  and scores a relative L2 near 1.0 while three of its four loss components
  look healthy. This is documented on the class and asserted by a test.

- `PeriodicBoundary` gained `derivative_order`. Matching values alone is not
  periodicity: it admits a field with a kink at the seam, which the residual
  never sees because no collocation point straddles the far face. The test that
  pins this down uses `u = x^2`, which matches by value at `x = +-1` while its
  slope there is `-2` and `+2`. Defaults to `0`, the previous behaviour.

- `allen_cahn_spectral` and `allen_cahn_reference_grid` in
  `pinnlab.solvers.reference`. Allen-Cahn has no closed form, so unlike the
  Cole-Hopf Burgers solution this reference is *numerical*: Fourier spectral in
  space, integrating-factor RK4 in time, 2/3-rule dealiasing. It is exposed as
  `reference_grid`, deliberately not `exact`, and problems now report
  `reference_kind` -- `"analytic"` or `"numerical"`.

  Verified against the two limits where the answer is known: with `gamma = 0`
  it reproduces the heat equation to 1.8e-15, with `nu = 0` the logistic ODE to
  5e-11, and halving the step divides the error by 16.2 as fourth order
  requires. Its resolution requirement is set by the interface width
  `sqrt(nu / gamma)`, about 0.0045 at the benchmark parameters, so below
  `n_x = 512` the phases are unresolved and the solution overshoots `|u| = 1`.

- `scripts/smoke_examples.py` runs all sixteen examples end to end, and CI runs
  it on every push. Every example now takes an `--adam-steps` or `--steps` flag
  defaulting to its previously hardcoded value, which took a full pass from
  about ten minutes to under three. Passing means an example runs, not that it
  converges.

- `FixedInteriorSampler` draws interior collocation points once and reuses
  them, which is what the solvers in `pinnlab.solvers` do and what the original
  Raissi implementations do. It is offered for reproducing fixed-point results,
  not as an improvement: measured on the heat problem from identical initial
  weights it is worse than the default resampling, 1.69e-2 against 7.53e-3
  under Adam and 1.17e-3 against 7.72e-4 with L-BFGS. The determinism L-BFGS
  needs comes from the trainer reseeding inside its closure, not from this
  sampler.

## [0.4.0] - 2026-08-10

### Fixed

- `import pinnlab` failed on a clean install. `utils/error_handling.py`
  imported `psutil` unguarded while four sibling modules already treated it as
  optional, so a `pip install` with only the declared dependencies raised
  `ModuleNotFoundError: No module named 'psutil'` before any user code ran.
  The import is now guarded to match, and its single use site checks for it.
- `matplotlib` and `seaborn` are declared as required dependencies. They were
  listed only in the `viz` extra, but `utils/__init__.py` imports the
  visualisation module eagerly, so the package could not be imported without
  them and the extra was advertising a configuration that does not work. The
  `viz` extra now holds `plotly` and `scipy`, which genuinely degrade with a
  warning when absent. Guarding the visualisation imports instead, and
  returning matplotlib to the extra, remains open.

Both were found by rehearsing the publish workflow rather than by the test
suite, which had never exercised an install limited to the declared
dependencies. `tests/unit/test_packaging.py` now does: it imports the package
with each optional dependency blocked, and with all of them blocked at once.

### Changed

- **The distribution and import package are now `pinnlab`.** `pinn` on PyPI is
  taken by an unrelated REST-API client, and that package occupies the
  `import pinn` name too, so publishing under a different distribution name
  while keeping `import pinn` would have broken any environment holding both.
  Install with `pip install pinnlab` and use `import pinnlab`; the console
  script is `pinnlab`. The repository, its URLs and the Zenodo DOI are
  unchanged. There is deliberately no `pinn` compatibility shim, since
  providing one would re-occupy the name this rename exists to avoid.

### Added

- `.github/workflows/publish.yml` publishes to PyPI on a version tag using
  trusted publishing, so no API token is stored in the repository. Before any
  upload it builds, runs `twine check --strict`, verifies the tag matches the
  version in `pyproject.toml`, installs the wheel into an isolated environment
  and imports it, and rejects artifacts containing caches or benchmark
  leftovers. A manual `workflow_dispatch` run performs every check without
  uploading, which is how to rehearse a release.

- Callback hooks for the composable trainer. `Trainer.train` accepts
  `callbacks=[...]`, and `TrainerCallback` exposes `on_train_begin`,
  `on_state_recorded`, `on_phase_end` and `on_train_end`, so logging,
  checkpointing and custom diagnostics can observe a run without editing the
  training loop or post-processing its history. `LoggingCallback` and
  `HistoryCollector` ship as usable implementations. Callback exceptions
  propagate deliberately: a checkpoint that silently fails to write is worse
  than a run that stops.
- `TrainingState.phase` records which optimizer produced a state (`"adam"`,
  `"lbfgs"`, or `"initial"` when no optimizer steps were requested), and
  `TrainingResult.states_for_phase` filters on it, making the Adam to L-BFGS
  transition inspectable rather than implicit in step numbering.
- Documented the responsibilities of every `TrainingState` and `TrainingResult`
  field, including why diagnostics are held apart from named losses.
- `profile_performance` accepts an explicit `name`, so published benchmark
  metrics are not named after a nested function's `__qualname__`.

### Changed

- `Trainer.train` evaluates its losses through a single `_evaluate` helper. The
  sample-evaluate-combine sequence had been repeated at four call sites (the
  Adam loop, the L-BFGS closure, the post-L-BFGS record and the no-step
  fallback), so refinement or sampling changes had to be made four times to
  stay consistent.

### Fixed

- The Performance workflow ran `pytest tests/performance`, which inherited the
  repository-wide `--cov-fail-under` from `addopts`. Measured over that subset
  coverage is about 29%, so the job failed a coverage gate rather than on the
  benchmarks, and never reached them.
- `scripts/benchmark_distributed.py` passed `ConfigFactory.create_burgers_config()`
  (a generic `PINNConfig`) to `ContinuousPINN`, which validates for
  `BurgersConfig`, so it raised before running a step.
- The Performance workflow emitted benchmark results as a
  `{"benchmarks": [...]}` object, but `customSmallerIsBetter` requires a bare
  array and rejected it. With these three fixed the workflow completes and
  publishes to the benchmark dashboard for the first time.
- `[tool.mypy] python_version` was `3.10`, which makes mypy abort on a PEP 695
  `type` statement in numpy's own stubs and check *zero* source files. The
  type-check step in CI was reporting one stub parse error and nothing about
  this codebase. With a 3.12 target it checks 89 files and reports 88 errors
  across 25 modules; `ROADMAP.md` now carries that figure.
- `benchmark_report.json` and `benchmark.json`, written into the working
  directory by the benchmark scripts, are gitignored.

## [0.3.0] - 2026-08-09

### Added

- Composable PINN core abstractions for problems, geometry, constraints,
  residuals and equation-agnostic training, with migrated heat, wave and
  Burgers problem adapters.
- Physical nondimensionalization and dtype-preserving scaling support for the
  composable trainer.
- Residual-adaptive refinement (RAR) in the composable trainer, including
  diversity-aware candidate selection and recorded diagnostics.
- Benchmark reporting primitives, an explicit finite-difference heat reference
  solver and scripted RAR comparisons for Burgers, heat and wave problems.
- SIREN network support for oscillatory and derivative-heavy PDE residuals.
- Composable sampling policies, including configurable interior samplers,
  Latin-hypercube interior sampling and adapters between the new
  `InteriorSampler` protocol and legacy `BaseSampler` consumers.

### Changed

- Legacy solver `latin_hypercube` names now re-export the shared
  `pinnlab.sampling.latin_hypercube` helper, keeping old import paths working while
  avoiding solver-local implementation drift.
- Development and runtime dependency floors were refreshed by Dependabot.
- Architecture, core API, performance and roadmap documentation now describe
  the composable trainer, benchmark provenance, RAR workflows and sampling
  migration path.

## [0.2.0] - 2026-08-08

### Added

- Exactly-solvable solvers, so accuracy is a number rather than a plot.
  `HeatPINN` is scored against the Fourier-series solution of the 1-D heat
  equation and `WavePINN` against the standing-wave solution of the 1-D wave
  equation, both through `relative_l2_error`. With the budgets used in the
  examples they reach 1.7e-3 and 4.0e-2 respectively; a two-mode heat initial
  condition, which is stiffer, reaches 2.3e-3.
- `pinnlab.solvers.reference.burgers_cole_hopf`: the exact solution of the viscous
  Burgers equation via the Cole-Hopf transform, so the existing Burgers solvers
  can be scored. Cross-checked against an independent explicit finite-difference
  solve, agreeing to 1.9e-5 relative L2 at `nu = 0.05`.
- `pinnlab.training.causal_weighting`: temporal causality weighting from Wang,
  Sankaran and Perdikaris (2022). The weighting formula is unit-tested, but no
  accuracy benefit has been demonstrated on the problems in this package -- see
  the module docstring for the measurements and `ROADMAP.md` for the follow-up.
- Examples `basic/heat_equation.py`, `basic/wave_equation.py` and
  `advanced/causal_weighting.py`, the last reporting both errors rather than
  assuming the weighted one wins.
- Notebook `basic/05_heat_equation.ipynb`, executed, covering scoring against a
  closed-form solution and the stiffer two-mode variant.

### Fixed

- `raissi_generic.latin_hypercube` was broken for `d > 1`: it left the interval
  bounds at shape `(n,)` rather than `(n, 1)`, so the broadcast against the
  `(n, d)` sample failed. `ContinuousPINNGeneric._make_data` calls it with
  `d = 2`, so the generic solver could not draw a single collocation point and
  the Allen-Cahn and Schrodinger demos raised before training began -- two of
  the four advertised PDE families were unusable. The module now imports the
  working implementation from `raissi_improved` instead of keeping a second copy.
- `examples/basic/burgers_equation.py` measured its error against
  `-sin(pi x) exp(-nu pi^2 t)`, which is the solution of the *heat* equation: it
  drops the nonlinear advection term and never forms a shock, so every metric it
  reported was meaningless. It now uses the Cole-Hopf solution and reports a
  relative L2 error.

## [0.1.1] - 2026-08-08

### Added

- The project governance and contribution files: `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `SECURITY.md` (including a threat model), issue and pull
  request templates, `CODEOWNERS`, `.gitattributes`, `.editorconfig` and a
  `.pre-commit-config.yaml` pinned to the same tool versions as CI.
- The Zenodo concept DOI (`10.5281/zenodo.21844101`) in `CITATION.cff`, the README
  citation block and a README badge.

### Changed

- Rewrote `tests/unit/test_visualization.py` to exercise the real implementations.
  It previously defined local stand-ins for `PINNVisualizer`, `VisualizationConfig`,
  `VisualizationError` and `create_custom_config`, shadowing the imports, so its
  assertions ran against its own test doubles and passed regardless of the
  library's behaviour. Coverage of `pinn/utils/visualization.py` rose from 14% to
  82%, and the save-format defect below is what the new tests caught first.
- `tests/conftest.py` selects the headless matplotlib backend before anything
  imports pyplot, so the plotting tests do not depend on an available display.
- Raised the coverage ratchet from 50% to 55%; measured coverage is now 58%.
- Unified the maintainer contact address on `dfr@esmad.ipp.pt`, matching the ESMAD
  affiliation declared in the citation metadata.

### Fixed

- Figures are saved in the format implied by the file extension. `_save_figure`
  forced `config.save_format` for every path, so a figure saved to `.pdf`, `.svg`
  or `.eps` contained PNG bytes under a misleading extension — despite
  `_validate_save_path` accepting exactly those extensions without warning.
- The README development setup told contributors to install from a
  `requirements-dev.txt` that does not exist.
- The README badges still advertised Python 3.8+ and PyTorch 1.9+; added a CI
  status badge alongside the corrected ones.

## [0.1.0] - 2026-08-07

First tagged release. The library has been in development for some time; this
release marks the point at which it is packaged, tested on every supported
Python version, and citable.

### Added

- `CITATION.cff` and `.zenodo.json`, making the project citable and archivable,
  validated in CI.
- Continuous integration that actually runs: lint, a six-cell test matrix across
  Python 3.10-3.13 on Linux plus macOS and Windows, and documentation publishing.
- Packaging metadata: license, keywords, trove classifiers and project URLs.

### Changed

- Raised the supported Python floor to 3.10 and modernised the dependency floors
  (numpy 2.2, torch 2.13, scipy 1.15, matplotlib 3.10, plotly 6.9, pytest 9,
  ipykernel 7).
- Replaced automated semantic-release publishing with a manual release process.
- Applied `black` across the codebase and adopted a working `flake8`
  configuration; `[tool.flake8]` in `pyproject.toml` had never taken effect,
  because flake8 cannot read that file.
- The default branch is now `main`.

### Fixed

- `ci.yml` was an invalid workflow file: a job-level `if: runner.os == 'Linux'`
  referenced a context that does not exist there, so no CI job had ever run.
- A bare `except:` in `pinnlab.utils.metrics` also caught `KeyboardInterrupt` and
  `SystemExit`.
- The Navier-Stokes example computed the true v-velocity field but never scored
  it; the v-velocity error is now reported alongside u.
- Documentation deployment failed with a 403 for want of `contents: write`.
- `performance.yml` had an inlined heredoc at column zero, which escaped its
  block scalar and left the workflow unparseable.

### Security

- Load checkpoints, cached samples and served models with
  `torch.load(..., weights_only=True)`. Checkpoint restore had explicitly passed
  `weights_only=False`, which unpickles arbitrary objects and allows code
  execution from a malicious checkpoint file. Checkpoints are now written with
  the default pickle protocol rather than protocol 5, because the weights-only
  unpickler cannot read protocol-5 frames; checkpoints written by earlier
  revisions cannot be loaded by this release.
- The serving entry point still binds all interfaces by default, as a container
  requires, but the host is now configurable through `PINN_HOST`.
- Documented the trust boundary on the cache's `pickle` loads, which only ever
  deserialise data the cache itself wrote.

[Unreleased]: https://github.com/DiogoRibeiro7/pinn/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/DiogoRibeiro7/pinn/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/DiogoRibeiro7/pinn/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/DiogoRibeiro7/pinn/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/DiogoRibeiro7/pinn/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/DiogoRibeiro7/pinn/releases/tag/v0.1.0

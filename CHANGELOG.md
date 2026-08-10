# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/).

Releases are cut manually: tag the commit (`git tag -a vX.Y.Z -m "vX.Y.Z"`), push the
tag, publish the GitHub release, and — if desired — build and upload with
`python -m build && twine upload dist/*`.

## [Unreleased]

### Added

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
  `pinn.sampling.latin_hypercube` helper, keeping old import paths working while
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
- `pinn.solvers.reference.burgers_cole_hopf`: the exact solution of the viscous
  Burgers equation via the Cole-Hopf transform, so the existing Burgers solvers
  can be scored. Cross-checked against an independent explicit finite-difference
  solve, agreeing to 1.9e-5 relative L2 at `nu = 0.05`.
- `pinn.training.causal_weighting`: temporal causality weighting from Wang,
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
- A bare `except:` in `pinn.utils.metrics` also caught `KeyboardInterrupt` and
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

[Unreleased]: https://github.com/DiogoRibeiro7/pinn/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/DiogoRibeiro7/pinn/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/DiogoRibeiro7/pinn/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/DiogoRibeiro7/pinn/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/DiogoRibeiro7/pinn/releases/tag/v0.1.0

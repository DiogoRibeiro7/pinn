# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/).

Releases are cut manually: tag the commit (`git tag -a vX.Y.Z -m "vX.Y.Z"`), push the
tag, publish the GitHub release, and — if desired — build and upload with
`python -m build && twine upload dist/*`.

## [Unreleased]

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

- The serving entry point still binds all interfaces by default, as a container
  requires, but the host is now configurable through `PINN_HOST`.
- Documented the trust boundary on the cache's `pickle` loads, which only ever
  deserialise data the cache itself wrote.

[Unreleased]: https://github.com/DiogoRibeiro7/pinn/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/DiogoRibeiro7/pinn/releases/tag/v0.1.0

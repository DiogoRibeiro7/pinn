# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/).

Releases are cut manually: tag the commit (`git tag -a vX.Y.Z -m "vX.Y.Z"`), push the
tag, publish the GitHub release, and — if desired — build and upload with
`python -m build && twine upload dist/*`.

## [Unreleased]

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

[Unreleased]: https://github.com/DiogoRibeiro7/pinn/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/DiogoRibeiro7/pinn/releases/tag/v0.1.0

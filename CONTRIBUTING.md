# Contributing to pinn

Thanks for considering a contribution. This document covers how to get set up, what
the checks expect, and how work gets reviewed.

By participating you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting set up

The project requires **Python 3.10 or newer**.

```bash
git clone https://github.com/DiogoRibeiro7/pinn.git
cd pinn
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev,viz]"
```

Install `viz` as well as `dev`, not just `dev`. Importing `pinn` pulls in
`matplotlib` and `seaborn`, so the package cannot be imported without them.

Optionally, install the pre-commit hooks so the formatting and lint checks run
before each commit rather than in CI:

```bash
pre-commit install
```

## Running the checks

CI runs exactly these. Running them locally first saves a round trip.

```bash
pytest                  # tests, with the coverage gate
black --check .         # formatting
flake8 .                # lint
bandit -r src -ll       # security scan
mypy src                # type check (currently advisory, see below)
```

`black`, `flake8` and `bandit` are **pinned to exact versions** in the `dev` extra.
This is deliberate: their output changes between releases, and an unpinned version
means CI disagrees with a local run that passed. If you upgrade one, upgrade the pin
in `pyproject.toml` in the same commit and reformat as needed.

`mypy` currently reports pre-existing errors and runs with `continue-on-error` in CI.
Please do not add new ones; clearing the backlog is tracked in [ROADMAP.md](ROADMAP.md).

### Tests

```bash
pytest                                   # everything
pytest tests/unit                        # unit tests only
pytest -m "not performance"              # skip benchmarks
pytest tests/unit/test_sampling.py -q    # a single file
```

Coverage is gated by `--cov-fail-under` in `pyproject.toml`, set just below the
current figure so coverage cannot regress. If your change raises coverage
meaningfully, raise the gate with it.

Some property-based tests use Hypothesis. `tests/conftest.py` registers a profile
that disables the timing-based health checks, because they measure the machine
rather than the code.

## Submitting a change

1. Open an issue first for anything substantial, so the approach can be agreed
   before you spend time on it. Small fixes can go straight to a pull request.
2. Branch from `main`. Name the branch for the work: `fix/burgers-residual-sign`,
   `feat/kdv-solver`, `docs/sampling-tutorial`.
3. Keep the commit history readable. One logical change per commit, with a message
   that explains *why* rather than restating the diff.
4. Add tests. A bug fix should come with a test that fails without it.
5. Update `CHANGELOG.md` under `## [Unreleased]` if the change is user-visible.
6. Open the pull request and fill in the template.

Pull requests need CI green and one maintainer approval before merge. Merges are
squashed.

## Adding a new PDE solver

The most common contribution. The path of least resistance:

1. Start from the generic residual framework in `pinnlab.solvers.raissi_generic`
   rather than writing a solver from scratch.
2. Express the PDE residual as a function of the network output and its autograd
   derivatives.
3. Validate against an exact solution wherever one exists. A solver without a
   quantitative accuracy check is hard to trust and harder to review.
4. Add a runnable example under `examples/` and, ideally, a notebook.

## Reporting bugs and security issues

Ordinary bugs go in the [issue tracker](https://github.com/DiogoRibeiro7/pinn/issues).
For anything security-sensitive, follow [SECURITY.md](SECURITY.md) instead of
opening a public issue.

## Licence

Contributions are accepted under the [MIT Licence](LICENSE), the same terms as the
rest of the project.

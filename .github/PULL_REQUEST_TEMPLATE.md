<!--
Thanks for the contribution. Keep this short — the aim is to give a reviewer the
context the diff cannot supply on its own.
-->

## What this changes

<!-- One or two sentences. What is different after this is merged? -->

## Why

<!--
The reasoning behind the change. If it fixes an issue, link it: "Fixes #123".
For a numerical or algorithmic change, say what makes the new behaviour correct.
-->

## How it was verified

<!--
How do you know it works? New tests, an exact solution it now matches, a
convergence plot, a benchmark before/after. "CI is green" is necessary but
rarely sufficient on its own.
-->

## Checklist

- [ ] Tests added or updated, and a bug fix has a test that fails without it
- [ ] `pytest` passes locally
- [ ] `black --check .`, `flake8 .` and `bandit -r src -ll` pass
- [ ] No new `mypy` errors introduced
- [ ] `CHANGELOG.md` updated under `## [Unreleased]`, if user-visible
- [ ] Docstrings and documentation updated, if the public API changed

## Breaking changes

<!--
Anything that changes behaviour for existing users: a renamed argument, a
different default, a checkpoint or serialisation format change. Write "None" if
there are none.
-->

None

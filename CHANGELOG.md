# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/).

Releases are cut manually: tag the commit (`git tag -a vX.Y.Z -m "vX.Y.Z"`), push the
tag, publish the GitHub release, and — if desired — build and upload with
`python -m build && twine upload dist/*`.

## [Unreleased]

- Added `CITATION.cff` and `.zenodo.json` so the project is citable and archivable.
- Raised the supported Python floor to 3.10 and modernised the dependency floors
  (numpy 2.2, torch 2.13, scipy 1.15, matplotlib 3.10, plotly 6.9, pytest 9, ipykernel 7).
- Replaced automated semantic-release publishing with a manual release process.

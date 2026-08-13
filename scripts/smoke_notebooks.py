#!/usr/bin/env python3
"""Execute every notebook and report which ones fail.

Why this exists: 18 of the 19 notebooks in this gallery once raised
``NameError`` in their setup cell, because the ``pinn`` -> ``pinnlab`` rename
updated the import and left the uses behind. Nothing noticed, because nothing
ran them. Their stored outputs still read version 0.1.0 at the time, which was
the only visible clue. That is the same shape as ``DiscreteRKPINN`` sitting in
the public API unable to take a single step: code that is never executed is code
whose breakage nobody sees.

This does not check that a notebook is *correct*, only that every cell runs.
Outputs are not written back -- see ``--write`` for that, which is a separate,
deliberate act because it produces a very large diff.

Cost
----
A full pass takes roughly 40 minutes on one CPU, dominated by
``basic/04_navier_stokes_2d`` (~9 min) and ``basic/06_composable_api`` (~7 min),
both of which train properly rather than for show. That is too slow for every
push, so CI runs this on a schedule and on demand rather than per commit. The
examples job, which is about three minutes, covers the same library code paths
on every push.

Usage
-----
    python scripts/smoke_notebooks.py              # execute all of them
    python scripts/smoke_notebooks.py -k burgers   # substring filter
    python scripts/smoke_notebooks.py --write      # also save regenerated outputs
"""

from __future__ import annotations

import argparse
import glob
import time
import warnings
from pathlib import Path

import nbformat
from nbclient import NotebookClient

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent

# Generous: the slowest notebook is about 9 minutes on a laptop, and a CI runner
# is slower still. A timeout that trips on a healthy notebook is worse than one
# that takes a while to notice a hang.
CELL_TIMEOUT_SECONDS = 1800


def notebook_paths(pattern: str | None) -> list[str]:
    """Return the notebooks to run, relative to the repository root."""
    paths = sorted(
        p.replace("\\", "/")
        for p in glob.glob(
            str(REPO_ROOT / "notebooks" / "**" / "*.ipynb"), recursive=True
        )
        if ".ipynb_checkpoints" not in p
    )
    if pattern:
        paths = [p for p in paths if pattern in p]
    return paths


def run_one(path: str, write: bool) -> tuple[bool, float, str]:
    """Execute one notebook and return (passed, seconds, detail)."""
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=CELL_TIMEOUT_SECONDS,
        kernel_name="python3",
        # Run each notebook from its own directory, so relative paths in it
        # resolve the way they do when a reader opens it.
        resources={"metadata": {"path": str(Path(path).parent)}},
    )
    started = time.perf_counter()
    try:
        client.execute()
    except Exception as exc:  # noqa: BLE001 - every failure mode is worth reporting
        last_line = str(exc).strip().splitlines()[-1] if str(exc).strip() else ""
        return (
            False,
            time.perf_counter() - started,
            f"{type(exc).__name__}: {last_line[:110]}",
        )
    elapsed = time.perf_counter() - started
    if write:
        nbformat.write(notebook, path)
    return True, elapsed, "written" if write else ""


def main() -> int:
    """Execute the selected notebooks and report a per-notebook table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-k", dest="pattern", help="substring filter on the path")
    parser.add_argument(
        "--write",
        action="store_true",
        help="save regenerated outputs for notebooks that execute cleanly",
    )
    args = parser.parse_args()

    paths = notebook_paths(args.pattern)
    if not paths:
        print("no notebooks matched")
        return 1

    print(f"Executing {len(paths)} notebooks\n", flush=True)

    failures: list[str] = []
    total = 0.0
    for path in paths:
        name = path.split("notebooks/")[-1]
        passed, elapsed, detail = run_one(path, args.write)
        total += elapsed
        status = "ok  " if passed else "FAIL"
        suffix = f"  {detail}" if detail else ""
        if not passed:
            # A failed notebook keeps its existing outputs, so a partial run
            # cannot half-update the gallery.
            suffix += "  (not written)" if args.write else ""
            failures.append(name)
        print(f"  {status} {name:46s} {elapsed:7.1f}s{suffix}", flush=True)

    print(f"\ntotal {total / 60:.1f} min")
    if failures:
        print(f"{len(failures)} of {len(paths)} failed: {', '.join(failures)}")
        return 1
    print(f"All {len(paths)} notebooks executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

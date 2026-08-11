#!/usr/bin/env python3
"""Run every example script end to end and report which ones fail.

Why this exists: ``DiscreteRKPINN.train_one_step`` sat in the public API
completely non-functional -- it raised on every call, for any tableau -- and
nothing noticed, because the only test touching that class checked it could be
imported. A type checker eventually found it. Actually running the code would
have found it years earlier and for free.

Budgets
-------
Every example is run at a token budget: the point is to execute each code path
once, not to converge. Fourteen take an explicit ``--adam-steps`` or ``--steps``
flag; ``custom_pde.py`` and ``performance_benchmark.py`` have none but are
already fast at their defaults, and that is marked ``budget=FULL`` in the output
rather than left implicit. The runtimes in ``EXAMPLES`` are real measurements.

Reducing the budget only changes how long the example trains, never what it
does, so every code path still executes. Passing here means the example runs,
not that it converges to anything useful.

Usage
-----
    python scripts/smoke_examples.py              # all of them
    python scripts/smoke_examples.py --fast       # only the budget-capable ones
    python scripts/smoke_examples.py -k burgers   # substring filter
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent

# A generous multiple of the measured runtime, so a slower CI runner does not
# turn a passing example into a spurious failure.
TIMEOUT_FACTOR = 6
MIN_TIMEOUT = 120


@dataclass(frozen=True)
class Example:
    """One example script and how to run it cheaply."""

    path: str
    #: Arguments that shrink the training budget. Empty means the script has no
    #: budget flags and runs at its full default.
    budget_args: List[str] = field(default_factory=list)
    #: The flag this script uses for its output directory. They disagree.
    out_flag: str = "--output-dir"
    #: Measured wall-clock seconds with the arguments above, for the timeout.
    measured_seconds: int = 40

    @property
    def name(self) -> str:
        """Return the script's basename without the extension."""
        return Path(self.path).stem

    @property
    def has_budget_flags(self) -> bool:
        """Return whether this example can be run at a reduced budget."""
        return bool(self.budget_args)


_TINY = ["--adam-steps", "5", "--lbfgs-iter", "2", "--collocation", "64"]

EXAMPLES: tuple[Example, ...] = (
    # Budget-capable: these run in seconds.
    Example("examples/basic/heat_equation.py", _TINY, "--out-dir", 7),
    Example("examples/basic/wave_equation.py", _TINY, "--out-dir", 6),
    Example("examples/advanced/causal_weighting.py", _TINY, "--out-dir", 6),
    Example(
        "examples/basic/allen_cahn.py",
        ["--steps", "5", "--collocation", "64"],
        measured_seconds=9,
    ),
    Example(
        "examples/benchmarks/method_comparison.py",
        ["--adam-steps", "5"],
        measured_seconds=6,
    ),
    Example(
        "examples/benchmarks/active_learning_comparison.py",
        ["--adam-steps", "5"],
        "--output",
        20,
    ),
    Example(
        "examples/benchmarks/multiscale_comparison.py",
        ["--epochs", "5", "--num-points", "64"],
        measured_seconds=20,
    ),
    # Config-driven examples: these take --adam-steps or --steps.
    Example(
        "examples/basic/burgers_equation.py", ["--adam-steps", "5"], measured_seconds=10
    ),
    Example(
        "examples/basic/navier_stokes_2d.py", ["--adam-steps", "5"], measured_seconds=15
    ),
    Example(
        "examples/advanced/inverse_problems.py", ["--steps", "5"], measured_seconds=10
    ),
    Example(
        "examples/advanced/multi_scale_training.py",
        ["--adam-steps", "5"],
        measured_seconds=10,
    ),
    Example(
        "examples/advanced/transfer_learning.py",
        ["--adam-steps", "5"],
        measured_seconds=10,
    ),
    Example(
        "examples/advanced/uncertainty_quantification.py",
        ["--adam-steps", "5"],
        measured_seconds=10,
    ),
    Example(
        "examples/benchmarks/convergence_study.py",
        ["--adam-steps", "5"],
        measured_seconds=10,
    ),
    # These two have no budget flag but are already fast at their defaults:
    # performance_benchmark already uses adam_steps=5, and custom_pde trains a
    # small network briefly.
    Example("examples/basic/custom_pde.py", measured_seconds=23),
    Example("examples/benchmarks/performance_benchmark.py", measured_seconds=10),
)


def run_one(example: Example, out_root: Path) -> tuple[bool, float, str]:
    """Run one example and return (passed, seconds, detail)."""
    out_dir = out_root / example.name
    command = [
        sys.executable,
        str(REPO_ROOT / example.path),
        *example.budget_args,
        example.out_flag,
        str(out_dir),
    ]
    timeout = max(MIN_TIMEOUT, example.measured_seconds * TIMEOUT_FACTOR)
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, cwd=REPO_ROOT
        )
    except subprocess.TimeoutExpired:
        return False, time.perf_counter() - started, f"timed out after {timeout}s"
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout).strip().splitlines()
        detail = tail[-1] if tail else f"exit {completed.returncode}"
        return False, elapsed, detail[:160]
    return True, elapsed, ""


def main() -> int:
    """Run the selected examples and report a per-example result table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="only run examples that support a reduced budget",
    )
    parser.add_argument("-k", dest="pattern", help="substring filter on the name")
    args = parser.parse_args()

    selected = [
        e
        for e in EXAMPLES
        if (not args.fast or e.has_budget_flags)
        and (not args.pattern or args.pattern in e.name)
    ]
    if not selected:
        print("no examples matched")
        return 1

    skipped = len(EXAMPLES) - len(selected)
    print(f"Running {len(selected)} examples ({skipped} not selected)\n", flush=True)

    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        out_root = Path(tmp)
        for example in selected:
            passed, elapsed, detail = run_one(example, out_root)
            budget = "tiny" if example.has_budget_flags else "FULL"
            status = "ok  " if passed else "FAIL"
            print(
                f"  {status} {example.name:34s} {elapsed:6.1f}s  budget={budget}"
                + (f"  {detail}" if detail else ""),
                flush=True,
            )
            if not passed:
                failures.append(example.name)

    print()
    if failures:
        print(f"{len(failures)} of {len(selected)} failed: {', '.join(failures)}")
        return 1
    print(f"All {len(selected)} examples ran.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

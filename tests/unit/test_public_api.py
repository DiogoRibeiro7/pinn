"""Compatibility tests for the public import surface.

Milestone 2 of the post-0.3 hardening stage moves behaviour out of the legacy
solver modules and behind the composable core. That migration is only safe if
the names people import keep working, so this module pins the surface down:
``public_api_snapshot.json`` records every name each package exports today, and
these tests fail if one disappears.

The snapshot is a floor, not a contract freeze. Adding exports is fine and does
not require touching the snapshot; removing or renaming one is a deliberate
break that has to be recorded by regenerating it, which is exactly the moment
to decide whether a deprecation shim is owed.

Regenerate with::

    python -c "import json,importlib; ..."   # see the commit that added this

The legacy names checked here are the ones ROADMAP.md commits to keeping
importable while the internals move.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

SNAPSHOT_PATH = Path(__file__).with_name("public_api_snapshot.json")
SNAPSHOT: dict[str, list[str]] = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))

# Named explicitly in ROADMAP.md as remaining importable through the migration.
LEGACY_SOLVER_CLASSES = [
    ("pinnlab.solvers.heat", "HeatPINN"),
    ("pinnlab.solvers.wave", "WavePINN"),
    ("pinnlab.solvers.raissi_improved", "ContinuousPINN"),
    ("pinnlab.solvers.raissi_improved", "DiscreteRKPINN"),
    ("pinnlab.solvers.raissi_generic", "ContinuousPINNGeneric"),
    ("pinnlab.solvers.navier_stokes", "NavierStokesPINN"),
]

# Legacy module paths that must keep resolving even as behaviour moves.
LEGACY_MODULES = [
    "pinnlab.solvers.raissi",
    "pinnlab.solvers.raissi_improved",
    "pinnlab.solvers.raissi_generic",
    "pinnlab.solvers.navier_stokes",
    "pinnlab.utils.sampling",
]

# Backward-compatibility aliases exposed at the top level.
LEGACY_ALIASES = [
    "pinn_raissi",
    "pinn_raissi_improved",
    "pinn_raissi_generic",
    "pinn_navier_stokes",
]


@pytest.mark.parametrize("module_name", sorted(SNAPSHOT))
def test_no_public_name_disappeared(module_name: str):
    """Every name the snapshot recorded still resolves.

    A failure here means the migration dropped part of the public surface. If
    that was intended, regenerate the snapshot in the same commit so the break
    is visible in review rather than discovered by a user.
    """
    module = importlib.import_module(module_name)
    missing = [name for name in SNAPSHOT[module_name] if not hasattr(module, name)]
    assert not missing, f"{module_name} no longer exports: {sorted(missing)}"


@pytest.mark.parametrize("module_name", sorted(SNAPSHOT))
def test_declared_exports_actually_resolve(module_name: str):
    """``__all__`` must not advertise names the module does not define.

    Distinct from the snapshot check: this catches a name added to ``__all__``
    without the corresponding import, which breaks ``from module import *``
    and star-import tooling while leaving normal imports working.
    """
    module = importlib.import_module(module_name)
    declared = getattr(module, "__all__", [])
    missing = [name for name in declared if not hasattr(module, name)]
    assert not missing, f"{module_name}.__all__ advertises undefined: {missing}"


class TestLegacySolverSurface:
    @pytest.mark.parametrize("module_name, symbol", LEGACY_SOLVER_CLASSES)
    def test_legacy_solver_classes_import(self, module_name: str, symbol: str):
        module = importlib.import_module(module_name)
        obj = getattr(module, symbol, None)
        assert obj is not None, f"{module_name}.{symbol} is missing"
        assert isinstance(obj, type), f"{module_name}.{symbol} is no longer a class"

    @pytest.mark.parametrize("module_name", LEGACY_MODULES)
    def test_legacy_modules_still_import(self, module_name: str):
        assert importlib.import_module(module_name) is not None

    @pytest.mark.parametrize("alias", LEGACY_ALIASES)
    def test_top_level_aliases_still_resolve(self, alias: str):
        import pinnlab

        assert getattr(pinnlab, alias, None) is not None

    def test_solver_classes_are_reachable_from_the_package_root(self):
        """`from pinnlab.solvers import HeatPINN` is the documented path."""
        from pinnlab.solvers import HeatPINN, WavePINN

        assert isinstance(HeatPINN, type)
        assert isinstance(WavePINN, type)


class TestCentralizedHelpers:
    """Helpers that were de-duplicated must stay one object, not copies.

    ``latin_hypercube`` had two implementations and the copy in
    ``raissi_generic`` was broken for more than one dimension, which left the
    Allen-Cahn and Schrodinger solvers unable to sample a single collocation
    point. Re-exporting one function is what keeps that from recurring.
    """

    def test_latin_hypercube_is_a_single_shared_function(self):
        from pinnlab.sampling import latin_hypercube as canonical
        from pinnlab.solvers.raissi_generic import latin_hypercube as generic
        from pinnlab.solvers.raissi_improved import latin_hypercube as improved

        assert generic is improved
        assert generic is canonical

    def test_shared_helper_handles_multiple_dimensions(self):
        from pinnlab.sampling import latin_hypercube

        for dimensions in (1, 2, 5):
            points = latin_hypercube(16, dimensions, 0.0, 1.0)
            assert points.shape == (16, dimensions)


class TestSnapshotIntegrity:
    def test_snapshot_covers_the_top_level_package(self):
        assert "pinnlab" in SNAPSHOT
        assert len(SNAPSHOT["pinnlab"]) > 100

    def test_snapshot_has_no_duplicate_entries(self):
        for module_name, names in SNAPSHOT.items():
            assert len(names) == len(set(names)), f"{module_name} lists duplicates"

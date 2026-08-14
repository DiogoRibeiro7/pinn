"""Tests that the declared dependencies match what importing the package needs.

A distribution can build cleanly, pass ``twine check`` and still be unusable:
if a module imported at package scope needs something that is not declared,
``pip install pinnlab`` succeeds and ``import pinnlab`` then fails. That is
exactly what happened before 0.4.0 -- ``utils/error_handling.py`` imported
``psutil`` unguarded while four sibling modules treated it as optional, and
``utils/__init__.py`` imported the visualisation module eagerly, so matplotlib
was required in practice while being advertised as an optional extra.

These tests fail fast on both shapes of that mistake.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

try:  # tomllib is stdlib from 3.11; this package supports 3.10
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on 3.10
    tomllib = pytest.importorskip(
        "tomli",
        reason="reading pyproject.toml needs tomllib (3.11+) or tomli",
    )

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "src" / "pinnlab"

# Third-party roots the package may touch. Anything here must either be a
# declared runtime dependency or be imported defensively.
OPTIONAL_ROOTS = {
    "matplotlib",
    "psutil",
    "scipy",
    "plotly",
    "seaborn",
    "fastapi",
    "uvicorn",
    "grpc",
    "sklearn",
}


def _declared_runtime_requirements() -> set[str]:
    """Return the distribution names in ``[project] dependencies``."""
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    names = set()
    for spec in data["project"]["dependencies"]:
        name = spec.split(">")[0].split("=")[0].split("[")[0].split(";")[0]
        names.add(name.strip().lower())
    return names


def _module_import_roots(path: Path) -> list[tuple[str, bool]]:
    """Return ``(root_module, guarded)`` for every import in a source file.

    ``guarded`` is True when the import sits inside a ``try`` block, which is
    how this package marks a dependency as optional.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    guarded_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for child in ast.walk(node):
                guarded_nodes.add(id(child))

    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((alias.name.split(".")[0], id(node) in guarded_nodes))
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.module.split(".")[0], id(node) in guarded_nodes))
    return found


class TestOptionalDependenciesAreGuarded:
    """A module reached at import time must not need an undeclared package.

    Deliberately no blanket "every optional import must sit in a try" check.
    That heuristic reports false positives against a legitimate pattern used
    here: ``deployment/server.py`` imports fastapi at the top, and
    ``deployment/__init__.py`` wraps ``from .server import ...`` in a try, so a
    missing fastapi degrades the feature without breaking ``import pinnlab``.
    Whether the guard sits on the import or on its importer cannot be decided
    by reading one file, so the authoritative check is the runtime probe in
    :class:`TestImportWithDeclaredDependenciesOnly`.
    """

    def test_psutil_specifically_is_optional_everywhere(self):
        """Regression: error_handling.py imported psutil unguarded, which broke
        `import pinnlab` on a clean install."""
        unguarded = [
            str(path.relative_to(REPO_ROOT))
            for path in sorted(PACKAGE_ROOT.rglob("*.py"))
            for root, guarded in _module_import_roots(path)
            if root == "psutil" and not guarded
        ]
        assert not unguarded, f"psutil imported unguarded in: {unguarded}"


class TestImportWithDeclaredDependenciesOnly:
    """The package must import with nothing but its declared requirements."""

    @staticmethod
    def _import_with_blocked(blocked: set[str]) -> None:
        """Import pinnlab while the named top-level modules are unavailable."""
        for name in [m for m in sys.modules if m.startswith("pinnlab")]:
            del sys.modules[name]

        real_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in blocked:
                raise ImportError(f"No module named {name.split('.')[0]!r}")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = guarded
        try:
            importlib.import_module("pinnlab")
        finally:
            builtins.__import__ = real_import
            for name in [m for m in sys.modules if m.startswith("pinnlab")]:
                del sys.modules[name]
            importlib.import_module("pinnlab")

    def test_imports_without_any_optional_dependency(self):
        """This is the check a clean `pip install` would perform."""
        self._import_with_blocked(set(OPTIONAL_ROOTS))

    @pytest.mark.parametrize("missing", sorted(OPTIONAL_ROOTS))
    def test_imports_without_each_optional_dependency(self, missing):
        self._import_with_blocked({missing})


class TestDeclaredDependenciesAreHonest:
    def test_matplotlib_is_not_required(self):
        """matplotlib moved to the `viz` extra once its import became lazy.

        pinnlab.utils imports the visualisation module on first access rather
        than at package import, so the solver path never touches matplotlib.
        This test replaces one asserting the opposite, which carried a note to
        do exactly this if the import were ever guarded.
        """
        assert "matplotlib" not in _declared_runtime_requirements()

    def test_no_plotting_dependency_is_required(self):
        """The whole plotting stack belongs to the extra, not the base install."""
        declared = _declared_runtime_requirements()
        for package in ("matplotlib", "seaborn", "plotly", "scipy"):
            assert package not in declared, f"{package} must live in the viz extra"

    def test_seaborn_is_not_required(self):
        """seaborn moved to the `viz` extra once its import became lazy.

        Its only use is a single set_palette call applied on first plot, so
        without it the palette differs and nothing else does. Requiring it cost
        every user the ipywidgets and IPython import chain it drags in.
        """
        assert "seaborn" not in _declared_runtime_requirements()

    def test_viz_extra_holds_only_genuinely_optional_packages(self):
        data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        viz = data["project"]["optional-dependencies"]["viz"]
        roots = {spec.split(">")[0].split("=")[0].strip().lower() for spec in viz}
        assert roots <= OPTIONAL_ROOTS, (
            f"{roots - OPTIONAL_ROOTS} are in the viz extra but are not treated "
            "as optional by the code"
        )


class TestImportCostStaysDown:
    """Importing the package must not drag in the plotting stack.

    ``pinnlab.utils.visualization`` used to run ``import seaborn`` and
    ``sns.set_palette`` at module scope, and ``pinnlab.utils`` imports that
    module eagerly, so ``import pinnlab`` imported seaborn -- and through it
    ipywidgets and IPython. Measured with ``python -X importtime``, that was
    2.3s of a 9.0s import, spent on one palette call. The palette is now
    applied on first plot instead.

    This has to run in a subprocess: by the time the test session reaches here,
    the parent process has almost certainly imported those modules already.
    """

    HEAVY = ("seaborn", "ipywidgets", "IPython")

    def test_importing_pinnlab_does_not_import_the_plotting_stack(self):
        code = (
            "import sys; import pinnlab; "
            f"print(','.join(m for m in {self.HEAVY!r} if m in sys.modules))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
        )
        assert result.returncode == 0, result.stderr
        leaked = [m for m in result.stdout.strip().split(",") if m]
        assert not leaked, f"import pinnlab pulled in: {leaked}"

    def test_the_palette_is_still_applied_when_something_plots(self):
        """The saving is only legitimate if the style still ends up applied."""
        code = (
            "import matplotlib; matplotlib.use('Agg'); import sys; "
            "import pinnlab.utils.visualization as v; "
            "before = v._STYLE_APPLIED; v.PINNVisualizer(); "
            "print(before, v._STYLE_APPLIED, 'seaborn' in sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
        )
        assert result.returncode == 0, result.stderr
        before, after, seaborn_loaded = result.stdout.strip().split()
        assert before == "False", "style was applied at import, not on first use"
        assert after == "True", "constructing a visualizer did not apply the style"
        assert seaborn_loaded == "True", "seaborn should load when plotting starts"

import os
import sys
from datetime import datetime

# Add project source path
docs_root = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(docs_root, ".."))
sys.path.insert(0, os.path.join(project_root, "src"))

project = "pinnlab"
author = "Diogo Ribeiro"
current_year = datetime.now().year
copyright = f"{current_year}, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
]

autosummary_generate = True

# Render numpydoc "Attributes" sections as :ivar: fields inside the class body
# rather than as separate attribute directives. Without this, any class with an
# Attributes section has each attribute described twice under the same name --
# once by napoleon and once by autodoc's :undoc-members: -- which is what
# produced the duplicate-object warnings that blocked building with -W. The
# alternative fix, dropping :undoc-members:, would also silence them but at the
# cost of hiding genuinely undocumented API from the reference.
napoleon_use_ivar = True
html_theme = "alabaster"

# Ensure doctests see package root
doctest_global_setup = "import numpy as np; import torch; import pinnlab"

# Resolve type references into the projects they come from. Without these, a
# nitpicky build reports 652 of its 708 warnings against torch, numpy, the
# standard library and matplotlib -- names this project mentions in signatures
# but does not define.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}
# Do not let a slow or unreachable inventory host stall a docs build; a missed
# inventory degrades to unresolved references rather than hanging CI.
intersphinx_timeout = 30

# nitpicky surfaces every unresolved cross-reference. It is on, and the Docs
# workflow builds with -W, so a new broken reference fails the build.
nitpicky = True

# What cannot be resolved, and why. Each entry is a category, not a convenience:
#
# pinnlab.*   Docstrings reference the public re-export -- `pinnlab.training.Trainer`,
#             the path a user types -- while autodoc registers each class under the
#             module that defines it, `pinnlab.training.trainer.Trainer`. The API
#             reference documents subpackages rather than every submodule, so the
#             defining path is never rendered and the public path has no target.
#             Fixing this properly means documenting submodules, which reintroduces
#             duplicate-object warnings; see ROADMAP.md.
# pathlib._local
#             Python 3.13 moved Path's implementation into pathlib._local, so
#             autodoc reports that path while intersphinx only knows pathlib.Path.
# torch.jit._script, fastapi
#             A private torch path with no public inventory entry, and a project
#             with no intersphinx inventory published.
# Bare names  Unqualified names in prose -- Module, Tensor, Path -- that Sphinx
#             cannot attach to a project without a full path.
nitpick_ignore_regex = [
    ("py:class", r"pinnlab\..*"),
    ("py:class", r"pathlib\._local\..*"),
    ("py:class", r"torch\.jit\._script\..*"),
    ("py:class", r"fastapi\..*"),
    ("py:class", r"^(Module|Tensor|Path|nn\.Module)$"),
    ("py:class", r"^(VisualizationError|ValidationError)$"),
    ("py:class", r"^store_(gradients|losses|samples)$"),
    ("py:class", r"plotly\..*"),
    # Exception and function references hit the same re-export problem as the
    # classes above, and need their own role entries: an ignore is per-role.
    ("py:exc", r"^(VisualizationError|ValidationError|ConfigError)$"),
    ("py:exc", r"pinnlab\..*"),
    ("py:func", r"pinnlab\..*"),
    ("py:meth", r"pinnlab\..*"),
    # Truncated generics. Sphinx splits a parameterised type at its comma and
    # tries to resolve the fragment, so `Dict[str, float]` produces a reference
    # to `Dict[str`. These appear under CI's Python 3.11 and not under 3.13
    # locally, which is why they were not caught before pushing. The pattern
    # matches an opening bracket with no closing one -- a fragment, never a real
    # target.
    ("py:class", r"^[^\[\]]*\[[^\]]*$"),
    ("py:obj", r"^[^\[\]]*\[[^\]]*$"),
]

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

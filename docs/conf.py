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
html_theme = "alabaster"

# Ensure doctests see package root
doctest_global_setup = "import numpy as np; import torch; import pinnlab"

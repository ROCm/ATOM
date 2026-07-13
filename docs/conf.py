# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "ATOM"
copyright = "Copyright (c) %Y Advanced Micro Devices, Inc. All rights reserved."
author = "Advanced Micro Devices, Inc."
version = "0.1.0"
release = version

# -- General configuration ---------------------------------------------------
extensions = [
    "rocm_docs",
    # "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    # "sphinx.ext.mathjax",
    # "myst_parser",
]

external_toc_path = "./sphinx/_toc.yml"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "rocm_docs_theme"
html_theme_options = {
    "flavor": "ai-ecosystem",
    "link_main_doc": True,
    "repository_url": "https://github.com/ROCm/ATOM",
    "use_repository_button": True,
    "use_issues_button": True,
}

html_logo = "assets/atom_logo.png"

# -- Extension configuration -------------------------------------------------

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
#
# # Intersphinx configuration
# intersphinx_mapping = {
#     "python": ("https://docs.python.org/3", None),
#     "torch": ("https://pytorch.org/docs/stable/", None),
# }

# MyST parser settings
myst_enable_extensions = {
    "colon_fence",
    "deflist",
}

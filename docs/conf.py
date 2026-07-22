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
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    # enabled by rocm_docs:
    # "sphinx.ext.autodoc",
    # "sphinx.ext.mathjax",
    # "myst_parser",
]

external_toc_path = "./sphinx/_toc.yml"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "DOCUMENTATION_AUDIT_REPORT.md"]

# -- Options for HTML output -------------------------------------------------
html_theme = "rocm_docs_theme"
html_theme_options = {
    "flavor": "ai-ecosystem",
    "link_main_doc": True,
    "repository_url": "https://github.com/ROCm/ATOM",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_download_button": True,
}

html_logo = "assets/atom_logo.png"

# -- Extension configuration -------------------------------------------------

# Publish the llms.txt index at the docs site root and let
# rocm-docs-core generate llms-full.txt after each build (the llms.txt standard,
# https://llmstxt.org/). See the rocm-docs-core guide:
# https://rocm.docs.amd.com/projects/rocm-docs-core/en/latest/user_guide/llms.html
rocm_docs_generate_llms = True

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# MyST parser settings
myst_enable_extensions = {
    "colon_fence",
    "deflist",
}

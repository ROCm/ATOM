# Configuration file for the Sphinx documentation builder.

import subprocess
import sys
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(DOCS_DIR / "_ext"))

# -- Project information -----------------------------------------------------
project = "ATOM"
copyright = "Copyright (c) %Y Advanced Micro Devices, Inc. All rights reserved."
author = "Advanced Micro Devices, Inc."


# Read the checked-out source revision, never an installed ATOM package. A
# development checkout must not be labeled as an old stable release.
def git_revision(*args):
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


release = git_revision("describe", "--tags", "--always", "--dirty")
version = release
source_revision = git_revision("rev-parse", "--short=12", "HEAD")
rst_prolog = f".. |source_revision| replace:: {source_revision}\n"
html_last_updated_fmt = "%Y-%m-%d"
html_title = f"ATOM {release} documentation"

# -- General configuration ---------------------------------------------------
extensions = [
    "rocm_docs",
    "model_registry",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    # enabled by rocm_docs:
    # "sphinx.ext.autodoc",
    # "sphinx.ext.mathjax",
    # "myst_parser",
]

external_toc_path = "./sphinx/_toc.yml"
external_projects_current_project = "atom"
# These guides use explicit URLs for external documentation, not intersphinx
# roles. Do not fetch every ROCm project's inventory for an ATOM build.
external_projects = []

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "DOCUMENTATION_AUDIT_REPORT.md",
    "_ext",
]

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

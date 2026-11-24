# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""FlyDSL kernels for AMD gfx950 (MI355X): ``amd_kernels.<category>.<kernel>``.

Each kernel package names the computation and the shape it serves (root README, rule 6) and holds its README.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("amd-kernels")
except PackageNotFoundError:  # a checkout on PYTHONPATH, not installed
    __version__ = "0.0.0+checkout"

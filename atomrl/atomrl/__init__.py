# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""AtomRL: High-performance RL post-training for LLMs on AMD GPUs."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("atomrl") or None
except PackageNotFoundError:
    __version__ = None

if __version__ is None:
    try:
        # Keep source-checkout compatibility when only the root ATOM package
        # has been installed.
        __version__ = version("atom")
    except PackageNotFoundError:
        __version__ = "0.1.0"

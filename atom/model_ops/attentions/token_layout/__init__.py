# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Where this step's tokens go: the per-token index arrays a forward needs.

The counterpart of `..pool_layout`, on the other axis. That package answers
where a byte lives and is a function of the config; this one answers where a
token goes and is a function of the batch, so it is rebuilt every step.

Like `..pool_layout`, this package imports neither `aiter` nor any other `atom`
module, so it is importable on a runner with no AITER build and no GPU --
which is what makes the arithmetic checkable against a naive reference while
the staging around it is not. `tests/test_layout_packages.py` enforces that
over both packages. Nothing is re-exported here.
"""

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The source locations of a mono build's trace.

While it traces a kernel, FlyDSL gives every op the user call-site chain that built it
(``flydsl.expr.meta.capture_user_location``: up to ``FLYDSL_DEBUG_MAX_LOC_DEPTH`` frames, default 5).

``scoped()`` (around a build's trace that keeps its locations: DeepSeek-V4.1's hi programs checked,
``hi.emit.compile.compile_traced``) keeps ``DEPTH`` frames a chain (only the locations change: the LLVM IR is the same
at any depth). ``unlocated()`` (the engine's builds) captures none.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

from flydsl._mlir import ir
from flydsl.compiler import ast_rewriter, kernel_function
from flydsl.expr import meta

_HOLDERS = (
    meta,
    ast_rewriter,
    kernel_function,
)  # the modules that call it by a name of their own
DEPTH = 1  # user frames a location's call-site chain keeps (FlyDSL's ``max_loc_depth``)
_DEPTH_ENV = "FLYDSL_DEBUG_MAX_LOC_DEPTH"  # read by FlyDSL at every capture


@contextmanager
def scoped():
    """Inside: FlyDSL captures ``DEPTH`` frames a location; its own setting again afterwards."""
    before = os.environ.get(_DEPTH_ENV)
    os.environ[_DEPTH_ENV] = str(DEPTH)
    try:
        yield
    finally:
        if before is None:
            del os.environ[_DEPTH_ENV]
        else:
            os.environ[_DEPTH_ENV] = before


def _unknown_location() -> ir.Location:
    return ir.Location.unknown()


@contextmanager
def unlocated():
    """Inside: the traced ops carry no source location (``ir.Location.unknown()``), for the builds that only need their
    binary (the engine's, from FlyDSL's disk cache where it holds them). Without debug info FlyDSL leaves the locations
    out of the binary and of its cache key, so the binary and its cache entry are the ones a located trace gives;
    only a failing build's diagnostics lose their source lines. With debug info on (``FLYDSL_DEBUG_ENABLE_DEBUG_INFO``)
    the locations are in the binary: ``scoped()`` instead. Capturing them was about a quarter of a hi build's trace.
    """
    from flydsl.utils import env

    if env.debug.enable_debug_info:
        with scoped():
            yield
        return
    before = [mod.capture_user_location for mod in _HOLDERS]
    for mod in _HOLDERS:
        mod.capture_user_location = _unknown_location
    try:
        yield
    finally:
        for mod, fn in zip(_HOLDERS, before, strict=True):
            mod.capture_user_location = fn

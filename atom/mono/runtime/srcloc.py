# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL's source locations for a mono build's trace, captured at a fraction of the cost.

While it traces a kernel, FlyDSL gives every op the user call-site chain that built it
(``flydsl.expr.meta.capture_user_location``: up to ``FLYDSL_DEBUG_MAX_LOC_DEPTH`` frames, default 5). It reads
each frame's position with ``inspect.getframeinfo``, which scans the function's ``co_positions()`` up to the
current instruction and looks the source file up again on every call: for large traced block functions most of the
trace time.

``scoped()`` (around a build's trace: DeepSeek-V4.1's hi programs, ``hi.emit.compile.compile_traced``) puts
``capture_user_location`` here in FlyDSL's place, in every FlyDSL module that holds a reference to it, and restores
FlyDSL's afterwards: the same frames, filenames, lines and columns, with each code object's position table built once,
each file's source path resolved once, and ``DEPTH`` frames a chain (only the locations change: the LLVM IR is the
same at any depth). The 108 hi programs of a rank trace in 587 s instead of 1284 s (compile-only on a CPU). It
replaces only the FlyDSL functions it was written against (``FINGERPRINT``: their source text); after a FlyDSL change
it leaves FlyDSL's in place and logs one warning a process (the builds then trace at FlyDSL's own cost, about twice
the time).
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import os
from contextlib import contextmanager
from functools import cache

from flydsl._mlir import ir
from flydsl.compiler import ast_rewriter, kernel_function
from flydsl.expr import meta

# sha256 (first 16 hex) of the FlyDSL functions this module reproduces (``_fingerprint``)
FINGERPRINT = (
    "8d4737fd2f8f5c2d"  # flydsl 0.4.0.dev976; the same source in 0.4.0 and 0.4.2
)
_HOLDERS = (
    meta,
    ast_rewriter,
    kernel_function,
)  # the modules that call it by a name of their own
logger = logging.getLogger("atom")
DEPTH = 1  # user frames a location's call-site chain keeps (FlyDSL's ``max_loc_depth``)

_ORIGINAL = meta.capture_user_location
_POSITIONS: dict = {}  # code object -> its co_positions(), one entry per instruction
_SOURCE: dict = (
    {}
)  # co_filename -> the location's filename (``inspect.getsourcefile`` + ``meta.file_location``)


@cache
def _fingerprint() -> str:
    h = hashlib.sha256()
    for fn in (_ORIGINAL, meta.file_location):
        h.update(inspect.getsource(fn).encode())
    return h.hexdigest()[:16]


def _position(frame) -> tuple[int, int]:
    """(line, column) as ``inspect.getframeinfo(frame).positions`` with FlyDSL's fallbacks."""
    code = frame.f_code
    table = _POSITIONS.get(code)
    if table is None:
        table = _POSITIONS[code] = tuple(code.co_positions())
    i = frame.f_lasti
    line, col = (None, None) if i < 0 else (table[i // 2][0], table[i // 2][2])
    return (frame.f_lineno if line is None else line), (0 if col is None else col)


def _filename(frame) -> str:
    """``meta.file_location``'s filename for ``inspect.getframeinfo(frame).filename``: kept per file only
    when it is an existing file at an absolute path (then neither the working directory nor the
    linecache can change it)."""
    name = _SOURCE.get(frame.f_code.co_filename)
    if name is not None:
        return name
    name = inspect.getsourcefile(frame) or inspect.getfile(frame)
    if name and not name.startswith("<"):
        name = os.path.abspath(name)
        if os.path.isabs(frame.f_code.co_filename) and os.path.exists(name):
            _SOURCE[frame.f_code.co_filename] = name
    return name


def capture_user_location() -> ir.Location:
    """``flydsl.expr.meta.capture_user_location``, line for line, with the frame lookups above and ``DEPTH``."""
    stack = getattr(meta._tls, "stack", None)
    boundary = stack[-1][0] if stack else None
    max_depth = DEPTH
    ctx = ir.Context.current
    locs = []
    boundary_loc = None

    frame = inspect.currentframe().f_back
    try:
        while frame is not None:
            code = frame.f_code
            is_boundary = boundary is not None and code is boundary
            if not meta._is_framework_file(code.co_filename):
                keep = len(locs) < max_depth
                if keep or is_boundary:
                    line, col = _position(frame)
                    floc = ir.Location.file(_filename(frame), line, col, context=ctx)
                    if keep:
                        locs.append(floc)
                    else:
                        boundary_loc = floc
            if is_boundary:
                break
            frame = frame.f_back
    finally:
        del frame

    if boundary_loc is not None:
        locs.append(boundary_loc)
    if not locs:
        return ir.Location.unknown()
    callee, callers = locs[0], locs[1:]
    if not callers:
        return callee
    return ir.Location.callsite(callee, callers)


@cache
def _warn_miss() -> None:
    """Once a process: FlyDSL's location capture is not the one this module reproduces."""
    logger.warning(
        "mono srcloc: FlyDSL's capture_user_location / file_location differ from the ones the fast source locations "
        "reproduce (fingerprint %s, expected %s): the builds' traces take FlyDSL's own location capture, about twice "
        "the build time; the programs are the same",
        _fingerprint(),
        FINGERPRINT,
    )


@contextmanager
def scoped():
    """Inside: FlyDSL captures locations with ``capture_user_location`` (when its functions are the ones this module
    reproduces); FlyDSL's own again afterwards."""
    if _fingerprint() != FINGERPRINT:
        _warn_miss()
        yield
        return
    before = [mod.capture_user_location for mod in _HOLDERS]
    for mod in _HOLDERS:
        mod.capture_user_location = capture_user_location
    try:
        yield
    finally:
        for mod, fn in zip(_HOLDERS, before, strict=True):
            mod.capture_user_location = fn

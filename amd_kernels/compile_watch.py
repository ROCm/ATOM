# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Report FlyDSL compiles of amd_kernels kernels that happen after a caller's warm-up.

Every variant a server can reach compiles before it serves (root README, rule 4). A caller
that has finished its warm-ups calls ``arm(warn)`` once: from then on every FlyDSL compile
whose call stack passes through amd_kernels calls ``warn(message)`` with the innermost
amd_kernels frame and is counted (``late_compiles``). Compiles of other code pass
silently. The hook is ``MlirCompiler.compile``, which a disk-cache hit does not reach: a
variant loaded from FlyDSL's cache is not reported.
"""

import os
import threading
import traceback
from collections.abc import Callable

_lock = threading.Lock()
_warn: Callable[[str], None] | None = None
_count = 0
_real = None
_PACKAGE = os.path.dirname(os.path.abspath(__file__)) + os.sep
_THIS = os.path.abspath(__file__)


def arm(warn: Callable[[str], None]) -> bool:
    """Start reporting (or send the reports to a new ``warn``). False when this FlyDSL
    has no ``MlirCompiler.compile`` to hook."""
    global _warn, _real
    try:
        from flydsl.compiler.jit_function import MlirCompiler
    except ImportError:
        return False
    with _lock:
        _warn = warn
        if _real is None:
            _real = MlirCompiler.compile

            def compile_and_report(self, *args, **kwargs):
                _report()
                return _real(self, *args, **kwargs)

            MlirCompiler.compile = compile_and_report
    return True


def disarm() -> None:
    """Stop reporting and restore FlyDSL's compile (tests)."""
    global _warn, _real
    with _lock:
        if _real is not None:
            from flydsl.compiler.jit_function import MlirCompiler

            MlirCompiler.compile = _real
        _warn, _real = None, None


def late_compiles() -> int:
    """amd_kernels compiles reported since the first ``arm``."""
    return _count


def _report() -> None:
    global _count
    warn = _warn
    if warn is None:
        return
    frames = [
        f
        for f in traceback.extract_stack()
        if os.path.abspath(f.filename).startswith(_PACKAGE) and os.path.abspath(f.filename) != _THIS
    ]
    if not frames:
        return
    with _lock:
        _count += 1
    f = frames[-1]
    warn(f"amd_kernels: FlyDSL compile after the warm-up (a variant the warm-up missed), from {f.filename}:{f.lineno}")

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The compile-or-dispatch call of this entry's hosts."""

import flydsl.compiler as flyc


def _run_compiled(exe, *args):
    """The first call compiles and runs (``flyc.compile``); later calls dispatch the cached CompiledFunction."""
    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)

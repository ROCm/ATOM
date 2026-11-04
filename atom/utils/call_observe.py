# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""`observe` from aiter's call hooks (`aiter.jit.utils.call_hook`), for ATOM's
own kernel launchers -- the GDN Triton ops -- so `ATOM_OP_RECORD_DIR` sees
their calls as it sees aiter's ops. With no hook registered a call pays one
truthiness test; an aiter without call hooks gets a decorator that does
nothing."""

try:
    from aiter.jit.utils.call_hook import observe
except ImportError:  # aiter predates call hooks: nothing records calls anyway

    def observe(fn=None, *, name=None):
        return fn if fn is not None else (lambda func: func)


__all__ = ["observe"]

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Optional per-CTA phase stamps inside a stage (diagnostics only). A stage marks its phases with ``point(k)``; nothing is
generated unless the trace sets ``HOOK[0]`` to a callable(slot) (the production build leaves it None).
"""

import flydsl.expr as fx

from atom.models.deepseek_v41.mono.gfx1250.kernels.device import TraceSlot

HOOK = TraceSlot()


def point(k):
    if fx.const_expr(HOOK[0] is not None):
        HOOK[0](k)

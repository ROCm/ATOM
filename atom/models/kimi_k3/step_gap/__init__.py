# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""What a Kimi-K3 DSpark decode step runs between two target forwards beside
the step tail and the draft's layers, on gfx1250 kernels of the model's own:
the draft's context path (``draft_context``, ``ATOM_KIMI_MONO_DRAFT_CONTEXT``).

Outside ``atom/models/kimi_k3/mono`` on purpose: the mono kernels' JIT keys
digest that directory, so a file added there would rebuild every one of them.
"""

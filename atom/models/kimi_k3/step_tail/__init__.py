# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Kimi-K3's decode-step tail on gfx1250 (``ATOM_KIMI_MONO_STEP_TAIL``): the
vocab-sharded LM head, the cross-rank greedy argmax and draft verification, and
the DSpark draft's Markov block sampling, in kernels of their own after the
model's mono launches; ``dispatch`` hands the engine the verdict
(``StepOverrideModel``).

Beside ``kimi_k3/mono`` rather than inside it: the mono kernels' build key
digests that directory, so an edit here rebuilds none of them.
"""

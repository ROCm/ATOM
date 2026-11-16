# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM's own host for the TileRT-derived DSpark decode step at TP8 on gfx950: the buffers (``alloc``), the code
objects and their launch (``codeobj``; our FlyDSL builds from source, ``flybuild``), the DSpark token tables built from
the checkpoint (``toktab_device``: on the GPU by the step's kernels, bit-exact; ``toktab``: a CPU re-derivation, dev
only), one rank's step, its kernargs and graphs (``host``), and the CPU checks (``check``). The step's launches are
``plan.decode_launches``'; the parameters ``weights.params``'.
"""

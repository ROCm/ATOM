# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The TileRT-structured DSpark decode step at TP8 on gfx950 (``tk``). As a V4.1 mono backend: its gate
(``gate``: the deployment it serves), ATOM's step override and decode round (``dispatch``), the step's runner on TkRank
(``runner``, ``kv``, ``outputs``). ATOM's own host for it: the buffers (``alloc``), the code
objects and their launch (``codeobj``; our FlyDSL builds from source, ``flybuild``), the DSpark token tables built from
the checkpoint (``toktab_device``: on the GPU by the step's kernels, bit-exact), one rank's step, its kernargs and
graphs (``host``). The step's launches are ``decode_launches``'; the parameters ``weights.params``'; the FlyDSL kernels
``kernels``.
"""

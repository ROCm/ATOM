# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Kernels behind the MiniMax-M3 launch-A seam (``launch_a.KERNEL`` names one module).

A module provides:

* ``available() -> bool``: its aiter entry imports (checked once, at model build);
* ``hosts(traits) -> bool``: the aiter gfx1250 PA launches (``Gfx1250Traits``) it runs in
  place of production's, i.e. the ones the attention op defers;
* ``run(call: launch_a.LaunchACall) -> (out, residual_out)``: one launch for the layer, with
  production's bits, on the deferred PA's arguments and production's k5 context.

A new version of a kernel with the same arguments is new aiter files; new arguments are a
new ``run``; a kernel that absorbs more of the layer (the index top-k, the QKV chain, the
router / shared gate_up) is a new module plus the deferral of that part in ``launch_a``.
"""

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""MiniMax-M3.1 decode launches on gfx1250 (MI455X, TP4, a C4 decode step of one request x 8 rows).

Host wrappers here, FlyDSL device code in ``kernels/``:
- ``aq``: the QKV chain (norm, RoPE, NVFP4 cache insert), the indexer-CP index scores and the cross-rank
  top-k exchange in one launch;
- ``mf``: the MoE front: o_proj + k5 all-reduce + residual + RMSNorm, the router and the shared expert;
- ``moe_tail_ar`` (+ ``moe_tail_ar_layout``): the routed GEMM2 + MoE tail + k5 all-reduce + RMSNorm + MXFP8 in
  one launch, registered with aiter's deferred MoE tail (``register_deferred_tail_launch``);
- ``dmf``: a dense layer's post-attention half (o_proj, all-reduce, norm, MLP) in one launch;
- ``kernels/rsh_tail``: device helpers the launches share.
Every launch is byte-identical to the separate launches it replaces; the model decides where each runs.
"""

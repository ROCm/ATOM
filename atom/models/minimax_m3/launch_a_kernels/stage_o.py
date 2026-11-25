# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Stage O: o_proj GEMV -> k5 all-reduce + residual + Gemma RMSNorm in one launch (aiter
``flydsl_oproj_ar``, MONO), in place of production's ``mxfp8_decode_gemm`` and the bf16 Gluon-order
k5 call after the sparse PA, which stays production's. Same bits; it runs k5's own inbox / counter
protocol, so it interleaves with production's k5 calls. Up to 32 rows."""

DEFERS_PA = False  # production's PA runs in the attention op; the launch starts at o_proj
MAX_ROWS = 32


def available() -> bool:
    try:
        from aiter.ops.flydsl.oproj_ar import flydsl_oproj_ar  # noqa: F401
    except ImportError:
        return False
    return True


def hosts(traits) -> bool:
    return False  # no PA launch is deferred


def rows_ok(m: int) -> bool:
    return 0 < m <= MAX_ROWS


def run(call):
    from aiter.ops.flydsl.oproj_ar import flydsl_oproj_ar

    from atom.models.minimax_m3.launch_a import k5_maxit

    k5 = call.k5
    m = call.x.shape[0]
    return flydsl_oproj_ar(
        call.x,
        call.x_scale,
        call.seam.o_proj.weight,
        call.seam.o_proj.weight_scale,
        call.residual.reshape(m, -1),
        call.norm_w,
        call.eps,
        k5.inbox,
        k5.ctr,
        k5.slot,
        k5.rank,
        max_rows=k5.max_rows,
        maxit=k5_maxit(),
    )

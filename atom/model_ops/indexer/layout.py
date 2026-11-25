# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Index cache layout names, shared without importing any GPU kernels."""

PRESHUFFLE = "preshuffle"
# gfx1250: FP4 rows in natural order, scored by the OPUS kernels.
NATURAL = "natural"
# gfx950 OPUS: MFMA-permuted scales, distinct from FlyDSL's preshuffle.
MFMA = "mfma"

# gfx1250 FP4 indexer: the per-layer weights_proj GEMM folded into the query
# RoPE/MXFP4-quant launch (aiter q_rope_mxfp4_quant), for a step of at most this
# many rows. The kernel returns the head weights as bf16 directly, so the fold
# is a single launch. It targets the small, launch-bound decode and
# short-prefill steps, matching the vLLM reference's 128-row gate; larger steps
# keep the standalone bf16 GEMM. `Indexer.project` applies the bound inline.
_MAX_FUSED_PROJECT_ROWS = 128


def fp4_cache_byte(row, packed_col):
    """OPUS gfx950: [group, 32-dim chunk, row, 16 packed bytes]."""
    return row // 64 * 4096 + (packed_col // 16 * 64 + row % 64) * 16 + packed_col % 16


def fp4_scale_byte(row, chunk):
    """OPUS gfx950: [group, chunk % 2, row % 32, chunk // 2, row // 32]."""
    return (
        row // 64 * 256
        + chunk % 2 * 128
        + row % 32 * 4
        + chunk // 2 * 2
        + row % 64 // 32
    )

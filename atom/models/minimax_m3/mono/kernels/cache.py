# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Optional cache contract for callers with independently allocated index pages.

Builders default to ``cache_mode="atom"``. ``"vllm"`` selects ABI version 1:
K/V scales are positive FP32 scalars, index Q is rounded BF16 -> unit E4M3,
and index addressing is independent of the packed page-16 main cache. Both
writers use AITER's nonsaturating hardware conversion, including overflow NaNs.
Attention follows AITER's PS/head-1 scalar mode: cast P directly to E4M3 and
multiply PV by the scalar V scale; the default mode retains dynamic P scaling.
The selected context is split evenly into eight ranges, each using online
softmax across its intersecting 256-key tiles. The default uses fixed tiles.

The launcher's optional ``cache_args`` is the device address of a contiguous
int64[3]: index slot-mapping address, index block-table address, and table row
stride in int32 entries. Slots are int64 token addresses into the index cache;
tables are int32 physical 128-token page IDs, with one row per query token.
The owner must keep the descriptor and all referenced storage alive and at
stable addresses through every captured graph's lifetime. No descriptor is
read in the default ATOM mode. The original positional stream argument and
ordered K1_ARGS remain unchanged.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr.typing import T

from atom.models.minimax_m3.mono.kernels.common import (
    bf16_round,
    fp8_pack4,
    fp8x8_bf16_pk,
    rsrc,
    uniform,
)


def is_vllm_cache(mode: str) -> bool:
    if mode not in ("atom", "vllm"):
        raise ValueError(f"Unknown mono cache mode: {mode!r}")
    return mode == "vllm"


def index_arguments(cache_args):
    """Device int64[3]: index slots pointer, table pointer, table row stride."""
    pointers = []
    for i in range(2):
        words = fx.Vector(
            bo.buffer_load(rsrc(cache_args), 2 * i, vec_width=2, dtype=T.i32)
        )
        pointers.append(
            (fx.Int64(uniform(words[1])) << 32) | fx.Int64(fx.Uint32(uniform(words[0])))
        )
    width = uniform(bo.buffer_load(rsrc(cache_args), 4, vec_width=1, dtype=T.i32))
    return pointers[0], pointers[1], width


def index_query_fp8(values):
    """Match the AITER writer's BF16 materialization then unit E4M3 rounding."""
    # AITER's opus::cast uses the hardware conversion without saturation;
    # overflows must retain its signed NaN representation.
    packed = fp8_pack4(*[bf16_round(value) for value in values])
    unpacked = fp8x8_bf16_pk([packed]).to(fx.Float32)
    return [unpacked[i] for i in range(4)]

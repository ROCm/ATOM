# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Typed argument groups composed at build time into unchanged scalar ABIs."""

import flydsl.expr as fx
from flydsl.expr.typing import Int32, Int64


@fx.struct
class HcArgs:
    hc_norm: Int64
    hc_down_inject: Int64
    hc_up: Int64


@fx.struct
class MoeArgs:
    router: Int64
    shared_w13: Int64
    shared_w2: Int64
    routed_w13: Int64
    routed_w13_scale: Int64
    routed_w2: Int64
    routed_w2_scale: Int64


@fx.struct
class OutputArgs:
    streams_out: Int64
    pending_out: Int64
    pending_raw: Int64
    moe_out: Int64


@fx.struct
class QsaProjectionArgs:
    qkv_weight: Int64
    q_norm: Int64
    k_norm: Int64
    index_weight: Int64
    index_q_norm: Int64
    index_k_norm: Int64
    cos: Int64
    sin: Int64
    index_cos: Int64
    index_sin: Int64
    positions: Int64
    position_rows: Int32
    block_tables: Int64
    table_stride: Int32
    slot_mapping: Int64
    compressed_slot_mapping: Int64
    token_to_req: Int64
    logical_positions: Int64
    page_size: Int32
    k_cache: Int64
    v_cache: Int64
    raw_key_cache: Int64
    compressed_key_cache: Int64
    rope_position_cache: Int64


@fx.struct
class QsaDecodeArgs:
    query: Int64
    gate: Int64
    index_query: Int64
    block_tables: Int64
    table_stride: Int32
    table_width: Int32
    seq_lens: Int64
    token_to_req: Int64
    logical_positions: Int64
    page_size: Int32
    cache_pages: Int32
    k_cache: Int64
    v_cache: Int64
    compressed_key_cache: Int64
    o_proj: Int64

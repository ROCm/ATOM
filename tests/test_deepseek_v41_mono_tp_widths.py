# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""V4.1 mono's LDS at every TP size it serves: each buffer holds what the
kernels write into it, and every build fits a CU's LDS."""

import pytest

pytest.importorskip("aiter")
pytest.importorskip("flydsl")

from atom.models.deepseek_v41.mono.config import MAX_MOE_ROWS, MAX_ROWS, SUPPORTED_DPA
from atom.models.deepseek_v41.mono.kernels import attn_post as k2a
from atom.models.deepseek_v41.mono.kernels import moe as k2b
from atom.models.deepseek_v41.mono.kernels.dims import Dims
from atom.models.deepseek_v41.mono.kernels.moe_shape import RouteShape
from atom.mono.device.ops import lds_bytes
from atom.mono.plan.execution import LDS_BYTES

TPS = [2, 4, 8]
# target verify widths (1, 2, 4, 8 requests) and the draft's (2, 8 requests of
# 5 rows), each with the ug_groups its runner builds with
SHAPES = [
    (6, 384, 6, 2), (12, 384, 6, 2), (24, 384, 6, 2), (48, 384, 6, 2),
    (10, 128, 3, 1), (40, 128, 3, 1),
]  # fmt: skip


@pytest.mark.parametrize("tp", TPS)
def test_a_wave_scale_region_holds_a_down_round(tp):
    d = Dims(tp)
    block = k2b.down_scale_words(d)
    assert block >= d.down_scale_cols * 8  # a 32-row block, rounded
    assert k2b.down_wsl_words(d) >= k2b.DOWN_R_MAX * block
    assert k2b.UG_WSL_WORDS >= 256  # an ug batch


@pytest.mark.parametrize("tp", TPS)
@pytest.mark.parametrize("tokens, experts, topk, ug_groups", SHAPES)
def test_k2b_fits_a_cu(tp, tokens, experts, topk, ug_groups):
    rs = RouteShape(experts, topk, tokens, tp, ug_groups)
    assert lds_bytes(k2b.moe_smem(tokens, rs, True)) <= LDS_BYTES
    # load_smid_tile copies a token tile's MXFP8 shared intermediate into sxl
    offsets = k2b.stage_lds(tokens, rs)["offsets"]
    rows = min(tokens, 16)
    assert rows * rs.dims.sh_inter // 4 <= offsets["sxsl"] - offsets["sxl"]
    assert rows * rs.dims.sh_inter // 32 <= offsets["contrib"] - offsets["sxsl"]


@pytest.mark.parametrize("ep", SUPPORTED_DPA)
@pytest.mark.parametrize("experts, topk, ug_groups", [(384, 6, 2), (128, 3, 1)])
@pytest.mark.parametrize("rows", [6, 48, 49, 72])
def test_k2b_fits_a_cu_under_dp_attention(ep, experts, topk, ug_groups, rows):
    # the MoE runs every rank's rows: up to MAX_MOE_ROWS, both down tasks'
    # contributions held at once; past MAX_ROWS in one stage_down call
    tokens = min(rows * ep, MAX_MOE_ROWS)
    rs = RouteShape(experts, topk, tokens, 1, ug_groups, ep=ep)
    assert rs.down_one_call == (tokens > MAX_ROWS)
    assert lds_bytes(k2b.moe_smem(tokens, rs, True)) <= LDS_BYTES


@pytest.mark.parametrize("dp", SUPPORTED_DPA)
@pytest.mark.parametrize("experts, topk, ug_groups", [(384, 6, 2), (128, 3, 1)])
@pytest.mark.parametrize("rows", [6, 48, 49, 72])
def test_k2b_fits_a_cu_under_dp_attention_moe_tp(dp, experts, topk, ug_groups, rows):
    # the MoE TP over the DP ranks: every rank's rows at TP dp's widths, a
    # rank's own rows routed by it
    tokens = min(rows * dp, MAX_MOE_ROWS)
    rs = RouteShape(experts, topk, tokens, dp, ug_groups, dp_attention=True)
    assert rs.dpa and rs.router_rows == tokens // dp
    assert lds_bytes(k2b.moe_smem(tokens, rs, True)) <= LDS_BYTES


@pytest.mark.parametrize("tp", TPS)
def test_shared_tasks_pair_into_quant_groups(tp):
    # smid_quant: an odd shared task quantizes its and the even task's 16 columns
    assert Dims(tp).shared_tasks % 2 == 0


@pytest.mark.parametrize("tp", TPS)
@pytest.mark.parametrize("tokens", [6, 12, 24, 48])
def test_k2a_fits_a_cu(tp, tokens):
    members = k2a.attn_post_smem(tokens, True, 1 << 20, Dims(tp))
    assert lds_bytes(k2a.lds_union("K2aLds", members)) <= LDS_BYTES

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""V4.1 mono's LDS at every TP size it serves: each buffer holds what the
kernels write into it, and every build fits a CU's LDS."""

import pytest

pytest.importorskip("flydsl")

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
    assert k2b.wsl_words(d) >= k2b.DOWN_R_MAX * block
    assert k2b.wsl_words(d) >= 256  # an ug batch


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


@pytest.mark.parametrize("tp", TPS)
def test_shared_tasks_pair_into_quant_groups(tp):
    # smid_quant: an odd shared task quantizes its and the even task's 16 columns
    assert Dims(tp).shared_tasks % 2 == 0


@pytest.mark.parametrize("tp", TPS)
@pytest.mark.parametrize("tokens", [6, 12, 24, 48])
def test_k2a_fits_a_cu(tp, tokens):
    members = k2a.attn_post_smem(tokens, True, 1 << 20, Dims(tp))
    assert lds_bytes(k2a.lds_union("K2aLds", members)) <= LDS_BYTES

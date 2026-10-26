# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A TP rank's share of a model's widths (``atom.mono.plan.shard``), and V4.1's
``Dims`` built on it at every TP size mono serves."""

import pytest

from atom.models.deepseek_v41.mono import moe_plan as mplan
from atom.models.deepseek_v41.mono.kernels.dims import UG_PART, Dims
from atom.mono.plan.shard import Shard, ShardError, Tiles, padded, shard_refusal


def test_split_is_exact_or_refused_by_name():
    assert Shard(8).split(2304, "inter") == 288
    with pytest.raises(ShardError, match="inter 2304 is not divisible by TP 5"):
        Shard(5).split(2304, "inter")


def test_tp_below_one_is_refused():
    with pytest.raises(ShardError):
        Shard(0)


def test_padded_rounds_up_to_the_alignment():
    assert padded(288, 128) == 384
    assert padded(1152, 128) == 1152


@pytest.mark.parametrize(
    "n, count, ragged", [(8, 1, True), (16, 1, False), (32, 2, False), (40, 3, True)]
)
def test_tiles_of_sixteen(n, count, ragged):
    tiles = Tiles(n, 16)
    assert (tiles.count, tiles.ragged) == (count, ragged)


def test_shard_refusal_names_the_width():
    assert shard_refusal(Dims, 4) is None
    assert "wo_a groups" in shard_refusal(Dims, 16)


@pytest.mark.parametrize("tp", [2, 4, 8])
def test_v41_dims_tile_the_full_model(tp):
    d = Dims(tp)
    assert d.heads * tp == 64 and d.groups * tp == 8
    assert d.o_rows == d.groups * 1024 and d.group_k == 8 * 512
    assert d.inter_real * tp == 2304 and d.sh_inter * tp == 2304
    assert d.inter % 128 == 0 and 0 <= d.inter - d.inter_real < 128
    assert d.down_scale_cols % 8 == 0 and d.down_scale_cols >= d.inter // 32
    # ug covers the real width, a task of 32 ug_groups columns, inside the
    # padded width
    for ug_groups in (1, 2):
        parts = mplan.ug_parts(d.inter_real // UG_PART, ug_groups)
        assert d.inter_real <= parts * ug_groups * UG_PART <= d.inter
    assert d.head_tiles.count * 16 >= d.heads


def test_v41_dims_at_tp8_leave_a_ragged_head_tile():
    d = Dims(8)
    assert (d.heads, d.head_tiles.count, d.head_tiles.ragged) == (8, 1, True)
    # the shared expert's down projection: K = 288, a last quarter FP8 step
    assert d.sh_inter % 64 == 32

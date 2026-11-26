# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""A K-padded MLP pair's loaders zero the pad behind every real shard they write (``linear._zero_pad_tail``), on
the CPU job too (AITER stubbed where it is not installed).

A weight sync's module-level loader writes the real shard over a weight that is already preshuffled: the bytes in
the pad's row-major place belong to other tiles, and re-shuffled they would land in the pad. Here the parameters
start out full of such bytes (0xAB) and every load must leave the real shard at the front of its slot and zeros
behind it: gate_up's rows (``MergedColumnParallelLinear``, per shard, weight and its 32-row scale blocks) and down's
columns (``RowParallelLinear``, weight and its 32-column scale groups)."""

from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_without_aiter

TP, INTER, HIDDEN = 8, 2304, 64  # DeepSeek-V4.1's shared expert at TP8: 288 -> 320
REAL, PADDED = INTER // TP, 320


@pytest.fixture(scope="module")
def L():
    with atom_without_aiter() as load:
        yield load("atom.model_ops.linear")


def _param(L, *shape):
    p = torch.nn.Parameter(
        torch.full(shape, 0xAB, dtype=torch.uint8), requires_grad=False
    )
    p.weight_loader_process = L.LinearBase.weight_loader_process
    return p


def _pattern(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, 200, shape, dtype=torch.uint8, generator=g)


@pytest.mark.parametrize("rank", [0, 3, 7])
def test_gate_up_rows_behind_each_shard_are_zeroed(L, rank):
    w, s = _param(L, 2 * PADDED, HIDDEN), _param(L, 2 * PADDED // 32, HIDDEN // 32)
    layer = SimpleNamespace(
        output_sizes=[PADDED * TP, PADDED * TP], _ckpt_output_sizes=[INTER, INTER], tp_size=TP, tp_rank=rank,
        tp_dim=0, weight_scale_row_group=32, quant_type=L.QuantType.per_1x32, weight_scale=s, input_scale=None,
    )  # fmt: skip
    loads = {}
    for shard in (0, 1):
        loads[shard] = (
            _pattern(INTER, HIDDEN, seed=shard),
            _pattern(INTER // 32, HIDDEN // 32, seed=10 + shard),
        )
        L.MergedColumnParallelLinear.weight_loader(layer, w, loads[shard][0], shard)
        L.MergedColumnParallelLinear.weight_loader(layer, s, loads[shard][1], shard)
    for shard, (lw, ls) in loads.items():
        w0, s0 = shard * PADDED, shard * PADDED // 32
        assert torch.equal(w.data[w0 : w0 + REAL], lw.chunk(TP, 0)[rank])
        assert torch.equal(s.data[s0 : s0 + REAL // 32], ls.chunk(TP, 0)[rank])
        assert (w.data[w0 + REAL : w0 + PADDED] == 0).all()
        assert (s.data[s0 + REAL // 32 : s0 + PADDED // 32] == 0).all()


@pytest.mark.parametrize("rank", [0, 3, 7])
def test_down_columns_behind_the_shard_are_zeroed(L, rank):
    w, s = _param(L, HIDDEN, PADDED), _param(L, HIDDEN // 32, PADDED // 32)
    layer = SimpleNamespace(
        tp_size=TP, tp_rank=rank, tp_dim=1, _k_padded=True, quant_type=L.QuantType.per_1x32, weight_scale=s,
        input_scale=None, bias=None,
    )  # fmt: skip
    lw, ls = _pattern(HIDDEN, INTER, seed=2), _pattern(
        HIDDEN // 32, INTER // 32, seed=3
    )
    L.RowParallelLinear.weight_loader(layer, w, lw)
    L.RowParallelLinear.weight_loader(layer, s, ls)
    assert (
        torch.equal(w.data[:, :REAL], lw.chunk(TP, 1)[rank])
        and (w.data[:, REAL:] == 0).all()
    )
    assert torch.equal(s.data[:, : REAL // 32], ls.chunk(TP, 1)[rank])
    assert (s.data[:, REAL // 32 :] == 0).all()

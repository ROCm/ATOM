# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""MiniMax-M3.1 DSpark stacked context K/V (`dspark_ctx_kv`): the stacking and the
writer's reach. The model-level cases are gone in the AgentX merge: MiniMaxM3DSpark
keeps ganyi's own stacked write (`write_context_rows(taps, ...)`), not this module's."""


import pytest
import torch

pytest.importorskip("triton")

from atom.model_ops.minimax_m3.dspark_ctx_kv import (
    CTX_KV_ROW_PAD,
    _element_offsets,
    stack_kv_rows,
)

LAYERS, HEADS, KV_HEADS, HEAD_DIM = 5, 64, 4, 128


def test_layers_in_one_allocation_are_reached_from_layer_0():
    pool = torch.zeros(5, 4, 8)
    views = [pool[i] for i in range(5)]
    assert _element_offsets(views, views[0]).tolist() == [32 * i for i in range(5)]


def test_writer_refuses_caches_it_cannot_reach_from_layer_0():
    separate = [torch.zeros(4, 8) for _ in range(3)]
    with pytest.raises(ValueError, match="layer 0's allocation"):
        _element_offsets(separate, separate[0])
    pool = torch.zeros(3, 4, 8)
    with pytest.raises(ValueError, match="at or above it"):
        _element_offsets([pool[2], pool[1], pool[0]], pool[2])


def test_stacked_rows_are_each_layers_kv_rows_then_padding():
    q, kv, k = 96, 64, 64
    weights = [torch.randn(q + kv, k).to(torch.float8_e4m3fn) for _ in range(LAYERS)]
    scales = [
        torch.randint(100, 140, (q + kv, k // 32), dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )
        for _ in weights
    ]
    w, s = stack_kv_rows(weights, scales, slice(q, q + kv))
    assert w.shape == (CTX_KV_ROW_PAD, k) and s.shape == (CTX_KV_ROW_PAD, k // 32)
    for i in range(LAYERS):
        rows = slice(i * kv, (i + 1) * kv)
        assert torch.equal(w[rows].view(torch.uint8), weights[i][q:].view(torch.uint8))
        assert torch.equal(s[rows].view(torch.uint8), scales[i][q:].view(torch.uint8))
    assert not w[LAYERS * kv :].view(torch.uint8).any()
    assert (s[LAYERS * kv :].view(torch.uint8) == 127).all()


gfx950 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="needs gfx950",
)

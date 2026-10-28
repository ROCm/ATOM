# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 FP4 index / NVFP4 sparse KV: the startup refusal and the pool layout.

Both run without a GPU. The refusal keeps a deployment the kernels cannot serve
from starting (it would otherwise fail inside a Gluon compile, or on gfx942 inside
an inline-asm build); the pool test pins the byte layout the kernels address.
"""

import pytest
import torch

from atom.config import m3_fp4_cache_unsupported_reason
from atom.model_ops.attentions.mha_kv_pool import MhaKvPool
from atom.model_ops.attentions.pool_layout.pool_rows import KvGeometry

# MiniMax-M3.1: 64 q heads, 4 kv heads.
HEADS, KV_HEADS = 64, 4


@pytest.mark.parametrize(
    "gfx, index_fp4, kv_nvfp4, tp, reuses_topk, refused",
    [
        ("gfx950", False, False, 8, True, None),  # neither cache: nothing to check
        ("gfx950", True, False, 4, False, None),
        ("gfx950", True, True, 4, False, None),  # TP4: 16 q heads per kv head
        ("gfx950", True, True, 2, False, None),  # TP2: 32 / 2 = 16
        ("gfx942", True, False, 4, False, "need gfx950"),
        ("gfx942", False, True, 4, False, "need gfx950"),
        ("gfx950", False, True, 8, False, "got 8 at TP8"),  # kv heads replicate
        ("gfx950", True, False, 8, False, None),  # FP4 index alone has no head rule
        ("gfx950", False, True, 4, True, "top-k-reusing"),
    ],
)
def test_refusal(gfx, index_fp4, kv_nvfp4, tp, reuses_topk, refused):
    reason = m3_fp4_cache_unsupported_reason(
        gfx, index_fp4, kv_nvfp4, tp, HEADS, KV_HEADS, reuses_topk
    )
    if refused is None:
        assert reason is None
    else:
        assert refused in reason


def test_nvfp4_pool_fields_and_views():
    layers, block, heads, dim = 60, 128, 1, 128
    pool = MhaKvPool(
        layers=layers,
        block_size=block,
        num_kv_heads=heads,
        head_dim=dim,
        kv_dtype=torch.float8_e4m3fn,
        kv_format="nvfp4",
    )
    # Per (token, head): D/2 payload bytes and D/16 E4M3 scales, for K and V.
    per_layer = 2 * heads * block * (dim // 2 + dim // 16)
    assert pool.entry_bytes == layers * per_layer
    fp8 = MhaKvPool(
        layers=layers,
        block_size=block,
        num_kv_heads=heads,
        head_dim=dim,
        kv_dtype=torch.float8_e4m3fn,
    )
    assert pool.entry_bytes < 0.6 * fp8.entry_bytes

    pool.allocate(3, "cpu")
    k, v = pool.kv_views(5)
    ks, vs = pool.scale_views(5)
    assert k.shape == v.shape == (3, heads, block, dim // 2) and k.dtype == torch.uint8
    assert ks.shape == vs.shape == (3, heads, block, dim // 16)
    assert ks.dtype == torch.uint8
    # Layers do not alias: a write to one layer's K leaves its neighbour alone.
    k.fill_(7)
    assert int(pool.kv_views(4)[0].sum()) == 0 and int(pool.kv_views(6)[0].sum()) == 0


def test_geometry_name():
    # The default name is what KV-transfer region roles were keyed on before.
    assert str(KvGeometry(1, 128)) == "h1d128"
    assert str(KvGeometry(1, 128, "nvfp4")) == "h1d128nvfp4"

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 FP4 index / NVFP4 sparse KV: the startup refusal and the pool layout.

Both run without a GPU. The refusal keeps a deployment the kernels cannot serve
from starting (it would otherwise fail inside a Gluon compile, or on gfx942 inside
an inline-asm build); the pool test pins the byte layout the kernels address.
"""

import pytest
import torch

from atom.config import (
    m3_fp4_cache_unsupported_reason,
    m3_nvfp4_stage_unsupported_reason,
    m3_reuses_index_topk,
)
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


@pytest.mark.parametrize("index_fp4, kv_nvfp4", [(True, False), (False, True)])
def test_plugin_mode_refuses_fp4_caches(index_fp4, kv_nvfp4):
    # The vLLM / SGLang bridges bind their own caches: refused, not silently FP8.
    reason = m3_fp4_cache_unsupported_reason(
        "gfx950", index_fp4, kv_nvfp4, 4, HEADS, KV_HEADS, False, plugin=True
    )
    assert "plugin mode" in reason
    assert (
        m3_fp4_cache_unsupported_reason(
            "gfx950", False, False, 4, HEADS, KV_HEADS, False, plugin=True
        )
        is None
    )


# Four sparse layers out of six (layers 1 and 3 dense).
SPARSE = {"sparse_attention_config": {"sparse_attention_freq": [1, 0, 1, 0, 1, 1]}}


@pytest.mark.parametrize(
    "fields, reuses",
    [
        ({}, False),
        ({"use_index_cache": False, "index_topk_pattern": ["F", "S"]}, False),
        ({"use_index_cache": True}, False),
        ({"use_index_cache": True, "index_topk_freq": 4}, True),
        # An explicit pattern takes precedence over the frequency, both ways.
        (
            {
                "use_index_cache": True,
                "index_topk_freq": 1,
                "index_topk_pattern": ["F", "S"],
            },
            True,
        ),
        (
            {
                "use_index_cache": True,
                "index_topk_freq": 4,
                "index_topk_pattern": ["F", "F"],
            },
            False,
        ),
        # The offset shifts sharing past every sparse layer: none reuses.
        (
            {
                "use_index_cache": True,
                "index_topk_freq": 4,
                "index_skip_topk_offset": 1000,
            },
            False,
        ),
        # An "S" past the last sparse layer's ordinal (3) is never read.
        (
            {"use_index_cache": True, "index_topk_pattern": ["F", "F", "F", "F", "S"]},
            False,
        ),
        ({"use_index_cache": True, "index_topk_pattern": ["F", "F", "F", "S"]}, True),
    ],
)
def test_reuses_index_topk_follows_the_model_rule(fields, reuses):
    """The startup check's rule is the model's schedule over its sparse layers."""
    from types import SimpleNamespace

    from atom.models.minimax_m3.index_schedule import should_skip_index_topk

    cfg = SimpleNamespace(**SPARSE, **fields)
    assert m3_reuses_index_topk(cfg) is reuses
    assert reuses is any(should_skip_index_topk(cfg, layer)[0] for layer in range(6))


@pytest.mark.parametrize(
    "stage_pages, index_pages, refused",
    [
        (2, 1, True),  # a 256-row capture bucket over a one-page pool
        (128, 127, True),
        (128, 128, False),
        (128, 57011, False),  # M3.1 TP4 at --gpu-memory-utilization 0.9
    ],
)
def test_nvfp4_staging_must_fit_a_persistent_index_pool(
    stage_pages, index_pages, refused
):
    """AITER's page-128 writer checks the index cache against the KV staging
    pages it is given, and graph capture runs buckets larger than the pool, so
    a pool smaller than the staging buffer is refused at bind time."""
    reason = m3_nvfp4_stage_unsupported_reason(stage_pages, index_pages)
    assert (reason is not None) is refused
    if refused:
        assert "--index-cache-dtype fp4" in reason


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

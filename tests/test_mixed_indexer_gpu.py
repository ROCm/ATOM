# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Mixed indexer top-k vs independent forwards with real preshuffled FP8 GEMM."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm"
)


@pytest.mark.parametrize("prefill_tokens", [1, 7, 32])
def test_mixed_indexer_topk_matches_separately_normalized_segments(
    monkeypatch, prefill_tokens
):
    from aiter import (
        QuantType,
        dtypes,
        gemm_a8w8_blockscale_bpreshuffle,
        indexer_k_quant_and_cache,
    )
    from aiter.jit.utils.chip_info import get_gfx
    from aiter.ops.shuffle import shuffle_weight

    if get_gfx() not in ("gfx942", "gfx950"):
        pytest.skip("tests gfx942/gfx950's FP8 blockscale preshuffle GEMM")

    import atom.models.deepseek_v4 as model
    from atom.model_ops.indexer import IndexCache, concat_rows, fp8_rows, score_topk
    from atom.model_ops.indexer.layout import PRESHUFFLE
    from atom.model_ops.indexer.rows import ConcatKeys
    from atom.model_ops.layernorm import _aiter_rms_quant
    from atom.model_ops.v4_kernels.qk_norm_rope_maybe_quant import QKNormRopeOut

    torch.manual_seed(2722)
    rows, rank, heads, dim = prefill_tokens + 8, 256, 64, 128
    block, length, topk = 128, 4096, 512  # context = 16k, beyond 4 * index_topk
    latent = torch.randn(rows, rank, dtype=torch.bfloat16, device="cuda")
    # Distinct group scales in each row expose the invalid apparent-row slice.
    latent[:, :128] *= torch.linspace(0.1, 10, rows, device="cuda")[:, None]
    norm_weight = torch.ones(rank, dtype=torch.bfloat16, device="cuda")
    weight = shuffle_weight(
        torch.randn(heads * dim, rank, device="cuda").to(dtypes.fp8),
        layout=(16, 16),
    )
    weight_scale = torch.rand(heads * dim // 128, rank // 128, device="cuda") + 0.5
    angles = torch.randn(rows, 32, device="cuda")
    positions = torch.arange(rows, dtype=torch.int64, device="cuda")
    x = torch.randn(rows, heads, dtype=torch.bfloat16, device="cuda")

    def normalize(tensor):
        q, scales, _ = _aiter_rms_quant(
            tensor, norm_weight, 1e-6, QuantType.per_1x128.value, True
        )
        return q, scales

    indexer = SimpleNamespace(
        wq_b=lambda q, x_scale: gemm_a8w8_blockscale_bpreshuffle(
            q, weight, x_scale, weight_scale, torch.bfloat16
        ),
        weights_proj=lambda values: values,
        rotary_emb=SimpleNamespace(
            cos_cache=angles.cos().bfloat16(), sin_cache=angles.sin().bfloat16()
        ),
        n_heads=heads,
        head_dim=dim,
        rope_head_dim=64,
        _indexer_fp4=False,
        _layout=PRESHUFFLE,
        _weights_scale=(heads * dim) ** -0.5,
        prefix="mixed_test",
    )
    indexer.forward_pre = model.Indexer.forward_pre.__get__(indexer)
    keys = torch.randn(length, dim, dtype=torch.bfloat16, device="cuda")
    cache = torch.zeros(
        length // block, block, dim + 4, dtype=torch.uint8, device="cuda"
    )
    table = torch.arange(length // block, dtype=torch.int32, device="cuda")[None]
    indexer_k_quant_and_cache(
        keys, cache, torch.arange(length, device="cuda"), dim, "ue8m0", preshuffle=True
    )
    cached = IndexCache(cache, None, block)
    metadata = SimpleNamespace(
        num_prefill_tokens=prefill_tokens,
        num_decode_tokens=8,
        prefill_attn_metadata=object(),
        decode_attn_metadata=object(),
    )
    ctx = SimpleNamespace(is_prefill=True, input_ids=positions)
    fc = SimpleNamespace(context=ctx, attn_metadata=metadata)
    monkeypatch.setattr(model, "get_forward_context", lambda: fc)

    def select(q, weights, scale):
        assert scale is None
        n = q.shape[0]
        ends = torch.full((n,), length, dtype=torch.int32, device="cuda")
        row_meta = (
            concat_rows(
                ends,
                ConcatKeys(
                    table,
                    torch.tensor([0, length], dtype=torch.int32, device="cuda"),
                    length,
                    torch.zeros(n, dtype=torch.int32, device="cuda"),
                ),
            )
            if ctx.is_prefill
            else fp8_rows(ends, length, None, table.expand(n, -1).contiguous())
        )
        return score_topk(q, weights, cached, row_meta, topk=topk, stable=True)

    indexer.score_topk_from = select
    expected = []
    for sl, prefill in (
        (slice(0, prefill_tokens), True),
        (slice(prefill_tokens, rows), False),
    ):
        ctx.is_prefill = prefill
        q, scales = normalize(latent[sl])
        expected.append(select(*indexer.forward_pre(x[sl], q, positions[sl], scales)))
    ctx.is_prefill = True
    selected = []

    def attend(qkn, pos, *, out, idx_q_quant, idx_weights, idx_q_scale, **kwargs):
        selected.append(
            model.Indexer.topk(
                indexer,
                kwargs["x"],
                None,
                pos,
                pre_q_quant=idx_q_quant,
                pre_weights=idx_weights,
                pre_q_scale=idx_q_scale,
            )
        )
        out.zero_()

    query = torch.zeros(rows, 2, 2, dtype=torch.bfloat16, device="cuda")
    attention = SimpleNamespace(
        n_local_heads=2,
        head_dim=2,
        indexer=indexer,
        skip_topk=False,
        _qk_norm_rope=lambda q, kv, pos, mixed=False: QKNormRopeOut(q_sa=query),
        _launch_compressors=lambda *args: None,
        _sparse_attention=attend,
    )
    q, scales = normalize(latent)
    model.DeepseekV4Attention._attn_mixed(
        attention, metadata, query, query, q, scales, x, positions
    )
    for actual, reference in zip(selected, expected, strict=True):
        assert (actual >= 0).all() and actual.shape[1] == topk
        torch.testing.assert_close(
            actual.sort(1).values, reference.sort(1).values, rtol=0, atol=0
        )

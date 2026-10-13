# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Qwen3.8 decode fusions must be bitwise equal to the launches they replace.

Each flag (ATOM_QWEN38_MOE_ROUTER_BUCKETS, ATOM_QWEN38_QSA_FUSED_INDEX,
ATOM_QWEN38_QSA_FUSED_QK) only cuts kernel launches; any numeric difference
would shift greedy outputs, so every case here compares with `torch.equal`.
"""

import types

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "exercises Triton decode kernels; needs a real GPU",
        allow_module_level=True,
    )

import aiter

from atom.model_ops.layernorm import DualRMSNorm, GemmaRMSNorm
from atom.model_ops.qwen4_exp import qsa_attention as QA
from atom.model_ops.qwen4_exp.ops import qsa as Q

DEV = "cuda"
PAGE = 64
# Qwen3.8's rope: 64 of 256 channels rotate, interleaved mRoPE sections.
CONFIG = types.SimpleNamespace(
    head_dim=256,
    max_position_embeddings=262144,
    rope_parameters={
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
        "partial_rotary_factor": 0.25,
        "rope_theta": 10000000,
        "rope_type": "default",
    },
)


def _norm(dim):
    norm = GemmaRMSNorm(dim, eps=1e-6).to(DEV).bfloat16()
    with torch.no_grad():
        norm.weight.copy_(torch.randn(dim) * 0.3)
    return norm


@pytest.mark.parametrize("tokens,offset", [(12, 0), (24, 0), (48, 5), (64, 0)])
def test_qsa_qk_prep_matches_norm_rope_and_cache_write(tokens, offset):
    torch.manual_seed(tokens)
    hq, hk, dim = 6, 1, 256
    rope = QA.build_qwen4_exp_rope(CONFIG, dim).to(DEV)
    dual = DualRMSNorm(_norm(dim), _norm(dim), hq, hk, dim, prefix="t")
    unit = torch.ones(1, dtype=torch.float32, device=DEV)
    qkv = torch.randn(tokens, (2 * hq + 2 * hk) * dim, device=DEV).bfloat16() * 3
    _, q, k, v = qkv.split([hq * dim, hq * dim, hk * dim, hk * dim], -1)
    pos = torch.randint(0, 30000, (tokens,), device=DEV)
    positions = torch.stack([pos, pos + offset, pos + 2 * offset])
    slots = torch.randperm(200 * PAGE, device=DEV)[:tokens].int()
    slots[::7] = -1  # padded rows write nothing
    k_cache = torch.randn(200, PAGE, hk, dim, device=DEV).bfloat16()
    v_cache = torch.randn_like(k_cache)
    k_ref, v_ref = k_cache.clone(), v_cache.clone()

    qn, kn = dual(q, k)
    q_ref, key_ref = Q.qsa_apply_mrope(
        rope, positions, qn.view(tokens, hq, dim), kn.view(tokens, hk, dim)
    )
    aiter.reshape_and_cache_flash(
        key_ref, v.view(tokens, hk, dim), k_ref, v_ref, slots, "auto", unit, unit
    )

    assert Q.qsa_qk_prep_supported(dual, rope, q, k, v, k_cache, v_cache)
    q_new, key_new = Q.qsa_qk_prep(
        dual, rope, positions, q, k, v, slots, k_cache, v_cache
    )
    assert torch.equal(q_new, q_ref)
    assert torch.equal(key_new, key_ref)
    assert torch.equal(k_cache, k_ref)
    assert torch.equal(v_cache, v_ref)


@pytest.mark.parametrize(
    "requests,steps,use_positions,offset",
    [
        (8, 3, True, 0),
        (16, 3, True, 0),
        (4, 3, False, 0),
        (8, 3, True, 7),
        (16, 1, True, 0),
    ],
)
def test_qsa_index_prep_matches_unfused_indexer(requests, steps, use_positions, offset):
    heads, dim, ratio, pages, width = 4, 128, 4, 512, 64
    g = torch.Generator().manual_seed(requests * 10 + steps + offset)
    torch.manual_seed(requests * 10 + steps + offset)
    rope = QA.build_qwen4_exp_rope(CONFIG, dim).to(DEV)
    ix = types.SimpleNamespace(
        index_n_heads=heads,
        index_kv_heads=1,
        index_head_dim=dim,
        compress_ratio=ratio,
        rotary_emb=rope,
        q_layernorm=_norm(dim),
        k_layernorm=_norm(dim),
        index_qk_proj=None,
    )
    # Each request owns 24 distinct pages; groups that close this step pool
    # rows both from this step's tokens and from earlier cache contents.
    ctx = torch.randint(4, 1500, (requests,), generator=g)
    table = torch.full((requests, width), -1, dtype=torch.int32)
    table[:, :24] = torch.randperm(pages, generator=g)[: requests * 24].view(-1, 24)
    req, logical, slots, cslots = [], [], [], []
    for r in range(requests):
        for s in range(steps):
            p = int(ctx[r]) + s
            req.append(r)
            logical.append(p)
            slots.append(int(table[r, p // PAGE]) * PAGE + p % PAGE)
            cslots.append(r * 512 + p // ratio if (p + 1) % ratio == 0 else -1)
    n = len(req)
    as_dev = lambda x: torch.tensor(x, dtype=torch.int32, device=DEV)
    req, logical, slots, cslots = map(as_dev, (req, logical, slots, cslots))
    table = table.to(DEV)
    base = logical.to(torch.int64)
    positions = torch.stack([base + offset, base + 2 * offset, base + 3 * offset])
    qk = torch.randn(n, (heads + 1) * dim, device=DEV).bfloat16()
    raw0 = torch.randn(pages, PAGE, 1, dim, device=DEV).bfloat16()
    pos0 = torch.randint(0, 3000, (pages, PAGE, 1, 3), device=DEV)
    comp0 = torch.randn(requests * 512 // PAGE + 8, PAGE, 1, dim, device=DEV)
    comp0 = comp0.bfloat16()
    rope_offset = 0 if use_positions else 1

    def caches():
        return raw0.clone(), pos0.clone() if use_positions else None, comp0.clone()

    raw, pc, comp = caches()
    q_raw, token_k = qk.split((heads * dim, dim), dim=-1)
    q = ix.q_layernorm(q_raw.reshape(-1, dim))
    q_ref, _ = Q.qsa_apply_mrope(rope, positions, q.view(-1, heads, dim))
    Q.qsa_store_rows(raw, slots, token_k.reshape(-1, 1, dim))
    if pc is not None:
        Q.qsa_store_rows(pc, slots, QA.canonical_rope_positions(positions))
    pooled, first = Q.qsa_compress_groups(
        raw,
        table,
        req,
        logical,
        cslots,
        ratio,
        position_cache=pc,
        rope_position_offset=rope_offset,
    )
    Q.qsa_store_rows(
        comp, cslots, QA.Qwen4ExpIndexer.normalize_compressed_keys(ix, pooled, first)
    )
    ref = (q_ref, raw, pc, comp)

    raw, pc, comp = caches()
    assert Q.qsa_index_prep_supported(ix, qk, raw)
    q_new = Q.qsa_index_prep(
        ix, qk, positions, slots, cslots, req, logical, table, raw, comp, pc,
        rope_offset,
    )  # fmt: skip
    for a, b in zip(ref, (q_new, raw, pc, comp)):
        assert (a is None and b is None) or torch.equal(a, b)


@pytest.mark.parametrize("tokens", [12, 24, 48, 64])
def test_router_buckets_match_planned_moe(tokens):
    from aiter import dtypes
    from aiter.ops.shuffle import shuffle_weight

    from atom.model_ops import topK
    from atom.model_ops.qwen4_exp.ops import moe_small_batch as M
    from atom.model_ops.qwen4_exp.ops import router

    torch.manual_seed(tokens)
    experts, local, dim, inter, top_k, tp, rank = 512, 128, 2560, 640, 10, 4, 1
    slots = local + 1  # local experts + the fused shared expert
    emap = torch.full((experts + 2,), -1, dtype=torch.int32, device=DEV)
    emap[rank * local : (rank + 1) * local] = torch.arange(
        local, dtype=torch.int32, device=DEV
    )
    emap[experts] = local
    w1 = (torch.randn(slots, 2 * inter, dim, device=DEV) / 10).to(dtypes.fp8)
    w2 = (torch.randn(slots, dim, inter, device=DEV) / 10).to(dtypes.fp8)
    s1 = torch.rand(slots, 2 * inter // 128, dim // 128, device=DEV) * 0.01
    s2 = torch.rand(slots, dim // 128, inter // 128, device=DEV) * 0.01
    w1, w2 = shuffle_weight(w1, (16, 16)), shuffle_weight(w2, (16, 16))

    # The shared-expert tail slots the router appends, as `topK` lays them out.
    weights = torch.zeros(64, top_k + 2, dtype=torch.float32, device=DEV)
    ids = torch.full((64, top_k + 2), experts + 1, dtype=torch.int32, device=DEV)
    ids[rank::tp, top_k] = experts
    saved = topK.aiter_topK_meta_data
    topK.aiter_topK_meta_data = (weights, ids)
    try:
        x = torch.randn(tokens, dim, device=DEV).bfloat16()
        logits = torch.randn(tokens, experts + 1, device=DEV).bfloat16()
        # Crowd a few local experts so some take more than one row block.
        logits[:, rank * local : rank * local + 4] += 3
        buckets = M.route_buckets(torch.device(DEV), slots)

        def run(bk):
            tw, ti = router.softmax_topk(
                None, logits, top_k, True, experts,
                buckets=(emap, bk) if bk else None,
            )  # fmt: skip
            out = M.moe_small_batch(x, ti, tw, emap, w1, s1, w2, s2, buckets=bk)
            return out.clone()

        planned = run(None)
        assert torch.equal(run(buckets), planned)
        # The last router program must leave the counts and ticket zeroed.
        assert torch.equal(run(buckets), planned)
    finally:
        topK.aiter_topK_meta_data = saved

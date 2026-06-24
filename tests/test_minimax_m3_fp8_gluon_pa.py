# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 fp8 KV cache + gluon PA port — subclass framework integration.

These tests cover the port of the fp8 KV-cache + gluon paged-attention sparse
attention from ``origin/ganyi/shuffle_kv_cache_fp8_eagle`` onto main's attention
framework, rebuilt as a first-class ``PagedAttentionImpl`` subclass
(``SparseMHAPagedAttentionImpl``) selected through ``Attention(impl_cls=...)``.

Source line ranges grafted by later tasks (origin/ganyi/shuffle_kv_cache_fp8_eagle):
  sparse_attn.py:
    971-973  ASM_PAGE_SIZE / PAGES_PER_SPARSE_BLOCK constants
    231-358  _fused_qknorm_rope_kv_insert_shuffle_kernel
    360-419  minimax_m3_fused_qknorm_rope_kv_insert_shuffle (host wrapper)
    976-1085 _build_sparse_block_table_kernel + minimax_m3_build_sparse_block_table
    1087-... minimax_m3_sparse_attn_decode_asm (topk->block table->gluon decode)
    1223-1330 _build_sparse_block_table_prefill_kernel + ..._prefill wrapper
    1332-... minimax_m3_sparse_attn_prefill_asm
    1412-... _run_prefill_fp8_gluon / gluon decode runner (run_pa_decode_gluon)
  index_topk.py:
    EMIT_SPARSE_BT constexpr branch + emit_sparse_block_table arg
    (exclude MAX_Q / causal_len eagle3 spec-decode machinery)

Run (most GPU-free; gluon/insert tests require GPU):
    source /opt/venv/bin/activate
    python -m pytest tests/test_minimax_m3_fp8_gluon_pa.py -v
"""

from __future__ import annotations

import inspect

import pytest  # noqa: F401


# The repo conftest stubs atom.config for CPU scheduler tests; evict so the real
# atom.model_ops modules import (mirrors tests/test_minimax_m3_sparse_attn_asm.py).
def _restore_real_atom_modules():
    import sys

    for mod_name in list(sys.modules):
        if mod_name == "atom" or mod_name.startswith("atom."):
            del sys.modules[mod_name]


_restore_real_atom_modules()


# ── Task 0: subclass selection ────────────────────────────────────────────────


def test_sparse_impl_is_paged_attention_subclass():
    """SparseMHAPagedAttentionImpl extends the standard MHA impl so it reuses
    forward / forward_impl and only overrides rope_cache + dispatch_backend."""
    from atom.model_ops.attention_mha import (
        PagedAttentionImpl,
        SparseMHAPagedAttentionImpl,
    )

    assert issubclass(SparseMHAPagedAttentionImpl, PagedAttentionImpl)


def test_sparse_impl_overrides_only_rope_cache_and_dispatch():
    """The subclass must override exactly rope_cache and dispatch_backend (the two
    framework hooks forward_impl calls) and inherit everything else."""
    from atom.model_ops.attention_mha import (
        PagedAttentionImpl,
        SparseMHAPagedAttentionImpl,
    )

    for name in ("rope_cache", "dispatch_backend"):
        assert (
            SparseMHAPagedAttentionImpl.__dict__.get(name) is not None
        ), f"{name} must be overridden on the subclass"
    # forward_impl/forward stay inherited (confirmed design: standard forward)
    assert "forward_impl" not in SparseMHAPagedAttentionImpl.__dict__
    assert "forward" not in SparseMHAPagedAttentionImpl.__dict__
    assert SparseMHAPagedAttentionImpl.forward_impl is PagedAttentionImpl.forward_impl


def test_attention_accepts_impl_cls_kwarg():
    """Attention.__init__ exposes an impl_cls override so a model can plug in the
    sparse impl while keeping the backend's metadata builder."""
    from atom.model_ops.paged_attention import Attention

    params = inspect.signature(Attention.__init__).parameters
    assert "impl_cls" in params, "Attention.__init__ must accept impl_cls"
    # Default None so existing models fall back to attn_backend.get_impl_cls().
    assert params["impl_cls"].default is None


# ── Task 1: page-16 constants + fused SHUFFLE KV-insert ────────────────────────

import torch  # noqa: E402

_HAS_CUDA = torch.cuda.is_available()
_gpu = pytest.mark.skipif(not _HAS_CUDA, reason="requires CUDA/ROCm")

HEAD_DIM = 128
ROTARY_DIM = 64


def _gemma_rmsnorm(x, weight, eps):
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    return xf * torch.rsqrt(var + eps) * (1.0 + weight.float())


def _apply_rope_neox_partial(x, positions, cos_sin_cache, rotary_dim):
    half = rotary_dim // 2
    cos_sin = cos_sin_cache[positions].float()
    cos = cos_sin[..., :half].unsqueeze(1)
    sin = cos_sin[..., half:].unsqueeze(1)
    rot = x[..., :rotary_dim]
    x1 = rot[..., :half]
    x2 = rot[..., half:]
    out = x.clone()
    out[..., :half] = x1 * cos - x2 * sin
    out[..., half:rotary_dim] = x2 * cos + x1 * sin
    return out


def _norm_rope_ref(x, weight, positions, cos_sin_cache, eps, dtype):
    normed = _gemma_rmsnorm(x.float(), weight, eps)
    return _apply_rope_neox_partial(normed, positions, cos_sin_cache, ROTARY_DIM).to(
        dtype
    )


def _make_cos_sin_cache(max_pos, rotary_dim, dtype):
    base = 5_000_000.0
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device="cuda")
            / rotary_dim
        )
    )
    positions = torch.arange(max_pos, dtype=torch.float32, device="cuda")
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dtype)


def test_page16_constants():
    from atom.model_ops.minimax_m3.sparse_attn import (
        ASM_PAGE_SIZE,
        PAGES_PER_SPARSE_BLOCK,
        SPARSE_BLOCK_SIZE,
    )

    assert ASM_PAGE_SIZE == 16
    assert PAGES_PER_SPARSE_BLOCK == SPARSE_BLOCK_SIZE // ASM_PAGE_SIZE == 8


@_gpu
def test_fused_qknorm_rope_kv_insert_shuffle_roundtrip():
    """The fused Gemma-RMSNorm + partial-NeoX-RoPE + page-16 SHUFFLE KV insert
    must match a pure-PyTorch reference for q_out/index_q_out and round-trip the
    K/V/index caches at each token's slot."""
    from atom.model_ops.minimax_m3.sparse_attn import (
        minimax_m3_fused_qknorm_rope_kv_insert_shuffle,
    )

    torch.manual_seed(123)
    dtype = torch.bfloat16
    eps = 1e-6
    max_pos = 4096
    num_tokens = 17
    block_size = 16  # ASM_PAGE
    num_heads, num_kv_heads, num_index_heads = 16, 4, 4
    x = 16 // dtype.itemsize  # bf16 -> 8

    q_w = torch.randn(HEAD_DIM, dtype=dtype, device="cuda") * 0.1
    k_w = torch.randn(HEAD_DIM, dtype=dtype, device="cuda") * 0.1
    iq_w = torch.randn(HEAD_DIM, dtype=dtype, device="cuda") * 0.1
    ik_w = torch.randn(HEAD_DIM, dtype=dtype, device="cuda") * 0.1
    cos_sin = _make_cos_sin_cache(max_pos, ROTARY_DIM, dtype)
    positions = torch.randint(
        0, max_pos, (num_tokens,), dtype=torch.int64, device="cuda"
    )

    q_size = num_heads * HEAD_DIM
    kv_size = num_kv_heads * HEAD_DIM
    iq_size = num_index_heads * HEAD_DIM
    ik_size = HEAD_DIM
    qkv = torch.randn(
        num_tokens, q_size + 2 * kv_size + iq_size + ik_size, dtype=dtype, device="cuda"
    )
    qkv_orig = qkv.clone()

    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    num_phys_blocks = num_blocks + 1
    slot_mapping = torch.randperm(
        num_phys_blocks * block_size, dtype=torch.int64, device="cuda"
    )[:num_tokens]

    kv_cache_k = torch.zeros(
        num_phys_blocks,
        num_kv_heads,
        HEAD_DIM // x,
        block_size,
        x,
        dtype=dtype,
        device="cuda",
    )
    kv_cache_v = torch.zeros(
        num_phys_blocks,
        num_kv_heads,
        block_size // x,
        HEAD_DIM,
        x,
        dtype=dtype,
        device="cuda",
    )
    index_cache = torch.zeros(
        num_phys_blocks, block_size, HEAD_DIM, dtype=dtype, device="cuda"
    )
    q_out = torch.empty(num_tokens, q_size, dtype=dtype, device="cuda")
    index_q_out = torch.empty(num_tokens, iq_size, dtype=dtype, device="cuda")

    minimax_m3_fused_qknorm_rope_kv_insert_shuffle(
        qkv,
        q_w,
        k_w,
        cos_sin,
        positions,
        num_heads,
        num_kv_heads,
        ROTARY_DIM,
        eps,
        iq_w,
        ik_w,
        num_index_heads,
        slot_mapping,
        kv_cache_k,
        kv_cache_v,
        index_cache,
        q_out,
        index_q_out,
        HEAD_DIM,
    )

    q_in, k_in, v_in, iq_in, ik_in = qkv_orig.split(
        [q_size, kv_size, kv_size, iq_size, ik_size], dim=-1
    )
    q_ref = _norm_rope_ref(
        q_in.view(num_tokens, num_heads, HEAD_DIM), q_w, positions, cos_sin, eps, dtype
    ).view(num_tokens, q_size)
    iq_ref = _norm_rope_ref(
        iq_in.view(num_tokens, num_index_heads, HEAD_DIM),
        iq_w,
        positions,
        cos_sin,
        eps,
        dtype,
    ).view(num_tokens, iq_size)
    k_ref = _norm_rope_ref(
        k_in.view(num_tokens, num_kv_heads, HEAD_DIM),
        k_w,
        positions,
        cos_sin,
        eps,
        dtype,
    )
    ik_ref = _norm_rope_ref(
        ik_in.view(num_tokens, 1, HEAD_DIM), ik_w, positions, cos_sin, eps, dtype
    ).view(num_tokens, HEAD_DIM)
    v_ref = v_in.view(num_tokens, num_kv_heads, HEAD_DIM)

    torch.testing.assert_close(q_out, q_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(index_q_out, iq_ref, rtol=1e-2, atol=1e-2)

    d = torch.arange(HEAD_DIM, device="cuda")
    for token in range(num_tokens):
        slot = slot_mapping[token].item()
        phys, intra = slot // block_size, slot % block_size
        for h in range(num_kv_heads):
            k_back = kv_cache_k[phys, h, d // x, intra, d % x]
            torch.testing.assert_close(k_back, k_ref[token, h], rtol=1e-2, atol=1e-2)
            v_back = kv_cache_v[phys, h, intra // x, d, intra % x]
            torch.testing.assert_close(v_back, v_ref[token, h], rtol=0, atol=0)
        torch.testing.assert_close(
            index_cache.view(-1, HEAD_DIM)[slot], ik_ref[token], rtol=1e-2, atol=1e-2
        )


# ── Task 2: sparse block-table builders + topk EMIT glue ───────────────────────


@_gpu
def test_build_sparse_block_table_expands_x8_tail_last():
    """Each selected logical 128-block expands to 8 contiguous physical 16-pages
    (logical_page*8 + j); the partial tail block is packed last; context_lens
    reflect full blocks (128 each) + the tail remainder."""
    from atom.model_ops.minimax_m3.sparse_attn import (
        PAGES_PER_SPARSE_BLOCK,
        SPARSE_BLOCK_SIZE,
        minimax_m3_build_sparse_block_table,
    )

    dev = "cuda"
    G = PAGES_PER_SPARSE_BLOCK  # 8
    topk = 3
    batch = 2
    # logical block_table maps logical 128-block -> physical 128-page id.
    # req 0: logical blocks {0,1,2} -> phys {5,6,7}
    # req 1: logical blocks {0,1}   -> phys {3,4}
    block_table = torch.tensor(
        [[5, 6, 7, 0], [3, 4, 0, 0]], dtype=torch.int32, device=dev
    )
    # req 0: seq_len 300 -> last (tail) logical block = (300-1)//128 = 2
    # req 1: seq_len 200 -> tail logical block = (200-1)//128 = 1
    seq_lens = torch.tensor([300, 200], dtype=torch.int32, device=dev)
    # topk_idx [1, batch, topk] 0-indexed logical 128-blocks, -1 pad.
    # req 0 selects {0,1,2} (2 is the tail). req 1 selects {0,1} + pad.
    topk_idx = torch.tensor([[[0, 1, 2], [0, 1, -1]]], dtype=torch.int32, device=dev)

    sparse_bt, sparse_ctx = minimax_m3_build_sparse_block_table(
        topk_idx, block_table, seq_lens
    )

    assert sparse_bt.shape == (batch, topk * G)
    assert sparse_ctx.shape == (batch,)

    # req 0: full blocks {0,1} -> phys logical {5,6} first (slots 0,1), tail
    # block 2 -> phys logical 7 last (slot 2). Each expands *8.
    # slot order: [block0 pages, block1 pages, tail(block2) pages]
    exp0 = []
    for logical_phys in (5, 6, 7):  # full(0)->5, full(1)->6, tail(2)->7
        exp0 += [logical_phys * G + j for j in range(G)]
    torch.testing.assert_close(
        sparse_bt[0].cpu(), torch.tensor(exp0, dtype=torch.int32)
    )
    # ctx: 2 full blocks * 128 + tail tokens (300 - 2*128 = 44) = 300
    assert sparse_ctx[0].item() == 300

    # req 1: full block {0}->phys5? no: req1 block_table phys {3,4}. tail is
    # logical block 1 -> phys 4 (packed last); full block 0 -> phys 3 (slot 0).
    exp1 = []
    for logical_phys in (3, 4):  # full(0)->3, tail(1)->4
        exp1 += [logical_phys * G + j for j in range(G)]
    # remaining (topk*G - 2*G) slots are zero-padded.
    exp1 += [0] * (topk * G - len(exp1))
    torch.testing.assert_close(
        sparse_bt[1].cpu(), torch.tensor(exp1, dtype=torch.int32)
    )
    # ctx: 1 full block *128 + tail (200 - 1*128 = 72) = 200
    assert sparse_ctx[1].item() == 200
    assert SPARSE_BLOCK_SIZE == 128


@_gpu
def test_topk_decode_emit_matches_standalone_builder():
    """The fused EMIT_SPARSE_BT path in minimax_m3_index_topk_decode must produce
    the same (sparse_bt, sparse_ctx) as the standalone builder run on its
    topk_idx (num_idx_heads == 1)."""
    from atom.model_ops.minimax_m3.index_topk import minimax_m3_index_topk_decode
    from atom.model_ops.minimax_m3.sparse_attn import (
        minimax_m3_build_sparse_block_table,
    )

    dev = "cuda"
    torch.manual_seed(7)
    head_dim = 128
    num_kv_heads = 1
    batch = 5
    topk, init_blocks, local_blocks = 4, 1, 1
    max_seq_len = 1024
    sm_scale = head_dim**-0.5

    max_blocks = max_seq_len // 128
    # index_kv_cache must hold every physical page referenced by block_table.
    num_blocks = batch * max_blocks + 1
    idx_q = torch.randn(batch, num_kv_heads, head_dim, device=dev) * 0.1
    index_kv_cache = torch.randn(num_blocks, 128, head_dim, device=dev) * 0.1
    block_table = torch.arange(
        batch * max_blocks, dtype=torch.int32, device=dev
    ).reshape(batch, max_blocks)
    seq_lens = torch.randint(130, max_seq_len, (batch,), dtype=torch.int32, device=dev)

    # non-emit: plain topk_idx, then standalone builder
    topk_idx = minimax_m3_index_topk_decode(
        idx_q,
        index_kv_cache,
        block_table,
        seq_lens,
        max_seq_len,
        topk,
        init_blocks,
        local_blocks,
        num_kv_heads,
        sm_scale,
    )
    sbt_ref, sctx_ref = minimax_m3_build_sparse_block_table(
        topk_idx, block_table, seq_lens
    )

    # emit: fused inside the merge kernel
    topk_idx2, sbt, sctx = minimax_m3_index_topk_decode(
        idx_q,
        index_kv_cache,
        block_table,
        seq_lens,
        max_seq_len,
        topk,
        init_blocks,
        local_blocks,
        num_kv_heads,
        sm_scale,
        emit_sparse_block_table=True,
    )

    torch.testing.assert_close(topk_idx2, topk_idx)
    torch.testing.assert_close(sctx, sctx_ref)
    torch.testing.assert_close(sbt, sbt_ref)

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Equivalence test: the sparse block-table fused into the topk kernels must
match the standalone builders byte-for-byte (decode + prefill)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from aiter import dtypes


def _restore_real_atom_modules():
    import sys

    for mod_name in list(sys.modules):
        if mod_name == "atom" or mod_name.startswith("atom."):
            del sys.modules[mod_name]


_restore_real_atom_modules()

from atom.model_ops.minimax_m3.index_topk import (  # noqa: E402
    minimax_m3_index_topk,
    minimax_m3_index_topk_decode,
    _topk_index_merge_kernel,
    PAGES_PER_SPARSE_BLOCK,
    SPARSE_BLOCK_SIZE,
)
from atom.model_ops.minimax_m3.sparse_attn import (  # noqa: E402
    minimax_m3_build_sparse_block_table,
    minimax_m3_build_sparse_block_table_prefill,
)
from atom.model_ops.attentions.aiter_attention import (  # noqa: E402
    AiterAttentionMetadataBuilder,
)
from aiter.ops.minimax_m3_topk_merge import minimax_m3_topk_index_merge  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA/ROCm"
)

HEAD_DIM = 128


def _fp8_dtype() -> torch.dtype:
    if dtypes.fp8 is not torch.uint8:
        return dtypes.fp8
    return getattr(torch, "float8_e4m3fnuz", getattr(torch, "float8_e4m3fn", None))


def _to_fp8(x: torch.Tensor) -> torch.Tensor:
    fp8_dtype = _fp8_dtype()
    assert fp8_dtype is not None
    return x.float().to(fp8_dtype)


def test_native_sparse_index_cache_allocation_uses_explicit_fp8_index_dtype():
    sparse_cfg = {
        "sparse_attention_freq": [True, False, True],
        "sparse_index_dim": HEAD_DIM,
    }
    hf_config = SimpleNamespace(
        num_hidden_layers=3,
        head_dim=HEAD_DIM,
        sparse_attention_config=sparse_cfg,
        use_index_cache=True,
    )
    config = SimpleNamespace(
        hf_config=hf_config,
        kv_cache_dtype="fp8",
        index_cache_dtype="fp8",
        torch_dtype=torch.bfloat16,
    )

    builder = object.__new__(AiterAttentionMetadataBuilder)
    builder.model_runner = SimpleNamespace(
        config=config,
        num_physical_kvcache_blocks=4,
        physical_block_size=128,
        is_mimo_v2=lambda: False,
    )

    tensors = builder.allocate_kv_cache_tensors(num_kv_heads=1, num_draft_layers=0)

    assert tensors["sparse_attention_index_cache"].dtype == dtypes.d_dtypes["fp8"]
    assert "sparse_attention_index_cache_scale" not in tensors


def test_native_sparse_index_cache_allocation_supports_fp8_index_dtype_with_bf16_kv():
    sparse_cfg = {
        "sparse_attention_freq": [True, False, True],
        "sparse_index_dim": HEAD_DIM,
    }
    hf_config = SimpleNamespace(
        num_hidden_layers=3,
        head_dim=HEAD_DIM,
        sparse_attention_config=sparse_cfg,
        use_index_cache=True,
    )
    config = SimpleNamespace(
        hf_config=hf_config,
        kv_cache_dtype="bf16",
        index_cache_dtype="fp8",
        torch_dtype=torch.bfloat16,
    )

    builder = object.__new__(AiterAttentionMetadataBuilder)
    builder.model_runner = SimpleNamespace(
        config=config,
        num_physical_kvcache_blocks=4,
        physical_block_size=128,
        is_mimo_v2=lambda: False,
    )

    tensors = builder.allocate_kv_cache_tensors(num_kv_heads=1, num_draft_layers=0)

    assert tensors["kv_cache"].dtype == dtypes.d_dtypes["bf16"]
    assert tensors["sparse_attention_index_cache"].dtype == dtypes.d_dtypes["fp8"]
    assert "sparse_attention_index_cache_scale" not in tensors


def test_native_sparse_index_cache_allocation_auto_uses_torch_dtype_with_bf16_kv():
    sparse_cfg = {
        "sparse_attention_freq": [True],
        "sparse_index_dim": HEAD_DIM,
    }
    hf_config = SimpleNamespace(
        num_hidden_layers=1,
        head_dim=HEAD_DIM,
        sparse_attention_config=sparse_cfg,
        use_index_cache=True,
    )
    config = SimpleNamespace(
        hf_config=hf_config,
        kv_cache_dtype="bf16",
        index_cache_dtype="auto",
        torch_dtype=torch.bfloat16,
    )

    builder = object.__new__(AiterAttentionMetadataBuilder)
    builder.model_runner = SimpleNamespace(
        config=config,
        num_physical_kvcache_blocks=4,
        physical_block_size=128,
        is_mimo_v2=lambda: False,
    )

    tensors = builder.allocate_kv_cache_tensors(num_kv_heads=1, num_draft_layers=0)

    assert tensors["sparse_attention_index_cache"].dtype == torch.bfloat16


@pytest.mark.parametrize(
    "seq_lens", [[300, 128, 129, 900, 50], [128], [127], [1, 256, 384]]
)
def test_fused_decode_block_table_matches_builder(seq_lens):
    torch.manual_seed(0)
    dev = "cuda"
    nidx = 1
    topk = 16
    batch = len(seq_lens)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
    max_seq = int(seq_lens_t.max())
    max_block = (max_seq + 127) // 128
    nblk_total = batch * max_block + 4
    idx_q = torch.randn(batch, nidx, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    idxc = torch.randn(nblk_total, 128, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    bt = (
        torch.randperm(nblk_total, device=dev)[: batch * max_block]
        .view(batch, max_block)
        .to(torch.int32)
    )
    sm = HEAD_DIM**-0.5

    tk, sbt_f, sctx_f = minimax_m3_index_topk_decode(
        idx_q,
        idxc,
        bt,
        seq_lens_t,
        max_seq,
        topk,
        0,
        1,
        nidx,
        sm,
        emit_sparse_block_table=True,
    )
    sbt_r, sctx_r = minimax_m3_build_sparse_block_table(tk, bt, seq_lens_t)
    assert torch.equal(sbt_f, sbt_r)
    assert torch.equal(sctx_f, sctx_r)


def test_decode_topk_accepts_fp8_index_cache():
    torch.manual_seed(1)
    dev = "cuda"
    nidx = 1
    topk = 4
    seq_lens_t = torch.tensor([256, 384], dtype=torch.int32, device=dev)
    batch = seq_lens_t.numel()
    max_seq = int(seq_lens_t.max())
    max_block = (max_seq + 127) // 128
    nblk_total = batch * max_block + 4
    idx_q = torch.randn(batch, nidx, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    idxc_src = torch.randn(nblk_total, 128, HEAD_DIM, dtype=torch.float32, device=dev)
    # Give each block a distinct offset so FP8 rounding does not alter top-k ties.
    idxc_src += torch.arange(nblk_total, dtype=torch.float32, device=dev).view(-1, 1, 1)
    idxc_fp8 = _to_fp8(idxc_src)
    idxc_dequant = idxc_fp8.float().to(torch.bfloat16)
    bt = torch.arange(batch * max_block, dtype=torch.int32, device=dev).view(
        batch, max_block
    )
    sm = HEAD_DIM**-0.5

    tk_fp8 = minimax_m3_index_topk_decode(
        idx_q,
        idxc_fp8,
        bt,
        seq_lens_t,
        max_seq,
        topk,
        0,
        0,
        nidx,
        sm,
    )
    tk_ref = minimax_m3_index_topk_decode(
        idx_q, idxc_dequant, bt, seq_lens_t, max_seq, topk, 0, 0, nidx, sm
    )
    assert torch.equal(tk_fp8, tk_ref)


@pytest.mark.parametrize(
    "seq_lens,prefix_lens",
    [
        ([200, 129, 300], [0, 0, 0]),
        ([300], [0]),
        ([129], [0]),
        ([200, 130, 300], [64, 128, 200]),  # chunked prefill
    ],
)
def test_fused_prefill_block_table_matches_builder(seq_lens, prefix_lens):
    torch.manual_seed(0)
    dev = "cuda"
    nidx = 1
    topk = 16
    batch = len(seq_lens)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
    prefix_lens_t = torch.tensor(prefix_lens, dtype=torch.int32, device=dev)
    q_lens = [seq_lens[b] - prefix_lens[b] for b in range(batch)]
    total_q = sum(q_lens)
    max_q = max(q_lens)
    cu = torch.zeros(batch + 1, dtype=torch.int32, device=dev)
    cu[1:] = torch.tensor(q_lens, dtype=torch.int32, device=dev).cumsum(0)
    max_seq = max(seq_lens)
    max_block = (max_seq + 127) // 128
    nblk_total = batch * max_block + 4
    idx_q = torch.randn(total_q, nidx, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    idxc = torch.randn(nblk_total, 128, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    bt = (
        torch.randperm(nblk_total, device=dev)[: batch * max_block]
        .view(batch, max_block)
        .to(torch.int32)
    )
    sm = HEAD_DIM**-0.5

    tk, sbt_f, sctx_f = minimax_m3_index_topk(
        idx_q,
        idxc,
        bt,
        cu,
        seq_lens_t,
        prefix_lens_t,
        max_q,
        max_seq,
        topk,
        0,
        1,
        nidx,
        sm,
        emit_sparse_block_table=True,
    )
    req = torch.repeat_interleave(
        torch.arange(batch, dtype=torch.int32, device=dev),
        torch.tensor(q_lens, device=dev),
    )
    absp = (
        prefix_lens_t[req]
        + (torch.arange(total_q, dtype=torch.int32, device=dev) - cu[req])
    ).to(torch.int32)
    sbt_r, sctx_r = minimax_m3_build_sparse_block_table_prefill(tk, bt, req, absp)
    assert torch.equal(sbt_f, sbt_r)
    assert torch.equal(sctx_f, sctx_r)


def test_prefill_topk_accepts_fp8_index_cache():
    torch.manual_seed(2)
    dev = "cuda"
    nidx = 1
    topk = 4
    seq_lens_t = torch.tensor([256, 384], dtype=torch.int32, device=dev)
    prefix_lens_t = torch.tensor([128, 256], dtype=torch.int32, device=dev)
    q_lens = (seq_lens_t - prefix_lens_t).tolist()
    total_q = sum(q_lens)
    max_q = max(q_lens)
    cu = torch.zeros(seq_lens_t.numel() + 1, dtype=torch.int32, device=dev)
    cu[1:] = torch.tensor(q_lens, dtype=torch.int32, device=dev).cumsum(0)
    max_seq = int(seq_lens_t.max())
    max_block = (max_seq + 127) // 128
    nblk_total = seq_lens_t.numel() * max_block + 4
    idx_q = torch.randn(total_q, nidx, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    idxc_src = torch.randn(nblk_total, 128, HEAD_DIM, dtype=torch.float32, device=dev)
    idxc_src += torch.arange(nblk_total, dtype=torch.float32, device=dev).view(-1, 1, 1)
    idxc_fp8 = _to_fp8(idxc_src)
    idxc_dequant = idxc_fp8.float().to(torch.bfloat16)
    bt = torch.arange(
        seq_lens_t.numel() * max_block, dtype=torch.int32, device=dev
    ).view(seq_lens_t.numel(), max_block)
    sm = HEAD_DIM**-0.5

    idx_q_fp8_ref = _to_fp8(idx_q.float()).float().to(torch.bfloat16)
    tk_fp8 = minimax_m3_index_topk(
        idx_q,
        idxc_fp8,
        bt,
        cu,
        seq_lens_t,
        prefix_lens_t,
        max_q,
        max_seq,
        topk,
        0,
        0,
        nidx,
        sm,
    )
    tk_ref = minimax_m3_index_topk(
        idx_q_fp8_ref,
        idxc_dequant,
        bt,
        cu,
        seq_lens_t,
        prefix_lens_t,
        max_q,
        max_seq,
        topk,
        0,
        0,
        nidx,
        sm,
    )
    assert torch.equal(tk_fp8, tk_ref)


def test_hip_decode_merge_matches_triton_merge():
    torch.manual_seed(5)
    dev = "cuda"
    num_chunks = 16
    nidx = 1
    total_q = 17
    topk = 16
    block_size_t = 16
    max_blocks = 128
    seq_lens = torch.tensor(
        [
            128,
            257,
            1024,
            2048,
            4096,
            8192,
            16384,
            300,
            129,
            64,
            777,
            3333,
            12000,
            555,
            666,
            7777,
            8888,
        ],
        dtype=torch.int32,
        device=dev,
    )
    bt = torch.arange(total_q * max_blocks, dtype=torch.int32, device=dev).view(
        total_q, max_blocks
    )
    ts = torch.full(
        (num_chunks, nidx, total_q, block_size_t),
        -1e30,
        dtype=torch.float32,
        device=dev,
    )
    ti = torch.zeros(
        num_chunks, nidx, total_q, block_size_t, dtype=torch.int32, device=dev
    )
    for row in range(total_q):
        valid_blocks = int(
            (int(seq_lens[row]) + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
        )
        candidates = torch.arange(1, valid_blocks + 1, dtype=torch.int32, device=dev)
        scores = torch.randn(valid_blocks, dtype=torch.float32, device=dev)
        order = torch.argsort(scores, descending=True)
        candidates = candidates[order]
        scores = scores[order]
        for chunk in range(num_chunks):
            start = chunk * block_size_t
            stop = min(start + block_size_t, valid_blocks)
            if start < stop:
                width = stop - start
                ts[chunk, 0, row, :width] = scores[start:stop]
                ti[chunk, 0, row, :width] = candidates[start:stop]
    ts[0, 0, :, 0] = 1e30
    ti[0, 0, :, 0] = 1
    ts[1, 0, :, 0] = 1e29
    ti[1, 0, :, 0] = 1

    def run_triton():
        out = torch.empty(nidx, total_q, topk, dtype=torch.int32, device=dev)
        sbt = torch.empty(
            total_q * nidx, topk * PAGES_PER_SPARSE_BLOCK, dtype=torch.int32, device=dev
        )
        ctx = torch.empty(total_q * nidx, dtype=torch.int32, device=dev)
        _topk_index_merge_kernel[(total_q, nidx)](
            ts,
            ti,
            out,
            seq_lens,
            SPARSE_BLOCK_SIZE,
            topk,
            ts.stride(0),
            ts.stride(1),
            ts.stride(2),
            ts.stride(3),
            ti.stride(0),
            ti.stride(1),
            ti.stride(2),
            ti.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            bt,
            sbt,
            ctx,
            bt.stride(0),
            sbt.stride(0),
            MAX_Q=1,
            NUM_KV_HEADS=nidx,
            NUM_TOPK_CHUNKS=num_chunks,
            pages_per_block=PAGES_PER_SPARSE_BLOCK,
            EMIT_SPARSE_BT=True,
        )
        return out, sbt, ctx

    def run_hip():
        out = torch.empty(nidx, total_q, topk, dtype=torch.int32, device=dev)
        sbt = torch.empty(
            total_q * nidx, topk * PAGES_PER_SPARSE_BLOCK, dtype=torch.int32, device=dev
        )
        ctx = torch.empty(total_q * nidx, dtype=torch.int32, device=dev)
        minimax_m3_topk_index_merge(
            ts,
            ti,
            out,
            seq_lens,
            SPARSE_BLOCK_SIZE,
            topk,
            bt,
            sbt,
            ctx,
            1,
            nidx,
            PAGES_PER_SPARSE_BLOCK,
            True,
        )
        return out, sbt, ctx

    ref = run_triton()
    got = run_hip()
    assert torch.equal(got[0], ref[0])
    assert torch.equal(got[1], ref[1])
    assert torch.equal(got[2], ref[2])

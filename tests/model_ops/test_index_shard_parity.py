# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GPU parity for a sharded MiniMax-M3 index cache (``ATOM_M3_INDEX_SHARD``).

Rank r keeps block b's index rows iff ``b % WORLD == r``, at local page
``b // WORLD`` (`minimax_m3/index_shard.py`). Against the replicated cache, on
one device (the collectives are data movement: a rank's planes are
``replicated[r::WORLD]``, the prefill all-gather is the ranks' send buffers
concatenated in rank order):

- the fused cache write with `owned_index_slots` writes each rank's rows, and
  only them, as the replicated write does;
- the CP decode scorers (Triton, FlyDSL) read the local planes through
  `page_view` and score exactly as from the replicated cache, and the CP chain
  selects what the TP path does;
- the prefill top-k over the gathered pages, read through the gather's table,
  selects and emits exactly what the replicated cache gives.

Every read stays inside its tensors: the scorers only read owned blocks, and
the planes sit after a pad as large as `page_view` reaches back.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("aiter", reason="requires the AITER runtime")
pytest.importorskip("triton", reason="requires Triton")

from atom.model_ops.minimax_m3.index_fp4 import index_fp4_store
from atom.model_ops.minimax_m3.index_shard import (
    gathered_planes,
    owned_index_slots,
    pack_index_pages,
    page_view,
    plan_index_gather,
)

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a ROCm GPU"
)

WORLD = 4
BLOCK = 128
D = 128
SCALE = 128**-0.5


def laned_tables(seq_lens, pages, generator):
    """Block tables whose logical block j holds a page ``== j mod WORLD``, in a
    random order within each lane, as the sharded allocator hands them out."""
    lanes = [
        (lane + WORLD * torch.randperm(pages // WORLD, generator=generator)).tolist()
        for lane in range(WORLD)
    ]
    tables = []
    for seq_len in seq_lens:
        tables.append([lanes[j % WORLD].pop() for j in range(-(-seq_len // BLOCK))])
    return tables


def fp4_cache(pages, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    keys = torch.randn(pages * BLOCK, D, device="cuda", generator=g) * 3
    payload = torch.zeros(pages, BLOCK, D // 2, dtype=torch.uint8, device="cuda")
    scale = torch.zeros(pages, BLOCK, D // 16, dtype=torch.uint8, device="cuda")
    index_fp4_store(
        keys.to(torch.bfloat16),
        torch.arange(pages * BLOCK, device="cuda"),
        payload,
        scale,
    )
    return payload, scale


def local_planes(plane, rank):
    """Rank ``rank``'s local plane of a replicated one, after a one-page pad
    (`page_view` reaches ``rank / WORLD`` of a page back)."""
    local = plane[rank::WORLD]
    store = torch.zeros(
        (local.shape[0] + 1, *local.shape[1:]), dtype=plane.dtype, device="cuda"
    )
    store[1:] = local
    return store[1:]


@needs_gpu
@pytest.mark.parametrize("tokens", [37, 2000])
def test_the_fused_write_lands_each_rows_at_its_owner(tokens):
    import aiter

    from atom.model_ops.attentions.mha_kv_pool import MhaKvPool

    heads, hkv, rot = 16, 1, 64
    blocks = -(-(-(-tokens // BLOCK) + 4) // WORLD) * WORLD
    g = torch.Generator(device="cuda").manual_seed(23)
    row = (heads + 3 * hkv + 1) * D
    qkv = torch.randn(tokens, row, generator=g, device="cuda").to(torch.bfloat16)
    w = [
        torch.randn(D, generator=g, device="cuda").to(torch.bfloat16) * 0.3 + 1
        for _ in range(4)
    ]
    cos_sin = torch.randn(8192, rot, generator=g, device="cuda").to(torch.bfloat16)
    pos = torch.randint(0, 8192, (tokens,), device="cuda")
    slots = torch.randperm(blocks * BLOCK, device="cuda")[:tokens].to(torch.int64)
    slots[2::29] = -1

    def write(index_pages, index_slots):
        pool = MhaKvPool(
            layers=1, block_size=BLOCK, num_kv_heads=hkv, head_dim=D,
            kv_dtype=torch.float8_e4m3fn, kv_format="nvfp4",
        )  # fmt: skip
        pool.allocate(blocks, torch.device("cuda"))
        (kp, vp), (ks, vs) = pool.kv_views(0), pool.scale_views(0)
        payload = torch.full(
            (index_pages, BLOCK, D // 2), 0x5A, dtype=torch.uint8, device="cuda"
        )
        scale = torch.full(
            (index_pages, BLOCK, D // 16), 0x5A, dtype=torch.uint8, device="cuda"
        )
        q_out = torch.empty(tokens, heads * D, dtype=torch.bfloat16, device="cuda")
        iq = torch.empty(tokens, hkv * D, dtype=torch.bfloat16, device="cuda")
        aiter.fused_qknorm_idxrqknorm(
            qkv.clone(), w[0], w[1], cos_sin, pos, heads, hkv, rot, 1e-6, w[2],
            w[3], hkv, slots, kp, vp, payload, BLOCK, q_out, iq, index_slots,
            kv_cache_dtype="nvfp4", index_cache_dtype="nvfp4", k_scale=ks,
            v_scale=vs, asm_layout=False, index_scale=scale,
        )  # fmt: skip
        return (kp, vp, ks, vs), payload, scale

    kv, payload, scale = write(blocks, slots)
    for rank in range(WORLD):
        local_slots = owned_index_slots(slots, rank, WORLD)
        kv_r, payload_r, scale_r = write(blocks // WORLD, local_slots)
        torch.cuda.synchronize()
        for got, want in zip(kv_r, kv):
            assert torch.equal(got, want)
        assert torch.equal(payload_r, payload[rank::WORLD])
        assert torch.equal(scale_r, scale[rank::WORLD])


def _decode_inputs(batch, max_seq_len, q, seed):
    g = torch.Generator().manual_seed(seed)
    blocks = -(-max_seq_len // BLOCK)
    pages = WORLD * batch * -(-blocks // WORLD)  # every lane holds what it serves
    lens = torch.randint(q, max_seq_len + 1, (batch,), generator=g)
    lens[0], lens[-1] = q, max_seq_len
    tables = laned_tables([max_seq_len] * batch, pages, g)
    block_table = torch.tensor(tables, dtype=torch.int32, device="cuda")
    idx_q = torch.randn(batch * q, WORLD, D, generator=g).to(
        device="cuda", dtype=torch.bfloat16
    )
    payload, scale = fp4_cache(pages, seed)
    return idx_q, (payload, scale), block_table, lens.to("cuda", torch.int32)


@needs_gpu
@pytest.mark.parametrize("flydsl", [False, True])
@pytest.mark.parametrize("max_seq_len,q", [(512, 1), (16384, 4)])
def test_cp_decode_reads_the_local_planes_by_block_id(
    flydsl, max_seq_len, q, monkeypatch
):
    from atom.model_ops.minimax_m3.indexer_candidate_exchange import (
        local_candidate_keys,
        merge_candidate_keys,
    )
    from atom.model_ops.minimax_m3.indexer_context_parallel import (
        indexer_context_scores,
    )

    monkeypatch.setenv("ATOM_M3_CP_SCORE_FLYDSL", "1" if flydsl else "0")
    batch, topk, init, local = 6, 16, 1, 2
    idx_q, (payload, scale), block_table, lens = _decode_inputs(
        batch, max_seq_len, q, seed=5
    )
    blocks = block_table.shape[1]

    def keys(rank, cache, cache_scale):
        return local_candidate_keys(
            indexer_context_scores(
                idx_q, cache, block_table, lens, max_seq_len, rank, WORLD, q, SCALE,
                index_scale=cache_scale, fill_dead_slots=False,
            ),
            lens, topk, rank, WORLD, q, blocks, init, local,
        )  # fmt: skip

    replicated = [keys(r, payload, scale) for r in range(WORLD)]
    sharded = [
        keys(
            r,
            page_view(local_planes(payload, r), r, WORLD),
            page_view(local_planes(scale, r), r, WORLD),
        )
        for r in range(WORLD)
    ]
    for r in range(WORLD):
        assert torch.equal(sharded[r], replicated[r]), f"rank {r} candidates"
    for head in range(WORLD):
        received = torch.stack([sharded[r][head] for r in range(WORLD)])
        got = merge_candidate_keys(received, block_table, lens, topk, init, local, q)
        want = merge_candidate_keys(
            torch.stack([replicated[r][head] for r in range(WORLD)]),
            block_table, lens, topk, init, local, q,
        )  # fmt: skip
        for a, b in zip(got, want):
            assert torch.equal(a, b)


@needs_gpu
@pytest.mark.parametrize("flydsl", [False, True])
def test_prefill_topk_on_gathered_pages_is_the_replicated_ones(flydsl, monkeypatch):
    from atom.model_ops.minimax_m3.index_topk import minimax_m3_index_topk

    monkeypatch.setenv("ATOM_M3_INDEX_SCORE_FLYDSL_PREFILL", "1" if flydsl else "0")
    monkeypatch.setenv("ATOM_M3_INDEX_SCORE_FLYDSL_MIN_ROWS", "1")
    g = torch.Generator().manual_seed(11)
    seq_lens = [700, 1500, 4096]
    q_lens = [128, 37, 1000]
    pages = 4 * WORLD * 16
    tables = laned_tables(seq_lens, pages, g)
    width = max(map(len, tables))
    block_table = torch.zeros(len(tables), width, dtype=torch.int32)
    for b, row in enumerate(tables):
        block_table[b, : len(row)] = torch.tensor(row)
    block_table = block_table.cuda()
    payload, scale = fp4_cache(pages, seed=11)
    cu = torch.tensor(np.concatenate([[0], np.cumsum(q_lens)]), dtype=torch.int32)
    seq = torch.tensor(seq_lens, dtype=torch.int32)
    prefix = seq - torch.tensor(q_lens, dtype=torch.int32)
    total_q = sum(q_lens)

    plans = [
        plan_index_gather(tables, np.array(seq_lens), r, WORLD, "cuda")
        for r in range(WORLD)
    ]
    sends = [
        pack_index_pages(
            plans[r], payload[r::WORLD].contiguous(), scale[r::WORLD].contiguous()
        )
        for r in range(WORLD)
    ]
    g_payload, g_scale = gathered_planes(torch.cat(sends), payload, scale)

    def topk(idx_q, cache, cache_scale, score_table=None):
        return minimax_m3_index_topk(
            idx_q, cache, block_table, cu.cuda(), seq.cuda(), prefix.cuda(),
            max(q_lens), max(seq_lens), 16, 1, 2, 1, SCALE,
            emit_sparse_block_table=True, index_kv_scale=cache_scale,
            score_block_table=score_table,
        )  # fmt: skip

    for head in range(WORLD):  # each rank's own head
        idx_q = torch.randn(total_q, 1, D, generator=g).to("cuda", torch.bfloat16)
        want = topk(idx_q, payload, scale)
        got = topk(idx_q, g_payload, g_scale, plans[0].score_table)
        for a, b in zip(got, want):
            assert torch.equal(a, b), f"head {head}"

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only tests for the SparseKV coordinator bookkeeping.

Exercises miss-detect + LRU allocation + logical->hot-slot translation without a
GPU. The GPU swap kernel is not invoked here: ``plan_swap_for_request`` is pure
bookkeeping, and the coordinator is built with ``device='cpu'`` so no HIP kernel
compiles. Data-movement correctness is left to on-GPU validation.
"""

import pytest
import torch

from atom.sparsekv.coordinator import _EMPTY, SparseKVCoordinator


def _make(num_layers=1, max_num_seqs=2, hot=4, max_ctx=32, kv_dim=8, ratio=8, page=16):
    return SparseKVCoordinator(
        num_layers=num_layers,
        max_num_seqs=max_num_seqs,
        hot_buffer_size=hot,
        max_context_len=max_ctx,
        kv_dim=kv_dim,
        kv_dtype=torch.bfloat16,
        device="cpu",
        host_to_device_ratio=ratio,
        page_size=page,
    )


def _seed_resident(c, layer, req, tokens):
    """Mark ``tokens`` resident in slots 0..len-1 with recency 1."""
    t = torch.tensor(tokens, dtype=torch.int32)
    slots = torch.arange(len(tokens), dtype=torch.int32)
    c.slot_token[layer, req, : len(tokens)] = t
    c.last_used[layer, req, : len(tokens)] = 1
    c.token_to_slot[layer, req, t.to(torch.int64)] = slots


def test_item_size_bytes():
    c = _make(kv_dim=576)
    # bf16 -> 2 bytes/elem
    assert c.item_size_bytes == 576 * 2


def test_all_hit_no_swap():
    c = _make()
    c.register_request(0, 10)
    c.alloc_host_pages(0, 0, 10)
    _seed_resident(c, 0, 0, [7, 8, 9])
    topk = torch.tensor([9, 8, 7], dtype=torch.int32)
    src, dst, _gs, _gd, tr = c.plan_swap_for_request(0, 0, topk)
    assert src.numel() == 0 and dst.numel() == 0
    hot_base = c._hot_base(0)
    # 9->slot2, 8->slot1, 7->slot0
    assert tr.tolist() == [hot_base + 2, hot_base + 1, hot_base + 0]


def test_cold_start_all_miss():
    c = _make(hot=4)
    c.register_request(0, 10)
    c.alloc_host_pages(0, 0, 10)
    topk = torch.tensor([1, 2, 3], dtype=torch.int32)
    src, dst, _gs, _gd, tr = c.plan_swap_for_request(0, 0, topk)
    # all three are misses, land in empty slots
    expected = sorted(int(c.req_to_host_pool[0, t].item()) for t in [1, 2, 3])
    assert sorted(src.tolist()) == expected
    assert src.numel() == 3 and dst.numel() == 3
    # every top-k entry now resolves to a resident hot slot
    for t in [1, 2, 3]:
        assert c.token_to_slot[0, 0, t].item() >= 0


def test_lru_eviction_picks_stale():
    c = _make(hot=3, max_ctx=32)  # 3 + 1 padded = 4 physical slots
    c.register_request(0, 20)
    c.alloc_host_pages(0, 0, 21)
    # fill all 4 slots with tokens 10,11,12,13 at increasing recency
    for i, tok in enumerate([10, 11, 12, 13]):
        c.slot_token[0, 0, i] = tok
        c.last_used[0, 0, i] = i + 1  # 10 is stalest (recency 1)
        c.token_to_slot[0, 0, tok] = i
    c._tick = 4  # tick must exceed any seeded recency (real-flow invariant)
    # request a brand new token -> must evict the stalest (token 10, slot 0)
    topk = torch.tensor([20], dtype=torch.int32)
    src, dst, _gs, _gd, tr = c.plan_swap_for_request(0, 0, topk)
    assert dst.tolist() == [c._hot_base(0) + 0]  # slot 0 evicted
    assert c.token_to_slot[0, 0, 10].item() == _EMPTY  # 10 evicted
    assert c.token_to_slot[0, 0, 20].item() == 0  # 20 now resident in slot 0


def test_hit_refreshes_recency_protects_from_eviction():
    c = _make(hot=3, max_ctx=32)
    c.register_request(0, 20)
    c.alloc_host_pages(0, 0, 21)
    for i, tok in enumerate([10, 11, 12, 13]):
        c.slot_token[0, 0, i] = tok
        c.last_used[0, 0, i] = i + 1
        c.token_to_slot[0, 0, tok] = i
    c._tick = 4  # tick must exceed any seeded recency (real-flow invariant)
    # touch token 10 (stalest) as a hit, then bring in a new token.
    # 10 should now be protected; 11 (next stalest) evicted instead.
    topk = torch.tensor([10, 20], dtype=torch.int32)
    src, dst, _gs, _gd, tr = c.plan_swap_for_request(0, 0, topk)
    assert c.token_to_slot[0, 0, 10].item() == 0  # still resident
    assert c.token_to_slot[0, 0, 11].item() == _EMPTY  # evicted instead


def test_padding_entries_ignored():
    c = _make()
    c.register_request(0, 10)
    _seed_resident(c, 0, 0, [5])
    topk = torch.tensor([5, -1, -1], dtype=torch.int32)
    src, dst, _gs, _gd, tr = c.plan_swap_for_request(0, 0, topk)
    assert src.numel() == 0  # only the real token, already resident
    hot_base = c._hot_base(0)
    assert tr[0].item() == hot_base + 0
    # padding entries map to hot_base (slot 0), harmless
    assert tr[1].item() == hot_base and tr[2].item() == hot_base


def test_duplicate_topk_tokens_dedup_misses():
    c = _make()
    c.register_request(0, 10)
    c.alloc_host_pages(0, 0, 10)
    topk = torch.tensor([3, 3, 3], dtype=torch.int32)
    src, dst, _gs, _gd, tr = c.plan_swap_for_request(0, 0, topk)
    assert src.numel() == 1  # deduped to a single swap
    hot_base = c._hot_base(0)
    assert tr.tolist() == [tr[0].item()] * 3  # all resolve to same slot
    assert tr[0].item() == hot_base + c.token_to_slot[0, 0, 3].item()


def test_plan_swap_dual_source():
    """Design Y: a mixed-home top-k splits into host and gpu swap groups."""
    c = SparseKVCoordinator(
        num_layers=1,
        max_num_seqs=2,
        hot_buffer_size=8,
        max_context_len=32,
        kv_dim=8,
        kv_dtype=torch.bfloat16,
        device="cpu",
        index_topk=4,
        host_to_device_ratio=8,
        page_size=16,
        num_gpu_cold_pages=4,
    )
    assert c.gpu_cold_enabled
    c.register_request(0, 20)
    # Disjoint homes: tokens 3,7 live in the host cold pool; 5,9 in the GPU tier.
    # A token has a valid row in exactly one table (the other stays -1).
    c.req_to_host_pool[0, 3] = 103
    c.req_to_host_pool[0, 7] = 107
    c.req_to_gpu_pool[0, 5] = 205
    c.req_to_gpu_pool[0, 9] = 209
    topk = torch.tensor([3, 5, 7, 9], dtype=torch.int32)
    hs, hd, gs, gd, tr = c.plan_swap_for_request(0, 0, topk)
    # host group carries only host-home sources; gpu group only gpu-home sources.
    assert sorted(hs.tolist()) == [103, 107]
    assert sorted(gs.tolist()) == [205, 209]
    assert hd.numel() == 2 and gd.numel() == 2
    # every top-k entry is now resident in some hot slot (mixed home, one cache).
    for t in [3, 5, 7, 9]:
        assert c.token_to_slot[0, 0, t].item() >= 0
    assert tr.numel() == 4


def test_plan_swap_gpu_disabled_all_host():
    """With the GPU tier off, plan_swap keeps everything host-home (empty gpu group)."""
    c = _make()  # num_gpu_cold_pages defaults to 0
    assert not c.gpu_cold_enabled
    c.register_request(0, 10)
    c.alloc_host_pages(0, 0, 10)
    topk = torch.tensor([1, 2, 3], dtype=torch.int32)
    hs, hd, gs, gd, _tr = c.plan_swap_for_request(0, 0, topk)
    assert hs.numel() == 3 and hd.numel() == 3
    assert gs.numel() == 0 and gd.numel() == 0


def test_request_lifecycle_reset():
    c = _make()
    c.register_request(0, 10)
    c.alloc_host_pages(0, 0, 10)
    _seed_resident(c, 0, 0, [1, 2, 3])
    c.unregister_request(0)
    assert not c.slot_active[0].item()
    assert (c.slot_token[:, 0, :] == _EMPTY).all()
    assert (c.token_to_slot[:, 0, :] == _EMPTY).all()
    # re-register reuses the slot cleanly
    c.register_request(0, 5)
    c.alloc_host_pages(0, 0, 5)
    topk = torch.tensor([1], dtype=torch.int32)
    src, _, _, _, _ = c.plan_swap_for_request(0, 0, topk)
    assert src.numel() == 1  # token 1 no longer resident -> miss


def test_per_layer_independent_state():
    c = _make(num_layers=2)
    c.register_request(0, 10)
    c.alloc_host_pages(0, 0, 10)
    _seed_resident(c, 0, 0, [7])  # resident only in layer 0
    topk = torch.tensor([7], dtype=torch.int32)
    src0, _, _, _, _ = c.plan_swap_for_request(0, 0, topk)
    src1, _, _, _, _ = c.plan_swap_for_request(1, 0, topk)
    assert src0.numel() == 0  # hit in layer 0
    assert src1.numel() == 1  # miss in layer 1


def test_backup_new_token_makes_resident():
    c = _make(hot=4)
    c.register_request(0, 10)
    kv = torch.ones(c.kv_dim, dtype=torch.bfloat16)
    c.backup_new_token(req_slot=0, layer_id=0, new_token_kv=kv, logical_pos=10)
    assert c.token_to_slot[0, 0, 10].item() >= 0
    assert c.context_len[0].item() == 11
    # cold pool row written (via paged host pool)
    cold_row = int(c.req_to_host_pool[0, 10].item())
    row = c.cold_pool[0, cold_row]
    assert torch.allclose(row.float(), torch.ones_like(row.float()))


def test_reqid_acquire_release_roundtrip():
    c = _make(max_num_seqs=2)
    s0 = c.acquire(req_id=100, context_len=5)
    s1 = c.acquire(req_id=101, context_len=5)
    assert {s0, s1} == {0, 1}
    assert c.is_registered(100) and c.slot_for_req(101) == s1
    # pool exhausted -> next acquire raises
    try:
        c.acquire(req_id=102, context_len=5)
        assert False, "expected RuntimeError on exhausted slots"
    except RuntimeError:
        pass
    c.release(100)
    assert not c.is_registered(100)
    # freed slot is reusable
    s2 = c.acquire(req_id=102, context_len=5)
    assert s2 == s0


def test_reqid_acquire_idempotent():
    c = _make(max_num_seqs=2)
    a = c.acquire(req_id=7, context_len=5)
    b = c.acquire(req_id=7, context_len=5)  # same id -> same slot, no new alloc
    assert a == b
    assert len(c._free_slots) == 1


def test_sync_active_releases_departed_requests():
    c = _make(max_num_seqs=4)
    c.acquire(10, 5)
    c.acquire(11, 5)
    c.acquire(12, 5)
    c.sync_active([11])  # 10 and 12 have left the batch
    assert c.is_registered(11)
    assert not c.is_registered(10) and not c.is_registered(12)
    assert sorted(c._free_slots) == sorted(set(range(4)) - {c.slot_for_req(11)})


# --- paged host pool allocator ------------------------------------------------


def _make_paged(max_num_seqs=2, hot=4, max_ctx=48, kv_dim=8, ratio=8, page=16):
    return _make(
        num_layers=1,
        max_num_seqs=max_num_seqs,
        hot=hot,
        max_ctx=max_ctx,
        kv_dim=kv_dim,
        ratio=ratio,
        page=page,
    )


def test_paged_alloc_pages_contiguous_within_page():
    c = _make_paged(page=16)
    slot = c.acquire(req_id=1, context_len=20)
    c.alloc_host_pages(slot, 0, 20)  # 20 tokens -> 2 pages (page-rounded to 32)
    assert c._req_host_alloc_len[slot] == 32
    row = c.req_to_host_pool[slot]
    # Each 16-token page is contiguous: slot value = page_base + offset.
    for pstart in (0, 16):
        base = int(row[pstart].item())
        assert base % 16 == 0  # page-aligned host slot
        assert row[pstart : pstart + 16].tolist() == list(range(base, base + 16))
    # The two pages are distinct.
    assert int(row[0].item()) // 16 != int(row[16].item()) // 16
    # Unbacked logical positions stay -1.
    assert int(row[32].item()) == _EMPTY


def test_paged_alloc_growth_and_free_recycles_pages():
    c = _make_paged(page=16)
    free0 = len(c._free_host_pages)
    slot = c.acquire(req_id=1, context_len=10)
    c.alloc_host_pages(slot, 0, 10)  # 1 page
    assert len(c._free_host_pages) == free0 - 1
    c.grow_host_for_new_tokens([slot], [16])  # crosses into a 2nd page
    assert c._req_host_alloc_len[slot] == 32
    assert len(c._free_host_pages) == free0 - 2
    c.release(req_id=1)  # frees pages + clears table row
    assert len(c._free_host_pages) == free0
    assert torch.all(c.req_to_host_pool[slot] == _EMPTY)


def test_sync_active_protects_reqs_awaiting_first_decode():
    # A request acquired at recv (slot + host pages reserved) whose prompt KV is
    # still transferring is absent from the decode batch. sync_active must not
    # reclaim it, or its pages vanish before the first decode (concurrency crash).
    c = _make_paged(max_num_seqs=4, ratio=8, page=16)
    c.acquire_at_recv(req_id=100, num_tokens=20)  # in-flight recv
    slot = c.slot_for_req(100)
    alloc_len = c._req_host_alloc_len[slot]
    assert alloc_len > 0
    assert 100 in c._awaiting_first_decode
    # Other requests decode; the in-flight request is not in their batches.
    c.sync_active([200])
    c.sync_active([201])
    assert c.is_registered(100), "recv-window request wrongly released"
    assert c._req_host_alloc_len[slot] == alloc_len
    # Once the first decode has run (tracked by leaving _awaiting_first_decode),
    # the request is a normal batch member and sync_active may reclaim it.
    c._awaiting_first_decode.discard(100)  # first decode reached (GPU hot-load)
    c.sync_active([200])  # 100 has genuinely left the batch now
    assert not c.is_registered(100)


def test_release_clears_awaiting_first_decode():
    # A request aborted during recv (released before first decode) must drop out
    # of the protection set, otherwise its slot leaks forever.
    c = _make_paged(max_num_seqs=4, ratio=8, page=16)
    c.acquire_at_recv(req_id=100, num_tokens=20)
    assert 100 in c._awaiting_first_decode
    c.release(100)
    assert 100 not in c._awaiting_first_decode
    assert not c.is_registered(100)


def test_paged_alloc_exhaustion_raises():
    # ratio=1 -> host_tokens = 1 * R(1) * H1(5) = 5 -> 1 page of 16 slots.
    c = _make_paged(max_num_seqs=1, hot=4, ratio=1, page=16)
    slot = c.acquire(req_id=1, context_len=16)
    c.alloc_host_pages(slot, 0, 16)  # consumes the only page
    with pytest.raises(RuntimeError, match="host pool exhausted"):
        c.grow_host_for_new_tokens([slot], [16])  # needs a 2nd page, none free


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused swap kernel needs a GPU + aiter"
)
def test_paged_fused_swap_matches_token_data():
    """Paged path: the table-lookup gather returns each top-k's KV.

    Token t's KV lives at host slot req_to_host_pool[slot][t], so a correct kernel
    must indirect through the page table.
    """
    dev = "cuda"
    L, R, H, C, D = 1, 1, 4, 48, 8
    c = SparseKVCoordinator(
        num_layers=L,
        max_num_seqs=R,
        hot_buffer_size=H,
        max_context_len=C,
        kv_dim=D,
        kv_dtype=torch.bfloat16,
        device=dev,
        host_to_device_ratio=8,
        page_size=16,
    )
    slot = c.acquire(req_id=1, context_len=10)
    c.alloc_host_pages(slot, 0, 10)  # back logical 0..9 (host slots may be scattered)
    for t in range(10):  # token t's KV -> value t, written at its host slot
        hs = int(c.req_to_host_pool[slot, t].item())
        c.cold_pool[0, hs] = torch.full((D,), float(t), dtype=torch.bfloat16)
    c.load_initial_hot_set(slot, 10)

    K = 3
    for step in ([9, 8, 7], [6, 9, 5], [9, 8, 4], [3, 2, 1]):
        topk = torch.tensor([step], dtype=torch.int32, device=dev)
        indptr = torch.tensor([0, K], dtype=torch.int32, device=dev)
        req_slots = torch.tensor([slot], dtype=torch.int32, device=dev)
        out = torch.zeros(K, dtype=torch.int32, device=dev)
        c.swap_in_for_layer_fused(0, topk, indptr, req_slots, out)
        torch.cuda.synchronize()
        for k, tok in enumerate(step):
            row = c.hot_buffer[0, int(out[k].item())].float()
            expected = torch.full((D,), float(tok), device=dev).float()
            assert torch.allclose(row, expected), (step, k, tok, row[0].item())


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused backup kernel needs a GPU + aiter"
)
def test_paged_fused_backup_writes_via_table():
    dev = "cuda"
    L, R, H, C, D = 1, 1, 4, 48, 8
    c = SparseKVCoordinator(
        num_layers=L,
        max_num_seqs=R,
        hot_buffer_size=H,
        max_context_len=C,
        kv_dim=D,
        kv_dtype=torch.bfloat16,
        device=dev,
        host_to_device_ratio=8,
        page_size=16,
    )
    slot = c.acquire(req_id=1, context_len=5)
    c.alloc_host_pages(slot, 0, 5)
    c.load_initial_hot_set(slot, 5)
    c.grow_host_for_new_tokens([slot], [5])  # back logical pos 5 before the backup
    layer_kv = torch.zeros(64, D, dtype=torch.bfloat16, device=dev)
    layer_kv[5] = torch.full((D,), 42.0, dtype=torch.bfloat16)
    c.backup_new_tokens_fused(
        0,
        layer_kv,
        torch.tensor([5], dtype=torch.int32, device=dev),
        torch.tensor([slot], dtype=torch.int32, device=dev),
        torch.tensor([5], dtype=torch.int32, device=dev),
    )
    torch.cuda.synchronize()
    assert int(c.token_to_slot[0, slot, 5].item()) >= 0
    hs = int(c.req_to_host_pool[slot, 5].item())
    assert torch.allclose(c.cold_pool[0, hs].float(), torch.full((D,), 42.0).float())


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused swap kernel needs a GPU + aiter"
)
def test_fused_swap_data_movement_matches_cold_pool():
    """End-to-end fused path: every translated hot row holds its top-k's cold KV.

    Eviction policy is free to differ from the reference; the observable invariant
    is that after swap-in the hot buffer row each top-k resolves to contains the
    cold-pool data for that logical token.
    """
    dev = "cuda"
    L, R, H, C, D = 1, 1, 4, 32, 8
    c = SparseKVCoordinator(
        num_layers=L,
        max_num_seqs=R,
        hot_buffer_size=H,
        max_context_len=C,
        kv_dim=D,
        kv_dtype=torch.bfloat16,
        device=dev,
        host_to_device_ratio=8,
        page_size=16,
    )
    slot = c.acquire(req_id=1, context_len=10)  # -> slot 0
    c.alloc_host_pages(slot, 0, C)
    for t in range(C):  # cold pool row for token t is the constant t
        hs = int(c.req_to_host_pool[slot, t].item())
        c.cold_pool[0, hs] = torch.full((D,), float(t), dtype=torch.bfloat16)
    c.load_initial_hot_set(slot, 10)

    K = 3
    for step in ([9, 8, 7], [6, 9, 5], [9, 8, 4], [3, 2, 1]):
        topk = torch.tensor([step], dtype=torch.int32, device=dev)
        indptr = torch.tensor([0, K], dtype=torch.int32, device=dev)
        req_slots = torch.tensor([0], dtype=torch.int32, device=dev)
        out = torch.zeros(K, dtype=torch.int32, device=dev)
        c.swap_in_for_layer_fused(0, topk, indptr, req_slots, out)
        torch.cuda.synchronize()
        for k, tok in enumerate(step):
            row = c.hot_buffer[0, int(out[k].item())].float()
            expected = torch.full((D,), float(tok), device=dev).float()
            assert torch.allclose(row, expected), (step, k, tok, row[0].item())


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="dual-source fused swap needs a GPU + aiter"
)
def test_fused_swap_dual_source_mixed_home():
    """Design Y: a top-k spanning host-home and gpu-home tokens gathers from both.

    Stage the full KV in the host cold pool, promote the first page to the GPU cold
    tier (so those tokens are gpu-home, their host row freed), then run the fused
    swap for a top-k that mixes a promoted (gpu) token with unpromoted (host) ones.
    Each translated hot row must hold that token's KV, proving the record-only
    detect + per-home gather (skip_gather path) moves both tiers correctly.
    """
    dev = "cuda"
    L, R, H, C, D, page = 1, 1, 4, 48, 8, 16
    c = SparseKVCoordinator(
        num_layers=L,
        max_num_seqs=R,
        hot_buffer_size=H,
        max_context_len=C,
        kv_dim=D,
        kv_dtype=torch.bfloat16,
        device=dev,
        index_topk=8,
        host_to_device_ratio=8,
        page_size=page,
        num_gpu_cold_pages=1,  # room for exactly one promoted page (tokens 0..15)
    )
    assert c.gpu_cold_enabled
    slot = c.acquire(req_id=1, context_len=C)
    c.alloc_host_pages(slot, 0, C)
    for t in range(C):  # host cold-pool row for token t holds the constant t
        hs = int(c.req_to_host_pool[slot, t].item())
        c.cold_pool[0, hs] = torch.full((D,), float(t), dtype=torch.bfloat16)
    # Promote the oldest page (tokens 0..15) to the GPU tier: those become gpu-home
    # (host row freed to -1), the rest stay host-home.
    promoted = c.promote_to_gpu(slot)
    torch.cuda.synchronize()
    assert promoted == 1
    assert int(c.req_to_gpu_pool[slot, 2].item()) >= 0  # token 2 now gpu-home
    assert int(c.req_to_host_pool[slot, 2].item()) == _EMPTY
    assert int(c.req_to_host_pool[slot, 40].item()) >= 0  # token 40 still host-home

    c.load_initial_hot_set(slot, C)  # seeds recency; recent tokens are host-home
    torch.cuda.synchronize()

    K = 3
    step = [2, 20, 40]  # 2 is gpu-home; 20 and 40 are host-home
    topk = torch.tensor([step], dtype=torch.int32, device=dev)
    indptr = torch.tensor([0, K], dtype=torch.int32, device=dev)
    req_slots = torch.tensor([slot], dtype=torch.int32, device=dev)
    out = torch.zeros(K, dtype=torch.int32, device=dev)
    c.swap_in_for_layer_fused(0, topk, indptr, req_slots, out)
    torch.cuda.synchronize()
    for k, tok in enumerate(step):
        row = c.hot_buffer[0, int(out[k].item())].float()
        expected = torch.full((D,), float(tok), device=dev).float()
        assert torch.allclose(row, expected), (step, k, tok, row[0].item())


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused backup kernel needs a GPU + aiter"
)
def test_fused_backup_makes_new_token_resident():
    dev = "cuda"
    L, R, H, C, D = 1, 1, 4, 32, 8
    c = SparseKVCoordinator(
        num_layers=L,
        max_num_seqs=R,
        hot_buffer_size=H,
        max_context_len=C,
        kv_dim=D,
        kv_dtype=torch.bfloat16,
        device=dev,
        host_to_device_ratio=8,
        page_size=16,
    )
    req_slot = c.acquire(req_id=1, context_len=5)
    c.alloc_host_pages(req_slot, 0, 6)
    c.load_initial_hot_set(req_slot, 5)
    # A fresh token at logical pos 5 lives at physical slot 5 of the layer cache.
    layer_kv = torch.zeros(64, D, dtype=torch.bfloat16, device=dev)
    layer_kv[5] = torch.full((D,), 42.0, dtype=torch.bfloat16)
    c.backup_new_tokens_fused(
        0,
        layer_kv,
        torch.tensor([5], dtype=torch.int32, device=dev),  # src slot
        torch.tensor([req_slot], dtype=torch.int32, device=dev),  # req slot
        torch.tensor([5], dtype=torch.int32, device=dev),  # logical pos
    )
    torch.cuda.synchronize()
    slot = int(c.token_to_slot[0, req_slot, 5].item())
    assert slot >= 0
    assert torch.allclose(
        c.hot_buffer[0, slot].float(), torch.full((D,), 42.0, device=dev).float()
    )
    hs = int(c.req_to_host_pool[req_slot, 5].item())
    assert torch.allclose(c.cold_pool[0, hs].float(), torch.full((D,), 42.0).float())


def test_mtp_multi_token_batch_translate():
    """MTP verify: multiple query tokens share one request's top-k runs."""
    c = _make(hot=8, max_ctx=64)
    c._run_swap = lambda *a, **k: None  # CPU test: skip the GPU gather kernel
    c.register_request(0, 40)
    c.alloc_host_pages(0, 0, 51)
    _seed_resident(c, 0, 0, [30, 31, 32, 33])
    # two query tokens, each with its own 3-wide top-k run
    out_translated = torch.zeros(6, dtype=torch.int32)
    out_indptr = torch.tensor([0, 3, 6], dtype=torch.int32)
    topk_per_req = [
        torch.tensor([30, 31, 32], dtype=torch.int32),
        torch.tensor([33, 30, 50], dtype=torch.int32),
    ]
    c.swap_in_for_layer(
        layer_id=0,
        batch_req_slots=[0, 0],
        topk_per_req=topk_per_req,
        out_translated=out_translated,
        out_indptr=out_indptr,
    )
    hot_base = c._hot_base(0)
    # first run: 30,31,32 all resident
    assert out_translated[:3].tolist() == [hot_base + 0, hot_base + 1, hot_base + 2]
    # second run: 33 resident (slot3), 30 resident (slot0), 50 was a miss -> resident now
    assert out_translated[3].item() == hot_base + 3
    assert out_translated[4].item() == hot_base + 0
    assert c.token_to_slot[0, 0, 50].item() >= 0


# --- Stage C: IndexShare group prefetch --------------------------------------


def test_prefetch_group_structure():
    """Anchor/shared grouping is derived correctly from the layer pattern.

    Pattern mirrors GLM-5.2: leading fulls with no shared layers, then repeating
    [full, shared, shared, shared] groups. Structure is built regardless of the
    ATOM_SPARSEKV_PREFETCH env (the enable gate is separate), so this needs no GPU.
    """
    # layers: 0,1,2 full (no group); 3=full anchors 4,5,6; 7=full anchors 8,9,10
    shared = [False, False, False, False, True, True, True, False, True, True, True]
    c = SparseKVCoordinator(
        num_layers=len(shared),
        max_num_seqs=2,
        hot_buffer_size=4,
        max_context_len=32,
        kv_dim=8,
        kv_dtype=torch.bfloat16,
        device="cpu",
        index_topk=4,
        shared_index_layers=shared,
    )
    assert c._prefetch_groups == {3: [4, 5, 6], 7: [8, 9, 10]}
    assert c._anchor_of[5] == 3 and c._anchor_of[9] == 7
    assert c._anchor_of[0] == 0 and c._anchor_of[3] == 3  # non-shared: self
    assert c._prefetch_slot[4] == 0 and c._prefetch_slot[6] == 2
    assert c._prefetch_slot[8] == 0 and c._prefetch_slot[10] == 2
    assert c._is_shared[4] and not c._is_shared[3]


def test_prefetch_disabled_by_default():
    """Without the env, prefetch is inert and the layer-role queries are False."""
    shared = [False, True, True]
    c = SparseKVCoordinator(
        num_layers=3,
        max_num_seqs=1,
        hot_buffer_size=4,
        max_context_len=16,
        kv_dim=8,
        kv_dtype=torch.bfloat16,
        device="cpu",
        index_topk=4,
        shared_index_layers=shared,
    )
    assert c.enable_prefetch is False
    assert c.is_prefetch_anchor(0) is False
    assert c.is_shared_layer(1) is False


def test_prefetch_bad_pattern_rejected():
    """A shared layer with no preceding anchor is a config error."""
    with pytest.raises(AssertionError):
        SparseKVCoordinator(
            num_layers=2,
            max_num_seqs=1,
            hot_buffer_size=4,
            max_context_len=16,
            kv_dim=8,
            kv_dtype=torch.bfloat16,
            device="cpu",
            index_topk=4,
            shared_index_layers=[True, False],
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="prefetch path needs a GPU + aiter"
)
def test_prefetch_matches_per_layer_fused(monkeypatch):
    """The IndexShare prefetch path is bit-identical to per-layer synchronous swap.

    A group's shared layers evolve in lockstep with their anchor, so replaying the
    anchor's plan + assigned-slot backup must produce the same hot buffer and the
    same translated indices as running the full fused kernel on every layer.
    """
    monkeypatch.setenv("ATOM_SPARSEKV_PREFETCH", "1")
    dev = "cuda"
    L, R, H, C, D, K = 3, 1, 4, 32, 8, 3
    shared = [False, True, True]  # layer 0 anchors 1, 2

    def build(shared_layers):
        c = SparseKVCoordinator(
            num_layers=L,
            max_num_seqs=R,
            hot_buffer_size=H,
            max_context_len=C,
            kv_dim=D,
            kv_dtype=torch.bfloat16,
            device=dev,
            index_topk=K,
            shared_index_layers=shared_layers,
            host_to_device_ratio=8,
            page_size=16,
        )
        slot = c.acquire(req_id=1, context_len=10)
        c.alloc_host_pages(slot, 0, C)
        for layer in range(L):  # distinct per-layer cold data: 100*layer + tok
            for t in range(C):
                hs = int(c.req_to_host_pool[slot, t].item())
                c.cold_pool[layer, hs] = torch.full(
                    (D,), float(100 * layer + t), dtype=torch.bfloat16
                )
        c.load_initial_hot_set(slot, 10)
        return c

    ref = build(None)  # prefetch disabled -> per-layer fused every layer
    pre = build(shared)  # prefetch enabled -> anchor records, shared replay
    assert pre.enable_prefetch and not ref.enable_prefetch

    req_slots = torch.tensor([0], dtype=torch.int32, device=dev)
    indptr = torch.tensor([0, K], dtype=torch.int32, device=dev)
    # A fresh token at logical pos 10 sits at physical slot 10 of each layer cache.
    layer_kv = [torch.zeros(64, D, dtype=torch.bfloat16, device=dev) for _ in range(L)]
    for layer in range(L):
        layer_kv[layer][10] = torch.full(
            (D,), float(1000 + layer), dtype=torch.bfloat16
        )
    src = torch.tensor([10], dtype=torch.int32, device=dev)
    pos = torch.tensor([10], dtype=torch.int32, device=dev)
    topk = torch.tensor([[10, 8, 3]], dtype=torch.int32, device=dev)

    # Reference: full fused (backup + swap+translate) on every layer.
    ref_tr = []
    for layer in range(L):
        out = torch.zeros(K, dtype=torch.int32, device=dev)
        ref.backup_new_tokens_fused(layer, layer_kv[layer], src, req_slots, pos)
        ref.swap_in_for_layer_fused(layer, topk, indptr, req_slots, out)
        ref_tr.append(out.clone())
    torch.cuda.synchronize()

    # Prefetch: anchor records + fires the group; shared layers replay + backup.
    anchor_out = torch.zeros(K, dtype=torch.int32, device=dev)
    pre.backup_new_tokens_fused(0, layer_kv[0], src, req_slots, pos)
    pre.swap_in_for_layer_fused(
        0, topk, indptr, req_slots, anchor_out, record_plan=True
    )
    pre.prefetch_group(0, req_slots)
    for layer in (1, 2):
        pre.backup_into_assigned_fused(layer, 0, layer_kv[layer], src, req_slots, pos)
        pre.wait_prefetch(layer)
    torch.cuda.synchronize()

    # Translate is shared across the group -> identical to the anchor's.
    assert torch.equal(anchor_out, ref_tr[0])
    # Each layer's hot buffer holds the same KV under both paths.
    for layer in range(L):
        assert torch.allclose(
            pre.hot_buffer[layer].float(), ref.hot_buffer[layer].float()
        ), layer


# --- GPU cold tier (Design Y) ------------------------------------------------


def _make_gpu_cold(
    num_layers=1,
    max_num_seqs=2,
    hot=4,
    max_ctx=32,
    kv_dim=8,
    ratio=8,
    page=16,
    gpu_pages=4,
):
    return SparseKVCoordinator(
        num_layers=num_layers,
        max_num_seqs=max_num_seqs,
        hot_buffer_size=hot,
        max_context_len=max_ctx,
        kv_dim=kv_dim,
        kv_dtype=torch.bfloat16,
        device="cpu",
        host_to_device_ratio=ratio,
        page_size=page,
        num_gpu_cold_pages=gpu_pages,
    )


def test_gpu_cold_disabled_by_default():
    c = _make()
    assert not c.gpu_cold_enabled
    assert c.gpu_cold_pool is None
    assert c.req_to_gpu_pool is None
    assert len(c._free_gpu_pages) == 0


def test_gpu_cold_enabled_allocates_pool():
    c = _make_gpu_cold(gpu_pages=4, page=16)
    assert c.gpu_cold_enabled
    assert c.gpu_cold_pool is not None
    assert c.gpu_cold_pool.shape == (1, 4 * 16, 8)
    assert c.req_to_gpu_pool is not None
    assert len(c._free_gpu_pages) == 4


def test_gpu_alloc_declines_after_partial_promote():
    """A request whose promote ran out of GPU pages keeps its tail on host. The
    decode-growth path must then decline GPU and fall back, not try to append
    past the gap — req_to_gpu_pool is filled from _req_gpu_alloc_len, so backing
    the new position on GPU would route the host-resident gap at
    [alloc_len, pos) to rows the promote never wrote."""
    c = _make_gpu_cold(gpu_pages=2, page=16, max_ctx=256, max_num_seqs=2)
    slot = c.acquire(req_id=1, context_len=64)

    # Only the first 32 logical positions get GPU pages (pool holds 2 pages).
    assert c.alloc_gpu_pages(slot, 0, 64) == 32
    assert c._req_gpu_alloc_len[slot] == 32

    # Position 63 sits past the promoted prefix: decline rather than assert.
    assert c.alloc_gpu_pages(slot, 63, 1) == 0
    # And the gap must not have been mapped into the GPU tier.
    assert int(c.req_to_gpu_pool[slot, 40].item()) == -1

    # grow_cold_for_new_token therefore lands it on host instead.
    c.grow_cold_for_new_token(slot, 63)
    assert int(c.req_to_host_pool[slot, 63].item()) >= 0


def test_gpu_page_bytes_matches_pool_footprint():
    """The auto-sizer divides free HBM by this, so it must equal what a page
    actually costs across all layers or the tier over- or under-shoots."""
    c = _make_gpu_cold(num_layers=3, gpu_pages=4, page=16, kv_dim=8)
    assert c.gpu_page_bytes == 3 * 16 * 8 * 2  # bf16
    assert c.gpu_cold_pool.numel() * c.gpu_cold_pool.element_size() == (
        4 * c.gpu_page_bytes
    )


def test_gpu_cold_tier_resizable_after_construction():
    """Auto sizing builds the coordinator with 0 pages so the hot buffer lands
    first, then re-inits the tier; both directions must leave consistent state."""
    c = _make_gpu_cold(gpu_pages=0, page=16)
    assert not c.gpu_cold_enabled and c.promote_stream is None

    c._init_gpu_cold_tier(4)
    assert c.gpu_cold_enabled
    assert c.gpu_cold_pool.shape == (1, 4 * 16, 8)
    assert c.req_to_gpu_pool is not None
    assert len(c._free_gpu_pages) == 4

    c._init_gpu_cold_tier(0)
    assert not c.gpu_cold_enabled
    assert c.gpu_cold_pool is None and c.req_to_gpu_pool is None
    assert len(c._free_gpu_pages) == 0


def test_autosize_is_noop_on_cpu():
    c = _make_gpu_cold(gpu_pages=0)
    assert c.autosize_gpu_cold_tier(0.15) == 0
    assert not c.gpu_cold_enabled


def test_gpu_alloc_and_free_pages():
    c = _make_gpu_cold(gpu_pages=4, page=16, max_ctx=64)
    slot = c.acquire(req_id=1, context_len=20)
    got = c.alloc_gpu_pages(slot, 0, 20)
    assert got == 20
    assert c._req_gpu_alloc_len[slot] == 32  # 2 pages, page-rounded
    assert len(c._free_gpu_pages) == 2
    row = c.req_to_gpu_pool[slot]
    for t in range(20):
        assert int(row[t].item()) >= 0
    assert int(row[32].item()) == _EMPTY
    c.free_gpu_pages(slot)
    assert len(c._free_gpu_pages) == 4
    assert torch.all(c.req_to_gpu_pool[slot] == _EMPTY)


def test_gpu_alloc_partial_when_full():
    c = _make_gpu_cold(gpu_pages=1, page=16)
    slot = c.acquire(req_id=1, context_len=30)
    got = c.alloc_gpu_pages(slot, 0, 30)
    assert got == 16  # only one page (16 tokens) available
    assert len(c._free_gpu_pages) == 0
    got2 = c.alloc_gpu_pages(slot, 16, 14)
    assert got2 == 0  # no pages left


def test_gpu_alloc_disabled_returns_zero():
    c = _make()
    slot = c.acquire(req_id=1, context_len=10)
    got = c.alloc_gpu_pages(slot, 0, 10)
    assert got == 0


def test_unregister_frees_both_pools():
    c = _make_gpu_cold(gpu_pages=4, page=16)
    slot = c.acquire(req_id=1, context_len=20)
    c.alloc_host_pages(slot, 0, 16)
    c.alloc_gpu_pages(slot, 0, 16)
    host_free_before = len(c._free_host_pages)
    gpu_free_before = len(c._free_gpu_pages)
    c.unregister_request(slot)
    assert len(c._free_host_pages) > host_free_before
    assert len(c._free_gpu_pages) > gpu_free_before
    assert torch.all(c.req_to_host_pool[slot] == _EMPTY)
    assert torch.all(c.req_to_gpu_pool[slot] == _EMPTY)


def test_grow_cold_prefers_gpu():
    c = _make_gpu_cold(gpu_pages=2, page=16)
    slot = c.acquire(req_id=1, context_len=20)
    c.grow_cold_for_new_token(slot, 0)
    assert int(c.req_to_gpu_pool[slot, 0].item()) >= 0
    assert int(c.req_to_host_pool[slot, 0].item()) == _EMPTY


def test_grow_cold_falls_back_to_host():
    c = _make_gpu_cold(gpu_pages=0, page=16)
    assert not c.gpu_cold_enabled
    slot = c.acquire(req_id=1, context_len=20)
    c.grow_cold_for_new_token(slot, 0)
    assert int(c.req_to_host_pool[slot, 0].item()) >= 0


def test_grow_cold_overflows_to_host():
    c = _make_gpu_cold(gpu_pages=1, page=16)
    slot = c.acquire(req_id=1, context_len=30)
    # Fill GPU tier (1 page = 16 tokens)
    for t in range(16):
        c.grow_cold_for_new_token(slot, t)
    assert len(c._free_gpu_pages) == 0
    # Next token must go to host
    c.grow_cold_for_new_token(slot, 16)
    assert int(c.req_to_host_pool[slot, 16].item()) >= 0
    assert int(c.req_to_gpu_pool[slot, 16].item()) == _EMPTY


def test_backup_new_token_writes_gpu_cold():
    c = _make_gpu_cold(gpu_pages=2, page=16, num_layers=1)
    slot = c.acquire(req_id=1, context_len=20)
    kv = torch.full((c.kv_dim,), 7.0, dtype=torch.bfloat16)
    c.backup_new_token(req_slot=slot, layer_id=0, new_token_kv=kv, logical_pos=0)
    gpu_row = int(c.req_to_gpu_pool[slot, 0].item())
    assert gpu_row >= 0
    row = c.gpu_cold_pool[0, gpu_row]
    assert torch.allclose(row.float(), torch.full_like(row.float(), 7.0))


def test_backup_new_token_falls_back_host():
    c = _make_gpu_cold(gpu_pages=0, page=16, num_layers=1)
    slot = c.acquire(req_id=1, context_len=20)
    kv = torch.full((c.kv_dim,), 9.0, dtype=torch.bfloat16)
    c.backup_new_token(req_slot=slot, layer_id=0, new_token_kv=kv, logical_pos=0)
    host_row = int(c.req_to_host_pool[slot, 0].item())
    assert host_row >= 0
    row = c.cold_pool[0, host_row]
    assert torch.allclose(row.float(), torch.full_like(row.float(), 9.0))


def test_gpu_locs_arg():
    c_off = _make()
    assert c_off._gpu_locs_arg().numel() == 0
    c_on = _make_gpu_cold(gpu_pages=2)
    assert c_on._gpu_locs_arg() is c_on.req_to_gpu_pool


def test_enqueue_drain_promote_queue():
    c = _make_gpu_cold(gpu_pages=4, page=16)
    slot = c.acquire(req_id=42, context_len=20)
    c.alloc_host_pages(slot, 0, 16)
    c.enqueue_promote(42)
    assert len(c._promote_queue) == 1
    # drain_promote_queue calls promote_to_gpu internally; on CPU device the
    # swap kernel is not called (promote_stream is None) but bookkeeping runs.
    result = c.drain_promote_queue()
    assert len(c._promote_queue) == 0
    # Promote moved pages from host to GPU
    if 42 in result:
        assert result[42] > 0
        # Some host pages freed, some GPU pages allocated
        for t in range(16):
            gpu_val = int(c.req_to_gpu_pool[slot, t].item())
            host_val = int(c.req_to_host_pool[slot, t].item())
            assert gpu_val >= 0 or host_val >= 0

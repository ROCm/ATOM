# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only tests for the HiSparse coordinator bookkeeping.

Exercises miss-detect + LRU allocation + logical->hot-slot translation without a
GPU. The GPU swap kernel is not invoked here: ``plan_swap_for_request`` is pure
bookkeeping, and the coordinator is built with ``device='cpu'`` so no HIP kernel
compiles. Data-movement correctness is left to on-GPU validation.
"""

import torch

from atom.hisparse.coordinator import _EMPTY, HiSparseCoordinator


def _make(num_layers=1, max_num_seqs=2, hot=4, max_ctx=32, kv_dim=8):
    return HiSparseCoordinator(
        num_layers=num_layers,
        max_num_seqs=max_num_seqs,
        hot_buffer_size=hot,
        max_context_len=max_ctx,
        kv_dim=kv_dim,
        kv_dtype=torch.bfloat16,
        device="cpu",
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
    _seed_resident(c, 0, 0, [7, 8, 9])
    topk = torch.tensor([9, 8, 7], dtype=torch.int32)
    src, dst, tr = c.plan_swap_for_request(0, 0, topk)
    assert src.numel() == 0 and dst.numel() == 0
    hot_base = c._hot_base(0)
    # 9->slot2, 8->slot1, 7->slot0
    assert tr.tolist() == [hot_base + 2, hot_base + 1, hot_base + 0]


def test_cold_start_all_miss():
    c = _make(hot=4)
    c.register_request(0, 10)
    topk = torch.tensor([1, 2, 3], dtype=torch.int32)
    src, dst, tr = c.plan_swap_for_request(0, 0, topk)
    # all three are misses, land in empty slots
    assert sorted(src.tolist()) == [c._cold_base(0) + t for t in [1, 2, 3]]
    assert src.numel() == 3 and dst.numel() == 3
    # every top-k entry now resolves to a resident hot slot
    for t in [1, 2, 3]:
        assert c.token_to_slot[0, 0, t].item() >= 0


def test_lru_eviction_picks_stale():
    c = _make(hot=3)  # 3 + 1 padded = 4 physical slots
    c.register_request(0, 20)
    # fill all 4 slots with tokens 10,11,12,13 at increasing recency
    for i, tok in enumerate([10, 11, 12, 13]):
        c.slot_token[0, 0, i] = tok
        c.last_used[0, 0, i] = i + 1  # 10 is stalest (recency 1)
        c.token_to_slot[0, 0, tok] = i
    c._tick = 4  # tick must exceed any seeded recency (real-flow invariant)
    # request a brand new token -> must evict the stalest (token 10, slot 0)
    topk = torch.tensor([20], dtype=torch.int32)
    src, dst, tr = c.plan_swap_for_request(0, 0, topk)
    assert dst.tolist() == [c._hot_base(0) + 0]  # slot 0 evicted
    assert c.token_to_slot[0, 0, 10].item() == _EMPTY  # 10 evicted
    assert c.token_to_slot[0, 0, 20].item() == 0  # 20 now resident in slot 0


def test_hit_refreshes_recency_protects_from_eviction():
    c = _make(hot=3)
    c.register_request(0, 20)
    for i, tok in enumerate([10, 11, 12, 13]):
        c.slot_token[0, 0, i] = tok
        c.last_used[0, 0, i] = i + 1
        c.token_to_slot[0, 0, tok] = i
    c._tick = 4  # tick must exceed any seeded recency (real-flow invariant)
    # touch token 10 (stalest) as a hit, then bring in a new token.
    # 10 should now be protected; 11 (next stalest) evicted instead.
    topk = torch.tensor([10, 20], dtype=torch.int32)
    src, dst, tr = c.plan_swap_for_request(0, 0, topk)
    assert c.token_to_slot[0, 0, 10].item() == 0  # still resident
    assert c.token_to_slot[0, 0, 11].item() == _EMPTY  # evicted instead


def test_padding_entries_ignored():
    c = _make()
    c.register_request(0, 10)
    _seed_resident(c, 0, 0, [5])
    topk = torch.tensor([5, -1, -1], dtype=torch.int32)
    src, dst, tr = c.plan_swap_for_request(0, 0, topk)
    assert src.numel() == 0  # only the real token, already resident
    hot_base = c._hot_base(0)
    assert tr[0].item() == hot_base + 0
    # padding entries map to hot_base (slot 0), harmless
    assert tr[1].item() == hot_base and tr[2].item() == hot_base


def test_duplicate_topk_tokens_dedup_misses():
    c = _make()
    c.register_request(0, 10)
    topk = torch.tensor([3, 3, 3], dtype=torch.int32)
    src, dst, tr = c.plan_swap_for_request(0, 0, topk)
    assert src.numel() == 1  # deduped to a single swap
    hot_base = c._hot_base(0)
    assert tr.tolist() == [tr[0].item()] * 3  # all resolve to same slot
    assert tr[0].item() == hot_base + c.token_to_slot[0, 0, 3].item()


def test_request_lifecycle_reset():
    c = _make()
    c.register_request(0, 10)
    _seed_resident(c, 0, 0, [1, 2, 3])
    c.unregister_request(0)
    assert not c.slot_active[0].item()
    assert (c.slot_token[:, 0, :] == _EMPTY).all()
    assert (c.token_to_slot[:, 0, :] == _EMPTY).all()
    # re-register reuses the slot cleanly
    c.register_request(0, 5)
    topk = torch.tensor([1], dtype=torch.int32)
    src, _, _ = c.plan_swap_for_request(0, 0, topk)
    assert src.numel() == 1  # token 1 no longer resident -> miss


def test_per_layer_independent_state():
    c = _make(num_layers=2)
    c.register_request(0, 10)
    _seed_resident(c, 0, 0, [7])  # resident only in layer 0
    topk = torch.tensor([7], dtype=torch.int32)
    src0, _, _ = c.plan_swap_for_request(0, 0, topk)
    src1, _, _ = c.plan_swap_for_request(1, 0, topk)
    assert src0.numel() == 0  # hit in layer 0
    assert src1.numel() == 1  # miss in layer 1


def test_backup_new_token_makes_resident():
    c = _make(hot=4)
    c.register_request(0, 10)
    kv = torch.ones(c.kv_dim, dtype=torch.bfloat16)
    c.backup_new_token(req_slot=0, layer_id=0, new_token_kv=kv, logical_pos=10)
    assert c.token_to_slot[0, 0, 10].item() >= 0
    assert c.context_len[0].item() == 11
    # cold pool row written
    row = c.cold_pool[0, c._cold_base(0) + 10]
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


def test_mtp_multi_token_batch_translate():
    """MTP verify: multiple query tokens share one request's top-k runs."""
    c = _make(hot=8, max_ctx=64)
    c._run_swap = lambda *a, **k: None  # CPU test: skip the GPU gather kernel
    c.register_request(0, 40)
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

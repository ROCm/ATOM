# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MLA landing to a DCP consumer writes exactly the per-token bytes.

The producer packs a decode rank's MLA rows into landing slots of the
consumer's pool and the consumer scatters them into its paged KV. Everything
runs on CPU tensors: the NIC is a memmove, the consumer's GPU scatter is a
memmove over the same segment table, and the ZMQ messages are routed by hand.
The per-token path is the byte reference, as in test_pd_mla_staging.py.
"""

import ctypes
import threading
import time
from contextlib import nullcontext

import msgpack
import numpy as np
import pytest
import torch

from atom.kv_transfer.disaggregation.landing_scatter import (
    landing_segments,
    segment_table,
)
from atom.kv_transfer.disaggregation.mooncake.mla_landing import (
    MSG_LANDING_CREDIT,
    MSG_LANDING_READY,
    LandingCredits,
    LandingReceiver,
    mla_landing_reserve_bytes,
)
from atom.kv_transfer.disaggregation.sharded_transfer import (
    build_dcp_shard_plan,
    pack_slots,
)
from atom.kv_transfer.disaggregation.types import MLA_KV_ROLE

BLOCK_SIZE = 16
TOKEN_BYTES = 576
PAGE_BYTES = BLOCK_SIZE * TOKEN_BYTES


def _mooncake():
    return pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dcp_size", [2, 4])
@pytest.mark.parametrize("interleave", [1, 2, 4])
@pytest.mark.parametrize("num_src", [1, 5, 13, 16])
def test_landing_rows_are_the_token_runs_in_destination_order(
    dcp_size, interleave, num_src
):
    rng = np.random.default_rng(num_src * 7 + interleave)
    src_ids = rng.permutation(64)[:num_src]
    dst_pages = -(-num_src // dcp_size)
    dst_ids = rng.permutation(40)[:dst_pages]
    for rank in range(dcp_size):
        plan = build_dcp_shard_plan(
            src_ids,
            block_size=BLOCK_SIZE,
            dcp_size=dcp_size,
            dcp_rank=rank,
            interleave_size=interleave,
            dst_pages=dst_pages,
        )
        rows = plan.landing_source_tokens()
        src, dst, length = plan.token_runs(dst_ids)
        want_src = np.concatenate(
            [np.arange(s, s + n) for s, n in zip(src, length)] or [np.empty(0)]
        )
        want_dst = np.concatenate(
            [np.arange(d, d + n) for d, n in zip(dst, length)] or [np.empty(0)]
        )
        np.testing.assert_array_equal(rows, want_src)
        # Row j lands at token j % block of destination page j // block.
        j = np.arange(rows.size)
        np.testing.assert_array_equal(
            dst_ids[j // BLOCK_SIZE] * BLOCK_SIZE + j % BLOCK_SIZE, want_dst
        )


@pytest.mark.parametrize("num_rows", [1, 15, 16, 17, 100, 333])
@pytest.mark.parametrize("slot_rows", [16, 40, 64, 200])
def test_pack_slots_covers_every_row_and_splits_on_pages(num_rows, slot_rows):
    widths = [TOKEN_BYTES, TOKEN_BYTES, TOKEN_BYTES]
    slots = pack_slots(widths, num_rows, slot_rows * TOKEN_BYTES, BLOCK_SIZE)
    seen = {r: np.zeros(num_rows, dtype=int) for r in range(len(widths))}
    for items in slots:
        end = 0
        for region, start, stop, offset in items:
            assert offset == end  # packed back to back
            end = offset + (stop - start) * TOKEN_BYTES
            seen[region][start:stop] += 1
            # A region only breaks at a page boundary, except at its end.
            assert start % BLOCK_SIZE == 0
            assert stop == num_rows or stop % BLOCK_SIZE == 0
        assert end <= slot_rows * TOKEN_BYTES
    for counts in seen.values():
        assert (counts == 1).all()


def test_pack_slots_rejects_a_slot_smaller_than_a_page():
    with pytest.raises(ValueError, match="cannot hold"):
        pack_slots([TOKEN_BYTES], 40, 15 * TOKEN_BYTES, BLOCK_SIZE)


def test_pack_slots_can_start_regions_at_later_pages():
    slots = pack_slots([PAGE_BYTES] * 3, 6, 4 * PAGE_BYTES, 1, [6, 2, 0])
    pages = {r: [] for r in range(3)}
    for items in slots:
        for region, start, stop, _ in items:
            pages[region].extend(range(start, stop))
    assert pages == {0: [], 1: [2, 3, 4, 5], 2: [0, 1, 2, 3, 4, 5]}


def test_landing_segments_match_a_row_by_row_reference():
    rng = np.random.default_rng(3)
    dst_blocks = rng.permutation(50)[:9]  # 144 rows
    items = np.array(
        [  # region, row_start, row_count, slot offset
            [0, 0, 144, 0],
            [1, 32, 50, 144 * TOKEN_BYTES],
            [2, 128, 16, 194 * TOKEN_BYTES],
        ]
    )
    bases = np.array([10_000_000, 30_000_000, 50_000_000])
    origin = 1_000_000
    src, dst, nbytes = landing_segments(
        items[:, 1],
        items[:, 2],
        items[:, 3] + 7 * TOKEN_BYTES,
        bases[items[:, 0]],
        np.full(3, PAGE_BYTES),
        dst_blocks,
        BLOCK_SIZE,
        origin,
    )
    assert (nbytes <= PAGE_BYTES).all()
    got = {}
    for s, d, n in zip(src, dst, nbytes):
        for k in range(n // TOKEN_BYTES):
            got[s + k * TOKEN_BYTES] = d + k * TOKEN_BYTES
    want = {}
    for region, start, count, offset in items:
        for i in range(count):
            row = start + i
            src_addr = offset + 7 * TOKEN_BYTES + i * TOKEN_BYTES
            want[src_addr] = (
                bases[region]
                - origin
                + dst_blocks[row // BLOCK_SIZE] * PAGE_BYTES
                + (row % BLOCK_SIZE) * TOKEN_BYTES
            )
    assert got == want
    table = np.zeros((3, src.size + 5), dtype=np.int64)
    n = segment_table(src, dst, nbytes, table)
    assert n == src.size
    np.testing.assert_array_equal(table[2, :n], nbytes // 4)


# ---------------------------------------------------------------------------
# producer credits
# ---------------------------------------------------------------------------


def test_credits_follow_the_advertised_partition_and_epoch():
    credits = LandingCredits()
    credits.sync("d0", {"epoch": 1, "slots": [4, 5]})
    got = {credits.acquire("d0", 1, 0), credits.acquire("d0", 1, 0)}
    assert got == {4, 5}
    assert credits.acquire("d0", 1, 0) is None  # exhausted
    assert credits.acquire("d0", 2, 0) is None  # stale epoch
    credits.release("d0", 1, [5])
    credits.release("d0", 1, [5])  # duplicate: ignored
    assert credits.acquire("d0", 1, 0) == 5
    assert credits.acquire("d0", 1, 0) is None
    # A restarted decode rank comes with a new epoch and a fresh partition.
    credits.sync("d0", {"epoch": 2, "slots": [0]})
    credits.release("d0", 1, [4])  # old epoch: ignored
    assert credits.acquire("d0", 2, 0) == 0


def test_credit_wait_wakes_on_release():
    credits = LandingCredits()
    credits.sync("d0", {"epoch": 9, "slots": [3]})
    assert credits.acquire("d0", 9, 0) == 3
    got = []
    waiter = threading.Thread(target=lambda: got.append(credits.acquire("d0", 9, 5)))
    waiter.start()
    time.sleep(0.05)
    credits.release("d0", 9, [3])
    waiter.join(timeout=5)
    assert got == [3]


# ---------------------------------------------------------------------------
# consumer receiver
# ---------------------------------------------------------------------------


class _Dest:
    """Consumer MLA regions plus a CPU receiver whose scatter is a memmove."""

    def __init__(self, num_regions=2, num_blocks=48, pool_slots=8, slot_rows=64):
        gen = torch.Generator().manual_seed(11)
        self.regions = [
            torch.randint(0, 256, (num_blocks, PAGE_BYTES), generator=gen).to(
                torch.uint8
            )
            for _ in range(num_regions)
        ]
        self.sent = []
        self.finished = []
        self.recv = LandingReceiver(
            device=None,
            pool_slots=pool_slots,
            slot_bytes=slot_rows * TOKEN_BYTES,
            block_size=BLOCK_SIZE,
            consumer_key="consumer:1",
            region_bases=[t.data_ptr() for t in self.regions],
            region_block_bytes=[PAGE_BYTES] * num_regions,
            mla_regions=list(range(num_regions)),
            send=lambda addr, parts: self.sent.append((addr, parts)),
            finish=lambda req, failed: self.finished.append((req, failed)),
        )
        pool = self.recv.pool

        def copy(src, dst, nbytes, _events):
            # The GPU kernel reads any offset: one outside the pool is a fault.
            assert ((src >= 0) & (src <= pool.numel() - nbytes)).all()
            for s, d, n in zip(src, dst, nbytes):
                ctypes.memmove(
                    self.recv._dst_origin + int(d), pool.data_ptr() + int(s), int(n)
                )

        self.recv._copy_segments = copy

    def credits(self):
        out = []
        for addr, (kind, payload) in self.sent:
            assert kind == MSG_LANDING_CREDIT
            out.append((addr, msgpack.loads(payload)["slots"]))
        self.sent.clear()
        return out

    def pump(self):
        self.recv._process(self.recv._drain(block=False))


def _ready(req, slot, seq, items, pp=0, nonce=5):
    return {
        "request_id": req,
        "write_nonce": nonce,
        "pp_rank": pp,
        "seq": seq,
        "slot": slot,
        "items": items,
    }


def _land(dest, slot, region, row_start, rows):
    """Put ``rows`` (uint8 [n, 576]) into a landing slot as the NIC would."""
    dest.recv.pool[slot, : rows.numel()].copy_(rows.reshape(-1))
    return [[region, row_start, rows.shape[0], 0]]


def test_receiver_partitions_the_pool_per_stage_endpoint():
    dest = _Dest(pool_slots=8)
    parts = [dest.recv.advertise(f"s{i}", 4)["slots"] for i in range(4)]
    assert sorted(s for p in parts for s in p) == list(range(8))
    assert all(len(p) == 2 for p in parts)
    assert dest.recv.advertise("s0", 4)["slots"] == parts[0]  # stable
    assert dest.recv.advertise("s4", 4) is None  # pool spent


def test_ready_then_write_done_completes_only_after_the_scatter():
    dest = _Dest()
    slots = dest.recv.advertise("stage0", 1)["slots"]
    dst_blocks = [5, 9]
    dest.recv.begin("r", 5, dst_blocks, {0: "stage0"})
    rows = torch.randint(0, 256, (20, TOKEN_BYTES), dtype=torch.uint8)
    dest.recv.on_ready(_ready("r", slots[0], 0, _land(dest, slots[0], 1, 0, rows)))
    # The write-done arrives while the slot is still queued: not complete.
    assert dest.recv.stage_done("r", 0, 5, True, [slots[0]])
    assert dest.finished == []
    dest.pump()
    assert dest.finished == [("r", False)]
    assert dest.credits() == [("stage0", [slots[0]])]
    region = dest.regions[1].view(-1, TOKEN_BYTES)
    for j in range(20):
        token = dst_blocks[j // BLOCK_SIZE] * BLOCK_SIZE + j % BLOCK_SIZE
        assert torch.equal(region[token], rows[j])


def test_duplicates_are_ignored_and_lost_ready_fails_the_request():
    dest = _Dest()
    slots = dest.recv.advertise("stage0", 1)["slots"]
    dest.recv.begin("r", 5, [1, 2], {0: "stage0"})
    rows = torch.zeros((4, TOKEN_BYTES), dtype=torch.uint8)
    ready = _ready("r", slots[0], 0, _land(dest, slots[0], 0, 0, rows))
    dest.recv.on_ready(ready)
    dest.recv.on_ready(ready)  # duplicate READY: no second scatter
    assert dest.recv._requests["r"].pending == 1
    dest.pump()
    # The stage landed 2 slots, but the second READY never arrived: the
    # request fails and the lost slot still goes back to the stage.
    dest.recv.stage_done("r", 0, 5, True, [slots[0], slots[1]])
    dest.recv.stage_done("r", 0, 5, True, [slots[0], slots[1]])  # duplicate
    assert dest.finished == [("r", True)]
    assert dest.credits() == [("stage0", [slots[0]]), ("stage0", [slots[1]])]


def test_a_ready_behind_its_write_done_returns_its_slot_once():
    # ZMQ keeps order only within one connection: after a reconnect, a
    # stage's write-done can overtake a READY still queued on the old one.
    # The write-done returns the slot as lost; the READY must not return it
    # again, or the stage could hand the slot to two transfers at once.
    dest = _Dest()
    s0 = dest.recv.advertise("stage0", 2)["slots"]
    dest.recv.advertise("stage1", 2)
    dest.recv.begin("r", 5, [1], {0: "stage0", 1: "stage1"})
    dest.recv.stage_done("r", 0, 5, True, [s0[0]])
    assert dest.credits() == [("stage0", [s0[0]])]
    dest.recv.on_ready(_ready("r", s0[0], 0, [[0, 0, 1, 0]]))
    assert dest.recv._queue.empty()
    assert dest.credits() == []
    assert dest.finished == []  # stage 1 may still be writing
    dest.recv.stage_done("r", 1, 5, True, [])
    assert dest.finished == [("r", True)]
    assert dest.credits() == []


def test_a_ready_after_its_request_retired_returns_nothing():
    # The same reordered READY, arriving once every stage ended: the
    # write-done that counted it lost already returned its slot.
    dest = _Dest()
    s0 = dest.recv.advertise("stage0", 2)["slots"]
    dest.recv.advertise("stage1", 2)
    dest.recv.begin("r", 5, [1], {0: "stage0", 1: "stage1"})
    dest.recv.stage_done("r", 0, 5, True, [s0[0]])
    dest.recv.stage_done("r", 1, 5, True, [])
    assert dest.finished == [("r", True)]
    assert dest.credits() == [("stage0", [s0[0]])]
    dest.recv.on_ready(_ready("r", s0[0], 0, [[0, 0, 1, 0]]))
    assert dest.recv._queue.empty()
    assert dest.credits() == []


@pytest.mark.parametrize(
    "landed", [5, [[1]], "ab", [None]], ids=["int", "nested", "string", "none-slot"]
)
def test_malformed_landed_slots_fail_the_request(landed):
    # They hide which READY never arrived, so the request cannot succeed;
    # it still settles, since the stage ended.
    dest = _Dest()
    dest.recv.advertise("stage0", 1)
    dest.recv.begin("r", 5, [1], {0: "stage0"})
    dest.recv.stage_done("r", 0, 5, True, landed)
    assert dest.finished == [("r", True)]
    assert dest.credits() == []


@pytest.mark.parametrize("landed", [[10_000], [True]], ids=["outside", "bool"])
def test_landed_slots_outside_the_partition_fail_the_request(landed):
    # The stage was never given such a slot, so the list cannot be trusted:
    # the request fails instead of settling without those rows.
    dest = _Dest()
    dest.recv.advertise("stage0", 1)
    dest.recv.begin("r", 5, [1], {0: "stage0"})
    dest.recv.stage_done("r", 0, 5, True, landed)
    assert dest.finished == [("r", True)]
    assert dest.credits() == []


@pytest.mark.parametrize("listed", ["other-slot", "missing"])
def test_a_write_done_that_disagrees_with_its_readys_fails_the_request(listed):
    # A READY that named another slot for its seq, or that the list leaves
    # out, was scattered from a slot the stage may not have written.
    dest = _Dest()
    s0 = dest.recv.advertise("stage0", 1)["slots"]
    dest.recv.begin("r", 5, [1], {0: "stage0"})
    dest.recv.on_ready(_ready("r", s0[0], 0, [[0, 0, 1, 0]]))
    dest.pump()
    assert dest.credits() == [("stage0", [s0[0]])]
    dest.recv.stage_done("r", 0, 5, True, [s0[1]] if listed == "other-slot" else [])
    assert dest.finished == [("r", True)]
    assert dest.credits() == []


def test_a_ready_missing_from_its_write_done_fails_the_request():
    # The write-done ended the stage without this READY, so its rows are not
    # part of the request; its slot goes back once.
    dest = _Dest()
    s0 = dest.recv.advertise("stage0", 2)["slots"]
    dest.recv.advertise("stage1", 2)
    dest.recv.begin("r", 5, [1], {0: "stage0", 1: "stage1"})
    dest.recv.stage_done("r", 0, 5, True, [])
    dest.recv.on_ready(_ready("r", s0[0], 0, [[0, 0, 1, 0]]))
    assert dest.recv._queue.empty()
    assert dest.credits() == [("stage0", [s0[0]])]
    dest.recv.stage_done("r", 1, 5, True, [])
    assert dest.finished == [("r", True)]


def test_a_trailing_ready_that_names_another_slot_fails_the_request():
    dest = _Dest()
    s0 = dest.recv.advertise("stage0", 2)["slots"]
    dest.recv.advertise("stage1", 2)
    dest.recv.begin("r", 5, [1], {0: "stage0", 1: "stage1"})
    dest.recv.stage_done("r", 0, 5, True, [s0[0]])
    assert dest.credits() == [("stage0", [s0[0]])]
    dest.recv.on_ready(_ready("r", s0[1], 0, [[0, 0, 1, 0]]))
    assert dest.recv._queue.empty()
    assert dest.credits() == []
    dest.recv.stage_done("r", 1, 5, True, [])
    assert dest.finished == [("r", True)]


def test_a_failed_stage_waits_for_the_others_and_late_slots_are_dropped():
    dest = _Dest()
    s0 = dest.recv.advertise("stage0", 2)["slots"]
    s1 = dest.recv.advertise("stage1", 2)["slots"]
    dest.recv.begin("r", 5, [1, 2, 3], {0: "stage0", 1: "stage1"})
    dest.recv.stage_done("r", 1, 5, False, [s1[0]])
    assert dest.credits() == [("stage1", [s1[0]])]
    assert dest.finished == []  # stage 0 may still be writing
    before = dest.regions[0].clone()
    rows = torch.ones((8, TOKEN_BYTES), dtype=torch.uint8)
    dest.recv.on_ready(_ready("r", s0[0], 0, _land(dest, s0[0], 0, 0, rows)))
    dest.pump()
    assert torch.equal(dest.regions[0], before)  # never scattered
    assert dest.credits() == [("stage0", [s0[0]])]
    dest.recv.stage_done("r", 0, 5, True, [s0[0]])
    assert dest.finished == [("r", True)]
    # A READY arriving after the request retired returns no credit: its
    # stage's write-done already returned every slot it listed.
    dest.recv.on_ready(_ready("r", s1[0], 0, [[0, 0, 1, 0]], pp=1))
    assert dest.credits() == []
    assert "r" not in dest.recv._requests


def test_slot_outside_the_stage_partition_fails_the_request():
    dest = _Dest()
    s0 = dest.recv.advertise("stage0", 2)["slots"]
    s1 = dest.recv.advertise("stage1", 2)["slots"]
    dest.recv.begin("r", 5, [1], {0: "stage0", 1: "stage1"})
    dest.recv.on_ready(_ready("r", s1[0], 0, [[0, 0, 1, 0]], pp=0))
    dest.recv.stage_done("r", 0, 5, True, [])
    dest.recv.stage_done("r", 1, 5, True, [])
    assert dest.finished == [("r", True)]
    assert s0 and dest.credits() == []


@pytest.mark.parametrize(
    "item",
    [
        [0, 0, 17, 0],  # 1 page = 16 rows
        [0, 0, 1, -4],
        [0, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 1, 2],
        [0, 2**62, 2**62, 0],
        [0, 0, 1, 2**63 - 4],
    ],
    ids=[
        "rows-past-the-pages",
        "negative-offset",
        "no-rows",
        "negative-rows",
        "unaligned-offset",
        "rows-wrap-int64",
        "offset-wraps-int64",
    ],
)
def test_malformed_items_fail_only_their_request(item):
    dest = _Dest()
    s0 = dest.recv.advertise("stage0", 1)["slots"]
    dest.recv.begin("r", 5, [1], {0: "stage0"})
    before = [t.clone() for t in dest.regions]
    dest.recv.on_ready(_ready("r", s0[0], 0, [item]))
    dest.pump()
    dest.recv.stage_done("r", 0, 5, True, [s0[0]])
    assert dest.finished == [("r", True)]
    assert all(torch.equal(a, b) for a, b in zip(dest.regions, before))
    assert dest.credits() == [("stage0", [s0[0]])]
    assert dest.recv.advertise("stage0", 1) is not None  # landing stays on


_NO_ITEMS = object()  # the READY has no items field


@pytest.mark.parametrize(
    "items",
    [
        [1],
        [[0, 0, 16]],
        [[0, 0, 16, 0], [1]],
        [],
        "abcd",
        None,
        [[None, 0, 16, 0]],
        [[2**63, 0, 16, 0]],
        _NO_ITEMS,
    ],
    ids=[
        "flat",
        "three-columns",
        "ragged",
        "empty",
        "string",
        "none",
        "none-value",
        "past-int64",
        "missing",
    ],
)
def test_unparsable_items_fail_their_request_and_return_the_slot(items):
    # on_ready runs on the listener thread: such items must not raise there,
    # nor reach the scatter, where they would fail the batch and disable landing.
    dest = _Dest()
    s0 = dest.recv.advertise("stage0", 2)["slots"]
    dest.recv.advertise("stage1", 2)
    dest.recv.begin("r", 5, [1], {0: "stage0", 1: "stage1"})
    ready = _ready("r", s0[0], 0, items)
    if items is _NO_ITEMS:
        del ready["items"]
    dest.recv.on_ready(ready)
    assert dest.recv._queue.empty()
    # The slot's RDMA write finished before READY, so it goes back at once.
    assert dest.credits() == [("stage0", [s0[0]])]
    dest.recv.stage_done("r", 0, 5, True, [s0[0]])
    assert dest.finished == []  # stage 1 may still be writing
    dest.recv.stage_done("r", 1, 5, True, [])
    assert dest.finished == [("r", True)]
    # Stage 0's write-done listed the slot, but it went back only once.
    assert dest.credits() == []
    assert dest.recv.advertise("stage0", 2) is not None  # landing stays on


def test_sweep_never_reports_a_failure_while_a_stage_still_runs():
    # Only a stage's write-done proves it stopped writing the request's
    # pages; reporting the failure earlier lets the scheduler reuse them.
    dest = _Dest()
    dest.recv.advertise("stage0", 2)
    dest.recv.advertise("stage1", 2)
    dest.recv.begin("r", 5, [1], {0: "stage0", 1: "stage1"})
    dest.recv.stage_done("r", 1, 5, False, [])
    dest.recv._requests["r"].failed_at -= 3600
    dest.recv.sweep()
    dest.recv.sweep()
    assert dest.finished == []
    assert dest.recv._requests["r"].stall_logged
    dest.recv.stage_done("r", 0, 5, True, [])
    assert dest.finished == [("r", True)]


def test_scatter_errors_fail_the_slots_and_disable_landing():
    dest = _Dest()
    s0 = dest.recv.advertise("stage0", 1)["slots"]
    dest.recv.begin("r", 5, [1, 2], {0: "stage0"})

    def boom(*_a):
        raise RuntimeError("gpu fault")

    dest.recv._copy_segments = boom
    rows = torch.zeros((4, TOKEN_BYTES), dtype=torch.uint8)
    dest.recv.on_ready(_ready("r", s0[0], 0, _land(dest, s0[0], 0, 0, rows)))
    dest.pump()
    dest.recv.stage_done("r", 0, 5, True, [s0[0]])
    assert dest.finished == [("r", True)]
    # Landing is off: no stage gets an offer, not even one with a partition.
    assert dest.recv.advertise("stage0", 1) is None
    assert dest.recv.advertise("stage9", 1) is None


def test_ready_for_an_unknown_request_returns_nothing():
    # It belongs to no transfer this rank tracks; returning its slot could
    # free one the stage holds for another transfer.
    dest = _Dest()
    s0 = dest.recv.advertise("stage0", 1)["slots"]
    dest.recv.begin("r", 5, [1], {0: "stage0"})
    dest.recv.on_ready(_ready("ghost", s0[1], 0, [[0, 0, 1, 0]]))
    dest.recv.on_ready(_ready("r", s0[1], 0, [[0, 0, 1, 0]], nonce=6))
    assert dest.recv._queue.empty()
    assert dest.credits() == []


def test_a_failed_credit_send_still_reports_the_request():
    # Credits go out after the bookkeeping. A raise there would end the
    # caller half done: a READY misreported as unread, a request never reported.
    dest = _Dest()
    s0 = dest.recv.advertise("stage0", 1)["slots"]
    dest.recv.begin("r", 5, [1], {0: "stage0"})

    def unreachable(_addr, _parts):
        raise RuntimeError("notify socket gone")

    dest.recv._send = unreachable
    rows = torch.zeros((4, TOKEN_BYTES), dtype=torch.uint8)
    dest.recv.on_ready(_ready("r", s0[0], 0, _land(dest, s0[0], 0, 0, rows)))
    dest.recv.on_ready(_ready("r", s0[1], 1, [1]))  # malformed: returned at once
    dest.pump()  # slot 0 is returned once its task completes
    dest.recv.stage_done("r", 0, 5, True, s0[:3])  # seq 2 lost: returned now
    assert dest.finished == [("r", True)]


# ---------------------------------------------------------------------------
# producer + consumer: bytes equal the per-token path
# ---------------------------------------------------------------------------


class _Nic:
    def __init__(self):
        self.labels = []

    def write(self, _target, src, dst, sizes, _req, label, *, engine=None):
        for s, d, n in zip(src, dst, sizes):
            ctypes.memmove(d, s, n)
        self.labels.append((label, len(src)))
        return True


class _Stream:
    def wait_event(self, _event):
        pass

    def synchronize(self):
        pass


def _source_regions(num_regions, num_blocks):
    gen = torch.Generator().manual_seed(num_regions * 1000 + num_blocks)
    return [
        torch.randint(0, 256, (num_blocks, PAGE_BYTES), generator=gen).to(torch.uint8)
        for _ in range(num_regions)
    ]


def _producer(mc, regions, *, staging_rows=None):
    conn = object.__new__(mc.MooncakeConnector)
    conn.dcp_size = 1
    conn.block_size = BLOCK_SIZE
    conn.pp_rank = 0
    conn.tp_rank = 0
    conn.kv_caches_base_addr = [t.data_ptr() for t in regions]
    conn._per_block_bytes_list = [PAGE_BYTES] * len(regions)
    conn._block_region_roles = [MLA_KV_ROLE] * len(regions)
    conn._block_region_consumer_indices = None
    conn._consumer_region_map = lambda *_a, **_k: list(range(len(regions)))
    conn._prepare_sharded_index = None
    conn._gather_sharded_index = None
    conn._index_staging_chunk_pages = 0
    conn._nic = _Nic()
    conn._rdma_write_with_retry = conn._nic.write
    if staging_rows:
        conn._mla_staging = torch.empty(
            (2, staging_rows * TOKEN_BYTES), dtype=torch.uint8
        )
        conn._mla_staging_free = [0, 1]
        conn._mla_staging_cv = threading.Condition()
        conn._stream = _Stream()
        conn._send_worker_stream = lambda: conn._stream
        conn._mla_token_views = {
            i: t.view(-1, TOKEN_BYTES) for i, t in enumerate(regions)
        }
        conn._landing_credits = LandingCredits()
    return conn


def _request(num_regions, dcp_size, dcp_rank, dst_bufs):
    return {
        "request_id": "req",
        "write_nonce": 5,
        "notify_host": "decode",
        "notify_port": 1,
        "consumer_base_addrs": [t.data_ptr() for t in dst_bufs],
        "consumer_num_layers": num_regions,
        "consumer_region_roles": [MLA_KV_ROLE] * num_regions,
        "consumer_block_bpb": [PAGE_BYTES] * num_regions,
        "consumer_dcp_size": dcp_size,
        "consumer_dcp_rank": dcp_rank,
        "consumer_dcp_interleave": 1,
    }


def _landed_vs_per_token(
    monkeypatch,
    *,
    dcp_size,
    dcp_rank,
    num_src,
    pool_slots,
    slot_rows=48,
    pump_during=True,
    credit_wait_ms=2000,
):
    mc = _mooncake()
    monkeypatch.setattr(mc.torch.cuda, "stream", lambda _s: nullcontext())
    monkeypatch.setattr(mc, "MLA_LANDING_MIN_SLOTS", 1)
    monkeypatch.setattr(mc, "MLA_LANDING_CREDIT_WAIT_S", credit_wait_ms / 1000)
    num_regions = 3
    regions = _source_regions(num_regions, 40)
    rng = np.random.default_rng(dcp_size * 10 + dcp_rank + num_src)
    src_ids = rng.permutation(40)[:num_src]
    dst_ids = rng.permutation(48)[: -(-num_src // dcp_size)]

    # Reference: per-token writes into a copy of the consumer regions.
    dest = _Dest(num_regions=num_regions, pool_slots=pool_slots, slot_rows=slot_rows)
    reference = [t.clone() for t in dest.regions]
    old = _producer(mc, regions)
    assert mc.MooncakeConnector._execute_block_transfer(
        old,
        _request(num_regions, dcp_size, dcp_rank, reference),
        "consumer:1",
        list(src_ids),
        list(dst_ids),
        "req",
        object(),
    )

    new = _producer(mc, regions, staging_rows=slot_rows)
    landing = dest.recv.advertise("stage0", 1)
    new._landing_credits.sync("consumer:1", landing)
    request = _request(num_regions, dcp_size, dcp_rank, dest.regions)
    request["mla_landing"] = landing
    dest.recv.begin("req", 5, list(dst_ids), {0: "stage0"})

    def route(_path, parts):
        kind, payload = parts
        assert kind == MSG_LANDING_READY
        dest.recv.on_ready(msgpack.loads(payload))

    new._send_on_socket = route
    stop = threading.Event()

    def pump():
        while not stop.is_set():
            dest.pump()
            for addr, slots in dest.credits():
                assert addr == "stage0"
                new._landing_credits.release("consumer:1", landing["epoch"], slots)
            time.sleep(0.001)

    pumper = threading.Thread(target=pump)
    if pump_during:
        pumper.start()
    try:
        assert mc.MooncakeConnector._execute_block_transfer(
            new, request, "consumer:1", list(src_ids), list(dst_ids), "req", object()
        )
    finally:
        if not pump_during:
            pumper.start()
        deadline = time.monotonic() + 10
        while dest.recv._requests.get("req") and dest.recv._requests["req"].pending:
            assert time.monotonic() < deadline
            time.sleep(0.005)
        stop.set()
        pumper.join()
    dest.recv.stage_done("req", 0, 5, True, request.get("_mla_landed", []))
    assert dest.finished == [("req", False)]
    for got, want in zip(dest.regions, reference):
        assert torch.equal(got, want)
    return new


@pytest.mark.parametrize(("dcp_size", "dcp_rank"), [(2, 0), (2, 1), (4, 0), (4, 3)])
@pytest.mark.parametrize("num_src", [4, 13, 32])
def test_landed_mla_matches_per_token_bytes(monkeypatch, dcp_size, dcp_rank, num_src):
    new = _landed_vs_per_token(
        monkeypatch,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        num_src=num_src,
        pool_slots=2,
    )
    labels = [label for label, _ in new._nic.labels]
    assert set(labels) == {"landed-mla"}
    # One descriptor per landing slot.
    assert all(n == 1 for _, n in new._nic.labels)


def test_landing_falls_back_to_staged_pages_without_credits(monkeypatch):
    # One slot, nobody returns it until the transfer is over: the first slot
    # lands, the rest goes out as staged pages, and the bytes still match.
    new = _landed_vs_per_token(
        monkeypatch,
        dcp_size=4,
        dcp_rank=1,
        num_src=32,
        pool_slots=1,
        pump_during=False,
        credit_wait_ms=0,
    )
    labels = [label for label, _ in new._nic.labels]
    assert labels[0] == "landed-mla"
    assert labels.count("landed-mla") == 1
    assert "staged-mla" in labels


def test_small_transfers_keep_the_staged_path(monkeypatch):
    mc = _mooncake()
    monkeypatch.setattr(mc.torch.cuda, "stream", lambda _s: nullcontext())
    monkeypatch.setattr(mc, "MLA_LANDING_MIN_SLOTS", 2)
    regions = _source_regions(1, 8)
    new = _producer(mc, regions, staging_rows=512)
    dst = [torch.zeros((8, PAGE_BYTES), dtype=torch.uint8)]
    request = _request(1, 4, 0, dst)
    request["mla_landing"] = {
        "epoch": 1,
        "base": 0,
        "slot_bytes": 1 << 20,
        "slots": [0],
    }
    new._landing_credits.sync("consumer:1", request["mla_landing"])
    assert mc.MooncakeConnector._execute_block_transfer(
        new, request, "consumer:1", [0, 1, 2, 3], [5], "req", object()
    )
    assert [label for label, _ in new._nic.labels] == ["staged-mla"]
    assert "_mla_landed" not in request


def test_a_failure_waits_out_another_stages_staged_fallback(monkeypatch):
    # Stage 1 finds no landing credit and writes the final KV page through
    # the staged path while stage 0 fails. However long that write takes, the
    # failure is reported only after stage 1's write-done, so the scheduler
    # never recomputes into a page stage 1 may still overwrite.
    mc = _mooncake()
    monkeypatch.setattr(mc.torch.cuda, "stream", lambda _s: nullcontext())
    monkeypatch.setattr(mc, "MLA_LANDING_MIN_SLOTS", 1)
    monkeypatch.setattr(mc, "MLA_LANDING_CREDIT_WAIT_S", 0)
    dest = _Dest(num_regions=1, pool_slots=2, slot_rows=16)
    dest.recv.advertise("stage0", 2)
    landing = dest.recv.advertise("stage1", 2)
    dest.recv.begin("req", 5, [3], {0: "stage0", 1: "stage1"})
    stage1 = _producer(mc, _source_regions(1, 8), staging_rows=16)
    stage1.pp_rank = 1
    stage1._landing_credits.sync("consumer:1", landing)
    # Another request holds the stage's only credit.
    assert stage1._landing_credits.acquire("consumer:1", landing["epoch"], 0) == 1
    request = _request(1, 4, 0, dest.regions)
    request["mla_landing"] = landing

    started, resume = threading.Event(), threading.Event()

    def slow_write(*args, **kwargs):
        started.set()
        assert resume.wait(timeout=10)
        return stage1._nic.write(*args, **kwargs)

    stage1._rdma_write_with_retry = slow_write
    result = []
    writer = threading.Thread(
        target=lambda: result.append(
            mc.MooncakeConnector._execute_block_transfer(
                stage1, request, "consumer:1", [0, 1, 2, 3], [3], "req", object()
            )
        )
    )
    writer.start()
    assert started.wait(timeout=10)
    dest.recv.stage_done("req", 0, 5, False, [])
    dest.recv._requests["req"].failed_at -= 3600
    dest.recv.sweep()
    assert dest.finished == []  # stage 1 is still writing page 3
    resume.set()
    writer.join(timeout=10)
    assert result == [True]
    assert [label for label, _ in stage1._nic.labels] == ["staged-mla"]
    dest.recv.stage_done("req", 1, 5, True, request.get("_mla_landed", []))
    assert dest.finished == [("req", True)]


def test_a_failed_ready_send_still_returns_the_written_slot(monkeypatch):
    # The slot's RDMA write finished but announcing it raised. The failure
    # write-done must still list the slot, or the consumer never returns it
    # and the stage loses the credit for good.
    mc = _mooncake()
    monkeypatch.setattr(mc.torch.cuda, "stream", lambda _s: nullcontext())
    monkeypatch.setattr(mc, "MLA_LANDING_MIN_SLOTS", 1)
    dest = _Dest(num_regions=1, pool_slots=1, slot_rows=16)
    landing = dest.recv.advertise("stage0", 1)
    dest.recv.begin("req", 5, [3], {0: "stage0"})
    new = _producer(mc, _source_regions(1, 8), staging_rows=16)
    new._landing_credits.sync("consumer:1", landing)
    request = _request(1, 4, 0, dest.regions)
    request["mla_landing"] = landing

    def unreachable(_path, _parts):
        raise RuntimeError("notify socket gone")

    new._send_on_socket = unreachable
    with pytest.raises(RuntimeError):
        mc.MooncakeConnector._execute_block_transfer(
            new, request, "consumer:1", [0, 1, 2, 3], [3], "req", object()
        )
    assert [label for label, _ in new._nic.labels] == ["landed-mla"]

    # What _execute_transfer's handler then sends.
    sent = []
    new._send_on_socket = lambda _path, parts, repeat=1: sent.append(parts)
    new._notify_transfer_result(request, success=False)
    done = msgpack.loads(sent[0][1])
    assert done["landed_slots"] == landing["slots"]
    dest.recv.stage_done(
        "req", done["pp_rank"], done["write_nonce"], False, done["landed_slots"]
    )
    assert dest.finished == [("req", True)]
    for _addr, slots in dest.credits():
        new._landing_credits.release("consumer:1", landing["epoch"], slots)
    assert new._landing_credits.acquire("consumer:1", landing["epoch"], 0) == 0


# ---------------------------------------------------------------------------
# connector wiring
# ---------------------------------------------------------------------------


def test_write_done_reports_landed_slots_only_when_offered():
    mc = _mooncake()
    conn = object.__new__(mc.MooncakeConnector)
    conn.pp_rank = 2
    conn.tp_rank = 0
    sent = []
    conn._send_on_socket = lambda path, parts, repeat=1: sent.append(parts)
    base = {"notify_host": "h", "notify_port": 1, "request_id": "r"}
    conn._notify_transfer_result(dict(base), success=True)
    conn._notify_transfer_result(
        {**base, "mla_landing": {}, "_mla_landed": [4, 4, 5]}, success=True
    )
    # A None offer is no offer, as for the write listener and the transfer.
    conn._notify_transfer_result({**base, "mla_landing": None}, success=True)
    first, second, third = (msgpack.loads(p[1]) for p in sent)
    assert first["landed_slots"] is None
    assert second["landed_slots"] == [4, 4, 5]
    assert third["landed_slots"] is None


def test_landing_requests_complete_through_the_receiver():
    mc = _mooncake()
    dest = _Dest()
    conn = object.__new__(mc.MooncakeConnector)
    conn._mla_landing = dest.recv
    conn._completion_lock = threading.Lock()
    conn._fence_lock = threading.Lock()
    conn._blocks_pending_fence = []
    conn._pending_recv = {"r"}
    conn._pending_recv_blocks = {"r": [1]}
    conn._pending_recv_slots = {}
    conn._pending_recv_expected = {"r": 1}
    conn._pending_recv_stages = {}
    conn._pending_recv_failed = set()
    conn._pending_recv_nonce = {"r": 5}
    conn._release_targets = {}
    conn.done_recving = set()
    conn.failed_recving = set()
    dest.recv._finish = conn._complete_recv
    s0 = dest.recv.advertise("stage0", 1)["slots"]
    dest.recv.begin("r", 5, [1], {0: "stage0"})
    rows = torch.zeros((4, TOKEN_BYTES), dtype=torch.uint8)
    dest.recv.on_ready(_ready("r", s0[0], 0, _land(dest, s0[0], 0, 0, rows)))
    conn._record_write_done("r", 0, 0, 5, success=True, landed_slots=[s0[0]])
    assert conn.done_recving == set()  # scatter still queued
    dest.pump()
    assert conn.done_recving == {"r"}
    assert conn._pending_recv_expected == {}
    assert conn._pending_recv == set()


def _listen(monkeypatch, mc, listener, messages):
    """Run a connector's listener loop over ``(type, payload)`` messages."""

    class _Stop(Exception):
        pass

    class _Socket:
        def recv_multipart(self):
            if not messages:
                raise _Stop
            return [b"peer", *messages.pop(0)]

    def fake_socket(*_args, **_kwargs):
        return nullcontext(_Socket())

    # The producer's listener binds its own socket; the consumer's adopts the
    # one bound when its port was chosen.
    monkeypatch.setattr(mc, "zmq_socket_ctx", fake_socket)
    monkeypatch.setattr(mc, "_owned_zmq_socket", fake_socket)
    with pytest.raises(_Stop):
        listener()


def _consumer(mc, dest):
    conn = object.__new__(mc.MooncakeConnector)
    conn._notification_port = 1
    conn._notification_ctx = conn._notification_sock = None
    conn._mla_landing = dest.recv
    return conn


def test_an_unreadable_ready_does_not_stop_the_listener(monkeypatch):
    # The READY behind unreadable ones is still received. One with no good
    # copy counts as lost: write-done fails the request and returns its slot.
    mc = _mooncake()
    dest = _Dest()
    s0 = dest.recv.advertise("stage0", 1)["slots"]
    dest.recv.begin("r", 5, [1], {0: "stage0"})
    rows = torch.zeros((4, TOKEN_BYTES), dtype=torch.uint8)
    ready = _ready("r", s0[0], 0, _land(dest, s0[0], 0, 0, rows))
    no_request_id = dict(ready)
    del no_request_id["request_id"]
    lost = _ready("r", s0[1], 1, [[0, 0, 4, 0]])
    payloads = [
        b"\xc1",  # not msgpack
        msgpack.dumps([ready]),  # not a dict
        msgpack.dumps(no_request_id),
        msgpack.dumps({**ready, "slot": "x"}),
        msgpack.dumps(ready),
        msgpack.dumps({**lost, "seq": "x"}),  # its only copy
    ]
    _listen(
        monkeypatch,
        mc,
        _consumer(mc, dest)._notification_listener,
        [(MSG_LANDING_READY, payload) for payload in payloads],
    )
    assert dest.recv._requests["r"].pending == 1  # the good READY was queued
    dest.pump()
    assert dest.credits() == [("stage0", [s0[0]])]
    dest.recv.stage_done("r", 0, 5, True, [s0[0], s0[1]])
    assert dest.finished == [("r", True)]
    assert dest.credits() == [("stage0", [s0[1]])]


def test_an_unreadable_write_done_does_not_stop_the_listener(monkeypatch):
    # Every later receive on the rank needs the listener.
    mc = _mooncake()
    dest = _Dest()
    dest.recv.advertise("stage0", 1)
    dest.recv.begin("r", 5, [1], {0: "stage0"})
    done = {"request_id": "r", "pp_rank": 0, "write_nonce": 5, "landed_slots": []}
    no_request_id = dict(done)
    del no_request_id["request_id"]
    payloads = [
        b"\xc1",  # not msgpack
        msgpack.dumps([done]),  # not a dict
        msgpack.dumps(no_request_id),
        msgpack.dumps(done),
    ]
    _listen(
        monkeypatch,
        mc,
        _consumer(mc, dest)._notification_listener,
        [(mc.MSG_WRITE_DONE, payload) for payload in payloads],
    )
    assert dest.finished == [("r", False)]


def test_unreadable_landing_fields_do_not_stop_the_write_listener(monkeypatch):
    # Write requests arrive on the same thread as the credits.
    mc = _mooncake()
    conn = object.__new__(mc.MooncakeConnector)
    conn._side_channel_port = 1
    conn._landing_credits = LandingCredits()
    conn._landing_credits.sync("d0", {"epoch": 1, "slots": [0]})
    assert conn._landing_credits.acquire("d0", 1, 0) == 0
    submitted = []

    class _Executor:
        def submit(self, _fn, request_data):
            submitted.append(request_data)

    conn._send_executor = _Executor()
    request = {
        "request_id": "r",
        "consumer_host": "d",
        "consumer_rpc_port": 1,
        "mla_landing": [1],  # not a partition
    }
    credit = {"consumer": "d0", "epoch": 1, "slots": [0]}
    _listen(
        monkeypatch,
        mc,
        conn._write_listener,
        [
            (mc.MSG_WRITE_REQUEST, msgpack.dumps(request)),
            (MSG_LANDING_CREDIT, b"\xc1"),  # not msgpack
            (MSG_LANDING_CREDIT, msgpack.dumps({**credit, "slots": [[0]]})),
            (MSG_LANDING_CREDIT, msgpack.dumps(credit)),
        ],
    )
    # The transfer still reaches a send worker, which reports its outcome.
    assert submitted == [request]
    assert conn._landing_credits.acquire("d0", 1, 0) == 0


# ---------------------------------------------------------------------------
# GPU scatter kernel
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gpu_scatter_matches_rows_across_allocations_and_slot_reuse():
    from atom.kv_transfer.disaggregation import landing_scatter as ls

    dev = torch.device("cuda", 0)
    arena = torch.zeros((3, 64 * PAGE_BYTES), dtype=torch.uint8, device=dev)
    extra = torch.zeros(64 * PAGE_BYTES, dtype=torch.uint8, device=dev)
    regions = [arena[0], arena[1], arena[2], extra]
    bases = np.array([r.data_ptr() for r in regions], dtype=np.int64)
    origin = int(bases.min())
    landing = torch.zeros((2, 1 << 20), dtype=torch.uint8, device=dev)
    table = torch.zeros((3, 4096), dtype=torch.int64, device=dev)
    host = np.zeros((3, 4096), dtype=np.int64)
    rng = np.random.default_rng(1)
    dst_blocks = rng.permutation(64)[:10]
    rows_per_region = 150  # partial last page
    for rep in range(6):  # slot reuse with no torch read in between
        slot = rep % 2
        src_rows = torch.randint(
            0, 256, (4, rows_per_region, TOKEN_BYTES), dtype=torch.uint8, device=dev
        )
        landing[slot, : src_rows.numel()].copy_(src_rows.reshape(-1))
        region = np.arange(4)
        src, dst, nbytes = ls.landing_segments(
            np.zeros(4, dtype=np.int64),
            np.full(4, rows_per_region),
            slot * (1 << 20) + region * rows_per_region * TOKEN_BYTES,
            bases[region],
            np.full(4, PAGE_BYTES),
            dst_blocks,
            BLOCK_SIZE,
            origin,
        )
        n = ls.segment_table(src, dst, nbytes, host)
        table[:, :n].copy_(torch.from_numpy(host[:, :n]))
        ls.scatter_segments(landing, ls.DevicePointer(origin, dev), table, n)
        torch.cuda.synchronize()
        j = np.arange(rows_per_region)
        tokens = torch.as_tensor(
            dst_blocks[j // BLOCK_SIZE] * BLOCK_SIZE + j % BLOCK_SIZE, device=dev
        )
        for r in range(4):
            got = regions[r].view(-1, TOKEN_BYTES)[tokens]
            assert torch.equal(got, src_rows[r]), (rep, r)


# ---------------------------------------------------------------------------
# KV budget reserve
# ---------------------------------------------------------------------------


def _config(transfer, dcp=4, kv_lora_rank=512):
    from types import SimpleNamespace

    return SimpleNamespace(
        kv_transfer_config=transfer,
        decode_context_parallel_size=dcp,
        hf_config=SimpleNamespace(kv_lora_rank=kv_lora_rank),
    )


def test_landing_reserve_matches_the_pool_and_its_gates(monkeypatch):
    consumer = {"kv_connector": "mooncake", "kv_role": "kv_consumer"}
    producer = {"kv_connector": "mooncake", "kv_role": "kv_producer"}
    monkeypatch.delenv("ATOM_PD_MLA_LANDING", raising=False)
    # On by default: the whole pool, 32 slots of 8 MiB.
    assert mla_landing_reserve_bytes(_config(consumer)) == 256 << 20
    assert mla_landing_reserve_bytes(_config(producer)) == 0
    assert mla_landing_reserve_bytes(_config(consumer, dcp=1)) == 0
    assert mla_landing_reserve_bytes(_config(consumer, kv_lora_rank=None)) == 0
    monkeypatch.setenv("ATOM_PD_MLA_LANDING", "0")
    assert mla_landing_reserve_bytes(_config(consumer)) == 0


def test_mooncake_answers_the_runner_kv_budget_reserve(monkeypatch):
    """The runner asks the factory; Mooncake answers with its two MLA pools."""
    _mooncake()  # the factory imports the worker class
    from atom.kv_transfer.disaggregation import KVConnectorFactory

    def reserve(transfer, **overrides):
        return KVConnectorFactory.kv_budget_reserve_bytes(
            _config(transfer, **overrides)
        )

    producer = {
        "kv_connector": "mooncake",
        "kv_role": "kv_producer",
        "num_worker_threads": 16,
    }
    consumer = {"kv_connector": "mooncake", "kv_role": "kv_consumer"}
    offload = {"kv_connector": "lmcache_mp", "kv_role": "offload"}
    monkeypatch.delenv("ATOM_PD_MLA_STAGING", raising=False)
    monkeypatch.delenv("ATOM_PD_MLA_LANDING", raising=False)
    # A prefill without DCP stages: one 8 MiB slot per send worker.
    assert reserve(producer, dcp=1) == 128 << 20
    # A DCP decode rank lands into its whole pool; the offload sub adds nothing.
    assert reserve(consumer) == 256 << 20
    assert reserve({"kv_connector": "multi", "connectors": [consumer, offload]}) == (
        256 << 20
    )
    monkeypatch.setenv("ATOM_PD_MLA_LANDING", "0")
    assert reserve(consumer) == 0

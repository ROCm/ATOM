# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM_MORIIO_INCREMENTAL_PUSH: P/D over the fabric, write mode, pushing only
the prompt blocks past the decode's own prefix-cache hit.

- D's lookup: the scheduler reports its local hit as ``num_computed_blocks``
  (flag on, no per-request state), and the received suffix is published so a
  later turn hits through it.
- Placement, byte for byte: D asks through `_request_write`, P serves the
  WRITE_REQ through `_write_blocks` on CPU pools; D's held blocks keep their
  bytes, every other destination block is P's source block, each byte once.
- ATOM_MORIIO_VERIFY: the pushed part is compared with the producer; each held
  block with what its own earlier push landed.

CPU only, on the aiter stand-ins.
"""

from __future__ import annotations

import threading
import types
from unittest import mock

import msgpack
import pytest
import torch
from aiter_stub import atom_under_stubs

# P's source blocks and D's destination blocks, paired by position.
SRC = [5, 6, 2, 0, 8]
DST = [3, 4, 9, 1, 7]
P_BLOCKS, D_BLOCKS = 10, 12
UNITS = (16, 4, 8)  # bytes per block of each PAGE region


@pytest.fixture(scope="module")
def stack():
    with atom_under_stubs() as load:
        yield types.SimpleNamespace(
            conn=load("atom.kv_transfer.disaggregation.moriio.moriio_connector"),
            fabric=load("atom.kv_transfer.disaggregation.moriio.fabric"),
            types=load("atom.kv_transfer.disaggregation.types"),
            bm=load("atom.model_engine.block_manager"),
        )


def side(blocks, seed):
    """A pool of random bytes holding one region per `UNITS`, back to back."""
    g = torch.Generator().manual_seed(seed)
    pool = torch.randint(0, 256, (blocks * sum(UNITS),), dtype=torch.uint8, generator=g)
    regions, views, off = [], [], 0
    for i, unit in enumerate(UNITS):
        regions.append((off, unit, f"region{i}"))
        views.append(pool[off : off + unit * blocks].view(blocks, unit))
        off += unit * blocks
    return types.SimpleNamespace(pool=pool, regions=regions, views=views)


def producer(stack, p, d, verify=False):
    conn = object.__new__(stack.conn.MoRIIOConnector)
    conn.use_fabric = True
    conn._verify_kv = verify
    conn._page_regions = p.regions
    conn._page_views = p.views
    conn.posted = []

    def write(sizes, l_offs, r_offs, session):
        conn.posted.append((list(l_offs), list(r_offs), list(sizes)))
        return "status"

    conn.moriio_wrapper = types.SimpleNamespace(write_remote_data_status=write)
    conn._get_or_build_page_session = lambda engine_id: (
        "session",
        [(o, u) for o, u, _ in d.regions],
    )
    return conn


def consumer(stack, d, verify=False):
    conn = object.__new__(stack.conn.MoRIIOConnector)
    conn.use_fabric = True
    conn._verify_kv = verify
    conn._page_regions = d.regions
    conn._page_views = d.views
    conn._regions = None
    conn._state_tensors = {}
    conn.tp_rank, conn.tp_size, conn.dp_rank = 0, 1, 0
    conn.local_ip = "10.0.0.2"
    conn.base_handshake_port = 7400
    conn.request_id_to_transfer_id = {"d": 9}
    conn._write_timeout_s = 600.0
    conn._awaiting_write = {}
    conn._failed_before_read = set()
    conn._verify_page_dst = {}
    conn._write_page_digests = {}
    conn._write_reports_lock = threading.Lock()
    conn._verify_block_sums = {}
    conn._verify_held = {}
    conn.sent = []
    conn.moriio_wrapper = types.SimpleNamespace(
        send_message=lambda frames, host, port: conn.sent.append(frames),
        send_notify=lambda *args: None,
    )
    return conn


def recv_meta(k, not_received=(), src=SRC):
    return types.SimpleNamespace(
        num_computed_blocks=k,
        verify_held_not_received=list(not_received),
        src_block_skip_factor=1,
        local_block_ids=list(DST),
        remote_block_ids=list(src),
        remote_host="10.0.0.1",
        remote_handshake_port=7301,
        remote_dp_rank=0,
        tp_size=1,
        transfer_id=9,
    )


def ask_and_push(stack, p, d, k, d_conn, not_received=()):
    """D asks with ``k`` blocks held, P serves the WRITE_REQ into ``d``'s pool.
    Returns the request and the writes per destination byte."""
    d_conn.sent = []
    d_conn._request_write("d", recv_meta(k, not_received))
    [frames] = d_conn.sent
    request = msgpack.loads(frames[1])
    p_conn = producer(stack, p, d, verify=d_conn._verify_kv)
    p_conn._write_blocks(request, "decode")
    written = torch.zeros(d.pool.numel(), dtype=torch.int32)
    for l_offs, r_offs, sizes in p_conn.posted:
        for lo, ro, n in zip(l_offs, r_offs, sizes):
            d.pool[ro : ro + n] = p.pool[lo : lo + n]
            written[ro : ro + n] += 1
    if d_conn._verify_kv:
        d_conn._write_page_digests["d"] = request["page_digest"]
    return request, written


def block_mask(s, blocks):
    mask = torch.zeros(s.pool.numel(), dtype=torch.bool)
    for off, unit, _ in s.regions:
        for b in blocks:
            mask[off + b * unit : off + (b + 1) * unit] = True
    return mask


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [0, 3, len(DST)])
def test_the_partial_push_lands_byte_exact(stack, k):
    pushed_from = min(k, len(DST) - 1)  # a hit never covers the last block
    p, d = side(P_BLOCKS, 1), side(D_BLOCKS, 2)
    before = d.pool.clone()
    request, written = ask_and_push(stack, p, d, k, consumer(stack, d))
    assert request["dst_block_ids"] == DST[pushed_from:]
    assert request["src_block_ids"] == SRC[pushed_from:]
    assert request["skipped_blocks"] == pushed_from
    held = block_mask(d, DST[:pushed_from])
    pushed = block_mask(d, DST[pushed_from:])
    # D's own prefix blocks: untouched. The rest: P's source, each byte once.
    assert torch.equal(d.pool[held], before[held])
    assert torch.equal(written, pushed.to(torch.int32))
    for dv, pv in zip(d.views, p.views):
        for s, t in zip(SRC[pushed_from:], DST[pushed_from:]):
            assert torch.equal(dv[t], pv[s])
    rest = ~(held | pushed)
    assert torch.equal(d.pool[rest], before[rest])


# ---------------------------------------------------------------------------
# ATOM_MORIIO_VERIFY
# ---------------------------------------------------------------------------


def verify_lines(stack, conn, caplog):
    """`_verify_page_landed`'s lines: (pushed part, held prefix)."""
    caplog.clear()
    # On a mismatch the check reads again after a device sync.
    with (
        caplog.at_level("INFO", logger="atom"),
        mock.patch("torch.cuda.synchronize", lambda: None),
        mock.patch.object(stack.conn.time, "sleep", lambda s: None),
    ):
        conn._verify_page_landed("d")
    lines = [r.getMessage() for r in caplog.records if "moriio-verify" in r.msg]
    held = [line for line in lines if "locally held" in line]
    return [line for line in lines if line not in held], held


def test_page_block_digests_are_each_blocks_page_digest(stack):
    views = side(6, 3).views
    blocks = [4, 0, 5]
    assert stack.fabric.page_block_digests(views, blocks) == [
        stack.fabric.page_digest(views, [b]) for b in blocks
    ]
    assert stack.fabric.page_block_digests(views, []) == []


def test_verify_checks_the_pushed_part_and_records_only_a_match(stack, caplog):
    p, d = side(P_BLOCKS, 1), side(D_BLOCKS, 2)
    conn = consumer(stack, d, verify=True)
    ask_and_push(stack, p, d, 3, conn)
    assert conn._verify_page_dst["d"] == DST[3:]
    d.views[1][DST[3]][2] ^= 1  # a pushed block, not the last
    pushed, _ = verify_lines(stack, conn, caplog)
    assert "differ from the producer" in pushed[0], pushed
    assert not conn._verify_block_sums


def push_then_hold(stack, caplog, not_received=()):
    """Turn 1 pushes every block (recorded once it matches), turn 2 holds the
    first three: D's pool, and the asker of turn 2 before its landing is
    verified."""
    p, d = side(P_BLOCKS, 1), side(D_BLOCKS, 2)
    conn = consumer(stack, d, verify=True)
    ask_and_push(stack, p, d, 0, conn)
    pushed, held = verify_lines(stack, conn, caplog)
    assert "of 5 blocks match the producer" in pushed[0] and not held, pushed
    # All but the last block: past the prompt decode writes its slots.
    assert sorted(conn._verify_block_sums) == sorted(DST[:-1])
    request, _ = ask_and_push(stack, p, d, 3, conn, not_received)
    assert conn._verify_held["d"] == (DST[:3], set(not_received))
    # The producer is not asked about the held blocks.
    assert not any("held" in key for key in request)
    return d, conn


def test_verify_holds_each_held_block_to_its_own_push(stack, caplog):
    _d, conn = push_then_hold(stack, caplog)
    pushed, held = verify_lines(stack, conn, caplog)
    assert "of 2 blocks match the producer" in pushed[0], pushed
    assert held == [
        (
            "[moriio-verify] req d: 3 of the 3 locally held prefix blocks hold "
            "what their push landed (0 completed by this decode, 0 with no "
            "matching push recorded: not compared)"
        )
    ], held


@pytest.mark.parametrize("region", [0, 1, 2])
@pytest.mark.parametrize("block", [0, 2])  # the held body, the last held block
def test_verify_reports_a_held_block_changed_since_its_push(
    stack, region, block, caplog
):
    d, conn = push_then_hold(stack, caplog)
    d.views[region][DST[block]][3] ^= 1
    pushed, held = verify_lines(stack, conn, caplog)
    assert "of 2 blocks match the producer" in pushed[0], pushed
    assert held == [
        (
            f"[moriio-verify] req d: 1 of the 3 locally held prefix blocks "
            f"changed since their push landed (positions [{block}]) -- decode "
            "will attend over a prefix-cache hit that is not the prompt's"
        )
    ], held


def test_verify_counts_a_held_block_this_decode_completed(stack, caplog):
    """A held block D's prefix cache did not publish from a push (its own
    decode filled it) has nothing to be compared with."""
    d, conn = push_then_hold(stack, caplog, not_received=[1])
    d.views[0][DST[1]][3] ^= 1
    _, held = verify_lines(stack, conn, caplog)
    assert held == [
        (
            "[moriio-verify] req d: 2 of the 3 locally held prefix blocks hold "
            "what their push landed (1 completed by this decode, 0 with no "
            "matching push recorded: not compared)"
        )
    ], held


# ---------------------------------------------------------------------------
# D's lookup: the hit it reports, and the suffix it publishes
# ---------------------------------------------------------------------------


def consumer_scheduler(stack, hash_block_size=128):
    sched = object.__new__(stack.conn.MoRIIOConnectorScheduler)
    sched.__dict__.update(
        is_producer=False,
        hash_block_size=hash_block_size,
        _reqs_need_recv={},
        request_id_to_transfer_id={},
        transfer_id_to_request_id={},
        _block_manager=None,
    )
    return sched


def flag(stack, name, on):
    return mock.patch.object(stack.conn.envs, name, on)


@pytest.mark.parametrize(
    "on, cached, per_req, expected",
    [
        (True, 5 * 128 + 17, False, 5),  # a partial hash block is not held
        (True, 0, False, 0),  # no hit: the full push
        (False, 5 * 128, False, 0),  # flag off: unchanged
        (True, 5 * 128, True, 0),  # per-request state: the full push
    ],
)
def test_d_reports_its_local_hit(stack, on, cached, per_req, expected):
    sched = consumer_scheduler(stack)
    seq = types.SimpleNamespace(
        id=11,
        block_table=list(range(20, 28)),
        state_slot=-1,
        num_cached_tokens=cached,
        has_per_req_cache=per_req,
        kv_transfer_params={"do_remote_prefill": True, "transfer_id": 11},
    )
    with flag(stack, "ATOM_MORIIO_INCREMENTAL_PUSH", on):
        sched.update_state_after_alloc(seq)
    assert seq.kv_transfer_params["num_computed_blocks"] == expected
    [meta] = sched.build_connector_meta().reqs_to_recv.values()
    assert meta.num_computed_blocks == expected


def test_the_received_suffix_is_published_for_the_next_turn(stack):
    """Turn 2 holds turn 1's blocks; what it receives past them is published,
    so turn 3 holds all of turn 2's full blocks and asks for less again."""
    from conftest import MockConfig

    from atom.model_engine.sequence import Sequence
    from atom.sampling_params import SamplingParams

    bm = stack.bm.BlockManager(
        MockConfig(enable_prefix_caching=True, num_kvcache_blocks=32)
    )
    sched = consumer_scheduler(stack, hash_block_size=bm.hash_block_size)
    sched.bind_block_manager(bm)

    def receive(tokens):
        seq = Sequence(tokens, 4, sampling_params=SamplingParams())
        seq.kv_transfer_params = {"do_remote_prefill": True, "transfer_id": seq.id}
        bm.allocate(seq, bm.can_allocate(seq))
        with (
            flag(stack, "ATOM_MORIIO_INCREMENTAL_PUSH", True),
            flag(stack, "ATOM_MORIIO_VERIFY", True),
        ):
            sched.update_state_after_alloc(seq)
        held = seq.kv_transfer_params["num_computed_blocks"]
        bm.register_received_prefix(seq)  # the push landed
        return seq, held

    turn1, held1 = receive(list(range(1, 11)))  # 2 full blocks + 2 tokens
    assert [bm.kv.block(b).received for b in turn1.block_table] == [1, 1, 0]
    turn2, held2 = receive(list(range(1, 19)))  # turn 1's prompt grown
    turn3, held3 = receive(list(range(1, 23)))
    assert (held1, held2, held3) == (0, 2, 4)
    assert turn2.block_table[:2] == turn1.block_table[:2]
    assert turn3.block_table[:4] == turn2.block_table[:4]
    # ATOM_MORIIO_VERIFY: every held block came from a push.
    assert turn3.kv_transfer_params["verify_held_not_received"] == []

    # Turn 2's last full block published by its own decode instead (here, by
    # hand): turn 4 holds it, and says it was not pushed.
    block = bm.kv.block(turn2.block_table[3])
    bm.kv.publish(block.block_id, block.hash, block.token_ids)
    assert not block.received
    turn4, held4 = receive(list(range(1, 23)))
    assert held4 == 5
    assert turn4.kv_transfer_params["verify_held_not_received"] == [3]
    meta = stack.types.ConnectorMetadata()
    meta.add_new_req_to_recv(
        request_id=turn4.id,
        local_block_ids=turn4.block_table,
        kv_transfer_params=turn4.kv_transfer_params,
    )
    assert meta.reqs_to_recv[turn4.id].verify_held_not_received == [3]


# ---------------------------------------------------------------------------
# Refusals and the read path
# ---------------------------------------------------------------------------


def test_a_source_stride_is_still_refused_before_any_ask(stack):
    conn = consumer(stack, side(D_BLOCKS, 2))
    meta = recv_meta(2)
    meta.src_block_skip_factor = 2
    conn._request_write("d", meta)
    assert conn._failed_before_read == {"d"} and not conn.sent


def test_a_short_source_is_still_refused_with_a_hit(stack):
    conn = consumer(stack, side(D_BLOCKS, 2))
    conn._request_write("d", recv_meta(2, src=SRC[:3]))
    assert conn._failed_before_read == {"d"} and not conn.sent


def test_a_read_still_moves_every_block(stack, caplog):
    meta = types.SimpleNamespace(num_computed_blocks=3, src_block_skip_factor=1)
    reject = stack.conn.MoRIIOConnector._rejects_transfer_subset
    with caplog.at_level("WARNING", logger="atom"):
        assert not reject("r", meta)
    assert "transferring all blocks" in caplog.text
    caplog.clear()
    with caplog.at_level("WARNING", logger="atom"):
        assert not reject("r", meta, skips_computed=True)
    assert caplog.text == ""

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM_MORIIO_INCREMENTAL_PUSH: MiniMax-M3.1 P/D over the fabric, write mode,
pushing only the prompt blocks past the decode's own prefix-cache hit.

- D's lookup: the scheduler reports its local hit as ``num_computed_blocks``
  (flag on, no per-request state), and the received suffix is published so a
  later turn hits through it.
- Placement, byte for byte, TP4 -> TP4 and PP4 x TP1 -> TP4: D asks through
  `_request_write`, each producer serves the WRITE_REQ it gets through
  `_write_blocks` on the real M3.1 pools; D's held blocks keep their bytes,
  every other destination block is what the full push writes there (and, at
  equal TP, P's source block), nothing else is touched.
- ATOM_MORIIO_VERIFY compares the pushed part with what P pushed, and D's
  held prefix blocks with P's source blocks at the same positions.
- No hit, or the flag off: the full push. Refusals still refuse.

CPU only, on the aiter stand-ins, like `test_moriio_pp_fabric.py`.
"""

from __future__ import annotations

import types
from unittest import mock

import msgpack
import pytest
import test_moriio_pp_fabric as pp_fabric
import torch
from test_moriio_fabric import M31_LAYERS, handshake
from test_moriio_pp_fabric import (
    D_TP,
    PARTITION,
    decode_connector,
    engine,
    fill,
    rank_pools,
    stage_connector,
    stage_layers,
)

from atom.model_engine.block_manager import BlockManager

# The stubbed ATOM stack and the write-path connectors, as that module's.
stack = pp_fabric.stack
write_path = pp_fabric.write_path

# P's source blocks and D's destination blocks, paired by position.
SRC = [5, 6, 2, 0, 8]
DST = [3, 4, 9, 1, 7]
P_BLOCKS, D_BLOCKS = 10, 12


@pytest.fixture(autouse=True)
def _device():
    with mock.patch("torch.cuda.current_device", lambda: 0):
        yield


@pytest.fixture(scope="module")
def pools(stack):
    """Random KV on P (four PP4 x TP1 stages and four TP4 ranks) and D (TP4)."""
    with mock.patch("torch.cuda.current_device", lambda: 0):
        stages = [
            rank_pools(stack, stage_layers(s), 1, s == 3, P_BLOCKS, "kv_producer")
            for s in range(len(PARTITION))
        ]
        p_ranks = [
            rank_pools(stack, range(M31_LAYERS), D_TP, True, P_BLOCKS, "kv_producer")
            for _ in range(D_TP)
        ]
        d_ranks = [
            rank_pools(stack, range(M31_LAYERS), D_TP, True, D_BLOCKS, "kv_consumer")
            for _ in range(D_TP)
        ]
    for i, side in enumerate(stages + p_ranks + d_ranks):
        fill(side.pool, 100 + i)
    return types.SimpleNamespace(stages=stages, p_ranks=p_ranks, d_ranks=d_ranks)


def producers_for(stack, pools, topology, rank):
    """``(connector, side)`` of every producer that pushes into D rank ``rank``."""
    if topology == "pp4":
        return [
            (stage_connector(stack, s, side), side)
            for s, side in enumerate(pools.stages)
        ]
    side = pools.p_ranks[rank]
    conn = stage_connector(stack, 0, side, partition=(M31_LAYERS,), tp=D_TP)
    conn.tp_rank = rank
    return [(conn, side)]


def asker(write_path, rank, verify_side=None):
    """D rank ``rank``'s write-path connector (it asks and verifies)."""
    conn = write_path.make(producer=False)
    conn.tp_size, conn.tp_rank = D_TP, rank
    conn.use_fabric = True
    conn.request_id_to_transfer_id = {"d": 9}
    if verify_side is not None:
        conn._verify_kv = True
        conn._stage_digests = {}
        conn._page_views = [p.view for p in verify_side.tt.pages]
        conn._page_regions = [
            (r.base_addr, r.unit_bytes, f"region{i}", r.shard_rank, r.shard_world)
            for i, r in enumerate(verify_side.tt.block_regions)
        ]
    return conn


def ask_and_push(stack, write_path, pools, topology, rank, k, dst_pool, verify=False):
    """D rank ``rank`` asks with ``k`` blocks held; every producer serves its
    WRITE_REQ into ``dst_pool``. Returns the asker, the requests, the bytes
    written per destination byte."""
    d_side = pools.d_ranks[rank]
    producers = producers_for(stack, pools, topology, rank)
    conn = asker(write_path, rank, d_side if verify else None)
    meta = write_path.recv_meta(DST, SRC, remote_pp_size=len(producers))
    meta.tp_size = 1 if topology == "pp4" else D_TP
    meta.num_computed_blocks = k
    conn._request_write("d", meta)
    requests = [msgpack.loads(frames[1]) for frames, _host, _port in conn.sent]
    assert len(requests) == len(producers)
    _consumer, d_meta = decode_connector(stack, d_side, rank)
    written = torch.zeros(dst_pool.numel(), dtype=torch.int32)
    for stage, ((p_conn, p_side), request) in enumerate(zip(producers, requests)):
        p_conn._verify_kv = verify
        if verify:
            p_conn._page_views = [p.view for p in p_side.tt.pages]
        handshake(stack, p_conn, d_meta)
        p_conn.remote_moriio_metadata[engine(rank)] = p_conn.remote_moriio_metadata.pop(
            "decode"
        )
        p_conn.posted = []
        p_conn._write_blocks(request, engine(rank))
        assert len(p_conn.posted) == 1
        for _session, l_offs, r_offs, sizes in p_conn.posted:
            for lo, ro, n in zip(l_offs, r_offs, sizes):
                dst_pool[ro : ro + n] = p_side.pool[lo : lo + n]
                written[ro : ro + n] += 1
        if verify:
            conn._record_write_report(
                "d",
                True,
                None,
                request["page_digest"],
                stage=stage,
                held_digest=request.get("held_digest"),
            )
    return conn, requests, written


def block_mask(side, blocks):
    """Every byte of ``blocks`` in each PAGE region of ``side``'s pool."""
    mask = torch.zeros(side.pool.numel(), dtype=torch.bool)
    base = side.pool.data_ptr()
    for region in side.tt.block_regions:
        off = region.base_addr - base
        for b in blocks:
            lo = off + b * region.unit_bytes
            mask[lo : lo + region.unit_bytes] = True
    return mask


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("topology", ["tp4", "pp4"])
@pytest.mark.parametrize("k", [0, 3, len(DST)])
def test_the_partial_push_lands_byte_exact(stack, write_path, pools, topology, k):
    pushed_from = min(k, len(DST) - 1)  # a hit never covers the last block
    for r, d_side in enumerate(pools.d_ranks):
        before = d_side.pool.clone()
        full = before.clone()
        ask_and_push(stack, write_path, pools, topology, r, 0, full)
        incr = before.clone()
        _, requests, written = ask_and_push(
            stack, write_path, pools, topology, r, k, incr
        )
        for request in requests:
            assert request["dst_block_ids"] == DST[pushed_from:]
            assert request["src_block_ids"] == SRC[pushed_from:]
            assert request["skipped_blocks"] == pushed_from
        held = block_mask(d_side, DST[:pushed_from])
        pushed = block_mask(d_side, DST[pushed_from:])
        # D's own prefix blocks: untouched, though the full push rewrites them.
        assert torch.equal(incr[held], before[held])
        if pushed_from:
            assert not torch.equal(full[held], before[held])
        # The missing blocks: exactly what the full push writes, each byte once.
        assert torch.equal(incr[pushed], full[pushed])
        assert torch.equal(written, pushed.to(torch.int32))
        rest = ~(held | pushed)
        assert torch.equal(incr[rest], before[rest])
        if topology == "tp4":
            # Equal TP pairs regions by position: P's source block itself.
            p_side = pools.p_ranks[r]
            d_base, p_base = d_side.pool.data_ptr(), p_side.pool.data_ptr()
            for d_reg, p_reg in zip(
                d_side.tt.block_regions, p_side.tt.block_regions, strict=True
            ):
                n = d_reg.unit_bytes
                for s, d in zip(SRC[pushed_from:], DST[pushed_from:]):
                    lo = d_reg.base_addr - d_base + d * n
                    plo = p_reg.base_addr - p_base + s * n
                    assert torch.equal(incr[lo : lo + n], p_side.pool[plo : plo + n])


def verify_lines(stack, conn, caplog):
    """`_verify_page_landed`'s lines: (pushed part, held prefix)."""
    caplog.clear()
    # On a mismatch the equal-TP check reads again after a device sync.
    with (
        caplog.at_level("INFO", logger="atom"),
        mock.patch("torch.cuda.synchronize", lambda: None),
        mock.patch.object(stack.conn.time, "sleep", lambda s: None),
    ):
        conn._verify_page_landed("d")
    lines = [r.getMessage() for r in caplog.records if "moriio-verify" in r.msg]
    held = [line for line in lines if "locally held" in line]
    return [line for line in lines if line not in held], held


@pytest.mark.parametrize("topology", ["tp4", "pp4"])
def test_verify_digests_the_pushed_part_and_the_held_prefix(
    stack, write_path, pools, topology, caplog
):
    rank, d_side = 1, pools.d_ranks[1]
    original = d_side.pool.clone()
    try:
        # A full push first: D's prefix blocks hold the prompt's KV, as its
        # own prefix cache does after an earlier turn.
        ask_and_push(stack, write_path, pools, topology, rank, 0, d_side.pool)
        conn, requests, _ = ask_and_push(
            stack, write_path, pools, topology, rank, 3, d_side.pool, verify=True
        )
        assert conn._verify_page_dst["d"] == DST[3:]
        assert conn._verify_held_dst["d"] == DST[:3]
        # Every producer stage digests the held blocks' source.
        assert all(r["verify_held_src_block_ids"] == SRC[:3] for r in requests)
        pushed, held = verify_lines(stack, conn, caplog)
        assert "of 2 blocks match the producer" in pushed[0], pushed
        assert (
            len(held) == 1
            and "3 locally held prefix blocks match the producer" in held[0]
        ), held

        conn, _, _ = ask_and_push(
            stack, write_path, pools, topology, rank, 3, d_side.pool, verify=True
        )
        d_side.tt.pages[7].view[DST[3]].view(-1)[5] ^= 1  # a pushed block
        pushed, held = verify_lines(stack, conn, caplog)
        assert "differ from the producer" in pushed[0], pushed
        assert "match the producer" in held[0], held
    finally:
        d_side.pool.copy_(original)


@pytest.mark.parametrize("topology", ["tp4", "pp4"])
@pytest.mark.parametrize("region", [0, 7, -1])  # first, a middle one, the last
@pytest.mark.parametrize("block", [0, 2])  # the held body, the last held block
def test_verify_reports_a_corrupt_held_prefix_block(
    stack, write_path, pools, topology, region, block, caplog
):
    """A byte of D's prefix-cache hit that is not P's: decode would attend over
    it, though the push never touched it."""
    rank, d_side = 2, pools.d_ranks[2]
    original = d_side.pool.clone()
    try:
        ask_and_push(stack, write_path, pools, topology, rank, 0, d_side.pool)
        conn, _, _ = ask_and_push(
            stack, write_path, pools, topology, rank, 3, d_side.pool, verify=True
        )
        d_side.tt.pages[region].view[DST[block]].view(-1)[3] ^= 1
        pushed, held = verify_lines(stack, conn, caplog)
        assert "of 2 blocks match the producer" in pushed[0], pushed
        assert len(held) == 1, held
        assert (
            "1 of " in held[0]
            and "3 locally held prefix blocks differ from the producer" in held[0]
        ), held
    finally:
        d_side.pool.copy_(original)


def test_verify_names_held_regions_no_stage_digested(stack, write_path, pools, caplog):
    """PP4: a stage whose report carries no held digest leaves its layers'
    held regions unchecked, which is reported, not taken as a match."""
    rank, d_side = 0, pools.d_ranks[0]
    original = d_side.pool.clone()
    try:
        ask_and_push(stack, write_path, pools, "pp4", rank, 0, d_side.pool)
        real = stack.conn.MoRIIOConnector._held_source
        calls = []

        def stage_2_digests_nothing(self, request):
            calls.append(self.pp_rank)
            return [] if self.pp_rank == 2 else real(self, request)

        with mock.patch.object(
            stack.conn.MoRIIOConnector, "_held_source", stage_2_digests_nothing
        ):
            conn, _, _ = ask_and_push(
                stack, write_path, pools, "pp4", rank, 3, d_side.pool, verify=True
            )
        assert sorted(calls) == [0, 1, 2, 3]
        _, held = verify_lines(stack, conn, caplog)
        assert "written by none of the 4 producer stages" in held[0], held
    finally:
        d_side.pool.copy_(original)


def test_without_verify_no_held_digest_is_asked_or_made(stack, write_path, pools):
    """The held check costs nothing with ATOM_MORIIO_VERIFY off."""
    d_side = pools.d_ranks[0]
    scratch = d_side.pool.clone()
    conn, requests, _ = ask_and_push(stack, write_path, pools, "pp4", 0, 3, scratch)
    assert not conn._verify_held_dst
    assert all("verify_held_src_block_ids" not in r for r in requests)
    assert all("held_digest" not in r for r in requests)


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
    )
    return sched


def remote_seq(num_cached_tokens, blocks=8, per_req=False):
    return types.SimpleNamespace(
        id=11,
        block_table=list(range(20, 20 + blocks)),
        state_slot=-1,
        num_cached_tokens=num_cached_tokens,
        has_per_req_cache=per_req,
        kv_transfer_params={"do_remote_prefill": True, "transfer_id": 11},
    )


def flag(stack, on):
    return mock.patch.object(stack.conn.envs, "ATOM_MORIIO_INCREMENTAL_PUSH", on)


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
    seq = remote_seq(cached, per_req=per_req)
    with flag(stack, on):
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

    bm = BlockManager(MockConfig(enable_prefix_caching=True, num_kvcache_blocks=32))
    sched = consumer_scheduler(stack, hash_block_size=bm.hash_block_size)

    def receive(tokens):
        seq = Sequence(tokens, 4, sampling_params=SamplingParams())
        seq.kv_transfer_params = {"do_remote_prefill": True, "transfer_id": seq.id}
        bm.allocate(seq, bm.can_allocate(seq))
        with flag(stack, True):
            sched.update_state_after_alloc(seq)
        held = seq.kv_transfer_params["num_computed_blocks"]
        bm.register_received_prefix(seq)  # the push landed
        return seq, held

    turn1, held1 = receive(list(range(1, 11)))  # 2 full blocks + 2 tokens
    turn2, held2 = receive(list(range(1, 19)))  # turn 1's prompt grown
    turn3, held3 = receive(list(range(1, 23)))
    assert (held1, held2, held3) == (0, 2, 4)
    assert turn2.block_table[:2] == turn1.block_table[:2]
    assert turn3.block_table[:4] == turn2.block_table[:4]


# ---------------------------------------------------------------------------
# Refusals and the read path
# ---------------------------------------------------------------------------


def test_a_source_stride_is_still_refused_before_any_ask(write_path):
    conn = asker(write_path, 0)
    meta = write_path.recv_meta(DST, SRC)
    meta.num_computed_blocks, meta.src_block_skip_factor = 2, 2
    conn._request_write("d", meta)
    assert conn._failed_before_read == {"d"} and not any(
        frames[0] == b"write_req" for frames, _, _ in conn.sent
    )


def test_a_short_source_is_still_refused_with_a_hit(write_path):
    conn = asker(write_path, 0)
    meta = write_path.recv_meta(DST, SRC[:3])
    meta.num_computed_blocks = 2
    conn._request_write("d", meta)
    assert conn._failed_before_read == {"d"}


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

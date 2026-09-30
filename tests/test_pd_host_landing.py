# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Decode host landing (`ATOM_PD_HOST_LANDING_BLOCKS`): GPU-free tests.

Scheduler half: the real `Scheduler` + `BlockManager` + Mooncake consumer
scheduler connector, driven through LANDING -> LANDED -> COPYING -> first
decode, the pool-full / HBM-full fallbacks, and abort or failure in every
phase. Worker half: the host pool's byte layout, checked by running the real
producer RDMA planner (`_execute_block_transfer`, incl. the DCP relayout)
against real CPU memory, once into "HBM" and once into the host pool followed
by the H2D copy -- both must leave identical bytes.
"""

from __future__ import annotations

import ctypes
import importlib
import sys
import threading
import time
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from conftest import MockConfig

# mooncake_connector imports `aiter.dist.parallel_state`, and a real aiter
# import JIT-builds kernels. Stub just that module when aiter is not already
# loaded, then drop the top-level name again (see test_pd_pp.py for why).
_stubbed: list[str] = []
if "aiter.dist.parallel_state" not in sys.modules:
    _aiter_pkg = types.ModuleType("aiter")
    _aiter_pkg.__path__ = []
    _dist = types.ModuleType("aiter.dist")
    _dist.__path__ = []
    _ps_stub = types.ModuleType("aiter.dist.parallel_state")
    for _fn in ("get_dp_group", "get_tp_group"):
        setattr(_ps_stub, _fn, MagicMock())
    for _name, _mod in (
        ("aiter", _aiter_pkg),
        ("aiter.dist", _dist),
        ("aiter.dist.parallel_state", _ps_stub),
    ):
        if _name not in sys.modules:
            sys.modules[_name] = _mod
            _stubbed.append(_name)

from atom.kv_transfer.disaggregation.mooncake.host_landing import (
    HOST_LANDING_COPY_CHANNEL,
    HostBlockAllocator,
    HostLandingBuffer,
    HostLandingController,
    HostLandingPhase,
    copy_runs,
)
from atom.kv_transfer.disaggregation.types import (
    MLA_KV_ROLE,
    ConnectorCompletion,
    ConnectorMetadata,
    KVConnectorOutput,
    LoadOperationId,
)
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence, SequenceStatus

if "aiter" in _stubbed:
    del sys.modules["aiter"]

BLOCK = 4  # MockConfig kv_cache_block_size; DCP 1, so also the hash block


def _mooncake():
    try:
        return importlib.import_module(
            "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
        )
    except Exception as exc:  # noqa: BLE001  # aiter/mooncake backend setup
        pytest.skip(f"mooncake_connector not importable here: {exc!r}")


# ---------------------------------------------------------------------------
# allocator / controller
# ---------------------------------------------------------------------------


def test_allocator_prefers_best_fit_contiguous_runs():
    alloc = HostBlockAllocator(16)
    a = alloc.allocate(4)
    b = alloc.allocate(6)
    c = alloc.allocate(2)
    assert (a, b, c) == ([0, 1, 2, 3], list(range(4, 10)), [10, 11])
    alloc.free(a)  # free runs: [0,4) and [12,16)
    # Both runs fit 3; best fit is either 4-long run -> the first.
    assert alloc.allocate(3) == [0, 1, 2]
    assert alloc.num_free == 16 - 3 - 6 - 2


def test_allocator_gathers_fragments_and_merges_on_free():
    alloc = HostBlockAllocator(10)
    ids = alloc.allocate(10)
    alloc.free(ids[0:2])
    alloc.free(ids[5:8])
    alloc.free(ids[9:10])
    got = alloc.allocate(5)
    assert sorted(got) == got and len(got) == 5
    assert set(got) <= {0, 1, 5, 6, 7, 9}
    assert alloc.allocate(2) is None  # one left
    alloc.free(got)
    alloc.free([2, 3, 4, 8])
    assert alloc.num_free == 10
    assert alloc.allocate(10) == list(range(10))  # merged back into one run


def test_allocator_refuses_double_free_and_overdraw():
    alloc = HostBlockAllocator(4)
    ids = alloc.allocate(2)
    alloc.free(ids)
    with pytest.raises(ValueError, match="freed twice"):
        alloc.free(ids)
    assert alloc.allocate(5) is None
    assert alloc.num_free == 4


def test_copy_runs_split_on_host_gaps_and_flag_hbm_contiguity():
    assert copy_runs([3, 4, 5, 9, 10], [7, 8, 9, 1, 5]) == [
        (0, 3, 3, True),
        (3, 9, 2, False),
    ]


def test_stale_copy_report_does_not_wake_the_request():
    ctl = HostLandingController(8)
    seq = SimpleNamespace(id=5, num_prompt_tokens=16, num_cached_tokens=0)
    assert ctl.reserve(seq, 0, 4)
    ctl.mark_landed(seq)
    ctl.queue_copy(seq, [10, 11, 12, 13])
    (copy,) = ctl.take_pending_copies()
    stale = ConnectorCompletion(
        HOST_LANDING_COPY_CHANNEL,
        LoadOperationId(5, copy.operation.generation + 1),
        True,
    )
    other = ConnectorCompletion("other.channel", LoadOperationId(5, 0), True)
    consumed = ctl.consume_completions({stale, other})
    assert consumed == {stale}
    assert ctl.take_copy_results() == (set(), set())
    good = ConnectorCompletion(HOST_LANDING_COPY_CHANNEL, copy.operation, True)
    ctl.consume_completions({good})
    assert ctl.take_copy_results() == ({5}, set())


# ---------------------------------------------------------------------------
# scheduler state machine
# ---------------------------------------------------------------------------


def _consumer_config():
    return SimpleNamespace(
        kv_transfer_config={"kv_connector": "mooncake", "kv_role": "kv_consumer"},
        tensor_parallel_size=1,
        parallel_config=SimpleNamespace(data_parallel_size=1, data_parallel_rank=0),
        pipeline_parallel_size=1,
        kv_cache_block_size=BLOCK,
        decode_context_parallel_size=1,
    )


def _scheduler(monkeypatch, *, hbm_blocks=64, host_blocks=32, reserve=0):
    mc = _mooncake()
    monkeypatch.setenv("ATOM_PD_HOST_LANDING_BLOCKS", str(host_blocks))
    monkeypatch.setenv("ATOM_PD_HOST_LANDING_HBM_RESERVE_BLOCKS", str(reserve))
    sched = Scheduler(
        MockConfig(
            enable_prefix_caching=True,
            num_kvcache_blocks=hbm_blocks,
            max_model_len=1024,
            max_num_batched_tokens=1024,
            max_num_seqs=16,
        )
    )
    sched.kv_connector = mc.MooncakeConnectorScheduler(_consumer_config())
    assert sched.kv_connector.host_landing is not None
    return sched


def _pd_seq(tokens, seq_id_hint=None, first_token=77):
    params = {
        "do_remote_prefill": True,
        "transfer_id": 1000 + (seq_id_hint or 0),
        "remote_block_ids": list(range(-(-len(tokens) // BLOCK))),
        "remote_host": "10.0.0.1",
        "remote_port": 1,
        "remote_handshake_port": 6301,
        "tp_size": 1,
        "block_size": BLOCK,
        "dcp_size": 1,
        "first_token_id": first_token,
    }
    return Sequence(list(tokens), BLOCK, kv_transfer_params=params)


def _warm_prefix(sched, tokens):
    """Leave ``tokens``' full blocks in the prefix cache, unreferenced."""
    bm = sched.block_manager
    warm = Sequence(list(tokens), BLOCK)
    assert bm.can_allocate(warm) >= 0
    bm.allocate(warm)
    bm.register_received_prefix(warm)
    bm.deallocate(warm)


def _host(sched):
    return sched.kv_connector.host_landing


def _copy_done(sched, copy, ok=True):
    sched._update_from_kv_xfer_finished(
        KVConnectorOutput(
            connector_completions={
                ConnectorCompletion(HOST_LANDING_COPY_CHANNEL, copy.operation, ok)
            }
        )
    )


def test_host_landed_pull_keeps_hbm_free_until_the_copy(monkeypatch):
    sched = _scheduler(monkeypatch)
    bm = sched.block_manager
    prefix = list(range(100, 108))  # two full blocks
    _warm_prefix(sched, prefix)
    seq = _pd_seq(prefix + list(range(200, 216)))  # 24 tokens -> 6 blocks
    free_before = bm.kv.num_free

    sched.add(seq)
    batch, _ = sched.schedule()

    # LANDING: only the HBM prefix hit is held; the suffix lands on host.
    rec = _host(sched).record(seq)
    assert rec.phase is HostLandingPhase.LANDING
    assert seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert len(seq.block_table) == 2 and seq.num_cached_tokens == 8
    assert len(rec.host_block_ids) == 4
    assert bm.kv.num_free == free_before - 2  # the two claimed cache blocks
    recv = batch.connector_meta_output.reqs_to_recv[seq.id]
    assert recv.host_landing is True
    assert recv.local_block_ids == rec.host_block_ids
    assert recv.num_computed_blocks == 2

    # Transfer done -> LANDED -> HBM admitted, copy queued, still parked.
    sched._update_from_kv_xfer_finished(KVConnectorOutput(finished_recving={seq.id}))
    batch, _ = sched.schedule()
    assert rec.phase is HostLandingPhase.COPYING
    assert seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
    (copy,) = batch.connector_meta_output.host_landing_copies
    assert list(copy.host_block_ids) == rec.host_block_ids
    assert list(copy.hbm_block_ids) == list(seq.block_table[2:])
    assert len(seq.block_table) == 6
    assert batch.total_seqs_num == 0

    # Copy done -> first decode, received prefix indexed, host pool empty.
    _copy_done(sched, copy)
    batch, _ = sched.schedule()
    assert seq.status == SequenceStatus.RUNNING
    assert batch.total_seqs_num_decode == 1
    assert seq.token_ids[-1] == 77  # injected T0
    assert _host(sched).record(seq) is None
    assert _host(sched).allocator.num_free == 32
    assert seq.num_hashed_tokens == 24
    assert sched._num_parked_remote_kv == 0


def test_full_host_pool_falls_back_to_the_direct_hbm_pull(monkeypatch):
    sched = _scheduler(monkeypatch, host_blocks=2)
    seq = _pd_seq(range(24))
    sched.add(seq)
    batch, _ = sched.schedule()
    assert _host(sched).record(seq) is None
    recv = batch.connector_meta_output.reqs_to_recv[seq.id]
    assert recv.host_landing is False
    assert recv.local_block_ids == list(seq.block_table)
    assert len(seq.block_table) == 6
    assert _host(sched).allocator.num_free == 2


def _land(sched, seq):
    sched.add(seq)
    sched.schedule()
    sched._update_from_kv_xfer_finished(KVConnectorOutput(finished_recving={seq.id}))


def _occupy(sched, n_blocks):
    """A running decode holding ``n_blocks`` HBM blocks."""
    runner = Sequence(list(range(500, 500 + n_blocks * BLOCK - 1)), BLOCK)
    sched.block_manager.allocate(runner)
    runner.status = SequenceStatus.RUNNING
    sched.running.append(runner)
    return runner


def test_landed_request_waits_for_hbm_and_blocks_direct_pulls(monkeypatch):
    sched = _scheduler(monkeypatch, hbm_blocks=10, host_blocks=6)
    bm = sched.block_manager
    seq = _pd_seq(range(24))  # 6 blocks
    _land(sched, seq)
    runner = _occupy(sched, 4)  # 6 free; need 6 + 1 reserve (one running)
    sched.schedule()
    assert _host(sched).record(seq).phase is HostLandingPhase.LANDED
    assert len(seq.block_table) == 0

    # Host pool is full; a second pull must not take HBM ahead of the landed one.
    other = _pd_seq(range(300, 308), seq_id_hint=1)
    sched.add(other)
    sched.schedule()
    assert other.status == SequenceStatus.WAITING
    assert len(other.block_table) == 0
    assert other in sched.waiting

    # HBM frees up -> the landed request goes first.
    sched.running.remove(runner)
    bm.deallocate(runner)
    batch, _ = sched.schedule()
    assert _host(sched).record(seq).phase is HostLandingPhase.COPYING
    assert len(batch.connector_meta_output.host_landing_copies) == 1


def test_landed_requests_get_hbm_first_come_first_served(monkeypatch):
    sched = _scheduler(monkeypatch, hbm_blocks=12, host_blocks=16)
    big = _pd_seq(range(24))  # 6 blocks
    small = _pd_seq(range(300, 308), seq_id_hint=1)  # 2 blocks
    sched.add(big)
    sched.add(small)
    sched.schedule()  # both parked on host
    runner = _occupy(sched, 7)  # 5 free: small (2 + 1) fits, big (6 + 1) not
    sched._update_from_kv_xfer_finished(
        KVConnectorOutput(finished_recving={big.id, small.id})
    )
    sched.schedule()
    assert _host(sched).record(big).phase is HostLandingPhase.LANDED
    assert _host(sched).record(small).phase is HostLandingPhase.LANDED

    sched.running.remove(runner)
    sched.block_manager.deallocate(runner)
    sched.schedule()
    assert _host(sched).record(big).phase is HostLandingPhase.COPYING
    assert _host(sched).record(small).phase is HostLandingPhase.COPYING


def test_abort_while_landing_waits_for_the_transfer(monkeypatch):
    sched = _scheduler(monkeypatch)
    bm = sched.block_manager
    prefix = list(range(100, 108))
    _warm_prefix(sched, prefix)
    free_before = bm.kv.num_free
    seq = _pd_seq(prefix + list(range(200, 216)))
    sched.add(seq)
    sched.schedule()

    assert sched.abort_request(seq.id)
    # RDMA may still be writing the host blocks: nothing is released yet.
    assert _host(sched).allocator.num_free == 32 - 4
    assert seq.id in sched.deferred_free_blocks

    sched._update_from_kv_xfer_finished(KVConnectorOutput(finished_recving={seq.id}))
    assert _host(sched).allocator.num_free == 32
    assert _host(sched).record(seq) is None
    assert seq.id not in sched.deferred_free_blocks
    assert bm.kv.num_free == free_before
    assert sched._num_parked_remote_kv == 0


def test_abort_while_landed_releases_at_once(monkeypatch):
    sched = _scheduler(monkeypatch, hbm_blocks=10, host_blocks=6)
    seq = _pd_seq(range(24))
    _land(sched, seq)
    _occupy(sched, 8)
    sched.schedule()
    assert _host(sched).record(seq).phase is HostLandingPhase.LANDED

    assert sched.abort_request(seq.id)
    assert _host(sched).allocator.num_free == 6
    assert seq.id not in sched.deferred_free_blocks
    assert sched._num_parked_remote_kv == 0


def test_abort_while_copying_waits_for_the_copy(monkeypatch):
    sched = _scheduler(monkeypatch)
    bm = sched.block_manager
    free_before = bm.kv.num_free
    seq = _pd_seq(range(24))
    _land(sched, seq)
    batch, _ = sched.schedule()
    (copy,) = batch.connector_meta_output.host_landing_copies

    assert sched.abort_request(seq.id)
    # The copy is writing these HBM blocks: they stay held until it reports.
    assert bm.kv.num_free == free_before - 6
    assert _host(sched).allocator.num_free == 32 - 6

    _copy_done(sched, copy)
    assert bm.kv.num_free == free_before
    assert _host(sched).allocator.num_free == 32
    assert seq.id not in sched.deferred_free_blocks
    assert sched._num_parked_remote_kv == 0


def test_failed_pull_falls_back_to_local_prefill(monkeypatch):
    sched = _scheduler(monkeypatch)
    seq = _pd_seq(range(24))
    sched.add(seq)
    sched.schedule()
    sched._update_from_kv_xfer_finished(KVConnectorOutput(failed_recving={seq.id}))
    batch, _ = sched.schedule()
    assert _host(sched).allocator.num_free == 32
    assert _host(sched).record(seq) is None
    assert batch.total_seqs_num_prefill == 1
    assert seq.status == SequenceStatus.RUNNING
    assert len(seq.block_table) == 6


def test_failed_copy_falls_back_to_local_prefill(monkeypatch):
    sched = _scheduler(monkeypatch)
    seq = _pd_seq(range(24))
    _land(sched, seq)
    batch, _ = sched.schedule()
    (copy,) = batch.connector_meta_output.host_landing_copies
    _copy_done(sched, copy, ok=False)
    batch, _ = sched.schedule()
    assert _host(sched).allocator.num_free == 32
    assert batch.total_seqs_num_prefill == 1


def test_disabled_by_default(monkeypatch):
    mc = _mooncake()
    monkeypatch.delenv("ATOM_PD_HOST_LANDING_BLOCKS", raising=False)
    assert mc.MooncakeConnectorScheduler(_consumer_config()).host_landing is None


# ---------------------------------------------------------------------------
# worker: byte layout, request body, copy reporting
# ---------------------------------------------------------------------------


class _Memory:
    """RDMA stand-in over real process memory."""

    def __init__(self):
        self.batches = 0

    def write(self, _target, src, dst, sizes, _req, _label, *, engine=None):
        for s, d, n in zip(src, dst, sizes):
            ctypes.memmove(d, s, n)
        self.batches += 1
        return True


def _producer(mc, regions, block_size):
    conn = object.__new__(mc.MooncakeConnector)
    conn.dcp_size = 1
    conn.block_size = block_size
    conn.kv_caches_base_addr = [t.data_ptr() for t in regions]
    conn._per_block_bytes_list = [t.shape[1] for t in regions]
    conn._block_region_roles = [MLA_KV_ROLE] * len(regions)
    conn._block_region_consumer_indices = None
    conn._consumer_region_map = lambda *_a, **_k: list(range(len(regions)))
    conn._prepare_sharded_index = None
    conn._gather_sharded_index = None
    conn._index_staging_chunk_pages = 0
    mem = _Memory()
    conn._rdma_write_with_retry = mem.write
    return conn


@pytest.mark.parametrize("consumer_dcp", [1, 2])
def test_host_landing_places_bytes_exactly_like_hbm_landing(consumer_dcp):
    mc = _mooncake()
    block_size = 4
    bytes_per_token = (8, 3)  # two regions of different widths
    gen = torch.Generator().manual_seed(0)
    src_blocks = 12
    producer = [
        torch.randint(0, 255, (src_blocks, block_size * w), generator=gen).to(
            torch.uint8
        )
        for w in bytes_per_token
    ]
    num_dst = 16
    hbm_a = [
        torch.zeros(num_dst, 1, block_size * w, dtype=torch.uint8)
        for w in bytes_per_token
    ]
    hbm_b = [torch.zeros_like(t) for t in hbm_a]
    # Scattered, non-monotonic destinations, as a real block table has.
    n_pages = src_blocks // consumer_dcp
    hbm_ids = [9, 2, 3, 4, 15, 0, 13, 12, 1, 5, 6, 7][:n_pages]
    src_ids = [11, 0, 5, 6, 7, 1, 2, 3, 8, 9, 10, 4][: n_pages * consumer_dcp]
    conn = _producer(mc, producer, block_size)
    request = {
        "consumer_num_layers": 2,
        "consumer_region_roles": [MLA_KV_ROLE, MLA_KV_ROLE],
        "consumer_block_bpb": [block_size * w for w in bytes_per_token],
        "consumer_dcp_size": consumer_dcp,
        "consumer_dcp_rank": consumer_dcp - 1,
        "consumer_dcp_interleave": 1,
    }

    # A: straight into HBM.
    req_a = dict(request, consumer_base_addrs=[t.data_ptr() for t in hbm_a])
    assert mc.MooncakeConnector._execute_block_transfer(
        conn, req_a, "t", src_ids, hbm_ids, "r"
    )

    # B: into the host pool, then the H2D copy.
    pool = HostLandingBuffer(hbm_b, 16, device="cpu", pin=False)
    host_ids = [5, 6, 7, 1, 2, 0, 8, 9, 10, 14, 15, 3][:n_pages]
    req_b = dict(request, consumer_base_addrs=pool.base_addrs)
    assert mc.MooncakeConnector._execute_block_transfer(
        conn, req_b, "t", src_ids, host_ids, "r"
    )
    start, done, nbytes = pool.copy_to_hbm(host_ids, hbm_ids)
    assert (start, done) == (None, None)
    assert nbytes == n_pages * block_size * sum(bytes_per_token)

    for a, b in zip(hbm_a, hbm_b):
        assert torch.equal(a, b)
        assert a.any()  # something actually moved


def _consumer(mc, host_pool):
    conn = object.__new__(mc.MooncakeConnector)
    conn.is_producer = False
    conn.tp_rank = 0
    conn.tp_size = 1
    conn.dp_rank = 0
    conn.local_ip = "10.0.0.2"
    conn.rpc_port = 7
    conn.ib_device = None
    conn._num_local_layers = 2
    conn._block_region_roles = [MLA_KV_ROLE, MLA_KV_ROLE]
    conn._block_regions = [(111, 32), (222, 12)]
    conn.kv_caches_base_addr = [111, 222]
    conn._has_slot_regions = False
    conn._fp4_index_layout = False
    conn._notification_port = 42000
    conn.dcp_size = 1
    conn.dcp_rank = 0
    conn.dcp_interleave_size = 1
    conn._host_landing = host_pool
    conn._completion_lock = threading.Lock()
    conn._pending_recv_expected = {}
    conn._pending_recv_nonce = {}
    conn._pending_recv = set()
    conn._pending_recv_blocks = {}
    conn._pending_recv_slots = {}
    conn._dispatch_in_flight = set()
    conn._deferred_failures = {}
    conn._release_targets = {}
    conn.failed_recving = set()
    conn.done_recving = set()
    conn.done_sending = set()
    conn._send_on_socket = MagicMock()
    return conn


def _recv_meta(host_landing, local_block_ids, off):
    meta = ConnectorMetadata()
    meta.reqs_to_recv["r0"] = ConnectorMetadata._build_req_meta(
        req_id="r0",
        local_block_ids=local_block_ids,
        kv_transfer_params={
            "remote_block_ids": list(range(100, 106)),
            "remote_host": "10.0.0.1",
            "remote_handshake_port": 6301,
            "tp_size": 1,
            "transfer_id": 9,
            "num_computed_blocks": off,
            "host_landing": host_landing,
        },
    )
    return meta


def test_consumer_advertises_host_addresses_and_suffix_blocks():
    mc = _mooncake()
    conn = _consumer(mc, SimpleNamespace(base_addrs=[900, 950]))
    conn.start_load_kv(_recv_meta(True, [40, 41, 42, 43], off=2))
    _addr, (_kind, payload) = conn._send_on_socket.call_args.args
    body = mc.msgpack.loads(payload)
    assert body["consumer_base_addrs"] == [900, 950]
    assert body["dst_block_ids"] == [40, 41, 42, 43]
    assert body["src_block_ids"] == [102, 103, 104, 105]
    assert body["num_computed_blocks"] == 2
    assert "r0" not in conn._pending_recv_blocks  # no HBM fence for host blocks

    # The HBM path is unchanged.
    conn = _consumer(mc, SimpleNamespace(base_addrs=[900, 950]))
    conn.start_load_kv(_recv_meta(False, [7, 8, 9, 10, 11, 12], off=2))
    body = mc.msgpack.loads(conn._send_on_socket.call_args.args[1][1])
    assert body["consumer_base_addrs"] == [111, 222]
    assert body["dst_block_ids"] == [9, 10, 11, 12]


def test_consumer_fails_a_host_landing_it_cannot_serve():
    mc = _mooncake()
    conn = _consumer(mc, None)
    conn.start_load_kv(_recv_meta(True, [40, 41], off=0))
    conn._send_on_socket.assert_not_called()
    assert conn.get_finished().failed_recving == {"r0"}
    assert conn._pending_recv_expected == {}


class _Event:
    def __init__(self, done=False):
        self.done = done

    def query(self):
        return self.done

    def elapsed_time(self, _other):
        return 2.0


def test_worker_reports_copies_once_their_event_completes():
    mc = _mooncake()
    conn = _consumer(mc, None)
    conn._host_landing_inflight = []
    conn._host_landing_completions = set()
    op = LoadOperationId(3, 1)
    done = _Event(False)
    conn._host_landing_inflight.append(
        (op, _Event(True), done, 4_000_000, time.monotonic(), True)
    )
    assert conn.get_finished().connector_completions == set()
    done.done = True
    out = conn.get_finished()
    assert out.connector_completions == {
        ConnectorCompletion(HOST_LANDING_COPY_CHANNEL, op, True)
    }
    assert conn.get_finished().connector_completions == set()


def test_worker_without_a_pool_fails_the_copy():
    mc = _mooncake()
    conn = _consumer(mc, None)
    conn._host_landing_inflight = []
    conn._host_landing_completions = set()
    copy = SimpleNamespace(
        operation=LoadOperationId(4, 1), host_block_ids=(0,), hbm_block_ids=(1,)
    )
    meta = ConnectorMetadata()
    meta.host_landing_copies = [copy]
    assert meta.has_work()
    conn.start_load_kv(meta)
    assert conn.get_finished().connector_completions == {
        ConnectorCompletion(HOST_LANDING_COPY_CHANNEL, copy.operation, False)
    }


def test_copy_mixes_direct_and_staged_runs():
    views = [torch.zeros(10, 1, 6, dtype=torch.uint8)]
    pool = HostLandingBuffer(views, 5, device="cpu", pin=False)
    assert pool.host.data_ptr() % 4096 == 0
    pool.host_views[0].copy_(torch.arange(30, dtype=torch.uint8).view(5, 6))
    pool.copy_to_hbm([0, 1, 3, 4], [6, 7, 2, 9])
    out = views[0].view(10, 6)
    assert torch.equal(out[6], pool.host_views[0][0])
    assert torch.equal(out[7], pool.host_views[0][1])
    assert torch.equal(out[2], pool.host_views[0][3])
    assert torch.equal(out[9], pool.host_views[0][4])
    assert not out[[0, 1, 3, 4, 5, 8]].any()


# ---------------------------------------------------------------------------
# review fixes
# ---------------------------------------------------------------------------


def test_mooncake_scheduler_leaves_multi_composites_one_completion_handler(
    monkeypatch,
):
    """`MultiConnectorScheduler` refuses >1 sub with `process_completions`;
    the prefill `[mooncake, lmcache_offload]` composite must keep working."""
    mc = _mooncake()
    from atom.kv_transfer.disaggregation.multi.multi_connector import (
        MultiConnectorScheduler,
    )

    monkeypatch.delenv("ATOM_PD_HOST_LANDING_BLOCKS", raising=False)
    producer_cfg = _consumer_config()
    producer_cfg.kv_transfer_config = {
        "kv_connector": "mooncake",
        "kv_role": "kv_producer",
    }
    pd = mc.MooncakeConnectorScheduler(producer_cfg)
    assert not hasattr(pd, "process_completions")
    seen = []

    class _Offload:
        def process_completions(self, output):
            seen.append(output)
            return output

    multi = object.__new__(MultiConnectorScheduler)
    multi._connectors = [pd, _Offload()]
    out = KVConnectorOutput()
    assert multi.process_completions(out) is out
    assert seen == [out]


def test_landed_request_is_not_wedged_by_the_reserve_when_idle(monkeypatch):
    # 6 free HBM blocks, need 6, reserve 256: with nothing running and no copy
    # in flight nothing will ever release the reserve, so it must not apply.
    sched = _scheduler(monkeypatch, hbm_blocks=6, host_blocks=8, reserve=256)
    seq = _pd_seq(range(24))
    _land(sched, seq)
    batch, _ = sched.schedule()
    assert _host(sched).record(seq).phase is HostLandingPhase.COPYING
    assert len(batch.connector_meta_output.host_landing_copies) == 1


def test_reserve_still_protects_running_decodes(monkeypatch):
    sched = _scheduler(monkeypatch, hbm_blocks=12, host_blocks=8, reserve=4)
    seq = _pd_seq(range(24))  # 6 blocks
    _land(sched, seq)
    _occupy(sched, 3)  # 9 free < 6 + 1 running + 4 reserve
    sched.schedule()
    assert _host(sched).record(seq).phase is HostLandingPhase.LANDED


def test_fifo_yields_when_nothing_can_free_hbm(monkeypatch):
    # big's suffix does not fit beside small's prefix claim, and small cannot
    # decode until it gets HBM: strict FIFO would wait forever.
    sched = _scheduler(monkeypatch, hbm_blocks=8, host_blocks=16)
    bm = sched.block_manager
    prefix = list(range(900, 908))  # 2 blocks
    _warm_prefix(sched, prefix)
    big = _pd_seq(range(24))  # 6 blocks, no hit
    small = _pd_seq(prefix + list(range(300, 304)), seq_id_hint=1)  # 3, hit 2
    sched.add(big)
    sched.add(small)
    sched.schedule()
    assert len(small.block_table) == 2  # claimed prefix: 6 blocks left
    runner = _occupy(sched, 1)  # 5 free
    sched._update_from_kv_xfer_finished(
        KVConnectorOutput(finished_recving={big.id, small.id})
    )
    sched.schedule()
    # A running decode will free HBM: strict FIFO holds small behind big.
    assert _host(sched).record(big).phase is HostLandingPhase.LANDED
    assert _host(sched).record(small).phase is HostLandingPhase.LANDED
    sched.running.remove(runner)
    bm.deallocate(runner)
    # Now nothing runs. Pin one more block with a claim nobody releases (as
    # a third request's prefix claim would): 5 free, big needs 6.
    _warm_prefix(sched, list(range(700, 704)))
    bm.claim_prefix_hit(Sequence(list(range(700, 704)) + [1], BLOCK))
    assert bm.kv.num_free == 5
    batch, _ = sched.schedule()
    assert _host(sched).record(big).phase is HostLandingPhase.LANDED
    assert _host(sched).record(small).phase is HostLandingPhase.COPYING
    (copy,) = batch.connector_meta_output.host_landing_copies
    assert copy.operation.req_id == small.id


class _FailingPool:
    def __init__(self):
        self.stream = None

    def copy_to_hbm(self, host_ids, hbm_ids):
        raise torch.cuda.OutOfMemoryError("staging")


def test_failed_copy_is_reported_only_after_queued_work_drains(monkeypatch):
    mc = _mooncake()
    pool = _FailingPool()
    pool.stream = object()  # a side stream with work possibly queued
    conn = _consumer(mc, pool)
    conn._host_landing_inflight = []
    conn._host_landing_completions = set()
    drained = _Event(False)

    class _EventCls:
        def __new__(cls, *a, **k):
            return drained

    drained.record = lambda _stream: None
    monkeypatch.setattr(mc.torch.cuda, "Event", _EventCls)
    copy = SimpleNamespace(
        operation=LoadOperationId(6, 1), host_block_ids=(0,), hbm_block_ids=(1,)
    )
    meta = ConnectorMetadata()
    meta.host_landing_copies = [copy]
    conn.start_load_kv(meta)
    assert conn.get_finished().connector_completions == set()
    drained.done = True
    assert conn.get_finished().connector_completions == {
        ConnectorCompletion(HOST_LANDING_COPY_CHANNEL, copy.operation, False)
    }


def test_host_pool_numa_preference_is_best_effort():
    views = [torch.zeros(4, 1, 8192, dtype=torch.uint8)]
    pool = HostLandingBuffer(views, 4, device="cpu", pin=False, numa_node=0)
    pool.host_views[0].fill_(3)
    assert pool.host_views[0].sum().item() == 3 * 4 * 8192


def test_composites_turn_host_landing_off_on_their_subs(monkeypatch):
    mc = _mooncake()
    from atom.kv_transfer.disaggregation.multi import multi_connector

    monkeypatch.setenv("ATOM_PD_HOST_LANDING_BLOCKS", "8")
    pd = mc.MooncakeConnectorScheduler(_consumer_config())
    assert pd.host_landing is not None
    worker = SimpleNamespace(_host_landing_blocks=8)
    monkeypatch.setattr(
        multi_connector,
        "_build_subconnectors",
        lambda _cfg, role: [pd] if role == "scheduler" else [worker],
    )
    multi_connector.MultiConnectorScheduler(SimpleNamespace())
    assert pd.host_landing is None
    multi_connector.MultiConnector(SimpleNamespace())
    assert worker._host_landing_blocks == 0

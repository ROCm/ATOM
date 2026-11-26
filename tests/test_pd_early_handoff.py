# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""ATOM_PD_EARLY_HANDOFF: P answers the router at DP dispatch and pushes chunks.

CPU only. Reuses the chunked-transfer protocol fixtures (real schedulers, real
producer/consumer workers, CPU buffers in place of HBM and the NIC).
"""

import asyncio
import ctypes
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from test_pd_chunked_transfer import (
    decode_worker,
    dispatch,
    idle_producer,
    mc,
    post_write_requests,
    publish,
)

from atom.kv_transfer.disaggregation.types import KVConnectorOutput

TRANSFER = "early-xfer"


def make_scheduler_connector(
    monkeypatch,
    is_producer,
    *,
    early,
    enable=False,
    skip_gate=None,
    num_kvcache_blocks=100,
):
    """`skip_gate`: None leaves ATOM_PD_EARLY_HANDOFF_SKIP_GATE at its
    default (on), else sets it. `num_kvcache_blocks`: P's pool, which the
    cap on early sources is a share of."""
    if early:
        monkeypatch.setenv("ATOM_PD_EARLY_HANDOFF", "1")
    else:
        monkeypatch.setenv("ATOM_PD_EARLY_HANDOFF", "0")
    if skip_gate is None:
        monkeypatch.delenv("ATOM_PD_EARLY_HANDOFF_SKIP_GATE", raising=False)
    else:
        monkeypatch.setenv("ATOM_PD_EARLY_HANDOFF_SKIP_GATE", "1" if skip_gate else "0")
    monkeypatch.setattr(mc, "get_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(mc, "get_open_port", lambda: 6400)
    cfg = {"kv_role": "kv_producer" if is_producer else "kv_consumer"}
    if enable:
        cfg["enable_chunked_transfer"] = True
    config = SimpleNamespace(
        kv_transfer_config=cfg,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        parallel_config=SimpleNamespace(data_parallel_size=1, data_parallel_rank=0),
        kv_cache_block_size=4,
        decode_context_parallel_size=1,
        num_kvcache_blocks=num_kvcache_blocks,
    )
    return mc.MooncakeConnectorScheduler(config)


def p_params(transfer_id=TRANSFER, early=True):
    """What `api_server._early_handoff_params` makes of the router's leg
    (`early=False`: a router-chunked leg, which never skips the gate)."""
    params = {
        "chunked_transfer": True,
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "transfer_id": transfer_id,
    }
    if early:
        params["pd_early_handoff"] = True
    return params


def d_params(transfer_id=TRANSFER):
    """What the legacy router relay hands D from P's early answer."""
    return {
        "chunked_transfer": True,
        "do_remote_prefill": True,
        "do_remote_decode": False,
        "transfer_id": transfer_id,
        "block_size": 4,
        "dcp_size": 1,
        "remote_host": "cpu",
        "remote_handshake_port": 6301,
        "remote_pp_size": 1,
        "tp_size": 1,
        "remote_tp_size": 1,
        "remote_dp_size": 1,
        "remote_dp_rank": 0,
    }


# ---------------------------------------------------------------- scheduler gate


def test_flag_off_keeps_existing_config(monkeypatch, seq_factory):
    plain = make_scheduler_connector(monkeypatch, True, early=False)
    assert not plain.enable_chunked_transfer
    # Without the flag (and without enable_chunked_transfer) a chunked P leg
    # is refused exactly as before.
    seq = seq_factory(list(range(10)), kv_transfer_params=p_params())
    with pytest.raises(ValueError, match="enable_chunked_transfer"):
        plain.admission_ready(seq)


def test_flag_enables_chunked_producer_but_not_consumer(monkeypatch):
    p = make_scheduler_connector(monkeypatch, True, early=True)
    assert p.enable_chunked_transfer
    d = make_scheduler_connector(monkeypatch, False, early=True)
    assert not d.enable_chunked_transfer


def _ack(p, seq, succeeded=True):
    """D's write request reached every P rank (the worker's report)."""
    p.process_pd_completions(
        KVConnectorOutput(
            connector_completions={
                mc.ConnectorCompletion("pd_admission_ready", seq.id, succeeded)
            }
        )
    )


def test_early_request_skips_the_gate_and_is_registered_once(monkeypatch, seq_factory):
    """Default (SKIP_GATE=1): P may allocate an early request at once, so the
    router->D hop overlaps P's prefill. It is still registered with the
    worker once, like a gated request, so D's arrival is reported and a write
    request that arrives while P queues is not expired as an orphan."""
    p = make_scheduler_connector(monkeypatch, True, early=True)
    seq = seq_factory(list(range(10)), kv_transfer_params=p_params())
    for _ in range(3):  # The scheduler asks on every pass.
        assert p.admission_ready(seq)
    assert seq.id in p._chunked_waiting
    assert p.build_connector_meta().chunked_admissions == {seq.id: TRANSFER}
    assert p.admission_ready(seq)
    assert p.build_connector_meta().chunked_admissions == {}


def test_only_flagged_early_requests_skip_the_gate(monkeypatch, seq_factory):
    """A router-chunked leg (no `pd_early_handoff`) keeps the gate even with
    early handoff on: its D may admit in any order."""
    p = make_scheduler_connector(monkeypatch, True, early=True, enable=True)
    seq = seq_factory(list(range(10)), kv_transfer_params=p_params(early=False))
    assert not p.admission_ready(seq)
    _ack(p, seq)
    assert p.admission_ready(seq)


def test_skip_gate_needs_early_handoff_on(monkeypatch, seq_factory):
    """The skip is an early-handoff knob: without ATOM_PD_EARLY_HANDOFF a
    flagged request on a router-chunked producer still waits for D."""
    p = make_scheduler_connector(monkeypatch, True, early=False, enable=True)
    seq = seq_factory(list(range(10)), kv_transfer_params=p_params())
    assert not p.admission_ready(seq)


def test_early_request_waits_for_d_and_is_registered_once(monkeypatch, seq_factory):
    """SKIP_GATE=0: P allocates an early request's source only after D
    reserved its pages (the chunked admission gate), so P admits in D's
    order. The transfer ID is registered with the worker once, on the first
    pass."""
    p = make_scheduler_connector(monkeypatch, True, early=True, skip_gate=False)
    seq = seq_factory(list(range(10)), kv_transfer_params=p_params())
    for _ in range(3):  # The scheduler asks on every pass.
        assert not p.admission_ready(seq)
    assert seq.id in p._chunked_waiting
    assert p.build_connector_meta().chunked_admissions == {seq.id: TRANSFER}
    assert not p.admission_ready(seq)
    assert p.build_connector_meta().chunked_admissions == {}
    # D's write request reached every P rank: the gate opens.
    _ack(p, seq)
    assert p.admission_ready(seq)


@pytest.mark.parametrize("skip_gate", [False, True])
def test_early_request_whose_d_never_comes_expires_without_a_source(
    monkeypatch, seq_factory, skip_gate
):
    """Gated, or skipping the gate but never allocated (P stays full): the
    admission timeout still aborts it before it takes a source."""
    from atom.model_engine.sequence import SequenceStatus

    p = make_scheduler_connector(monkeypatch, True, early=True, skip_gate=skip_gate)
    seq = seq_factory(list(range(10)), kv_transfer_params=p_params())
    clock = [100.0]
    monkeypatch.setattr(mc.time, "monotonic", lambda: clock[0])
    assert p.admission_ready(seq) is skip_gate
    clock[0] += p._chunked_admission_timeout_s
    assert not p.admission_ready(seq)
    assert seq.status == SequenceStatus.ABORTED
    assert not seq.block_table and not p._chunked_waiting


# ------------------------------------------------ real P and D, D arrives late


def _p_scheduler(monkeypatch, num_blocks=6, skip_gate=None):
    from conftest import MockConfig

    from atom.model_engine.scheduler import Scheduler

    p = Scheduler(MockConfig(num_kvcache_blocks=num_blocks, max_num_batched_tokens=4))
    p.kv_connector = make_scheduler_connector(
        monkeypatch,
        True,
        early=True,
        skip_gate=skip_gate,
        num_kvcache_blocks=num_blocks,
    )
    return p


def _d_scheduler(monkeypatch, num_blocks=6):
    from conftest import MockConfig

    from atom.model_engine.scheduler import Scheduler

    d = Scheduler(MockConfig(num_kvcache_blocks=num_blocks, max_num_batched_tokens=4))
    d.kv_connector = make_scheduler_connector(monkeypatch, False, early=True)
    return d


def _wire(pw, dw):
    import msgpack

    def receive(_addr, message):
        assert message[0] == mc.MSG_WRITE_REQUEST
        dispatch(pw, msgpack.loads(message[1]))

    def complete(data, success):
        assert success
        dw._record_write_done(
            data["request_id"],
            0,
            0,
            data["write_nonce"],
            handoff=data["prefill_handoff"],
        )

    dw._send_on_socket.side_effect = receive
    pw._notify_transfer_result.side_effect = complete
    pw._ack_write_request = MagicMock(
        side_effect=lambda data: dw._record_write_ack(
            {
                "request_id": data["request_id"],
                "write_nonce": data["write_nonce"],
                "pp_rank": 0,
                "tp_rank": 0,
                "write_window_s": mc._write_window_s(),
            }
        )
    )


@pytest.mark.parametrize("p_passes_before_d", [0, 1, 3])
def test_d_arriving_while_p_queues_receives_every_page(
    monkeypatch, seq_factory, p_passes_before_d
):
    """P answered the router at dispatch; D arrives while P still queues.

    `p_passes_before_d` counts P scheduling passes before D's reservation
    (0 = D's write request reaches P before P registered the transfer). With
    SKIP_GATE=0 P allocates nothing until D reserved; then every page must
    land byte-identical, D gets the first token, and P frees its source only
    after D's reader finished.
    """
    from atom.model_engine.scheduler import ScheduledBatchOutput
    from atom.sampling_params import SamplingParams

    p = _p_scheduler(monkeypatch, skip_gate=False)
    d = _d_scheduler(monkeypatch)
    prompt = list(range(10))
    ps = seq_factory(
        prompt,
        sampling_params=SamplingParams(max_tokens=1),
        kv_transfer_params=p_params(),
    )
    ds = seq_factory(
        prompt,
        sampling_params=SamplingParams(max_tokens=2),
        kv_transfer_params=d_params(),
    )
    pw, _, src, destinations = idle_producer()
    dw = decode_worker(destinations[0])
    _wire(pw, dw)
    p.extend([ps])

    def d_arrives():
        d.extend([ds])
        batch, seqs = d.schedule()
        assert not seqs and len(ds.block_table) == 3  # Full prompt reserved.
        dw.start_load_kv(batch.connector_meta_output)
        post_write_requests(dw, monkeypatch)
        p._update_from_kv_xfer_finished(pw.get_finished())

    pool = ThreadPoolExecutor(1)
    pw._send_executor = pool
    try:
        for _ in range(p_passes_before_d):
            batch, seqs = p.schedule()
            # The admission gate: no source before D reserved its pages.
            assert not seqs and not ps.block_table
            pw.start_load_kv(batch.connector_meta_output)
        d_arrives()
        for _ in range(3):  # D's readiness reaches P's scheduler.
            batch, seqs = p.schedule()
            if seqs:
                break
            pw.start_load_kv(batch.connector_meta_output)
            p._update_from_kv_xfer_finished(pw.get_finished())
        for forward in range(3):
            if forward:
                batch, seqs = p.schedule()
            assert list(seqs) == [ps.id]
            assert batch.context_lens.tolist() == [min(4 * (forward + 1), 10)]
            pw.start_load_kv(batch.connector_meta_output)
            state = pw._chunked_prefills[TRANSFER]
            assert not p._is_preemptable(ps)
            publish(state, int(batch.context_lens[0]))
            pw._dispatch_ready_chunked_requests()
            p.postprocess(
                list(seqs.values()),
                ScheduledBatchOutput([ps.id], [(42,)], None, None, None),
                batch=batch,
            )
            pw.start_load_kv(p.kv_connector.build_connector_meta())
            if forward < 2:
                # Source pinned while D's reader is mid-read.
                assert ps.block_table
        pool.shutdown(wait=True)
    finally:
        pool.shutdown(wait=True)
    pw._send_executor = SimpleNamespace(submit=lambda fn, *a, **kw: fn(*a, **kw))
    p._update_from_kv_xfer_finished(pw.get_finished())
    assert not ps.block_table, "P must free its source once D's reader finished"
    assert not p.deferred_free_blocks
    d._update_from_kv_xfer_finished(dw.get_finished())
    batch, seqs = d.schedule()
    assert list(seqs) == [ds.id]
    assert batch.total_seqs_num_decode == 1 and batch.total_seqs_num_prefill == 0
    assert ds.kv_transfer_params["first_token_id"] == 42
    # Byte-identical pages: P block b -> D block d for the whole prompt.
    p_blocks = pw._notify_transfer_result.call_args.args[0]
    assert pw._notify_transfer_result.call_count == 1
    sent = []
    while not pw.transfer_engine.writes.empty():
        sources, dsts, sizes = pw.transfer_engine.writes.get_nowait()
        sent += list(zip(sources, dsts, sizes, strict=True))
    base_src = ctypes.addressof(src)
    base_dst = ctypes.addressof(destinations[0])
    assert sum(size for _, _, size in sent) == 12
    for s, t, size in sent:
        assert bytes(src[s - base_src : s - base_src + size]) == bytes(
            destinations[0][t - base_dst : t - base_dst + size]
        )
    assert p_blocks["transfer_id"] == TRANSFER
    assert not pw._pending_chunked_requests


def _assert_pages_landed(pw, src, dst, total):
    sent = []
    while not pw.transfer_engine.writes.empty():
        sources, dsts, sizes = pw.transfer_engine.writes.get_nowait()
        sent += list(zip(sources, dsts, sizes, strict=True))
    base_src = ctypes.addressof(src)
    base_dst = ctypes.addressof(dst)
    assert sum(size for _, _, size in sent) == total
    for s, t, size in sent:
        assert bytes(src[s - base_src : s - base_src + size]) == bytes(
            dst[t - base_dst : t - base_dst + size]
        )


@pytest.mark.parametrize("forwards_before_d", [1, 3])
def test_p_prefills_before_d_arrives_and_d_receives_every_page(
    monkeypatch, seq_factory, forwards_before_d
):
    """Default (SKIP_GATE=1): P allocates and prefills at once, before D's
    write request (3 = the whole prefill, the handoff included, is done
    before D arrives). D's reader then starts from block 0: every page lands
    byte-identical, D gets the first token, and P frees its source only once
    D's reader finished."""
    from atom.model_engine.scheduler import ScheduledBatchOutput
    from atom.sampling_params import SamplingParams

    p = _p_scheduler(monkeypatch)
    d = _d_scheduler(monkeypatch)
    prompt = list(range(10))
    ps = seq_factory(
        prompt,
        sampling_params=SamplingParams(max_tokens=1),
        kv_transfer_params=p_params(),
    )
    ds = seq_factory(
        prompt,
        sampling_params=SamplingParams(max_tokens=2),
        kv_transfer_params=d_params(),
    )
    pw, _, src, destinations = idle_producer()
    dw = decode_worker(destinations[0])
    _wire(pw, dw)
    p.extend([ps])

    def p_forward(forward):
        batch, seqs = p.schedule()
        assert list(seqs) == [ps.id]
        assert batch.context_lens.tolist() == [min(4 * (forward + 1), 10)]
        pw.start_load_kv(batch.connector_meta_output)
        publish(pw._chunked_prefills[TRANSFER], int(batch.context_lens[0]))
        pw._dispatch_ready_chunked_requests()
        p.postprocess(
            list(seqs.values()),
            ScheduledBatchOutput([ps.id], [(42,)], None, None, None),
            batch=batch,
        )
        pw.start_load_kv(p.kv_connector.build_connector_meta())

    pool = ThreadPoolExecutor(1)
    pw._send_executor = pool
    try:
        for forward in range(forwards_before_d):
            p_forward(forward)
            # No D yet: the source is pinned, nothing is sent.
            assert ps.block_table and not p._is_preemptable(ps)
            assert pw.transfer_engine.writes.empty()
        d.extend([ds])
        batch, seqs = d.schedule()
        assert not seqs and len(ds.block_table) == 3
        dw.start_load_kv(batch.connector_meta_output)
        post_write_requests(dw, monkeypatch)
        p._update_from_kv_xfer_finished(pw.get_finished())
        for forward in range(forwards_before_d, 3):
            p_forward(forward)
        pool.shutdown(wait=True)
    finally:
        pool.shutdown(wait=True)
    pw._send_executor = SimpleNamespace(submit=lambda fn, *a, **kw: fn(*a, **kw))
    p._update_from_kv_xfer_finished(pw.get_finished())
    assert not ps.block_table, "P must free its source once D's reader finished"
    assert not p.deferred_free_blocks
    assert not p.kv_connector._chunked_waiting
    d._update_from_kv_xfer_finished(dw.get_finished())
    batch, seqs = d.schedule()
    assert list(seqs) == [ds.id]
    assert ds.kv_transfer_params["first_token_id"] == 42
    assert pw._notify_transfer_result.call_count == 1
    _assert_pages_landed(pw, src, destinations[0], 12)
    assert not pw._pending_chunked_requests


@pytest.mark.parametrize("skip_gate", [False, True])
@pytest.mark.parametrize("byte_limit", [False, True])
def test_opposite_admission_order_progresses_in_early_mode(
    monkeypatch, seq_factory, byte_limit, skip_gate
):
    """Room for one prompt per side; P queues A then B, D admits B then A.

    P answered both at dispatch, so both decode legs reach D. Had P allocated
    A's source before D reserved it, D's pages (or, with `byte_limit`, its
    admission bytes) would hold B, P's would hold a finished A that only D's
    A reader can free, and neither side could admit its other request (the
    pair stalls until the worker's lookup timeout). With the gate (SKIP_GATE=0), or with
    the skip and the default cap (a 3-page prompt is past half of P's 3
    pages, so it waits for D), P follows D's order: both requests finish, no
    transfer fails, nothing is recomputed.
    """
    from atom.kv_transfer.disaggregation.pd_admission import PdAdmissionController
    from atom.model_engine.scheduler import ScheduledBatchOutput
    from atom.sampling_params import SamplingParams

    monkeypatch.delenv("ATOM_PD_EARLY_HANDOFF_MAX_PINNED_FRACTION", raising=False)
    p = _p_scheduler(monkeypatch, num_blocks=3, skip_gate=skip_gate)
    d_blocks = 6 if byte_limit else 3
    d = _d_scheduler(monkeypatch, num_blocks=d_blocks)
    if byte_limit:
        d.pd_admission = PdAdmissionController(12, 4, d.kv_connector, d.block_manager)

    def seqs_for(name):
        ps = seq_factory(
            list(range(10)),
            sampling_params=SamplingParams(max_tokens=1),
            kv_transfer_params=p_params(name),
        )
        ds = seq_factory(
            list(range(10)),
            sampling_params=SamplingParams(max_tokens=2),
            kv_transfer_params=d_params(name),
        )
        return ps, ds

    pa, da = seqs_for("early-a")
    pb, db = seqs_for("early-b")
    p.extend([pa, pb])
    d.extend([db, da])
    pw, _, _src, destinations = idle_producer()
    dw = decode_worker(destinations[0])
    _wire(pw, dw)
    finished, started = [], []
    peer = {pa.id: da, pb.id: db}
    pool = ThreadPoolExecutor(1)
    pw._send_executor = pool
    try:
        for _ in range(20):
            if len(finished) == 2:
                break
            time.sleep(0.02)  # let the send thread write what was published
            d._update_from_kv_xfer_finished(dw.get_finished())
            out = d.schedule()
            if out is not None:
                batch, seqs = out
                dw.start_load_kv(batch.connector_meta_output)
                if seqs:
                    ids = list(seqs)
                    finished += d.postprocess(
                        list(seqs.values()),
                        ScheduledBatchOutput(ids, [(43,)] * len(ids), None, None, None),
                        batch=batch,
                    )
            if dw._write_queue:
                post_write_requests(dw, monkeypatch)
            p._update_from_kv_xfer_finished(pw.get_finished())
            out = p.schedule()
            if out is None:
                continue
            batch, seqs = out
            pw.start_load_kv(batch.connector_meta_output)
            for sid, seq in seqs.items():
                if sid not in started:
                    # P never takes a source D has not reserved pages for.
                    assert peer[sid].block_table
                    started.append(sid)
                state = pw._chunked_prefills[seq.kv_transfer_params["transfer_id"]]
                publish(state, int(batch.context_lens[0]))
                pw._dispatch_ready_chunked_requests()
                p.postprocess(
                    [seq],
                    ScheduledBatchOutput([sid], [(42,)], None, None, None),
                    batch=batch,
                )
                pw.start_load_kv(p.kv_connector.build_connector_meta())
    finally:
        pool.shutdown(wait=True)
    # Order on D is its own (B first); P followed it.
    assert started == [pb.id, pa.id]
    assert [s.id for s in finished] == [db.id, da.id]
    assert pw._notify_transfer_result.call_count == 2  # `_wire` asserts success
    assert da.kv_transfer_params["first_token_id"] == 42
    assert p.block_manager.kv.num_free == 3
    assert d.block_manager.kv.num_free == d_blocks
    assert not p.deferred_free_blocks and not d.deferred_free_blocks
    assert not pw._pending_chunked_requests


# ------------------------------------------------- safety valve for the skip


def _allocate(p, seq, blocks):
    """What the scheduler does once `admission_ready` said yes."""
    from atom.model_engine.sequence import new_block_table

    seq.block_table = new_block_table(blocks)
    p.update_state_after_alloc(seq)
    return p.build_connector_meta()


def _finish_prefill(p, seq, token=42):
    seq.prefill_completed = True
    seq.append_token(token)
    p.request_finished(seq)


@pytest.mark.parametrize("fraction,ready", [("0.5", False), ("1", True)])
def test_cap_sends_early_requests_past_it_through_the_gate(
    monkeypatch, seq_factory, fraction, ready
):
    """Sources allocated before D's report may hold at most the cap's share
    of P's pool (default half). Past it an early request waits for D like a
    gated one, and may skip again once D reached the pinned request."""
    monkeypatch.setenv("ATOM_PD_EARLY_HANDOFF_MAX_PINNED_FRACTION", fraction)
    p = make_scheduler_connector(monkeypatch, True, early=True, num_kvcache_blocks=6)
    a = seq_factory(list(range(10)), kv_transfer_params=p_params("early-a"))
    b = seq_factory(list(range(10)), kv_transfer_params=p_params("early-b"))
    assert p.admission_ready(a)  # 3 pages of 6: at the cap.
    _allocate(p, a, [0, 1, 2])
    assert p._early_pins == {str(a.id): 3}
    assert p.admission_ready(b) is ready
    assert b.id in p._chunked_waiting  # Registered either way.
    _ack(p, a)  # D reached A: its source no longer waits on D.
    assert str(a.id) not in p._early_pins
    assert p.admission_ready(b)


def test_cap_needs_a_sized_pool(monkeypatch):
    """The cap is a share of `num_kvcache_blocks`, which the engine sets
    before it builds the scheduler. An unsized pool is refused rather than
    turning the cap off; without the skip the pool size is not read."""
    with pytest.raises(ValueError, match="num_kvcache_blocks=-1"):
        make_scheduler_connector(monkeypatch, True, early=True, num_kvcache_blocks=-1)
    make_scheduler_connector(
        monkeypatch, True, early=True, skip_gate=False, num_kvcache_blocks=-1
    )
    make_scheduler_connector(monkeypatch, True, early=False, num_kvcache_blocks=-1)


def test_cap_counts_only_sources_allocated_before_d(monkeypatch, seq_factory):
    """A request D already reached is allocated as a gated one would be: it
    takes no room under the cap. A zero cap never skips the gate."""
    monkeypatch.setenv("ATOM_PD_EARLY_HANDOFF_MAX_PINNED_FRACTION", "0")
    p = make_scheduler_connector(monkeypatch, True, early=True, num_kvcache_blocks=6)
    a = seq_factory(list(range(10)), kv_transfer_params=p_params("early-a"))
    assert not p.admission_ready(a)
    _ack(p, a)
    assert p.admission_ready(a)
    _allocate(p, a, [0, 1, 2])
    assert not p._early_pins


def test_undone_allocation_leaves_the_cap(monkeypatch, seq_factory):
    p = make_scheduler_connector(monkeypatch, True, early=True, num_kvcache_blocks=6)
    a = seq_factory(list(range(10)), kv_transfer_params=p_params("early-a"))
    assert p.admission_ready(a)
    from atom.model_engine.sequence import new_block_table

    a.block_table = new_block_table([0, 1, 2])
    p.update_state_after_alloc(a)
    assert str(a.id) in p._early_pins
    p.cancel_preadmission(a)
    assert not p._early_pins and a.id in p._chunked_waiting


def test_reallocation_after_d_reached_p_takes_no_pin(monkeypatch, seq_factory):
    """`admission_ready` re-admits a request that already holds its source
    (e.g. resuming a MultiConnector load), and the scheduler allocates it
    again. Once D has reached P that source no longer waits on D, so the
    second allocation must not count it against the cap again."""
    p = make_scheduler_connector(monkeypatch, True, early=True, num_kvcache_blocks=6)
    a = seq_factory(list(range(10)), kv_transfer_params=p_params("early-a"))
    b = seq_factory(list(range(10)), kv_transfer_params=p_params("early-b"))
    assert p.admission_ready(a)
    _allocate(p, a, [0, 1, 2])
    assert p._early_pins == {str(a.id): 3}
    _ack(p, a)
    assert not p._early_pins
    assert p.admission_ready(a)  # It holds its send claim: re-admitted.
    _allocate(p, a, [0, 1, 2, 3])
    assert not p._early_pins
    assert p.admission_ready(b)  # 3 pages of 6: the cap has room.


def test_worker_lookup_timeout_releases_a_finished_pin(monkeypatch, seq_factory):
    """P itself never cancels a finished early source D has not reached: it
    keeps its pin (and its send claim) until the worker's lookup timeout
    (`ChunkedPrefill.source_safe`) cancels it and reports it safe. The
    scheduler's `send_finished` for that report takes it out of the cap at
    once, so an early request past the cap can skip the gate again."""
    from atom.model_engine.scheduler import ScheduledBatchOutput
    from atom.sampling_params import SamplingParams

    p = _p_scheduler(monkeypatch, num_blocks=6)
    a, b = (
        seq_factory(
            list(range(10)),
            sampling_params=SamplingParams(max_tokens=1),
            kv_transfer_params=p_params(name),
        )
        for name in ("early-a", "early-b")
    )
    p.extend([a])
    for _ in range(3):
        batch, seqs = p.schedule()
        assert list(seqs) == [a.id]
        p.postprocess(
            list(seqs.values()),
            ScheduledBatchOutput([a.id], [(42,)], None, None, None),
            batch=batch,
        )
    handoff = p.kv_connector.build_connector_meta().reqs_to_save[a.id]
    assert not handoff.prefill_aborted
    assert p.kv_connector._early_pins == {str(a.id): 3}
    assert a.id in p.deferred_free_blocks
    assert not p.kv_connector.admission_ready(b)  # 3 + 3 pages > half of 6.
    p._update_from_kv_xfer_finished(
        KVConnectorOutput(
            connector_completions={mc.ConnectorCompletion("pd_source_safe", a.id, True)}
        )
    )
    assert not p.kv_connector._early_pins
    assert not p.deferred_free_blocks and p.block_manager.kv.num_free == 6
    assert p.kv_connector.admission_ready(b)


@pytest.mark.parametrize("as_str", [False, True])
def test_send_finished_releases_the_pin(monkeypatch, seq_factory, as_str):
    """`send_finished` is the release: by the time the scheduler frees the
    blocks (`source_blocks_released`) the pin is already gone. It may name
    the request either way, as for the send claim."""
    p = make_scheduler_connector(monkeypatch, True, early=True, num_kvcache_blocks=6)
    a = seq_factory(list(range(10)), kv_transfer_params=p_params("early-a"))
    assert p.admission_ready(a)
    _allocate(p, a, [0, 1, 2])
    _finish_prefill(p, a)
    assert p._early_pins == {str(a.id): 3} and p.should_defer_free(a)
    p.send_finished(str(a.id) if as_str else a.id)
    assert not p._early_pins and not p.should_defer_free(a)


def test_aborted_early_request_leaves_the_cap(monkeypatch, seq_factory):
    p = make_scheduler_connector(monkeypatch, True, early=True, num_kvcache_blocks=6)
    a = seq_factory(list(range(10)), kv_transfer_params=p_params("early-a"))
    assert p.admission_ready(a)
    _allocate(p, a, [0, 1, 2])
    a.leave_reason = "aborted"
    p.request_finished(a)
    assert not p._early_pins
    assert p.build_connector_meta().reqs_to_save[a.id].prefill_aborted


def test_gate_mode_tracks_no_pins(monkeypatch, seq_factory):
    p = make_scheduler_connector(
        monkeypatch, True, early=True, skip_gate=False, num_kvcache_blocks=6
    )
    a = seq_factory(list(range(10)), kv_transfer_params=p_params("early-a"))
    assert not p.admission_ready(a)
    _ack(p, a)
    assert p.admission_ready(a)
    _allocate(p, a, [0, 1, 2])
    assert not p._early_pins


@pytest.mark.parametrize(
    "name,raw,expected",
    [
        ("ATOM_PD_EARLY_HANDOFF_MAX_PINNED_FRACTION", None, 0.5),
        ("ATOM_PD_EARLY_HANDOFF_MAX_PINNED_FRACTION", "1.5", 0.5),
        ("ATOM_PD_EARLY_HANDOFF_MAX_PINNED_FRACTION", "half", 0.5),
        ("ATOM_PD_EARLY_HANDOFF_MAX_PINNED_FRACTION", "0.25", 0.25),
        ("ATOM_PD_EARLY_HANDOFF_MAX_PINNED_FRACTION", "0", 0.0),
        ("ATOM_PD_EARLY_HANDOFF", None, True),
        ("ATOM_PD_EARLY_HANDOFF", "0", False),
        ("ATOM_PD_EARLY_HANDOFF_SKIP_GATE", None, True),
        ("ATOM_PD_EARLY_HANDOFF_SKIP_GATE", "0", False),
    ],
)
def test_valve_env_defaults_and_validation(monkeypatch, name, raw, expected):
    from atom.utils import envs

    if raw is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, raw)
    assert getattr(envs, name) == expected


def test_lookup_timeout_breaks_the_opposite_order_cycle(monkeypatch, seq_factory):
    """Without the cap (fraction 1) the reviewer's cycle forms for real: P
    allocates and finishes A before D reached it, D reserved B, and P cannot
    admit B. The worker's lookup timeout (PREFILL_LOOKUP_TIMEOUT, shortened
    here) cancels A's source: B then runs on both sides, D's late A write
    request fails and D aborts A. Both pools end empty."""
    from atom.model_engine.scheduler import ScheduledBatchOutput
    from atom.model_engine.sequence import SequenceStatus
    from atom.sampling_params import SamplingParams

    monkeypatch.setenv("ATOM_PD_EARLY_HANDOFF_MAX_PINNED_FRACTION", "1")
    monkeypatch.setattr(mc, "PREFILL_LOOKUP_TIMEOUT", 0.05)
    monkeypatch.setenv("ATOM_PD_FAILED_WRITE_HOLD_S", "0")
    p = _p_scheduler(monkeypatch, num_blocks=3)
    d = _d_scheduler(monkeypatch, num_blocks=3)

    def seqs_for(name):
        ps = seq_factory(
            list(range(10)),
            sampling_params=SamplingParams(max_tokens=1),
            kv_transfer_params=p_params(name),
        )
        ds = seq_factory(
            list(range(10)),
            sampling_params=SamplingParams(max_tokens=2),
            kv_transfer_params=d_params(name),
        )
        return ps, ds

    pa, da = seqs_for("early-a")
    pb, db = seqs_for("early-b")
    p.extend([pa, pb])
    d.extend([db, da])
    pw, _, _src, destinations = idle_producer()
    dw = decode_worker(destinations[0])
    _wire(pw, dw)
    results = []

    def complete(data, success):
        results.append((data["transfer_id"], success))
        dw._record_write_done(
            data["request_id"],
            0,
            0,
            data["write_nonce"],
            success=success,
            handoff=data.get("prefill_handoff"),
        )

    pw._notify_transfer_result.side_effect = complete
    finished, started = [], []
    pool = ThreadPoolExecutor(1)
    pw._send_executor = pool
    try:
        for _ in range(200):
            if db in finished and da.leave_reason == "aborted":
                break
            time.sleep(0.01)
            d._update_from_kv_xfer_finished(dw.get_finished())
            out = d.schedule()
            if out is not None:
                batch, seqs = out
                dw.start_load_kv(batch.connector_meta_output)
                if seqs:
                    ids = list(seqs)
                    finished += d.postprocess(
                        list(seqs.values()),
                        ScheduledBatchOutput(ids, [(43,)] * len(ids), None, None, None),
                        batch=batch,
                    )
            if dw._write_queue:
                post_write_requests(dw, monkeypatch)
            p._update_from_kv_xfer_finished(pw.get_finished())
            out = p.schedule()
            if out is None:
                # The engine's idle drain still ships the connector's meta.
                pw.start_load_kv(p.kv_connector.build_connector_meta())
                continue
            batch, seqs = out
            pw.start_load_kv(batch.connector_meta_output)
            for sid, seq in seqs.items():
                if sid not in started:
                    started.append(sid)
                state = pw._chunked_prefills[seq.kv_transfer_params["transfer_id"]]
                publish(state, int(batch.context_lens[0]))
                pw._dispatch_ready_chunked_requests()
                p.postprocess(
                    [seq],
                    ScheduledBatchOutput([sid], [(42,)], None, None, None),
                    batch=batch,
                )
                pw.start_load_kv(p.kv_connector.build_connector_meta())
    finally:
        pool.shutdown(wait=True)
    # P skipped the gate for A (before D reached it), so the cycle formed.
    assert started == [pa.id, pb.id]
    assert db in finished and db.kv_transfer_params["first_token_id"] == 42
    # D aborted A on the failed receive, which never reached a decode step.
    assert da.status == SequenceStatus.FINISHED and da.leave_reason == "aborted"
    assert da not in finished and not d.waiting and not d.running
    assert sorted(results) == [("early-a", False), ("early-b", True)]
    assert p.block_manager.kv.num_free == 3
    assert d.block_manager.kv.num_free == 3
    assert not p.deferred_free_blocks and not d.deferred_free_blocks
    assert not p.kv_connector._early_pins


@pytest.mark.parametrize(
    "skip_gate,fraction,a_aborted",
    [
        (None, None, True),  # the defaults: SKIP_GATE=1, cap 0.5
        (False, None, False),  # SKIP_GATE=0: P follows D's order
        (None, "0.3", False),  # cap <= 1 - 4/6: A cannot skip either
    ],
)
def test_unequal_prompts_under_the_cap_end_in_a_bounded_abort(
    monkeypatch, seq_factory, skip_gate, fraction, a_aborted
):
    """P has 6 pages, D 4. A's prompt takes 3 pages, B's 4. P queues A then
    B, D admits B then A. At the default cap (half of P's 6 pages) A's 3
    pages fit, so P pins a finished A before D reached it. D's 4 pages hold
    B, and P's 3 free pages cannot take B: neither side can admit its other
    request. The cap rules this out only for a D prompt that fits in P's
    unpinned share. The worker's lookup timeout (PREFILL_LOOKUP_TIMEOUT,
    shortened here) cancels A's source: B then runs on both sides and D
    aborts A on the failed receive. Nothing hangs and both pools end empty.
    With SKIP_GATE=0, or a cap at most 1 - (largest prompt / P's pool), P
    follows D's order and both requests finish."""
    from atom.model_engine.scheduler import ScheduledBatchOutput
    from atom.model_engine.sequence import SequenceStatus
    from atom.sampling_params import SamplingParams

    if fraction is None:
        monkeypatch.delenv("ATOM_PD_EARLY_HANDOFF_MAX_PINNED_FRACTION", raising=False)
    else:
        monkeypatch.setenv("ATOM_PD_EARLY_HANDOFF_MAX_PINNED_FRACTION", fraction)
    monkeypatch.setattr(mc, "PREFILL_LOOKUP_TIMEOUT", 0.05)
    monkeypatch.setenv("ATOM_PD_FAILED_WRITE_HOLD_S", "0")
    p = _p_scheduler(monkeypatch, num_blocks=6, skip_gate=skip_gate)
    d = _d_scheduler(monkeypatch, num_blocks=4)

    def seqs_for(name, prompt):
        ps = seq_factory(
            prompt,
            sampling_params=SamplingParams(max_tokens=1),
            kv_transfer_params=p_params(name),
        )
        ds = seq_factory(
            prompt,
            sampling_params=SamplingParams(max_tokens=2),
            kv_transfer_params=d_params(name),
        )
        return ps, ds

    pa, da = seqs_for("early-a", list(range(10)))  # 3 pages
    pb, db = seqs_for("early-b", list(range(100, 113)))  # 4 pages
    p.extend([pa, pb])
    d.extend([db, da])
    pw, _, _src, destinations = idle_producer()
    dw = decode_worker(destinations[0])
    _wire(pw, dw)
    results = []

    def complete(data, success):
        results.append((data["transfer_id"], success))
        dw._record_write_done(
            data["request_id"],
            0,
            0,
            data["write_nonce"],
            success=success,
            handoff=data.get("prefill_handoff"),
        )

    pw._notify_transfer_result.side_effect = complete
    finished, started = [], []
    stalled = False
    pool = ThreadPoolExecutor(1)
    pw._send_executor = pool
    try:
        for _ in range(300):
            if db in finished and (da in finished or da.leave_reason == "aborted"):
                break
            time.sleep(0.01)
            d._update_from_kv_xfer_finished(dw.get_finished())
            out = d.schedule()
            if out is not None:
                batch, seqs = out
                dw.start_load_kv(batch.connector_meta_output)
                if seqs:
                    ids = list(seqs)
                    finished += d.postprocess(
                        list(seqs.values()),
                        ScheduledBatchOutput(ids, [(43,)] * len(ids), None, None, None),
                        batch=batch,
                    )
            if dw._write_queue:
                post_write_requests(dw, monkeypatch)
            p._update_from_kv_xfer_finished(pw.get_finished())
            # The cycle: D reached B (B is ready on P) but a finished A still
            # holds P's pages, and D's pages all hold B.
            stalled |= (
                pb.id in p.kv_connector._chunked_ready
                and str(pa.id) in p.kv_connector._early_pins
                and p.block_manager.kv.num_free < 4
                and not da.block_table
            )
            out = p.schedule()
            if out is None:
                # The engine's idle drain still ships the connector's meta.
                pw.start_load_kv(p.kv_connector.build_connector_meta())
                continue
            batch, seqs = out
            pw.start_load_kv(batch.connector_meta_output)
            for sid, seq in seqs.items():
                if sid not in started:
                    started.append(sid)
                state = pw._chunked_prefills[seq.kv_transfer_params["transfer_id"]]
                publish(state, int(batch.context_lens[0]))
                pw._dispatch_ready_chunked_requests()
                p.postprocess(
                    [seq],
                    ScheduledBatchOutput([sid], [(42,)], None, None, None),
                    batch=batch,
                )
                pw.start_load_kv(p.kv_connector.build_connector_meta())
    finally:
        pool.shutdown(wait=True)
    assert db in finished and db.kv_transfer_params["first_token_id"] == 42
    if a_aborted:
        # P skipped the gate for A under the cap, so the cycle formed.
        assert stalled and started == [pa.id, pb.id]
        assert da.status == SequenceStatus.FINISHED and da.leave_reason == "aborted"
        assert da not in finished
        assert sorted(results) == [("early-a", False), ("early-b", True)]
    else:
        assert not stalled and started == [pb.id, pa.id]
        assert [s.id for s in finished] == [db.id, da.id]
        assert da.kv_transfer_params["first_token_id"] == 42
        assert sorted(results) == [("early-a", True), ("early-b", True)]
    assert not d.waiting and not d.running
    assert p.block_manager.kv.num_free == 6
    assert d.block_manager.kv.num_free == 4
    assert not p.deferred_free_blocks and not d.deferred_free_blocks
    assert not p.kv_connector._early_pins


def test_late_write_request_before_p_allocation_is_not_expired(monkeypatch):
    """D can reach P's listener while P still queues the request. Registration
    at admission keeps it from being retired as an orphan after the lookup
    timeout."""
    pw, request, _, _ = idle_producer()
    request["transfer_id"] = TRANSFER
    from atom.kv_transfer.disaggregation.types import ConnectorMetadata

    meta = ConnectorMetadata()
    meta.chunked_admissions = {7: TRANSFER}
    pw.start_load_kv(meta)
    with pw._completed_prefills_lock:
        pw._pending_chunked_requests.append(
            (request, mc.time.monotonic() - mc.PREFILL_LOOKUP_TIMEOUT - 1)
        )
    pw._dispatch_ready_chunked_requests()
    assert pw._pending_chunked_requests, "a registered transfer must keep waiting"
    pw._notify_transfer_result.assert_not_called()


# ----------------------------------------------------------- API early answer


@pytest.fixture
def api(monkeypatch):
    from atom.entrypoints.openai import api_server

    monkeypatch.setattr(api_server, "_early_handoff_endpoint_cache", [])
    monkeypatch.setattr("atom.utils.network.get_ip", lambda: "10.0.0.1")
    return api_server


def _engine(api, monkeypatch, *, rank=2, finish_after=False, multinode=False):
    seq = SimpleNamespace(
        id=11,
        prompt_token_ids=[1, 2, 3],
        num_prompt_tokens=3,
        max_tokens=1,
        pd_routed_dp_rank=None,
    )
    calls = SimpleNamespace(aborted=[], params=None, callback=None)

    def preprocess(_prompt, _sampling, stream_callback=None, **kw):
        calls.params = kw["kv_transfer_params"]
        calls.callback = stream_callback
        return seq

    class CoreMgr:
        reports_dp_rank = True

        def add_request(self, seqs):
            if rank is not None:
                seqs[0].pd_routed_dp_rank = rank

        def abort_request(self, seq_id):
            calls.aborted.append(seq_id)

    sub = {
        "kv_connector": "mooncake",
        "kv_role": "kv_producer",
        "handshake_port": 31520,
    }
    engine = SimpleNamespace(
        io_processor=SimpleNamespace(preprocess=preprocess, requests={11: seq}),
        core_mgr=CoreMgr(),
        config=SimpleNamespace(
            kv_transfer_config={"kv_connector": "multi", "connectors": [sub]},
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            kv_cache_block_size=16,
            decode_context_parallel_size=1,
            parallel_config=SimpleNamespace(is_multinode_dp=multinode),
        ),
    )
    monkeypatch.setattr(api, "engine", engine)
    monkeypatch.setattr(api, "_validate_sequence_context_length", lambda *_a: None)
    return seq, calls


def _run(api, params, return_ids=True):
    from atom.sampling_params import SamplingParams

    async def go():
        outs = [
            o
            async for o in api.generate_async(
                [1, 2, 3],
                SamplingParams(max_tokens=1),
                "req",
                kv_transfer_params=params,
                return_token_ids=return_ids,
            )
        ]
        # Checked inside the loop: asyncio.run cancels the detached drain on
        # exit, which a long-lived server loop never does.
        registered = 11 in api.engine.io_processor.requests
        return outs, registered

    return asyncio.run(go())


def test_api_answers_at_dispatch_with_rank_and_keeps_prefill_running(monkeypatch, api):
    monkeypatch.setenv("ATOM_PD_EARLY_HANDOFF", "1")
    _, calls = _engine(api, monkeypatch)
    router_params = {"do_remote_decode": True, "do_remote_prefill": False}
    outs, registered = _run(api, router_params)
    assert len(outs) == 1
    out = outs[0]
    kv = out["kv_transfer_output_meta_info"]
    tid = kv["transfer_id"]
    assert tid.startswith("early-")
    assert calls.params == {
        **router_params,
        "chunked_transfer": True,
        "transfer_id": tid,
        "pd_early_handoff": True,
    }
    assert router_params == {"do_remote_decode": True, "do_remote_prefill": False}
    assert kv == {
        "remote_host": "10.0.0.1",
        "remote_handshake_port": 31520,
        "remote_pp_size": 1,
        "block_size": 16,
        "dcp_size": 1,
        "tp_size": 1,
        "do_remote_prefill": True,
        "do_remote_decode": False,
        "chunked_transfer": True,
        "transfer_id": tid,
        "dp_rank": 2,
    }
    assert out["prompt_token_ids"] == [1, 2, 3]
    assert out["num_tokens_output"] == 0 and out["text"] == ""
    # Not aborted, and its bookkeeping lives until the engine finishes it.
    assert calls.aborted == []
    assert registered


def test_api_drain_releases_request_when_engine_finishes(monkeypatch, api):
    monkeypatch.setenv("ATOM_PD_EARLY_HANDOFF", "1")
    _, calls = _engine(api, monkeypatch)

    async def go():
        agen = api.generate_async(
            [1, 2, 3],
            __import__("atom.sampling_params", fromlist=["x"]).SamplingParams(
                max_tokens=1
            ),
            "req",
            kv_transfer_params={"do_remote_decode": True},
        )
        (out,) = [o async for o in agen]
        assert out["kv_transfer_output_meta_info"]["chunked_transfer"]
        assert 11 in api.engine.io_processor.requests
        calls.callback(
            SimpleNamespace(
                output_tokens=[5],
                logprobs=None,
                finished=True,
                finish_reason="length",
                num_cached_tokens=0,
                kv_transfer_params_output=None,
            )
        )
        await asyncio.gather(*api._early_handoff_tasks)
        assert 11 not in api.engine.io_processor.requests

    asyncio.run(go())
    assert calls.aborted == []


def test_api_fails_closed_without_rank(monkeypatch, api):
    monkeypatch.setenv("ATOM_PD_EARLY_HANDOFF", "1")
    _, calls = _engine(api, monkeypatch, rank=None)
    with pytest.raises(RuntimeError, match="DP rank unknown"):
        _run(api, {"do_remote_decode": True})
    assert calls.aborted == [11]
    assert 11 not in api.engine.io_processor.requests


@pytest.mark.parametrize(
    "flag,params,multinode",
    [
        ("0", {"do_remote_decode": True}, False),  # flag off
        ("1", {"do_remote_decode": True}, True),  # multi-node DP
        ("1", None, False),  # not a PD prefill leg
        ("1", {"do_remote_prefill": True}, False),  # a decode leg
        ("1", {"do_remote_decode": True, "chunked_transfer": True}, False),
    ],
)
def test_api_keeps_legacy_path_otherwise(monkeypatch, api, flag, params, multinode):
    monkeypatch.setenv("ATOM_PD_EARLY_HANDOFF", flag)
    _engine(api, monkeypatch, multinode=multinode)
    assert api._early_handoff_params(params) is None


def test_core_managers_that_do_not_report_the_rank_keep_the_legacy_path(
    monkeypatch, api
):
    """Only `CoreManager` records the dispatched rank. RapidServe's
    `DisaggCoreManager` inherits `_add_prepared_requests` but overrides
    `add_request`, and a front-end worker's `RemoteCoreManager` leaves
    dispatch to the hub: both must keep the legacy path, not fail closed."""
    from aiter_stub import stubbed_aiter

    with stubbed_aiter():
        from atom.entrypoints.openai.frontend_workers import RemoteCoreManager
        from atom.model_engine.engine_core_mgr import CoreManager, DisaggCoreManager

    assert CoreManager.reports_dp_rank is True
    assert DisaggCoreManager.reports_dp_rank is False
    assert RemoteCoreManager.reports_dp_rank is False

    monkeypatch.setenv("ATOM_PD_EARLY_HANDOFF", "1")
    for mgr_cls in (DisaggCoreManager, RemoteCoreManager):
        _engine(api, monkeypatch)
        api.engine.core_mgr = mgr_cls.__new__(mgr_cls)
        api._early_handoff_endpoint_cache.clear()
        assert api._early_handoff_params({"do_remote_decode": True}) is None
    # Positive control: the routing CoreManager takes the early path.
    _engine(api, monkeypatch)
    api.engine.core_mgr = CoreManager.__new__(CoreManager)
    api._early_handoff_endpoint_cache.clear()
    assert api._early_handoff_params({"do_remote_decode": True}) is not None


def test_unset_routed_rank_is_not_pickled(seq_factory):
    import pickle

    seq = seq_factory([1, 2, 3])
    assert seq.pd_routed_dp_rank is None
    assert "pd_routed_dp_rank" not in seq.__getstate__()
    seq.pd_routed_dp_rank = 3  # as CoreManager sets it after the send
    assert pickle.loads(pickle.dumps(seq)).pd_routed_dp_rank == 3


@pytest.mark.parametrize("flag", ["0", "1"])
def test_kv_transfer_info_does_not_advertise_early_mode(monkeypatch, api, flag):
    """The router must keep its legacy relay: no chunked template advertised,
    with the flag on or off. The positive control is the same config with
    `enable_chunked_transfer`, which must advertise it."""
    monkeypatch.setenv("ATOM_PD_EARLY_HANDOFF", flag)
    _engine(api, monkeypatch)
    api.engine.config.parallel_config.data_parallel_size = 4
    info = asyncio.run(api.kv_transfer_info())
    assert info["kv_role"] == "kv_producer"
    assert "chunked_transfer" not in info
    (sub,) = api.engine.config.kv_transfer_config["connectors"]
    sub["enable_chunked_transfer"] = True
    info = asyncio.run(api.kv_transfer_info())
    assert info["chunked_transfer"]["remote_handshake_port"] == 31520


def test_api_early_answer_carries_the_pp_size(monkeypatch, api):
    """A PP>1 P answers with its stage count; D writes to every stage."""
    monkeypatch.setenv("ATOM_PD_EARLY_HANDOFF", "1")
    _engine(api, monkeypatch, rank=0)
    api.engine.config.pipeline_parallel_size = 2
    (out,), _ = _run(api, {"do_remote_decode": True})
    kv = out["kv_transfer_output_meta_info"]
    assert kv["remote_pp_size"] == 2 and kv["dp_rank"] == 0


# --------------------------------------------- CoreManager: dispatched DP rank


def _core_mgr(monkeypatch, *, pp_size=1, engines=1):
    from aiter_stub import stubbed_aiter

    with stubbed_aiter():
        from atom.model_engine import engine_core_mgr as ecm

    mgr = ecm.CoreManager.__new__(ecm.CoreManager)
    mgr.label = "t"
    mgr.pp_size = pp_size
    mgr._seq_id_to_callback = {}
    sent = []
    mgr._send_request = lambda rank, frame: sent.append(rank)
    monkeypatch.setattr(ecm, "_add_frame", lambda seqs: b"")
    monkeypatch.setattr(
        ecm.CoreManager, "_routable_engine_count", engines, raising=False
    )
    monkeypatch.setattr(ecm.tracing, "bind_sequences", lambda _s: None)
    return mgr, sent


@pytest.mark.parametrize("pp_size", [1, 2, 4])
def test_core_manager_reports_rank_0_for_a_single_engine_or_pp_head(
    monkeypatch, pp_size
):
    # PP>1 is one DP rank: requests enter at stage 0 and D reaches every
    # stage through remote_pp_size. Before the fix the PP branch left the
    # rank unset and the API failed every early prefill closed.
    mgr, sent = _core_mgr(monkeypatch, pp_size=pp_size)
    seqs = [SimpleNamespace(id=i, stream_callback=None) for i in range(2)]
    mgr._add_prepared_requests(seqs)
    assert [s.pd_routed_dp_rank for s in seqs] == [0, 0] and sent == [0]


def test_dispatch_to_dp_ranks_records_each_rank_after_the_send():
    import pickle

    from test_dp_load_balance import _FakeSeq, _make_mgr

    class Socket:
        def __init__(self, fail=False):
            self.fail, self.messages = fail, []

        def send_multipart(self, parts, copy=False):
            if self.fail:
                raise RuntimeError("send failed")
            self.messages.append(parts)

    mgr = _make_mgr(4, strategy="least_requests")
    mgr.engine_core_identities = [b"e0", b"e1", b"e2", b"e3"]
    mgr.input_sockets = [Socket() for _ in range(4)]
    seqs = [_FakeSeq(i, 10, data_parallel_rank=r) for i, r in enumerate([2, 0, 3])]
    seqs.append(_FakeSeq(3, 10))  # load-balanced: the only idle rank, 1
    mgr._dispatch_to_dp_ranks(seqs)
    assert [s.pd_routed_dp_rank for s in seqs] == [2, 0, 3, 1]
    for rank, sock in enumerate(mgr.input_sockets):
        (msg,) = sock.messages
        _, sent = pickle.loads(msg[1])
        # Set after pickling: the rank's own copy never carries it.
        assert all(not hasattr(s, "pd_routed_dp_rank") for s in sent)

    # A rank whose send failed (and the ones after it) report no rank, so
    # the API fails those requests closed instead of naming a wrong rank.
    mgr = _make_mgr(4, strategy="least_requests")
    mgr.engine_core_identities = [b"e0", b"e1", b"e2", b"e3"]
    mgr.input_sockets = [Socket(), Socket(fail=True), Socket(), Socket()]
    seqs = [_FakeSeq(10 + r, 10, data_parallel_rank=r) for r in range(4)]
    with pytest.raises(RuntimeError):
        mgr._dispatch_to_dp_ranks(seqs)
    assert [getattr(s, "pd_routed_dp_rank", None) for s in seqs] == [
        0,
        None,
        None,
        None,
    ]

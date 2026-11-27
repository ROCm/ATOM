# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""P/D early handoff, negotiated per request.

P offers it (`/kv_transfer_info` `early_handoff`), D declares what it can
receive (`chunked_receive`), the router asks per request. P answers at DP
dispatch with an NDJSON body whose second line marks the end of the prefill,
and may allocate before D's write request only while its pool provably keeps
room for every registered chunked request.

CPU only. Reuses the chunked-transfer protocol fixtures (real schedulers, real
producer/consumer workers, CPU buffers in place of HBM and the NIC).
"""

import asyncio
import json
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
    monkeypatch, is_producer, *, early, speculative=None, num_kvcache_blocks=100
):
    monkeypatch.setenv("ATOM_PD_EARLY_HANDOFF", "1" if early else "0")
    if speculative is None:
        monkeypatch.delenv("ATOM_PD_EARLY_HANDOFF_SPECULATIVE", raising=False)
    else:
        monkeypatch.setenv(
            "ATOM_PD_EARLY_HANDOFF_SPECULATIVE", "1" if speculative else "0"
        )
    monkeypatch.setattr(mc, "get_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(mc, "get_open_port", lambda: 6400)
    config = SimpleNamespace(
        kv_transfer_config={"kv_role": "kv_producer" if is_producer else "kv_consumer"},
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
    (`early=False`: a router-chunked leg, which always keeps the gate)."""
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
    """What the router relay hands D from P's early answer."""
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


def _ack(p, seq, succeeded=True):
    """D's write request reached every P rank (the worker's report)."""
    p.process_pd_completions(
        KVConnectorOutput(
            connector_completions={
                mc.ConnectorCompletion("pd_admission_ready", seq.id, succeeded)
            }
        )
    )


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


def _prompt(pages, start=0):
    """A prompt of exactly `pages` 4-token pages (the last one partial)."""
    return list(range(start, start + 4 * pages - 2))


# ------------------------------------------------------------ admission rule


def test_flag_off_keeps_existing_config(monkeypatch, seq_factory):
    plain = make_scheduler_connector(monkeypatch, True, early=False)
    assert not plain.enable_chunked_transfer and not plain._speculative
    seq = seq_factory(list(range(10)), kv_transfer_params=p_params())
    with pytest.raises(ValueError, match="enable_chunked_transfer"):
        plain.admission_ready(seq)


def test_flag_enables_chunked_producer_but_not_consumer(monkeypatch):
    assert make_scheduler_connector(
        monkeypatch, True, early=True
    ).enable_chunked_transfer
    d = make_scheduler_connector(monkeypatch, False, early=True)
    assert not d.enable_chunked_transfer and not d._speculative


def test_early_request_on_an_idle_pool_allocates_before_d(monkeypatch, seq_factory):
    """Nothing else registered: P may allocate at once, so P's prefill overlaps
    the router->D->P hop. It is still registered with the worker once."""
    p = make_scheduler_connector(monkeypatch, True, early=True)
    seq = seq_factory(list(range(10)), kv_transfer_params=p_params())
    for _ in range(3):  # The scheduler asks on every pass.
        assert p.admission_ready(seq)
    assert p.build_connector_meta().chunked_admissions == {seq.id: TRANSFER}
    assert p.admission_ready(seq)
    assert p.build_connector_meta().chunked_admissions == {}


def test_a_prompt_over_half_the_pool_still_skips_on_an_idle_pool(
    monkeypatch, seq_factory
):
    """The fixed 50% cap of #3773 sent this to the gate; the rule does not:
    with nothing else registered, a 5-page prompt fits a 6-page pool."""
    p = make_scheduler_connector(monkeypatch, True, early=True, num_kvcache_blocks=6)
    big = seq_factory(_prompt(5), kv_transfer_params=p_params("early-big"))
    assert p.admission_ready(big)


def test_router_chunked_leg_keeps_the_gate(monkeypatch, seq_factory):
    """A router-chunked leg (no `pd_early_handoff`): D got the request at the
    same time as P and may reserve requests P has not registered yet, so the
    rule's premise does not hold and it waits for D."""
    p = make_scheduler_connector(monkeypatch, True, early=True)
    seq = seq_factory(list(range(10)), kv_transfer_params=p_params(early=False))
    assert not p.admission_ready(seq)
    _ack(p, seq)
    assert p.admission_ready(seq)


def test_speculative_off_keeps_the_gate(monkeypatch, seq_factory):
    p = make_scheduler_connector(monkeypatch, True, early=True, speculative=False)
    seq = seq_factory(list(range(10)), kv_transfer_params=p_params())
    assert not p.admission_ready(seq)
    _ack(p, seq)
    assert p.admission_ready(seq)
    _allocate(p, seq, [0, 1, 2])
    assert not p._spec_pins


@pytest.mark.parametrize(
    "pool,a_ready,b_ready",
    [
        (6, False, False),  # 3 + 4 > 6 and 4 + 3 > 6: both wait for D
        (7, True, True),  # 3 + 4 <= 7
    ],
)
def test_room_is_kept_for_the_largest_other_registered_request(
    monkeypatch, seq_factory, pool, a_ready, b_ready
):
    p = make_scheduler_connector(monkeypatch, True, early=True, num_kvcache_blocks=pool)
    a = seq_factory(_prompt(3), kv_transfer_params=p_params("early-a"))
    b = seq_factory(_prompt(4, 100), kv_transfer_params=p_params("early-b"))
    p.admission_ready(a)  # Both registered, as `Scheduler.add` does.
    p.admission_ready(b)
    assert p.admission_ready(a) is a_ready
    assert p.admission_ready(b) is b_ready
    # D reached B: B no longer needs room kept by anybody else's rule.
    _ack(p, b)
    assert p.admission_ready(b)


def test_pins_count_until_d_reaches_p(monkeypatch, seq_factory):
    """A source allocated before D waits on D: it counts against the pool
    until D's report, then it is an ordinary allocation."""
    p = make_scheduler_connector(monkeypatch, True, early=True, num_kvcache_blocks=7)
    a = seq_factory(_prompt(3), kv_transfer_params=p_params("early-a"))
    b = seq_factory(_prompt(4, 100), kv_transfer_params=p_params("early-b"))
    c = seq_factory(_prompt(1, 200), kv_transfer_params=p_params("early-c"))
    for s in (a, b, c):
        p.admission_ready(s)
    assert p.admission_ready(a)  # 0 + 3 + 4 <= 7
    _allocate(p, a, [0, 1, 2])
    assert p._spec_pins == {str(a.id): 3}
    assert not p.admission_ready(c)  # 3 pinned + 1 + room for B (4) > 7
    assert not p.admission_ready(b)  # 3 pinned + 4 + room for C (1) > 7
    _ack(p, a)  # D reached A: its source no longer waits on D.
    assert not p._spec_pins
    assert p.admission_ready(c) and p.admission_ready(b)


def test_allocated_and_aborted_requests_leave_the_waiting_set(monkeypatch, seq_factory):
    from atom.model_engine.sequence import SequenceStatus

    p = make_scheduler_connector(monkeypatch, True, early=True, num_kvcache_blocks=7)
    a = seq_factory(_prompt(3), kv_transfer_params=p_params("early-a"))
    big = seq_factory(_prompt(5, 100), kv_transfer_params=p_params("early-big"))
    p.admission_ready(a)
    p.admission_ready(big)
    assert not p.admission_ready(a)  # 3 + 5 > 7
    big.status = SequenceStatus.ABORTED
    big.leave_reason = "aborted"
    p.request_finished(big)  # leaves `_chunked_waiting`; its heap entry is stale
    assert p._largest_waiting_pages(a) == 0
    assert p.admission_ready(a)


def test_waiting_heap_stays_bounded(monkeypatch, seq_factory):
    p = make_scheduler_connector(monkeypatch, True, early=True)
    for i in range(300):
        s = seq_factory(_prompt(1, i * 10), kv_transfer_params=p_params(f"x{i}"))
        p.admission_ready(s)
        _ack(p, s)
        _allocate(p, s, [0])
    assert len(p._waiting_pages) <= 2 * len(p._chunked_waiting) + 65


def test_speculation_needs_a_sized_pool(monkeypatch):
    with pytest.raises(ValueError, match="sized KV pool"):
        make_scheduler_connector(monkeypatch, True, early=True, num_kvcache_blocks=-1)


def test_allocation_after_d_reached_p_takes_no_pin(monkeypatch, seq_factory):
    """D reached P first: the source never waits on D, so it is an ordinary
    allocation and keeps no room from later early requests."""
    p = make_scheduler_connector(monkeypatch, True, early=True, num_kvcache_blocks=6)
    a = seq_factory(_prompt(3), kv_transfer_params=p_params("early-a"))
    b = seq_factory(_prompt(3, 100), kv_transfer_params=p_params("early-b"))
    p.admission_ready(a)
    _ack(p, a)
    assert p.admission_ready(a)
    _allocate(p, a, [0, 1, 2])
    assert not p._spec_pins
    assert p.admission_ready(b)  # 0 pinned + 3 <= 6


def test_reallocation_after_d_reached_p_takes_no_pin(monkeypatch, seq_factory):
    p = make_scheduler_connector(monkeypatch, True, early=True, num_kvcache_blocks=6)
    a = seq_factory(_prompt(3), kv_transfer_params=p_params("early-a"))
    assert p.admission_ready(a)
    _allocate(p, a, [0, 1, 2])
    assert p._spec_pins == {str(a.id): 3}
    _ack(p, a)
    assert not p._spec_pins
    assert p.admission_ready(a)  # It holds its send claim: re-admitted.
    _allocate(p, a, [0, 1, 2, 3])
    assert not p._spec_pins


def test_undone_allocation_releases_the_pin(monkeypatch, seq_factory):
    p = make_scheduler_connector(monkeypatch, True, early=True, num_kvcache_blocks=6)
    a = seq_factory(_prompt(3), kv_transfer_params=p_params("early-a"))
    assert p.admission_ready(a)
    from atom.model_engine.sequence import new_block_table

    a.block_table = new_block_table([0, 1, 2])
    p.update_state_after_alloc(a)
    assert str(a.id) in p._spec_pins
    p.cancel_preadmission(a)
    assert not p._spec_pins and a.id in p._chunked_waiting
    assert p._largest_waiting_pages(seq_factory([1])) == 3  # back in the heap


@pytest.mark.parametrize("as_str", [False, True])
def test_send_finished_releases_the_pin(monkeypatch, seq_factory, as_str):
    p = make_scheduler_connector(monkeypatch, True, early=True, num_kvcache_blocks=6)
    a = seq_factory(_prompt(3), kv_transfer_params=p_params("early-a"))
    assert p.admission_ready(a)
    _allocate(p, a, [0, 1, 2])
    _finish_prefill(p, a)
    assert p._spec_pins == {str(a.id): 3} and p.should_defer_free(a)
    p.send_finished(str(a.id) if as_str else a.id)
    assert not p._spec_pins and not p.should_defer_free(a)


def test_aborted_early_request_releases_the_pin(monkeypatch, seq_factory):
    p = make_scheduler_connector(monkeypatch, True, early=True, num_kvcache_blocks=6)
    a = seq_factory(_prompt(3), kv_transfer_params=p_params("early-a"))
    assert p.admission_ready(a)
    _allocate(p, a, [0, 1, 2])
    a.leave_reason = "aborted"
    p.request_finished(a)
    assert not p._spec_pins
    assert p.build_connector_meta().reqs_to_save[a.id].prefill_aborted


@pytest.mark.parametrize("speculative", [False, True])
def test_early_request_whose_d_never_comes_expires_without_a_source(
    monkeypatch, seq_factory, speculative
):
    """D never arrives and P never allocated (gate, or no room): the
    admission timeout aborts it before it takes a source."""
    from atom.model_engine.sequence import SequenceStatus

    p = make_scheduler_connector(
        monkeypatch, True, early=True, speculative=speculative, num_kvcache_blocks=3
    )
    other = seq_factory(_prompt(3, 100), kv_transfer_params=p_params("early-o"))
    seq = seq_factory(_prompt(3), kv_transfer_params=p_params())
    clock = [100.0]
    monkeypatch.setattr(mc.time, "monotonic", lambda: clock[0])
    p.admission_ready(other)
    assert not p.admission_ready(seq)  # 3 + 3 > 3 either way
    clock[0] += p._chunked_admission_timeout_s
    assert not p.admission_ready(seq)
    assert seq.status == SequenceStatus.ABORTED
    assert not seq.block_table and seq.id not in p._chunked_waiting


@pytest.mark.parametrize(
    "name,raw,expected",
    [
        ("ATOM_PD_EARLY_HANDOFF", None, False),
        ("ATOM_PD_EARLY_HANDOFF", "1", True),
        ("ATOM_PD_EARLY_HANDOFF_SPECULATIVE", None, True),
        ("ATOM_PD_EARLY_HANDOFF_SPECULATIVE", "0", False),
    ],
)
def test_env_defaults(monkeypatch, name, raw, expected):
    from atom.utils import envs

    if raw is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, raw)
    assert getattr(envs, name) == expected


# ------------------------------------------- real P and D schedulers and workers


def _p_scheduler(monkeypatch, num_blocks=6, speculative=None):
    from conftest import MockConfig

    from atom.model_engine.scheduler import Scheduler

    p = Scheduler(MockConfig(num_kvcache_blocks=num_blocks, max_num_batched_tokens=4))
    p.kv_connector = make_scheduler_connector(
        monkeypatch,
        True,
        early=True,
        speculative=speculative,
        num_kvcache_blocks=num_blocks,
    )
    return p


def _d_scheduler(monkeypatch, num_blocks=6):
    from conftest import MockConfig

    from atom.model_engine.scheduler import Scheduler

    d = Scheduler(MockConfig(num_kvcache_blocks=num_blocks, max_num_batched_tokens=4))
    d.kv_connector = make_scheduler_connector(monkeypatch, False, early=True)
    return d


def _wire(pw, dw, results=None):
    import msgpack

    def receive(_addr, message):
        assert message[0] == mc.MSG_WRITE_REQUEST
        dispatch(pw, msgpack.loads(message[1]))

    def complete(data, success):
        if results is None:
            assert success
        else:
            results.append((data["transfer_id"], success))
        dw._record_write_done(
            data["request_id"],
            0,
            0,
            data["write_nonce"],
            success=success,
            handoff=data.get("prefill_handoff"),
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


def _pair(seq_factory, name, prompt):
    from atom.sampling_params import SamplingParams

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


def _drive(p, d, pw, dw, monkeypatch, done, rounds=300, watch=None):
    """Step D and P alternately until `done()`; returns (finished on D, the
    order P started requests in)."""
    from atom.model_engine.scheduler import ScheduledBatchOutput

    finished, started = [], []
    pool = ThreadPoolExecutor(1)
    pw._send_executor = pool
    try:
        for _ in range(rounds):
            if done(finished):
                break
            time.sleep(0.01)  # let the send thread write what was published
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
            if watch is not None:
                watch()
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
    return finished, started


@pytest.mark.parametrize("forwards_before_d", [1, 3])
def test_p_prefills_before_d_arrives_and_d_receives_every_page(
    monkeypatch, seq_factory, forwards_before_d
):
    """An idle P allocates and prefills at once (3 = the whole prefill, the
    handoff included, is done before D arrives). D's reader starts from block
    0: every page lands byte-identical, D gets the first token, and P frees its
    source only once D's reader finished."""
    import ctypes

    from atom.model_engine.scheduler import ScheduledBatchOutput

    p = _p_scheduler(monkeypatch)
    d = _d_scheduler(monkeypatch)
    ps, ds = _pair(seq_factory, TRANSFER, list(range(10)))
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
            assert ps.block_table and not p._is_preemptable(ps)
            assert pw.transfer_engine.writes.empty()
        assert p.kv_connector._spec_pins == {str(ps.id): 3}
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
    assert not ps.block_table and not p.deferred_free_blocks
    assert not p.kv_connector._chunked_waiting and not p.kv_connector._spec_pins
    d._update_from_kv_xfer_finished(dw.get_finished())
    batch, seqs = d.schedule()
    assert list(seqs) == [ds.id]
    assert ds.kv_transfer_params["first_token_id"] == 42
    sent = []
    while not pw.transfer_engine.writes.empty():
        sources, dsts, sizes = pw.transfer_engine.writes.get_nowait()
        sent += list(zip(sources, dsts, sizes, strict=True))
    base_src, base_dst = ctypes.addressof(src), ctypes.addressof(destinations[0])
    assert sum(size for _, _, size in sent) == 12
    for s, t, size in sent:
        assert bytes(src[s - base_src : s - base_src + size]) == bytes(
            destinations[0][t - base_dst : t - base_dst + size]
        )


@pytest.mark.parametrize("byte_limit", [False, True])
def test_opposite_admission_order_progresses(monkeypatch, seq_factory, byte_limit):
    """Copilot's / valar-cop's cycle: room for one prompt per side; P queues A
    then B, D admits B then A. The rule keeps room for B while B is
    registered, so P does not pin A before D: it follows D's order, both
    finish, nothing is cancelled."""
    from atom.kv_transfer.disaggregation.pd_admission import PdAdmissionController

    p = _p_scheduler(monkeypatch, num_blocks=3)
    d_blocks = 6 if byte_limit else 3
    d = _d_scheduler(monkeypatch, num_blocks=d_blocks)
    if byte_limit:
        d.pd_admission = PdAdmissionController(12, 4, d.kv_connector, d.block_manager)
    pa, da = _pair(seq_factory, "early-a", list(range(10)))
    pb, db = _pair(seq_factory, "early-b", list(range(10)))
    p.extend([pa, pb])
    d.extend([db, da])
    pw, _, _src, destinations = idle_producer()
    dw = decode_worker(destinations[0])
    _wire(pw, dw)
    finished, started = _drive(
        p, d, pw, dw, monkeypatch, lambda f: len(f) == 2, rounds=60
    )
    assert started == [pb.id, pa.id]
    assert [s.id for s in finished] == [db.id, da.id]
    assert pw._notify_transfer_result.call_count == 2  # `_wire` asserts success
    assert p.block_manager.kv.num_free == 3
    assert d.block_manager.kv.num_free == d_blocks
    assert not p.deferred_free_blocks and not d.deferred_free_blocks


def test_unequal_prompts_progress_without_an_abort(monkeypatch, seq_factory):
    """Copilot's unequal case, which #3773's 50% cap could only end in a
    60 s stall and an abort: P has 6 pages, D 4; A takes 3, B 4; P queues A
    then B, D admits B then A. 3 + 4 > 6, so A waits for D: P follows D's
    order and both finish."""
    monkeypatch.setattr(mc, "PREFILL_LOOKUP_TIMEOUT", 0.05)
    p = _p_scheduler(monkeypatch, num_blocks=6)
    d = _d_scheduler(monkeypatch, num_blocks=4)
    pa, da = _pair(seq_factory, "early-a", _prompt(3))
    pb, db = _pair(seq_factory, "early-b", _prompt(4, 100))
    p.extend([pa, pb])
    d.extend([db, da])
    pw, _, _src, destinations = idle_producer()
    dw = decode_worker(destinations[0])
    results = []
    _wire(pw, dw, results)
    finished, started = _drive(
        p,
        d,
        pw,
        dw,
        monkeypatch,
        lambda f: db in f and (da in f or da.leave_reason == "aborted"),
    )
    assert started == [pb.id, pa.id]
    assert [s.id for s in finished] == [db.id, da.id]
    assert sorted(results) == [("early-a", True), ("early-b", True)]
    assert p.block_manager.kv.num_free == 6 and d.block_manager.kv.num_free == 4
    assert not p.kv_connector._spec_pins


def test_late_larger_request_reordered_by_d_ends_in_a_bounded_abort(
    monkeypatch, seq_factory
):
    """The rule's remaining gap, kept visible. A (3 pages) is the only
    registered request, so P pins it before D. Only then does B (4 pages)
    reach P, and D reserves B *ahead of* A (in production D got A's answer
    first, so this needs D to reorder). P's 3 free pages cannot take B and
    D's 4 pages hold B. The worker's lookup timeout (shortened here) cancels
    A's source: B completes, D aborts A, both pools end empty."""
    from atom.model_engine.sequence import SequenceStatus

    monkeypatch.setattr(mc, "PREFILL_LOOKUP_TIMEOUT", 0.05)
    monkeypatch.setenv("ATOM_PD_FAILED_WRITE_HOLD_S", "0")
    p = _p_scheduler(monkeypatch, num_blocks=6)
    d = _d_scheduler(monkeypatch, num_blocks=4)
    pa, da = _pair(seq_factory, "early-a", _prompt(3))
    pb, db = _pair(seq_factory, "early-b", _prompt(4, 100))
    p.extend([pa])
    pw, _, _src, destinations = idle_producer()
    dw = decode_worker(destinations[0])
    results = []
    _wire(pw, dw, results)
    arrived = []

    def b_arrives_after_a_is_pinned():
        if not arrived and str(pa.id) in p.kv_connector._spec_pins:
            arrived.append(True)
            p.extend([pb])
            d.extend([db, da])

    finished, started = _drive(
        p,
        d,
        pw,
        dw,
        monkeypatch,
        lambda f: db in f and da.leave_reason == "aborted",
        watch=b_arrives_after_a_is_pinned,
    )
    assert started == [pa.id, pb.id]
    assert db in finished and da.status == SequenceStatus.FINISHED
    assert da.leave_reason == "aborted"
    assert sorted(results) == [("early-a", False), ("early-b", True)]
    assert p.block_manager.kv.num_free == 6 and d.block_manager.kv.num_free == 4
    assert not p.kv_connector._spec_pins


def test_d_never_arrives_for_a_finished_source(monkeypatch, seq_factory):
    """P prefilled an early request before D; D never comes (router or D died
    after the early answer). P keeps the pin and send claim until the worker's
    lookup timeout reports the source safe; then the pool is whole again."""
    from atom.model_engine.scheduler import ScheduledBatchOutput
    from atom.sampling_params import SamplingParams

    p = _p_scheduler(monkeypatch, num_blocks=6)
    a = seq_factory(
        _prompt(3),
        sampling_params=SamplingParams(max_tokens=1),
        kv_transfer_params=p_params("early-a"),
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
    assert p.kv_connector._spec_pins == {str(a.id): 3}
    assert a.id in p.deferred_free_blocks
    p._update_from_kv_xfer_finished(
        KVConnectorOutput(
            connector_completions={mc.ConnectorCompletion("pd_source_safe", a.id, True)}
        )
    )
    assert not p.kv_connector._spec_pins
    assert not p.deferred_free_blocks and p.block_manager.kv.num_free == 6


def test_late_write_request_before_p_allocation_is_not_expired(monkeypatch):
    """D can reach P's listener while P still queues the request (gated
    early request). Registration keeps it from being retired as an orphan."""
    from atom.kv_transfer.disaggregation.types import ConnectorMetadata

    pw, request, _, _ = idle_producer()
    request["transfer_id"] = TRANSFER
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


# ------------------------------- a retired transfer releases its early source


def test_worker_reports_a_retired_transfers_source_safe():
    """A save for a transfer the worker already retired creates no source
    (no reader can acquire it), so the worker reports the send claim safe;
    `continue`-ing silently left the finished request's blocks and state slot
    deferred forever (c2048 hang)."""
    from atom.kv_transfer.disaggregation.types import ConnectorMetadata

    pw, _, _, _ = idle_producer()
    with pw._completed_prefills_lock:
        pw._retire_chunked_transfer(TRANSFER)
    meta = ConnectorMetadata()
    meta.add_new_req_to_save(
        7,
        [0, 1, 2],
        {"transfer_id": TRANSFER, "chunked_transfer": True, "num_prompt_tokens": 10},
    )
    pw.start_load_kv(meta)
    assert TRANSFER not in pw._chunked_prefills
    out = pw.get_finished()
    assert mc.ConnectorCompletion("pd_source_safe", 7, True) in (
        out.connector_completions
    )


def test_failed_report_for_an_allocated_early_source_aborts_it(
    monkeypatch, seq_factory
):
    """`pd_admission_ready(False)` for a source allocated before D (not in
    `_chunked_waiting` any more) used to be ignored, so P prefilled for
    nobody. Now the scheduler is asked to abort it."""
    p = make_scheduler_connector(monkeypatch, True, early=True, num_kvcache_blocks=6)
    a = seq_factory(_prompt(3), kv_transfer_params=p_params("early-a"))
    b = seq_factory(_prompt(1, 100), kv_transfer_params=p_params("early-b"))
    for s in (a, b):
        assert p.admission_ready(s)
        _allocate(p, s, [0, 1, 2] if s is a else [3])
    assert set(p._early_sources) == {str(a.id), str(b.id)}
    _ack(p, a, succeeded=False)
    _ack(p, b)  # D reached B: not aborted, no longer tracked
    assert p.take_sources_to_abort() == [a.id]
    assert p.take_sources_to_abort() == []
    assert not p._early_sources and not p._spec_pins


def _p_scheduler_with_state_slots(monkeypatch, num_blocks, state_slots):
    from conftest import MockConfig

    from atom.model_engine.kv_block import STATE_SLOT_CLASS
    from atom.model_engine.scheduler import Scheduler

    p = Scheduler(
        MockConfig(
            num_kvcache_blocks=num_blocks,
            max_num_batched_tokens=4,
            pool_entries={STATE_SLOT_CLASS: state_slots},
        )
    )
    p.kv_connector = make_scheduler_connector(
        monkeypatch, True, early=True, num_kvcache_blocks=num_blocks
    )
    return p


@pytest.mark.parametrize("per_req_state", [False, True])
def test_p_stall_past_the_lookup_timeout_releases_the_early_source(
    monkeypatch, seq_factory, per_req_state
):
    """The c2048 P0 wedge, end to end. P allocates an early source before D
    (idle pool), then that step stalls past PREFILL_LOOKUP_TIMEOUT before its
    metadata (the registration and the save) reaches the worker. D's write
    request queues meanwhile; out of the stall, `get_finished` runs before
    the metadata and retires it as an orphan, telling D it failed. The
    registration and the save then arrive for a retired transfer.

    P must stop the prefill, and the source -- its blocks and its state slot
    -- must come back (before the fix: `released 0 of N`, `state: 128/128`
    and Running 0 for good). D must abort the receive and free its pages."""
    from atom.model_engine.scheduler import ScheduledBatchOutput
    from atom.model_engine.sequence import SequenceStatus
    from atom.sampling_params import SamplingParams

    monkeypatch.setattr(mc, "PREFILL_LOOKUP_TIMEOUT", 0.05)
    monkeypatch.setenv("ATOM_PD_FAILED_WRITE_HOLD_S", "0")
    p = _p_scheduler_with_state_slots(monkeypatch, 6, 1 if per_req_state else 0)
    d = _d_scheduler(monkeypatch)
    ps = seq_factory(
        _prompt(3),
        sampling_params=SamplingParams(max_tokens=1),
        kv_transfer_params=p_params(TRANSFER),
        has_per_req_cache=per_req_state,
    )
    _, ds = _pair(seq_factory, TRANSFER, _prompt(3))
    pw, _, _src, destinations = idle_producer()
    dw = decode_worker(destinations[0])
    results = []
    _wire(pw, dw, results)
    p_free, d_free = p.block_manager.kv.num_free, d.block_manager.kv.num_free

    p.extend([ps])
    batch, seqs = p.schedule()
    assert list(seqs) == [ps.id] and str(ps.id) in p.kv_connector._spec_pins
    if per_req_state:
        assert ps.state_slot >= 0 and not p.block_manager.state.has_free(1)
    stalled_meta = batch.connector_meta_output  # this step stalls
    # D got P's early answer and reaches P's listener during the stall.
    d.extend([ds])
    d_batch, _ = d.schedule()
    dw.start_load_kv(d_batch.connector_meta_output)
    post_write_requests(dw, monkeypatch)
    assert pw._pending_chunked_requests
    time.sleep(0.1)  # the stall outlasts the (shortened) lookup timeout
    # Out of the stall: get_finished runs before the step's metadata.
    p._update_from_kv_xfer_finished(pw.get_finished())
    assert results == [(TRANSFER, False)]  # retired as an orphan, D told
    pw.start_load_kv(stalled_meta)  # registration + save of a retired transfer
    p._update_from_kv_xfer_finished(pw.get_finished())
    assert ps.status == SequenceStatus.ABORTED, "P must not prefill for nobody"
    # The stalled forward returns; the aborted request leaves.
    p.postprocess(
        [ps], ScheduledBatchOutput([ps.id], [(42,)], None, None, None), batch=batch
    )
    pw.start_load_kv(p.kv_connector.build_connector_meta())
    p._update_from_kv_xfer_finished(pw.get_finished())
    assert ps.status == SequenceStatus.FINISHED and ps.leave_reason == "aborted"
    assert not p.deferred_free_blocks and not ps.block_table
    assert p.block_manager.kv.num_free == p_free
    if per_req_state:
        assert p.block_manager.state.has_free(1)
    conn = p.kv_connector
    assert not conn._spec_pins and not conn._early_sources
    assert not conn._awaiting_send and not conn._chunked_waiting
    assert not pw._chunked_prefills and not pw._chunked_local_ids
    # D gives up on the receive and frees its reservation.
    d._update_from_kv_xfer_finished(dw.get_finished())
    assert ds.leave_reason == "aborted"
    assert d.block_manager.kv.num_free == d_free

    # The rank is not wedged: the next early request allocates (its state
    # slot included) and completes with D.
    ps2, ds2 = _pair(seq_factory, "early-next", _prompt(3, 100))
    ps2.has_per_req_cache = per_req_state
    p.extend([ps2])
    d.extend([ds2])
    finished, _ = _drive(p, d, pw, dw, monkeypatch, lambda f: ds2 in f)
    assert ds2 in finished and ("early-next", True) in results
    assert not p.deferred_free_blocks and p.block_manager.kv.num_free == p_free


# ----------------------------------- answering when P schedules the request


@pytest.mark.parametrize("at_schedule", [False, True])
def test_answer_waits_for_allocation_or_the_gate(monkeypatch, seq_factory, at_schedule):
    """ATOM_PD_EARLY_HANDOFF_AT_SCHEDULE: an early request is answered when
    its source is allocated, or at once when the rule sends it to wait for D
    (D must learn of it, or it would wait on itself). Queuing on P's own
    capacity answers nothing. Each request is answered once."""
    monkeypatch.setenv("ATOM_PD_EARLY_HANDOFF_AT_SCHEDULE", "1" if at_schedule else "0")
    p = make_scheduler_connector(monkeypatch, True, early=True, num_kvcache_blocks=7)
    a = seq_factory(_prompt(3), kv_transfer_params=p_params("early-a"))
    b = seq_factory(_prompt(4, 100), kv_transfer_params=p_params("early-b"))
    c = seq_factory(_prompt(1, 200), kv_transfer_params=p_params("early-c"))
    for s in (a, b):
        assert p.admission_ready(s)  # registered by `add`; 3 + 4 <= 7
    assert p.take_early_answers() == []
    _allocate(p, a, [0, 1, 2])
    assert p.admission_ready(a)  # re-asked: no second answer
    assert p.take_early_answers() == ([a.id] if at_schedule else [])
    assert not p.admission_ready(c)  # 3 pinned + 1 + room for B (4) > 7: gate
    assert not p.admission_ready(c)
    assert p.take_early_answers() == ([c.id] if at_schedule else [])
    _ack(p, c)
    assert p.admission_ready(c)
    _allocate(p, c, [3])  # D reached it first; it was answered already
    assert p.take_early_answers() == []
    c.leave_reason = "aborted"
    p.request_finished(c)
    assert c.id not in p._answered


def test_router_chunked_leg_is_never_answered_by_the_scheduler(
    monkeypatch, seq_factory
):
    monkeypatch.setenv("ATOM_PD_EARLY_HANDOFF_AT_SCHEDULE", "1")
    p = make_scheduler_connector(monkeypatch, True, early=True)
    seq = seq_factory(list(range(10)), kv_transfer_params=p_params(early=False))
    assert not p.admission_ready(seq)
    _ack(p, seq)
    _allocate(p, seq, [0, 1, 2])
    assert p.take_early_answers() == []


def test_engine_sends_each_answer_as_a_tokenless_stream_output(monkeypatch):
    import queue

    from aiter_stub import stubbed_aiter

    with stubbed_aiter():
        from atom.model_engine.engine_core import send_early_handoff_answers
    from atom.model_engine.request import PD_EARLY_SCHEDULED_KEY

    answers = [[3, 5], []]
    core = SimpleNamespace()
    core.scheduler = SimpleNamespace(
        kv_connector=SimpleNamespace(take_early_answers=lambda: answers.pop(0))
    )
    core.output_queue = queue.Queue()
    send_early_handoff_answers(core)
    send_early_handoff_answers(core)  # nothing due: nothing sent
    kind, outputs = core.output_queue.get_nowait()
    assert core.output_queue.empty() and kind == "STREAM"
    assert [rid for rid, _ in outputs] == [3, 5]
    for rid, out in outputs:
        assert out.request_id == rid and out.output_tokens == []
        assert not out.finished
        assert out.kv_transfer_params_output == {PD_EARLY_SCHEDULED_KEY: True}
    core.scheduler = SimpleNamespace(kv_connector=None)
    send_early_handoff_answers(core)  # no connector: no-op
    assert core.output_queue.empty()


def test_multi_connector_forwards_answers_and_aborts():
    from atom.kv_transfer.disaggregation.multi.multi_connector import (
        MultiConnectorScheduler,
    )

    multi = MultiConnectorScheduler.__new__(MultiConnectorScheduler)
    multi._connectors = [
        SimpleNamespace(
            take_early_answers=lambda: [1, 2], take_sources_to_abort=lambda: [9]
        ),
        SimpleNamespace(),  # an offload tier keeps neither
    ]
    assert multi.take_early_answers() == [1, 2]
    assert multi.take_sources_to_abort() == [9]


def test_at_schedule_answers_progress_with_real_schedulers(monkeypatch, seq_factory):
    """D only learns of a request from P's answer, which now leaves when P
    schedules it. P has room for both prompts but one prefill chunk a step,
    so B is answered only once P gets to it; both complete, and P's pool and
    D's pool end whole."""
    monkeypatch.setenv("ATOM_PD_EARLY_HANDOFF_AT_SCHEDULE", "1")
    p = _p_scheduler(monkeypatch, num_blocks=6)
    d = _d_scheduler(monkeypatch, num_blocks=6)
    pa, da = _pair(seq_factory, "early-a", _prompt(3))
    pb, db = _pair(seq_factory, "early-b", _prompt(3, 100))
    d_of = {pa.id: da, pb.id: db}
    p.extend([pa, pb])
    pw, _, _src, destinations = idle_producer()
    dw = decode_worker(destinations[0])
    _wire(pw, dw)
    answered = []

    def relay_answers():
        for rid in p.kv_connector.take_early_answers():
            answered.append((rid, len(pb.block_table)))
            d.extend([d_of[rid]])

    finished, started = _drive(
        p, d, pw, dw, monkeypatch, lambda f: len(f) == 2, watch=relay_answers
    )
    assert [rid for rid, _ in answered] == [pa.id, pb.id]
    assert answered[0][1] == 0, "B must not be answered before P schedules it"
    assert started == [pa.id, pb.id]
    assert {s.id for s in finished} == {da.id, db.id}
    assert p.block_manager.kv.num_free == 6 and d.block_manager.kv.num_free == 6
    assert not p.deferred_free_blocks and not d.deferred_free_blocks


@pytest.mark.parametrize("pool", [3, 6])
def test_at_schedule_gate_answers_without_deadlock(monkeypatch, seq_factory, pool):
    """Copilot's opposite-order case under answer-at-schedule: with room for
    one prompt the rule gates both requests, and a gated request is answered
    at once, so D can reserve it and P follows D's order. With room for both,
    answers follow P's order. Both complete either way."""
    monkeypatch.setenv("ATOM_PD_EARLY_HANDOFF_AT_SCHEDULE", "1")
    p = _p_scheduler(monkeypatch, num_blocks=pool)
    d = _d_scheduler(monkeypatch, num_blocks=3)
    pa, da = _pair(seq_factory, "early-a", list(range(10)))
    pb, db = _pair(seq_factory, "early-b", list(range(10)))
    d_of = {pa.id: da, pb.id: db}
    p.extend([pa, pb])
    pw, _, _src, destinations = idle_producer()
    dw = decode_worker(destinations[0])
    _wire(pw, dw)

    def relay_answers():
        # D admits in reverse answer order when both arrive together.
        d.extend([d_of[rid] for rid in reversed(p.kv_connector.take_early_answers())])

    finished, _ = _drive(
        p, d, pw, dw, monkeypatch, lambda f: len(f) == 2, watch=relay_answers
    )
    assert {s.id for s in finished} == {da.id, db.id}
    assert p.block_manager.kv.num_free == pool and d.block_manager.kv.num_free == 3
    assert not p.deferred_free_blocks and not d.deferred_free_blocks


# ----------------------------------------------------- negotiation (P and D API)


@pytest.fixture
def api(monkeypatch):
    from atom.entrypoints.openai import api_server

    monkeypatch.setattr(api_server, "_early_handoff_endpoint_cache", [])
    monkeypatch.setattr("atom.utils.network.get_ip", lambda: "10.0.0.1")
    return api_server


def _config(role="kv_producer", **kw):
    sub = {"kv_connector": "mooncake", "kv_role": role, "handshake_port": 31520}
    cfg = SimpleNamespace(
        kv_transfer_config={"kv_connector": "multi", "connectors": [sub]},
        tensor_parallel_size=kw.get("tp", 1),
        pipeline_parallel_size=kw.get("pp", 1),
        kv_cache_block_size=16,
        decode_context_parallel_size=kw.get("dcp", 1),
        parallel_config=SimpleNamespace(
            is_multinode_dp=kw.get("multinode", False), data_parallel_size=4
        ),
        hf_config=SimpleNamespace(model_type=kw.get("model_type", "deepseek_v3")),
    )
    return cfg, sub


def _engine(api, monkeypatch, *, rank=2, multinode=False, cfg=None):
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

    if cfg is None:
        cfg, _ = _config(multinode=multinode)
    engine = SimpleNamespace(
        io_processor=SimpleNamespace(preprocess=preprocess, requests={11: seq}),
        core_mgr=CoreMgr(),
        config=cfg,
    )
    monkeypatch.setattr(api, "engine", engine)
    monkeypatch.setattr(api, "_validate_sequence_context_length", lambda *_a: None)
    return seq, calls


ROUTER_ASKS = {
    "do_remote_decode": True,
    "do_remote_prefill": False,
    "early_handoff": True,
}


def _finished(calls, reason="length"):
    calls.callback(
        SimpleNamespace(
            output_tokens=[5],
            logprobs=None,
            finished=True,
            finish_reason=reason,
            num_cached_tokens=0,
            kv_transfer_params_output=None,
        )
    )


@pytest.mark.parametrize("flag", ["0", "1"])
def test_p_offers_early_handoff_only_with_the_flag(monkeypatch, api, flag):
    monkeypatch.setenv("ATOM_PD_EARLY_HANDOFF", flag)
    _engine(api, monkeypatch)
    info = asyncio.run(api.kv_transfer_info())
    assert info["kv_role"] == "kv_producer"
    assert "chunked_transfer" not in info and "chunked_receive" not in info
    assert ("early_handoff" in info) is (flag == "1")
    if flag == "1":
        assert info["early_handoff"] == {"version": 1}


def test_router_chunked_mode_takes_precedence(monkeypatch, api):
    monkeypatch.setenv("ATOM_PD_EARLY_HANDOFF", "1")
    _engine(api, monkeypatch)
    (sub,) = api.engine.config.kv_transfer_config["connectors"]
    sub["enable_chunked_transfer"] = True
    info = asyncio.run(api.kv_transfer_info())
    assert info["chunked_transfer"]["remote_handshake_port"] == 31520
    assert "early_handoff" not in info


@pytest.mark.parametrize(
    "shape,accepts",
    [
        ({}, True),
        ({"tp": 8, "dcp": 8, "model_type": "deepseek_v3"}, True),  # GLM/DSA-like DCP
        ({"tp": 8, "dcp": 8, "model_type": "kimi_linear"}, False),  # Kimi-K3 D
        ({"tp": 8, "dcp": 8, "model_type": "deepseek_v4"}, False),  # V4.1 under DCP
        ({"pp": 2}, False),
    ],
)
def test_d_declares_what_it_can_receive_and_refuses_exactly_that(
    monkeypatch, api, shape, accepts
):
    """D's declaration and D's 400 come from the same checks: whatever D
    declares it cannot receive, it also refuses; whatever it declares it can,
    it accepts (from a P whose TP divides D's)."""
    from fastapi import HTTPException

    from atom.model_engine.llm_engine import InputOutputProcessor

    if "model_type" in shape and not accepts:
        # The shapes this test calls per-request-state models must be ones.
        assert shape["model_type"] in InputOutputProcessor._per_req_cache_model_types()
    cfg, _ = _config("kv_consumer", **shape)
    _engine(api, monkeypatch, cfg=cfg)
    info = asyncio.run(api.kv_transfer_info())
    assert info["kv_role"] == "kv_consumer"
    cap = info["chunked_receive"]
    assert cap["accepts"] is accepts
    assert cap["tp_size"] == cfg.tensor_parallel_size
    assert (cap["reason"] is None) is accepts
    params = {"chunked_transfer": True, "do_remote_prefill": True, "remote_tp_size": 1}
    if accepts:
        api._validate_chunked_receive(params)
    else:
        with pytest.raises(HTTPException) as err:
            api._validate_chunked_receive(params)
        assert err.value.status_code == 400


def test_a_d_without_a_mooncake_consumer_declares_no_receive(monkeypatch, api):
    cfg, sub = _config("kv_consumer")
    sub["kv_connector"] = "moriio"
    _engine(api, monkeypatch, cfg=cfg)
    cap = api.chunked_receive_capability(cfg)
    assert cap["accepts"] is False and "Mooncake" in cap["reason"]


@pytest.mark.parametrize(
    "flag,params,multinode",
    [
        ("0", ROUTER_ASKS, False),  # P does not offer it
        ("1", {"do_remote_decode": True}, False),  # router did not ask
        ("1", ROUTER_ASKS, True),  # multi-node DP
        ("1", None, False),  # not a PD prefill leg
        ("1", {"do_remote_prefill": True, "early_handoff": True}, False),
        ("1", {**ROUTER_ASKS, "chunked_transfer": True}, False),
    ],
)
def test_p_answers_after_prefill_unless_asked_and_able(
    monkeypatch, api, flag, params, multinode
):
    monkeypatch.setenv("ATOM_PD_EARLY_HANDOFF", flag)
    _engine(api, monkeypatch, multinode=multinode)
    assert api._early_handoff_params(params) is None


def test_core_managers_that_do_not_report_the_rank_keep_the_legacy_path(
    monkeypatch, api
):
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
        assert api._early_handoff_params(ROUTER_ASKS) is None
        assert "early_handoff" not in asyncio.run(api.kv_transfer_info())
    _engine(api, monkeypatch)
    api.engine.core_mgr = CoreManager.__new__(CoreManager)
    api._early_handoff_endpoint_cache.clear()
    assert api._early_handoff_params(ROUTER_ASKS) is not None


def _generate(api, params, return_ids=True):
    from atom.sampling_params import SamplingParams

    return api.generate_async(
        [1, 2, 3],
        SamplingParams(max_tokens=1),
        "req",
        kv_transfer_params=params,
        return_token_ids=return_ids,
    )


async def _body_lines(response):
    lines = []
    async for chunk in response.body_iterator:
        lines.append(chunk if isinstance(chunk, str) else chunk.decode())
    return lines


def test_p_answers_at_dispatch_and_ends_the_body_when_the_prefill_ends(
    monkeypatch, api
):
    """Line 1 leaves at dispatch with the chunked metadata; line 2 only once
    the engine finished the prefill. A router reading the body to its end
    therefore holds P's load for the whole prefill."""
    monkeypatch.setenv("ATOM_PD_EARLY_HANDOFF", "1")
    _, calls = _engine(api, monkeypatch)

    async def go():
        (out,) = [o async for o in _generate(api, dict(ROUTER_ASKS))]
        kv = out["kv_transfer_output_meta_info"]
        tid = kv["transfer_id"]
        assert tid.startswith("early-")
        assert calls.params == {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "chunked_transfer": True,
            "transfer_id": tid,
            "pd_early_handoff": True,
        }
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
        body = {"kv_transfer_params": kv, "prompt_token_ids": out["prompt_token_ids"]}
        response = api._nonstream_response(body, out)
        assert response.headers[api.EARLY_HANDOFF_HEADER] == "1"
        assert response.media_type == "application/x-ndjson"
        it = response.body_iterator
        line1 = await it.__anext__()
        assert json.loads(line1) == body
        line2 = asyncio.ensure_future(it.__anext__())
        await asyncio.sleep(0.05)
        assert not line2.done(), "line 2 must wait for the end of the prefill"
        assert 11 in api.engine.io_processor.requests
        _finished(calls)
        assert json.loads(await line2) == {
            api.EARLY_HANDOFF_DONE_KEY: {"ok": True, "finish_reason": "length"}
        }
        with pytest.raises(StopAsyncIteration):
            await it.__anext__()
        assert 11 not in api.engine.io_processor.requests

    asyncio.run(go())
    assert calls.aborted == []


def test_p_failure_after_the_early_answer_reaches_the_router(monkeypatch, api):
    monkeypatch.setenv("ATOM_PD_EARLY_HANDOFF", "1")
    _, calls = _engine(api, monkeypatch)

    async def go():
        (out,) = [o async for o in _generate(api, dict(ROUTER_ASKS))]
        response = api._nonstream_response({"x": 1}, out)
        _finished(calls, reason="aborted")
        lines = await _body_lines(response)
        assert json.loads(lines[1])[api.EARLY_HANDOFF_DONE_KEY]["ok"] is False

    asyncio.run(go())


def test_router_hanging_up_aborts_the_prefill(monkeypatch, api):
    """D failed (or the client left) and the router dropped P's body: nobody
    will read this prefill, so P stops computing it."""
    monkeypatch.setenv("ATOM_PD_EARLY_HANDOFF", "1")
    _, calls = _engine(api, monkeypatch)

    async def go():
        (out,) = [o async for o in _generate(api, dict(ROUTER_ASKS))]
        response = api._nonstream_response({"x": 1}, out)
        it = response.body_iterator
        await it.__anext__()
        pending = asyncio.ensure_future(it.__anext__())
        await asyncio.sleep(0.01)
        pending.cancel()  # Starlette cancels the body on http.disconnect
        with pytest.raises(asyncio.CancelledError):
            await pending
        await asyncio.sleep(0)
        assert 11 not in api.engine.io_processor.requests

    asyncio.run(go())
    assert calls.aborted == [11]


def _scheduled(calls):
    from atom.model_engine.request import PD_EARLY_SCHEDULED_KEY

    calls.callback(
        SimpleNamespace(
            output_tokens=[],
            logprobs=None,
            finished=False,
            finish_reason=None,
            num_cached_tokens=0,
            kv_transfer_params_output={PD_EARLY_SCHEDULED_KEY: True},
        )
    )


def test_at_schedule_the_answer_waits_for_the_scheduler(monkeypatch, api):
    """Line 1 leaves only once P's scheduler reports the request scheduled;
    line 2 still waits for the end of the prefill."""
    monkeypatch.setenv("ATOM_PD_EARLY_HANDOFF", "1")
    monkeypatch.setenv("ATOM_PD_EARLY_HANDOFF_AT_SCHEDULE", "1")
    _, calls = _engine(api, monkeypatch)

    async def go():
        gen = _generate(api, dict(ROUTER_ASKS))
        first = asyncio.ensure_future(gen.__anext__())
        await asyncio.sleep(0.05)
        assert not first.done(), "no answer before P schedules the request"
        _scheduled(calls)
        out = await first
        assert out["kv_transfer_output_meta_info"]["dp_rank"] == 2
        response = api._nonstream_response({"x": 1}, out)
        it = response.body_iterator
        await it.__anext__()
        line2 = asyncio.ensure_future(it.__anext__())
        await asyncio.sleep(0.02)
        assert not line2.done()
        _finished(calls)
        assert json.loads(await line2)[api.EARLY_HANDOFF_DONE_KEY]["ok"] is True

    asyncio.run(go())
    assert calls.aborted == []


def test_at_schedule_a_request_ending_unscheduled_fails(monkeypatch, api):
    """Ended before it was scheduled (aborted, e.g. its transfer was retired
    or it timed out at the gate): the router gets an error, not line 1."""
    monkeypatch.setenv("ATOM_PD_EARLY_HANDOFF", "1")
    monkeypatch.setenv("ATOM_PD_EARLY_HANDOFF_AT_SCHEDULE", "1")
    _, calls = _engine(api, monkeypatch)

    async def go():
        gen = _generate(api, dict(ROUTER_ASKS))
        first = asyncio.ensure_future(gen.__anext__())
        await asyncio.sleep(0)
        _finished(calls, reason="aborted")
        with pytest.raises(RuntimeError, match="before it was scheduled"):
            await first

    asyncio.run(go())
    assert calls.aborted == []  # it already ended
    assert 11 not in api.engine.io_processor.requests


def test_at_schedule_router_leaving_before_the_answer_aborts(monkeypatch, api):
    monkeypatch.setenv("ATOM_PD_EARLY_HANDOFF", "1")
    monkeypatch.setenv("ATOM_PD_EARLY_HANDOFF_AT_SCHEDULE", "1")
    seq, calls = _engine(api, monkeypatch)

    async def go():
        gen = _generate(api, dict(ROUTER_ASKS))
        first = asyncio.ensure_future(gen.__anext__())
        for _ in range(500):  # until it was submitted and waits for the answer
            await asyncio.sleep(0.01)
            if seq.pd_routed_dp_rank is not None:
                break
        await asyncio.sleep(0.01)
        assert not first.done()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

    asyncio.run(go())
    assert calls.aborted == [11]
    assert 11 not in api.engine.io_processor.requests


def test_legacy_answer_is_plain_json(monkeypatch, api):
    monkeypatch.setenv("ATOM_PD_EARLY_HANDOFF", "1")
    _engine(api, monkeypatch)
    response = api._nonstream_response({"x": 1}, {"text": ""})
    assert response.media_type == "application/json"
    assert api.EARLY_HANDOFF_HEADER not in response.headers


def test_p_fails_closed_without_a_rank(monkeypatch, api):
    monkeypatch.setenv("ATOM_PD_EARLY_HANDOFF", "1")
    _, calls = _engine(api, monkeypatch, rank=None)

    async def go():
        return [o async for o in _generate(api, dict(ROUTER_ASKS))]

    with pytest.raises(RuntimeError, match="DP rank unknown"):
        asyncio.run(go())
    assert calls.aborted == [11]
    assert 11 not in api.engine.io_processor.requests


def test_early_answer_carries_the_pp_size(monkeypatch, api):
    monkeypatch.setenv("ATOM_PD_EARLY_HANDOFF", "1")
    _engine(api, monkeypatch, rank=0)
    api.engine.config.pipeline_parallel_size = 2

    async def go():
        (out,) = [o async for o in _generate(api, dict(ROUTER_ASKS))]
        out["_early_prefill"].task.cancel()
        return out

    kv = asyncio.run(go())["kv_transfer_output_meta_info"]
    assert kv["remote_pp_size"] == 2 and kv["dp_rank"] == 0


@pytest.mark.parametrize(
    "reason,ok",
    [
        ("length", True),
        ("stop", True),
        ("aborted", False),
        ("unschedulable: too long", False),
        ("logprob_error", False),
        (None, False),
    ],
)
def test_prefill_success_follows_the_engine_leave_reason(api, reason, ok):
    assert api._prefill_succeeded(reason) is ok


# --------------------------------------------- CoreManager: dispatched DP rank


def test_unset_routed_rank_is_not_pickled(seq_factory):
    import pickle

    seq = seq_factory([1, 2, 3])
    assert seq.pd_routed_dp_rank is None
    assert "pd_routed_dp_rank" not in seq.__getstate__()
    seq.pd_routed_dp_rank = 3
    assert pickle.loads(pickle.dumps(seq)).pd_routed_dp_rank == 3


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
    for sock in mgr.input_sockets:
        (msg,) = sock.messages
        _, sent = pickle.loads(msg[1])
        assert all(not hasattr(s, "pd_routed_dp_rank") for s in sent)

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

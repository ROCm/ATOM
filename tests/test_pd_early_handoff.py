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


def make_scheduler_connector(monkeypatch, is_producer, *, early, enable=False):
    if early:
        monkeypatch.setenv("ATOM_PD_EARLY_HANDOFF", "1")
    else:
        monkeypatch.delenv("ATOM_PD_EARLY_HANDOFF", raising=False)
    monkeypatch.setattr(mc, "get_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(mc, "get_open_port", lambda: 6400)
    cfg = {"kv_role": "kv_producer" if is_producer else "kv_consumer"}
    if enable:
        cfg["enable_chunked_transfer"] = True
    return mc.MooncakeConnectorScheduler(
        SimpleNamespace(
            kv_transfer_config=cfg,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            parallel_config=SimpleNamespace(data_parallel_size=1, data_parallel_rank=0),
            kv_cache_block_size=4,
            decode_context_parallel_size=1,
        )
    )


def p_params(transfer_id=TRANSFER):
    """What `api_server._early_handoff_params` makes of the router's leg."""
    return {
        "chunked_transfer": True,
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "transfer_id": transfer_id,
    }


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
    assert not plain.early_handoff and not plain.enable_chunked_transfer
    # Without the flag (and without enable_chunked_transfer) a chunked P leg
    # is refused exactly as before.
    seq = seq_factory(list(range(10)), kv_transfer_params=p_params())
    with pytest.raises(ValueError, match="enable_chunked_transfer"):
        plain.admission_ready(seq)


def test_flag_enables_chunked_producer_but_not_consumer(monkeypatch):
    p = make_scheduler_connector(monkeypatch, True, early=True)
    assert p.early_handoff and p.enable_chunked_transfer
    d = make_scheduler_connector(monkeypatch, False, early=True)
    assert not d.early_handoff and not d.enable_chunked_transfer


def test_early_request_waits_for_d_and_is_registered_once(monkeypatch, seq_factory):
    """P allocates an early request's source only after D reserved its pages
    (the chunked admission gate), so P admits in D's order. The transfer ID
    is registered with the worker once, on the first pass."""
    p = make_scheduler_connector(monkeypatch, True, early=True)
    seq = seq_factory(list(range(10)), kv_transfer_params=p_params())
    for _ in range(3):  # The scheduler asks on every pass.
        assert not p.admission_ready(seq)
    assert seq.id in p._chunked_waiting
    assert p.build_connector_meta().chunked_admissions == {seq.id: TRANSFER}
    assert not p.admission_ready(seq)
    assert p.build_connector_meta().chunked_admissions == {}
    # D's write request reached every P rank: the gate opens.
    p.process_pd_completions(
        KVConnectorOutput(
            connector_completions={
                mc.ConnectorCompletion("pd_admission_ready", seq.id, True)
            }
        )
    )
    assert p.admission_ready(seq)


def test_early_request_whose_d_never_comes_expires_without_a_source(
    monkeypatch, seq_factory
):
    from atom.model_engine.sequence import SequenceStatus

    p = make_scheduler_connector(monkeypatch, True, early=True)
    seq = seq_factory(list(range(10)), kv_transfer_params=p_params())
    clock = [100.0]
    monkeypatch.setattr(mc.time, "monotonic", lambda: clock[0])
    assert not p.admission_ready(seq)
    clock[0] += p._chunked_admission_timeout_s
    assert not p.admission_ready(seq)
    assert seq.status == SequenceStatus.ABORTED
    assert not seq.block_table and not p._chunked_waiting


# ------------------------------------------------ real P and D, D arrives late


def _p_scheduler(monkeypatch, num_blocks=6):
    from conftest import MockConfig

    from atom.model_engine.scheduler import Scheduler

    p = Scheduler(MockConfig(num_kvcache_blocks=num_blocks, max_num_batched_tokens=4))
    p.kv_connector = make_scheduler_connector(monkeypatch, True, early=True)
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
    (0 = D's write request reaches P before P registered the transfer). P
    allocates nothing until D reserved; then every page must land
    byte-identical, D gets the first token, and P frees its source only after
    D's reader finished.
    """
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


@pytest.mark.parametrize("byte_limit", [False, True])
def test_opposite_admission_order_progresses_in_early_mode(
    monkeypatch, seq_factory, byte_limit
):
    """Room for one prompt per side; P queues A then B, D admits B then A.

    P answered both at dispatch, so both decode legs reach D. Had P allocated
    A's source before D reserved it, D's pages (or, with `byte_limit`, its
    admission bytes) would hold B, P's would hold a finished A that only D's
    A reader can free, and neither side could admit its other request (the
    pair stalls until the transfer timeout). With the gate P follows D's
    order: both requests finish, no transfer fails, nothing is recomputed.
    """
    from atom.kv_transfer.disaggregation.pd_admission import PdAdmissionController
    from atom.model_engine.scheduler import ScheduledBatchOutput
    from atom.sampling_params import SamplingParams

    p = _p_scheduler(monkeypatch, num_blocks=3)
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
        id=11, prompt_token_ids=[1, 2, 3], num_prompt_tokens=3, max_tokens=1
    )
    calls = SimpleNamespace(aborted=[], params=None, callback=None)

    def preprocess(_prompt, _sampling, stream_callback=None, **kw):
        calls.params = kw["kv_transfer_params"]
        calls.callback = stream_callback
        return seq

    class CoreMgr:
        def _add_prepared_requests(self, seqs):  # marks a routing CoreManager
            pass

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

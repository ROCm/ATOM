# SPDX-License-Identifier: MIT
"""ATOM_PD_EARLY_HANDOFF: P answers the router at DP dispatch and pushes chunks.

CPU only. Reuses the chunked-transfer protocol fixtures (real schedulers, real
producer/consumer workers, CPU buffers in place of HBM and the NIC).
"""

import asyncio
import ctypes
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


def p_params(early=True):
    params = {
        "chunked_transfer": True,
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "transfer_id": TRANSFER,
    }
    if early:
        params["pd_early_handoff"] = True
    return params


def d_params():
    """What the legacy router relay hands D from P's early answer."""
    return {
        "chunked_transfer": True,
        "do_remote_prefill": True,
        "do_remote_decode": False,
        "transfer_id": TRANSFER,
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


def test_flag_off_keeps_existing_gate_and_config(monkeypatch, seq_factory):
    plain = make_scheduler_connector(monkeypatch, True, early=False)
    assert not plain.early_handoff and not plain.enable_chunked_transfer
    gated = make_scheduler_connector(monkeypatch, True, early=False, enable=True)
    seq = seq_factory(list(range(10)), kv_transfer_params=p_params())
    # Without the flag a stray pd_early_handoff key changes nothing: D first.
    assert not gated.admission_ready(seq)
    assert seq.id in gated._chunked_waiting


def test_flag_enables_chunked_producer_but_not_consumer(monkeypatch):
    p = make_scheduler_connector(monkeypatch, True, early=True)
    assert p.early_handoff and p.enable_chunked_transfer
    d = make_scheduler_connector(monkeypatch, False, early=True)
    assert not d.early_handoff and not d.enable_chunked_transfer


def test_early_request_is_admitted_at_once_and_registered_once(
    monkeypatch, seq_factory
):
    p = make_scheduler_connector(monkeypatch, True, early=True)
    seq = seq_factory(list(range(10)), kv_transfer_params=p_params())
    for _ in range(3):  # The scheduler asks on every pass.
        assert p.admission_ready(seq)
    assert not p._chunked_waiting and not p._chunked_ready
    assert p.build_connector_meta().chunked_admissions == {seq.id: TRANSFER}
    assert p.admission_ready(seq)
    assert p.build_connector_meta().chunked_admissions == {}
    # A late worker readiness report for it is dropped, not mistaken for a gate.
    p.process_pd_completions(
        KVConnectorOutput(
            connector_completions={
                mc.ConnectorCompletion("pd_admission_ready", seq.id, True)
            }
        )
    )
    assert not p._chunked_ready


def test_router_chunked_request_keeps_gate_with_flag(monkeypatch, seq_factory):
    """Only P's own early requests skip the gate; router-chunked ones wait."""
    p = make_scheduler_connector(monkeypatch, True, early=True)
    seq = seq_factory(list(range(10)), kv_transfer_params=p_params(early=False))
    assert not p.admission_ready(seq)


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


@pytest.mark.parametrize("d_arrives_after", [0, 1, 2, 3])
def test_p_computes_before_d_exists_and_d_receives_every_page(
    monkeypatch, seq_factory, d_arrives_after
):
    """P starts without D; D attaches before, during or after the prefill.

    `d_arrives_after` counts finished P forwards (3 = after the final
    sampling). Every case must land byte-identical pages, hand D the first
    token, and free P's source only after D's reader finished.
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
        if d_arrives_after == 0:
            d_arrives()
        for forward in range(3):
            batch, seqs = p.schedule()
            # No D gate: P schedules every chunk whether or not D is there.
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
            if forward + 1 == d_arrives_after < 3:
                d_arrives()
            if forward < 2:
                # Source pinned and held while D is absent or mid-read.
                assert ps.block_table
        if d_arrives_after == 3:
            # Finished on P, still pinned for the reader that has not come.
            assert p.kv_connector.should_defer_free(ps)
            assert ps.block_table
            d_arrives()
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


def test_kv_transfer_info_does_not_advertise_early_mode(monkeypatch, api):
    """The router must keep its legacy relay: no chunked template advertised."""
    monkeypatch.setenv("ATOM_PD_EARLY_HANDOFF", "1")
    _engine(api, monkeypatch)
    api.engine.config.parallel_config.data_parallel_size = 4
    info = asyncio.run(api.kv_transfer_info())
    assert "chunked_transfer" not in info


def test_core_manager_reports_dispatched_rank(monkeypatch):
    from aiter_stub import stubbed_aiter

    with stubbed_aiter():
        from atom.model_engine import engine_core_mgr as ecm

    mgr = ecm.CoreManager.__new__(ecm.CoreManager)
    mgr.label = "t"
    mgr.pp_size = 1
    mgr._seq_id_to_callback = {}
    sent = []
    mgr._send_request = lambda rank, frame: sent.append(rank)
    monkeypatch.setattr(ecm, "_add_frame", lambda seqs: b"")
    seq = SimpleNamespace(id=1, stream_callback=None)
    monkeypatch.setattr(ecm.CoreManager, "_routable_engine_count", 1, raising=False)
    monkeypatch.setattr(ecm.tracing, "bind_sequences", lambda _s: None)
    mgr._add_prepared_requests([seq])
    assert seq.pd_routed_dp_rank == 0 and sent == [0]

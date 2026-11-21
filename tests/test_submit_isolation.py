# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A request that does not encode fails alone, and a failed add leaves nothing
of its requests behind: online or offline, on every dispatch path."""

import asyncio
import itertools
import queue
from collections import deque
from threading import Lock
from types import SimpleNamespace

import pytest
from engine_v1_helpers import received

from atom.model_engine.engine_core_mgr import CoreManager, DisaggCoreManager
from atom.model_engine.request import RequestOutput
from atom.model_engine.zmq_protocol import wire
from atom.model_engine.zmq_protocol.codec import REJECTED_REASON, decode_client_message
from atom.model_engine.zmq_protocol.requests import IngressRequest
from atom.sampling_params import SamplingParams

_ids = itertools.count(1000)


def request(*, bad=False, callback=None):
    req = IngressRequest(
        id=next(_ids), prompt_token_ids=[1, 2, 3], sampling_params=SamplingParams()
    )
    if bad:
        req.dp_session_id = "s\ud800"  # a lone surrogate: not UTF-8
    req.stream_callback = callback
    return req


def manager(cls=CoreManager, *, ranks=1, depth=0, keep=True):
    """A manager whose sends decode what they send, per rank."""
    mgr = cls.__new__(cls)
    mgr.label = "test"
    mgr.pp_size = 1
    mgr.global_engine_count = mgr.local_engine_count = ranks
    mgr._dp_lb_strategy = "least_requests"
    mgr._dp_lb_req_equiv = 512
    mgr._dp_session_affinity_enabled = False
    mgr._rank_rotation_cursor = 0
    mgr._rank_reqs = [0] * ranks
    mgr._rank_tokens = [0] * ranks
    mgr._rank_routed_total = [0] * ranks
    mgr._dp_route_counters = {"explicit_total": 0, "load_balanced_total": 0}
    mgr._seq_load = {}
    mgr._lb_lock = Lock()
    mgr._bp_depth = depth
    mgr._bp_pending = deque()
    mgr._bp_offline_batch = depth > 0
    mgr._seq_id_to_callback = {}
    mgr._structured_ids = set()
    mgr._flush_stream_batch_fn = None
    mgr.keep_finished_outputs = keep
    mgr.outputs_queue = queue.Queue()
    mgr.sent = [[] for _ in range(ranks)]

    def send(rank, frames):
        _, specs, _ = decode_client_message(received(frames))
        mgr.sent[rank].extend(spec.seq_id for spec in specs)

    mgr._send_request = send
    return mgr


def rejected_ids(mgr):
    out = []
    while not mgr.outputs_queue.empty():
        out += [r.id for r in mgr.outputs_queue.get_nowait() if r.rejected]
    return out


def test_one_request_that_does_not_encode_is_rejected_alone():
    delivered = []
    good, bad = request(), request(bad=True, callback=delivered.append)
    mgr = manager()
    mgr.add_request([good, bad])
    assert mgr.sent == [[good.id]]
    # Its client hears why it ended; nothing of it stays registered.
    [out] = delivered
    assert isinstance(out, RequestOutput) and out.finished
    assert out.finish_reason.startswith(f"{REJECTED_REASON}: ")
    assert "UnicodeEncodeError" in out.finish_reason
    assert bad.id not in mgr._seq_id_to_callback
    # An offline run gets the rejected result, so generate() ends.
    assert rejected_ids(mgr) == [bad.id]


def test_dp_dispatch_releases_the_charge_of_one_that_does_not_encode():
    mgr = manager(ranks=2)
    good, bad = request(), request(bad=True)
    mgr.add_request([good, bad])
    assert sorted(itertools.chain(*mgr.sent)) == [good.id]
    assert sum(mgr._rank_reqs) == 1 and bad.id not in mgr._seq_load
    assert rejected_ids(mgr) == [bad.id]


def test_backpressure_ends_one_that_does_not_encode_rather_than_retry_it():
    mgr = manager(ranks=2, depth=2)
    reqs = [request(), request(bad=True), request()]
    mgr.add_request(reqs)
    assert sorted(itertools.chain(*mgr.sent)) == [reqs[0].id, reqs[2].id]
    assert not mgr._bp_pending and not mgr._bp_dispatch_paused
    assert reqs[1].id not in mgr._seq_load
    assert rejected_ids(mgr) == [reqs[1].id]


def test_disaggregated_prefill_gets_only_what_decode_took():
    mgr = manager(DisaggCoreManager, ranks=2)
    good, bad = request(), request(bad=True)
    mgr.add_request([good, bad])
    assert mgr.sent == [[good.id], [good.id]]
    # Rejected once, not once per engine.
    assert rejected_ids(mgr) == [bad.id]


def test_a_failed_send_leaves_no_callback_registered():
    mgr = manager()
    req = request(callback=lambda out: None)

    def fail(rank, frames):
        raise RuntimeError("socket gone")

    mgr._send_request = fail
    with pytest.raises(RuntimeError, match="socket gone"):
        mgr.add_request([req])
    assert mgr._seq_id_to_callback == {}


def test_offline_add_that_fails_leaves_no_request_to_wait_on():
    from atom.model_engine.llm_engine import InputOutputProcessor, LLMEngine

    engine = LLMEngine.__new__(LLMEngine)
    io = InputOutputProcessor.__new__(InputOutputProcessor)
    io.requests, io._external_to_internal, io._internal_to_external = {}, {}, {}

    def preprocess_fanout(prompt, params, **kw):
        req = request()
        io.requests[req.id] = req
        io._internal_to_external[req.id] = f"ext-{req.id}"
        io._external_to_internal[f"ext-{req.id}"] = req.id
        return [req]

    io.preprocess_fanout = preprocess_fanout
    engine.io_processor = io

    def add_request(reqs):
        raise RuntimeError("no engine")

    engine.core_mgr = SimpleNamespace(add_request=add_request)
    with pytest.raises(RuntimeError, match="no engine"):
        engine.add_request(["a", "b"], SamplingParams())
    assert not io.has_pending_requests()
    assert io._external_to_internal == io._internal_to_external == {}


def test_online_submit_that_fails_leaves_no_request_state(monkeypatch):
    from atom.entrypoints.openai import api_server

    seq = SimpleNamespace(id=77, sampling_params=SamplingParams())
    aborted = []

    def add_request(seqs):
        raise ValueError("refused")

    engine = SimpleNamespace(
        core_mgr=SimpleNamespace(add_request=add_request, abort_request=aborted.append),
        io_processor=SimpleNamespace(requests={77: seq}),
    )
    monkeypatch.setattr(api_server, "engine", engine)
    monkeypatch.setitem(api_server._seq_id_to_request_id, 77, "req")
    monkeypatch.setitem(api_server._stream_loops, "req", object())
    monkeypatch.setitem(api_server._request_start_times, "req", 0.0)
    with pytest.raises(ValueError, match="refused"):
        asyncio.run(api_server._submit_engine_requests([seq], stream_request_id="req"))
    assert aborted == [77]
    assert not engine.io_processor.requests
    assert 77 not in api_server._seq_id_to_request_id
    assert "req" not in api_server._stream_loops
    assert "req" not in api_server._request_start_times


def test_a_value_nested_too_deep_is_rejected_like_a_bad_string():
    mgr = manager()
    deep = request()
    value = {}
    for _ in range(wire.MAX_VALUE_DEPTH + 1):
        value = {"k": value}
    deep.kv_transfer_params = value
    mgr.add_request([deep, request()])
    assert len(mgr.sent[0]) == 1
    assert rejected_ids(mgr) == [deep.id]

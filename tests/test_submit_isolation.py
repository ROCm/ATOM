# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""A failed add leaves nothing of its requests behind, online or offline, and
a request that does not encode is rejected once, by the engine it reaches."""

import asyncio
from types import SimpleNamespace

import pytest

from atom.model_engine.engine_core_mgr import CoreManager, DisaggCoreManager
from atom.model_engine.engine_core_protocol import EngineCoreRequestType as T
from atom.model_engine.sequence import Sequence
from atom.model_engine.zmq_protocol import codec
from atom.sampling_params import SamplingParams


def request(seq_id, *, bad=False, callback=None):
    return Sequence(
        [1, 2, 3],
        16,
        SamplingParams(),
        id=seq_id,
        # A lone surrogate: not UTF-8, so the request does not encode.
        dp_session_id="s\ud800" if bad else None,
        stream_callback=callback,
    )


def manager(cls=CoreManager):
    """A one-rank manager whose sends decode what they send, per rank."""
    mgr = cls.__new__(cls)
    mgr.label = "test"
    mgr.pp_size = 1
    mgr.global_engine_count = mgr.local_engine_count = 1
    mgr._seq_id_to_callback = {}
    mgr.sent = {}

    def send(rank, payload):
        kind, items = codec.decode_input(payload)
        assert kind == T.ADD
        mgr.sent.setdefault(rank, []).extend(
            (item.seq_id, item.refused is not None) for item in items
        )

    mgr._send_request = send
    return mgr


@pytest.mark.parametrize(
    "cls, sent",
    [
        (CoreManager, {0: [(1, False), (2, True)]}),
        # Decode rejects it once; prefill never sees it.
        (DisaggCoreManager, {1: [(1, False), (2, True)], 0: [(1, False)]}),
    ],
)
def test_one_request_that_does_not_encode_goes_refused_and_the_rest_go(cls, sent):
    mgr = manager(cls)
    mgr.add_request([request(1), request(2, bad=True)])
    # Its Core rejects it, so it ends like any refused request.
    assert mgr.sent == sent


@pytest.mark.parametrize("cls", [CoreManager, DisaggCoreManager])
def test_a_failed_send_leaves_no_callback_registered(cls):
    mgr = manager(cls)

    def fail(rank, payload):
        raise RuntimeError("socket gone")

    mgr._send_request = fail
    with pytest.raises(RuntimeError, match="socket gone"):
        mgr.add_request([request(1, callback=lambda out: None)])
    assert mgr._seq_id_to_callback == {}


def test_offline_add_that_fails_leaves_no_request_to_wait_on():
    from atom.model_engine.llm_engine import InputOutputProcessor, LLMEngine

    engine = LLMEngine.__new__(LLMEngine)
    io = InputOutputProcessor.__new__(InputOutputProcessor)
    io.requests, io._external_to_internal, io._internal_to_external = {}, {}, {}
    ids = iter(range(100, 200))

    def preprocess_fanout(prompt, params, **kw):
        seq = request(next(ids))
        io.requests[seq.id] = seq
        io._internal_to_external[seq.id] = f"ext-{seq.id}"
        io._external_to_internal[f"ext-{seq.id}"] = seq.id
        return [seq]

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

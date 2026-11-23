# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""AtomEngine's Generate on CPU: ATOM's API server module over a fake engine."""

import asyncio
import itertools
import json
import socket
import struct
import sys
import threading
import time
from argparse import Namespace
from collections import deque
from dataclasses import replace
from types import SimpleNamespace

import grpc
import pytest

from atom.entrypoints.grpc import atom_grpc_pb2 as pb
from atom.entrypoints.grpc import atom_grpc_pb2_grpc as pb_grpc
from atom.entrypoints.grpc import frames
from atom.entrypoints.grpc import server as grpc_server
from atom.entrypoints.grpc import session as grpc_session
from atom.entrypoints.grpc.server import AtomEngineServicer
from atom.entrypoints.openai import api_server
from atom.entrypoints.openai.protocol import DEFAULT_MAX_TOKENS
from atom.entrypoints.openai.reasoning_dialects import DIALECTS, FALLBACK_DIALECT
from atom.entrypoints.openai.tool_parser.registry import PARSERS_BY_NAME
from atom.model_engine.request import RequestOutput
from atom.model_engine.zmq_protocol import engine_v1_pb2 as v1
from atom.sampling_params import SamplingParams


class FakeEngine:
    """Registers each preprocessed seq as ATOM's io_processor does; on submit, runs
    `steps`, each a list of (token ids, finish reason or None), one flush apiece."""

    def __init__(self, steps=()):
        self.steps, self.ids = list(steps), itertools.count(1)
        self.config = SimpleNamespace(
            max_model_len=1 << 20, tensor_parallel_size=1, pipeline_parallel_size=1
        )
        self.io_processor = SimpleNamespace(
            requests={},
            preprocess=self.preprocess,
            preprocess_fanout=self.preprocess_fanout,
        )
        self.aborted, self.made = [], []
        self.core_mgr = SimpleNamespace(
            add_request=self.add_request,
            abort_request=self.abort_request,
            max_pool_tokens=None,
        )
        self.running = {}  # submitted seqs no step has ended
        self.grammars = []  # the SamplingParams each structured request prepared
        self.preprocessed, self.submitted = [], []
        self.release = threading.Event()
        self.release.set()
        self.flush = None  # the servicer's per-step flush

    def preprocess(self, prompt, params, *, seq_id=None, **kwargs):
        """The Session's entry: one seq, under the id it assigns."""
        seq_ids = None if seq_id is None else [seq_id]
        return self.preprocess_fanout(prompt, params, seq_ids=seq_ids, **kwargs)[0]

    def preprocess_fanout(self, prompt, params, *, stream_callback, **kwargs):
        self.preprocessed.append((prompt, params, kwargs))
        self.release.wait(10)
        seqs, ids = [], kwargs.get("seq_ids") or [None] * params.n
        for seq_id in ids:
            seq = SimpleNamespace(
                id=seq_id or next(self.ids),
                num_prompt_tokens=len(prompt),
                max_tokens=params.max_tokens,
                callback=stream_callback,
                arrive_time=time.time(),
                json_schema=params.json_schema,  # structured: submitted off the loop
            )
            self.io_processor.requests[seq.id] = seq
            self.made.append(seq.id)
            seqs.append(seq)
        return seqs

    def add_request(self, seqs):
        self.submitted += seqs
        for tokens, reason in self.steps:
            for seq in seqs:
                seq.callback(RequestOutput(seq.id, tokens, reason is not None, reason))
            self.flush()
        if not self.steps or self.steps[-1][1] is None:
            self.running.update((seq.id, seq) for seq in seqs)

    def prepare_structured_request(self, params, kv_transfer, multimodal):
        self.grammars.append(params)

    def abort_request(self, seq_id):
        """As the engine aborts a running seq: a terminal output at its next step."""
        self.aborted.append(seq_id)
        if (seq := self.running.pop(seq_id, None)) is not None:
            seq.callback(RequestOutput(seq_id, [], True, "aborted"))
            self.flush()


class Context:
    def __init__(self):
        self.written = []

    async def write(self, response):
        self.written.append(response)


@pytest.fixture
def serve(monkeypatch):
    monkeypatch.setattr(api_server, "_request_logger", None)
    monkeypatch.setattr(api_server, "custom_message_encoder", None)

    def serve(engine):
        monkeypatch.setattr(api_server, "engine", engine)
        servicer = AtomEngineServicer(api_server, "/models/m")
        engine.flush = servicer.steps.flush
        return servicer

    return serve


def _generate(servicer, **fields):
    fields.setdefault("prompt_token_ids", [1, 2, 3])
    request, context = pb.GenerateRequest(request_id="r", **fields), Context()
    asyncio.run(asyncio.wait_for(servicer.Generate(request, context), 10))
    return context.written


def _error(written):
    assert [r.WhichOneof("response") for r in written] == ["error"]
    return written[0].error


def test_generate_streams_one_chunk_per_step_then_completes(serve):
    engine = FakeEngine([([5], None), ([6, 7], "eos")])
    stop = v1.TokenIds(ids=[9, 9])
    written = _generate(serve(engine), stream=True, stop_token_sequences=[stop])
    assert [list(r.chunk.token_ids) for r in written[:2]] == [[5], [6, 7]]
    assert [r.chunk.completion_tokens for r in written[:2]] == [1, 3]
    complete = written[2].complete
    assert (complete.finish_reason, complete.completion_tokens) == ("eos", 3)
    prompt, params, kwargs = engine.preprocessed[0]
    # As /v1/completions admits ids: its default max_tokens, stops given as ids.
    assert prompt == [1, 2, 3] and params.max_tokens == DEFAULT_MAX_TOKENS
    assert params.max_tokens_is_default and params.stop_strings is None
    assert kwargs["stop_token_sequences"] == [[9, 9]]
    assert engine.io_processor.requests == {} and engine.aborted == []


def test_generate_answers_once_when_not_streaming(serve):
    engine = FakeEngine([([5], None), ([6], "max_tokens")])
    sampling = v1.SamplingParams(max_tokens=2)
    (written,) = _generate(serve(engine), sampling=sampling)
    assert list(written.complete.output_ids) == [5, 6]
    assert not engine.preprocessed[0][1].max_tokens_is_default


@pytest.mark.parametrize("stream", [False, True])
def test_the_complete_carries_the_http_usage_timing(serve, monkeypatch, stream):
    clock = itertools.count()  # arrival 0, the two steps' ids 1 and 2, the end 3
    monkeypatch.setattr(grpc_server, "time", SimpleNamespace(time=lambda: next(clock)))
    engine = FakeEngine([([5], None), ([6, 7], "eos")])
    complete = _generate(serve(engine), stream=stream)[-1].complete
    assert (complete.ttft_s, complete.tpot_s, complete.latency_s) == (1, 0.5, 3)


@pytest.mark.parametrize("stream", [False, True])
def test_token_times_are_taken_as_the_engine_delivers_them(serve, monkeypatch, stream):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(grpc_server, "time", SimpleNamespace(time=lambda: clock.now))
    engine = FakeEngine()

    def add_request(seqs):  # the steps end at 1 s and 2 s; the call reads them at 10 s
        for now, tokens, reason in ((1.0, [5], None), (2.0, [6, 7], "eos")):
            clock.now = now
            seqs[0].callback(
                RequestOutput(seqs[0].id, tokens, reason is not None, reason)
            )
            engine.flush()
        clock.now = 10.0

    engine.core_mgr.add_request = add_request
    complete = _generate(serve(engine), stream=stream)[-1].complete
    assert (complete.ttft_s, complete.tpot_s, complete.latency_s) == (1, 0.5, 10)


def test_a_complete_without_output_ids_has_no_token_timing(serve, monkeypatch):
    clock = itertools.count()  # arrival 0, the step's delivery 1, the end 2
    monkeypatch.setattr(grpc_server, "time", SimpleNamespace(time=lambda: next(clock)))
    complete = _generate(serve(FakeEngine([([], "eos")])))[-1].complete
    assert (complete.ttft_s, complete.tpot_s, complete.latency_s) == (0, 0, 2)


@pytest.mark.parametrize(
    "sampling, message",
    [
        (v1.SamplingParams(n=2, temperature=1.0), "a streamed n above 1"),
        (v1.SamplingParams(max_tokens=0), "max_tokens must be at least 1"),
        (v1.SamplingParams(logprobs_flag=True), "logprobs_flag"),
        (v1.SamplingParams(top_logprobs=2), "top_logprobs"),
        (v1.SamplingParams(json_schema="{}"), "json_schema"),
        (v1.SamplingParams(structured_reasoning_end="</think>"), "reasoning_end"),
    ],
)
def test_generate_refuses_what_it_cannot_serve_with_a_400(serve, sampling, message):
    engine = FakeEngine()
    error = _error(_generate(serve(engine), sampling=sampling, stream=True))
    assert error.http_status == 400 and message in error.message
    assert engine.preprocessed == []


def test_session_affinity_ids_reach_preprocess(serve):
    engine = FakeEngine([([5], "eos")])
    _generate(serve(engine), dp_session_id="s", dp_parent_session_id="p")
    _generate(serve(engine))
    (_, _, given), (_, _, absent) = engine.preprocessed
    assert (given["dp_session_id"], given["dp_parent_session_id"]) == ("s", "p")
    assert (absent["dp_session_id"], absent["dp_parent_session_id"]) == (None, None)


def test_n_above_1_answers_one_complete_per_sibling(serve):
    engine = FakeEngine([([5], None), ([6], "eos")])
    sampling = v1.SamplingParams(n=2, temperature=1.0)
    written = _generate(serve(engine), sampling=sampling)
    assert [r.complete.index for r in written] == [0, 1]
    assert [list(r.complete.output_ids) for r in written] == [[5, 6], [5, 6]]
    assert engine.preprocessed[0][1].n == 2 and len(engine.submitted) == 2
    assert engine.io_processor.requests == {} and engine.aborted == []


@pytest.mark.parametrize("stream", [False, True])
def test_greedy_n_above_1_is_served_as_1_as_on_http(serve, stream):
    engine = FakeEngine([([5], "eos")])
    sampling = v1.SamplingParams(n=3, temperature=0.0)
    written = _generate(serve(engine), sampling=sampling, stream=stream)
    assert [r.WhichOneof("response") for r in written][-1] == "complete"
    assert engine.preprocessed[0][1].n == 1


def test_n_above_1_with_kv_transfer_params_is_a_400(serve):
    engine = FakeEngine()
    kv = json.dumps({"do_remote_decode": True})
    sampling = v1.SamplingParams(n=2, temperature=1.0)
    written = _generate(serve(engine), sampling=sampling, kv_transfer_params_json=kv)
    error = _error(written)
    assert error.http_status == 400 and "kv_transfer_params" in error.message
    assert engine.preprocessed == []


def test_abort_ends_every_sibling(serve):
    engine = FakeEngine([([5], None)])  # both siblings run on after their first step
    servicer = serve(engine)

    async def abort_running():
        sampling = v1.SamplingParams(n=2, temperature=1.0)
        request = pb.GenerateRequest(
            request_id="r", prompt_token_ids=[1], sampling=sampling
        )
        context = Context()
        call = asyncio.create_task(servicer.Generate(request, context))
        await _until(lambda: len(engine.running) == 2)
        await servicer.Abort(pb.AbortRequest(request_id="r"), None)
        await asyncio.wait_for(call, 10)
        return context.written

    written = asyncio.run(abort_running())
    assert [r.complete.finish_reason for r in written] == ["aborted", "aborted"]
    assert sorted(engine.aborted) == [1, 2] and engine.io_processor.requests == {}
    assert servicer.live == {}


def test_a_partly_dispatched_fan_out_aborts_every_sibling(serve):
    engine = FakeEngine()

    def dispatch_one_then_fail(seqs):  # as CoreManager when a later rank's send fails
        engine.running[seqs[0].id] = seqs[0]
        raise RuntimeError("the second rank's send failed")

    engine.core_mgr.add_request = dispatch_one_then_fail
    sampling = v1.SamplingParams(n=2, temperature=1.0)
    error = _error(_generate(serve(engine), sampling=sampling))
    assert error.http_status == 500
    assert engine.running == {} and engine.io_processor.requests == {}


def test_a_prompt_only_in_kv_transfer_params_is_admitted(serve):
    engine = FakeEngine([([5], "eos")])
    kv = json.dumps({"prompt_token_ids": [4, 5], "do_remote_prefill": False})
    written = _generate(serve(engine), prompt_token_ids=[], kv_transfer_params_json=kv)
    assert written[-1].HasField("complete")
    prompt, _, kwargs = engine.preprocessed[0]
    assert prompt == [4, 5]
    assert kwargs["kv_transfer_params"] == {"do_remote_prefill": False}


def test_a_request_without_a_prompt_is_a_400(serve):
    error = _error(_generate(serve(FakeEngine()), prompt_token_ids=[]))
    assert (error.http_status, error.message) == (400, "prompt_token_ids is required")


def test_a_structured_output_error_ends_the_call_with_a_500(serve):
    engine = FakeEngine([([5], None), ([6], "structured_output_error")])
    written = _generate(serve(engine), stream=True)
    assert [r.WhichOneof("response") for r in written] == ["chunk", "error"]
    assert written[-1].error.http_status == 500
    assert engine.io_processor.requests == {} and engine.aborted == []


async def _until(condition):
    for _ in range(500):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition never held")


def test_a_call_cancelled_during_preprocess_leaves_no_seq(serve):
    engine = FakeEngine()
    engine.release.clear()
    servicer = serve(engine)

    async def cancel_while_preprocessing():
        request = pb.GenerateRequest(request_id="r", prompt_token_ids=[1, 2])
        call = asyncio.create_task(servicer.Generate(request, Context()))
        await _until(lambda: engine.preprocessed)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        engine.release.set()
        await _until(lambda: engine.made and not engine.io_processor.requests)

    asyncio.run(cancel_while_preprocessing())
    assert engine.submitted == []


def test_a_call_cancelled_mid_stream_aborts_its_seq(serve):
    engine = FakeEngine([([5], None)])
    servicer = serve(engine)

    async def cancel_mid_stream():
        request = pb.GenerateRequest(request_id="r", prompt_token_ids=[1], stream=True)
        context = Context()
        call = asyncio.create_task(servicer.Generate(request, context))
        await _until(lambda: context.written)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call

    asyncio.run(cancel_mid_stream())
    assert engine.aborted == [1] and engine.io_processor.requests == {}
    assert servicer.live == {}


def test_request_log_records_the_request_and_each_answer(serve, monkeypatch):
    lines = []
    monkeypatch.setattr(
        api_server, "_request_logger", SimpleNamespace(info=lines.append)
    )
    _generate(serve(FakeEngine([([5], None), ([6], "eos")])), stream=True)
    entries = [json.loads(line) for line in lines]
    assert [e["type"] for e in entries] == [
        "request",
        "stream_chunk",
        "stream_chunk",
        "response",
    ]
    assert {e["request_id"] for e in entries} == {"r"}
    assert entries[0]["data"]["prompt_token_ids"] == [1, 2, 3]
    assert entries[-1]["data"]["complete"]["finish_reason"] == "eos"


def _envelope(message) -> v1.Envelope:
    return v1.Envelope.FromString(message[0])


def _submit(prompt=(1, 2), **sampling):
    return frames.SubmitRequest(
        seq_id=1,
        prompt_token_ids=list(prompt),
        sampling_params=SamplingParams(**sampling),
    )


def test_the_session_refuses_logprobs_it_would_not_serve(serve):
    engine = FakeEngine()
    servicer = serve(engine)
    session = servicer.sessions.open()
    with pytest.raises(ValueError, match="logprobs"):
        asyncio.run(servicer._submit(session, _submit(logprobs=True)))
    assert engine.preprocessed == []


def test_the_session_refuses_stop_strings_without_their_ids(serve):
    engine = FakeEngine()
    servicer = serve(engine)
    session = servicer.sessions.open()
    with pytest.raises(ValueError, match="stop_token_sequences"):
        asyncio.run(servicer._submit(session, _submit(stop_strings=["END"])))
    assert grpc_server._rejected(1, ValueError("x")).code == v1.RejectedRequest.INVALID
    assert engine.preprocessed == []


def test_a_session_grammar_failure_is_rejected_as_internal(serve):
    engine = FakeEngine()
    session = serve(engine).sessions.open()
    seq = SimpleNamespace(id=session.seq_id(1), num_prompt_tokens=2, arrive_time=0.0)
    engine.io_processor.requests[seq.id] = seq
    session.add(1, seq, "r")
    session.put(RequestOutput(seq.id, [], True, "structured_output_error"))
    session.flush()
    rejected = _envelope(session.outbox.get_nowait()).rejected.items
    assert [(r.seq_id, r.code) for r in rejected] == [(1, v1.RejectedRequest.INTERNAL)]
    assert session.outbox.empty() and engine.io_processor.requests == {}


def test_a_session_ended_during_preprocess_leaves_no_seq(serve):
    engine = FakeEngine()
    engine.release.clear()
    servicer = serve(engine)
    session = servicer.sessions.open()

    async def end_while_preprocessing():
        submit = asyncio.create_task(servicer._submit(session, _submit()))
        await _until(lambda: engine.preprocessed)
        submit.cancel()
        with pytest.raises(asyncio.CancelledError):
            await submit
        engine.release.set()
        await _until(lambda: engine.made and not engine.io_processor.requests)

    asyncio.run(end_while_preprocessing())
    assert engine.submitted == [] and session.requests == {}


def test_a_session_ended_during_preprocess_keeps_its_window_until_the_seq_drops(serve):
    engine = FakeEngine()
    engine.release.clear()
    servicer = serve(engine)
    session = servicer.sessions.open()
    window = session.window

    async def end_while_preprocessing():
        submit = asyncio.create_task(servicer._submit(session, _submit()))
        await _until(lambda: engine.preprocessed)
        submit.cancel()  # as the Session RPC ends: its reader cancelled, then close
        with pytest.raises(asyncio.CancelledError):
            await submit
        session.close()
        held = window not in servicer.sessions.free
        engine.release.set()
        await _until(lambda: engine.made and not engine.io_processor.requests)
        return held

    assert asyncio.run(end_while_preprocessing())
    assert window in servicer.sessions.free and session.window is None


def test_a_submission_cancelled_once_it_reached_the_engine_holds_its_window(
    serve, monkeypatch
):
    engine, entered, admit = FakeEngine(), threading.Event(), threading.Event()
    servicer = serve(engine)
    sessions = servicer.sessions
    session = sessions.open()
    window = session.window
    sessions.free.clear()  # every other window held: a next Session needs this one

    def add_request(seqs):  # the structured admission, on its thread
        entered.set()
        admit.wait(10)
        engine.running.update((seq.id, seq) for seq in seqs)

    monkeypatch.setattr(engine.core_mgr, "add_request", add_request)
    # The engine ends an aborted seq at its next step, not at once.
    monkeypatch.setattr(engine.core_mgr, "abort_request", engine.aborted.append)

    async def cancel_then_resubmit():
        request = _submit(json_schema='{"type": "object"}')
        submit = asyncio.create_task(servicer._submit(session, request))
        await asyncio.to_thread(entered.wait, 10)
        submit.cancel()  # as the Session RPC ends: its reader cancelled, then close
        session.close()
        admit.set()
        with pytest.raises(asyncio.CancelledError):
            await submit
        reopened = sessions.open()  # before the abort's end: no window
        (seq,) = engine.running.values()
        seq.callback(RequestOutput(seq.id, [], True, "aborted"))
        engine.flush()
        await _until(lambda: window in sessions.free)
        return reopened

    assert asyncio.run(cancel_then_resubmit()) is None
    assert engine.aborted and engine.io_processor.requests == {}
    assert sessions.open().window == window


def test_engine_ids_are_not_reused_by_a_client_seq_id_or_a_windows_next_session(
    serve,
):
    # CoreManager's finished-seq message may still name a completed request's id
    # after the client reads its Completions and sends its seq_id again.
    engine = FakeEngine([([5], "eos")])
    servicer = serve(engine)
    sessions = servicer.sessions
    sessions.free = deque([sessions.free[0]])  # one window, for both Sessions

    async def submit_seq_id_1_twice(session):
        for _ in range(2):
            await servicer._submit(session, _submit())
            await _until(lambda: not session.requests)
        session.close()

    asyncio.run(submit_seq_id_1_twice(sessions.open()))
    asyncio.run(submit_seq_id_1_twice(sessions.open()))
    engine_ids = [seq.id for seq in engine.submitted]
    assert len(set(engine_ids)) == 4


def test_a_client_that_stops_reading_fails_its_session_past_the_outbox_bound(
    serve, monkeypatch
):
    monkeypatch.setattr(grpc_session, "OUTBOX", 2)
    engine = FakeEngine([([5], None)])  # the request runs on, a message per step
    servicer = serve(engine)
    open_session, opened = servicer.sessions.open, []

    def record_open():
        opened.append(open_session())
        return opened[-1]

    monkeypatch.setattr(servicer.sessions, "open", record_open)

    class Context:
        async def abort(self, code, details):
            self.code, self.details = code, details
            raise grpc.RpcError()

    async def client():
        await asyncio.Event().wait()  # sends nothing more, ends never
        yield

    async def stop_reading():
        context = Context()
        call = servicer.Session(client(), context)
        assert _envelope(list((await anext(call)).frames)).WhichOneof("body") == "ready"
        session = opened[0]
        await servicer._submit(session, _submit())
        await _until(lambda: session.outbox.qsize() == 1)
        seq = engine.running[next(iter(session.client_ids))]
        for _ in range(2):  # the engine's next steps, which the client does not read
            seq.callback(RequestOutput(seq.id, [6], False))
            engine.flush()
        await _until(lambda: session.failed is not None)
        with pytest.raises(grpc.RpcError):
            await anext(call)  # its messages dropped: the failure is all that is left
        return context

    context = asyncio.run(stop_reading())
    assert context.code == grpc.StatusCode.RESOURCE_EXHAUSTED
    assert "left 2 Session messages unread" in context.details
    assert engine.running == {} and engine.aborted  # its request aborted


def test_a_failed_session_admits_no_submission_still_in_preprocess(serve, monkeypatch):
    monkeypatch.setattr(grpc_session, "OUTBOX", 0)  # its next message fails it
    engine = FakeEngine([([5], None)])
    engine.release.clear()
    servicer = serve(engine)
    open_session, opened = servicer.sessions.open, []
    monkeypatch.setattr(
        servicer.sessions, "open", lambda: opened.append(open_session()) or opened[-1]
    )

    async def client():
        yield pb.Frames(frames=_submit_batch((3, "a", [1, 2])))
        await asyncio.Event().wait()

    async def fail_during_preprocess():
        call = servicer.Session(client(), SimpleNamespace())
        await anext(call)  # Ready
        await _until(lambda: engine.preprocessed)
        opened[0].send([b"a step the client does not read"])
        engine.release.set()
        await _until(lambda: engine.made and not engine.io_processor.requests)
        await call.aclose()
        return opened[0]

    session = asyncio.run(fail_during_preprocess())
    assert session.failed is not None and session.requests == {}
    assert engine.submitted == []


def test_a_session_failed_in_structured_admission_aborts_once_its_seq_is_enqueued(
    serve, monkeypatch
):
    monkeypatch.setattr(grpc_session, "OUTBOX", 0)  # its next message fails it
    engine, entered, admit, events = (
        FakeEngine(),
        threading.Event(),
        threading.Event(),
        [],
    )
    servicer = serve(engine)
    open_session, opened = servicer.sessions.open, []
    monkeypatch.setattr(
        servicer.sessions, "open", lambda: opened.append(open_session()) or opened[-1]
    )

    def add_request(seqs):  # the structured admission, on its thread
        entered.set()
        admit.wait(10)
        events.append("enqueued")

    monkeypatch.setattr(engine.core_mgr, "add_request", add_request)
    monkeypatch.setattr(
        engine.core_mgr, "abort_request", lambda seq_id: events.append("aborted")
    )

    class Context:
        async def abort(self, code, details):
            raise grpc.RpcError()

    async def client():
        batch = _submit_batch((3, "a", [1, 2]))
        envelope = v1.Envelope.FromString(batch[0])
        envelope.submit.requests[0].sampling.json_schema = '{"type": "object"}'
        yield pb.Frames(frames=[envelope.SerializeToString(), *batch[1:]])
        await asyncio.Event().wait()

    async def fail_during_admission():
        call = servicer.Session(client(), Context())
        await anext(call)  # Ready
        await asyncio.to_thread(entered.wait, 10)
        opened[0].send([b"a step the client does not read"])  # cancels the reader
        await asyncio.sleep(0.05)  # which settles its submission
        with pytest.raises(grpc.RpcError):
            await anext(call)  # the call ends
        admit.set()
        with pytest.raises(asyncio.CancelledError):
            await opened[0].reader  # its submission settled, then aborted

    asyncio.run(fail_during_admission())
    # close() aborted it before it was enqueued, the admission once it was.
    assert events == ["aborted", "enqueued", "aborted"]


def test_a_session_submission_that_fails_after_a_send_aborts_its_seq(serve):
    engine = FakeEngine()
    servicer = serve(engine)
    session = servicer.sessions.open()

    def decode_then_fail(seqs):  # as RapidServe's: decode has it, prefill's send fails
        engine.running.update((seq.id, seq) for seq in seqs)
        raise RuntimeError("the prefill send failed")

    engine.core_mgr.add_request = decode_then_fail
    with pytest.raises(RuntimeError, match="prefill send"):
        asyncio.run(servicer._submit(session, _submit()))
    assert engine.running == {} and engine.aborted
    assert session.requests == {} and engine.io_processor.requests == {}


def test_a_session_admits_a_prompt_only_in_kv_transfer_params(serve):
    engine = FakeEngine()
    servicer = serve(engine)
    kv = {"prompt_token_ids": [4, 5], "do_remote_prefill": False}
    submit = replace(_submit(prompt=()), kv_transfer_params=kv)
    asyncio.run(servicer._submit(servicer.sessions.open(), submit))
    prompt, _, kwargs = engine.preprocessed[0]
    assert prompt == [4, 5]
    assert kwargs["kv_transfer_params"] == {"do_remote_prefill": False}


def test_a_session_request_without_a_prompt_is_refused(serve):
    engine = FakeEngine()
    servicer = serve(engine)
    with pytest.raises(ValueError, match="prompt_token_ids is required"):
        asyncio.run(servicer._submit(servicer.sessions.open(), _submit(prompt=())))
    assert engine.preprocessed == []


def test_abort_during_preprocess_keeps_the_request_out_of_the_engine(serve):
    engine = FakeEngine([([5], "eos")])
    engine.release.clear()
    servicer = serve(engine)

    async def abort_while_preprocessing():
        request = pb.GenerateRequest(request_id="r", prompt_token_ids=[1], stream=True)
        context = Context()
        call = asyncio.create_task(servicer.Generate(request, context))
        await _until(lambda: engine.preprocessed)
        await servicer.Abort(pb.AbortRequest(request_id="r"), None)
        engine.release.set()
        await asyncio.wait_for(call, 10)
        return context.written

    (written,) = asyncio.run(abort_while_preprocessing())
    assert written.complete.finish_reason == "aborted"
    assert engine.submitted == [] and engine.io_processor.requests == {}
    assert servicer.live == {}


def test_abort_during_submission_aborts_the_request_once_submitted(serve, monkeypatch):
    engine = FakeEngine([([5], None)])
    servicer = serve(engine)
    submit = api_server._submit_engine_requests

    async def abort_then_submit(seqs):
        await servicer.Abort(pb.AbortRequest(request_id="r"), None)
        await submit(seqs)

    monkeypatch.setattr(api_server, "_submit_engine_requests", abort_then_submit)
    written = _generate(servicer, stream=True)
    assert written[-1].complete.finish_reason == "aborted"
    assert engine.running == {} and engine.io_processor.requests == {}


SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}}
RESPONSE_FORMAT = json.dumps(
    {"type": "json_schema", "json_schema": {"name": "s", "schema": SCHEMA}}
)


@pytest.fixture
def chat(serve, monkeypatch):
    """`serve` with the chat route's startup state: no reasoning switch, no tool
    parser, the fallback dialect."""
    for name, value in {
        "reasoning_dialect": FALLBACK_DIALECT,
        "reasoning_toggle": None,
        "model_starts_in_reasoning": False,
        "required_tool_channel": None,
        "tool_call_parser_cls": None,
        "synthetic_token_text": None,
        "tokenizer": SimpleNamespace(),
    }.items():
        monkeypatch.setattr(api_server, name, value)
    return serve


def _prepare(servicer, structured, **fields):
    request = pb.PrepareStructuredOutputRequest(structured_output=structured, **fields)
    return asyncio.run(servicer.PrepareStructuredOutput(request, None))


def test_prepare_answers_a_response_schema_as_the_chat_route_derives_it(chat):
    engine = FakeEngine()
    structured = pb.StructuredOutput(response_format_json=RESPONSE_FORMAT)
    answer = _prepare(chat(engine), structured).sampling
    assert json.loads(answer.json_schema) == SCHEMA
    assert answer.structured_reasoning_end == ""
    assert len(engine.grammars) == 1  # each rank's grammar prepared


def test_a_prompt_opening_reasoning_suspends_the_grammar_until_its_end(chat):
    structured = pb.StructuredOutput(
        response_format_json=RESPONSE_FORMAT, prompt_opens_reasoning=True
    )
    answer = _prepare(chat(FakeEngine()), structured).sampling
    assert answer.structured_reasoning_end == FALLBACK_DIALECT.think_end_marker


def test_generate_admits_a_chat_with_the_fields_prepare_answers(chat):
    engine = FakeEngine([([5], "eos")])
    structured = pb.StructuredOutput(response_format_json=RESPONSE_FORMAT)
    written = _generate(chat(engine), structured_output=structured)
    assert written[-1].complete.finish_reason == "eos"
    params = engine.preprocessed[0][1]
    assert json.loads(params.json_schema) == SCHEMA
    assert engine.submitted and engine.io_processor.requests == {}


def test_message_declared_tools_reach_the_tool_call_grammar(chat, monkeypatch):
    seen = {}

    async def tool_call_grammar(engine, config, params, channel, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(api_server, "tool_call_parser_cls", object)
    monkeypatch.setattr(api_server, "configure_tool_call_grammar", tool_call_grammar)
    top = {"type": "function", "function": {"name": "top", "parameters": {}}}
    declared = {"type": "function", "function": {"name": "own", "parameters": {}}}
    structured = pb.StructuredOutput(
        tools_json=json.dumps([top]), message_tools_json=json.dumps([None, [declared]])
    )
    _prepare(chat(FakeEngine()), structured)
    names = [tool["function"]["name"] for tool in seen["tools"]]
    assert sorted(names) == ["own", "top"]


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"sampling": v1.SamplingParams(ignore_eos=True)}, "ignore_eos"),
        (
            {"stop_token_sequences": [v1.TokenIds(ids=[7])]},
            "custom stop strings",
        ),
        ({"sampling": v1.SamplingParams(top_logprobs=2)}, "top_logprobs requires"),
    ],
)
def test_prepare_refuses_what_the_chat_route_refuses_with_a_400(chat, fields, message):
    engine = FakeEngine()
    structured = pb.StructuredOutput(response_format_json=RESPONSE_FORMAT)
    error = _prepare(chat(engine), structured, **fields).error
    assert error.http_status == 400 and message in error.message
    assert engine.grammars == []


def test_generate_refuses_stops_with_a_response_schema_as_the_chat_route_does(chat):
    engine = FakeEngine()
    structured = pb.StructuredOutput(response_format_json=RESPONSE_FORMAT)
    stops = [v1.TokenIds(ids=[7])]
    written = _generate(
        chat(engine), structured_output=structured, stop_token_sequences=stops
    )
    error = _error(written)
    assert error.http_status == 400 and "custom stop strings" in error.message
    assert engine.grammars == [] and engine.preprocessed == []


def test_chat_template_kwargs_must_be_an_object(chat):
    structured = pb.StructuredOutput(chat_template_kwargs_json="[]")
    error = _prepare(chat(FakeEngine()), structured).error
    assert (error.http_status, error.message) == (
        400,
        "chat_template_kwargs must be an object",
    )


def test_grpc_port_refuses_a_tokenizer_with_named_chat_templates(monkeypatch, capsys):
    templates = {"default": "{{ messages }}", "tool_use": "{{ tools }}"}

    def frontend(args):
        monkeypatch.setattr(
            api_server, "tokenizer", SimpleNamespace(chat_template=templates)
        )

    def no_engine(*args, **kwargs):
        raise AssertionError("the engine started")

    monkeypatch.setattr(api_server, "_setup_frontend", frontend)
    monkeypatch.setattr(api_server.EngineArgs, "from_cli_args", no_engine)
    monkeypatch.setattr(sys, "argv", ["atom", "--model", "m", "--grpc-port", "1"])
    with pytest.raises(SystemExit):
        api_server.main()
    error = capsys.readouterr().err
    assert "(default, tool_use); pick one with --chat-template" in error


def test_a_step_ending_as_the_listener_starts_reaches_its_call(monkeypatch):
    core_mgr = SimpleNamespace(_flush_stream_batch_fn=lambda: None)  # HTTP's
    api, delivered = SimpleNamespace(engine=SimpleNamespace(core_mgr=core_mgr)), []

    async def listen(address, servicer, listening):
        try:  # a call's last output, buffered and flushed on this thread
            queue = asyncio.Queue()
            servicer.steps.callback(queue.put_nowait)(
                RequestOutput(1, [5], True, "eos")
            )
            core_mgr._flush_stream_batch_fn()
            delivered.append(await asyncio.wait_for(queue.get(), 1))
        finally:
            listening.set_result((asyncio.get_running_loop(), None))

    monkeypatch.setattr(grpc_server, "_serve", listen)
    monkeypatch.setattr(grpc_server.Sequence, "counter", grpc_server.Sequence.counter)
    args = Namespace(host="127.0.0.1", grpc_port=1, model="m")
    grpc_server.start_grpc_server(api, args)
    assert [output.finish_reason for output in delivered] == ["eos"]


def _listen(serve, monkeypatch, engine):
    """`engine` behind a listening gRPC server: its stop and a channel to it."""
    serve(engine)  # api_server's engine; start_grpc_server makes its own servicer
    engine.core_mgr._flush_stream_batch_fn = None
    engine.flush = lambda: engine.core_mgr._flush_stream_batch_fn()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    monkeypatch.setattr(grpc_server.Sequence, "counter", grpc_server.Sequence.counter)
    args = Namespace(host="127.0.0.1", grpc_port=port, model="m")
    stop = grpc_server.start_grpc_server(api_server, args)
    return stop, grpc.insecure_channel(f"127.0.0.1:{port}")


def test_a_generate_past_4_mib_is_admitted(serve, monkeypatch):
    stop, channel = _listen(serve, monkeypatch, FakeEngine([([5], "eos")]))
    prompt = [1 << 28] * 900_000  # 5 bytes apiece: 4.5 MB
    request = pb.GenerateRequest(request_id="r", prompt_token_ids=prompt)
    assert request.ByteSize() > 4 << 20
    try:
        answer = list(pb_grpc.AtomEngineStub(channel).Generate(request, timeout=30))
        assert [r.WhichOneof("response") for r in answer] == ["complete"]
        assert answer[0].complete.prompt_tokens == len(prompt)
    finally:
        channel.close()
        stop()


def test_shutdown_ends_running_calls_before_it_returns(serve, monkeypatch):
    engine = FakeEngine([([5], None)])  # the seq runs on after its first step
    stop, channel = _listen(serve, monkeypatch, engine)
    request = pb.GenerateRequest(request_id="r", prompt_token_ids=[1], stream=True)
    stub = pb_grpc.AtomEngineStub(channel)
    streaming = stub.Generate(request, timeout=10)
    assert list(next(streaming).chunk.token_ids) == [5]
    engine.release.clear()
    preprocessing = stub.Generate(request, timeout=10)
    for _ in range(500):
        if len(engine.preprocessed) == 2:
            break
        time.sleep(0.01)
    threading.Timer(0.3, engine.release.set).start()
    stop()
    # Before the engine closes: one seq aborted, the other dropped once made.
    assert engine.made == [1, 2] and engine.aborted == [1]
    assert engine.io_processor.requests == {} and engine.submitted[1:] == []
    for call in (streaming, preprocessing):
        with pytest.raises(grpc.RpcError):
            list(call)
    channel.close()


def _submit_batch(*requests) -> list[bytes]:
    """A SubmitBatch of (seq_id, external id, prompt) requests, no sampling field set."""
    envelope = v1.Envelope(protocol_version=frames.PROTOCOL_VERSION)
    envelope.submit.SetInParent()
    data = []
    for seq_id, external, prompt in requests:
        data.append(struct.pack(f"<{len(prompt)}i", *prompt))
        envelope.submit.requests.add(
            seq_id=seq_id,
            external_request_id=external,
            prompt_token_ids=v1.BufferRef(frame=len(data), nbytes=len(data[-1])),
        )
    return [envelope.SerializeToString(), *data]


def test_a_session_serves_a_submit_batch_over_the_wire(serve, monkeypatch):
    engine = FakeEngine([([5], None), ([6], "eos")])
    stop, channel = _listen(serve, monkeypatch, engine)
    read = threading.Event()

    def client():
        yield pb.Frames(frames=_submit_batch((3, "a", [1, 2]), (9, "b", [4])))
        read.wait(10)  # the client's side ends once it has read its answers

    try:
        answers = pb_grpc.AtomEngineStub(channel).Session(client(), timeout=10)
        messages = [_envelope(list(next(answers).frames)) for _ in range(7)]
        read.set()
        assert list(answers) == []
    finally:
        channel.close()
        stop()
    assert messages[0].WhichOneof("body") == "ready"
    for seq_id, external in ((3, "a"), (9, "b")):
        mine = [
            (m.WhichOneof("body"), m)
            for m in messages
            if seq_id in m.stream.seq_ids
            or seq_id in [c.seq_id for c in m.completions.items]
        ]
        assert [body for body, _ in mine] == ["stream", "stream", "completions"]
        assert [list(m.stream.tokens) for _, m in mine[:2]] == [[5], [6]]
        (done,) = mine[2][1].completions.items  # its ids went in the StreamBatches
        assert (done.leave_reason, done.external_request_id) == ("eos", external)
        assert done.completion_token_ids.nbytes == 0
    # Omitted sampling fields read as /v1/completions fills them.
    for prompt, params, _ in engine.preprocessed:
        assert params.max_tokens == DEFAULT_MAX_TOKENS and params.max_tokens_is_default
    assert [p for p, _, _ in engine.preprocessed] == [[1, 2], [4]]
    assert engine.io_processor.requests == {} and engine.aborted == []


def test_a_step_flushed_after_the_server_stopped_is_dropped():
    steps = grpc_server.StepOutputs()

    async def buffer():
        steps.callback(asyncio.Queue().put_nowait)(RequestOutput(1, [5], True, "eos"))

    asyncio.run(buffer())  # its loop is closed once this returns
    steps.flush()


def test_model_info_leaves_an_absent_chat_template_unset(serve, monkeypatch):
    engine = FakeEngine()
    engine.config.end_token_ids = [2]
    servicer = serve(engine)
    monkeypatch.setattr(api_server, "tokenizer", SimpleNamespace(chat_template=None))
    info = asyncio.run(servicer.GetModelInfo(pb.GetModelInfoRequest(), None))
    assert not info.HasField("chat_template") and list(info.eos_token_ids) == [2]


@pytest.mark.parametrize(
    "atom_parser, router_parser",
    [
        ("dsml_v41", "dsml_v41"),
        ("kimi", "kimik2"),
        ("qwen", "qwen_xml"),
        ("minimax", "minimax"),
        ("glm", None),  # atomesh's model-name mapping tells GLM's dialects apart
        (None, None),
    ],
)
def test_router_tool_call_parser_names_atoms_format(atom_parser, router_parser):
    parser_cls = PARSERS_BY_NAME.get(atom_parser)
    names = grpc_server.router_parser_names(parser_cls, FALLBACK_DIALECT)
    assert names == (router_parser, "qwen3")


@pytest.mark.parametrize(
    "dialect, router_parser",
    [
        (FALLBACK_DIALECT, "qwen3"),  # <think>: starts outside reasoning
        (DIALECTS[1], None),  # MiniMax-M3's <mm:think>
        (DIALECTS[0], None),  # Kimi-K3's channels
        (None, None),
    ],
)
def test_router_reasoning_parser_is_qwen3_only_for_the_think_dialect(
    dialect, router_parser
):
    assert grpc_server.router_parser_names(None, dialect) == (None, router_parser)


def test_model_info_names_the_router_parsers(serve, monkeypatch):
    engine = FakeEngine()
    engine.config.end_token_ids = [2]
    monkeypatch.setattr(api_server, "tokenizer", SimpleNamespace(chat_template=None))
    monkeypatch.setattr(api_server, "tool_call_parser_cls", PARSERS_BY_NAME["dsml_v41"])
    monkeypatch.setattr(api_server, "reasoning_dialect", FALLBACK_DIALECT)
    info = asyncio.run(serve(engine).GetModelInfo(pb.GetModelInfoRequest(), None))
    assert (info.tool_call_parser, info.reasoning_parser) == ("dsml_v41", "qwen3")
    monkeypatch.setattr(api_server, "tool_call_parser_cls", None)
    monkeypatch.setattr(api_server, "reasoning_dialect", DIALECTS[1])
    info = asyncio.run(serve(engine).GetModelInfo(pb.GetModelInfoRequest(), None))
    assert not info.HasField("tool_call_parser")
    assert not info.HasField("reasoning_parser")


@pytest.mark.parametrize(
    "encoder, usage_on_finish", [(None, False), ("encoding_dsv41", True)]
)
def test_model_info_places_streamed_usage_as_the_encoder_does(
    serve, monkeypatch, encoder, usage_on_finish
):
    engine = FakeEngine()
    engine.config.end_token_ids = [2]
    monkeypatch.setattr(api_server, "tokenizer", SimpleNamespace(chat_template=None))
    encoder = encoder and SimpleNamespace(name=encoder)
    monkeypatch.setattr(api_server, "custom_message_encoder", encoder)
    info = asyncio.run(serve(engine).GetModelInfo(pb.GetModelInfoRequest(), None))
    assert info.usage_on_finish is usage_on_finish


def test_model_info_says_how_a_reply_starts_in_reasoning(serve, monkeypatch):
    engine = FakeEngine()
    engine.config.end_token_ids = [2]
    monkeypatch.setattr(api_server, "tokenizer", SimpleNamespace(chat_template=None))
    info = asyncio.run(serve(engine).GetModelInfo(pb.GetModelInfoRequest(), None))
    assert not info.HasField("implicit_reasoning_opener")
    assert info.reasoning_toggle_json == ""
    # DeepSeek-R1's template closes a <think> it never opens.
    monkeypatch.setattr(api_server, "model_starts_in_reasoning", True)
    monkeypatch.setattr(api_server, "reasoning_dialect", FALLBACK_DIALECT)
    monkeypatch.setattr(api_server, "reasoning_toggle", ("thinking", False, True))
    info = asyncio.run(serve(engine).GetModelInfo(pb.GetModelInfoRequest(), None))
    assert info.implicit_reasoning_opener == FALLBACK_DIALECT.prompt_open_marker
    assert json.loads(info.reasoning_toggle_json) == ["thinking", False, True]


def test_server_info_carries_the_kv_event_sources_body(serve, monkeypatch):
    engine = FakeEngine()
    sources = {"0": {"endpoint": "tcp://*:5557", "topic": "kv", "block_size": 16}}
    engine.core_mgr.kv_event_sources = sources
    monkeypatch.setattr(api_server, "kv_transfer_info_body", dict)
    monkeypatch.setattr(api_server, "server_info_body", dict)
    servicer = serve(engine)
    info = asyncio.run(servicer.GetServerInfo(pb.GetServerInfoRequest(), None))
    assert json.loads(info.kv_event_sources_json) == sources

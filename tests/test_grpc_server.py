# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""AtomEngine's Generate on CPU: ATOM's API server module over a fake engine."""

import asyncio
import itertools
import json
import threading
from types import SimpleNamespace

import pytest

from atom.entrypoints.grpc import atom_grpc_pb2 as pb
from atom.entrypoints.grpc.server import AtomEngineServicer
from atom.entrypoints.openai import api_server
from atom.entrypoints.openai.protocol import DEFAULT_MAX_TOKENS
from atom.model_engine.request import RequestOutput
from atom.model_engine.zmq_protocol import engine_v1_pb2 as v1


class FakeEngine:
    """Registers each preprocessed seq as ATOM's io_processor does; on submit, runs
    `steps`, each a list of (token ids, finish reason or None), one flush apiece."""

    def __init__(self, steps=()):
        self.steps, self.ids = list(steps), itertools.count(1)
        self.config = SimpleNamespace(
            max_model_len=1 << 20, tensor_parallel_size=1, pipeline_parallel_size=1
        )
        self.io_processor = SimpleNamespace(requests={}, preprocess=self.preprocess)
        self.aborted, self.made = [], []
        self.core_mgr = SimpleNamespace(
            add_request=self.add_request,
            abort_request=self.abort_request,
            max_pool_tokens=None,
        )
        self.running = {}  # submitted seqs no step has ended
        self.preprocessed, self.submitted = [], []
        self.release = threading.Event()
        self.release.set()
        self.flush = None  # the servicer's per-step flush

    def preprocess(self, prompt, params, *, stream_callback, **kwargs):
        self.preprocessed.append((prompt, params, kwargs))
        self.release.wait(10)
        seq = SimpleNamespace(
            id=next(self.ids),
            num_prompt_tokens=len(prompt),
            max_tokens=params.max_tokens,
            callback=stream_callback,
        )
        self.io_processor.requests[seq.id] = seq
        self.made.append(seq.id)
        return seq

    def add_request(self, seqs):
        self.submitted += seqs
        for tokens, reason in self.steps:
            for seq in seqs:
                seq.callback(RequestOutput(seq.id, tokens, reason is not None, reason))
            self.flush()
        if not self.steps or self.steps[-1][1] is None:
            self.running.update((seq.id, seq) for seq in seqs)

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


@pytest.mark.parametrize(
    "sampling, message",
    [
        (v1.SamplingParams(n=2, temperature=1.0), "n above 1"),
        (v1.SamplingParams(max_tokens=0), "max_tokens must be at least 1"),
        (v1.SamplingParams(logprobs_flag=True), "logprobs_flag"),
        (v1.SamplingParams(top_logprobs=2), "top_logprobs"),
        (v1.SamplingParams(json_schema="{}"), "json_schema"),
        (v1.SamplingParams(structured_reasoning_end="</think>"), "reasoning_end"),
    ],
)
def test_generate_refuses_what_it_cannot_serve_with_a_400(serve, sampling, message):
    engine = FakeEngine()
    error = _error(_generate(serve(engine), sampling=sampling))
    assert error.http_status == 400 and message in error.message
    assert engine.preprocessed == []


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

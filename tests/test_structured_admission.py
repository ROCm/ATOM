# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import asyncio
import subprocess
import sys
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from atom.structured_output import manager as so
from atom.structured_output import tokenizer_fingerprint
from atom.structured_output.validation import (
    WorkerGrammarPreflight,
)


@pytest.fixture
def worker(monkeypatch):
    from atom.model_engine import llm_engine

    class Head:
        num_embeddings = 8

    monkeypatch.setitem(
        sys.modules, "atom.model_ops.embed_head", NS(ParallelLMHead=Head)
    )
    tokenizer = NS(
        eos_token_id=2, backend_tokenizer=NS(to_str=lambda: '{"vocab":{"a":1}}')
    )
    loader = Mock(return_value=tokenizer)
    monkeypatch.setattr(llm_engine, "load_tokenizer", loader)
    config = NS(
        model="test-model",
        trust_remote_code=False,
        stop_token_ids=[2],
        end_token_ids=(2,),
    )
    runner = NS(config=config, model=NS(modules=lambda: [Head()]))
    compile_schema = Mock()
    constructor = Mock(
        side_effect=lambda tok, vocab, stops, fingerprint: NS(
            fingerprint=fingerprint, vocab_size=vocab, compile=compile_schema
        )
    )
    monkeypatch.setattr(so, "StructuredOutput", constructor)
    return runner, so.tokenizer_fingerprint(tokenizer, [2]), loader, compile_schema


def test_worker_reuses_execution_compiler(worker):
    runner, fingerprint, loader, compile_schema = worker
    result = so._compile_worker_grammar(runner, "{}", None, "", fingerprint, 8)
    controller = runner.structured_output
    so._compile_worker_grammar(runner, '{"type":"string"}', None, "", fingerprint, 8)
    assert runner.structured_output is controller
    assert result == {"fingerprint": fingerprint, "vocab_size": 8}
    loader.assert_called_once_with("test-model", False)
    assert compile_schema.call_count == 2


@pytest.mark.parametrize("fingerprint,vocab", [("different", 8), (None, 16)])
def test_mismatch_fails_before_worker_compilation(worker, fingerprint, vocab):
    runner, actual, _loader, compile_schema = worker
    with pytest.raises(ValueError):
        so._compile_worker_grammar(runner, "{}", None, "", fingerprint or actual, vocab)
    compile_schema.assert_not_called()
    assert not hasattr(runner, "structured_output")


def test_compile_failure_does_not_poison_next_request(worker):
    runner, fingerprint, _loader, compile_schema = worker
    compile_schema.side_effect = [ValueError("unsupported grammar"), None]
    with pytest.raises(ValueError, match="unsupported grammar"):
        so._compile_worker_grammar(runner, "bad", None, "", fingerprint, 8)
    so._compile_worker_grammar(runner, "{}", None, "", fingerprint, 8)
    assert compile_schema.call_count == 2


def test_compile_rpc_returns_while_background_compiler_is_busy(monkeypatch):
    from collections import OrderedDict
    from threading import Event

    entered, release = Event(), Event()
    runner = NS(structured_compile_executor=None, structured_compilations=OrderedDict())

    def compile_worker(*args):
        entered.set()
        assert release.wait(3)
        return {"fingerprint": "fp", "vocab_size": 8}

    monkeypatch.setattr(so, "_compile_worker_grammar", compile_worker)
    try:
        assert so.prepare_worker_grammar(runner, "{}", None, "", "fp", 8) == {
            "ready": False
        }
        assert entered.wait(2)
        # A second poll also returns; the worker can execute another forward.
        assert so.prepare_worker_grammar(runner, "{}", None, "", "fp", 8) == {
            "ready": False
        }
        release.set()
        next(iter(runner.structured_compilations.values())).result(timeout=2)
        assert so.prepare_worker_grammar(runner, "{}", None, "", "fp", 8)["ready"]
    finally:
        release.set()
        runner.structured_compile_executor.shutdown(wait=True)


def frontend(rpc):
    tokenizer = NS(
        eos_token_id=2, backend_tokenizer=NS(to_str=lambda: '{"vocab":{"a":1}}')
    )
    config = NS(hf_config=NS(vocab_size=8), stop_token_ids=[2], end_token_ids=(2,))
    return WorkerGrammarPreflight(
        NS(core_mgr=NS(collective_rpc=rpc)), tokenizer, config
    )


def test_all_rank_success_is_cached_and_tool_format_is_part_of_key():
    rpc = Mock()
    preflight = frontend(rpc)
    value = {
        "fingerprint": tokenizer_fingerprint(preflight.tokenizer, [2]),
        "vocab_size": 8,
    }
    rpc.return_value = [NS(ok=True, value=value), NS(ok=True, value=value)]

    async def run():
        await asyncio.gather(*(preflight.prepare("{}", None) for _ in range(4)))
        await preflight.prepare("{}", "tool-envelope")

    asyncio.run(run())
    assert rpc.call_count == 2
    assert rpc.call_args.kwargs["args"] == (
        "{}",
        "tool-envelope",
        "",
        preflight.fingerprint,
        8,
    )


@pytest.mark.parametrize("kind", ["one_rank_failed", "empty", "mismatch", "timeout"])
def test_failed_preparation_is_never_cached(kind):
    rpc = Mock()
    preflight = frontend(rpc)
    value = {
        "fingerprint": tokenizer_fingerprint(preflight.tokenizer, [2]),
        "vocab_size": 8,
    }
    if kind == "one_rank_failed":
        rpc.return_value = [
            NS(ok=True, value=value),
            NS(ok=False, error="invalid grammar"),
        ]
    elif kind == "empty":
        rpc.return_value = []
    elif kind == "mismatch":
        rpc.return_value = [NS(ok=True, value={**value, "vocab_size": 16})]
    else:
        rpc.side_effect = TimeoutError("timeout")
    with pytest.raises((ValueError, TimeoutError)):
        asyncio.run(preflight.prepare("{}", None))
    assert not preflight.ready
    rpc.side_effect = None
    rpc.return_value = [NS(ok=True, value=value)]
    asyncio.run(preflight.prepare("{}", None))
    assert rpc.call_count == 2


def test_fingerprint_covers_decoder_and_stops():
    a = NS(backend_tokenizer=NS(to_str=lambda: '{"vocab":{"a":1},"decoder":"A"}'))
    b = NS(backend_tokenizer=NS(to_str=lambda: '{"decoder":"A","vocab":{"a":1}}'))
    c = NS(backend_tokenizer=NS(to_str=lambda: '{"vocab":{"a":1},"decoder":"B"}'))
    assert tokenizer_fingerprint(a, [2, 3]) == tokenizer_fingerprint(b, [3, 2])
    assert tokenizer_fingerprint(a, [2]) != tokenizer_fingerprint(c, [2])
    assert tokenizer_fingerprint(a, [2]) != tokenizer_fingerprint(a, [3])


def test_unstructured_request_never_uses_worker_rpc(monkeypatch):
    from atom.entrypoints.openai import api_server

    rpc = Mock(side_effect=AssertionError("ordinary request made an RPC"))
    monkeypatch.setattr(api_server, "engine", NS(core_mgr=NS(collective_rpc=rpc)))
    asyncio.run(api_server._configure_structured_output(NS(), None, {}))
    rpc.assert_not_called()


def test_worker_failure_is_value_error_before_schema_is_attached(monkeypatch):
    from atom.entrypoints.openai import api_server

    rpc = Mock(return_value=[NS(ok=False, error="invalid schema on worker")])
    preflight = frontend(rpc)
    preflight.engine.prepare_structured_request = (
        lambda params, *args: preflight.submit(
            params.json_schema,
            params.structured_tool_format,
            params.structured_reasoning_end,
        ).result()
    )
    monkeypatch.setattr(api_server, "engine", preflight.engine)
    monkeypatch.setattr(api_server, "tokenizer", preflight.tokenizer)
    monkeypatch.setattr(
        api_server, "_get_engine_config", lambda: NS(speculative_config=None)
    )
    monkeypatch.setattr(api_server, "synthetic_token_text", None)
    monkeypatch.setattr(api_server, "tool_call_parser_cls", None)
    monkeypatch.setattr(
        api_server._configure_structured_output, "preflight", preflight, raising=False
    )
    params = NS(ignore_eos=False, stop_strings=None, logprobs=False)
    from atom.entrypoints.openai.reasoning import NO_REASONING

    with pytest.raises(ValueError, match="Worker JSON Schema preparation failed"):
        asyncio.run(api_server._configure_structured_output(params, "{}", NO_REASONING))
    assert not preflight.ready


@pytest.mark.parametrize("opened", [False, True])
def test_admission_uses_actual_channel_state(monkeypatch, opened):
    from atom.entrypoints.openai import api_server
    from atom.entrypoints.openai.reasoning import ReasoningChannel

    rpc = Mock()
    preflight = frontend(rpc)
    preflight.tokenizer.encode = lambda text, **kw: list(text.encode())
    rpc.return_value = [
        NS(
            ok=True,
            value={
                "fingerprint": tokenizer_fingerprint(preflight.tokenizer, [2]),
                "vocab_size": 8,
            },
        )
    ]
    preflight.engine.prepare_structured_request = (
        lambda params, *args: preflight.submit(
            params.json_schema,
            params.structured_tool_format,
            params.structured_reasoning_end,
        ).result()
    )
    monkeypatch.setattr(api_server, "engine", preflight.engine)
    monkeypatch.setattr(api_server, "tokenizer", preflight.tokenizer)
    monkeypatch.setattr(
        api_server, "_get_engine_config", lambda: NS(speculative_config=None)
    )
    monkeypatch.setattr(api_server, "synthetic_token_text", None)
    monkeypatch.setattr(api_server, "tool_call_parser_cls", None)
    monkeypatch.setattr(
        api_server._configure_structured_output, "preflight", preflight, raising=False
    )
    params = NS(ignore_eos=False, stop_strings=None, logprobs=False)
    asyncio.run(
        api_server._configure_structured_output(
            params, "{}", ReasoningChannel(starts_open=opened)
        )
    )
    assert params.structured_reasoning_end == ("</think>" if opened else "")
    rpc.assert_called_once()


def test_framed_channel_rejected_before_worker_rpc(monkeypatch):
    from atom.entrypoints.openai import api_server
    from atom.entrypoints.openai.reasoning import ReasoningChannel
    from atom.entrypoints.openai.reasoning_dialects import resolve_dialect

    rpc = Mock(side_effect=AssertionError("unsupported channel must fail before RPC"))
    monkeypatch.setattr(api_server, "engine", NS(core_mgr=NS(collective_rpc=rpc)))
    monkeypatch.setattr(
        api_server, "_get_engine_config", lambda: NS(speculative_config=None)
    )
    monkeypatch.setattr(api_server, "synthetic_token_text", None)
    dialect, _ = resolve_dialect("<|open|>think<|sep|>")
    params = NS(ignore_eos=False, stop_strings=None, logprobs=False)
    with pytest.raises(ValueError, match="framed reasoning channels"):
        asyncio.run(
            api_server._configure_structured_output(
                params, "{}", ReasoningChannel(dialect, True)
            )
        )
    rpc.assert_not_called()
    assert not hasattr(params, "json_schema")


def test_cached_schema_bypasses_unrelated_slow_compile():
    from threading import Event

    entered, release = Event(), Event()
    preflight = frontend(Mock())
    ready = {
        "ready": True,
        "fingerprint": tokenizer_fingerprint(preflight.tokenizer, [2]),
        "vocab_size": 8,
    }

    def rpc(method, *, args, timeout):
        if args[0] == "slow":
            entered.set()
            assert release.wait(3)
        return [NS(ok=True, value=ready)]

    preflight.engine.core_mgr.collective_rpc = rpc

    async def run():
        await preflight.prepare("cached", None)
        slow = asyncio.create_task(preflight.prepare("slow", None))
        assert await asyncio.to_thread(entered.wait, 2)
        try:
            await asyncio.wait_for(preflight.prepare("cached", None), 0.5)
            assert not slow.done()
        finally:
            release.set()
        await slow

    try:
        asyncio.run(run())
    finally:
        release.set()
        preflight.close()


# A JSON Schema answer, or strict tool arguments alone: both are structured.
STRUCTURED = [
    pytest.param({"json_schema": "{}"}, id="json_schema"),
    pytest.param(
        {"json_schema": None, "structured_tool_format": "{}"}, id="strict_tools"
    ),
]


@pytest.mark.parametrize("fields", STRUCTURED)
def test_engine_boundary_checks_structured_requests_before_tokenization(fields):
    from atom.model_engine.llm_engine import InputOutputProcessor
    from atom.sampling_params import SamplingParams

    processor = InputOutputProcessor.__new__(InputOutputProcessor)
    processor.config = NS()
    processor.structured_admission = Mock(side_effect=ValueError("admission rejected"))
    with pytest.raises(ValueError, match="admission rejected"):
        processor.preprocess_fanout("prompt", SamplingParams(**fields))
    processor.structured_admission.assert_called_once()


def test_speculative_schema_logprobs_refused_instead_of_wrong_distribution():
    from atom.sampling_params import SamplingParams
    from atom.structured_output.validation import validate_execution_options

    params = SamplingParams(json_schema="{}", logprobs=True)
    config = NS(
        speculative_config=NS(num_speculative_tokens=5, synthetic_acceptance_rates=None)
    )
    with pytest.raises(
        ValueError, match="JSON Schema, speculative decoding and logprobs"
    ):
        validate_execution_options(config, params)
    params.logprobs = False
    validate_execution_options(config, params)


"""Keep optional structured-output dependencies off ordinary engine imports."""


def test_package_entry_points_do_not_load_backend_or_gpu_helpers():
    code = """
import sys
import atom.structured_output as so

assert "prepare_worker_grammar" in so.__all__
assert "StructuredOutput" in so.__all__
assert not any(n == "llguidance" or n.startswith("llguidance.") for n in sys.modules)
assert all(
    "atom.structured_output." + n not in sys.modules
    for n in (
        "backend_guidance", "utils", "schema", "validation",
    )
)
assert "atom.model_ops.triton_structured_output" not in sys.modules
assert "atom.entrypoints.openai.api_server" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )


def test_plain_format_validation_keeps_optional_dependencies_unloaded():
    code = """
import sys
from atom.structured_output.validation import schema_from_format, validate_output
assert schema_from_format(None) is None
assert schema_from_format({"type": "text"}) is None
validate_output(None, "ordinary answer", "stop")
assert "atom.structured_output.schema" not in sys.modules
assert "atom.structured_output.utils" not in sys.modules
assert all(name not in sys.modules for name in ("torch", "llguidance", "jsonschema", "re2"))
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )


def test_cleanup_method_is_allowed_by_collective_rpc():
    from atom.model_engine.collective_rpc import COLLECTIVE_RPC_METHODS

    assert "release_structured_requests" in COLLECTIVE_RPC_METHODS


def test_cold_compile_cache_hit_prunes_completed_futures_without_losing_active_work():
    from collections import OrderedDict
    from concurrent.futures import Future

    active = Future()
    records = OrderedDict([(0, active)])
    controller = NS(
        fingerprint="fp", vocab_size=8, grammars={("{}", None, ""): object()}, states={}
    )
    runner = NS(structured_output=controller, structured_compilations=records)
    for i in range(1, 300):
        done = Future()
        done.set_result(None)
        records[i] = done
        assert so.prepare_worker_grammar(runner, "{}", None, "", "fp", 8)["ready"]
        assert len(records) <= 128
        assert records[0] is active


@pytest.mark.parametrize("fields", STRUCTURED)
@pytest.mark.parametrize("failure", [False, True])
def test_submission_owns_matchers_and_releases_all_replicas(failure, fields):
    from atom.model_engine.engine_core_mgr import CoreManager

    mgr = CoreManager.__new__(CoreManager)
    replicas = [{}, {}]
    events = []
    seq = NS(id=77, leave_reason="stop", **fields)

    def prepare(seqs):
        events.append("prepare")
        for replica in replicas:
            replica[77] = object()
        if failure:
            raise ValueError("one rank failed")

    mgr.structured_request_prepare = prepare
    mgr._add_prepared_requests = lambda seqs: events.append("enqueue")

    def broadcast(cmd, **kw):
        assert cmd == "release_structured_requests"
        for replica in replicas:
            for i in kw["request_ids"]:
                replica.pop(i, None)

    mgr.broadcast_utility_command = broadcast
    mgr._seq_id_to_callback = {}
    if failure:
        with pytest.raises(ValueError):
            mgr.add_request([seq])
        assert events == ["prepare"]
    else:
        mgr.add_request([seq])
        assert events == ["prepare", "enqueue"]
        assert all(77 in r for r in replicas)
        mgr._deliver_terminal_output(seq)
        mgr._deliver_terminal_output(seq)  # completion cleanup is idempotent
    assert all(not replica for replica in replicas)


@pytest.mark.parametrize("fields", STRUCTURED)
def test_api_cancel_waits_for_submission_before_aborting(monkeypatch, fields):
    import threading

    from atom.entrypoints.openai import api_server

    entered, release = threading.Event(), threading.Event()
    events = []
    seq = NS(id=77, **fields)

    def submit(seqs):
        entered.set()
        assert release.wait(3)
        events.append("enqueued")

    engine = NS(
        core_mgr=NS(
            add_request=submit, abort_request=lambda i: events.append("aborted")
        ),
        io_processor=NS(requests={77: seq}),
    )
    monkeypatch.setattr(api_server, "engine", engine)

    async def run():
        task = asyncio.create_task(api_server._submit_engine_requests([seq]))
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        asyncio.run(run())
    finally:
        release.set()
    assert events == ["enqueued", "aborted"]
    assert not engine.io_processor.requests


@pytest.mark.parametrize("fields", STRUCTURED)
@pytest.mark.parametrize("pp_size", [2, 4])
def test_pipeline_schema_rejected_before_engine_preflight(pp_size, fields):
    from atom.model_engine.llm_engine import LLMEngine
    from atom.sampling_params import SamplingParams

    engine = LLMEngine.__new__(LLMEngine)
    engine.config = NS(pipeline_parallel_size=pp_size)
    engine.core_mgr = NS(collective_rpc=Mock())
    with pytest.raises(ValueError, match="pipeline parallelism"):
        engine.prepare_structured_request(SamplingParams(**fields))
    engine.prepare_structured_request(SamplingParams())
    engine.core_mgr.collective_rpc.assert_not_called()


@pytest.mark.parametrize("pp_size", [2, 4])
def test_pipeline_schema_rejected_before_api_preflight(pp_size):
    from atom.sampling_params import SamplingParams
    from atom.structured_output.validation import configure_request

    engine = NS(prepare_structured_request=Mock())
    for schema in ("{}", None):
        request = configure_request(
            engine,
            None,
            NS(pipeline_parallel_size=pp_size),
            SamplingParams(),
            schema,
            None,
        )
        if schema is None:
            asyncio.run(request)
        else:
            with pytest.raises(ValueError, match="pipeline parallelism"):
                asyncio.run(request)
    engine.prepare_structured_request.assert_not_called()


@pytest.mark.parametrize("with_callback", [False, True])
def test_cleanup_send_failure_does_not_stop_terminal_deliveries(with_callback):
    from atom.model_engine.engine_core_mgr import CoreManager

    mgr = CoreManager.__new__(CoreManager)
    mgr.label = "test"
    first, second = Mock(), Mock()
    mgr._seq_id_to_callback = {78: second}
    if with_callback:
        mgr._seq_id_to_callback[77] = first
    mgr.broadcast_utility_command = Mock(
        side_effect=[RuntimeError("socket closed"), None, None]
    )
    assert (
        mgr._deliver_terminal_output(NS(id=77, leave_reason="stop", json_schema="{}"))
        == with_callback
    )
    assert mgr._deliver_terminal_output(
        NS(id=78, leave_reason="stop", json_schema="{}")
    )
    assert first.call_count == int(with_callback)
    second.assert_called_once()
    assert second.call_args.args[0].finished
    mgr._release_structured_requests([NS(id=77, json_schema="{}")])
    assert not mgr._seq_id_to_callback


def test_bind_groups_matching_grammars_without_merging_request_states():
    from atom.model_engine.llm_engine import LLMEngine

    engine = LLMEngine.__new__(LLMEngine)
    engine._structured_preflight = Mock()

    def seq(i, schema="{}", tool=None, reasoning=""):
        return NS(
            id=i,
            json_schema=schema,
            structured_tool_format=tool,
            structured_reasoning_end=reasoning,
        )

    engine.bind_structured_requests(
        [
            seq(1),
            seq(2, schema='{"type":"object"}'),
            seq(3),
            seq(4, tool="tool"),
            seq(5, reasoning="</think>"),
            seq(6),
        ]
    )
    calls = engine._structured_preflight.prepare_requests.call_args_list
    assert [c.args for c in calls] == [
        (("{}", None, ""), (1, 3, 6)),
        (('{"type":"object"}', None, ""), (2,)),
        (("{}", "tool", ""), (4,)),
        (("{}", None, "</think>"), (5,)),
    ]


@pytest.mark.parametrize(
    "fields", [pytest.param({"json_schema": None}, id="plain"), *STRUCTURED]
)
@pytest.mark.parametrize("pending", [False, True])
def test_cancellation_cleanup_follows_terminal_sequence(fields, pending):
    from atom.model_engine.engine_core_mgr import CoreManager

    mgr = CoreManager.__new__(CoreManager)
    mgr.label = "test"
    seq = NS(id=19, leave_reason="aborted", **fields)
    callback = Mock()
    mgr._seq_id_to_callback = {19: callback}
    mgr._release_seq_load = Mock()
    mgr._bp_take_pending = Mock(return_value=seq if pending else None)
    mgr.broadcast_utility_command = Mock()
    mgr.abort_request(19)
    if not pending:
        mgr.broadcast_utility_command.assert_called_once_with(
            "abort_request", req_id=19
        )
        callback.assert_not_called()
        mgr._deliver_terminal_output(seq)
    callback.assert_called_once()
    releases = [
        c
        for c in mgr.broadcast_utility_command.call_args_list
        if c.args[0] == "release_structured_requests"
    ]
    assert len(releases) == int(fields != {"json_schema": None})
    if releases:
        assert releases[0].kwargs == {"request_ids": [19]}


def test_cleanup_failure_preserves_original_admission_error():
    from atom.model_engine.engine_core_mgr import CoreManager

    mgr = CoreManager.__new__(CoreManager)
    mgr.label = "test"
    mgr.structured_request_prepare = Mock(side_effect=ValueError("compile failed"))
    mgr.broadcast_utility_command = Mock(side_effect=RuntimeError("socket closed"))
    mgr._add_prepared_requests = Mock()
    with pytest.raises(ValueError, match="compile failed"):
        mgr.add_request([NS(id=19, json_schema="{}")])
    mgr._add_prepared_requests.assert_not_called()
    mgr.broadcast_utility_command.assert_called_once_with(
        "release_structured_requests", request_ids=[19]
    )


def test_preflight_construction_and_close_keep_optional_dependencies_unloaded():
    code = """
import sys
from atom.structured_output.validation import WorkerGrammarPreflight

# Deliberately opaque inputs: construction/close must not inspect tokenizer
# or model configuration before any structured request arrives.
preflight = WorkerGrammarPreflight(object(), object(), object())
preflight.close()
assert all(name not in sys.modules for name in ("torch", "llguidance", "jsonschema", "re2"))
assert "atom.structured_output.manager" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )


def test_concurrent_first_requests_initialize_one_preflight_identity(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from atom import structured_output

    fingerprint = Mock(return_value="identity")
    monkeypatch.setattr(structured_output, "tokenizer_fingerprint", fingerprint)
    rpc = Mock(
        return_value=[NS(ok=True, value={"fingerprint": "identity", "vocab_size": 8})]
    )
    preflight = frontend(rpc)
    gate = Barrier(8)

    def submit():
        gate.wait(timeout=3)
        preflight.submit("{}", None).result(timeout=3)

    try:
        fingerprint.assert_not_called()
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(lambda _: submit(), range(8)))
        fingerprint.assert_called_once()
        rpc.assert_called_once()
    finally:
        preflight.close()


def test_failed_identity_initialization_can_retry(monkeypatch):
    from atom import structured_output

    fingerprint = Mock(side_effect=[ValueError("bad tokenizer"), "identity"])
    monkeypatch.setattr(structured_output, "tokenizer_fingerprint", fingerprint)
    rpc = Mock(
        return_value=[NS(ok=True, value={"fingerprint": "identity", "vocab_size": 8})]
    )
    preflight = frontend(rpc)
    try:
        with pytest.raises(ValueError, match="bad tokenizer"):
            preflight.submit("{}", None)
        assert preflight.fingerprint is None
        assert not preflight.pending
        rpc.assert_not_called()
        preflight.submit("{}", None).result(timeout=3)
        assert preflight.fingerprint == "identity"
        rpc.assert_called_once()
    finally:
        preflight.close()


@pytest.mark.parametrize("fields", STRUCTURED)
def test_a_scheduled_batch_carries_every_structured_request(
    scheduler, seq_factory, fields
):
    """The worker masks what the batch hands it: a request constrained by
    strict tool arguments alone must be in it, or it decodes unconstrained."""
    from atom.sampling_params import SamplingParams

    seq = seq_factory([1, 2, 3, 4], sampling_params=SamplingParams(**fields))
    scheduler.add(seq)
    batch, _ = scheduler.schedule()
    [request] = batch.structured_requests.values()
    assert request.schema[:2] == (
        fields["json_schema"],
        fields.get("structured_tool_format"),
    )


@pytest.mark.parametrize("fields", STRUCTURED)
def test_a_finished_structured_request_releases_its_grammar(seq_factory, fields):
    from atom.model_engine.engine_core import EngineCore
    from atom.sampling_params import SamplingParams

    seq = seq_factory([1, 2, 3, 4], sampling_params=SamplingParams(**fields))
    calls = []
    core = NS(runner_mgr=NS(call_func=lambda *args: calls.append(args)))
    EngineCore._release_request_resources(core, [seq])
    assert ("release_structured_requests", [seq.id]) in calls

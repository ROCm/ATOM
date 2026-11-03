# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import asyncio
import json
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from atom.entrypoints.openai.reasoning import ReasoningChannel
from atom.entrypoints.openai.sse import data_frame
from atom.structured_output.validation import (
    schema_from_format,
    structured_reasoning_marker,
    structured_response_channel,
    validate_chat_stream,
    validate_output,
    validate_responses_output,
)

pytest.importorskip("jsonschema")


@pytest.mark.parametrize(
    "name", ["$ref", "$dynamicRef", "const", "enum", "default", "examples"]
)
@pytest.mark.parametrize("position", ["properties", "$defs", "dependentSchemas"])
def test_schema_map_names_are_not_keywords(name, position):
    schema = {position: {name: {"type": "string"}}}
    fmt = {"type": "json_schema", "name": "test", "schema": schema}
    assert json.loads(schema_from_format(fmt, responses=True)) == schema
    schema[position][name] = {"$ref": "https://invalid.example/schema"}
    with pytest.raises(ValueError, match="Only local"):
        schema_from_format(fmt, responses=True)


@pytest.mark.parametrize("keyword", ["const", "enum", "default", "examples"])
def test_literal_objects_do_not_trigger_reference_checks(keyword):
    literal = {"$ref": "https://invalid.example/literal"}
    value = [literal] if keyword in {"enum", "examples"} else literal
    fmt = {"type": "json_schema", "name": "literal", "schema": {keyword: value}}
    schema_from_format(fmt, responses=True)


@pytest.mark.parametrize(
    "prompt", ["assistant", "assistant<think></think>", "assistant<think>"]
)
@pytest.mark.parametrize("tokenized", [False, True])
def test_grammar_activation_uses_rendered_prompt(prompt, tokenized):
    from atom.entrypoints.openai.reasoning import (
        ReasoningChannel,
        prompt_starts_in_reasoning,
        prompt_tokens_start_in_reasoning,
    )

    tokenizer = NS(
        encode=lambda value, **kw: list(value.encode()),
        decode=lambda ids: bytes(ids).decode(),
    )
    opened = (
        prompt_tokens_start_in_reasoning(list(prompt.encode()), tokenizer.decode)
        if tokenized
        else prompt_starts_in_reasoning(prompt)
    )
    channel = ReasoningChannel(starts_open=opened)
    assert structured_reasoning_marker(channel, tokenizer) == (
        "</think>" if prompt.endswith("<think>") else ""
    )


@pytest.mark.parametrize("opened", [False, True])
def test_framed_channels_fail_admission_even_without_explicit_think_close(opened):
    from atom.entrypoints.openai.reasoning import ReasoningChannel
    from atom.entrypoints.openai.reasoning_dialects import resolve_dialect

    dialect, _ = resolve_dialect("<|open|>think<|sep|>")
    with pytest.raises(ValueError, match="framed reasoning channels"):
        structured_reasoning_marker(ReasoningChannel(dialect, opened), object())


@pytest.mark.parametrize(
    "choice",
    [
        "auto",
        "required",
        {"function": {"name": "files::lookup"}},
        {"function": {"name": "lookup"}, "namespace": "files"},
    ],
)
def test_namespaced_tools_keep_dispatch_identity(choice):
    from atom.entrypoints.openai.tool_parser.deepseekv41_tool_parser import (
        DsmlV41Parser,
    )
    from atom.structured_output.validation import (
        structured_tool_format,
    )

    tools = [
        {"function": {"name": "lookup"}, "namespace": ns} for ns in ("search", "files")
    ]
    fmt = json.loads(structured_tool_format(DsmlV41Parser, tools, choice))
    assert fmt["names"] == (
        ["files::lookup"]
        if isinstance(choice, dict)
        else ["search::lookup", "files::lookup"]
    )


@pytest.mark.parametrize("responses", [False, True])
def test_nested_schema(responses):
    schema = {
        "type": "object",
        "properties": {
            "items": {"type": "array", "items": {"enum": ["red", "blue"]}},
            "ok": {"type": "boolean"},
        },
        "required": ["items", "ok"],
        "additionalProperties": False,
    }
    spec = {"name": "answer", "schema": schema}
    fmt = (
        {"type": "json_schema", **spec}
        if responses
        else {"type": "json_schema", "json_schema": spec}
    )
    compiled = schema_from_format(fmt, responses=responses)
    validate_output(compiled, '{"items":["red"],"ok":true}', "stop")
    for bad in [
        "{}",
        '{"items":["green"],"ok":true}',
        '{"items":[],"ok":true,"extra":1}',
    ]:
        with pytest.raises(RuntimeError):
            validate_output(compiled, bad, "stop")


@pytest.mark.parametrize(
    "fmt",
    [
        {},
        {"type": "other"},
        {"type": "json_schema"},
        {"type": "json_schema", "name": "x", "schema": {"type": "bogus"}},
        {
            "type": "json_schema",
            "name": "x",
            "schema": {"$ref": "https://example.com/schema"},
        },
    ],
)
def test_invalid_schema_rejected(fmt):
    with pytest.raises(ValueError):
        schema_from_format(fmt, responses=True)


def test_length_is_incomplete_not_a_false_success():
    validate_output('{"type":"object"}', '{"x":', "length")
    with pytest.raises(RuntimeError):
        validate_output('{"type":"object"}', '{"x":', "stop")


def test_tool_only_turn_has_no_text_but_text_still_must_conform():
    validate_output('{"type":"object"}', "", "tool_calls", tool_calls=True)
    with pytest.raises(RuntimeError):
        validate_output('{"type":"object"}', "not JSON", "tool_calls", tool_calls=True)


def test_tool_format_matches_the_selected_parser_and_tool_name():
    import json

    from atom.entrypoints.openai.tool_parser.deepseekv41_tool_parser import (
        DsmlV41Parser,
    )
    from atom.structured_output.validation import (
        structured_tool_format,
    )

    tools = [{"type": "function", "function": {"name": "weather"}}]
    assert structured_tool_format(DsmlV41Parser, tools, "none") is None
    spec = json.loads(
        structured_tool_format(
            DsmlV41Parser, tools, {"type": "function", "function": {"name": "weather"}}
        )
    )
    assert spec["prefix"] == "｜DSML｜ " and spec["section"] == "calls"
    assert spec["names"] == ["weather"] and spec["required"]
    with pytest.raises(ValueError, match="declared"):
        structured_tool_format(
            DsmlV41Parser, tools, {"type": "function", "function": {"name": "unknown"}}
        )


def test_stream_failure_withholds_success_and_closes_source():
    closed = []

    async def source():
        try:
            yield data_frame(
                {"choices": [{"index": 0, "delta": {"content": "invalid"}}]}
            )
            yield data_frame(
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            )
        finally:
            closed.append(True)

    async def run():
        return [
            item async for item in validate_chat_stream(source(), '{"type":"object"}')
        ]

    frames = "".join(asyncio.run(run()))
    assert "structured_output_error" in frames
    assert "finish_reason" not in frames
    assert frames.count("[DONE]") == 1
    assert closed == [True]


@pytest.mark.parametrize(
    "kind", ["function_call", "custom_tool_call", "tool_search_call"]
)
def test_tool_only_responses_skip_answer_validation_after_rewriting(kind):
    validate_responses_output(
        '{"type":"object"}', {"output": [{"type": kind}]}, "tool_calls"
    )
    # A tool call must not excuse invalid visible answer text.
    with pytest.raises(RuntimeError):
        validate_responses_output(
            '{"type":"object"}',
            {
                "output": [
                    {"type": kind},
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "bad"}],
                    },
                ]
            },
            "stop",
        )


@pytest.mark.parametrize("opened", [False, True])
@pytest.mark.parametrize("width", [1, 2, 7, 1024])
def test_grammar_answer_keeps_literal_reasoning_tags(opened, width):
    answer = json.dumps({"v": "<think>literal</think>"})
    raw = ("initial reasoning</think>" if opened else "") + answer
    channel = structured_response_channel(ReasoningChannel(starts_open=opened), "{}")
    expected_reasoning = "initial reasoning" if opened else None
    assert channel.split(raw) == (expected_reasoning, answer)
    reader = channel.stream()
    events = []
    for start in range(0, len(raw), width):
        events.extend(reader.process(raw[start : start + width]))
    events.extend(reader.flush())
    assert "".join(v for k, v in events if k == "content") == answer
    assert (
        "".join(v for k, v in events if k == "reasoning_content") or None
    ) == expected_reasoning


def test_unstructured_channel_is_unchanged():
    channel = ReasoningChannel()
    assert structured_response_channel(channel, None) is channel


def test_relaxed_mtp_rejected_before_frontend_rpc(monkeypatch):
    from atom.entrypoints.openai import api_server
    from atom.utils import envs

    monkeypatch.setattr(envs, "ATOM_ENABLE_RELAXED_MTP", True)
    rpc = Mock(side_effect=AssertionError("request must fail before RPC"))
    monkeypatch.setattr(api_server, "engine", NS(core_mgr=NS(collective_rpc=rpc)))
    with pytest.raises(ValueError, match="relaxed MTP"):
        asyncio.run(
            api_server._configure_structured_output(NS(), "{}", ReasoningChannel())
        )
    asyncio.run(api_server._configure_structured_output(NS(), None, ReasoningChannel()))
    rpc.assert_not_called()


def test_relaxed_mtp_rejected_on_execution_worker(monkeypatch):
    from atom.structured_output.manager import _compile_worker_grammar
    from atom.utils import envs

    monkeypatch.setattr(envs, "ATOM_ENABLE_RELAXED_MTP", True)
    with pytest.raises(ValueError, match="relaxed MTP"):
        _compile_worker_grammar(NS(), "{}", None, "", "fingerprint", 10)


@pytest.mark.parametrize("text", ['{"answer":42}', ""])
@pytest.mark.parametrize("multi", [False, True])
def test_chat_rejects_raw_grammar_error_before_success_normalization(text, multi):
    from atom.entrypoints.openai.serving_chat import (
        build_chat_response,
        build_chat_response_multi,
    )

    output = {
        "text": text,
        "finish_reason": "structured_output_error",
        "num_tokens_input": 1,
        "num_tokens_output": 1,
    }
    with pytest.raises(RuntimeError, match="Structured decoding failed"):
        if multi:
            build_chat_response_multi(
                "id", "model", [dict(output, finish_reason="stop"), output]
            )
        else:
            build_chat_response("id", "model", text, output)


@pytest.mark.parametrize("multi", [False, True])
def test_chat_stream_grammar_error_never_emits_success_even_after_valid_json(multi):
    from atom.entrypoints.openai.serving_chat import (
        stream_chat_response,
        stream_chat_response_fanout,
    )
    from atom.entrypoints.openai.streaming_dispatch import StreamOutputCollector

    async def run():
        collector = StreamOutputCollector("id")
        cleaned = []
        good = {"text": '{"answer":42}', "token_ids": [1], "finished": False}
        error = {
            "text": "",
            "token_ids": [],
            "finished": True,
            "finish_reason": "structured_output_error",
        }
        if multi:
            collector.put_nowait((0, good))
            collector.put_nowait((1, error))
            source = stream_chat_response_fanout(
                "id",
                "model",
                collector,
                [1, 2],
                1,
                lambda sid, **kw: cleaned.append(sid),
                lambda *a: None,
            )
        else:
            collector.put_nowait(good)
            collector.put_nowait(error)
            source = stream_chat_response(
                "id",
                "model",
                collector,
                1,
                1,
                lambda sid, **kw: cleaned.append(sid),
                lambda *a: None,
            )
        frames = [f async for f in validate_chat_stream(source, '{"type":"object"}')]
        assert set(cleaned) == ({1, 2} if multi else {1})
        raw = "".join(frames)
        assert "structured_output_error" in raw
        assert '"finish_reason": "stop"' not in raw
        assert '"finish_reason": "tool_calls"' not in raw
        assert raw.count("[DONE]") == 1

    asyncio.run(run())

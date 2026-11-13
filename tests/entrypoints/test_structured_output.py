# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import asyncio
import json
import re
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
    assert [tool["name"] for tool in fmt["tools"]] == (
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
    assert spec["tools"] == [{"name": "weather", "parameters": None}]
    assert spec["required"]
    # Free text may contain nothing the parser reads as a call's start.
    assert spec["markers"] == list(DsmlV41Parser.START_MARKERS)
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
    assert channel.split(raw, stopped=True) == (expected_reasoning, answer)
    reader = channel.stream()
    events = []
    for start in range(0, len(raw), width):
        events.extend(reader.process(raw[start : start + width]))
    events.extend(reader.flush(stopped=True))
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


STRICT = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}


def _function(parameters=STRICT, strict=True):
    return {
        "type": "function",
        "function": {"name": "weather", "strict": strict, "parameters": parameters},
    }


def test_strict_tool_schemas_keep_only_strict_functions():
    from atom.structured_output.validation import strict_tool_schemas

    tools = [
        _function(),
        {
            **_function(strict=False),
            "function": {**_function(strict=False)["function"], "name": "other"},
        },
    ]
    assert strict_tool_schemas(tools) == {"weather": STRICT}


@pytest.mark.parametrize(
    "parameters, strict, message",
    [
        ({**STRICT, "required": []}, True, "required"),
        (
            {k: v for k, v in STRICT.items() if k != "additionalProperties"},
            True,
            "additionalProperties",
        ),
        (
            {**STRICT, "properties": {"city": {"type": "string", "maxLength": 3}}},
            True,
            "maxLength",
        ),
        (
            {
                **STRICT,
                "properties": {
                    "city": {"type": "object", "properties": {"x": {"type": "integer"}}}
                },
            },
            True,
            "additionalProperties",
        ),
        ({"type": "array"}, True, "object"),
        (STRICT, "yes", "boolean"),
    ],
)
def test_strict_parameters_follow_deepseek_rules(parameters, strict, message):
    from atom.structured_output.validation import strict_tool_schemas

    with pytest.raises(ValueError, match=message):
        strict_tool_schemas([_function(parameters, strict)])


def test_property_named_like_a_keyword_is_not_a_keyword():
    from atom.structured_output.validation import strict_tool_schemas

    parameters = {
        **STRICT,
        "properties": {"maxLength": {"type": "integer"}},
        "required": ["maxLength"],
    }
    assert strict_tool_schemas([_function(parameters)])


def _call(arguments):
    return {"function": {"name": "weather", "arguments": arguments}}


def test_strict_tool_calls_are_validated_after_generation():
    from atom.structured_output.validation import validate_tool_calls

    schemas = {"weather": STRICT}
    validate_tool_calls(schemas, [_call('{"city": "Paris"}')], "tool_calls")
    # A truncated reply is reported as such, not as a schema violation.
    validate_tool_calls(schemas, [_call('{"city": ')], "length")
    for bad in ('{"city": 3}', '{"city": "a", "x": 1}', "not json"):
        with pytest.raises(RuntimeError, match="strict"):
            validate_tool_calls(schemas, [_call(bad)], "tool_calls")


def test_strict_tools_alone_make_a_request_structured():
    from atom.structured_output import is_structured

    assert not is_structured(NS(json_schema=None, structured_tool_format=None))
    assert is_structured(NS(json_schema="{}", structured_tool_format=None))
    assert is_structured(NS(json_schema=None, structured_tool_format="{}"))


@pytest.mark.parametrize(
    "schema, good, bad",
    [
        ({"type": "string", "pattern": "^[0-9]{5}$"}, "12345", "1234"),
        ({"type": "integer", "multipleOf": 2}, 4, 3),
        ({"type": "number", "exclusiveMinimum": 0}, 1, 0),
        ({"anyOf": [{"type": "string"}, {"type": "integer"}]}, 7, [1]),
    ],
)
def test_strict_arguments_are_rechecked_per_keyword(schema, good, bad):
    from atom.structured_output.validation import validate_tool_calls

    schemas = {"weather": {**STRICT, "properties": {"city": schema}}}
    validate_tool_calls(schemas, [_call(json.dumps({"city": good}))], "tool_calls")
    with pytest.raises(RuntimeError, match="strict"):
        validate_tool_calls(schemas, [_call(json.dumps({"city": bad}))], "tool_calls")


def test_def_definitions_follow_the_strict_rules_too():
    from atom.structured_output.validation import strict_tool_schemas

    author = {
        "type": "object",
        "properties": {"name": {"type": "string", "maxLength": 3}},
        "required": ["name"],
        "additionalProperties": False,
    }
    parameters = {
        **STRICT,
        "properties": {"a": {"$ref": "#/$def/author"}},
        "required": ["a"],
        "$def": {"author": author},
    }
    with pytest.raises(ValueError, match="maxLength"):
        strict_tool_schemas([_function(parameters)])


def test_def_is_read_as_defs_with_its_references():
    from atom.structured_output.validation import strict_tool_schemas

    parameters = {
        **STRICT,
        "properties": {"a": {"$ref": "#/$def/author"}},
        "required": ["a"],
        "$def": {"author": STRICT},
    }
    [canonical] = strict_tool_schemas([_function(parameters)]).values()
    assert "$def" not in canonical
    assert canonical["$defs"] == {"author": STRICT}
    assert canonical["properties"]["a"] == {"$ref": "#/$defs/author"}


@pytest.mark.parametrize(
    "parameters, message",
    [
        # Each parameter is its own grammar: `#` would name the parameter.
        ({**STRICT, "properties": {"city": {"$ref": "#"}}}, "ref"),
        ({**STRICT, "properties": {"city": {"$ref": "#/properties/city"}}}, "ref"),
        ({**STRICT, "properties": {"city": {**STRICT, "$def": {"x": STRICT}}}}, "top"),
        ({**STRICT, "$def": {"x": STRICT}, "$defs": {"x": STRICT}}, "both"),
        ({**STRICT, "properties": {"city": {"type": "string", "$id": "x"}}}, "id"),
    ],
)
def test_strict_references_stay_within_the_definitions(parameters, message):
    from atom.structured_output.validation import strict_tool_schemas

    with pytest.raises(ValueError, match=message):
        strict_tool_schemas([_function(parameters)])


def _tool_call_frames(arguments):
    call = {"index": 0, "id": "c", "type": "function"}
    return [
        data_frame({"choices": [{"index": 0, "delta": {"content": "Checking."}}]}),
        data_frame(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    **call,
                                    "function": {"name": "weather", "arguments": ""},
                                }
                            ]
                        },
                    }
                ]
            }
        ),
        data_frame(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": arguments}}
                            ]
                        },
                    }
                ]
            }
        ),
        data_frame(
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
        ),
        "data: [DONE]\n\n",
    ]


def _validated_stream(arguments):
    async def source():
        for frame in _tool_call_frames(arguments):
            yield frame

    async def run():
        return [
            f async for f in validate_chat_stream(source(), None, {"weather": STRICT})
        ]

    return asyncio.run(run())


def test_a_valid_strict_call_streams_unchanged():
    good = '{"city": "Paris"}'
    assert _validated_stream(good) == _tool_call_frames(good)


def test_an_invalid_strict_call_never_reaches_the_client():
    """As a non-streamed reply fails before anything is sent: the text before
    the call goes out as it came, the call does not."""
    raw = "".join(_validated_stream('{"city": 3}'))
    assert "Checking." in raw
    assert "structured_output_error" in raw
    assert '"tool_calls"' not in raw and "finish_reason" not in raw
    assert raw.count("[DONE]") == 1


def test_a_null_strict_is_unset():
    from atom.structured_output.validation import strict_tool_schemas

    assert strict_tool_schemas([_function(STRICT, None)]) == {}


@pytest.mark.parametrize(
    "keyword, value",
    [
        ("allOf", [{"properties": {"city": {"type": "string", "pattern": "^P"}}}]),
        ("anyOf", [STRICT]),
        ("not", {"required": ["city"]}),
        ("if", {"properties": {"city": {"const": "x"}}}),
        ("minProperties", 1),
    ],
)
def test_strict_parameters_constrain_through_properties_only(keyword, value):
    """The grammar constrains each property; a keyword beside them would only
    be checked after generation."""
    from atom.structured_output.validation import strict_tool_schemas

    with pytest.raises(ValueError, match=keyword):
        strict_tool_schemas([_function({**STRICT, keyword: value})])


def test_a_call_to_an_undeclared_function_fails_strict_validation():
    """What the parser read as another call, out of a value, is not a call."""
    from atom.structured_output.validation import validate_tool_calls

    schemas = {"weather": STRICT}
    good = _call('{"city": "Paris"}')
    rogue = {"function": {"name": "rm_rf", "arguments": "{}"}}
    validate_tool_calls(schemas, [good], "tool_calls", {"weather"})
    with pytest.raises(RuntimeError, match="rm_rf"):
        validate_tool_calls(schemas, [good, rogue], "tool_calls", {"weather"})
    # Without strict tools nothing is checked, as before.
    validate_tool_calls({}, [rogue], "tool_calls", {"weather"})


@pytest.mark.parametrize(
    "city, message",
    [
        # Each of these lets an argument carry keys the request never declared.
        ({"type": ["object", "null"]}, "additionalProperties"),
        ({}, "type"),
        (True, "object"),
        ({"type": "array"}, "items"),
        ({"type": "array", "items": {}}, "type"),
        (
            {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
                "patternProperties": {"^x": {"type": "integer"}},
            },
            "patternProperties",
        ),
        # Only the formats DeepSeek documents; the grammar enforces them.
        ({"type": "string", "format": "date"}, "format"),
        # References point into the top's definitions only.
        ({**STRICT, "$defs": {"x": STRICT}}, "top"),
    ],
)
def test_strict_schemas_close_every_object(city, message):
    from atom.structured_output.validation import strict_tool_schemas

    with pytest.raises(ValueError, match=message):
        strict_tool_schemas([_function({**STRICT, "properties": {"city": city}})])


@pytest.mark.parametrize(
    "city",
    [
        {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        {"type": ["string", "null"]},
        {
            "type": ["object", "null"],
            "properties": {"n": {"type": "integer"}},
            "required": ["n"],
            "additionalProperties": False,
        },
        {"type": "array", "items": {"type": "string", "format": "email"}},
        {"enum": ["a", "b"]},
    ],
)
def test_closed_strict_schemas_are_accepted(city):
    from atom.structured_output.validation import strict_tool_schemas

    assert strict_tool_schemas([_function({**STRICT, "properties": {"city": city}})])


def test_strict_tools_alone_configure_a_grammar():
    from atom.entrypoints.openai.tool_parser.deepseekv41_tool_parser import (
        DsmlV41Parser,
    )
    from atom.sampling_params import SamplingParams
    from atom.structured_output.validation import configure_request

    engine = NS(prepare_structured_request=Mock())
    params = SamplingParams()
    asyncio.run(
        configure_request(
            engine,
            None,
            NS(),
            params,
            None,
            ReasoningChannel(),
            parser_cls=DsmlV41Parser,
            tools=[_function()],
            tool_choice="auto",
            strict_schemas={"weather": STRICT},
        )
    )
    assert params.json_schema is None
    spec = json.loads(params.structured_tool_format)
    assert spec["tools"] == [{"name": "weather", "parameters": STRICT}]
    engine.prepare_structured_request.assert_called_once()


def test_a_draft_07_dialect_is_read_as_2020_12():
    """OpenAI's zod helpers name draft-07 and keep definitions in
    `definitions`; strict parameters mean the same in either."""
    from atom.structured_output.validation import strict_tool_schemas

    zod = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        **STRICT,
        "properties": {"city": {"$ref": "#/definitions/City"}},
        "definitions": {"City": {"type": "string"}},
    }
    [parameters] = strict_tool_schemas([_function(zod)]).values()
    assert "$schema" not in parameters
    with pytest.raises(ValueError, match="2020-12"):
        strict_tool_schemas(
            [
                _function(
                    {**STRICT, "$schema": "http://json-schema.org/draft-04/schema#"}
                )
            ]
        )


def test_an_array_gives_its_items_beside_a_prefix():
    from atom.structured_output.validation import strict_tool_schemas

    pair = {"type": "array", "prefixItems": [{"type": "string"}]}
    with pytest.raises(ValueError, match="items"):
        strict_tool_schemas([_function({**STRICT, "properties": {"city": pair}})])
    closed = {**pair, "items": False}
    assert strict_tool_schemas([_function({**STRICT, "properties": {"city": closed}})])


def test_each_distinct_tool_is_checked_once():
    from atom.structured_output import validation

    validation._checked_strict_parameters.cache_clear()
    for _ in range(3):
        validation.strict_tool_schemas([_function()])
    info = validation._checked_strict_parameters.cache_info()
    assert (info.misses, info.hits) == (1, 2)


@pytest.mark.parametrize(
    "keyword, value",
    [
        ("minLength", 1),
        ("maxLength", 1),
        ("minItems", 1),
        ("maxItems", 1),
        ("$anchor", "x"),
        ("$dynamicAnchor", "x"),
    ],
)
def test_strict_keywords_deepseek_refuses(keyword, value):
    from atom.structured_output.validation import strict_tool_schemas

    city = {"type": "string", keyword: value}
    with pytest.raises(ValueError, match=re.escape(keyword)):
        strict_tool_schemas([_function({**STRICT, "properties": {"city": city}})])


def test_a_stream_without_strict_tools_is_never_held():
    """Only strict calls wait for their validation: `/v1` with
    `response_format` and ordinary tools streams as it comes."""
    events = []

    async def source():
        for frame in _tool_call_frames('{"city": "Paris"}'):
            events.append("produced")
            yield frame

    async def run():
        async for _ in validate_chat_stream(source(), None, {}):
            events.append("sent")

    asyncio.run(run())
    assert events == ["produced", "sent"] * len(_tool_call_frames("{}"))


def test_strict_rules_reach_into_every_branch():
    """An object inside `anyOf` is held to the rules a property is."""
    from atom.structured_output.validation import strict_tool_schemas

    loose = {"type": "object", "properties": {"x": {"type": "string"}}}
    city = {"anyOf": [{**loose, "required": ["x"]}, {"type": "null"}]}
    with pytest.raises(ValueError, match="additionalProperties"):
        strict_tool_schemas([_function({**STRICT, "properties": {"city": city}})])


def test_a_streamed_call_to_an_undeclared_function_never_reaches_the_client():
    async def source():
        for frame in _tool_call_frames('{"city": "Paris"}'):
            yield frame

    async def run():
        stream = validate_chat_stream(source(), None, {"weather": STRICT}, {"search"})
        return "".join([frame async for frame in stream])

    raw = asyncio.run(run())
    assert "structured_output_error" in raw
    assert '"tool_calls"' not in raw


def test_strict_arguments_must_be_standard_json():
    """The parser reads 1e999 as inf and writes it back as `Infinity`."""
    from atom.structured_output.validation import validate_tool_calls

    number = {**STRICT, "properties": {"v": {"type": "number"}}, "required": ["v"]}
    validate_tool_calls({"weather": number}, [_call('{"v": 1.5}')], "tool_calls")
    with pytest.raises(RuntimeError, match="strict parameters"):
        validate_tool_calls(
            {"weather": number}, [_call('{"v": Infinity}')], "tool_calls"
        )


@pytest.mark.parametrize("where", ["tool", "parameter"])
def test_strict_names_cannot_hold_a_quote(where):
    """The DSML tags quote names unescaped; the parser would cut one short."""
    from atom.structured_output.validation import strict_tool_schemas

    if where == "tool":
        tool = _function()
        tool["function"]["name"] = 'we"ather'
    else:
        quoted = {"type": "string"}
        tool = _function(
            {**STRICT, "properties": {'a">b': quoted}, "required": ['a">b']}
        )
    with pytest.raises(ValueError, match="cannot contain"):
        strict_tool_schemas([tool])


@pytest.mark.parametrize("key", ["a</invoke>b", "a<invoke b", "a</|DSML|invoke>b"])
def test_strict_parameter_names_cannot_hold_a_tag(key):
    """The parser would read the tag inside the name as the call's end, or
    another call."""
    from atom.structured_output.validation import strict_tool_schemas

    tool = _function(
        {**STRICT, "properties": {key: {"type": "string"}}, "required": [key]}
    )
    with pytest.raises(ValueError, match="cannot contain"):
        strict_tool_schemas([tool])


def test_a_reference_target_is_held_to_the_strict_rules():
    """A $ref into a definition's `default` names an untyped `{}`."""
    from atom.structured_output.validation import strict_tool_schemas

    holder = {"type": "string", "default": {}}
    loose = {
        **STRICT,
        "properties": {"city": {"$ref": "#/$defs/Holder/default"}},
        "$defs": {"Holder": holder},
    }
    with pytest.raises(ValueError, match="give its type"):
        strict_tool_schemas([_function(loose)])
    scalar = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    part = {
        **STRICT,
        "properties": {"city": {"$ref": "#/$defs/Scalar/anyOf/1"}},
        "$defs": {"Scalar": scalar},
    }
    assert strict_tool_schemas([_function(part)])


def test_a_call_the_stream_never_finished_is_not_sent():
    """The engine failed mid-call (a logprob error, say): the call was never
    checked, so only the text before it and the error go out."""
    frames = _tool_call_frames('{"city": Infinity}')[:3] + [
        data_frame({"error": {"type": "logprob_error", "message": "failed"}}),
        "data: [DONE]\n\n",
    ]

    async def source():
        for frame in frames:
            yield frame

    async def run():
        stream = validate_chat_stream(source(), None, {"weather": STRICT})
        return "".join([frame async for frame in stream])

    raw = asyncio.run(run())
    assert "Checking." in raw and "logprob_error" in raw
    assert '"tool_calls"' not in raw
    assert raw.count("[DONE]") == 1


def test_a_ref_inside_an_annotation_is_data():
    """`example` and `x-*` annotations hold instances, where `$ref` is a key."""
    from atom.structured_output.validation import strict_tool_schemas

    data = {"$ref": "https://example.com/data"}
    city = {
        "type": "object",
        "properties": {"$ref": {"type": "string"}},
        "required": ["$ref"],
        "additionalProperties": False,
        "example": data,
        "x-sample": data,
    }
    assert strict_tool_schemas([_function({**STRICT, "properties": {"city": city}})])

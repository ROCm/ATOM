import asyncio

import pytest

pytest.importorskip("jsonschema")

from atom.entrypoints.openai.sse import data_frame
from atom.entrypoints.openai.structured_output import (
    schema_from_format,
    validate_chat_stream,
    validate_output,
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

    from atom.entrypoints.openai.structured_output import structured_tool_format
    from atom.entrypoints.openai.tool_parser.deepseekv41_tool_parser import (
        DsmlV41Parser,
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

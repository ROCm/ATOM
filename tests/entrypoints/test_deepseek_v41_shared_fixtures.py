# SPDX-License-Identifier: MIT
"""One fixture source for Python and mesh Rust DSML parsing."""

import json
from pathlib import Path

from atom.entrypoints.openai.tool_parser import flatten_tool_events, parse_tool_calls
from atom.entrypoints.openai.tool_parser.deepseekv41_tool_parser import DsmlV41Parser
from atom.entrypoints.openai.tool_parser.stream import ToolCallStreamParser


def test_shared_rust_fixtures():
    fixtures = json.loads(
        (Path(__file__).parent / "fixtures/deepseek_v41_dsml.json").read_text()
    )
    for fixture in fixtures:
        tools = [
            {
                "type": "function",
                "function": {"name": c["name"], "parameters": {"type": "object"}},
            }
            for c in fixture["calls"]
        ]
        text, calls = parse_tool_calls(fixture["text"], tools, DsmlV41Parser)
        assert text == fixture["content"]
        for actual, expected in zip(calls, fixture["calls"], strict=True):
            assert actual.function["name"] == expected["name"]
            assert json.loads(actual.function["arguments"]) == expected["arguments"]
        for offset in range(len(fixture["text"]) + 1):
            parser = ToolCallStreamParser(tools=tools, parser_cls=DsmlV41Parser)
            events = []
            for part in (fixture["text"][:offset], fixture["text"][offset:]):
                events.extend(parser.process(part))
            events.extend(parser.flush())
            text, streamed = flatten_tool_events(events)
            assert text == fixture["content"]
            assert [c.function for c in streamed] == [c.function for c in calls]

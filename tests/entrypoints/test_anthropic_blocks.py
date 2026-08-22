# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Anthropic content blocks: one open at a time, and nothing falls between them.

A response is framed as indexed blocks of a kind, and a change of kind is a
close and an open. Those transitions used to be written out at each of the
four places a segment could arrive, each covering the subset its author
needed — and the one nobody needed, text -> thinking, was missing. A reasoning
segment arriving after content had started matched no branch and was dropped
with no error and no log.

So the properties here are about totality: every segment handed in comes back
out, whatever order the kinds arrive in.
"""

from __future__ import annotations

import json
from typing import ClassVar

import pytest

from atom.entrypoints.openai.serving_anthropic import (
    AnthropicBlocks,
    completes_a_tool_call,
    tool_event_frames,
)
from atom.entrypoints.openai.tool_parser.glm_tool_parser import GlmParser

KINDS = ("text", "thinking", "tool_use")


def events(frames: list[str]) -> list[tuple[str, int, str]]:
    """(event name, block index, delta text) for each frame."""
    out = []
    for f in frames:
        name = f.split("event: ", 1)[1].split("\n", 1)[0]
        data = json.loads(f.split("data: ", 1)[1])
        delta = data.get("delta", {})
        text = delta.get("text") or delta.get("thinking") or delta.get("partial_json")
        out.append((name, data["index"], text or ""))
    return out


def drive(pairs: list[tuple[str, str]]) -> list[str]:
    """Feed (kind, text) in order and close at the end, as the server does."""
    blocks = AnthropicBlocks()
    frames: list[str] = []
    for kind, text in pairs:
        frames += list(blocks.delta(kind, text))
    frames += list(blocks.close())
    return frames


class TestNothingIsDropped:
    def test_reasoning_after_content_is_delivered(self):
        """The bug, stated as a test.

        A model that answers, opens a `<think>` block and answers again used to
        lose the whole reasoning block: `started_text` was set, `started_thinking`
        was not, and neither branch fired.
        """
        frames = drive(
            [
                ("text", "Let me look that up. "),
                ("thinking", "Paris weather."),
                ("text", "Sunny."),
            ]
        )
        thinking = "".join(t for name, _, t in events(frames) if "delta" in name and t)
        assert "Paris weather." in thinking

    @pytest.mark.parametrize("first", KINDS)
    @pytest.mark.parametrize("second", KINDS)
    def test_every_kind_change_delivers_both_sides(self, first, second):
        """Nine orderings, none of which may swallow anything."""
        frames = drive([(first, "AAA"), (second, "BBB")])
        delivered = "".join(t for _, _, t in events(frames))
        assert "AAA" in delivered and "BBB" in delivered

    def test_text_handed_in_is_text_handed_out(self):
        pairs = [
            ("text", "one "),
            ("thinking", "two "),
            ("text", "three "),
            ("tool_use", "{}"),
        ]
        delivered = "".join(t for _, _, t in events(drive(pairs)))
        assert delivered == "".join(t for _, t in pairs)


class TestBlockFraming:
    def test_a_block_is_closed_before_the_next_one_opens(self):
        names = [n for n, _, _ in events(drive([("text", "a"), ("thinking", "b")]))]
        # start, delta, [signature] stop, start, delta, ... and never two
        # starts without a stop between them.
        depth = 0
        for n in names:
            if n == "content_block_start":
                assert depth == 0, "a block opened while another was open"
                depth = 1
            elif n == "content_block_stop":
                depth = 0
        assert depth == 0, "the last block was left open"

    def test_indices_are_unique_and_ascending(self):
        idx = [
            i
            for n, i, _ in events(
                drive([("text", "a"), ("thinking", "b"), ("text", "c")])
            )
        ]
        starts = sorted({i for i in idx})
        assert idx == sorted(idx) and starts == list(range(len(starts)))

    def test_a_thinking_block_signs_off_before_it_stops(self):
        """Anthropic requires the signature delta while the block is still open."""
        names = [n for n, _, _ in events(drive([("thinking", "why"), ("text", "so")]))]
        stop = names.index("content_block_stop")
        assert names[stop - 1] == "content_block_delta"

    def test_closing_twice_emits_nothing_the_second_time(self):
        blocks = AnthropicBlocks()
        list(blocks.delta("text", "a"))
        assert list(blocks.close())
        assert list(blocks.close()) == []

    def test_closing_before_anything_opened_emits_nothing(self):
        assert list(AnthropicBlocks().close()) == []


class TestAResponseIsNeverEmpty:
    """A reply that produced only reasoning still says something.

    `/v1/messages` drops reasoning when the request did not ask for thinking.
    That was safe while an unseeded filter sent most output down the content
    channel; once seeding is right, a reasoning model stopped at `max_tokens`
    produces *nothing else*, and the client got pings, an empty text block and
    `stop_reason=end_turn`. Measured: 20 pings, zero delta frames.

    The block machine cannot fix this on its own -- the decision is the
    endpoint's -- so what is pinned here is the shape the endpoint relies on:
    an untouched machine reports `index == 0`, which is how it knows nothing
    was delivered.
    """

    def test_an_untouched_machine_reports_nothing_delivered(self):
        assert AnthropicBlocks().index == 0

    def test_one_delivered_block_advances_the_index(self):
        blocks = AnthropicBlocks()
        list(blocks.delta("text", "hi"))
        list(blocks.close())
        assert blocks.index == 1

    def test_opening_without_delivering_does_not_advance_it(self):
        """The endpoint opens a trailing text block before it checks."""
        blocks = AnthropicBlocks()
        list(blocks.open("text"))
        assert blocks.index == 0


class TestToolEventFrames:
    """The tool-parser's events as Anthropic frames, in one place.

    This dispatch was written out twice in the streaming endpoint -- once for
    `process` and once for `flush` -- twenty-two identical lines each. Two
    copies of a dispatch is a fix that lands in one of them and says nothing,
    which is the hazard `AnthropicBlocks` itself was extracted to remove. It
    was also untestable there: the endpoint body is an async generator inside
    a route handler no unit test reaches.
    """

    @staticmethod
    def _kinds(events, blocks=None):
        frames = list(tool_event_frames(events, blocks or AnthropicBlocks()))
        return [json.loads(f.split("data: ", 1)[1])["type"] for f in frames]

    START = (
        "tool_call_start",
        {"id": "call_1", "function": {"name": "get_weather", "arguments": ""}},
    )
    ARGS = ("tool_call_args", {"function": {"arguments": '{"city":'}})
    END = ("tool_call_end", None)

    def test_a_whole_call_opens_streams_and_closes(self):
        assert self._kinds([self.START, self.ARGS, self.END]) == [
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
        ]

    def test_content_before_a_call_lands_in_a_text_block(self):
        frames = list(
            tool_event_frames([("content", "Checking."), self.START], AnthropicBlocks())
        )
        payloads = [json.loads(f.split("data: ", 1)[1]) for f in frames]
        assert payloads[0]["type"] == "content_block_start"
        assert payloads[0]["content_block"]["type"] == "text"
        # The text block is closed before the tool block opens.
        assert [p["type"] for p in payloads[1:]] == [
            "content_block_delta",
            "content_block_stop",
            "content_block_start",
        ]
        assert payloads[-1]["content_block"]["type"] == "tool_use"

    def test_the_call_carries_its_id_and_name(self):
        frames = list(tool_event_frames([self.START], AnthropicBlocks()))
        block = json.loads(frames[0].split("data: ", 1)[1])["content_block"]
        assert block["id"] == "call_1" and block["name"] == "get_weather"

    def test_an_unknown_event_type_is_ignored_not_crashed(self):
        assert self._kinds([("something_new", {})]) == []

    def test_no_events_emit_nothing(self):
        assert self._kinds([]) == []

    @pytest.mark.parametrize(
        "events, expected",
        [
            ([], False),
            ([("content", "hi")], False),
            # A name and nothing else: announced early, then the stream was
            # cut off. Not a usable call, so not `tool_use`.
            ([("content", "hi"), START], False),
            ([START, ARGS], True),
            ([("tool_call_args", {}), ("tool_call_end", None)], True),
        ],
    )
    def test_completes_a_tool_call_reads_the_batch(self, events, expected):
        """`stop_reason` turns on this, and it is asked of both batches.

        Keyed on the arguments: a name can be sent before the call is known
        to close, so a name alone does not mean the client has a tool to run.
        """
        assert completes_a_tool_call(events) is expected


class TestNoBlockWithoutAnIdAndAName:
    """`delta("tool_use", ...)` opens a block when none is open, and it was
    given nothing to open one with.

    Anything landing between a call's name and its arguments -- text, another
    kind, a stray event -- re-opened the tool_use block with `id: ""` and
    `name: ""`. That is syntactically a complete tool_use: a client cannot
    dispatch it (no name) and cannot return a result for it (no id), and
    Claude Code treats a well-formed zero-argument block as a call to make.
    """

    START = (
        "tool_call_start",
        {"id": "call_1", "function": {"name": "get_weather", "arguments": ""}},
    )
    ARGS = ("tool_call_args", {"function": {"arguments": '{"city": "Paris"}'}})

    @staticmethod
    def _blocks(events):
        out = []
        for frame in tool_event_frames(events, AnthropicBlocks()):
            payload = json.loads(frame.split("data: ", 1)[1])
            if payload["type"] == "content_block_start":
                out.append(payload["content_block"])
        return out

    def test_arguments_with_no_name_open_nothing(self):
        assert self._blocks([self.ARGS, ("tool_call_end", None)]) == []

    def test_text_between_the_name_and_the_arguments_keeps_both(self):
        blocks = self._blocks(
            [self.START, ("content", "oops"), self.ARGS, ("tool_call_end", None)]
        )
        tool_blocks = [b for b in blocks if b["type"] == "tool_use"]
        assert len(tool_blocks) == 2, "the re-opened block went missing"
        for b in tool_blocks:
            assert b["id"] == "call_1" and b["name"] == "get_weather"

    def test_no_tool_use_block_is_ever_nameless(self):
        for events in (
            [self.ARGS],
            [self.ARGS, self.ARGS],
            [self.START, ("content", "x"), self.ARGS],
            [("tool_call_end", None), self.ARGS],
        ):
            for block in self._blocks(events):
                if block["type"] == "tool_use":
                    assert block["id"] and block["name"], block


class TestOneCallSpansTwoParserBatches:
    """A name announced early and its arguments do not arrive together.

    `tool_event_frames` runs once per parser batch, and announcing the name as
    soon as the region reveals it puts the two events that describe one call
    in different batches -- the name from `process`, the arguments from
    `flush`. Which call is open therefore cannot be a local in that function,
    and while it was, every streamed tool call on `/v1/messages` reached the
    client as `input: {}` with `stop_reason: tool_use`. Claude Code ran the
    tool with no arguments.

    Driven through a real parser rather than a hand-built event list: the
    tests above pass one batch each and so could not see this.
    """

    TOOLS: ClassVar[list] = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]
    CALL = (
        "<tool_call>get_weather<arg_key>city</arg_key>"
        "<arg_value>Paris</arg_value></tool_call>"
    )

    def _frames(self, chunk_size):
        parser = GlmParser(tools=self.TOOLS)
        blocks = AnthropicBlocks()
        out = []
        for i in range(0, len(self.CALL), chunk_size):
            batch = parser.process(self.CALL[i : i + chunk_size])
            out += list(tool_event_frames(batch, blocks))
        out += list(tool_event_frames(parser.flush(), blocks))
        return [json.loads(f.split("data: ", 1)[1]) for f in out]

    @pytest.mark.parametrize("chunk_size", [1, 7, len(CALL)])
    def test_the_arguments_reach_the_client(self, chunk_size):
        deltas = [
            p["delta"]["partial_json"]
            for p in self._frames(chunk_size)
            if p["type"] == "content_block_delta"
            and p["delta"]["type"] == "input_json_delta"
        ]
        assert "".join(deltas) == '{"city": "Paris"}'

    @pytest.mark.parametrize("chunk_size", [1, 7, len(CALL)])
    def test_they_land_in_the_block_that_carries_the_name(self, chunk_size):
        frames = self._frames(chunk_size)
        named = [
            p["index"]
            for p in frames
            if p["type"] == "content_block_start"
            and p["content_block"]["name"] == "get_weather"
        ]
        argued = [
            p["index"]
            for p in frames
            if p["type"] == "content_block_delta"
            and p["delta"]["type"] == "input_json_delta"
        ]
        assert named and set(argued) <= set(named), (named, argued)

    def test_a_call_that_ended_does_not_adopt_the_next_arguments(self):
        """The call outlives a block close, but not its own end.

        Surviving `close` is what fixes the case above; surviving
        `tool_call_end` would hand a later orphan batch of arguments to a
        call the client has already been told is finished.
        """
        blocks = AnthropicBlocks()
        start = (
            "tool_call_start",
            {"id": "call_1", "function": {"name": "get_weather", "arguments": ""}},
        )
        orphan = ("tool_call_args", {"function": {"arguments": '{"city": "Rome"}'}})
        list(tool_event_frames([start, ("tool_call_end", None)], blocks))
        assert blocks.open_call is None
        frames = list(tool_event_frames([orphan], blocks))
        assert frames == [], "arguments were adopted by a call that had ended"

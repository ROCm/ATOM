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

import pytest

from atom.entrypoints.openai.serving_anthropic import (
    AnthropicBlocks,
    starts_a_tool_call,
    tool_event_frames,
)

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
            ([("tool_call_args", {}), ("tool_call_end", None)], False),
            ([("content", "hi"), START], True),
        ],
    )
    def test_starts_a_tool_call_reads_the_batch(self, events, expected):
        """`stop_reason` turns on this, and it is asked of both batches."""
        assert starts_a_tool_call(events) is expected

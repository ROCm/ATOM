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

from atom.entrypoints.openai.reasoning import ReasoningChannel
from atom.entrypoints.openai.reasoning_dialects import DIALECTS
from atom.entrypoints.openai.serving_anthropic import (
    AnthropicBlocks,
    _blocks_in_order,
    build_anthropic_response,
    completes_a_tool_call,
    read_whole_blocks,
    stream_failure_frames,
    stream_message_start,
    tool_event_frames,
)
from atom.entrypoints.openai.tool_parser import ToolCallStreamParser
from atom.entrypoints.openai.tool_parser.glm_tool_parser import GlmParser
from entrypoints.wire_corpus import REAL_CALLS, TYPED_TOOLS

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
            tool_event_frames(
                [("content", "Checking."), self.START, self.ARGS], AnthropicBlocks()
            )
        )
        payloads = [json.loads(f.split("data: ", 1)[1]) for f in frames]
        assert payloads[0]["type"] == "content_block_start"
        assert payloads[0]["content_block"]["type"] == "text"
        # The text block is closed before the tool block opens.
        assert [p["type"] for p in payloads[1:]] == [
            "content_block_delta",
            "content_block_stop",
            "content_block_start",
            "content_block_delta",
        ]
        assert payloads[-2]["content_block"]["type"] == "tool_use"

    def test_the_call_carries_its_id_and_name(self):
        frames = list(tool_event_frames([self.START, self.ARGS], AnthropicBlocks()))
        block = json.loads(frames[0].split("data: ", 1)[1])["content_block"]
        assert block["id"] == "call_1" and block["name"] == "get_weather"

    def test_a_name_with_no_arguments_puts_no_block_on_the_wire(self):
        """`content_block_start` of type `tool_use` carries `"input": {}` and
        is, on its own, a complete zero-argument call -- there is no frame for
        "the name is known, the arguments are coming". So the name cannot be
        sent early on this protocol, and a response that announced one and was
        then cut off must leave nothing behind. It used to leave a
        syntactically perfect call the model never made."""
        assert self._kinds([("content", "hi"), self.START]) == [
            "content_block_start",
            "content_block_delta",
        ]

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

    def test_one_call_never_arrives_as_two_blocks(self):
        """Re-opening on the same id is worse than dropping the arguments.

        A client iterating content blocks sees two `tool_use` entries with one
        id, the first carrying no input -- so it runs `get_weather({})` and
        then `get_weather({"city": "Paris"})`. Anthropic has no spelling for
        "the block you already closed, continued".

        The shape is not reachable from any registered parser -- driven over
        every format's real call, three leading shapes and every chunking,
        content never once landed between an announced name and its arguments
        -- so this pins the degradation rather than a behaviour anyone gets.
        """
        blocks = self._blocks(
            [self.START, ("content", "oops"), self.ARGS, ("tool_call_end", None)]
        )
        tool_blocks = [b for b in blocks if b["type"] == "tool_use"]
        assert len(tool_blocks) == 1, f"one call, {len(tool_blocks)} blocks"
        ids = [b["id"] for b in tool_blocks]
        assert len(set(ids)) == len(ids), f"a tool id was used twice: {ids}"

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

    def _frames(self, chunk_size, text=None, reasoning_at=None):
        parser = ToolCallStreamParser(tools=self.TOOLS, parser_cls=GlmParser)
        blocks = AnthropicBlocks()
        text = self.CALL if text is None else text
        out = []
        for i in range(0, len(text), chunk_size):
            if reasoning_at is not None and i >= reasoning_at:
                # One thinking segment, delivered between the name and the
                # arguments. It closes whatever block was open, which is the
                # whole point: the call's identity has to outlive that.
                out += list(blocks.delta("thinking", "hm"))
                reasoning_at = None
            batch = parser.process(text[i : i + chunk_size])
            out += list(tool_event_frames(batch, blocks))
        out += list(tool_event_frames(parser.flush(), blocks))
        return [json.loads(f.split("data: ", 1)[1]) for f in out]

    @pytest.mark.parametrize("chunk_size", [1, 7])
    def test_a_thinking_block_in_between_does_not_lose_the_arguments(self, chunk_size):
        """The shape that broke it: reasoning arriving after the name was
        announced closed the tool block, and the id and name lived on that
        block. The arguments then had nothing to open a block with and were
        dropped -- `tool_use` with `input: {}` and `stop_reason: tool_use`,
        while the same bytes returned `{"city": "Paris"}` unstreamed."""
        frames = self._frames(chunk_size, reasoning_at=chunk_size)
        deltas = [
            p["delta"]["partial_json"]
            for p in frames
            if p["type"] == "content_block_delta"
            and p["delta"]["type"] == "input_json_delta"
        ]
        assert "".join(deltas) == '{"city": "Paris"}'

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

    def test_arguments_for_a_call_that_was_never_named_are_dropped(self):
        """Every call is keyed by the index it was named at, so a batch that
        closes one block cannot make the next batch's arguments land on it --
        and arguments for an index nobody named open nothing at all."""
        blocks = AnthropicBlocks()
        start = (
            "tool_call_start",
            {
                "index": 0,
                "id": "call_1",
                "function": {"name": "get_weather", "arguments": ""},
            },
        )
        args = ("tool_call_args", {"index": 0, "function": {"arguments": "{}"}})
        later = (
            "tool_call_args",
            {"index": 1, "function": {"arguments": '{"city": "Rome"}'}},
        )
        list(tool_event_frames([start, args, ("tool_call_end", None)], blocks))
        assert blocks.open_call_index is None
        frames = list(tool_event_frames([later], blocks))
        assert frames == [], "arguments were adopted by a call that never existed"


# ── The two delivery modes must build the same blocks ──────────────────


def blocks_from_frames(frames: list[str]) -> list[dict]:
    """The blocks a client accumulates from the stream."""
    out: list[dict] = []
    for f in frames:
        name = f.split("event: ", 1)[1].split("\n", 1)[0]
        data = json.loads(f.split("data: ", 1)[1])
        if name == "content_block_start":
            out.append(dict(data["content_block"]))
        elif name == "content_block_delta":
            delta, blk = data["delta"], out[-1]
            for key in ("text", "thinking"):
                if key in delta:
                    blk[key] = blk.get(key, "") + delta[key]
            if "partial_json" in delta:
                blk["_json"] = blk.get("_json", "") + delta["partial_json"]
    for blk in out:
        if "_json" in blk:
            raw = blk.pop("_json")
            try:
                blk["input"] = json.loads(raw or "{}")
            except json.JSONDecodeError:
                blk["input"] = raw
    return out


def shape_of(blocks: list[dict]) -> list[tuple]:
    """Type and payload only -- a thinking block's signature is random."""
    return [
        (b["type"], b.get("text", b.get("thinking", b.get("input")))) for b in blocks
    ]


def stream_side(raw: str, channel, parser_cls, tools=None) -> list[dict]:
    """The server's streaming loop, over one chunk.

    Copied in shape from `api_server`'s Anthropic branch on purpose: phase 1
    is the reasoning filter, phase 2 is the tool parser *on the content
    segments only*, and the interleaving of the two is what a client sees.
    """
    rf, tp = channel.stream(), ToolCallStreamParser(tools=tools, parser_cls=parser_cls)
    blocks, frames = AnthropicBlocks(), []
    for field, text in rf.process(raw) + rf.flush():
        if not text:
            continue
        if field == "reasoning_content":
            frames += list(blocks.delta("thinking", text))
        else:
            frames += list(tool_event_frames(tp.process(text), blocks))
    frames += list(tool_event_frames(tp.flush(), blocks))
    frames += list(blocks.close())
    return blocks_from_frames(frames)


def nonstream_side(raw: str, channel, parser_cls, tools=None) -> list[dict]:
    """The same generation through `stream=false`, as `api_server` builds it."""
    events = read_whole_blocks(channel, parser_cls, raw, tools)
    return build_anthropic_response(
        request_id="r", model="m", events=events, stop_reason="end_turn"
    )["content"]


class TestBothDeliveryModesBuildTheSameBlocks:
    """`/v1/messages` renders blocks in order, so order is part of the answer.

    The streaming branch interleaves as the model wrote: reasoning filter
    first, tool parser on the content segments it yields. The non-streaming
    branch calls `split()`, which returns `(reasoning, content)` -- position
    gone at that line -- and then rebuilds blocks from the flat pair. So a
    model that answers, thinks, and answers again came back as two blocks with
    the two answers glued together where streaming sent three.

    Parity is the right property here, unlike the reasoning split one stage
    down where both readers were wrong together: streaming is correct and
    non-streaming is not, so the canonical shape is asserted outright too.
    """

    SHAPES: ClassVar[dict[str, str]] = {
        "answer only": "The answer is 42.",
        "reasoning then answer": "<think>hmm</think>The answer is 42.",
        "answer, reasoning, answer": "Sure.<think>hmm</think>Answer.",
        "reasoning only": "<think>hmm</think>",
        "answer then a call": "Checking.{call}",
        "reasoning, answer, call": "<think>hmm</think>Checking.{call}",
        "call then answer": "{call}All done.",
        # The shape this branch has broken twice: a sentence the model wrote
        # between two calls, which an extra span competing for a call hoists
        # in front of both.
        "call, sentence, call": "{call}Now Rome.{other}",
        "reasoning between two calls": "{call}<think>and Rome</think>{other}",
    }

    @staticmethod
    def _channel():
        return ReasoningChannel(dialect=DIALECTS[2], starts_open=False)

    @pytest.mark.parametrize("name", sorted(SHAPES))
    def test_the_block_sequence_is_identical(self, name):
        raw = self.SHAPES[name].format(
            call=REAL_CALLS["glm"],
            other=REAL_CALLS["glm"].replace("get_weather", "get_time"),
        )
        ch = self._channel()
        streamed = shape_of(stream_side(raw, ch, GlmParser, TYPED_TOOLS))
        whole = shape_of(nonstream_side(raw, ch, GlmParser, TYPED_TOOLS))
        assert whole == streamed, (
            f"{name}: stream=false built {whole}\n"
            f"{' ' * len(name)}  stream=true  built {streamed}"
        )

    def test_and_the_order_is_the_one_the_model_wrote(self):
        """The canonical shape, asserted rather than only compared.

        Both paths agreeing on the wrong order would satisfy the property
        above; this says which order is right.
        """
        ch = self._channel()
        raw = "Sure.<think>hmm</think>Answer."
        assert shape_of(stream_side(raw, ch, None)) == [
            ("text", "Sure."),
            ("thinking", "hmm"),
            ("text", "Answer."),
        ]

    def test_and_a_call_whose_arguments_are_not_an_object_keeps_them(self):
        """Kimi-K2 passes the wire bytes through, so `arguments` need not be a
        JSON object. Streaming forwards them and the SDK accumulates the real
        value; the non-streaming builder replaced anything but a dict with
        `{}`, so the tool was invoked with no arguments on one path and the
        right ones on the other -- silently, with no log."""
        events = [
            (
                "tool_call_start",
                {
                    "index": 0,
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "t", "arguments": ""},
                },
            ),
            ("tool_call_args", {"index": 0, "function": {"arguments": "[1, 2]"}}),
        ]
        streamed = shape_of(
            blocks_from_frames(list(tool_event_frames(events, AnthropicBlocks())))
        )
        whole = shape_of(_blocks_in_order(events))
        assert whole == streamed == [("tool_use", [1, 2])]


class TestAStreamThatFailedStillOwesTheClientAWholeMessage:
    """The error tail, which the endpoint used to write out inline.

    It emitted `error` / `message_delta` / `message_stop` without checking
    whether `message_start` had gone out. The flag is only set inside the
    loop, after the first `await stream_collector.get()` returns -- so an
    engine error, a detokenizer failure or an abort surfacing on that first
    call produced a stream that never opened, and the SDK raises on a
    `message_delta` arriving before `message_start`. The client got an
    SDK-internal error instead of the `error` frame this exists to deliver.
    """

    @staticmethod
    def _names(frames):
        return [f.split("event: ", 1)[1].split("\n", 1)[0] for f in frames]

    def test_it_opens_the_message_when_nothing_opened_it(self):
        frames = list(
            stream_failure_frames(
                RuntimeError("boom"),
                AnthropicBlocks(),
                0,
                opening=stream_message_start("r", "m", 0, 0),
            )
        )
        names = self._names(frames)
        assert names[0] == "message_start", names
        assert names[-1] == "message_stop", names
        assert "error" in names

    def test_but_does_not_open_it_twice(self):
        frames = list(
            stream_failure_frames(
                RuntimeError("boom"), AnthropicBlocks(), 0, opening=None
            )
        )
        names = self._names(frames)
        assert "message_start" not in names, names
        assert names[-1] == "message_stop", names

    def test_and_closes_a_block_that_was_left_open(self):
        blocks = AnthropicBlocks()
        list(blocks.delta("text", "half an ans"))
        frames = list(
            stream_failure_frames(RuntimeError("boom"), blocks, 0, opening=None)
        )
        assert self._names(frames)[0] == "content_block_stop", self._names(frames)

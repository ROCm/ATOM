# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 replies written straight into the prompt-opened think channel.

On ~1M-token prompts K3 sometimes skips its think channel: it writes the reply
where reasoning goes and closes it with the *answer's* closer, never emitting
`<|close|>think<|sep|>` or `<|open|>response<|sep|>`. Measured on BEAM (1M)
under the Kimi Vendor Verifier, one reply in five had this shape, and both
delivery modes returned it as `reasoning_content` with `content` empty, so the
benchmark scored it 0. The answer's closer before any think exit is what tells
the two apart -- the same reading as sgl-project/sglang#37743.
"""

import pytest

from atom.entrypoints.openai.reasoning import ReasoningChannel
from atom.entrypoints.openai.reasoning_dialects import (
    CHANNEL_MESSAGE_END,
    CHANNEL_RESPONSE_END,
    CHANNEL_RESPONSE_START,
    CHANNEL_THINK_END,
    CHANNEL_THINK_START,
    CHANNEL_TOOLS_START,
    DIALECTS,
    ReasoningDialect,
)
from atom.entrypoints.openai.tool_parser import parse_tool_calls
from atom.entrypoints.openai.tool_parser.kimi_k3_tool_parser import KimiK3Parser

K3 = next(d for d in DIALECTS if d.think_end_marker == CHANNEL_THINK_END)
PROMPT_OPENED = ReasoningChannel(dialect=K3, starts_open=True)
SELF_OPENED = ReasoningChannel(dialect=K3, starts_open=False)
CHUNKS = (1, 2, 3, 7, 10_000)

REPLY = "To find it:\n\n1. **Convert** the coordinates.\n2. Apply the formula."
SKIPPED = REPLY + CHANNEL_RESPONSE_END + CHANNEL_MESSAGE_END
CALL = (
    CHANNEL_TOOLS_START
    + '<|open|>call tool="get_weather"<|sep|>'
    + '<|open|>argument key="city"<|sep|>Paris<|close|>argument'
    + "<|close|>call<|close|>tools<|sep|>"
)


def streamed(channel: ReasoningChannel, text: str, chunk: int) -> tuple[str, str]:
    f = channel.stream()
    segs = []
    for i in range(0, len(text), chunk):
        segs += f.process(text[i : i + chunk])
    segs += f.flush()
    return (
        "".join(t for k, t in segs if k == "reasoning_content"),
        "".join(t for k, t in segs if k == "content"),
    )


class TestASkippedChannelIsTheAnswer:
    def test_the_split_returns_the_reply_as_content(self):
        assert PROMPT_OPENED.split(SKIPPED) == (None, REPLY)

    @pytest.mark.parametrize("chunk", CHUNKS)
    def test_the_stream_delivers_the_same_answer(self, chunk):
        _, content = streamed(PROMPT_OPENED, SKIPPED, chunk)
        assert content == REPLY

    @pytest.mark.parametrize("chunk", CHUNKS)
    def test_and_its_reasoning_holds_only_what_had_already_gone_out(self, chunk):
        """The one divergence, pinned so it stays the only one: text released
        as reasoning before the closer arrived cannot be recalled, so it is
        repeated, never anything more."""
        reasoning, _ = streamed(PROMPT_OPENED, SKIPPED, chunk)
        assert reasoning == REPLY

    def test_a_closer_with_no_message_end_still_counts(self):
        """`max_tokens` can stop between the two closers."""
        assert PROMPT_OPENED.split(REPLY + CHANNEL_RESPONSE_END) == (None, REPLY)
        for chunk in CHUNKS:
            text = REPLY + CHANNEL_RESPONSE_END
            assert streamed(PROMPT_OPENED, text, chunk)[1] == REPLY

    def test_a_channel_the_model_opened_itself_reads_the_same(self):
        text = "Note. " + CHANNEL_THINK_START + SKIPPED
        assert SELF_OPENED.split(text) == (None, "Note. " + REPLY)
        for chunk in CHUNKS:
            assert streamed(SELF_OPENED, text, chunk)[1] == "Note. " + REPLY

    def test_a_call_after_the_reply_is_still_parsed(self):
        """Tools after the skipped reply reach the tool parser, and the reply
        is the content around the call -- the reply's own closer says so."""
        text = REPLY + CHANNEL_RESPONSE_END + CALL + CHANNEL_MESSAGE_END
        reasoning, content = PROMPT_OPENED.split(text)
        assert reasoning is None
        answer, calls = parse_tool_calls(content, None, KimiK3Parser)
        assert [c.function["name"] for c in calls] == ["get_weather"]
        assert answer == REPLY
        for chunk in CHUNKS:
            assert streamed(PROMPT_OPENED, text, chunk)[1] == content


class TestRealReasoningIsUntouched:
    """The answer's closer only means a skipped channel when it comes first."""

    @pytest.mark.parametrize(
        "text, expected",
        [
            (
                "Let me think."
                + CHANNEL_THINK_END
                + CHANNEL_RESPONSE_START
                + "42"
                + CHANNEL_RESPONSE_END
                + CHANNEL_MESSAGE_END,
                ("Let me think.", "42"),
            ),
            (
                "Let me think."
                + CHANNEL_RESPONSE_START
                + "42"
                + CHANNEL_RESPONSE_END
                + CHANNEL_MESSAGE_END,
                ("Let me think.", "42"),
            ),
            (
                "Still thinking when max_tokens hit",
                ("Still thinking when max_tokens hit", ""),
            ),
        ],
        ids=["closed-then-answered", "response-opened", "truncated"],
    )
    def test_both_paths_keep_the_reasoning(self, text, expected):
        reasoning, content = PROMPT_OPENED.split(text)
        assert (reasoning, content) == expected
        for chunk in CHUNKS:
            assert streamed(PROMPT_OPENED, text, chunk) == expected

    def test_a_call_straight_out_of_the_channel_keeps_its_reasoning(self):
        """`<|open|>tools<|sep|>` before any answer closer is the existing
        implicit exit, not a skipped channel."""
        text = "I should look it up." + CALL + CHANNEL_MESSAGE_END
        reasoning, content = PROMPT_OPENED.split(text)
        assert reasoning == "I should look it up."
        _, calls = parse_tool_calls(content, None, KimiK3Parser)
        assert [c.function["name"] for c in calls] == ["get_weather"]


def test_an_answer_end_marker_must_be_framing():
    """The filter drops the marker itself; a split that kept it would hand the
    two delivery modes different answers."""
    with pytest.raises(ValueError, match="content_framing"):
        ReasoningDialect(
            prompt_open_marker="<a>",
            output_open_marker="<a>",
            think_end_marker="</a>",
            split=lambda text, starts: None,
            answer_end_markers=("</answer>",),
        )

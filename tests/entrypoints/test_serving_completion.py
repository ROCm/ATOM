# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`/v1/completions` reports why generation stopped, in OpenAI's vocabulary.

There was no test file for this endpoint at all, and no `finish_reason`
assertion anywhere for it -- which is how it stayed the one path that never
normalised. The chat path and atomesh's completion twin both do; this one
forwarded the engine's own word on the content chunk and hardcoded `"stop"`
on the terminal one, so `max_tokens` -- the only reason that carries a warning
-- was lost on the wire while `stream=false` still reported it.
"""

from __future__ import annotations

import json

import pytest

from atom.entrypoints.openai.protocol import openai_stop_reason
from atom.entrypoints.openai.serving_completion import (
    build_completion_response,
    build_completion_response_multi,
    create_completion_chunk,
)

# Everything `scheduler.py` assigns to `leave_reason`, which is what reaches
# these builders. `stop_<token_id>` is not exotic: `stop_token_ids` is
# `generation_config.eos_token_id` minus the single `tokenizer.eos_token_id`,
# so every model declaring more than one EOS ends there normally -- Qwen3,
# Qwen3.5, gpt-oss.
ENGINE_REASONS = [
    "max_tokens",
    "eos",
    "stop_163586",
    "stop_sequence",
    "aborted",
    "unschedulable: no kv blocks",
]

OPENAI_VOCABULARY = {"stop", "length", "tool_calls"}


def _reason(frame: str) -> str | None:
    """The finish_reason a client reads off one SSE frame."""
    for line in frame.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            return json.loads(line[6:])["choices"][0]["finish_reason"]
    raise AssertionError(f"no data frame in {frame!r}")


def _output(reason: str, text: str = "hi") -> dict:
    return {
        "text": text,
        "finish_reason": reason,
        "num_tokens_input": 1,
        "num_tokens_output": 1,
    }


@pytest.mark.parametrize("engine_reason", ENGINE_REASONS)
class TestTheReasonIsTranslatedBeforeItLeaves:
    """One question, three paths, and they have to answer it the same way."""

    def test_the_non_streaming_response_speaks_openai(self, engine_reason):
        resp = build_completion_response("r", "m", _output(engine_reason))
        got = resp.choices[0]["finish_reason"]
        assert got == openai_stop_reason(engine_reason)
        assert got in OPENAI_VOCABULARY, f"{got!r} is not a reason OpenAI defines"

    def test_and_so_does_every_fan_out_sibling(self, engine_reason):
        resp = build_completion_response_multi(
            "r", "m", [_output(engine_reason), _output("eos")]
        )
        assert [c["finish_reason"] for c in resp.choices] == [
            openai_stop_reason(engine_reason),
            "stop",
        ]

    def test_and_the_streamed_answer_agrees_with_the_whole_one(self, engine_reason):
        """The terminal chunk carries it, as on every other streaming path.

        Hardcoding `"stop"` there while the engine's own word went out on the
        *content* chunk meant a client reading the documented place saw a
        truncated response reported as a clean stop -- and `stream=false`
        reported `max_tokens` for the same generation.
        """
        terminal = _reason(
            create_completion_chunk(
                "r", "m", "", openai_stop_reason(engine_reason) or "stop"
            )
        )
        whole = build_completion_response("r", "m", _output(engine_reason)).choices[0]
        assert terminal == whole["finish_reason"]


class TestTheContentChunkDoesNotCarryIt:
    """A reason on a content chunk is a second place to read one from.

    The engine's raw word went out there while the terminal chunk said
    `"stop"`, so the two disagreed within a single response. atomesh's twin
    puts `None` on content chunks for the same reason.
    """

    def test_a_content_chunk_reports_no_reason(self):
        assert _reason(create_completion_chunk("r", "m", "hi")) is None

    def test_an_unfinished_engine_chunk_reports_no_reason(self):
        assert (
            _reason(create_completion_chunk("r", "m", "hi", finish_reason=None)) is None
        )


class TestNothingButOpenAIsOwnWordsReachTheClient:
    def test_no_engine_spelling_survives(self):
        """The whole point, stated once: the engine's vocabulary and OpenAI's
        overlap only at `stop`, and everything else has to be translated."""
        leaked = [
            r for r in ENGINE_REASONS if openai_stop_reason(r) not in OPENAI_VOCABULARY
        ]
        assert not leaked, f"reported verbatim to the client: {leaked}"

    def test_truncation_is_the_one_reason_that_is_not_stop(self):
        """`length` is the only warning in the vocabulary. Collapsing it to
        `stop` is the loss that matters -- the others are all ways of saying
        the model finished."""
        assert openai_stop_reason("max_tokens") == "length"
        assert {openai_stop_reason(r) for r in ENGINE_REASONS if r != "max_tokens"} == {
            "stop"
        }

    def test_no_reason_at_all_stays_none(self):
        """A chunk mid-generation has no reason, and `None` is how the wire
        says so -- not `"stop"`, which would end the response early."""
        assert openai_stop_reason(None) is None

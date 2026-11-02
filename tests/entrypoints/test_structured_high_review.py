# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import asyncio
import json
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from atom.entrypoints.openai.reasoning import ReasoningChannel
from atom.structured_output.validation import (
    structured_response_channel,
    validate_responses_output,
)


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
    from atom.structured_output import prepare_worker_grammar
    from atom.utils import envs

    monkeypatch.setattr(envs, "ATOM_ENABLE_RELAXED_MTP", True)
    with pytest.raises(ValueError, match="relaxed MTP"):
        prepare_worker_grammar(NS(), "{}", None, "fingerprint", 10)

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""V4.1 protocol adaptation and differential checks against the pinned encoder.

Set ATOM_DSV41_MODEL to a local checkpoint for the reference cases. Parser,
request-validation and loader tests also run without a checkpoint or GPU.
"""

import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from atom.entrypoints.openai import api_server
from atom.entrypoints.openai.chat_encoder_adapters import build_message_encoder_adapter
from atom.entrypoints.openai.chat_encoders import (
    _load_encoder_from_dir,
    apply_chat_template,
    resolve_reasoning_toggle,
)
from atom.entrypoints.openai.protocol import ChatCompletionRequest
from atom.entrypoints.openai.reasoning import ReasoningChannel
from atom.entrypoints.openai.serving_chat import resolve_thinking, validate_chat_request
from atom.entrypoints.openai.tool_parser import flatten_tool_events, parse_tool_calls
from atom.entrypoints.openai.tool_parser.deepseekv4_tool_parser import DsmlParser
from atom.entrypoints.openai.tool_parser.deepseekv41_tool_parser import DsmlV41Parser
from atom.entrypoints.openai.tool_parser.registry import resolve_tool_call_parser
from atom.entrypoints.openai.tool_parser.schema import build_param_types
from atom.entrypoints.openai.tool_parser.stream import ToolCallStreamParser
from atom.entrypoints.openai.tool_parser.tool_parser import qualified_tool_name

TOOLS = [
    {
        "type": "function",
        "namespace": {"name": ns, "description": ns + " tools"},
        "function": {
            "name": "lookup",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
            },
        },
    }
    for ns in ("search", "files")
]
CALL = (
    '<｜DSML｜ calls><｜DSML｜ invoke name="search::lookup">'
    '<｜DSML｜ parameter name="query" string="true">中文 & <tag>  </｜DSML｜ parameter>'
    '<｜DSML｜ parameter name="limit" string="false">2</｜DSML｜ parameter>'
    '<｜DSML｜ parameter name="options" string="false">[true, null, {"k": 3}]</｜DSML｜ parameter>'
    "</｜DSML｜ invoke></｜DSML｜ calls>"
)


def call_values(calls):
    return [(c.function["name"], json.loads(c.function["arguments"])) for c in calls]


@pytest.fixture(scope="module")
def encoder():
    directory = Path(os.environ.get("ATOM_DSV41_MODEL", "/mnt/DeepSeek-V4.1-Flash"))
    source = directory / "encoding/encoding.py"
    if not source.is_file():
        pytest.skip(
            "set ATOM_DSV41_MODEL for pinned official-encoder differential tests"
        )
    manifest = json.loads(
        (
            Path(__file__).parents[1]
            / "models/deepseek_v41/fixtures/reference_manifest.json"
        ).read_text()
    )
    assert (
        hashlib.sha256(source.read_bytes()).hexdigest()
        == manifest["sha256"]["encoding/encoding.py"]
    )
    return _load_encoder_from_dir(str(directory))


@pytest.mark.parametrize(
    "model_type, expected",
    [("deepseek_v41", "encoding_dsv41"), ("other", "encoding"), (None, "encoding")],
)
def test_generic_filename_selects_hooks_only_for_declared_model(
    tmp_path, model_type, expected
):
    (tmp_path / "encoding").mkdir()
    (tmp_path / "encoding/encoding.py").write_text(
        "def encode_messages(messages, thinking_mode): return thinking_mode"
    )
    if model_type:
        (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    adapter = _load_encoder_from_dir(str(tmp_path))
    assert adapter.name == expected
    assert adapter.supports_tools == (model_type == "deepseek_v41")


def test_ambiguous_encoder_files_are_not_guessed(tmp_path):
    (tmp_path / "encoding").mkdir()
    for name in ("encoding.py", "encoding_dsv4.py"):
        (tmp_path / "encoding" / name).write_text(
            "raise RuntimeError('must not import')"
        )
    assert _load_encoder_from_dir(str(tmp_path)) is None


@pytest.mark.parametrize("effort", [1, 37, 100])
def test_numeric_effort_preserves_toggle_precedence(effort):
    assert resolve_thinking(ChatCompletionRequest(reasoning_effort=effort)) == (
        None,
        effort,
    )
    assert resolve_thinking(
        ChatCompletionRequest(reasoning_effort=effort, thinking={"type": "disabled"})
    ) == (False, effort)
    assert resolve_thinking(
        ChatCompletionRequest(
            reasoning_effort=effort, thinking={"type": "enabled", "effort": "low"}
        )
    ) == (True, "low")
    assert resolve_thinking(
        ChatCompletionRequest(
            reasoning_effort="none", thinking={"type": "enabled", "effort": effort}
        )
    ) == (False, effort)


@pytest.mark.parametrize("effort", [True, False, 0, 101, 75.0, 1.5])
def test_api_rejects_invalid_numeric_effort(effort):
    with pytest.raises(ValidationError):
        ChatCompletionRequest(reasoning_effort=effort)


@pytest.mark.parametrize("effort", [True, False, 0, 101, 75.0, "75", "medium", [], {}])
def test_adapter_rejects_invalid_native_or_common_effort(effort):
    adapter = build_message_encoder_adapter("encoding_dsv41", lambda messages, **kw: kw)
    for key in ("thinking_effort", "reasoning_effort"):
        with pytest.raises(ValueError, match="reasoning_effort"):
            adapter([], **{key: effort})


def test_namespaces_use_one_identity_for_validation_and_schema_lookup():
    validate_chat_request(
        ChatCompletionRequest(
            tools=TOOLS,
            tool_choice={"type": "function", "function": {"name": "files::lookup"}},
        )
    )
    assert set(build_param_types(TOOLS)) == {"search::lookup", "files::lookup"}
    duplicate = copy.deepcopy(TOOLS[0])
    duplicate.pop("namespace")
    duplicate["function"]["name"] = "search::lookup"
    with pytest.raises(ValueError, match="duplicate"):
        validate_chat_request(ChatCompletionRequest(tools=[TOOLS[0], duplicate]))


@pytest.mark.parametrize(
    "name, namespace",
    [
        ("lookup", {}),
        ("lookup", ""),
        ("lookup", "a::b"),
        ("a::lookup", "b"),
        ("a::b::lookup", None),
        ("::lookup", None),
        ("a::", None),
        ("a:lookup", None),
    ],
)
def test_invalid_namespaces_are_rejected(name, namespace):
    tool = {"function": {"name": name}, "namespace": namespace}
    with pytest.raises(ValueError):
        qualified_tool_name(tool)


@pytest.mark.parametrize("mode", ["chat", "thinking"])
@pytest.mark.parametrize("effort", [None, 1, 37, 100, "low", "high", "max"])
def test_multiturn_prompt_matches_official_encoder(encoder, mode, effort):
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer", "reasoning_content": "earlier"},
        {"role": "system", "content": "updated policy"},
    ]
    original = copy.deepcopy(messages)
    actual = apply_chat_template(
        None, encoder, messages, thinking_mode=mode, thinking_effort=effort
    )
    expected = encoder.encode(messages, thinking_mode=mode, reasoning_effort=effort)
    assert actual == expected
    assert actual.count("Reasoning Effort:") == (mode == "thinking")
    assert actual.endswith(
        "<｜Assistant｜>" + ("<think>" if mode == "thinking" else "</think>")
    )
    assert messages == original


@pytest.mark.parametrize("mode", ["chat", "thinking"])
def test_tool_history_matches_official_order_and_retains_reasoning(encoder, mode):
    calls = [
        {
            "id": str(i),
            "type": "function",
            "namespace": ns,
            "function": {"name": "lookup", "arguments": {"query": ns, "limit": 2}},
        }
        for i, ns in enumerate(("search", "files"))
    ]
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "look up both"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "look in both",
            "tool_calls": calls,
        },
        {"role": "tool", "tool_call_id": "1", "content": "files result"},
        {"role": "tool", "tool_call_id": "0", "content": "search result"},
    ]
    original = copy.deepcopy(messages)
    request = ChatCompletionRequest(messages=messages, tools=TOOLS)
    actual = apply_chat_template(
        None, encoder, request.messages, tools=TOOLS, thinking_mode=mode
    )
    expected_messages = copy.deepcopy(messages)
    expected_messages[0]["tools"] = TOOLS
    expected = encoder.encode(expected_messages, thinking_mode=mode)
    assert actual == expected
    assert actual.index("<tool_result>search result") < actual.index(
        "<tool_result>files result"
    )
    assert ("look in both" in actual) == (mode == "thinking")
    assert messages == original


def test_startup_detects_v41_dialect_and_thinking_switch(encoder):
    assert encoder.name == "encoding_dsv41"
    assert resolve_tool_call_parser(None, None, encoder) is DsmlV41Parser
    assert resolve_reasoning_toggle(None, encoder) == (
        "thinking_mode",
        "chat",
        "thinking",
    )
    prompt = apply_chat_template(
        None,
        encoder,
        [{"role": "user", "content": "hi"}],
        reasoning_effort=20,
        thinking_effort=88,
    )
    assert "Reasoning Effort: 88 " in prompt
    assert DsmlParser.detect(CALL) is False


def test_typed_call_matches_reference_and_all_stream_boundaries(encoder):
    parse_reference = encoder.encode.__globals__["parse_message_from_completion_text"]
    reference_text = encoder.encode(
        [
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "reason.",
                "tool_calls": [
                    {
                        "type": "function",
                        "namespace": "search",
                        "function": {
                            "name": "lookup",
                            "arguments": {
                                "query": "中文 & <tag>  ",
                                "limit": 2,
                                "options": [True, None, {"k": 3}],
                            },
                        },
                    }
                ],
            }
        ],
        context=[{"role": "user", "content": "question"}],
        thinking_mode="thinking",
    )
    expected = parse_reference(reference_text, thinking_mode="thinking")
    reasoning, content = ReasoningChannel(starts_open=True).split(
        reference_text.removesuffix("<｜end▁of▁sentence｜>")
    )
    assert reasoning == expected["reasoning_content"]
    text, calls = parse_tool_calls(content, TOOLS, DsmlV41Parser)
    expected_calls = [
        (qualified_tool_name(c), json.loads(c["function"]["arguments"]))
        for c in expected["tool_calls"]
    ]
    assert call_values(calls) == expected_calls
    assert text.strip() == expected["content"]
    for boundary in range(len(content) + 1):
        stream = ToolCallStreamParser(tools=TOOLS, parser_cls=DsmlV41Parser)
        events = (
            stream.process(content[:boundary])
            + stream.process(content[boundary:])
            + stream.flush()
        )
        actual_text, actual_calls = flatten_tool_events(events)
        assert actual_text == text
        assert call_values(actual_calls) == expected_calls


@pytest.mark.parametrize("suppressed", [True, False])
def test_tool_dispatch_preserves_surrounding_text(suppressed):
    text, calls = parse_tool_calls(
        "before " + CALL + " after", TOOLS, DsmlV41Parser, suppress_calls=suppressed
    )
    assert text == "before  after"
    assert bool(calls) is not suppressed


def test_truncated_namespaced_call_uses_qualified_schema():
    text = '<｜DSML｜ calls><｜DSML｜ invoke name="files::lookup"><｜DSML｜ parameter name="limit">2'
    _, calls = parse_tool_calls(text, TOOLS, DsmlV41Parser)
    assert call_values(calls) == [("files::lookup", {"limit": 2})]


def test_text_content_blocks_reach_official_encoder(encoder):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "first block"},
                {"type": "text", "text": "second block"},
            ],
        }
    ]
    request = ChatCompletionRequest(messages=messages)
    assert apply_chat_template(None, encoder, request.messages) == encoder.encode(
        messages, thinking_mode="thinking"
    )


def test_jinja_keeps_existing_flat_text_message_shape():
    request = ChatCompletionRequest(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first"},
                    {"type": "text", "text": "second"},
                ],
            }
        ]
    )

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return messages

    assert apply_chat_template(Tokenizer(), None, request.messages) == [
        {"role": "user", "content": "first\nsecond"}
    ]


@pytest.fixture
def chat_endpoint(encoder, monkeypatch):
    prompts = []
    for name, value in {
        "tokenizer": None,
        "custom_message_encoder": encoder,
        "reasoning_toggle": ("thinking_mode", "chat", "thinking"),
        "reasoning_dialect": None,
        "model_starts_in_reasoning": True,
        "model_name": "v41",
        "tool_call_parser_cls": DsmlV41Parser,
        "default_chat_template_kwargs": {
            "thinking_mode": "chat",
            "reasoning_effort": 20,
        },
    }.items():
        monkeypatch.setattr(api_server, name, value)

    async def generate(prompt, *args, **kwargs):
        prompts.append(prompt)
        text = "reason.</think>answer" if prompt.endswith("<think>") else "answer"
        yield {
            "text": text,
            "finish_reason": "stop",
            "num_tokens_input": 4,
            "num_tokens_output": 3,
            "ttft": 0.1,
            "tpot": 0.1,
            "latency": 0.3,
        }

    async def consume(generator, *args):
        async for output in generator:
            pass
        return output

    monkeypatch.setattr(api_server, "generate_async", generate)
    monkeypatch.setattr(api_server, "_run_nonstream_with_disconnect", consume)

    def invoke(**kwargs):
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "question"}], **kwargs
        )
        return asyncio.run(api_server.chat_completions(request, None))

    return invoke, prompts


@pytest.mark.parametrize(
    "controls, budget",
    [
        ({"reasoning_effort": 88}, None),
        ({"reasoning_effort": 88, "thinking": {"type": "enabled"}}, 88),
        (
            {
                "reasoning_effort": 88,
                "chat_template_kwargs": {
                    "thinking_mode": "thinking",
                    "reasoning_effort": 37,
                },
            },
            88,
        ),
        (
            {
                "reasoning_effort": 88,
                "thinking": {"type": "disabled"},
                "chat_template_kwargs": {"thinking_mode": "thinking"},
            },
            None,
        ),
        ({"reasoning_effort": "none", "thinking": {"type": "enabled"}}, None),
        (
            {"reasoning_effort": 37, "thinking": {"type": "enabled", "effort": "high"}},
            75,
        ),
    ],
)
def test_actual_chat_handler_merges_controls_before_encoding(
    chat_endpoint, controls, budget
):
    invoke, prompts = chat_endpoint
    response = invoke(**controls)
    message = response.choices[0]["message"]
    assert message["content"] == "answer"
    assert bool(message.get("reasoning_content")) == (budget is not None)
    if budget is None:
        assert "Reasoning Effort:" not in prompts[0]
        assert prompts[0].endswith("</think>")
    else:
        assert f"Reasoning Effort: {budget} " in prompts[0]
        assert prompts[0].endswith("<think>")


@pytest.mark.parametrize(
    "controls",
    [
        {"thinking": {"effort": True}},
        {"thinking": {"effort": 75.0}},
        {"thinking": {"effort": 101}},
        {"chat_template_kwargs": {"reasoning_effort": 0}},
        {"chat_template_kwargs": {"thinking_effort": []}},
    ],
)
def test_invalid_effort_is_http_400_before_generation(chat_endpoint, controls):
    invoke, prompts = chat_endpoint
    with pytest.raises(HTTPException) as exc:
        invoke(**controls)
    assert exc.value.status_code == 400
    assert not prompts


def test_null_outer_namespace_keeps_function_namespace():
    tool = {
        "namespace": None,
        "function": {
            "name": "lookup",
            "namespace": "search",
            "parameters": {"properties": {"limit": {"type": "integer"}}},
        },
    }
    assert qualified_tool_name(tool) == "search::lookup"
    assert build_param_types([tool]) == {"search::lookup": {"limit": "integer"}}


@pytest.mark.parametrize("effort", [0, 101, True])
def test_responses_bad_effort_is_a_client_error(chat_endpoint, effort):
    class Request:
        async def json(self):
            return {"input": "hi", "reasoning": {"effort": effort}}

    response = asyncio.run(api_server.responses_create(Request()))
    assert response.status_code == 400
    assert json.loads(response.body)["error"]["type"] == "invalid_request_error"
    assert not chat_endpoint[1]

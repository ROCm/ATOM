# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The render endpoints report the prompt token IDs their inference routes run."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from atom.entrypoints.openai import api_server
from atom.entrypoints.openai.chat_encoder_adapters import build_message_encoder_adapter
from atom.model_engine.segment_tokenizer import PrefixEncoding

MODEL = "served-model"
TOOL = {
    "type": "function",
    "function": {"name": "lookup", "parameters": {"type": "object"}},
}
ASK = {"role": "user", "content": "Look up ATOM."}
CALLED = {
    "role": "assistant",
    "content": None,
    "reasoning_content": "The user wants ATOM looked up.",
    "tool_calls": [
        {
            "id": "call_0",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"query": "ATOM"}'},
        }
    ],
}
RESULT = {"role": "tool", "tool_call_id": "call_0", "content": "An inference engine."}
CHATS = {
    "plain": {"messages": [ASK]},
    "system": {"messages": [{"role": "system", "content": "Be terse."}, ASK]},
    "tools": {"messages": [ASK], "tools": [TOOL], "tool_choice": "auto"},
    "tool_history": {"messages": [ASK, CALLED, RESULT], "tools": [TOOL]},
    "effort": {"messages": [ASK], "reasoning_effort": "max"},
    "effort_none": {"messages": [ASK], "reasoning_effort": "none"},
    "thinking_off": {"messages": [ASK], "thinking": {"type": "disabled"}},
    "template_kwargs": {
        "messages": [ASK],
        "chat_template_kwargs": {"answer_style": "short"},
    },
}


def tokenize(text: str) -> list[int]:
    """The engine's tokenizer here: one ID per character."""
    return [ord(c) for c in text]


def render(messages, **kwargs) -> str:
    """A template that writes out everything it renders from."""
    return json.dumps([messages, kwargs], sort_keys=True, default=str)


def run_ids(prompt) -> list[int]:
    """The IDs the engine runs for a prompt an inference route submitted."""
    return tokenize(prompt) if isinstance(prompt, str) else list(prompt)


@pytest.fixture
def submitted(monkeypatch):
    """Serve MODEL through DeepSeek-V4.1's adapter over `render`, and collect
    the prompts the inference routes give the engine."""
    prompts = []
    adapter = build_message_encoder_adapter(
        "encoding", render, model_type="deepseek_v41"
    )
    for name, value in {
        "custom_message_encoder": adapter,
        "reasoning_toggle": ("thinking_mode", "chat", "thinking"),
        "default_chat_template_kwargs": {},
        "model_name": MODEL,
        "model_starts_in_reasoning": False,
        "reasoning_dialect": None,
        "tool_call_parser_cls": None,
        "_request_logger": None,
        "tokenizer": SimpleNamespace(
            apply_chat_template=render, encode=tokenize, decode=lambda *a, **kw: ""
        ),
        "engine": SimpleNamespace(
            config=SimpleNamespace(
                generation_config=SimpleNamespace(), max_model_len=8192
            ),
            io_processor=SimpleNamespace(
                encode_prompt=lambda text, **kw: tokenize(text)
            ),
        ),
    }.items():
        monkeypatch.setattr(api_server, name, value)

    async def generate(prompt_or_tokens, *args, **kwargs):
        prompts.append(prompt_or_tokens)
        yield {
            "text": "answer",
            "finish_reason": "eos",
            "num_cached_tokens": 0,
            "num_tokens_input": 1,
            "num_tokens_output": 1,
            "ttft": 0.0,
            "tpot": 0.0,
            "latency": 0.0,
        }

    monkeypatch.setattr(api_server, "generate_async", generate)
    return prompts


def post(path, body):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api_server.app), base_url="http://test"
        ) as client:
            return await client.post(path, json=body)

    return asyncio.run(run())


@pytest.mark.parametrize("encoder", ["deepseek_v41", "jinja"])
@pytest.mark.parametrize("defaults", [{}, {"thinking_mode": "chat"}])
@pytest.mark.parametrize("chat", CHATS.values(), ids=CHATS.keys())
def test_chat_render_is_the_prompt_the_chat_endpoint_runs(
    submitted, monkeypatch, chat, defaults, encoder
):
    monkeypatch.setattr(api_server, "default_chat_template_kwargs", defaults)
    if encoder == "jinja":
        monkeypatch.setattr(api_server, "custom_message_encoder", None)
        monkeypatch.setattr(
            api_server, "reasoning_toggle", ("enable_thinking", False, True)
        )
    body = {"model": MODEL, **chat}
    rendered = post("/v1/chat/completions/render", body)
    assert rendered.status_code == 200, rendered.text
    assert post("/v1/chat/completions", body).status_code == 200
    (prompt,) = submitted
    data = rendered.json()
    assert data["token_ids"] == run_ids(prompt)
    assert data["model"] == MODEL
    assert data["request_id"].startswith("chatcmpl-")


def test_chat_render_returns_the_prompt_token_ids_a_chat_carries(submitted):
    body = {"model": MODEL, "messages": [ASK], "prompt_token_ids": [7, 8, 9]}
    rendered = post("/v1/chat/completions/render", body)
    assert post("/v1/chat/completions", body).status_code == 200
    assert rendered.json()["token_ids"] == run_ids(submitted[0]) == [7, 8, 9]


REFUSED_CHATS = {
    "another_model": {"model": "another-model", "messages": [ASK]},
    "no_messages": {"model": MODEL},
    "untyped_tool": {"model": MODEL, "messages": [ASK], "tools": [{"name": "x"}]},
    "required_tool_while_thinking": {
        "model": MODEL,
        "messages": [ASK],
        "tools": [TOOL],
        "tool_choice": "required",
        "reasoning_effort": "high",
    },
}


@pytest.mark.parametrize("body", REFUSED_CHATS.values(), ids=REFUSED_CHATS.keys())
def test_chat_render_refuses_what_the_chat_endpoint_refuses(submitted, body):
    assert post("/v1/chat/completions", body).status_code == 400
    assert post("/v1/chat/completions/render", body).status_code == 400
    assert submitted == []


def test_chat_render_refuses_image_input(submitted):
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
    body = {"model": MODEL, "messages": [{"role": "user", "content": [image]}]}
    response = post("/v1/chat/completions/render", body)
    assert response.status_code == 400
    assert response.json()["detail"] == "render does not take image input"


PROMPTS = {
    "text": {"prompt": "Once upon a time"},
    "token_ids": {"prompt": [3, 1, 4]},
    "prompt_token_ids": {"prompt_token_ids": [1, 5, 9]},
}


@pytest.mark.parametrize("prompt", PROMPTS.values(), ids=PROMPTS.keys())
def test_completions_render_is_the_prompt_the_completions_endpoint_runs(
    submitted, prompt
):
    body = {"model": MODEL, **prompt}
    rendered = post("/v1/completions/render", body)
    assert rendered.status_code == 200, rendered.text
    assert post("/v1/completions", body).status_code == 200
    (item,) = rendered.json()
    assert item["token_ids"] == run_ids(submitted[0])
    assert item["model"] == MODEL
    assert item["request_id"].startswith("cmpl-")


@pytest.mark.parametrize(
    "body", [{"model": MODEL}, {"model": "another-model", "prompt": "Hi"}]
)
def test_completions_render_refuses_what_the_completions_endpoint_refuses(
    submitted, body
):
    assert post("/v1/completions", body).status_code == 400
    assert post("/v1/completions/render", body).status_code == 400
    assert submitted == []


def test_render_reports_the_ids_of_a_prefix_encoding(submitted, monkeypatch):
    encoded = PrefixEncoding(token_ids=[4, 2], block_hashes=None)
    monkeypatch.setattr(
        api_server.engine.io_processor, "encode_prompt", lambda text, **kw: encoded
    )
    (item,) = post("/v1/completions/render", {"model": MODEL, "prompt": "Hi"}).json()
    assert item["token_ids"] == [4, 2]
    chat = post("/v1/chat/completions/render", {"model": MODEL, "messages": [ASK]})
    assert chat.json()["token_ids"] == [4, 2]

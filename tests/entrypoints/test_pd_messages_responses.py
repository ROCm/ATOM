# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""PD handoff on /v1/messages and /v1/responses, as on /v1/chat/completions.

A PD router sends the prefill leg the request with ``kv_transfer_params``
(``do_remote_decode``), reads the KV location back from the top level of the
non-streaming response, and forwards it to the decode leg. These endpoints used
to drop the field on both legs, so the decode recomputed the whole prompt.
"""

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from atom.entrypoints.openai import api_server
from atom.entrypoints.openai.serving_anthropic import (
    AnthropicMessage,
    AnthropicMessagesRequest,
)

HANDOFF = {
    "do_remote_prefill": True,
    "remote_engine_id": "prefill-0",
    "remote_block_ids": [10, 11],
    "transfer_id": 7,
}
PREFILL_OUTPUT = {
    "do_remote_prefill": True,
    "remote_block_ids": [3, 4],
    "remote_host": "127.0.0.1",
    "transfer_id": 7,
}
MAX_CTX = 8192
LONG_PROMPT_TOKENS = 8000


class _FinishedCollector:
    """A stream that ends on its first chunk."""

    async def get(self):
        return {"text": "", "token_ids": [], "finished": True, "finish_reason": "eos"}


@pytest.fixture
def server(monkeypatch):
    """Stub the engine and record what each endpoint hands to it."""
    calls = SimpleNamespace(generate=[], setup=[], template="<rendered prompt>")

    def encode(text, **_kw):
        return [0] * (LONG_PROMPT_TOKENS if text == "<long prompt>" else 2)

    # Only what the engine does: count what it is handed. Any cut is the
    # production code's (`_prepare_prompt`).
    async def fake_generate_async(prompt, _sampling, _request_id, **kwargs):
        calls.generate.append((prompt, kwargs))
        output = {
            "text": "hello",
            "finish_reason": "eos",
            "num_cached_tokens": 0,
            "num_tokens_input": len(prompt),
            "num_tokens_output": 1,
        }
        if kwargs.get("kv_transfer_params") is not None:
            output["kv_transfer_output_meta_info"] = PREFILL_OUTPUT
        yield output

    async def fake_setup_streaming_request(prompt, _sampling, _request_id, **kwargs):
        calls.setup.append((prompt, kwargs))
        return 1, _FinishedCollector(), len(prompt)

    monkeypatch.setattr(api_server, "model_name", "m")
    monkeypatch.setattr(api_server, "_request_logger", None)
    monkeypatch.setattr(api_server, "default_chat_template_kwargs", {})
    monkeypatch.setattr(api_server, "reasoning_toggle", None)
    monkeypatch.setattr(api_server, "custom_message_encoder", None)
    monkeypatch.setattr(api_server, "tool_call_parser_cls", None)
    monkeypatch.setattr(
        api_server,
        "tokenizer",
        SimpleNamespace(encode=encode, decode=lambda ids, **_kw: f"<{len(ids)}>"),
    )
    monkeypatch.setattr(
        api_server, "apply_chat_template", lambda *_a, **_kw: calls.template
    )
    monkeypatch.setattr(
        api_server,
        "engine",
        SimpleNamespace(
            config=SimpleNamespace(
                generation_config=SimpleNamespace(), max_model_len=MAX_CTX
            ),
            io_processor=SimpleNamespace(encode_prompt=encode),
        ),
    )
    monkeypatch.setattr(api_server, "generate_async", fake_generate_async)
    monkeypatch.setattr(
        api_server, "setup_streaming_request", fake_setup_streaming_request
    )
    monkeypatch.setattr(api_server, "cleanup_stream", lambda *_a, **_kw: None)
    return calls


def _messages(**kwargs):
    request = AnthropicMessagesRequest(
        model="m", messages=[AnthropicMessage(role="user", content="hi")], **kwargs
    )
    return asyncio.run(api_server.anthropic_messages(request, None))


def _responses(body):
    async def post():
        transport = httpx.ASGITransport(app=api_server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            return await client.post("/v1/responses", json=body)

    return asyncio.run(post())


class TestMessages:
    def test_handoff_reaches_the_engine_and_comes_back(self, server):
        response = _messages(
            max_tokens=1,
            kv_transfer_params={**HANDOFF, "prompt_token_ids": [1, 2, 3]},
        )

        assert server.generate[0][1]["kv_transfer_params"] == HANDOFF
        assert response["kv_transfer_params"] == PREFILL_OUTPUT

    def test_streaming_decode_gets_the_handoff(self, server):
        _messages(stream=True, kv_transfer_params=HANDOFF)

        assert server.setup[0][1]["kv_transfer_params"] == HANDOFF

    def test_plain_request_is_unchanged(self, server):
        response = _messages()

        assert server.generate[0][1].get("kv_transfer_params") is None
        assert "kv_transfer_params" not in response


class TestResponses:
    def test_handoff_reaches_the_engine_and_comes_back(self, server):
        response = _responses(
            {
                "model": "m",
                "input": "hi",
                "max_output_tokens": 1,
                "kv_transfer_params": HANDOFF,
            }
        )

        assert response.status_code == 200
        assert server.generate[0][1]["kv_transfer_params"] == HANDOFF
        assert response.json()["kv_transfer_params"] == PREFILL_OUTPUT

    def test_streaming_decode_gets_the_handoff(self, server):
        response = _responses(
            {"model": "m", "input": "hi", "stream": True, "kv_transfer_params": HANDOFF}
        )

        assert response.status_code == 200
        assert server.setup[0][1]["kv_transfer_params"] == HANDOFF

    def test_plain_request_is_unchanged(self, server):
        response = _responses({"model": "m", "input": "hi"})

        assert server.generate[0][1].get("kv_transfer_params") is None
        assert "kv_transfer_params" not in response.json()

    def test_non_object_handoff_is_rejected(self, server):
        response = _responses({"model": "m", "input": "hi", "kv_transfer_params": "x"})

        assert response.status_code == 400
        assert response.json()["error"]["type"] == "invalid_request_error"
        assert server.generate == []


class TestBothLegsTruncateAlike:
    """Prefill gets max_tokens=1, decode the client's value: same prompt."""

    @staticmethod
    def _sent_prompt(server, endpoint, max_tokens, kv):
        server.generate.clear()
        if endpoint == "messages":
            _messages(max_tokens=max_tokens, kv_transfer_params=kv)
        else:
            body = {"model": "m", "input": "hi", "max_output_tokens": max_tokens}
            if kv is not None:
                body["kv_transfer_params"] = kv
            _responses(body)
        return server.generate[0][0]

    @pytest.mark.parametrize("endpoint", ["messages", "responses"])
    def test_with_handoff_the_prompt_does_not_depend_on_max_tokens(
        self, server, endpoint
    ):
        server.template = "<long prompt>"

        prefill = self._sent_prompt(server, endpoint, 1, HANDOFF)
        decode = self._sent_prompt(server, endpoint, 4096, HANDOFF)

        assert len(prefill) == len(decode) == MAX_CTX - MAX_CTX // 8

    @pytest.mark.parametrize("endpoint", ["messages", "responses"])
    def test_without_handoff_truncation_still_follows_max_tokens(
        self, server, endpoint
    ):
        server.template = "<long prompt>"

        # max_tokens=1 leaves room for the whole prompt; 4096 does not.
        assert len(self._sent_prompt(server, endpoint, 1, None)) == LONG_PROMPT_TOKENS
        assert (
            len(self._sent_prompt(server, endpoint, 4096, None))
            == MAX_CTX - MAX_CTX // 8
        )

    @pytest.mark.parametrize("endpoint", ["messages", "responses"])
    def test_input_tokens_are_the_engine_count_of_what_it_ran(self, server, endpoint):
        server.template = "<long prompt>"
        if endpoint == "messages":
            usage = _messages(max_tokens=4096)["usage"]
        else:
            body = {"model": "m", "input": "hi", "max_output_tokens": 4096}
            usage = _responses(body).json()["usage"]
        assert usage["input_tokens"] == MAX_CTX - MAX_CTX // 8


class TestStructuredResponsesCutFirst:
    """A structured /v1/responses request builds its reasoning channel into the
    sampling params before the engine call; both are seeded from the text the
    engine runs."""

    def test_over_the_cap_the_engine_gets_the_kept_ids(self, server, monkeypatch):
        server.template = "<long prompt>"
        configured = []
        seeds = []

        async def configure(_sampling, schema, channel, **_kw):
            configured.append(schema)

        monkeypatch.setattr(api_server, "_configure_structured_output", configure)
        monkeypatch.setattr(
            api_server,
            "prompt_starts_in_reasoning",
            lambda text: seeds.append(text) or False,
        )
        # The fake engine answers "hello", which is not JSON.
        monkeypatch.setattr(
            api_server, "validate_responses_output", lambda *a, **kw: None
        )
        response = _responses(
            {
                "model": "m",
                "input": "hi",
                "max_output_tokens": 4096,
                "text": {"format": {"type": "json_object"}},
            }
        )

        assert response.status_code == 200, response.text
        sent, _kwargs = server.generate[0]
        kept = MAX_CTX - MAX_CTX // 8
        assert configured == ['{"type": "object"}']
        # The engine is handed the kept ids, and does not tokenize again.
        assert sent == [0] * kept
        # The channel was seeded from the cut text, before the engine call.
        assert seeds[0] == f"<{kept}>"

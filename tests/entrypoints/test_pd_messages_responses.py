# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""PD handoff on /v1/messages and /v1/responses, as on /v1/chat/completions.

A PD router sends the prefill leg the request with ``kv_transfer_params``
(``do_remote_decode``), reads the KV location back from the top level of the
non-streaming response, and forwards it to the decode leg. These endpoints used
to drop the field on both legs, so the decode recomputed the whole prompt.
"""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from jinja2 import TemplateError

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

    # Only what the engine does: count what it is handed.
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


def _post(path, body):
    async def post():
        transport = httpx.ASGITransport(app=api_server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            return await client.post(path, json=body)

    return asyncio.run(post())


def _responses(body):
    return _post("/v1/responses", body)


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


def _admit(monkeypatch):
    """Run the admission check the real engine call runs."""

    async def admit(prompt, sampling, _request_id, **_kw):
        seq = SimpleNamespace(
            num_prompt_tokens=len(prompt), max_tokens=sampling.max_tokens
        )
        api_server._validate_sequence_context_length(seq, sampling)
        yield {}

    monkeypatch.setattr(api_server, "generate_async", admit)


class TestNotCut:
    """Neither endpoint cuts an over-long prompt: a cut drops its end, the
    latest turn and the generation prompt, and answers 200 on the rest.
    Admission refuses it instead, and both PD legs run the same prompt."""

    @pytest.mark.parametrize("endpoint", ["messages", "responses"])
    @pytest.mark.parametrize("kv", [None, HANDOFF])
    @pytest.mark.parametrize("max_tokens", [1, 4096])
    def test_the_whole_prompt_reaches_the_engine(
        self, server, endpoint, kv, max_tokens
    ):
        # The stub engine skips admission, so with 4096 this request, which
        # admission refuses, still shows what the engine would be handed.
        server.template = "<long prompt>"
        if endpoint == "messages":
            _messages(max_tokens=max_tokens, kv_transfer_params=kv)
        else:
            body = {"model": "m", "input": "hi", "max_output_tokens": max_tokens}
            if kv is not None:
                body["kv_transfer_params"] = kv
            _responses(body)
        assert len(server.generate[0][0]) == LONG_PROMPT_TOKENS

    @pytest.mark.parametrize("endpoint", ["messages", "responses"])
    def test_input_tokens_are_the_whole_prompt(self, server, endpoint):
        server.template = "<long prompt>"
        if endpoint == "messages":
            usage = _messages(max_tokens=1)["usage"]
        else:
            body = {"model": "m", "input": "hi", "max_output_tokens": 1}
            usage = _responses(body).json()["usage"]
        assert usage["input_tokens"] == LONG_PROMPT_TOKENS

    def test_responses_refuses_with_openais_context_code(self, server, monkeypatch):
        # Codex reads this code: it compacts, or reports an exhausted context.
        _admit(monkeypatch)
        server.template = "<long prompt>"
        reply = _responses({"model": "m", "input": "hi", "max_output_tokens": 4096})
        assert reply.status_code == 400
        error = reply.json()["error"]
        assert error["code"] == "context_length_exceeded"
        assert error["message"].startswith("This model's maximum context length")

    @pytest.mark.parametrize(
        "path, body",
        [
            ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}),
            ("/v1/completions", {"prompt": "<long prompt>"}),
        ],
    )
    def test_chat_and_completions_refuse_with_openais_context_code(
        self, server, monkeypatch, path, body
    ):
        async def refuse(*_a, **_kw):
            # What admission raises for an over-long request.
            api_server._validate_context_length(8000, 4096, MAX_CTX)
            yield {}

        monkeypatch.setattr(api_server, "generate_async", refuse)
        reply = _post(path, {"model": "m", "max_tokens": 4096, **body})
        assert reply.status_code == 400
        error = reply.json()["error"]
        assert error["code"] == "context_length_exceeded"
        assert error["message"].startswith("This model's maximum context length")

    @pytest.mark.parametrize(
        "path, body",
        [
            ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}),
            ("/v1/responses", {"input": "hi"}),
        ],
    )
    def test_any_other_value_error_keeps_code_400(
        self, server, monkeypatch, path, body
    ):
        async def refuse(*_a, **_kw):
            raise ValueError("bad request")
            yield {}

        monkeypatch.setattr(api_server, "generate_async", refuse)
        reply = _post(path, {"model": "m", "max_tokens": 16, **body})
        assert reply.status_code == 400
        assert reply.json()["error"] == {
            "message": "bad request",
            "type": "invalid_request_error",
            "code": 400,
        }

    @pytest.mark.parametrize(
        "prompt_tokens, max_tokens, message",
        [
            # The prompt fits, its output does not: Claude Code lowers max_tokens.
            (
                8000,
                4096,
                (
                    "input length and `max_tokens` exceed context limit: 8000 + "
                    "4096 > 8192, decrease input length or `max_tokens` and try again"
                ),
            ),
            # The prompt alone fills the context: Claude Code compacts.
            (8192, 16, "prompt is too long: 8192 tokens > 8191 maximum"),
            # max_tokens omitted: admission clamps it, and refuses only a full one.
            (8192, None, "prompt is too long: 8192 tokens > 8191 maximum"),
        ],
    )
    def test_an_over_long_request_is_refused_in_anthropic_wording(
        self, server, monkeypatch, prompt_tokens, max_tokens, message
    ):
        monkeypatch.setattr(
            api_server.engine.io_processor,
            "encode_prompt",
            lambda _text, **_kw: [0] * prompt_tokens,
        )

        _admit(monkeypatch)
        kwargs = {} if max_tokens is None else {"max_tokens": max_tokens}
        reply = _messages(**kwargs)
        assert reply.status_code == 400
        assert json.loads(reply.body)["error"] == {
            "type": "invalid_request_error",
            "message": message,
        }

    @pytest.mark.parametrize("max_tokens", [16, 4096])
    def test_a_prompt_the_kv_pool_cannot_hold_is_too_long(
        self, server, monkeypatch, max_tokens
    ):
        # With 4096 the context limit fails too (8000 + 4096 > 8192); the pool
        # answers first, since a smaller max_tokens cannot help.
        monkeypatch.setattr(api_server, "_get_engine_max_pool_tokens", lambda: 4000)
        _admit(monkeypatch)
        server.template = "<long prompt>"
        reply = _messages(max_tokens=max_tokens)
        assert reply.status_code == 400
        assert json.loads(reply.body)["error"]["message"] == (
            "prompt is too long: 8000 tokens > 4000 maximum"
        )

    def test_a_pool_larger_than_the_context_is_not_the_limit_reported(
        self, server, monkeypatch
    ):
        # The pool is sized from free memory, so it is often larger than the
        # context. A prompt past both is too long for the context, not the pool.
        monkeypatch.setattr(api_server, "_get_engine_max_pool_tokens", lambda: 20000)
        monkeypatch.setattr(
            api_server.engine.io_processor,
            "encode_prompt",
            lambda _text, **_kw: [0] * 30000,
        )
        _admit(monkeypatch)
        reply = _messages(max_tokens=16)
        assert reply.status_code == 400
        assert json.loads(reply.body)["error"]["message"] == (
            "prompt is too long: 30000 tokens > 8191 maximum"
        )

    def test_a_smaller_pool_answers_before_the_default_output_clamp(
        self, server, monkeypatch
    ):
        # max_tokens omitted and the prompt fills the context: the pool, the
        # tighter limit, is what the client is told.
        monkeypatch.setattr(api_server, "_get_engine_max_pool_tokens", lambda: 4000)
        monkeypatch.setattr(
            api_server.engine.io_processor,
            "encode_prompt",
            lambda _text, **_kw: [0] * MAX_CTX,
        )
        _admit(monkeypatch)
        reply = _messages()
        assert reply.status_code == 400
        assert json.loads(reply.body)["error"]["message"] == (
            f"prompt is too long: {MAX_CTX} tokens > 4000 maximum"
        )

    @staticmethod
    def _with_mid_system(server, monkeypatch, template, **request_kw):
        monkeypatch.setattr(api_server, "apply_chat_template", template)
        messages = request_kw.pop(
            "messages",
            [
                AnthropicMessage(role="user", content="hi"),
                AnthropicMessage(role="system", content="reminder"),
            ],
        )
        request = AnthropicMessagesRequest(
            model="m", max_tokens=16, messages=messages, **request_kw
        )
        return asyncio.run(api_server.anthropic_messages(request, None))

    def test_a_template_that_refuses_a_system_message_is_a_400(
        self, server, monkeypatch
    ):
        def refuse(_tok, _enc, messages, **_kw):
            if any(m.role == "system" for m in messages[1:]):
                raise TemplateError("System message must be at the beginning.")
            return "<rendered prompt>"

        reply = self._with_mid_system(server, monkeypatch, refuse)
        assert reply.status_code == 400
        # The wording Claude Code retries on, folding the message into a user turn.
        assert json.loads(reply.body)["error"]["message"].startswith(
            'Unexpected role "system"'
        )

    def test_a_template_that_fails_for_another_reason_is_not_blamed_on_it(
        self, server, monkeypatch
    ):
        def fail(*_a, **_kw):
            raise TemplateError("Tool names may not contain '::'.")

        reply = self._with_mid_system(server, monkeypatch, fail)
        # Not the retry-on-fold 400: folding would fail the same way.
        assert reply.status_code == 500
        assert json.loads(reply.body)["error"]["message"] == (
            "Tool names may not contain '::'."
        )

    def test_a_template_with_no_system_role_is_not_blamed_on_a_mid_one(
        self, server, monkeypatch
    ):
        # Gemma style: no system message at all. With only the top-level system
        # prompt there is nothing mid-conversation to fold, so the original
        # error comes back.
        def no_system(_tok, _enc, messages, **_kw):
            if any(m.role == "system" for m in messages):
                raise TemplateError("System role not supported")
            return "<rendered prompt>"

        reply = self._with_mid_system(
            server,
            monkeypatch,
            no_system,
            system="top",
            messages=[AnthropicMessage(role="user", content="hi")],
        )
        assert reply.status_code == 500
        assert json.loads(reply.body)["error"]["message"] == "System role not supported"

    def test_a_failing_re_render_keeps_the_original_error(self, server, monkeypatch):
        def refuse_then_raise(_tok, _enc, messages, **_kw):
            if any(m.role == "system" for m in messages[1:]):
                raise TemplateError("System message must be at the beginning.")
            raise ValueError("something unrelated")

        reply = self._with_mid_system(server, monkeypatch, refuse_then_raise)
        assert reply.status_code == 500
        assert json.loads(reply.body)["error"]["message"] == (
            "System message must be at the beginning."
        )


class TestStructuredResponsesSeeding:
    """A structured /v1/responses request builds its reasoning channel into the
    sampling params before the engine call; both are seeded from the text the
    engine runs."""

    def test_the_engine_gets_the_encoded_ids_and_the_channel_its_text(
        self, server, monkeypatch
    ):
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
                "max_output_tokens": 16,
                "text": {"format": {"type": "json_object"}},
            }
        )

        assert response.status_code == 200, response.text
        sent, _kwargs = server.generate[0]
        assert configured == ['{"type": "object"}']
        # The engine is handed the encoded ids, and does not tokenize again.
        assert sent == [0] * LONG_PROMPT_TOKENS
        assert seeds[0] == "<long prompt>"

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Test prefill prompt-ID responses and decode reuse without tokenizing."""

import asyncio
from types import SimpleNamespace
from uuid import UUID

import fastapi.routing
import httpx
import pytest
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from atom.entrypoints.openai import api_server
from atom.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
)

PROMPT_IDS = [101, 102, 103, 104]


@pytest.fixture
def server(monkeypatch):
    """Stub generation and record the handler's inputs."""
    calls = SimpleNamespace(templates=0, generate=[])

    def record_template(*_args, **_kwargs):
        calls.templates += 1
        return "<rendered prompt>"

    async def fake_generate_async(prompt_or_tokens, _sampling, _request_id, **kwargs):
        calls.generate.append((prompt_or_tokens, kwargs))
        output = {
            "text": "hello",
            "finish_reason": "eos",
            "num_tokens_input": 4,
            "num_tokens_output": 1,
        }
        if kwargs.get("return_token_ids"):
            output["prompt_token_ids"] = list(PROMPT_IDS)
        yield output

    monkeypatch.setattr(api_server, "model_name", "m")
    monkeypatch.setattr(api_server, "_request_logger", None)
    monkeypatch.setattr(api_server, "default_chat_template_kwargs", {})
    monkeypatch.setattr(api_server, "reasoning_toggle", None)
    monkeypatch.setattr(api_server, "custom_message_encoder", None)
    monkeypatch.setattr(api_server, "tool_call_parser_cls", None)
    monkeypatch.setattr(
        api_server,
        "tokenizer",
        SimpleNamespace(
            encode=lambda text, **_kw: [1, 2],
            decode=lambda _ids, **_kw: "tail",
        ),
    )
    monkeypatch.setattr(api_server, "apply_chat_template", record_template)
    monkeypatch.setattr(api_server, "generate_async", fake_generate_async)
    return calls


def _request(endpoint, **kwargs):
    if endpoint == "chat":
        return (
            ChatCompletionRequest(
                model="m",
                messages=[ChatMessage(role="user", content="hi")],
                temperature=1.0,
                **kwargs,
            ),
            api_server.chat_completions,
        )
    return (
        CompletionRequest(model="m", temperature=1.0, **kwargs),
        api_server.completions,
    )


class TestPromptTokenIdRequests:
    @pytest.mark.parametrize("endpoint", ["chat", "completion"])
    @pytest.mark.parametrize("use_ids", [False, True])
    def test_top_level_ids_or_text_reach_generation(self, server, endpoint, use_ids):
        kwargs = {}
        if use_ids:
            kwargs["prompt_token_ids"] = PROMPT_IDS
        elif endpoint == "completion":
            kwargs["prompt"] = "hi"
        request, handler = _request(endpoint, **kwargs)

        asyncio.run(handler(request, None))

        text = "<rendered prompt>" if endpoint == "chat" else "hi"
        assert server.generate[0][0] == (PROMPT_IDS if use_ids else text)
        assert server.templates == int(endpoint == "chat" and not use_ids)

    @pytest.mark.parametrize("endpoint", ["chat", "completion"])
    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize("n", [1, 2])
    def test_handoff_preserves_kv_metadata_and_original_request(
        self, monkeypatch, server, endpoint, stream, n
    ):
        async def fake_fanout(prompt, _sampling, _rid, **kwargs):
            server.generate.append((prompt, kwargs))
            return [
                {
                    "text": "hello",
                    "finish_reason": "eos",
                    "num_tokens_input": len(PROMPT_IDS),
                    "num_tokens_output": 1,
                }
                for _ in range(n)
            ]

        async def fake_setup(prompt, _sampling, _rid, **kwargs):
            server.generate.append((prompt, kwargs))
            return ([7, 8] if n > 1 else 7), object(), len(PROMPT_IDS)

        monkeypatch.setattr(api_server, "generate_async_fanout", fake_fanout)
        monkeypatch.setattr(api_server, "setup_streaming_request", fake_setup)
        monkeypatch.setattr(api_server, "setup_streaming_request_fanout", fake_setup)
        metadata = {
            "do_remote_prefill": True,
            "remote_engine_id": "prefill-0",
            "remote_block_ids": [10, 11],
            "connector_metadata": {"transfer_id": "transfer-1"},
        }
        request, handler = _request(
            endpoint,
            kv_transfer_params={**metadata, "prompt_token_ids": PROMPT_IDS},
            stream=stream,
            n=n,
            # Both default and explicit opt-out allow streaming.
            return_token_ids=None if n == 1 else False,
        )
        original = request.kv_transfer_params

        asyncio.run(handler(request, None))

        assert len(server.generate) == 1
        sent, options = server.generate[0]
        assert sent == PROMPT_IDS
        assert options["kv_transfer_params"] == metadata
        assert options["kv_transfer_params"] is not original
        assert (
            options["kv_transfer_params"]["remote_block_ids"]
            is original["remote_block_ids"]
        )
        assert request.kv_transfer_params is original
        assert original == {**metadata, "prompt_token_ids": PROMPT_IDS}
        assert server.templates == 0

    @pytest.mark.parametrize("endpoint", ["chat", "completion"])
    def test_conflicting_ids_are_rejected_before_cleanup(self, server, endpoint):
        # Invalid ID values are covered by test_protocol.py.
        request, handler = _request(
            endpoint,
            prompt_token_ids=PROMPT_IDS,
            kv_transfer_params={"prompt_token_ids": [999]},
        )

        with pytest.raises(api_server.HTTPException) as excinfo:
            asyncio.run(handler(request, None))

        assert excinfo.value.status_code == 400
        assert server.generate == []
        assert request.kv_transfer_params["prompt_token_ids"] == [999]

    @pytest.mark.parametrize("endpoint", ["chat", "completion"])
    def test_streaming_cannot_return_prompt_ids(self, server, endpoint):
        kwargs = {"prompt": "hi"} if endpoint == "completion" else {}
        request, handler = _request(
            endpoint, return_token_ids=True, stream=True, **kwargs
        )

        with pytest.raises(api_server.HTTPException) as excinfo:
            asyncio.run(handler(request, None))

        assert excinfo.value.status_code == 400
        assert "stream" in excinfo.value.detail
        assert server.generate == []


class TestHttpResponseSerialization:
    @pytest.mark.parametrize("endpoint", ["chat", "completion"])
    @pytest.mark.parametrize("return_ids", [False, True])
    @pytest.mark.parametrize("n", [1, 2])
    def test_wire_response_is_preserved_without_fastapi_recursive_conversion(
        self, monkeypatch, server, endpoint, return_ids, n
    ):
        # Check the HTTP path bypasses FastAPI's recursive ID conversion.
        suffix = "chat/completions" if endpoint == "chat" else "completions"
        builder = f"build_{endpoint}_response" + ("_multi" if n > 1 else "")
        original_builder = getattr(api_server, builder)
        expected_bodies = []
        ids = list(range(1000, 5096))
        metadata = {
            "do_remote_prefill": True,
            "remote_engine_id": UUID("b1382773-c171-4e07-b42d-e7ff18a825ab"),
            "remote_block_ids": (7, 8),
        }
        output = {
            "text": '你好，世界 🌍\n"quoted"',
            "finish_reason": "eos",
            "num_tokens_input": len(ids),
            "num_tokens_output": 1,
            "num_cached_tokens": 256,
            "kv_transfer_output_meta_info": metadata,
        }
        outputs = [dict(output) for _ in range(n)]
        if return_ids:
            # Fanout returns shared prompt IDs only on its first output.
            outputs[0]["prompt_token_ids"] = ids

        async def generate(*_args, **kwargs):
            assert kwargs["return_token_ids"] is return_ids
            yield outputs[0]

        async def fanout(*_args, **kwargs):
            assert kwargs["return_token_ids"] is return_ids
            return outputs

        def build(*args, **kwargs):
            response = original_builder(*args, **kwargs)
            # Preserve the previous JSON encoding, including Unicode and metadata.
            expected_bodies.append(JSONResponse(jsonable_encoder(response)).body)
            return response

        def reject_automatic_conversion(*_args, **_kwargs):
            pytest.fail("FastAPI recursively converted the completed response")

        monkeypatch.setattr(api_server, "generate_async", generate)
        monkeypatch.setattr(api_server, "generate_async_fanout", fanout)
        monkeypatch.setattr(api_server, builder, build)
        monkeypatch.setattr(
            fastapi.routing, "jsonable_encoder", reject_automatic_conversion
        )
        payload = {
            "model": "m",
            "temperature": 1.0,
            "n": n,
            "return_token_ids": return_ids,
        }
        payload.update(
            {"messages": [{"role": "user", "content": "hi"}]}
            if endpoint == "chat"
            else {"prompt": "hi"}
        )

        async def post():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=api_server.app),
                base_url="http://test",
            ) as client:
                return await client.post(f"/v1/{suffix}", json=payload)

        response = asyncio.run(post())
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        assert response.content == expected_bodies[0]
        body = response.json()
        assert body["prompt_token_ids"] == (ids if return_ids else None)
        if n == 1:
            assert body["kv_transfer_params"]["remote_block_ids"] == [7, 8]
        else:
            # Existing multi-choice builders do not return KV transfer metadata.
            assert body["kv_transfer_params"] is None
        assert len(body["choices"]) == n

    @pytest.mark.parametrize("endpoint", ["chat", "completion"])
    def test_http_prefill_ids_can_be_reused_by_decode(self, server, endpoint):
        suffix = "chat/completions" if endpoint == "chat" else "completions"
        payload = {"model": "m", "return_token_ids": True}
        payload.update(
            {"messages": [{"role": "user", "content": "hi"}]}
            if endpoint == "chat"
            else {"prompt": "hi"}
        )

        async def round_trip():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=api_server.app),
                base_url="http://test",
            ) as client:
                prefill = await client.post(f"/v1/{suffix}", json=payload)
                assert prefill.status_code == 200
                payload["return_token_ids"] = False
                payload["kv_transfer_params"] = {
                    "do_remote_prefill": True,
                    "prompt_token_ids": prefill.json()["prompt_token_ids"],
                }
                return await client.post(f"/v1/{suffix}", json=payload)

        response = asyncio.run(round_trip())
        assert response.status_code == 200
        assert server.generate[1][0] == PROMPT_IDS
        assert server.generate[1][1]["kv_transfer_params"] == {
            "do_remote_prefill": True
        }
        assert server.templates == (1 if endpoint == "chat" else 0)


@pytest.mark.parametrize("return_ids", [False, True])
def test_generate_async_reads_prompt_ids_from_local_sequence(monkeypatch, return_ids):
    seq = SimpleNamespace(
        id=7,
        prompt_token_ids=list(PROMPT_IDS),
        num_prompt_tokens=len(PROMPT_IDS),
        max_tokens=16,
    )

    def preprocess(_prompt, _sampling, stream_callback=None, **_kwargs):
        # Run the completion callback from the preprocessing executor thread.
        stream_callback(
            SimpleNamespace(
                output_tokens=[5],
                finished=True,
                finish_reason="eos",
                num_cached_tokens=0,
            )
        )
        return seq

    engine = SimpleNamespace(
        io_processor=SimpleNamespace(preprocess=preprocess, requests={}),
        core_mgr=SimpleNamespace(add_request=lambda _seqs: None),
    )
    monkeypatch.setattr(api_server, "engine", engine)
    monkeypatch.setattr(
        api_server, "_validate_sequence_context_length", lambda _s: None
    )
    monkeypatch.setattr(api_server, "delivered_text", lambda _ids: "hello")
    monkeypatch.setattr(api_server, "record_nonstream_first_token", lambda: None)

    async def collect():
        return [
            output
            async for output in api_server.generate_async(
                PROMPT_IDS, object(), "req-1", return_token_ids=return_ids
            )
        ]

    outputs = asyncio.run(collect())
    assert len(outputs) == 1
    output = outputs[0]
    assert output["num_tokens_input"] == len(PROMPT_IDS)
    if return_ids:
        assert output["prompt_token_ids"] == PROMPT_IDS
    else:
        assert "prompt_token_ids" not in output

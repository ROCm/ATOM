# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""ATOM_PD_LATE_KV: decode leg of the router's concurrent P/D dispatch."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from starlette.requests import Request

from atom.entrypoints.openai import api_server
from atom.entrypoints.openai.protocol import ChatCompletionRequest

PROMPT_IDS = [101, 102, 103, 104]
PATH = api_server.PD_LATE_KV_PATH
KV = {
    "do_remote_prefill": True,
    "do_remote_decode": False,
    "transfer_id": "early-abc",
    "remote_dp_rank": 3,
    "remote_tp_size": 1,
    "remote_dp_size": 4,
}


@pytest.fixture
def server(monkeypatch):
    """Stub generation and record what reaches the engine."""
    calls = SimpleNamespace(templates=0, generate=[], events=[])

    def record_template(*_args, **_kwargs):
        calls.templates += 1
        return "<rendered prompt>"

    async def fake_generate_async(prompt_or_tokens, _sampling, _request_id, **kwargs):
        calls.events.append("enqueue")
        calls.generate.append((prompt_or_tokens, kwargs))
        yield {
            "text": "hello",
            "finish_reason": "eos",
            "num_tokens_input": len(PROMPT_IDS),
            "num_tokens_output": 1,
        }

    class SpyRequest(ChatCompletionRequest):
        @classmethod
        def model_validate(cls, *args, **kwargs):
            calls.events.append("validated")
            return super().model_validate(*args, **kwargs)

    monkeypatch.setattr(api_server, "ChatCompletionRequest", SpyRequest)
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
    monkeypatch.setenv("ATOM_PD_LATE_KV", "1")
    return calls


def _head(**extra):
    body = {
        "model": "m",
        "temperature": 1.0,
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "line one\nline two"}],
        "prompt_token_ids": PROMPT_IDS,
    }
    body.update(extra)
    return json.dumps(body, separators=(",", ":")).encode() + b"\n"


def _kv_line(kv=KV):
    return json.dumps({"kv_transfer_params": kv}).encode() + b"\n"


def _post(chunks, events=None, gap_s=0.05):
    """POST a streamed body; `None` in `chunks` is a pause before the next one."""

    async def body():
        for chunk in chunks:
            if chunk is None:
                await asyncio.sleep(gap_s)
                if events is not None:
                    events.append("kv_sent")
                continue
            yield chunk

    async def post():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api_server.app), base_url="http://d"
        ) as client:
            return await client.post(
                PATH,
                content=body(),
                headers={"content-type": "application/json", "x-atom-pd-late-kv": "1"},
            )

    return asyncio.run(asyncio.wait_for(post(), 10))


def test_flag_off_route_is_404_so_the_router_falls_back(server, monkeypatch):
    monkeypatch.setenv("ATOM_PD_LATE_KV", "0")
    response = _post([_head(), _kv_line()])
    assert response.status_code == 404
    assert server.generate == []


def test_request_is_validated_before_kv_arrives_and_enqueued_after(server):
    response = _post([_head(), None, _kv_line()], events=server.events)
    assert response.status_code == 200, response.text
    # Parsing and validation overlapped the wait for P; the engine saw
    # nothing until the KV metadata was in.
    assert server.events == ["validated", "kv_sent", "enqueue"]
    sent, options = server.generate[0]
    assert sent == PROMPT_IDS
    assert options["kv_transfer_params"] == KV
    assert server.templates == 0
    assert response.json()["choices"][0]["message"]["content"] == "hello"


@pytest.mark.parametrize("chunk_size", [1, 7, 4096])
def test_arbitrary_chunk_boundaries(server, chunk_size):
    raw = _head() + _kv_line()
    chunks = [raw[i : i + chunk_size] for i in range(0, len(raw), chunk_size)]
    response = _post(chunks)
    assert response.status_code == 200, response.text
    assert server.generate[0][1]["kv_transfer_params"] == KV


def test_echoed_ids_in_kv_line_are_reused_without_tokenizing(server):
    head = json.loads(_head())
    del head["prompt_token_ids"]
    line = json.dumps(head).encode() + b"\n"
    response = _post([line, _kv_line({**KV, "prompt_token_ids": PROMPT_IDS})])
    assert response.status_code == 200, response.text
    sent, options = server.generate[0]
    assert sent == PROMPT_IDS
    assert "prompt_token_ids" not in options["kv_transfer_params"]
    assert server.templates == 0


@pytest.mark.parametrize(
    "chunks,detail",
    [
        ([], "no request line"),
        ([_head().rstrip(b"\n")], "no request line"),
        ([_head()], "ended before kv_transfer_params"),
        ([b"[1,2]\n", _kv_line()], "must be a JSON object"),
        ([b"{nope\n", _kv_line()], "not JSON"),
        ([_head(kv_transfer_params=KV), _kv_line()], "must not carry"),
        ([_head(), b'{"kv_transfer_params": 5}\n'], "kv_transfer_params object"),
        ([_head(), b"{oops\n"], "not JSON"),
        ([_head(), _kv_line(), b'{"x":1}\n'], "data after"),
    ],
)
def test_malformed_bodies_are_rejected_before_enqueue(server, chunks, detail):
    response = _post(chunks)
    assert response.status_code == 400, response.text
    assert detail in response.text
    assert server.generate == []


def test_trailing_whitespace_is_tolerated(server):
    response = _post([_head(), _kv_line(), b"\n  \n"])
    assert response.status_code == 200, response.text


def test_invalid_request_line_is_422_like_the_plain_route(server):
    response = _post([_head(messages="not-a-list"), _kv_line()])
    assert response.status_code == 422, response.text
    assert server.generate == []


def test_router_disconnect_before_kv_enqueues_nothing(server):
    """P failed: the router drops the leg after line 1."""
    messages = [
        {"type": "http.request", "body": _head(), "more_body": True},
        {"type": "http.disconnect"},
    ]

    async def receive():
        return messages.pop(0)

    scope = {
        "type": "http",
        "method": "POST",
        "path": PATH,
        "headers": [(b"content-type", b"application/json")],
        "query_string": b"",
    }
    response = asyncio.run(
        api_server.pd_late_kv_chat_completions(Request(scope, receive))
    )
    assert response.status_code == 499
    assert server.events == ["validated"]
    assert server.generate == []


def test_plain_chat_route_is_unchanged(server):
    async def post():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api_server.app), base_url="http://d"
        ) as client:
            return await client.post(
                "/v1/chat/completions", content=_head() + _kv_line()
            )

    # Two JSON documents are not a chat request on the normal route.
    response = asyncio.run(post())
    assert response.status_code in (400, 422)
    assert server.generate == []


def test_chunked_http_body_through_uvicorn(server):
    """The router streams with Transfer-Encoding: chunked; check the real
    uvicorn + middleware stack hands the lines over as they arrive."""
    import socket
    import threading

    import uvicorn

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    config = uvicorn.Config(
        api_server.app, host="127.0.0.1", port=port, log_level="error"
    )
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    try:
        for _ in range(200):
            if srv.started:
                break
            threading.Event().wait(0.02)
        assert srv.started

        async def body():
            yield _head()
            await asyncio.sleep(0.2)
            server.events.append("kv_sent")
            yield _kv_line()

        async def post():
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as c:
                return await c.post(
                    PATH, content=body(), headers={"x-atom-pd-late-kv": "1"}
                )

        response = asyncio.run(asyncio.wait_for(post(), 10))
        assert response.status_code == 200, response.text
        assert server.events == ["validated", "kv_sent", "enqueue"]
        assert server.generate[0][1]["kv_transfer_params"] == KV
    finally:
        srv.should_exit = True
        thread.join(5)


def test_env_default_is_off(monkeypatch):
    from atom.utils import envs

    monkeypatch.delenv("ATOM_PD_LATE_KV", raising=False)
    assert envs.ATOM_PD_LATE_KV is False
    monkeypatch.setenv("ATOM_PD_LATE_KV", "1")
    assert envs.ATOM_PD_LATE_KV is True

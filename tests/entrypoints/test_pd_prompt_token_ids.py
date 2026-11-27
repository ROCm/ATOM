# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Test prefill prompt-ID responses and decode reuse without tokenizing."""

import array
import asyncio
import base64
import importlib.util
import pathlib
from types import SimpleNamespace
from uuid import UUID

import fastapi.routing
import httpx
import numpy as np
import pytest
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from atom.entrypoints.openai import api_server
from atom.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
)
from atom.model_engine.multimodal_runtime import multimodal_cache_seed
from atom.model_engine.sequence import Sequence, SequenceStatus
from atom.sampling_params import SamplingParams
from tests.test_scheduler import _ConnectorNamespace

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
@pytest.mark.parametrize("multimodal", [False, True])
def test_generate_async_reads_prompt_ids_from_local_sequence(
    monkeypatch, return_ids, multimodal
):
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
                logprobs=None,
                finished=True,
                finish_reason="eos",
                num_cached_tokens=0,
                kv_transfer_params_output=None,
            )
        )
        return seq

    engine = SimpleNamespace(
        io_processor=SimpleNamespace(preprocess=preprocess, requests={}),
        core_mgr=SimpleNamespace(add_request=lambda _seqs: None),
    )
    monkeypatch.setattr(api_server, "engine", engine)
    monkeypatch.setattr(
        api_server, "_validate_sequence_context_length", lambda _s, _sampling=None: None
    )
    monkeypatch.setattr(api_server, "delivered_text", lambda _ids: "hello")
    monkeypatch.setattr(api_server, "record_nonstream_first_token", lambda: None)

    async def collect():
        # Read the IDs retained by preprocessing, rather than echoing its
        # input: multimodal preprocessing may have expanded placeholders.
        generate = (
            api_server.generate_async_multimodal
            if multimodal
            else api_server.generate_async
        )
        args = (
            ([11, 12], {"image": "fixture"}, SamplingParams(), "req-1")
            if multimodal
            else ([11, 12], SamplingParams(), "req-1")
        )
        return [output async for output in generate(*args, return_token_ids=return_ids)]

    outputs = asyncio.run(collect())
    assert len(outputs) == 1
    output = outputs[0]
    assert output["num_tokens_input"] == len(PROMPT_IDS)
    if return_ids:
        assert output["prompt_token_ids"] == PROMPT_IDS
    else:
        assert "prompt_token_ids" not in output


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("multi", [False, True])
def test_chunked_transfer_discovery_advertises_pp_and_page_geometry(
    monkeypatch, enabled, multi
):
    sub = {
        "kv_connector": "mooncake",
        "kv_role": "kv_producer",
        "enable_chunked_transfer": enabled,
        "handshake_port": 6401,
    }
    config = SimpleNamespace(
        kv_transfer_config=(
            {"kv_connector": "multi", "connectors": [sub]} if multi else sub
        ),
        tensor_parallel_size=4,
        pipeline_parallel_size=3,
        parallel_config=SimpleNamespace(data_parallel_size=2),
        kv_cache_block_size=64,
        decode_context_parallel_size=2,
    )
    monkeypatch.setattr(api_server, "engine", SimpleNamespace(config=config))
    monkeypatch.setattr("atom.utils.network.get_ip", lambda: "127.0.0.1")
    info = asyncio.run(api_server.kv_transfer_info())
    assert info["kv_role"] == "kv_producer"
    assert info["handshake_port"] == 6401
    assert ("chunked_transfer" in info) is enabled
    if enabled:
        assert info["chunked_transfer"] == {
            "remote_host": "127.0.0.1",
            "remote_handshake_port": 6401,
            "remote_pp_size": 3,
            "block_size": 64,
            "dcp_size": 2,
        }


class TestChunkedAdmissionTopology:
    @pytest.mark.parametrize(
        "endpoint", ["chat/completions", "completions", "messages", "responses"]
    )
    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize(
        "p_tp,d_tp,d_pp", [(2, 1, 1), (3, 4, 1), (1, 1, 2), (0, 1, 1)]
    )
    def test_incompatible_pair_is_http_400_before_enqueue(
        self, monkeypatch, server, endpoint, stream, p_tp, d_tp, d_pp
    ):
        monkeypatch.setattr(
            api_server,
            "engine",
            SimpleNamespace(
                config=SimpleNamespace(
                    tensor_parallel_size=d_tp,
                    pipeline_parallel_size=d_pp,
                )
            ),
        )

        async def unexpected_enqueue(*_args, **_kwargs):
            pytest.fail("incompatible D request must be rejected before enqueue")

        monkeypatch.setattr(api_server, "setup_streaming_request", unexpected_enqueue)
        body = {
            "model": "m",
            "max_tokens": 2,
            "stream": stream,
            "kv_transfer_params": {
                "chunked_transfer": True,
                "do_remote_prefill": True,
                "transfer_id": "rejected-pair",
                "remote_tp_size": p_tp,
            },
        }
        if endpoint == "completions":
            body["prompt"] = "hi"
        elif endpoint == "responses":
            body.pop("max_tokens")
            body["input"] = "hi"
        else:
            body["messages"] = [{"role": "user", "content": "hi"}]

        async def post():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=api_server.app),
                base_url="http://d",
            ) as client:
                return await client.post(f"/v1/{endpoint}", json=body)

        response = asyncio.run(asyncio.wait_for(post(), 5))
        assert response.status_code == 400, response.text
        assert "Chunked P/D requires" in response.text
        assert not server.generate and not server.templates

    @pytest.mark.parametrize("p_tp,d_tp", [(1, 1), (1, 4), (2, 4), (4, 4)])
    def test_supported_receive_keeps_admission_path(self, monkeypatch, p_tp, d_tp):
        monkeypatch.setattr(
            api_server,
            "engine",
            SimpleNamespace(
                config=SimpleNamespace(
                    tensor_parallel_size=d_tp,
                    pipeline_parallel_size=1,
                    decode_context_parallel_size=1,
                )
            ),
        )
        api_server._validate_chunked_receive(
            {
                "chunked_transfer": True,
                "do_remote_prefill": True,
                "remote_tp_size": p_tp,
            }
        )

    @pytest.mark.parametrize("model_type", ["kimi_linear", "deepseek_v3"])
    def test_a_dcp_decode_refuses_chunked_receive_only_for_a_stateful_model(
        self, monkeypatch, model_type
    ):
        monkeypatch.setattr(
            api_server,
            "engine",
            SimpleNamespace(
                config=SimpleNamespace(
                    tensor_parallel_size=8,
                    pipeline_parallel_size=1,
                    decode_context_parallel_size=8,
                    hf_config=SimpleNamespace(model_type=model_type),
                )
            ),
        )
        params = {"chunked_transfer": True, "do_remote_prefill": True}
        if model_type == "deepseek_v3":
            api_server._validate_chunked_receive(params)
            return
        with pytest.raises(api_server.HTTPException) as refused:
            api_server._validate_chunked_receive(params)
        assert refused.value.status_code == 400
        assert "per-request state to a DCP decode" in refused.value.detail

    @pytest.mark.parametrize(
        "params",
        [
            None,
            {},
            {"do_remote_prefill": True},
            {"chunked_transfer": True, "do_remote_decode": True},
        ],
    )
    def test_legacy_offload_and_producer_do_not_require_decode_topology(
        self, monkeypatch, params
    ):
        monkeypatch.setattr(api_server, "engine", None)
        api_server._validate_chunked_receive(params)


@pytest.mark.parametrize("return_ids", [False, True])
@pytest.mark.parametrize("n", [1, 2])
def test_multimodal_chat_returns_requested_prompt_ids(
    monkeypatch, server, return_ids, n
):
    media = {"image": "fixture"}
    calls = []

    def output():
        result = {
            "text": "red",
            "finish_reason": "stop",
            "num_tokens_input": len(PROMPT_IDS),
            "num_tokens_output": 1,
        }
        if return_ids:
            result["prompt_token_ids"] = list(PROMPT_IDS)
        return result

    async def single(ids, images, _sampling, _request_id, **kwargs):
        assert ids == PROMPT_IDS and images is media
        assert kwargs["return_token_ids"] is return_ids
        calls.append("single")
        yield output()

    async def fanout(ids, _sampling, _request_id, **kwargs):
        assert ids == PROMPT_IDS and kwargs["multimodal_data"] is media
        assert kwargs["return_token_ids"] is return_ids
        calls.append("fanout")
        return [output() for _ in range(n)]

    monkeypatch.setattr(api_server, "_get_multimodal_processor", lambda: object())
    monkeypatch.setattr(
        api_server,
        "_prepare_multimodal_inputs",
        lambda *_args: (list(PROMPT_IDS), media),
    )
    monkeypatch.setattr(api_server, "generate_async_multimodal", single)
    monkeypatch.setattr(api_server, "generate_async_fanout", fanout)
    payload = {
        "model": "m",
        "temperature": 1.0,
        "n": n,
        "return_token_ids": return_ids,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AA=="},
                    },
                ],
            }
        ],
    }

    async def post():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api_server.app), base_url="http://test"
        ) as client:
            return await client.post("/v1/chat/completions", json=payload)

    response = asyncio.run(post())
    assert response.status_code == 200
    body = response.json()
    assert body["prompt_token_ids"] == (PROMPT_IDS if return_ids else None)
    assert body["usage"]["prompt_tokens"] == len(PROMPT_IDS)
    assert len(body["choices"]) == n
    assert calls == ["single" if n == 1 else "fanout"] and server.templates == 0


@pytest.mark.parametrize("endpoint", ["chat", "completion"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "cap",
    [
        {},
        {"max_tokens": None},
        {"max_tokens": 512},
        {"max_completion_tokens": 512},
        {"max_tokens": 8192},
    ],
)
def test_output_cap_default_presence_reaches_sampling(
    server, monkeypatch, endpoint, stream, cap
):
    from fastapi import HTTPException

    monkeypatch.setattr(
        api_server, "_get_engine_config", lambda: SimpleNamespace(max_model_len=4096)
    )
    original = api_server._build_sampling_params
    seen = []

    def capture(**kwargs):
        result = original(**kwargs)
        seen.append(result)
        return result

    monkeypatch.setattr(api_server, "_build_sampling_params", capture)
    req, handler = _request(endpoint, stream=stream, **cap)
    if endpoint == "completion":
        req.prompt = "hi"

    # Stop after sampling construction; generation is covered by this module's
    # endpoint tests and the context-budget tests exercise engine prompt lengths.
    def stop(*args, **kwargs):
        raise RuntimeError("sampling captured")

    monkeypatch.setattr(api_server, "apply_chat_template", stop)
    monkeypatch.setattr(type(req), "get_prompt_or_tokens", stop, raising=False)
    if cap.get("max_tokens") == 8192:
        with pytest.raises(HTTPException) as exc:
            asyncio.run(handler(req, SimpleNamespace()))
        assert exc.value.status_code == 400
        return
    try:
        asyncio.run(handler(req, SimpleNamespace()))
    except (RuntimeError, HTTPException):
        pass
    assert len(seen) == 1
    explicit = any(value is not None for value in cap.values())
    assert seen[0].max_tokens_is_default == (not explicit)
    assert seen[0].max_tokens == (512 if explicit else 4096)


IMAGE_MESSAGES = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the image."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
        ],
    }
]
# Before the `server` fixture stubs it: the PD tests run the real generator.
_REAL_GENERATE_ASYNC = api_server.generate_async


@pytest.fixture
def pd_engine(monkeypatch, server):
    """Real Sequences and the real generators over an engine that hands
    every prefill off at once."""
    media = {
        "pixel_values": np.arange(24, dtype=np.float32).reshape(4, 6),
        "image_grid_thw": np.array([[1, 2, 2]]),
    }
    engine = SimpleNamespace(seqs=[], prepared=[])
    handoff = {
        "do_remote_prefill": True,
        "do_remote_decode": False,
        "remote_engine_id": "prefill-0",
        "remote_block_ids": [3, 4],
    }

    def preprocess(tokens, sampling, stream_callback=None, **kwargs):
        multimodal_data = kwargs.get("multimodal_data")
        seq = Sequence(
            list(tokens),
            16,
            sampling,
            kv_transfer_params=kwargs.get("kv_transfer_params"),
            multimodal_data=multimodal_data,
            # As an MRoPE model's `get_mrope_input_positions` returns it.
            mrope_position_delta=-2 if multimodal_data is not None else 0,
        )
        engine.seqs.append(seq)
        produces = (seq.kv_transfer_params or {}).get("do_remote_decode")
        stream_callback(
            SimpleNamespace(
                output_tokens=[5],
                logprobs=None,
                finished=True,
                finish_reason="stop",
                num_cached_tokens=0,
                kv_transfer_params_output=dict(handoff) if produces else None,
            )
        )
        return seq

    def prepare(*_args):
        engine.prepared.append(_args)
        return list(PROMPT_IDS), dict(media)

    engine.io_processor = SimpleNamespace(preprocess=preprocess, requests={})
    engine.core_mgr = SimpleNamespace(add_request=lambda _seqs: None)
    monkeypatch.setattr(api_server, "engine", engine)
    monkeypatch.setattr(api_server, "generate_async", _REAL_GENERATE_ASYNC)
    monkeypatch.setattr(api_server, "_get_multimodal_processor", lambda: object())
    monkeypatch.setattr(api_server, "_prepare_multimodal_inputs", prepare)
    monkeypatch.setattr(
        api_server, "_validate_sequence_context_length", lambda _s, _sp=None: None
    )
    monkeypatch.setattr(api_server, "record_nonstream_first_token", lambda: None)
    engine.media_seed = multimodal_cache_seed(media)
    return engine


def _post_chat(payloads):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api_server.app), base_url="http://test"
        ) as client:
            return [
                await client.post("/v1/chat/completions", json=payload)
                for payload in payloads
            ]

    return asyncio.run(run())


class TestMultimodalPD:
    def test_decode_reuses_prefill_ids_and_cache_seed(self, pd_engine):
        prefill_body = {
            "model": "m",
            "messages": IMAGE_MESSAGES,
            "max_tokens": 1,
            "return_token_ids": True,
            "kv_transfer_params": {
                "do_remote_decode": True,
                "do_remote_prefill": False,
            },
        }
        (prefill,) = _post_chat([prefill_body])
        assert prefill.status_code == 200, prefill.text
        kv = prefill.json()["kv_transfer_params"]
        assert kv["cache_seed"] == pd_engine.media_seed
        assert kv["mrope_position_delta"] == -2

        # As ATOMesh's relay builds D's body: P's metadata plus its ids.
        kv["prompt_token_ids"] = prefill.json()["prompt_token_ids"]
        decode_body = {
            "model": "m",
            "messages": IMAGE_MESSAGES,
            "kv_transfer_params": kv,
        }
        (decode,) = _post_chat([decode_body])
        assert decode.status_code == 200, decode.text

        p_seq, d_seq = pd_engine.seqs
        assert len(pd_engine.prepared) == 1  # P only
        assert p_seq.kv_transfer_params["do_remote_decode"] is True
        assert d_seq.multimodal_data is None
        assert list(d_seq.token_ids) == PROMPT_IDS
        assert d_seq.kv_transfer_params["remote_block_ids"] == [3, 4]
        assert d_seq.cache_seed == p_seq.cache_seed == pd_engine.media_seed
        assert d_seq.mrope_position_delta == p_seq.mrope_position_delta == -2
        assert d_seq.media_on_prefill_instance
        assert not p_seq.media_on_prefill_instance

    @pytest.mark.parametrize("with_cache_seed", [True, False])
    def test_b64_decode_leg_takes_prefill_expanded_ids(
        self, pd_engine, monkeypatch, with_cache_seed
    ):
        # ATOM_PROMPT_IDS_B64: the client body's b64 ids have the image
        # placeholders not expanded, and the router forwards that body to D
        # unchanged; P's expanded ids reach D in kv_transfer_params.
        monkeypatch.setenv("ATOM_PROMPT_IDS_B64", "1")
        unexpanded = [101, 7]
        assert unexpanded != PROMPT_IDS
        client_body = {
            "model": "m",
            "messages": IMAGE_MESSAGES,
            "prompt_token_ids_b64": base64.b64encode(
                array.array("I", unexpanded).tobytes()
            ).decode("ascii"),
        }
        (prefill,) = _post_chat(
            [
                {
                    **client_body,
                    "max_tokens": 1,
                    "return_token_ids": True,
                    "kv_transfer_params": {
                        "do_remote_decode": True,
                        "do_remote_prefill": False,
                    },
                }
            ]
        )
        assert prefill.status_code == 200, prefill.text
        # A media prefill still echoes its expanded ids.
        assert prefill.json()["prompt_token_ids"] == PROMPT_IDS
        kv = prefill.json()["kv_transfer_params"]
        if not with_cache_seed:
            kv.pop("cache_seed")
        kv["prompt_token_ids"] = prefill.json()["prompt_token_ids"]

        (decode,) = _post_chat([{**client_body, "kv_transfer_params": kv}])
        assert decode.status_code == 200, decode.text
        _p_seq, d_seq = pd_engine.seqs
        assert list(d_seq.token_ids) == PROMPT_IDS
        assert d_seq.media_on_prefill_instance == with_cache_seed
        # P preprocessed the media; D only when P handed off no cache seed.
        assert len(pd_engine.prepared) == (1 if with_cache_seed else 2)

    def test_decode_without_cache_seed_preprocesses_the_media(self, pd_engine):
        # As from a prefill instance that hands off no cache seed.
        kv = {"do_remote_prefill": True, "prompt_token_ids": PROMPT_IDS}
        (decode,) = _post_chat(
            [{"model": "m", "messages": IMAGE_MESSAGES, "kv_transfer_params": kv}]
        )
        assert decode.status_code == 200, decode.text
        (seq,) = pd_engine.seqs
        assert len(pd_engine.prepared) == 1 and seq.multimodal_data is not None
        assert seq.cache_seed == pd_engine.media_seed
        assert seq.mrope_position_delta == -2
        assert not seq.media_on_prefill_instance

    @pytest.mark.parametrize(
        "key,value",
        [
            ("cache_seed", "9"),
            ("cache_seed", 9.0),
            ("cache_seed", True),
            ("cache_seed", None),
            ("mrope_position_delta", -2.5),
            ("mrope_position_delta", "-2"),
        ],
    )
    def test_non_integer_handoff_key_is_400_before_enqueue(self, pd_engine, key, value):
        kv = {"do_remote_prefill": True, "prompt_token_ids": PROMPT_IDS, key: value}
        (decode,) = _post_chat(
            [{"model": "m", "messages": IMAGE_MESSAGES, "kv_transfer_params": kv}]
        )
        assert decode.status_code == 400, decode.text
        assert key in decode.text
        assert pd_engine.seqs == [] and pd_engine.prepared == []

    def test_prefill_without_handoff_returns_no_kv_params(self, pd_engine):
        (response,) = _post_chat([{"model": "m", "messages": IMAGE_MESSAGES}])
        assert response.status_code == 200, response.text
        assert response.json()["kv_transfer_params"] is None
        assert len(pd_engine.prepared) == 1

    def test_ids_without_remote_prefill_still_preprocess_media(self, pd_engine):
        (response,) = _post_chat(
            [
                {
                    "model": "m",
                    "messages": IMAGE_MESSAGES,
                    "prompt_token_ids": PROMPT_IDS,
                }
            ]
        )
        assert response.status_code == 200, response.text
        (seq,) = pd_engine.seqs
        assert len(pd_engine.prepared) == 1 and seq.multimodal_data is not None


IMAGE_DECODE_KV = {
    "do_remote_prefill": True,
    "cache_seed": 9,
    "mrope_position_delta": -2,
}


def _pd_connector(scheduler, **methods):
    """A P/D consumer's connector, answering the scheduler with `methods`."""
    scheduler.kv_connector = _ConnectorNamespace(
        is_producer=False, is_offload=False, **methods
    )


def _received_image_decode(scheduler, seq_factory, **connector):
    """An image decode leg whose KV has arrived: allocated, parked for the
    receive and reported finished, as a P/D consumer's connector leaves it."""
    _pd_connector(scheduler, **connector)
    seq = seq_factory(list(PROMPT_IDS), kv_transfer_params=dict(IMAGE_DECODE_KV))
    scheduler.add(seq)
    bm = scheduler.block_manager
    assert bm.allocate(seq, bm.can_allocate(seq))
    seq.status = SequenceStatus.WAITING_FOR_REMOTE_KVS
    scheduler._count_inflight_load(seq)
    scheduler.finished_recving_kv_req_ids.append(seq.id)
    return seq


def _image_decode_on_way_in(way_in, scheduler, seq_factory):
    """An image decode leg about to be admitted along `way_in`."""
    if way_in == "preempted":
        # Its receive done and decoding, then preempted; the connector has
        # nothing left to load for it.
        seq = _received_image_decode(
            scheduler,
            seq_factory,
            get_num_new_matched_tokens=lambda _seq: (0, False),
        )
        scheduler.schedule()
        seq.append_token(5)
        scheduler.running.remove(seq)
        assert scheduler.preempt(seq)
        return seq
    if way_in == "failed_receive":
        # Its receive failed; the connector cleared `do_remote_prefill` when
        # it queued that receive, so it has nothing to load again.
        seq = _received_image_decode(
            scheduler,
            seq_factory,
            get_num_new_matched_tokens=lambda _seq: (0, False),
        )
        scheduler.finished_recving_kv_req_ids.remove(seq.id)
        scheduler.failed_recving_kv_req_ids.append(seq.id)
        return seq
    seq = seq_factory(list(PROMPT_IDS), kv_transfer_params=dict(IMAGE_DECODE_KV))
    if way_in == "requeued":
        # Taken back after its receive was queued, which ends the connector's
        # remote-prefill answer: its next query finds nothing to load.
        _pd_connector(scheduler, get_num_new_matched_tokens=lambda _seq: (0, False))
    # The rest reach an instance whose connector is an offload tier alone
    # (a P/D consumer beside a tier does not start, `validate_multi_config`);
    # nothing stops a client from sending it a decode leg.
    elif way_in == "tier_declined_after_alloc":
        scheduler.kv_connector = _ConnectorNamespace(
            is_offload=True,
            get_num_new_matched_tokens=lambda _seq: (4, True),
            update_state_after_alloc=lambda _seq: None,
            should_park_for_load_after_alloc=lambda _seq: False,
        )
    elif way_in == "tier_resume":
        bm = scheduler.block_manager
        scheduler.kv_connector = _ConnectorNamespace(is_offload=True)
        assert bm.allocate(seq, bm.can_allocate(seq))
        seq.num_cached_tokens = 2
        seq.offload_loaded = True
    scheduler.add(seq)
    if way_in == "tier_early_loaded":
        # The tier loads its prefix ahead of admission; it lands.
        scheduler.kv_connector = _ConnectorNamespace(
            is_offload=True,
            supports_early_load=True,
            lookup_ready=lambda _seq: True,
            get_num_new_matched_tokens=lambda _seq: (len(PROMPT_IDS), True),
            should_park_for_load_after_alloc=lambda _seq: True,
        )
        scheduler.config.offload_min_load_tokens = 0
        scheduler.schedule()
        assert seq.id in scheduler._early_load.entries
        scheduler.finished_recving_kv_req_ids.append(seq.id)
    return seq


@pytest.mark.parametrize(
    "way_in",
    [
        "no_connector",
        "failed_receive",
        "requeued",
        "preempted",
        "tier_declined_after_alloc",
        "tier_resume",
        "tier_early_loaded",
    ],
)
def test_media_dropped_leg_is_never_prefilled_locally(scheduler, seq_factory, way_in):
    seq = _image_decode_on_way_in(way_in, scheduler, seq_factory)
    assert seq.media_on_prefill_instance

    scheduled = scheduler.schedule()

    assert scheduled is None or seq.id not in scheduled[1]
    assert seq.status == SequenceStatus.FINISHED
    assert seq.leave_reason.startswith("unschedulable:")
    assert scheduler.take_rejected() == [seq]
    assert seq not in scheduler.waiting and seq not in scheduler.running


def test_fresh_image_leg_waits_for_remote_kv(scheduler, seq_factory):
    _pd_connector(
        scheduler, get_num_new_matched_tokens=lambda _seq: (len(PROMPT_IDS), True)
    )
    seq = seq_factory(list(PROMPT_IDS), kv_transfer_params=dict(IMAGE_DECODE_KV))
    scheduler.add(seq)

    scheduler.schedule()

    assert seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert scheduler.take_rejected() == []


def test_a_leg_carrying_its_media_is_an_image_request(seq_factory):
    seq = seq_factory(
        list(PROMPT_IDS),
        kv_transfer_params=dict(IMAGE_DECODE_KV),
        multimodal_data={"cache_seed": 5},
        mrope_position_delta=-1,
    )
    assert not seq.media_on_prefill_instance
    assert (seq.cache_seed, seq.mrope_position_delta) == (5, -1)


@pytest.mark.parametrize(
    "kv",
    [
        {"cache_seed": 9},
        {"mrope_position_delta": 7},
        {"do_remote_prefill": True, "mrope_position_delta": 7},
    ],
)
def test_text_request_with_a_handoff_key_prefills_as_text(scheduler, seq_factory, kv):
    seq = seq_factory(list(PROMPT_IDS), kv_transfer_params=kv)
    assert not seq.media_on_prefill_instance
    assert (seq.cache_seed, seq.mrope_position_delta) == (-1, 0)
    scheduler.add(seq)

    batch, scheduled = scheduler.schedule()

    assert seq.id in scheduled and seq.status == SequenceStatus.RUNNING
    assert batch.mrope_position_deltas == {}


def _decode_batch_of_handed_off_seq(scheduler, seq_factory):
    """Run a received image decode leg past its first step; return its
    decode batch."""
    seq = _received_image_decode(scheduler, seq_factory)
    assert seq.mrope_positions is None and seq.mrope_position_delta == -2
    scheduler.schedule()
    seq.append_token(5)
    batch, _ = scheduler.schedule()
    return seq, batch


def test_handed_off_mrope_delta_reaches_the_batch(scheduler, seq_factory):
    seq, batch = _decode_batch_of_handed_off_seq(scheduler, seq_factory)
    assert batch.mrope_position_deltas == {seq.id: -2}


# numpy-only, loaded by path: no AITER or GPU stack.
_DECODE_LAYOUT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "atom/model_ops/attentions/token_layout/decode.py"
)
_spec = importlib.util.spec_from_file_location("_decode_layout", _DECODE_LAYOUT)
decode_layout = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(decode_layout)


def test_handed_off_mrope_delta_offsets_decode_positions(scheduler, seq_factory):
    _, batch = _decode_batch_of_handed_off_seq(scheduler, seq_factory)
    planes = np.full((3, 1), -1, dtype=np.int64)
    decode_layout.mrope_decode_positions(
        batch.req_ids,
        np.asarray(batch.context_lens),
        batch.num_scheduled_tokens,
        batch.mrope_position_deltas,
        planes,
    )
    # The sampled token sits at index 4; P's delta moves it to position 2.
    assert planes.tolist() == [[2], [2], [2]]


def test_text_decode_positions_are_their_indices():
    planes = np.full((3, 3), -1, dtype=np.int64)
    decode_layout.mrope_decode_positions([1, 2], [5, 9], [1, 2], {}, planes)
    assert planes.tolist() == [[4, 7, 8]] * 3


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("n", [1, 2])
def test_image_decode_takes_pretokenized_branch(monkeypatch, server, stream, n):
    async def fake_fanout(prompt, _sampling, _rid, **kwargs):
        server.generate.append((prompt, kwargs))
        output = {"text": "hi", "finish_reason": "eos", "num_tokens_input": 4}
        return [{**output, "num_tokens_output": 1} for _ in range(n)]

    async def fake_setup(prompt, _sampling, _rid, **kwargs):
        server.generate.append((prompt, kwargs))
        return ([7, 8] if n > 1 else 7), object(), len(PROMPT_IDS)

    def refuse(*_args):
        raise AssertionError("decode preprocessed the image")

    monkeypatch.setattr(api_server, "generate_async_fanout", fake_fanout)
    monkeypatch.setattr(api_server, "setup_streaming_request", fake_setup)
    monkeypatch.setattr(api_server, "setup_streaming_request_fanout", fake_setup)
    monkeypatch.setattr(api_server, "_prepare_multimodal_inputs", refuse)
    kv = {"do_remote_prefill": True, "cache_seed": 9, "prompt_token_ids": PROMPT_IDS}
    request = ChatCompletionRequest(
        model="m",
        messages=IMAGE_MESSAGES,
        temperature=1.0,
        stream=stream,
        n=n,
        kv_transfer_params=kv,
    )

    asyncio.run(api_server.chat_completions(request, None))

    ((sent, options),) = server.generate
    assert sent == PROMPT_IDS and options.get("multimodal_data") is None
    assert options["kv_transfer_params"] == {"do_remote_prefill": True, "cache_seed": 9}


MAX_MODEL_LEN = 4096
OUT_OF_RANGE_HANDOFF = [
    ("cache_seed", -1),
    ("cache_seed", -2),
    ("cache_seed", 2**64),
    ("cache_seed", "9"),
    ("cache_seed", None),
    ("cache_seed", True),
    ("mrope_position_delta", MAX_MODEL_LEN + 1),
    ("mrope_position_delta", -MAX_MODEL_LEN - 1),
    ("mrope_position_delta", 2**63),
    ("mrope_position_delta", -2.5),
    ("mrope_position_delta", True),
]


@pytest.fixture
def d_engine(monkeypatch):
    monkeypatch.setattr(
        api_server,
        "engine",
        SimpleNamespace(
            config=SimpleNamespace(
                max_model_len=MAX_MODEL_LEN,
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
            )
        ),
    )


@pytest.mark.parametrize(
    "endpoint", ["chat/completions", "completions", "messages", "responses"]
)
@pytest.mark.parametrize("key,value", OUT_OF_RANGE_HANDOFF)
def test_out_of_range_handoff_key_is_400_on_every_endpoint(
    monkeypatch, server, d_engine, endpoint, key, value
):
    async def unexpected_enqueue(*_args, **_kwargs):
        pytest.fail("an out-of-range handoff key must be refused before enqueue")

    monkeypatch.setattr(api_server, "setup_streaming_request", unexpected_enqueue)
    body = {
        "model": "m",
        "max_tokens": 2,
        "kv_transfer_params": {
            "do_remote_prefill": True,
            "cache_seed": 9,
            "prompt_token_ids": PROMPT_IDS,
            key: value,
        },
    }
    if endpoint == "completions":
        body["prompt"] = "hi"
    elif endpoint == "responses":
        body.pop("max_tokens")
        body["input"] = "hi"
    else:
        body["messages"] = [{"role": "user", "content": "hi"}]

    async def post():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api_server.app), base_url="http://d"
        ) as client:
            return await client.post(f"/v1/{endpoint}", json=body)

    response = asyncio.run(asyncio.wait_for(post(), 5))
    assert response.status_code == 400, response.text
    assert key in response.text
    assert not server.generate and not server.templates


@pytest.mark.parametrize(
    "handoff",
    [
        {"cache_seed": 0, "mrope_position_delta": MAX_MODEL_LEN},
        {"cache_seed": 2**64 - 1, "mrope_position_delta": -MAX_MODEL_LEN},
    ],
)
def test_handoff_keys_at_their_bounds_are_accepted(d_engine, handoff):
    api_server._validate_kv_transfer_params({"do_remote_prefill": True, **handoff})


def _intake(hf_config=None, *, num_draft_tokens=0, rapidserve=False):
    """The engine's request intake, as the server's executor runs it."""
    from atom.model_engine.llm_engine import InputOutputProcessor

    processor = InputOutputProcessor.__new__(InputOutputProcessor)
    processor.__dict__.update(
        config=SimpleNamespace(
            hf_config=hf_config or SimpleNamespace(),
            enable_dp_attention=False,
            enable_rapidserve=rapidserve,
        ),
        num_speculative_tokens=num_draft_tokens,
        block_size=16,
        has_per_req_cache=False,
        requests={},
        _external_to_internal={},
        _internal_to_external={},
        prefix_hash_block_size=0,
    )
    return processor


def _decode_leg(processor, delta=-2):
    kv = dict(IMAGE_DECODE_KV, mrope_position_delta=delta)
    return processor.preprocess(
        list(PROMPT_IDS), SamplingParams(), kv_transfer_params=kv
    )


@pytest.mark.parametrize("delta", [1 - len(PROMPT_IDS), 0])
def test_a_handed_off_delta_within_the_prompt_is_admitted(delta):
    seq = _decode_leg(_intake(), delta)
    assert seq.media_on_prefill_instance and seq.mrope_position_delta == delta


@pytest.mark.parametrize(
    "delta", [-len(PROMPT_IDS), 1, MAX_MODEL_LEN - len(PROMPT_IDS)]
)
def test_a_handed_off_delta_past_the_prompt_is_refused(delta):
    processor = _intake()
    with pytest.raises(ValueError, match="mrope_position_delta"):
        _decode_leg(processor, delta)
    assert processor.requests == {}


def test_rapidserve_refuses_a_media_dropped_leg():
    with pytest.raises(ValueError, match="RapidServe"):
        _decode_leg(_intake(rapidserve=True))


@pytest.mark.parametrize("num_draft_tokens", [0, 5])
def test_dspark_refuses_a_media_dropped_leg_as_an_image_request(num_draft_tokens):
    from atom.models.deepseek_v41.config import DeepseekV41TextConfig

    processor = _intake(DeepseekV41TextConfig(), num_draft_tokens=num_draft_tokens)
    if num_draft_tokens:
        with pytest.raises(ValueError, match="DSpark currently supports text"):
            _decode_leg(processor)
    else:
        assert _decode_leg(processor).media_on_prefill_instance

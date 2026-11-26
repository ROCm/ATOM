# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""``prompt_token_ids_b64`` (ATOM_PROMPT_IDS_B64): the P/D decode-ingest fast path."""

import array
import asyncio
import base64
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from atom.entrypoints.openai import api_server
from atom.entrypoints.openai.protocol import (
    PROMPT_TOKEN_IDS_B64,
    ChatCompletionRequest,
    decode_prompt_token_ids_b64,
)
from atom.entrypoints.openai.reasoning import prompt_tokens_start_in_reasoning
from atom.model_engine.sequence import Sequence

PROMPT_IDS = [0, 101, 102, 129279, 2**31 - 1]
MESSAGES = [{"role": "user", "content": "hi"}]


def encode(ids):
    """The exact-tokenizer front's `prompt_ids_codec.encode_ids`."""
    a = array.array("I", ids)
    if a.itemsize != 4:  # pragma: no cover - every supported host
        pytest.skip("array('I') is not 32-bit here")
    return base64.b64encode(a.tobytes()).decode("ascii")


@pytest.fixture
def b64_on(monkeypatch):
    monkeypatch.setenv("ATOM_PROMPT_IDS_B64", "1")


@pytest.fixture
def b64_off(monkeypatch):
    monkeypatch.setenv("ATOM_PROMPT_IDS_B64", "0")


class TestDecode:
    def test_round_trip_is_an_int32_array(self):
        ids = decode_prompt_token_ids_b64(encode(PROMPT_IDS))
        assert isinstance(ids, array.array) and ids.typecode == "i"
        assert ids.tolist() == PROMPT_IDS

    @pytest.mark.parametrize(
        "text",
        [
            "",  # empty
            "!!!!",  # not base64
            base64.b64encode(b"\x01\x02\x03").decode(),  # ragged
            encode([2**31]),  # past int32
            encode([2**32 - 1]),
        ],
    )
    def test_malformed_is_a_value_error(self, text):
        with pytest.raises(ValueError):
            decode_prompt_token_ids_b64(text)

    def test_non_string_is_a_value_error(self):
        with pytest.raises(ValueError):
            decode_prompt_token_ids_b64([1, 2])


class TestRequestModel:
    def test_default_is_on(self, monkeypatch):
        monkeypatch.delenv("ATOM_PROMPT_IDS_B64", raising=False)
        req = ChatCompletionRequest.model_validate(
            {"messages": MESSAGES, PROMPT_TOKEN_IDS_B64: encode(PROMPT_IDS)}
        )
        assert req.prompt_ids_from_b64()
        assert req.get_prompt_token_ids().tolist() == PROMPT_IDS

    def test_default_without_the_field_is_unchanged(self, monkeypatch):
        monkeypatch.delenv("ATOM_PROMPT_IDS_B64", raising=False)
        req = ChatCompletionRequest.model_validate(
            {"messages": MESSAGES, "prompt_token_ids": PROMPT_IDS}
        )
        assert not req.prompt_ids_from_b64()
        assert req.get_prompt_token_ids() == PROMPT_IDS
        bare = ChatCompletionRequest.model_validate({"messages": MESSAGES})
        assert bare.get_prompt_token_ids() is None

    def test_off_ignores_the_field_as_before(self, b64_off):
        req = ChatCompletionRequest.model_validate(
            {"messages": MESSAGES, PROMPT_TOKEN_IDS_B64: encode(PROMPT_IDS)}
        )
        assert req.get_prompt_token_ids() is None
        assert not req.prompt_ids_from_b64()
        assert PROMPT_TOKEN_IDS_B64 not in req.model_dump()

    @pytest.mark.parametrize("junk", ["not even base64", 7, [1, 2], {"a": 1}])
    def test_off_keeps_a_json_list_and_refuses_nothing(self, b64_off, junk):
        req = ChatCompletionRequest.model_validate(
            {
                "messages": MESSAGES,
                "prompt_token_ids": PROMPT_IDS,
                PROMPT_TOKEN_IDS_B64: junk,
            }
        )
        assert req.get_prompt_token_ids() == PROMPT_IDS

    def test_on_non_string_is_a_validation_error(self, b64_on):
        with pytest.raises(ValidationError, match="base64 string"):
            ChatCompletionRequest.model_validate(
                {"messages": MESSAGES, PROMPT_TOKEN_IDS_B64: [1, 2]}
            )

    def test_on_decodes_and_drops_the_string(self, b64_on):
        req = ChatCompletionRequest.model_validate(
            {"messages": MESSAGES, PROMPT_TOKEN_IDS_B64: encode(PROMPT_IDS)}
        )
        ids = req.get_prompt_token_ids()
        assert isinstance(ids, array.array) and ids.tolist() == PROMPT_IDS
        assert req.prompt_ids_from_b64()
        assert req.prompt_token_ids is None and req.prompt_token_ids_b64 is None
        dumped = req.model_dump()
        assert PROMPT_TOKEN_IDS_B64 not in dumped
        assert dumped["prompt_token_ids"] is None

    def test_on_from_json_bytes(self, b64_on):
        import json

        body = json.dumps(
            {"messages": MESSAGES, PROMPT_TOKEN_IDS_B64: encode(PROMPT_IDS)}
        )
        req = ChatCompletionRequest.model_validate_json(body)
        assert req.get_prompt_token_ids().tolist() == PROMPT_IDS

    def test_on_agreeing_list_is_accepted(self, b64_on):
        req = ChatCompletionRequest.model_validate(
            {
                "messages": MESSAGES,
                "prompt_token_ids": PROMPT_IDS,
                PROMPT_TOKEN_IDS_B64: encode(PROMPT_IDS),
            }
        )
        assert req.get_prompt_token_ids().tolist() == PROMPT_IDS

    def test_on_disagreeing_list_is_rejected(self, b64_on):
        with pytest.raises(ValidationError, match="disagree"):
            ChatCompletionRequest.model_validate(
                {
                    "messages": MESSAGES,
                    "prompt_token_ids": PROMPT_IDS[:-1],
                    PROMPT_TOKEN_IDS_B64: encode(PROMPT_IDS),
                }
            )

    def test_on_malformed_is_a_validation_error(self, b64_on):
        with pytest.raises(ValidationError, match="base64"):
            ChatCompletionRequest.model_validate(
                {"messages": MESSAGES, PROMPT_TOKEN_IDS_B64: "@@@@"}
            )

    def test_on_kv_carried_ids_are_checked_against_b64(self, b64_on):
        # An unpatched P still echoes the ids; the router carries them to D.
        ok = ChatCompletionRequest.model_validate(
            {
                "messages": MESSAGES,
                PROMPT_TOKEN_IDS_B64: encode(PROMPT_IDS),
                "kv_transfer_params": {"prompt_token_ids": list(PROMPT_IDS)},
            }
        )
        assert ok.get_prompt_token_ids().tolist() == PROMPT_IDS
        bad = ChatCompletionRequest.model_validate(
            {
                "messages": MESSAGES,
                PROMPT_TOKEN_IDS_B64: encode(PROMPT_IDS),
                "kv_transfer_params": {"prompt_token_ids": [7]},
            }
        )
        with pytest.raises(ValueError, match="disagree"):
            bad.get_prompt_token_ids()

    def test_media_decode_leg_takes_the_kv_ids_unchecked(self, b64_on):
        # The b64 ids are the client's, placeholders not expanded; P's
        # expanded ids in kv_transfer_params are the prompt.
        expanded = [101, 9, 9, 9, 102]
        req = ChatCompletionRequest.model_validate(
            {
                "messages": MESSAGES,
                PROMPT_TOKEN_IDS_B64: encode([101, 102]),
                "kv_transfer_params": {
                    "do_remote_prefill": True,
                    "prompt_token_ids": expanded,
                },
            }
        )
        assert req.get_prompt_token_ids(media_decode_leg=True) == expanded
        with pytest.raises(ValueError, match="disagree"):
            req.get_prompt_token_ids()


class TestEngineInputs:
    def test_sequence_copies_the_array(self):
        ids = decode_prompt_token_ids_b64(encode(PROMPT_IDS))
        seq = Sequence(ids, 16)
        assert seq.token_ids == ids and seq.token_ids is not ids
        assert seq.num_prompt_tokens == len(PROMPT_IDS)
        assert seq.last_token == PROMPT_IDS[-1]
        ids[0] = 5  # the request's buffer is not the sequence's
        assert seq.token_ids[0] == PROMPT_IDS[0]

    def test_reasoning_probe_hands_decode_a_list(self):
        seen = []

        def decode(ids):
            seen.append(ids)
            return "text"

        ids = decode_prompt_token_ids_b64(encode(PROMPT_IDS))
        prompt_tokens_start_in_reasoning(ids, decode)
        assert seen and all(type(s) is list for s in seen)


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
            "num_tokens_input": len(PROMPT_IDS),
            "num_tokens_output": 1,
        }
        if kwargs.get("return_token_ids"):
            output["prompt_token_ids"] = list(PROMPT_IDS)
        yield output

    async def fake_setup(prompt, _sampling, _rid, **kwargs):
        calls.generate.append((prompt, kwargs))
        raise RuntimeError("streaming setup reached")

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
    monkeypatch.setattr(api_server, "setup_streaming_request", fake_setup)
    return calls


def _post(payload):
    async def post():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api_server.app), base_url="http://p"
        ) as client:
            return await client.post("/v1/chat/completions", json=payload)

    return asyncio.run(post())


class TestHandler:
    def test_prefill_leg_skips_template_and_does_not_echo_ids(self, server, b64_on):
        # The router's P leg: stream off, return_token_ids on.
        response = _post(
            {
                "model": "m",
                "messages": MESSAGES,
                PROMPT_TOKEN_IDS_B64: encode(PROMPT_IDS),
                "return_token_ids": True,
                "stream": False,
                "max_tokens": 1,
                "kv_transfer_params": {"do_remote_decode": True},
            }
        )
        assert response.status_code == 200, response.text
        assert server.templates == 0
        sent, kwargs = server.generate[0]
        assert isinstance(sent, array.array) and sent.tolist() == PROMPT_IDS
        assert kwargs["return_token_ids"] is False
        assert response.json()["prompt_token_ids"] is None

    def test_off_renders_and_echoes_as_before(self, server, b64_off):
        response = _post(
            {
                "model": "m",
                "messages": MESSAGES,
                PROMPT_TOKEN_IDS_B64: encode(PROMPT_IDS),
                "return_token_ids": True,
            }
        )
        assert response.status_code == 200, response.text
        assert server.templates == 1
        assert server.generate[0][0] == "<rendered prompt>"
        assert server.generate[0][1]["return_token_ids"] is True
        assert response.json()["prompt_token_ids"] == PROMPT_IDS

    def test_list_ids_still_echo_with_the_flag_on(self, server, b64_on):
        response = _post(
            {
                "model": "m",
                "messages": MESSAGES,
                "prompt_token_ids": PROMPT_IDS,
                "return_token_ids": True,
            }
        )
        assert response.status_code == 200, response.text
        assert server.generate[0][1]["return_token_ids"] is True
        assert response.json()["prompt_token_ids"] == PROMPT_IDS

    def test_decode_leg_streams_from_the_array(self, server, b64_on):
        # The router's D leg: client body (b64) + kv params without ids.
        response = _post(
            {
                "model": "m",
                "messages": MESSAGES,
                PROMPT_TOKEN_IDS_B64: encode(PROMPT_IDS),
                "stream": True,
                "kv_transfer_params": {
                    "do_remote_prefill": True,
                    "remote_engine_id": "p0",
                },
            }
        )
        assert response.status_code == 500  # the stub stops at setup
        assert server.templates == 0
        sent, kwargs = server.generate[0]
        assert isinstance(sent, array.array) and sent.tolist() == PROMPT_IDS
        assert kwargs["kv_transfer_params"] == {
            "do_remote_prefill": True,
            "remote_engine_id": "p0",
        }

    def test_malformed_b64_is_a_client_error(self, server, b64_on):
        response = _post(
            {"model": "m", "messages": MESSAGES, PROMPT_TOKEN_IDS_B64: "@@@@"}
        )
        assert response.status_code == 422
        assert server.generate == [] and server.templates == 0

    def test_multimodal_prefill_still_echoes(self, server, b64_on, monkeypatch):
        # A multimodal prefill renders from messages; its echoed ids carry the
        # expanded image placeholders that the decode leg needs.
        async def fake_multimodal(prompt, mm_data, _sampling, _rid, **kwargs):
            server.generate.append((prompt, kwargs))
            yield {
                "text": "hello",
                "finish_reason": "eos",
                "num_tokens_input": len(PROMPT_IDS),
                "num_tokens_output": 1,
                "prompt_token_ids": list(PROMPT_IDS),
            }

        monkeypatch.setattr(api_server, "_has_multimodal_content", lambda _m: True)
        monkeypatch.setattr(api_server, "_get_multimodal_processor", lambda: None)
        monkeypatch.setattr(
            api_server,
            "_prepare_multimodal_inputs",
            lambda *_a, **_kw: ("<mm prompt>", {"image": 1}),
        )
        monkeypatch.setattr(api_server, "generate_async_multimodal", fake_multimodal)
        response = _post(
            {
                "model": "m",
                "messages": MESSAGES,
                PROMPT_TOKEN_IDS_B64: encode(PROMPT_IDS),
                "return_token_ids": True,
                "stream": False,
                "max_tokens": 1,
                "kv_transfer_params": {"do_remote_decode": True},
            }
        )
        assert response.status_code == 200, response.text
        assert server.generate[0][1]["return_token_ids"] is True

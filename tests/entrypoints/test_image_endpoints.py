# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Images on /v1/responses and /v1/messages: translated to chat image parts,
checked as chat's are, and run on the multimodal path. Where DeepSeek's
contract takes them: user turns (Responses' developer role too) and tool
results."""

import asyncio
import base64
import io
import json
import re
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from atom.entrypoints.openai import api_server
from atom.entrypoints.openai.images import MAX_IMAGES, NO_FILES_API
from atom.entrypoints.openai.serving_anthropic import (
    AnthropicMessage,
    anthropic_to_openai_messages,
)
from atom.entrypoints.openai.serving_responses import (
    ResponsesRequest,
    normalize_codex_request,
    responses_to_openai_messages,
)


def _png() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(buffer, "PNG")
    return base64.b64encode(buffer.getvalue()).decode()


PNG = _png()
DATA_URL = f"data:image/png;base64,{PNG}"
URL = "https://example.com/cat.png"
TOKENS = [5, 6, 7]
DATA = {"pixel_values": "px"}


def _responses_messages(body: dict) -> list[dict]:
    body, _context = normalize_codex_request({"model": "m", **body})
    return responses_to_openai_messages(ResponsesRequest.model_validate(body))


def _anthropic_messages(content, role="user", system=None) -> list[dict]:
    return anthropic_to_openai_messages(
        [AnthropicMessage(role=role, content=content)], system
    )


def _image(url=URL):
    return {"type": "image", "source": {"type": "url", "url": url}}


def _base64_image():
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": PNG},
    }


def _input_image(url=DATA_URL):
    return {"type": "input_image", "image_url": url}


class TestResponsesTranslation:
    def test_user_image_keeps_its_place_among_the_text(self):
        content = [
            {"type": "input_text", "text": "what is"},
            {"type": "input_image", "image_url": URL, "detail": "low"},
            {"type": "input_text", "text": "this?"},
        ]
        messages = _responses_messages(
            {"input": [{"role": "user", "content": content}]}
        )
        assert messages == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is"},
                    {"type": "image_url", "image_url": {"url": URL, "detail": "low"}},
                    {"type": "text", "text": "this?"},
                ],
            }
        ]

    def test_developer_image_is_a_user_image(self):
        item = {"type": "message", "role": "developer", "content": [_input_image()]}
        assert _responses_messages({"input": [item]}) == [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": DATA_URL}}],
            }
        ]

    @pytest.mark.parametrize(
        "item",
        [
            {"type": "function_call_output", "call_id": "c1"},
            {"type": "custom_tool_call_output", "call_id": "c1"},
            # A call that carries its own output.
            {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
        ],
    )
    def test_tool_output_image_reaches_the_tool_message(self, item):
        output = [
            {"type": "input_text", "text": "screenshot:"},
            {"type": "input_image", "image_url": URL},
        ]
        messages = _responses_messages({"input": [{**item, "output": output}]})
        assert messages[-1] == {
            "role": "tool",
            "tool_call_id": "c1",
            "content": [
                {"type": "text", "text": "screenshot:"},
                {"type": "image_url", "image_url": {"url": URL}},
            ],
        }

    def test_chat_shaped_image_url_is_read_too(self):
        image = {"type": "input_image", "image_url": {"url": URL, "detail": "high"}}
        messages = _responses_messages(
            {"input": [{"role": "user", "content": [image]}]}
        )
        assert messages[0]["content"] == [
            {"type": "image_url", "image_url": {"url": URL, "detail": "high"}}
        ]

    def test_file_id_is_refused(self):
        image = {"type": "input_image", "file_id": "file-1"}
        with pytest.raises(ValueError, match=re.escape(NO_FILES_API)):
            _responses_messages({"input": [{"role": "user", "content": [image]}]})

    def test_without_images_nothing_changes(self):
        output = [{"type": "input_text", "text": "ok"}]
        texts = [
            {"type": "input_text", "text": "a"},
            {"type": "input_text", "text": "b"},
        ]
        messages = _responses_messages(
            {
                "input": [
                    {"role": "user", "content": texts},
                    {"type": "function_call_output", "call_id": "c1", "output": output},
                ]
            }
        )
        assert messages[0] == {"role": "user", "content": "ab"}
        assert messages[1]["content"] == json.dumps(output)


class TestAnthropicTranslation:
    def test_base64_and_url_images_keep_their_place(self):
        messages = _anthropic_messages(
            [_base64_image(), {"type": "text", "text": "and"}, _image()]
        )
        assert messages == [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": DATA_URL}},
                    {"type": "text", "text": "and"},
                    {"type": "image_url", "image_url": {"url": URL}},
                ],
            }
        ]

    def test_tool_result_image_reaches_the_tool_message(self):
        result = {
            "type": "tool_result",
            "tool_use_id": "t1",
            "content": [{"type": "text", "text": "screenshot:"}, _image()],
        }
        assert _anthropic_messages([result]) == [
            {
                "role": "tool",
                "tool_call_id": "t1",
                "content": [
                    {"type": "text", "text": "screenshot:"},
                    {"type": "image_url", "image_url": {"url": URL}},
                ],
            }
        ]

    @pytest.mark.parametrize(
        ("source", "message"),
        [
            ({"type": "file", "file_id": "file_1"}, re.escape(NO_FILES_API)),
            ({"type": "base64", "data": PNG}, "media_type and data"),
            ({"type": "s3", "url": URL}, "base64 or url source"),
            (None, "base64 or url source"),
        ],
    )
    def test_unusable_source_is_refused(self, source, message):
        with pytest.raises(ValueError, match=message):
            _anthropic_messages([{"type": "image", "source": source}])

    @pytest.mark.parametrize(
        ("role", "system", "where"),
        [
            ("assistant", None, "assistant messages"),
            ("system", None, "system messages"),
            ("user", "system", "the system prompt"),
        ],
    )
    def test_images_outside_user_turns_are_refused(self, role, system, where):
        content, system = ("hi", [_image()]) if system else ([_image()], None)
        with pytest.raises(ValueError, match=f"not in {where}"):
            _anthropic_messages(content, role=role, system=system)

    def test_without_images_nothing_changes(self):
        texts = [{"type": "text", "text": "x"}, {"type": "text", "text": "y"}]
        messages = _anthropic_messages(
            [
                {"type": "text", "text": "a"},
                {"type": "tool_result", "tool_use_id": "t1", "content": texts},
                {"type": "text", "text": "b"},
            ]
        )
        assert messages == [
            {"role": "user", "content": "a\nb"},
            {"role": "tool", "tool_call_id": "t1", "content": "x\ny"},
        ]


class _FinishedCollector:
    async def get(self):
        return {"text": "", "token_ids": [], "finished": True, "finish_reason": "eos"}


@pytest.fixture
def server(monkeypatch):
    """Stub the engine and the model's processor; the images are checked,
    loaded and parsed for real, then handed over."""
    calls = SimpleNamespace(generate=[], multimodal=[], setup=[], prepared=[])
    output = {
        "text": "hello",
        "finish_reason": "eos",
        "num_tokens_input": len(TOKENS),
        "num_tokens_output": 1,
    }

    async def fake_generate_async(prompt, _sampling, _request_id, **kwargs):
        calls.generate.append((prompt, kwargs))
        yield output

    async def fake_generate_async_multimodal(tokens, data, _sampling, _rid, **kw):
        calls.multimodal.append((tokens, data, kw))
        yield output

    async def fake_setup_streaming_request(prompt, _sampling, _request_id, **kwargs):
        calls.setup.append((prompt, kwargs))
        return 1, _FinishedCollector(), len(prompt)

    def fake_prepare(_config, _processor, conversation, media, _kwargs, tools=None):
        calls.prepared.append((conversation, media))
        return TOKENS, DATA

    monkeypatch.delenv("ATOM_IMAGE_URL_ALLOW_PRIVATE", raising=False)
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
            encode=lambda *_a, **_kw: [0, 0], decode=lambda ids, **_kw: "<x>"
        ),
    )
    monkeypatch.setattr(api_server, "apply_chat_template", lambda *_a, **_kw: "p")
    monkeypatch.setattr(
        api_server,
        "engine",
        SimpleNamespace(
            config=SimpleNamespace(
                generation_config=SimpleNamespace(), max_model_len=8192
            ),
            io_processor=SimpleNamespace(encode_prompt=lambda *_a, **_kw: [0, 0]),
        ),
    )
    monkeypatch.setattr(api_server, "_get_multimodal_processor", lambda: "processor")
    monkeypatch.setattr(api_server, "prepare_multimodal_inputs", fake_prepare)
    monkeypatch.setattr(api_server, "generate_async", fake_generate_async)
    monkeypatch.setattr(
        api_server, "generate_async_multimodal", fake_generate_async_multimodal
    )
    monkeypatch.setattr(
        api_server, "setup_streaming_request", fake_setup_streaming_request
    )
    monkeypatch.setattr(api_server, "cleanup_stream", lambda *_a, **_kw: None)
    return calls


def _post(path, body):
    """The whole response, a stream read to its end."""

    async def post():
        transport = httpx.ASGITransport(app=api_server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            return await client.post(path, json={"model": "m", **body})

    return asyncio.run(post())


def _responses(body):
    return _post("/v1/responses", body)


def _messages(content, **body):
    return _post(
        "/v1/messages", {"messages": [{"role": "user", "content": content}], **body}
    )


def _user_image(text="what?"):
    content = [{"type": "input_text", "text": text}, _input_image()]
    return [{"role": "user", "content": content}]


class TestResponsesEndpoint:
    def test_image_runs_on_the_multimodal_path(self, server):
        response = _responses({"input": _user_image()})

        assert response.status_code == 200, response.text
        assert [(t, d) for t, d, _kw in server.multimodal] == [(TOKENS, DATA)]
        assert server.generate == []
        conversation, media = server.prepared[0]
        assert conversation[-1]["content"] == [
            {"type": "text", "text": "what?"},
            {"type": "image"},
        ]
        assert [image.size for image in media["image"]] == [(2, 2)]

    def test_streamed_image_runs_on_the_multimodal_path(self, server):
        response = _responses({"input": _user_image(), "stream": True})

        assert response.status_code == 200, response.text
        assert "response.completed" in response.text
        prompt, kwargs = server.setup[0]
        assert prompt == TOKENS
        assert kwargs["multimodal_data"] == DATA

    @pytest.mark.parametrize("kind", ["function_call", "custom_tool_call"])
    def test_tool_output_image_is_loaded(self, server, kind):
        call = {"type": kind, "call_id": "c1", "name": "shot", "arguments": "{}"}
        result = {
            "type": f"{kind}_output",
            "call_id": "c1",
            "output": [_input_image()],
        }
        response = _responses({"input": [call, result]})

        assert response.status_code == 200, response.text
        conversation, media = server.prepared[0]
        assert conversation[-1]["role"] == "tool"
        assert conversation[-1]["content"] == [{"type": "image"}]
        assert len(media["image"]) == 1

    def test_structured_output_is_told_of_the_images(self, server, monkeypatch):
        seen = []

        async def configure(*_a, **kwargs):
            seen.append(kwargs["multimodal"])

        monkeypatch.setattr(api_server, "_configure_structured_output", configure)
        monkeypatch.setattr(
            api_server, "validate_responses_output", lambda *a, **kw: None
        )
        text = {"format": {"type": "json_object"}}
        assert _responses({"input": _user_image(), "text": text}).status_code == 200
        assert _responses({"input": "hi", "text": text}).status_code == 200
        assert seen == [True, False]

    @pytest.mark.parametrize(
        ("item", "message"),
        [
            (
                {"type": "message", "role": "assistant", "content": [_input_image()]},
                "not in assistant messages",
            ),
            (
                {"type": "message", "role": "system", "content": [_input_image()]},
                "not in system messages",
            ),
            (
                {
                    "role": "user",
                    "content": [{"type": "input_image", "file_id": "file-1"}],
                },
                NO_FILES_API,
            ),
            (
                {"role": "user", "content": [_input_image("/etc/passwd")]},
                "never a server path",
            ),
            (
                # Checked before anything is fetched.
                {"role": "user", "content": [_input_image()] * (MAX_IMAGES + 1)},
                f"at most {MAX_IMAGES} images",
            ),
        ],
    )
    def test_refusals_are_400(self, server, item, message):
        response = _responses({"input": [item]})

        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert message in error["message"]
        assert server.prepared == server.multimodal == server.generate == []

    def test_without_images_the_text_path_runs(self, server):
        assert _responses({"input": "hi"}).status_code == 200
        assert server.prepared == [] and len(server.generate) == 1


class TestMessagesEndpoint:
    def test_image_runs_on_the_multimodal_path(self, server):
        response = _messages([{"type": "text", "text": "what?"}, _base64_image()])

        assert response.status_code == 200, response.text
        assert [(t, d) for t, d, _kw in server.multimodal] == [(TOKENS, DATA)]
        assert server.generate == []
        conversation, media = server.prepared[0]
        assert conversation[-1]["content"] == [
            {"type": "text", "text": "what?"},
            {"type": "image"},
        ]
        assert [image.size for image in media["image"]] == [(2, 2)]

    @pytest.mark.parametrize("stream", [False, True])
    def test_tool_result_image_runs_on_the_multimodal_path(self, server, stream):
        result = {
            "type": "tool_result",
            "tool_use_id": "t1",
            "content": [_image(DATA_URL)],
        }
        response = _messages([result], stream=stream)

        assert response.status_code == 200, response.text
        if stream:
            assert "event: message_stop" in response.text
            prompt, kwargs = server.setup[0]
            assert prompt == TOKENS
            assert kwargs["multimodal_data"] == DATA
        else:
            assert [(t, d) for t, d, _kw in server.multimodal] == [(TOKENS, DATA)]
        conversation, _media = server.prepared[0]
        assert conversation[-1]["role"] == "tool"
        assert conversation[-1]["content"] == [{"type": "image"}]

    @pytest.mark.parametrize(
        ("content", "message"),
        [
            (
                [{"type": "image", "source": {"type": "file", "file_id": "file_1"}}],
                NO_FILES_API,
            ),
            # Checked before anything is fetched.
            ([_base64_image()] * (MAX_IMAGES + 1), f"at most {MAX_IMAGES} images"),
            # Fetched in the worker: still the client's error.
            ([_image("http://127.0.0.1/x.png")], "public addresses"),
            (
                [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "AAAA",
                        },
                    }
                ],
                "invalid image data",
            ),
        ],
    )
    def test_refusals_are_400(self, server, content, message):
        response = _messages(content)

        assert response.status_code == 400
        body = response.json()
        assert body["type"] == "error"
        assert body["error"]["type"] == "invalid_request_error"
        assert message in body["error"]["message"]
        assert server.prepared == server.multimodal == server.generate == []

    def test_without_images_the_text_path_runs(self, server):
        assert _messages("hi").status_code == 200
        assert server.prepared == [] and len(server.generate) == 1


@pytest.mark.parametrize("stream", [False, True])
def test_a_pd_handoff_with_images_is_refused(server, stream):
    """A prefill leg cannot hand an image request off: 400 on either leg,
    before anything is fetched."""
    handoff = {"kv_transfer_params": {"do_remote_decode": True}, "stream": stream}
    responses = _responses({"input": _user_image(), **handoff})
    messages = _messages([_base64_image()], **handoff)

    assert responses.status_code == messages.status_code == 400
    for message in (responses.json()["error"], messages.json()["error"]):
        assert "kv_transfer_params" in message["message"]
    assert server.prepared == server.multimodal == server.setup == []

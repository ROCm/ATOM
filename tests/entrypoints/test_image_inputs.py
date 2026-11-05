# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Image input across Chat, Responses and Anthropic Messages.

Every endpoint must deliver its images to the multimodal path or refuse the
request; none may answer as if the image were not there.
"""

import asyncio
import base64
import http.server
import io
import json
import socket
import threading
import urllib.parse
from types import SimpleNamespace

import pytest
from PIL import Image

from atom.entrypoints.openai import api_server, images
from atom.entrypoints.openai.protocol import ChatCompletionRequest
from atom.entrypoints.openai.serving_anthropic import (
    AnthropicMessage,
    anthropic_to_openai_messages,
)
from atom.entrypoints.openai.serving_responses import (
    ResponsesRequest,
    responses_to_openai_messages,
)


def encoded(fmt="PNG", size=(32, 16), color="red", mode="RGB"):
    buffer = io.BytesIO()
    Image.new(mode, size, color).save(buffer, fmt)
    return base64.b64encode(buffer.getvalue()).decode()


def data_url(fmt="PNG", **kwargs):
    return f"data:image/{fmt.lower()};base64,{encoded(fmt, **kwargs)}"


PNG = data_url()


# --- sources and parts -----------------------------------------------------------


@pytest.mark.parametrize("url", [PNG, "https://example.com/a.png", "http://h/x.jpg"])
def test_supported_sources_become_image_url_parts(url):
    assert images.image_url_part(url, "low") == {
        "type": "image_url",
        "image_url": {"url": url, "detail": "low"},
    }


@pytest.mark.parametrize(
    "url, detail",
    [
        # Never a path on the server.
        ("/etc/hostname", None),
        ("file:///etc/hostname", None),
        ("ftp://example.com/a.png", None),
        ("data:image/png,rawbytes", None),
        ("data:text/plain;base64,aGk=", None),
        ("https://example.com/" + "a" * images.MAX_URL_LENGTH, None),
        ("", None),
        (None, None),
        (PNG, "medium"),
    ],
)
def test_other_sources_are_refused(url, detail):
    with pytest.raises(ValueError):
        images.image_url_part(url, detail)


def test_chat_file_parts_carry_inline_images_and_refuse_file_ids():
    parts = images.chat_image_parts(
        [
            {"type": "text", "text": "a"},
            {"type": "file", "file": {"file_data": PNG, "filename": "x.png"}},
            {"type": "image_url", "image_url": {"url": PNG, "detail": "high"}},
        ]
    )
    assert [p["type"] for p in parts] == ["text", "image_url", "image_url"]
    assert parts[2]["image_url"]["detail"] == "high"
    with pytest.raises(ValueError, match="Files API"):
        images.chat_image_parts([{"type": "file", "file": {"file_id": "file-1"}}])


def test_chat_processor_style_image_parts_are_checked_as_image_urls():
    # They reached the media loader as they came: a server path, any URL.
    assert images.chat_image_parts([{"type": "image", "image": PNG}]) == [
        {"type": "image_url", "image_url": {"url": PNG}}
    ]
    for source in ("/etc/hostname", "file:///etc/hostname", None, {"url": PNG}):
        with pytest.raises(ValueError):
            images.chat_image_parts([{"type": "image", "image": source}])


def test_responses_content_keeps_order_and_is_none_without_images():
    assert images.responses_content([{"type": "input_text", "text": "a"}]) is None
    parts = images.responses_content(
        [
            {"type": "input_text", "text": "before"},
            {"type": "input_image", "image_url": PNG, "detail": "low"},
            {"type": "input_text", "text": "after"},
        ]
    )
    assert [p.get("text") or p["image_url"]["url"] for p in parts] == [
        "before",
        PNG,
        "after",
    ]
    with pytest.raises(ValueError, match="Files API"):
        images.responses_content([{"type": "input_image", "file_id": "file-1"}])


@pytest.mark.parametrize(
    "source, expected",
    [
        (
            {"type": "base64", "media_type": "image/png", "data": "QQ=="},
            "data:image/png;base64,QQ==",
        ),
        (
            {"type": "url", "url": "https://example.com/a.webp"},
            "https://example.com/a.webp",
        ),
    ],
)
def test_anthropic_sources(source, expected):
    part = images.anthropic_image_part({"type": "image", "source": source})
    assert part["image_url"]["url"] == expected


@pytest.mark.parametrize(
    "source",
    [
        {"type": "base64", "media_type": "image/bmp", "data": "QQ=="},
        {"type": "base64", "media_type": "image/png"},
        {"type": "file", "file_id": "file-1"},
        {"type": "path", "path": "/etc/hostname"},
        None,
    ],
)
def test_anthropic_bad_sources_are_refused(source):
    with pytest.raises(ValueError):
        images.anthropic_image_part({"type": "image", "source": source})


def test_images_belong_to_allowed_roles_only():
    image = [images.image_url_part(PNG)]
    images.check_image_roles([{"role": "user", "content": image}], frozenset({"user"}))
    for role in ("system", "assistant", "tool"):
        with pytest.raises(ValueError, match=f"{role} messages"):
            images.check_image_roles(
                [{"role": role, "content": image}], frozenset({"user"})
            )


# --- loading ---------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "GIF", "WEBP"])
def test_supported_formats_decode_by_content(fmt):
    # The declared type is ignored; the bytes decide.
    url = "data:image/png;base64," + encoded(fmt)
    [message] = images.load_images(
        [{"role": "user", "content": [images.image_url_part(url)]}]
    )
    image = message["content"][0]["image"]
    assert message["content"][0]["type"] == "image"
    assert image.mode == "RGB" and image.size == (32, 16)


def test_low_detail_is_downscaled_to_512():
    url = data_url(size=(2048, 1024))
    [message] = images.load_images(
        [{"role": "user", "content": [images.image_url_part(url, "low")]}]
    )
    assert message["content"][0]["image"].size == (512, 256)


@pytest.mark.parametrize(
    "url, message",
    [
        (data_url("BMP"), "unsupported image format BMP"),
        (data_url(size=(images.MAX_IMAGE_SIDE + 1, 1)), "px a side"),
        (
            "data:image/png;base64," + base64.b64encode(b"not an image").decode(),
            "invalid image",
        ),
        ("data:image/png;base64,***", "base64"),
        ("http://127.0.0.1:9/a.png", "could not fetch"),
    ],
)
def test_unusable_images_are_client_errors(url, message):
    with pytest.raises(ValueError, match=message):
        images.load_images([{"role": "user", "content": [images.image_url_part(url)]}])


def test_image_count_is_bounded(monkeypatch):
    monkeypatch.setattr(images, "MAX_IMAGES", 2)
    content = [images.image_url_part(PNG)] * 3
    with pytest.raises(ValueError, match="at most 2"):
        images.load_images([{"role": "user", "content": content}])


URL = "https://example.com/a.png"


@pytest.mark.parametrize(
    "translate, block",
    [
        (images.chat_image_parts, [{"type": "image_url"}]),
        (images.chat_image_parts, [{"type": "image_url", "image_url": {}}]),
        (images.chat_image_parts, [{"type": "file", "file": {}}]),
        (
            images.chat_image_parts,
            [{"type": "file", "file": {"file_data": PNG, "file_id": "f-1"}}],
        ),
        (images.responses_content, [{"type": "input_image"}]),
        (
            images.responses_content,
            [{"type": "input_image", "image_url": PNG, "file_id": "f-1"}],
        ),
        (images.anthropic_image_part, {"type": "image"}),
        (images.anthropic_image_part, {"type": "image", "source": {"type": "url"}}),
        (
            images.anthropic_image_part,
            {"type": "image", "source": {"type": "base64", "media_type": "image/png"}},
        ),
        (
            images.anthropic_image_part,
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": encoded(),
                    "url": URL,
                },
            },
        ),
        (
            images.anthropic_image_part,
            {"type": "image", "source": {"type": "url", "url": URL, "data": encoded()}},
        ),
    ],
)
def test_missing_and_conflicting_image_sources_are_refused(translate, block):
    with pytest.raises(ValueError):
        translate(block)


def test_video_reaches_the_media_path_and_is_refused_there():
    from atom.multimodal.processing import prepare_multimodal_inputs

    part = {"type": "video_url", "video_url": {"url": "https://example.com/v.mp4"}}
    assert images.chat_image_parts([part]) == [part]
    with pytest.raises(ValueError, match="supported modalities: image"):
        prepare_multimodal_inputs(None, None, [], {"video": [part["video_url"]["url"]]})


# --- URL fetching ---------------------------------------------------------------


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1",
        "localhost",
        "0.0.0.0",
        "10.1.2.3",
        "169.254.169.254",
        "[::1]",
        "[::ffff:192.168.0.1]",
    ],
)
def test_urls_to_internal_hosts_are_refused(host):
    with pytest.raises(ValueError, match="public addresses"):
        images._read(f"http://{host}:9/a.png")


def test_one_internal_answer_is_enough_to_refuse(monkeypatch):
    # Otherwise the socket connects to whichever answer comes first.
    answers = [("93.184.215.14", 80), ("10.0.0.1", 80)]
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", answer) for answer in answers
        ],
    )
    with pytest.raises(ValueError, match="public addresses"):
        images._read("http://images.example/a.png")


@pytest.mark.parametrize("allow_private", ["0", "1"])
def test_a_url_http_client_cannot_use_is_a_client_error(monkeypatch, allow_private):
    # http.client raises InvalidURL here, which is not an OSError.
    monkeypatch.setenv("ATOM_IMAGE_URL_ALLOW_PRIVATE", allow_private)
    monkeypatch.setenv("no_proxy", "*")
    with pytest.raises(ValueError, match="could not fetch image"):
        images._read("http://images.example:abc/a.png")


def test_a_reply_that_is_not_http_is_a_client_error(monkeypatch):
    # http.client raises BadStatusLine, which is not an OSError either.
    listener = socket.create_server(("127.0.0.1", 0))

    def serve():
        conn, _ = listener.accept()
        with conn:
            conn.recv(65536)
            conn.sendall(b"not http\r\n\r\n")

    threading.Thread(target=serve, daemon=True).start()
    monkeypatch.setattr(images, "_is_public", lambda ip: ip == "127.0.0.1")
    try:
        with pytest.raises(ValueError, match="could not fetch image"):
            images._read(f"http://127.0.0.1:{listener.getsockname()[1]}/a.png")
    finally:
        listener.close()


@pytest.fixture
def image_server():
    """A PNG at /a.png on 127.0.0.1; /hop?to=URL redirects to URL."""
    png = base64.b64decode(encoded())

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            url = urllib.parse.urlsplit(self.path)
            if url.path == "/hop":
                self.send_response(302)
                self.send_header("Location", urllib.parse.parse_qs(url.query)["to"][0])
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(png)))
            self.end_headers()
            self.wfile.write(png)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


@pytest.mark.parametrize(
    "target, message",
    [
        ("http://127.0.0.2:9/a.png", "public addresses"),
        ("ftp://127.0.0.1/a.png", "unknown url type"),
        ("file:///etc/hostname", "not allowed"),
    ],
)
def test_every_redirect_hop_is_checked(image_server, monkeypatch, target, message):
    # Let this server stand in for the public internet; nothing else is.
    monkeypatch.setattr(images, "_is_public", lambda ip: ip == "127.0.0.1")
    assert images._read(f"{image_server}/a.png").startswith(b"\x89PNG")
    with pytest.raises(ValueError, match=message):
        images._read(f"{image_server}/hop?to={urllib.parse.quote(target)}")


def test_internal_urls_can_be_allowed(image_server, monkeypatch):
    monkeypatch.setenv("ATOM_IMAGE_URL_ALLOW_PRIVATE", "1")
    monkeypatch.setenv("no_proxy", "*")  # straight to the local server
    target = urllib.parse.quote(f"{image_server}/a.png")
    url = images.image_url_part(f"{image_server}/hop?to={target}")
    [message] = images.load_images([{"role": "user", "content": [url]}])
    assert message["content"][0]["image"].size == (32, 16)


# --- protocol translation --------------------------------------------------------


def test_anthropic_images_keep_their_place_in_user_and_tool_result_content():
    image = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": encoded()},
    }
    messages = anthropic_to_openai_messages(
        [
            AnthropicMessage(
                role="user",
                content=[
                    {"type": "text", "text": "a"},
                    image,
                    {"type": "text", "text": "b"},
                ],
            ),
            AnthropicMessage(
                role="user",
                content=[
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [{"type": "text", "text": "shot"}, image],
                    }
                ],
            ),
        ]
    )
    assert [p["type"] for p in messages[0]["content"]] == ["text", "image_url", "text"]
    assert messages[1]["role"] == "tool"
    assert [p["type"] for p in messages[1]["content"]] == ["text", "image_url"]


def test_anthropic_text_only_content_is_unchanged():
    messages = anthropic_to_openai_messages(
        [
            AnthropicMessage(
                role="user",
                content=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
            )
        ]
    )
    assert messages == [{"role": "user", "content": "a\nb"}]


@pytest.mark.parametrize("where", ["assistant", "system"])
def test_anthropic_images_outside_user_turns_are_refused(where):
    image = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "QQ=="},
    }
    with pytest.raises(ValueError, match=f"{where} messages"):
        if where == "system":
            anthropic_to_openai_messages(
                [AnthropicMessage(role="user", content="hi")], [image]
            )
        else:
            anthropic_to_openai_messages(
                [AnthropicMessage(role="assistant", content=[image])]
            )


def test_responses_images_in_user_developer_and_tool_output():
    image = {"type": "input_image", "image_url": PNG}
    request = ResponsesRequest(
        input=[
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": "d"}, image],
            },
            {"role": "user", "content": [image, {"type": "input_text", "text": "q"}]},
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "shot",
                "arguments": "{}",
            },
            {"type": "function_call_output", "call_id": "c1", "output": [image]},
        ]
    )
    messages = responses_to_openai_messages(request)
    assert [m["role"] for m in messages] == ["user", "user", "assistant", "tool"]
    assert all(images.has_image(m["content"]) for i, m in enumerate(messages) if i != 2)


# --- endpoints -------------------------------------------------------------------


@pytest.fixture
def engine(monkeypatch):
    """Record what reaches generation; images must arrive as media, not text."""
    state = SimpleNamespace(text=[], media=[])

    async def text(prompt, *_a, **_k):
        state.text.append(prompt)
        yield {
            "text": "ok",
            "finish_reason": "eos",
            "num_tokens_input": 1,
            "num_tokens_output": 1,
        }

    async def media(tokens, data, *_a, **_k):
        state.media.append((tokens, data))
        yield {
            "text": "ok",
            "finish_reason": "eos",
            "num_tokens_input": 1,
            "num_tokens_output": 1,
        }

    async def consume(generator, *_args):
        async for output in generator:
            pass
        return output

    def prepare(messages, *_args):
        loaded = images.load_images(api_server._role_contents(messages))
        count = sum(
            p.get("type") == "image"
            for m in loaded
            if isinstance(m["content"], list)
            for p in m["content"]
        )
        return [7] * count, {"images": count}

    for name, value in {
        "model_name": "m",
        "_request_logger": None,
        "default_chat_template_kwargs": {},
        "reasoning_toggle": None,
        "reasoning_dialect": None,
        "custom_message_encoder": None,
        "tool_call_parser_cls": None,
        "tokenizer": SimpleNamespace(
            encode=lambda t, **k: [1] * 4, decode=lambda ids, **k: ""
        ),
        "engine": SimpleNamespace(
            config=SimpleNamespace(generation_config=None, max_model_len=4096)
        ),
        "generate_async": text,
        "generate_async_multimodal": media,
        "_run_nonstream_with_disconnect": consume,
        "_prepare_multimodal_inputs": prepare,
        "_get_multimodal_processor": lambda: None,
        "apply_chat_template": lambda *a, **k: "<text prompt>",
    }.items():
        monkeypatch.setattr(api_server, name, value)
    return state


def run(coro):
    """(status, body); a successful handler returns its body directly."""
    response = asyncio.run(coro)
    if isinstance(response, dict):
        return 200, response
    return response.status_code, json.loads(response.body)


def responses_request(content, role="user"):
    from starlette.requests import Request

    body = json.dumps(
        {
            "model": "m",
            "input": [{"role": role, "content": content}],
            "max_output_tokens": 8,
        }
    ).encode()

    async def receive():
        return {"type": "http.request", "body": body}

    return Request(
        {"type": "http", "method": "POST", "path": "/v1/responses", "headers": []},
        receive,
    )


def test_responses_delivers_images_to_the_multimodal_path(engine):
    status, _ = run(
        api_server.responses_create(
            responses_request(
                [
                    {"type": "input_image", "image_url": PNG},
                    {"type": "input_text", "text": "what?"},
                ]
            )
        )
    )
    assert status == 200
    assert engine.media == [([7], {"images": 1})] and not engine.text


def test_responses_refuses_system_images_before_generation(engine):
    status, body = run(
        api_server.responses_create(
            responses_request(
                [{"type": "input_image", "image_url": PNG}], role="system"
            )
        )
    )
    assert status == 400 and "system messages" in json.dumps(body)
    assert not engine.media and not engine.text


def test_anthropic_delivers_images_to_the_multimodal_path(engine):
    from atom.entrypoints.openai.serving_anthropic import AnthropicMessagesRequest

    request = AnthropicMessagesRequest(
        model="m",
        max_tokens=8,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": encoded(),
                        },
                    },
                    {"type": "text", "text": "what?"},
                ],
            }
        ],
    )
    status, _ = run(api_server.anthropic_messages(request, None))
    assert status == 200
    assert engine.media == [([7], {"images": 1})] and not engine.text


@pytest.mark.parametrize(
    "messages",
    [
        [
            {
                "role": "system",
                "content": [{"type": "image_url", "image_url": {"url": PNG}}],
            }
        ],
        [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [{"type": "image_url", "image_url": {"url": PNG}}],
            },
        ],
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "/etc/hostname"}}
                ],
            }
        ],
        # The processor-style part goes through the same checks.
        [{"role": "user", "content": [{"type": "image", "image": "/etc/hostname"}]}],
        [{"role": "system", "content": [{"type": "image", "image": PNG}]}],
    ],
)
def test_chat_refuses_misplaced_or_local_images_before_generation(engine, messages):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            api_server.chat_completions(
                ChatCompletionRequest(model="m", messages=messages), None
            )
        )
    assert exc.value.status_code == 400
    assert not engine.media and not engine.text

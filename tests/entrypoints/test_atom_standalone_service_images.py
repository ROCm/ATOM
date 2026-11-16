# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Image input through the ATOMesh standalone service's chat endpoints.

The service is driven as the mesh router drives it, over a fake engine that
records what reaches it. The model's processor is a stub; the image checks,
loading and placeholder expansion between the two are the server's own.
"""

import base64
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

Image = pytest.importorskip("PIL.Image")
standalone = pytest.importorskip(
    "atom.entrypoints.atomesh.atom_standalone_service", exc_type=ImportError
)
from atom.entrypoints.openai import api_server

PLACEHOLDER = 42
ANSWER_TOKEN = 7
QUESTION = {"type": "text", "text": "What color is this?"}


def _png_url(color):
    out = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(out, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(out.getvalue()).decode()}"


def _image(url):
    return {"type": "image_url", "image_url": {"url": url}}


class Tokenizer:
    chat_template = "{{ messages }}"

    def apply_chat_template(self, messages, tools=None, **_kwargs):
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<bot>"

    def encode(self, text, **_kwargs):
        return [1] * len(text)

    def decode(self, token_ids, **_kwargs):
        return "red" if ANSWER_TOKEN in list(token_ids) else ""


class Engine:
    """What `AtomEngineService` calls, recording each request it is given."""

    def __init__(self):
        self.config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["KimiK3ForConditionalGeneration"]),
            multimodal_config=SimpleNamespace(
                media_placeholder_token_id=PLACEHOLDER,
                vision_config=SimpleNamespace(merge_kernel_size=2),
            ),
        )
        self.io_processor = self
        self.core_mgr = self
        self.requests = {}
        self.preprocessed = []
        self._callbacks = []

    def preprocess(self, prompt, sampling_params, stream_callback=None, **kwargs):
        self.preprocessed.append({"prompt": prompt, **kwargs})
        self._callbacks.append(stream_callback)
        length = len(prompt) if isinstance(prompt, list) else len(prompt.split())
        return SimpleNamespace(id=len(self.preprocessed), num_prompt_tokens=length)

    def add_request(self, seqs):
        callback = self._callbacks.pop()
        callback(
            SimpleNamespace(
                output_tokens=[ANSWER_TOKEN], finished=True, finish_reason="stop"
            )
        )


def processor(**kwargs):
    """A Kimi-style processor: one placeholder per image, expanded by ATOM."""
    images = [media["image"] for media in kwargs["medias"]]
    return {
        "input_ids": np.array([[9] + [PLACEHOLDER, 8] * len(images)]),
        "pixel_values": np.stack([np.asarray(image) for image in images]),
        "grid_thws": np.array([[1, 4, 4]] * len(images)),
    }


@pytest.fixture
def service(tmp_path, monkeypatch):
    loaded = []

    def load(*args):
        loaded.append(args)
        return processor

    monkeypatch.setattr(api_server, "get_native_multimodal_processor", load)
    service = standalone.AtomStandaloneService(Engine(), Tokenizer(), str(tmp_path))
    service.processors_loaded = loaded
    yield service
    service.close()


def _chat(service, messages, stream):
    """The response's text and usage, through either delivery mode."""
    request = {"model": service.model_name, "messages": messages}
    if not stream:
        response = service.chat_completions(request)
        return response["choices"][0]["message"]["content"], response["usage"]
    stream_id = service.start_chat_completions_stream(request)
    text, usage = "", None
    for _ in range(100):
        for chunk in service.drain_chat_completions_stream(stream_id, timeout=0.1):
            payload = chunk.removeprefix("data: ").strip()
            if payload == "[DONE]":
                return text, usage
            data = json.loads(payload)
            usage = data.get("usage") or usage
            for choice in data.get("choices", ()):  # the usage chunk has none
                text += choice["delta"].get("content") or ""
    raise AssertionError("the stream never finished")


@pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
def test_an_image_reaches_the_engine_with_its_placeholders(service, stream):
    messages = [{"role": "user", "content": [_image(_png_url("red")), QUESTION]}]

    text, usage = _chat(service, messages, stream)

    (request,) = service.engine.preprocessed
    # Each image's placeholder expanded to its 2x2 merged grid of 4x4 patches.
    assert request["prompt"] == [9] + [PLACEHOLDER] * 4 + [8]
    data = request["multimodal_data"]
    assert data["embedding_spans"] == ((1, 4),)
    assert data["pixel_values"][0, 0, 0].tolist() == [255, 0, 0]
    assert text == "red"
    assert usage["prompt_tokens"] == len(request["prompt"])


def test_the_processor_is_loaded_once(service):
    messages = [{"role": "user", "content": [_image(_png_url("blue")), QUESTION]}]
    _chat(service, messages, stream=False)
    _chat(service, messages, stream=True)
    assert len(service.processors_loaded) == 1


@pytest.mark.parametrize(
    "message",
    [
        {"role": "user", "content": [_image("/etc/passwd"), QUESTION]},
        {"role": "user", "content": [_image("file:///etc/passwd"), QUESTION]},
        {"role": "user", "content": [_image("data:image/png;base64,!!"), QUESTION]},
        {
            "role": "user",
            "content": [_image("data:image/png;base64,aGVsbG8="), QUESTION],
        },
        {"role": "assistant", "content": [_image(_png_url("red"))]},
        {
            "role": "user",
            "content": [{"type": "video_url", "video_url": {"url": "x.mp4"}}],
        },
    ],
    ids=["server-path", "file-url", "bad-base64", "not-an-image", "assistant", "video"],
)
@pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
def test_a_malformed_image_part_is_the_clients_error(service, message, stream):
    """A `ValueError` is what the mesh router answers with a 400."""
    with pytest.raises(ValueError) as error:
        _chat(service, [message], stream)
    assert error.type is ValueError
    assert service.engine.preprocessed == []


@pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
def test_a_text_only_request_is_rendered_as_text(service, stream):
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": [QUESTION, {"type": "text", "text": "Answer."}]},
    ]

    text, _usage = _chat(service, messages, stream)

    (request,) = service.engine.preprocessed
    assert request["prompt"] == (
        "<system>Be brief.<user>What color is this?\nAnswer.<bot>"
    )
    assert request.get("multimodal_data") is None
    assert text == "red"
    assert service.processors_loaded == []

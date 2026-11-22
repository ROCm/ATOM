# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The encoder-only server: the chat app answering with ec_transfer_params from
a stub tower and the real embedding pool over a fake Mooncake engine."""

import asyncio
import base64
import ctypes
import io
import sys
import types
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock

import httpx
import pytest
import torch
from PIL import Image
from safetensors.torch import save_file
from torch import nn

# The pool's module imports the KV connector, which imports aiter's parallel
# state: stub what a CPU runner lacks, as tests/test_ec_transport.py does.
_stubbed = []
for _name in ("aiter", "aiter.dist", "aiter.dist.parallel_state"):
    if _name not in sys.modules:
        sys.modules[_name] = types.ModuleType(_name)
        sys.modules[_name].__path__ = []
        _stubbed.append(_name)
for _fn in ("get_dp_group", "get_tp_group"):
    if not hasattr(sys.modules["aiter.dist.parallel_state"], _fn):
        setattr(sys.modules["aiter.dist.parallel_state"], _fn, MagicMock())

from atom.kv_transfer.disaggregation.mooncake import ec_transport as ec
from atom.kv_transfer.disaggregation.mooncake import mooncake_connector as mc

if "aiter" in _stubbed:
    del sys.modules["aiter"]

from atom.entrypoints.openai import api_server
from atom.model_engine import encoder_only
from atom.model_engine.arg_utils import EngineArgs
from atom.model_engine.multimodal_runtime import multimodal_cache_seed

HIDDEN = 8
# Two images: 4 and 2 placeholder tokens.
SPANS = [(2, 4), (8, 2)]
TOKENS = list(range(12))


def _data_url() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class FakeEngine:
    """Moves bytes between registered regions of engines in this process."""

    def __init__(self, sessions):
        self.sessions = sessions

    def initialize(self, ip, metadata, protocol, ib_device):
        self.port = 17000 + len(self.sessions)
        self.sessions[f"{ip}:{self.port}"] = self
        return 0

    def get_rpc_port(self):
        return self.port

    def register_memory(self, ptr, length):
        return 0

    def transfer_sync_read(self, session, dst, src, length):
        ctypes.memmove(dst, src, length)
        return 0


class StubVisionModel:
    """A model class as `VisionEncoder` reads one."""

    packed_modules_mapping: ClassVar[dict] = {}
    weights_mapping: ClassVar[dict] = {}

    def get_vision_embeddings(self, pixel_values, grid_thw):
        return self.vision_tower(pixel_values)


def _encoder():
    tower = nn.Linear(HIDDEN, HIDDEN, bias=False)
    nn.init.eye_(tower.weight)
    tower.weight.data *= 2
    return encoder_only.VisionEncoder(StubVisionModel, {"vision_tower": tower})


@pytest.fixture
def server(monkeypatch):
    """An encoder-only app; the model's processor returns two images' spans."""
    sessions = {}
    monkeypatch.setattr(mc, "_MOONCAKE_AVAILABLE", True)
    monkeypatch.setattr(
        mc, "TransferEngine", lambda: FakeEngine(sessions), raising=False
    )
    monkeypatch.setattr(ec, "get_ip", lambda: "192.0.2.7")
    monkeypatch.setenv("ATOM_MOONCAKE_IB_DEVICE", "rocep139s0")
    monkeypatch.setenv("ATOM_EC_LEASE_S", "60")
    calls = SimpleNamespace(prepared=[], encoded=[])

    def fake_prepare(_config, _processor, conversation, media, _kwargs, tools=None):
        rows = sum(count for _, count in SPANS)
        data = {
            "pixel_values": torch.arange(rows * HIDDEN, dtype=torch.float32).view(
                rows, HIDDEN
            ),
            "image_grid_thw": torch.tensor([[1, 4, 4], [1, 2, 4]]),
            "embedding_spans": list(SPANS),
        }
        calls.prepared.append((conversation, media, data))
        return list(TOKENS), data

    config = SimpleNamespace(
        torch_dtype=torch.float32,
        max_model_len=4096,
        hf_config=SimpleNamespace(max_position_embeddings=4096),
    )

    def make(pool_bytes=1 << 16, encoder=None):
        pool = ec.EcPool(pool_bytes, "cpu")
        server_encoder = encoder_only.EncoderOnly(
            config, encoder or _encoder(), pool, "cpu"
        )
        monkeypatch.setattr(api_server, "encoder", server_encoder)
        calls.pool = pool
        return server_encoder

    monkeypatch.setattr(api_server, "engine", None)
    monkeypatch.setattr(api_server, "encoder", None)
    monkeypatch.setattr(api_server, "model_name", "m")
    monkeypatch.setattr(api_server, "_request_logger", None)
    monkeypatch.setattr(api_server, "default_chat_template_kwargs", {})
    monkeypatch.setattr(api_server, "reasoning_toggle", None)
    monkeypatch.setattr(api_server, "custom_message_encoder", None)
    monkeypatch.setattr(api_server, "tool_call_parser_cls", None)
    monkeypatch.setattr(api_server, "_get_multimodal_processor", lambda: "processor")
    monkeypatch.setattr(api_server, "prepare_multimodal_inputs", fake_prepare)
    calls.make = make
    yield calls
    if api_server.encoder is not None:
        api_server.encoder.close()


def _request(method, path, body=None):
    async def send():
        transport = httpx.ASGITransport(app=api_server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            return await client.request(method, path, json=body)

    return asyncio.run(send())


def _two_image_chat(**body):
    image = {"type": "image_url", "image_url": {"url": _data_url()}}
    content = [image, {"type": "text", "text": "compare"}, image]
    return _request(
        "POST",
        "/v1/chat/completions",
        {"model": "m", "messages": [{"role": "user", "content": content}], **body},
    )


def test_two_image_request_answers_with_ec_transfer_params(server):
    server.make()
    response = _two_image_chat()

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["choices"] == []
    assert body["usage"]["prompt_tokens"] == len(TOKENS)
    params = body["ec_transfer_params"]
    _conversation, media, data = server.prepared[0]
    assert len(media["image"]) == 2
    assert params["rows"] == sum(count for _, count in SPANS)
    assert params["cache_seed"] == multimodal_cache_seed(data)

    # The pool holds the tower's output for P to read.
    reader = ec.EcReader(1 << 12, "cpu")
    rows = reader.read(params, params["cache_seed"], torch.float32)
    assert torch.equal(rows, data["pixel_values"] * 2)


def test_streamed_request_gets_the_same_answer(server):
    server.make()
    response = _two_image_chat(stream=True)

    assert response.status_code == 200, response.text
    assert response.json()["ec_transfer_params"]["rows"] == 6


def test_full_pool_answers_503(server):
    # Room for one request's rows: the second finds the first inside its lease.
    server.make(pool_bytes=256)
    assert _two_image_chat().status_code == 200

    response = _two_image_chat()

    assert response.status_code == 503
    assert "full" in response.json()["error"]["message"]
    assert "choices" not in response.json()


def test_encode_failure_answers_500(server):
    class Failing(StubVisionModel):
        def get_vision_embeddings(self, pixel_values, grid_thw):
            raise RuntimeError("HIP error: out of memory")

    server.make(encoder=encoder_only.VisionEncoder(Failing, {}))

    response = _two_image_chat()

    assert response.status_code == 500
    assert "out of memory" in response.json()["error"]["message"]


def test_rows_other_than_the_span_total_answer_500(server):
    class Short(StubVisionModel):
        def get_vision_embeddings(self, pixel_values, grid_thw):
            return pixel_values[:-1]

    server.make(encoder=encoder_only.VisionEncoder(Short, {}))

    response = _two_image_chat()

    assert response.status_code == 500
    assert "rows" in response.json()["error"]["message"]


def test_text_request_is_refused(server):
    server.make()
    response = _request(
        "POST",
        "/v1/chat/completions",
        {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 400
    assert "image requests only" in response.json()["error"]["message"]


def test_health_and_model_discovery_answer_without_an_engine(server):
    server.make()

    assert _request("GET", "/health").json() == {"status": "ok"}
    models = _request("GET", "/v1/models")
    assert models.status_code == 200
    assert models.json()["data"][0]["id"] == "m"


def test_health_without_encoder_or_engine_is_503(server):
    assert _request("GET", "/health").status_code == 503


def test_main_builds_the_encoder_and_no_engine(monkeypatch):
    built = []
    monkeypatch.setattr(
        sys, "argv", ["openai_server", "--model", "m", "--encoder-only"]
    )
    monkeypatch.setattr(api_server, "encoder", None)
    monkeypatch.setattr(api_server, "_setup_frontend", lambda args: None)
    monkeypatch.setattr(api_server, "_setup_reasoning_and_tools", lambda args: None)
    monkeypatch.setattr(api_server, "_serve", lambda args: built.append("serve"))
    monkeypatch.setattr(
        EngineArgs, "create_encoder", lambda self: built.append("encoder") or "E"
    )
    monkeypatch.setattr(
        EngineArgs, "create_engine", lambda self, **kw: built.append("engine")
    )

    api_server.main()

    assert built == ["encoder", "serve"]
    assert api_server.encoder == "E"


def test_weights_load_without_the_language_model(tmp_path, monkeypatch):
    save_file(
        {
            "vision_tower.weight": torch.full((HIDDEN, HIDDEN), 3.0),
            "language_model.model.embed_tokens.weight": torch.ones(4, HIDDEN),
            "language_model.lm_head.weight": torch.ones(4, HIDDEN),
        },
        str(tmp_path / "model.safetensors"),
    )
    yielded = []
    iterate = encoder_only.safetensors_weights_iterator

    def recording_iterator(*args):
        for name, tensor in iterate(*args):
            yielded.append(name)
            yield name, tensor

    monkeypatch.setattr(
        encoder_only, "safetensors_weights_iterator", recording_iterator
    )
    encoder = _encoder()
    config = SimpleNamespace(
        model=str(tmp_path),
        hf_config=SimpleNamespace(num_hidden_layers=1),
        load_dummy=None,
    )

    encoder_only.load_vision_weights(encoder, config)

    assert yielded == ["vision_tower.weight"]
    assert torch.equal(encoder.vision_tower.weight, torch.full((HIDDEN, HIDDEN), 3.0))

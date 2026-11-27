# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`/v1/completions` refuses a prompt that encodes to no tokens with a 400.

Qwen3's and gpt-oss's tokenizers add no BOS, so `"prompt": ""` encodes to
nothing, and building its `Sequence` raised `IndexError`: a 500.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from atom.entrypoints.openai import api_server


@pytest.fixture
def engine(monkeypatch):
    """The route's real `InputOutputProcessor`, with a tokenizer that adds no
    BOS, and an engine that records what reaches it."""
    proc = api_server.InputOutputProcessor.__new__(api_server.InputOutputProcessor)
    proc.config = MagicMock()
    proc.config.enable_rapidserve = False
    proc.config.kv_transfer_config = None
    proc.config.speculative_config = None
    proc.tokenizer = SimpleNamespace(encode=lambda text, **_kw: [ord(c) for c in text])
    proc.block_size = 4
    proc.requests = {}
    proc.has_per_req_cache = False
    proc.num_speculative_tokens = 0
    proc._greedy_ids_only = False
    state = SimpleNamespace(io_processor=proc, submitted=[])
    state.core_mgr = SimpleNamespace(add_request=state.submitted.append)
    monkeypatch.setattr(api_server, "engine", state)
    # `_build_sampling_params` reads the output cap from a full engine config.
    monkeypatch.setattr(api_server, "_get_engine_config", lambda: None)
    monkeypatch.setattr(api_server, "_stream_batch_dispatcher", MagicMock())
    monkeypatch.setattr(api_server, "model_name", "m")
    monkeypatch.setattr(api_server, "_request_logger", None)
    return state


@pytest.mark.parametrize("stream", [False, True])
def test_an_empty_prompt_gets_a_400(engine, stream):
    async def post():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api_server.app),
            base_url="http://test",
        ) as client:
            return await client.post(
                "/v1/completions",
                json={"model": "m", "prompt": "", "stream": stream},
            )

    response = asyncio.run(post())
    assert response.status_code == 400
    assert "The decoder prompt cannot be empty" in response.text
    assert engine.submitted == []
    assert engine.io_processor.requests == {}

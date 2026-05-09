# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Integration tests for the ATOM OpenAI-compatible server.

These tests spawn a real ATOM server with a small model (Qwen/Qwen3-0.6B)
and verify the API endpoints via HTTP requests. Requires a GPU.

Usage:
    # Run from repo root (requires GPU + model downloaded):
    pytest tests/entrypoints/test_openai_server.py -v --timeout=300

    # Skip if no GPU available:
    pytest tests/entrypoints/test_openai_server.py -v -k "not gpu"
"""

import json
import os
import signal
import socket
import subprocess
import sys
import time
from typing import List, Optional

import pytest
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = os.environ.get("ATOM_TEST_MODEL", "Qwen/Qwen3-0.6B")
HOST = "127.0.0.1"
TIMEOUT = int(os.environ.get("ATOM_TEST_TIMEOUT", "300"))  # server startup timeout


def _find_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Skip marker: skip entire module if no GPU or torch not available
# ---------------------------------------------------------------------------

try:
    import torch

    _HAS_GPU = torch.cuda.is_available() or (
        hasattr(torch, "hip") and torch.hip.is_available()
    )
except ImportError:
    _HAS_GPU = False

pytestmark = pytest.mark.skipif(not _HAS_GPU, reason="No GPU available")


# ---------------------------------------------------------------------------
# RemoteATOMServer — spawns a real server subprocess
# ---------------------------------------------------------------------------


class RemoteATOMServer:
    """Context manager that spawns a real ATOM OpenAI server.

    Inspired by vLLM's RemoteOpenAIServer test utility.
    """

    def __init__(
        self,
        model: str,
        host: str = HOST,
        port: Optional[int] = None,
        extra_args: Optional[List[str]] = None,
        timeout: int = TIMEOUT,
    ):
        self.model = model
        self.host = host
        self.port = port or _find_free_port()
        self.base_url = f"http://{self.host}:{self.port}"
        self.timeout = timeout
        self.proc: Optional[subprocess.Popen] = None

        cmd = [
            sys.executable,
            "-m",
            "atom.entrypoints.openai_server",
            "--model",
            model,
            "--host",
            host,
            "--server-port",
            str(self.port),
            "--kv_cache_dtype",
            "fp8",
        ]
        if extra_args:
            cmd.extend(extra_args)

        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={**os.environ, "AITER_LOG_LEVEL": "WARNING"},
        )

    def wait_for_ready(self):
        """Poll /health until the server is ready or timeout."""
        start = time.time()
        while time.time() - start < self.timeout:
            try:
                r = requests.get(f"{self.base_url}/health", timeout=2)
                if r.status_code == 200:
                    return
            except (requests.ConnectionError, requests.Timeout):
                pass
            # Check if process died
            if self.proc.poll() is not None:
                stdout = self.proc.stdout.read().decode() if self.proc.stdout else ""
                raise RuntimeError(
                    f"ATOM server exited with code {self.proc.returncode}.\n"
                    f"Output:\n{stdout[-2000:]}"
                )
            time.sleep(2)
        raise RuntimeError(f"ATOM server did not become ready within {self.timeout}s")

    def shutdown(self):
        """Gracefully shut down the server."""
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()

    def __enter__(self):
        self.wait_for_ready()
        return self

    def __exit__(self, *args):
        self.shutdown()


# ---------------------------------------------------------------------------
# Module-scoped server fixture (starts once, shared across all tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def server():
    """Spawn a real ATOM server for the test module."""
    with RemoteATOMServer(MODEL) as srv:
        yield srv


@pytest.fixture
def base_url(server):
    return server.base_url


# ---------------------------------------------------------------------------
# Health & Models Tests
# ---------------------------------------------------------------------------


class TestHealthAndModels:
    """Test /health and /v1/models endpoints."""

    def test_health(self, base_url):
        r = requests.get(f"{base_url}/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_models_list(self, base_url):
        r = requests.get(f"{base_url}/v1/models")
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == MODEL
        assert data["data"][0]["owned_by"] == "atom"


# ---------------------------------------------------------------------------
# Chat Completion Tests (Non-Streaming)
# ---------------------------------------------------------------------------


class TestChatCompletionNonStreaming:
    """Test /v1/chat/completions without streaming."""

    def test_basic_chat(self, base_url):
        r = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": "What is 1+1? Answer with just the number.",
                    }
                ],
                "max_tokens": 128,
                "stream": False,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == MODEL
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert isinstance(data["choices"][0]["message"]["content"], str)
        assert data["usage"]["prompt_tokens"] > 0
        assert data["usage"]["completion_tokens"] > 0
        assert data["usage"]["total_tokens"] > 0

    def test_reasoning_content_in_response(self, base_url):
        """Thinking models should return reasoning_content separately."""
        r = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "max_tokens": 512,
                "stream": False,
            },
        )
        assert r.status_code == 200
        msg = r.json()["choices"][0]["message"]
        # Content should exist (may or may not have reasoning depending on model)
        assert "content" in msg
        # If model produces thinking, reasoning_content should be present
        if msg.get("reasoning_content"):
            assert isinstance(msg["reasoning_content"], str)
            assert len(msg["reasoning_content"]) > 0

    def test_multimodal_content_format(self, base_url):
        """Server should accept multimodal content array format."""
        r = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Say hi"}],
                    }
                ],
                "max_tokens": 64,
                "stream": False,
            },
        )
        assert r.status_code == 200
        msg = r.json()["choices"][0]["message"]
        # Content or reasoning_content should exist
        has_content = len(msg.get("content", "")) > 0
        has_reasoning = len(msg.get("reasoning_content", "")) > 0
        assert has_content or has_reasoning

    def test_extra_fields_ignored(self, base_url):
        """Server should not return 422 for unknown OpenAI fields."""
        r = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 32,
                "stream": False,
                "stream_options": {"include_usage": True},
                "tools": [],
                "tool_choice": "auto",
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "n": 1,
            },
        )
        assert r.status_code == 200

    def test_system_message(self, base_url):
        r = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say hello."},
                ],
                "max_tokens": 64,
                "stream": False,
            },
        )
        assert r.status_code == 200

    def test_multi_turn(self, base_url):
        r = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [
                    {"role": "user", "content": "My name is Alice."},
                    {"role": "assistant", "content": "Nice to meet you, Alice!"},
                    {"role": "user", "content": "What is my name?"},
                ],
                "max_tokens": 64,
                "stream": False,
            },
        )
        assert r.status_code == 200

    def test_usage_timing_fields(self, base_url):
        """ATOM-specific: usage should include ttft_s, tpot_s, latency_s."""
        r = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 32,
                "stream": False,
            },
        )
        usage = r.json()["usage"]
        assert "ttft_s" in usage
        assert "tpot_s" in usage
        assert "latency_s" in usage
        assert usage["latency_s"] > 0


# ---------------------------------------------------------------------------
# Chat Completion Tests (Streaming)
# ---------------------------------------------------------------------------


class TestChatCompletionStreaming:
    """Test /v1/chat/completions with streaming."""

    def _collect_stream(self, base_url, messages, max_tokens=256):
        """Helper: send streaming request and collect chunks."""
        r = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
                "stream": True,
            },
            stream=True,
        )
        assert r.status_code == 200

        chunks = []
        for line in r.iter_lines():
            line = line.decode("utf-8").strip()
            if not line or not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                break
            chunk = json.loads(line[6:])
            # Skip usage-only chunks (no choices key)
            if "choices" in chunk:
                chunks.append(chunk)
        return chunks

    def test_streaming_basic(self, base_url):
        chunks = self._collect_stream(
            base_url,
            [{"role": "user", "content": "Say hello in one word."}],
            max_tokens=128,
        )
        assert len(chunks) > 0
        # First chunk should have role
        first = chunks[0]
        assert first["object"] == "chat.completion.chunk"
        assert first["choices"][0]["delta"].get("role") == "assistant"

    def test_streaming_content_accumulation(self, base_url):
        chunks = self._collect_stream(
            base_url,
            [{"role": "user", "content": "Count from 1 to 3."}],
            max_tokens=512,
        )
        content = ""
        reasoning = ""
        for chunk in chunks:
            delta = chunk["choices"][0].get("delta", {})
            if "content" in delta:
                content += delta["content"]
            if "reasoning_content" in delta:
                reasoning += delta["reasoning_content"]
        # At least one of content or reasoning should be non-empty
        assert len(content) + len(reasoning) > 0

    def test_streaming_reasoning_content(self, base_url):
        """Streaming should have reasoning_content in delta for thinking models."""
        chunks = self._collect_stream(
            base_url,
            [{"role": "user", "content": "What is 2+2?"}],
            max_tokens=512,
        )
        reasoning = ""
        content = ""
        for chunk in chunks:
            delta = chunk["choices"][0].get("delta", {})
            if "reasoning_content" in delta:
                reasoning += delta["reasoning_content"]
            if "content" in delta:
                content += delta["content"]

        # At least one of reasoning or content should be non-empty
        assert len(reasoning) + len(content) > 0

        # If model produces thinking, reasoning should be separate
        if reasoning:
            assert "content" not in reasoning or len(content) > 0

    def test_streaming_finish_reason(self, base_url):
        chunks = self._collect_stream(
            base_url,
            [{"role": "user", "content": "Hi"}],
            max_tokens=64,
        )
        # Find the chunk with finish_reason
        finish_reasons = [
            c["choices"][0].get("finish_reason")
            for c in chunks
            if c["choices"][0].get("finish_reason") is not None
        ]
        assert len(finish_reasons) > 0
        assert finish_reasons[-1] == "stop"

    def test_streaming_done_message(self, base_url):
        """Stream should end with data: [DONE]."""
        r = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 32,
                "stream": True,
            },
            stream=True,
        )
        lines = [
            line.decode("utf-8").strip()
            for line in r.iter_lines()
            if line.decode("utf-8").strip()
        ]
        assert lines[-1] == "data: [DONE]"


# ---------------------------------------------------------------------------
# Error Handling Tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Test error responses."""

    def test_wrong_model_name(self, base_url):
        r = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "nonexistent-model",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert r.status_code == 400

    def test_missing_messages(self, base_url):
        """Request without messages or prompt should fail."""
        r = requests.post(
            f"{base_url}/v1/chat/completions",
            json={"model": MODEL, "max_tokens": 32},
        )
        # Should fail with 400 or 500 (ValueError from get_messages())
        assert r.status_code in (400, 422, 500)


# ---------------------------------------------------------------------------
# Multi-sample (n > 1) Tests — issue #618
# ---------------------------------------------------------------------------


class TestMultiSampleChat:
    """Verify that ``n > 1`` returns multiple sampled completions."""

    def test_n_greater_than_one_returns_multiple_choices(self, base_url):
        r = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": "Give me any single word.",
                    }
                ],
                "max_tokens": 64,
                "temperature": 0.9,
                "top_p": 0.95,
                "stream": False,
                "n": 3,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["choices"]) == 3
        assert {c["index"] for c in data["choices"]} == {0, 1, 2}

    def test_n_greedy_collapses_to_one(self, base_url):
        """temperature=0 with n>1 should collapse to a single choice."""
        r = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 32,
                "temperature": 0.0,
                "stream": False,
                "n": 4,
            },
        )
        assert r.status_code == 200
        assert len(r.json()["choices"]) == 1

    def test_n_streaming_multiplexes_by_index(self, base_url):
        """Streaming with n>1 should tag every chunk by choices[0].index."""
        r = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Say a word."}],
                "max_tokens": 64,
                "temperature": 0.9,
                "stream": True,
                "n": 2,
            },
            stream=True,
        )
        assert r.status_code == 200
        indices_seen = set()
        finish_reasons_by_index = {}
        for line in r.iter_lines():
            line = line.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if "choices" not in chunk:
                continue
            idx = chunk["choices"][0]["index"]
            indices_seen.add(idx)
            fr = chunk["choices"][0].get("finish_reason")
            if fr is not None:
                finish_reasons_by_index[idx] = fr
        assert indices_seen == {0, 1}
        # Both siblings should have emitted a terminal finish_reason.
        assert set(finish_reasons_by_index.keys()) == {0, 1}


class TestMultiSampleCompletion:
    """n>1 on /v1/completions."""

    def test_completion_n_returns_multiple_choices(self, base_url):
        r = requests.post(
            f"{base_url}/v1/completions",
            json={
                "model": MODEL,
                "prompt": "The quick brown",
                "max_tokens": 16,
                "temperature": 0.9,
                "stream": False,
                "n": 2,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["choices"]) == 2
        assert {c["index"] for c in data["choices"]} == {0, 1}

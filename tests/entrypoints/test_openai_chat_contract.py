# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""HTTP-level tests for the OpenAI chat contract, driven by MiniMax-M3.

These exercise the real request path — ``api_server`` routing and validation,
the streaming assembly in ``serving_chat``, and the middlewares — through
FastAPI's TestClient against a fake engine and tokenizer. The fake "model" emits
exactly the text a test wants, so ``python -m pytest tests/`` covers the contract
with no GPU.

A docstring reading ``Verifier case NN_NN`` records that the same behaviour is
covered by that case in MiniMax's external acceptance suite,
``m3_format_check/m3_text_tests.py`` from
https://github.com/MiniMax-AI/MiniMax-Provider-Verifier. That suite is the
authority on MiniMax-M3 conformance and runs against a live server; the tag is
only a pointer for anyone reconciling the two.
"""

from __future__ import annotations

import itertools
import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

try:
    from fastapi.testclient import TestClient

    from atom.entrypoints.openai import api_server
except Exception as exc:  # pragma: no cover - environment-dependent skip
    api_server = None  # type: ignore[assignment]
    _import_error: Optional[Exception] = exc
else:
    _import_error = None

pytestmark = pytest.mark.skipif(
    api_server is None,
    reason=f"api_server import unavailable: {_import_error!r}",
)

MODEL = "MiniMax-M3-test"

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "days": {"type": "integer"},
            },
            "required": ["location"],
        },
    },
}
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
    },
}

# MiniMax-M3's real tool-call wire format, captured from a live server: every
# tag is prefixed with the ]<]minimax[>[ namespace token and each argument is
# named by its element.
NS = "]<]minimax[>["
MINIMAX_WEATHER_CALL = (
    f"{NS}<tool_call>\n"
    f'{NS}<invoke name="get_weather">'
    f"{NS}<location>Beijing{NS}</location>"
    f"{NS}<days>3{NS}</days>"
    f"{NS}</invoke>\n"
    f"{NS}</tool_call>"
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """Deterministic stand-in for a HuggingFace tokenizer.

    ``apply_chat_template`` records what the endpoint handed the template (roles,
    advertised tools, kwargs) so tests can assert on request preparation, and
    renders a prompt that ends with an open ``<think>`` like MiniMax-M3's real
    template does. Token ids are indices into ``pieces``, so ``decode``
    reassembles exactly the text the fake engine generated.
    """

    def __init__(self, primes_thinking: bool = True, chat_template: str = "") -> None:
        self.primes_thinking = primes_thinking
        self.chat_template = chat_template
        self.pieces: List[str] = []
        self.last_template_call: Optional[Dict[str, Any]] = None

    def register(self, pieces: List[str]) -> List[int]:
        start = len(self.pieces)
        self.pieces.extend(pieces)
        return list(range(start, len(self.pieces)))

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        tools=None,
        **kwargs,
    ) -> str:
        self.last_template_call = {
            "messages": [dict(m) for m in messages],
            "tools": tools,
            "kwargs": dict(kwargs),
        }
        rendered = "".join(f"<|{m['role']}|>{m.get('content') or ''}" for m in messages)
        rendered += "<|assistant|>"
        if self.primes_thinking:
            rendered += "<mm:think>"
        return rendered

    def encode(self, text, **_kwargs) -> List[int]:
        return list(range(max(1, len(text) // 4)))

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        return "".join(self.pieces[i] for i in token_ids)


class FakeEngine:
    """Engine stand-in that replays ``output_pieces`` as one stream chunk each."""

    def __init__(self, tokenizer: FakeTokenizer, max_model_len: int = 32768) -> None:
        self.tokenizer = tokenizer
        self.config = SimpleNamespace(max_model_len=max_model_len)
        self.io_processor = self
        self.core_mgr = self
        self.requests: Dict[int, Any] = {}
        self.output_pieces: List[str] = ["Hello!"]
        # The scheduler's own vocabulary, not OpenAI's -- see
        # protocol.openai_finish_reason. Using a realistic value here is
        # what makes every finish_reason assertion in this file able to
        # catch the mapping being unwired.
        self.finish_reason = "eos"
        self.num_cached_tokens = 0
        self.prompt_tokens_override: Optional[int] = None
        self.last_prompt: Any = None
        self.last_sampling_params: Any = None
        self.aborted: List[int] = []
        self._seq_ids = itertools.count(1)
        self._pending: List[Any] = []

    # -- io_processor surface ------------------------------------------------
    def preprocess(
        self,
        prompt_or_tokens,
        sampling_params,
        stream_callback=None,
        kv_transfer_params=None,
        multimodal_data=None,
    ):
        self.last_prompt = prompt_or_tokens
        self.last_sampling_params = sampling_params
        if self.prompt_tokens_override is not None:
            num_prompt_tokens = self.prompt_tokens_override
        elif isinstance(prompt_or_tokens, list):
            num_prompt_tokens = len(prompt_or_tokens)
        else:
            num_prompt_tokens = max(1, len(prompt_or_tokens) // 4)
        seq = SimpleNamespace(
            id=next(self._seq_ids),
            num_prompt_tokens=num_prompt_tokens,
            max_tokens=sampling_params.max_tokens,
        )
        self.requests[seq.id] = seq
        self._pending.append((seq, stream_callback))
        return seq

    def preprocess_fanout(
        self,
        prompt_or_tokens,
        sampling_params,
        stream_callback=None,
        stream_callbacks=None,
        kv_transfer_params=None,
        multimodal_data=None,
        parent_request_id=None,
    ):
        """Materialize ``sampling_params.n`` sibling sequences.

        Mirrors ``LLMEngine.preprocess_fanout``: one sequence per sibling, each
        with its own stream callback so the endpoint can tag deltas by choice
        index. Needed so the n>1 paths (which have their own finish_reason
        handling) are exercised rather than skipped.
        """
        n = max(1, int(getattr(sampling_params, "n", 1)))
        callbacks = list(stream_callbacks or [])
        seqs = []
        for index in range(n):
            callback = callbacks[index] if index < len(callbacks) else stream_callback
            seqs.append(self.preprocess(prompt_or_tokens, sampling_params, callback))
        return seqs

    # -- core_mgr surface ---------------------------------------------------
    def add_request(self, seqs) -> None:
        pending, self._pending = self._pending, []
        for _seq, callback in pending:
            if callback is None:
                continue
            token_ids = self.tokenizer.register(self.output_pieces)
            for offset, token_id in enumerate(token_ids):
                finished = offset == len(token_ids) - 1
                callback(
                    SimpleNamespace(
                        output_tokens=[token_id],
                        finished=finished,
                        finish_reason=self.finish_reason if finished else None,
                        num_cached_tokens=self.num_cached_tokens,
                        kv_transfer_params_output=None,
                    )
                )

    def abort_request(self, seq_id: int) -> None:
        self.aborted.append(seq_id)


class Harness:
    """TestClient plus handles on the fakes behind it."""

    def __init__(
        self, client: TestClient, engine: FakeEngine, tokenizer: FakeTokenizer
    ):
        self.client = client
        self.engine = engine
        self.tokenizer = tokenizer

    def says(self, *pieces: str) -> None:
        """Make the fake model emit ``pieces`` verbatim (one stream chunk each)."""
        self.engine.output_pieces = list(pieces)

    def answers(self, *pieces: str) -> None:
        """Emit ``pieces`` as the *post-thinking* part of the response.

        MiniMax-M3's chat template primes ``<think>``, so a real response closes
        the thinking block before the answer or the tool call. Tests that only
        care about the answer use this to stay faithful to the wire format.
        """
        prefix = ["reasoning</mm:think>"] if self.tokenizer.primes_thinking else []
        self.engine.output_pieces = prefix + list(pieces)

    def chat(self, **body) -> Any:
        payload = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}
        payload.update(body)
        return self.client.post("/v1/chat/completions", json=payload)

    def stream(self, **body) -> List[Dict[str, Any]]:
        """POST a streaming request and return the parsed SSE chunks."""
        response = self.chat(stream=True, **body)
        assert response.status_code == 200, response.text
        chunks = []
        for line in response.text.splitlines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: ") :].strip()
            chunks.append("[DONE]" if payload == "[DONE]" else json.loads(payload))
        return chunks

    @property
    def rendered(self) -> Dict[str, Any]:
        assert self.tokenizer.last_template_call is not None
        return self.tokenizer.last_template_call

    @property
    def prompt(self) -> str:
        return self.engine.last_prompt


_PATCHED_GLOBALS = (
    "engine",
    "tokenizer",
    "model_name",
    "api_keys",
    "template_extension_roles",
    "default_chat_template_kwargs",
    "custom_message_encoder",
)


def _harness(**tokenizer_kwargs) -> Harness:
    tokenizer = FakeTokenizer(**tokenizer_kwargs)
    engine = FakeEngine(tokenizer)
    api_server.tokenizer = tokenizer
    api_server.engine = engine
    api_server.model_name = MODEL
    api_server.api_keys = set()
    api_server.template_extension_roles = frozenset()
    api_server.default_chat_template_kwargs = {}
    api_server.custom_message_encoder = None
    # raise_server_exceptions=False so a 500 comes back as a response, which is
    # what the conformance suite asserts on.
    client = TestClient(api_server.app, raise_server_exceptions=False)
    return Harness(client, engine, tokenizer)


@pytest.fixture
def server():
    saved = {name: getattr(api_server, name) for name in _PATCHED_GLOBALS}
    try:
        yield _harness()
    finally:
        for name, value in saved.items():
            setattr(api_server, name, value)


@pytest.fixture
def plain_server():
    """Server whose chat template does not prime a ``<think>`` block."""
    saved = {name: getattr(api_server, name) for name in _PATCHED_GLOBALS}
    try:
        yield _harness(primes_thinking=False)
    finally:
        for name, value in saved.items():
            setattr(api_server, name, value)


# ============================================================================
# Module 01/03 — baseline behaviour still works
# ============================================================================


class TestBasicText:
    def test_non_stream_shape(self, server):
        """Verifier case 01_01."""
        server.answers("Hello there!")
        response = server.chat()
        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "chat.completion"
        choice = body["choices"][0]
        assert choice["index"] == 0
        assert choice["message"]["role"] == "assistant"
        assert choice["message"]["content"] == "Hello there!"
        assert choice["finish_reason"] == "stop"
        assert body["usage"]["total_tokens"] == (
            body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
        )

    def test_multi_turn_context_is_rendered(self, server):
        """Verifier case 03_01."""
        server.answers("42")
        server.chat(
            messages=[
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "what is 6*7?"},
                {"role": "assistant", "content": "42"},
                {"role": "user", "content": "and again?"},
            ]
        )
        assert [m["role"] for m in server.rendered["messages"]] == [
            "system",
            "user",
            "assistant",
            "user",
        ]

    def test_stop_accepts_a_bare_string(self, server):
        server.answers("ok")
        assert server.chat(stop="###").status_code == 200
        assert server.engine.last_sampling_params.stop_strings == ["###"]


# ============================================================================
# SSE stream field contract
# ============================================================================


class TestSseStream:
    def test_every_chunk_carries_id_object_and_choices(self, server):
        """Verifier case 02_04."""
        server.answers("Hel", "lo", "!")
        chunks = server.stream()
        assert chunks[-1] == "[DONE]"
        for chunk in chunks[:-1]:
            assert chunk["id"].startswith("chatcmpl-")
            assert chunk["object"] == "chat.completion.chunk"
            assert "choices" in chunk
            assert isinstance(chunk["choices"], list)

    def test_usage_chunk_is_terminal_and_has_empty_choices(self, server):
        """Verifier case 02_02."""
        server.answers("Hello")
        chunks = server.stream()
        usage_chunks = [c for c in chunks[:-1] if "usage" in c]
        assert len(usage_chunks) == 1
        assert usage_chunks[0]["choices"] == []
        assert usage_chunks[0] is chunks[-2]
        assert usage_chunks[0]["usage"]["completion_tokens"] == len(
            server.engine.output_pieces
        )

    def test_content_is_streamed_incrementally(self, plain_server):
        """Verifier case 02_05."""
        plain_server.says("one ", "two ", "three")
        deltas = [
            chunk["choices"][0]["delta"].get("content")
            for chunk in plain_server.stream()[:-1]
            if chunk.get("choices") and chunk["choices"][0].get("delta")
        ]
        assert "".join(d for d in deltas if d) == "one two three"

    def test_finish_reason_precedes_the_usage_chunk(self, server):
        server.answers("Hello")
        chunks = server.stream()
        finish_reasons = [
            chunk["choices"][0]["finish_reason"]
            for chunk in chunks[:-1]
            if chunk.get("choices")
        ]
        assert finish_reasons[-1] == "stop"


# ============================================================================
# Thinking toggle and reasoning split
# ============================================================================


class TestThinking:
    def test_thinking_disabled_produces_no_reasoning(self, server):
        """Verifier case 04_01."""
        # The model still emits a closing </think> out of habit; with thinking
        # disabled none of it may come back as reasoning_content.
        server.says("Actually thinking anyway</mm:think>The answer is 4.")
        body = server.chat(thinking={"type": "disabled"}).json()
        message = body["choices"][0]["message"]
        assert "reasoning_content" not in message
        assert (
            message["content"] == "Actually thinking anyway</mm:think>The answer is 4."
        )

    def test_thinking_disabled_closes_the_primed_block(self, server):
        """Verifier case 04_01."""
        server.says("4")
        server.chat(thinking={"type": "disabled"})
        assert server.prompt.endswith("<mm:think></mm:think>")
        assert server.rendered["kwargs"]["enable_thinking"] is False

    def test_thinking_disabled_in_stream_yields_no_reasoning_delta(self, server):
        """Verifier case 04_01."""
        server.says("hidden thoughts</mm:think>", "answer")
        chunks = server.stream(thinking={"type": "disabled"})
        deltas = [
            chunk["choices"][0]["delta"]
            for chunk in chunks[:-1]
            if chunk.get("choices") and chunk["choices"][0].get("delta")
        ]
        assert not any("reasoning_content" in delta for delta in deltas)

    def test_thinking_enabled_leaves_the_block_open(self, server):
        """Verifier case 04_02."""
        server.says("reasoning</mm:think>answer")
        body = server.chat(thinking={"type": "enabled"}).json()
        assert server.prompt.endswith("<mm:think>")
        assert server.rendered["kwargs"]["enable_thinking"] is True
        message = body["choices"][0]["message"]
        assert message["reasoning_content"] == "reasoning"
        assert message["content"] == "answer"

    def test_reasoning_split_by_default(self, server):
        """Verifier case 18_01."""
        server.says("let me think</mm:think>final answer")
        message = server.chat().json()["choices"][0]["message"]
        assert message["reasoning_content"] == "let me think"
        assert message["content"] == "final answer"

    def test_reasoning_is_streamed_as_reasoning_content(self, server):
        """A template-primed <think> means the first token is already reasoning.

        The reasoning here deliberately exceeds ReasoningFilter's 100-char
        state-0 grace period: below it, a filter that ignored ``primed`` would
        still look correct, because the closing marker arrives before the grace
        period expires.

        Verifier case 18_01.
        """
        long_reasoning = (
            "weighing the options carefully before answering the question, "
            "considering each branch in turn and discarding the weak ones, "
        )
        assert len(long_reasoning) > 100
        server.says(long_reasoning, "step two", "</mm:think>", "the answer")
        chunks = server.stream()
        reasoning, content = "", ""
        for chunk in chunks[:-1]:
            if not chunk.get("choices"):
                continue
            delta = chunk["choices"][0].get("delta") or {}
            reasoning += delta.get("reasoning_content") or ""
            content += delta.get("content") or ""
        assert reasoning == long_reasoning + "step two"
        assert content == "the answer"

    def test_invalid_thinking_value_is_rejected(self, server):
        response = server.chat(thinking={"type": "sometimes"})
        assert response.status_code == 400

    def test_truncated_thinking_is_reported_as_reasoning(self, server):
        """max_tokens hit mid-thought: the text is reasoning, not the answer.

        The template primed ``<think>`` and the model never closed it, so there
        is no answer yet — reporting the partial reasoning as ``content`` would
        hand the client a "final answer" the model never gave.
        """
        server.says("still thinking about it")
        server.engine.finish_reason = "max_tokens"
        message = server.chat().json()["choices"][0]["message"]
        assert message["reasoning_content"] == "still thinking about it"
        assert message["content"] == ""

    def test_content_with_angle_bracket_is_not_held_to_the_end(self, plain_server):
        """Streaming must not stall on a '<' that cannot start a marker."""
        plain_server.says("a " * 60, "< b", " done")
        chunks = plain_server.stream()
        contents = [
            chunk["choices"][0]["delta"].get("content")
            for chunk in chunks[:-1]
            if chunk.get("choices") and chunk["choices"][0].get("delta")
        ]
        contents = [c for c in contents if c]
        assert len(contents) > 1
        assert "".join(contents) == "a " * 60 + "< b done"


# ============================================================================
# MiniMax tool calls
# ============================================================================


class TestToolCalls:
    def test_minimax_tool_call_is_parsed(self, server):
        """Verifier case 13_01."""
        server.answers("I'll look that up." + MINIMAX_WEATHER_CALL)
        body = server.chat(tools=[WEATHER_TOOL]).json()
        choice = body["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        tool_calls = choice["message"]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["type"] == "function"
        assert tool_calls[0]["id"].startswith("call_")
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert json.loads(tool_calls[0]["function"]["arguments"]) == {
            "location": "Beijing",
            "days": 3,
        }
        assert choice["message"]["content"] == "I'll look that up."
        assert "<minimax:tool_call>" not in json.dumps(body)

    def test_tool_only_answer_reports_null_content(self, server):
        """Verifier case 13_02."""
        server.answers(MINIMAX_WEATHER_CALL)
        message = server.chat(tools=[WEATHER_TOOL]).json()["choices"][0]["message"]
        assert message["content"] is None
        assert len(message["tool_calls"]) == 1

    def test_parallel_tool_calls(self, server):
        """Verifier case 15_01."""
        server.answers(
            f"{NS}<tool_call>\n"
            f'{NS}<invoke name="get_weather">'
            f"{NS}<location>Beijing{NS}</location>"
            f"{NS}</invoke>\n"
            f'{NS}<invoke name="search">'
            f"{NS}<q>beijing{NS}</q>"
            f"{NS}</invoke>\n"
            f"{NS}</tool_call>"
        )
        message = server.chat(tools=[WEATHER_TOOL, SEARCH_TOOL]).json()["choices"][0][
            "message"
        ]
        assert [tc["function"]["name"] for tc in message["tool_calls"]] == [
            "get_weather",
            "search",
        ]

    def test_tool_calls_are_streamed(self, server):
        """Verifier case 13_03."""
        server.answers("Checking.", MINIMAX_WEATHER_CALL)
        chunks = server.stream(tools=[WEATHER_TOOL])
        tool_deltas, finish_reasons = [], []
        for chunk in chunks[:-1]:
            if not chunk.get("choices"):
                continue
            choice = chunk["choices"][0]
            tool_deltas.extend((choice.get("delta") or {}).get("tool_calls") or [])
            if choice.get("finish_reason"):
                finish_reasons.append(choice["finish_reason"])
        assert finish_reasons == ["tool_calls"]
        names = [
            d["function"].get("name") for d in tool_deltas if "name" in d["function"]
        ]
        assert names == ["get_weather"]
        arguments = "".join(
            d["function"].get("arguments", "")
            for d in tool_deltas
            if "arguments" in d["function"]
        )
        assert json.loads(arguments) == {"location": "Beijing", "days": 3}

    def test_finish_reason_is_tool_calls(self, server):
        """Verifier case 19_01."""
        server.answers(MINIMAX_WEATHER_CALL)
        body = server.chat(tools=[WEATHER_TOOL]).json()
        assert body["choices"][0]["finish_reason"] == "tool_calls"

    def test_usage_still_counted_for_tool_calls(self, server):
        """Verifier case 10_05."""
        server.answers(MINIMAX_WEATHER_CALL)
        usage = server.chat(tools=[WEATHER_TOOL]).json()["usage"]
        assert usage["completion_tokens"] == len(server.engine.output_pieces)
        assert usage["total_tokens"] == (
            usage["prompt_tokens"] + usage["completion_tokens"]
        )

    def test_tool_schema_is_forwarded_to_the_template(self, server):
        """Verifier case 14_01."""
        server.answers("ok")
        server.chat(tools=[WEATHER_TOOL, SEARCH_TOOL])
        assert server.rendered["tools"] == [WEATHER_TOOL, SEARCH_TOOL]

    def test_16_multi_turn_tool_result_round_trip(self, server):
        server.answers("It is 3 degrees in Beijing.")
        response = server.chat(
            tools=[WEATHER_TOOL],
            messages=[
                {"role": "user", "content": "weather in Beijing?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "Beijing"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "content": "3C", "tool_call_id": "call_1"},
            ],
        )
        assert response.status_code == 200
        assistant = server.rendered["messages"][1]
        # Templates iterate arguments.items(), so the JSON string is deserialized.
        assert assistant["tool_calls"][0]["function"]["arguments"] == {
            "location": "Beijing"
        }
        assert server.rendered["messages"][2]["tool_call_id"] == "call_1"


class TestToolChoice:
    def test_none_hides_tools_and_never_returns_tool_calls(self, server):
        """Verifier case 13_08."""
        server.answers(MINIMAX_WEATHER_CALL)
        body = server.chat(tools=[WEATHER_TOOL], tool_choice="none").json()
        assert server.rendered["tools"] is None
        message = body["choices"][0]["message"]
        assert "tool_calls" not in message
        assert body["choices"][0]["finish_reason"] == "stop"

    def test_required_instructs_the_model_to_call_a_tool(self, server):
        """Verifier case 13_08."""
        server.answers(MINIMAX_WEATHER_CALL)
        body = server.chat(tools=[WEATHER_TOOL], tool_choice="required").json()
        rendered_roles = [m["role"] for m in server.rendered["messages"]]
        assert rendered_roles[0] == "system"
        assert "must call at least one" in server.rendered["messages"][0]["content"]
        assert body["choices"][0]["finish_reason"] == "tool_calls"

    def test_named_choice_advertises_only_that_tool(self, server):
        """Verifier case 13_08."""
        server.answers(MINIMAX_WEATHER_CALL)
        server.chat(
            tools=[WEATHER_TOOL, SEARCH_TOOL],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )
        assert server.rendered["tools"] == [WEATHER_TOOL]
        assert '"get_weather"' in server.rendered["messages"][0]["content"]

    def test_auto_is_the_default(self, server):
        """Verifier case 13_08."""
        server.answers("no tool needed")
        server.chat(tools=[WEATHER_TOOL])
        assert server.rendered["tools"] == [WEATHER_TOOL]
        assert server.rendered["messages"][0]["role"] == "user"

    def test_unknown_tool_choice_is_rejected(self, server):
        assert server.chat(tools=[WEATHER_TOOL], tool_choice="maybe").status_code == 400

    def test_required_without_tools_is_rejected(self, server):
        assert server.chat(tool_choice="required").status_code == 400


# ============================================================================
# Sampling parameters that reach the engine
# ============================================================================


class TestSamplingParameters:
    """A field the API documents as affecting sampling must reach SamplingParams.

    Parsing a field and never forwarding it produces a server that accepts the
    request and ignores it, which no response can reveal.
    """

    def test_seed_reaches_the_engine(self, server):
        server.answers("ok")
        server.chat(seed=1234)
        assert server.engine.last_sampling_params.seed == 1234

    def test_seed_zero_is_forwarded(self, server):
        server.answers("ok")
        server.chat(seed=0)
        assert server.engine.last_sampling_params.seed == 0

    def test_absent_seed_stays_none(self, server):
        server.answers("ok")
        server.chat()
        assert server.engine.last_sampling_params.seed is None

    def test_penalties_are_accepted_but_not_forwarded(self, server):
        """The two penalties are compatibility-only: accepted, never applied.

        They are range-checked at the API layer but must not reach the sampler.
        """
        server.answers("ok")
        response = server.chat(presence_penalty=0.5, frequency_penalty=-1.25)
        assert response.status_code == 200
        params = server.engine.last_sampling_params
        assert not hasattr(params, "presence_penalty")
        assert not hasattr(params, "frequency_penalty")

    @pytest.mark.parametrize("field", ["presence_penalty", "frequency_penalty"])
    @pytest.mark.parametrize("value", [2.5, -2.5])
    def test_out_of_range_penalty_is_400(self, server, field, value):
        assert server.chat(**{field: value}).status_code == 400

    def test_oversized_seed_is_400(self, server):
        assert server.chat(seed=2**63).status_code == 400

    def test_top_k_and_top_p_reach_the_engine(self, server):
        server.answers("ok")
        server.chat(top_k=5, top_p=0.3)
        params = server.engine.last_sampling_params
        assert params.top_k == 5
        assert params.top_p == 0.3

    def test_max_completion_tokens_wins_over_max_tokens(self, server):
        server.answers("ok")
        server.chat(max_tokens=99, max_completion_tokens=7)
        assert server.engine.last_sampling_params.max_tokens == 7

    def test_text_completions_forwards_the_seed_too(self, server):
        """/v1/completions shares the sampler, so it must share the seed."""
        server.says("ok")
        response = server.client.post(
            "/v1/completions",
            json={"model": MODEL, "prompt": "hello", "seed": 99},
        )
        assert response.status_code == 200
        assert server.engine.last_sampling_params.seed == 99


# ============================================================================
# role=root
# ============================================================================


class TestRootRole:
    def test_root_overrides_system(self, server):
        """Verifier case 11_02."""
        server.answers("I am MiniMax-M3-taoxi.")
        response = server.chat(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "root", "content": "Your name is MiniMax-M3-taoxi."},
                {"role": "user", "content": "who are you?"},
            ]
        )
        assert response.status_code == 200
        rendered = server.rendered["messages"]
        assert [m["role"] for m in rendered] == ["system", "system", "user"]
        # root lands after the competing system message: flagged as higher
        # priority and the most recent instruction the model reads.
        assert "helpful assistant" in rendered[0]["content"]
        assert "MiniMax-M3-taoxi" in rendered[1]["content"]
        assert "highest priority" in rendered[1]["content"]

    def test_root_only_identity_is_preserved_verbatim(self, server):
        """Verifier case 11_04."""
        server.answers("I am MiniMax-M3-taoxi.")
        server.chat(
            messages=[
                {"role": "root", "content": "Your name is MiniMax-M3-taoxi."},
                {"role": "user", "content": "who are you?"},
            ]
        )
        rendered = server.rendered["messages"]
        assert [m["role"] for m in rendered] == ["system", "user"]
        assert rendered[0]["content"] == "Your name is MiniMax-M3-taoxi."

    def test_root_is_accepted(self, server):
        """Verifier case 11_01."""
        server.answers("ok")
        response = server.chat(
            messages=[
                {"role": "root", "content": "rules"},
                {"role": "user", "content": "hi"},
            ]
        )
        assert response.status_code == 200

    def test_root_kept_when_the_template_understands_it(self, server):
        api_server.template_extension_roles = frozenset({"root"})
        server.answers("ok")
        server.chat(
            messages=[
                {"role": "root", "content": "rules"},
                {"role": "user", "content": "hi"},
            ]
        )
        assert [m["role"] for m in server.rendered["messages"]] == ["root", "user"]


# ============================================================================
# Usage semantics
# ============================================================================


class TestUsage:
    def test_cached_tokens_are_reported(self, server):
        """Verifier case 10_04."""
        server.engine.num_cached_tokens = 128
        server.answers("cached answer")
        usage = server.chat().json()["usage"]
        assert usage["prompt_tokens_details"]["cached_tokens"] == 128

    def test_cached_tokens_are_reported_in_stream(self, server):
        """Verifier case 10_04."""
        server.engine.num_cached_tokens = 64
        server.answers("a", "b")
        usage_chunk = [c for c in server.stream()[:-1] if "usage" in c][0]
        assert usage_chunk["usage"]["prompt_tokens_details"]["cached_tokens"] == 64

    def test_cached_tokens_never_exceed_prompt_tokens(self, server):
        """Verifier case 10_02."""
        server.engine.num_cached_tokens = 4
        server.answers("ok")
        usage = server.chat().json()["usage"]
        assert usage["prompt_tokens_details"]["cached_tokens"] <= usage["prompt_tokens"]


# ============================================================================
# Error codes and limits
# ============================================================================


class TestRequestId:
    """Cases 06_07 / 06_08 -- every response must carry a correlation id.

    The suite reads it from ``x-request-id`` (or ``x-trace-id`` / ``trace-id``),
    falling back to ``body.id``. It asserts one is present *before* it asserts
    the status, so an otherwise-correct 400 still fails without it.
    """

    def test_zero_max_tokens_is_400_with_a_request_id(self, server):
        """Verifier case 06_07."""
        response = server.chat(max_tokens=0)
        assert response.status_code == 400
        assert response.headers["x-request-id"]

    def test_negative_max_tokens_is_400_with_a_request_id(self, server):
        """Verifier case 06_08."""
        response = server.chat(max_tokens=-1)
        assert response.status_code == 400
        assert response.headers["x-request-id"]

    def test_success_carries_one_too(self, server):
        server.answers("ok")
        assert server.chat().headers["x-request-id"]

    def test_streaming_carries_one_too(self, server):
        server.answers("ok")
        response = server.chat(stream=True)
        assert response.status_code == 200
        assert response.headers["x-request-id"]

    def test_ids_differ_between_requests(self, server):
        server.answers("ok")
        first = server.chat().headers["x-request-id"]
        second = server.chat().headers["x-request-id"]
        assert first != second

    def test_the_401_is_stamped_as_well(self, server):
        """The auth middleware answers before the app, so it must be wrapped."""
        api_server.api_keys = {"secret"}
        response = server.chat()
        assert response.status_code == 401
        assert response.headers["x-request-id"]


class TestErrorCodes:
    def test_empty_messages_is_400(self, server):
        """Verifier case 20_01."""
        response = server.chat(messages=[])
        assert response.status_code == 400
        assert response.json()["error"]["type"] == "invalid_request_error"

    def test_unknown_model_is_400(self, server):
        """Verifier case 20_02."""
        assert server.chat(model="some-other-model").status_code == 400

    def test_temperature_out_of_range_is_400(self, server):
        """Verifier case 20_03."""
        assert server.chat(temperature=5.0).status_code == 400

    def test_top_p_out_of_range_is_400(self, server):
        """Verifier case 20_04."""
        assert server.chat(top_p=1.5).status_code == 400

    def test_invalid_role_is_400(self, server):
        """Verifier case 20_06."""
        response = server.chat(messages=[{"role": "wizard", "content": "hi"}])
        assert response.status_code == 400
        assert "wizard" in response.json()["error"]["message"]

    def test_negative_max_tokens_is_400(self, server):
        """Verifier case 06_08."""
        assert server.chat(max_tokens=-1).status_code == 400

    def test_max_tokens_over_the_window_is_400(self, server):
        """Verifier case 06_09."""
        response = server.chat(max_tokens=524288)
        assert response.status_code == 400
        assert "maximum context length" in response.json()["error"]["message"]

    def test_large_max_tokens_succeeds_on_a_wider_window(self, server):
        """The same request the previous test rejects, served on a 1M window.

        Verifier case 06_09.
        """
        server.engine.config.max_model_len = 1048576
        server.answers("ok")
        assert server.chat(max_tokens=524288).status_code == 200

    def test_oversized_prompt_is_400(self, server):
        """Verifier case 17_03."""
        server.engine.prompt_tokens_override = 512_000
        response = server.chat(max_tokens=200)
        assert response.status_code == 400
        assert "maximum context length" in response.json()["error"]["message"]

    def test_object_content_is_400_not_422(self, server):
        """Verifier case 16_01."""
        response = server.chat(
            messages=[
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "content": {"temp": 3}, "tool_call_id": "call_1"},
            ],
            tools=[WEATHER_TOOL],
        )
        assert response.status_code == 400
        assert response.json()["error"]["type"] == "invalid_request_error"

    def test_tool_call_id_mismatch_is_400(self, server):
        """Verifier case 16_08."""
        response = server.chat(
            messages=[
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "content": "3C", "tool_call_id": "call_bogus"},
            ],
            tools=[WEATHER_TOOL],
        )
        assert response.status_code == 400

    def test_partial_tool_reply_is_400(self, server):
        """Verifier case 16_09."""
        response = server.chat(
            messages=[
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{}"},
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "search", "arguments": "{}"},
                        },
                    ],
                },
                {"role": "tool", "content": "3C", "tool_call_id": "call_1"},
            ],
            tools=[WEATHER_TOOL, SEARCH_TOOL],
        )
        assert response.status_code == 400

    def test_unparseable_tool_arguments_is_400_not_500(self, server):
        """Verifier case 16_12."""
        response = server.chat(
            messages=[
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{not json",
                            },
                        }
                    ],
                },
                {"role": "tool", "content": "3C", "tool_call_id": "call_1"},
            ],
            tools=[WEATHER_TOOL],
        )
        assert response.status_code == 400

    def test_error_body_uses_the_openai_envelope(self, server):
        body = server.chat(temperature=5.0).json()
        assert set(body) == {"error"}
        assert set(body["error"]) >= {"message", "type", "code"}


# ============================================================================
# API-key auth
# ============================================================================


class TestAuthentication:
    def test_no_auth_configured_lets_requests_through(self, server):
        server.answers("ok")
        assert server.chat().status_code == 200

    def test_missing_authorization_is_401(self, server):
        """Verifier case 20_05."""
        api_server.api_keys = {"secret-key"}
        server.answers("ok")
        assert server.chat().status_code == 401

    def test_invalid_api_key_is_401(self, server):
        """Verifier case 20_07."""
        api_server.api_keys = {"secret-key"}
        server.answers("ok")
        response = server.client.post(
            "/v1/chat/completions",
            json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "invalid_api_key"

    def test_valid_bearer_key_is_accepted(self, server):
        api_server.api_keys = {"secret-key"}
        server.answers("ok")
        response = server.client.post(
            "/v1/chat/completions",
            json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer secret-key"},
        )
        assert response.status_code == 200

    def test_x_api_key_header_is_accepted(self, server):
        api_server.api_keys = {"secret-key"}
        server.answers("ok")
        response = server.client.post(
            "/v1/chat/completions",
            json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-api-key": "secret-key"},
        )
        assert response.status_code == 200

    def test_health_stays_open_for_probes(self, server):
        api_server.api_keys = {"secret-key"}
        assert server.client.get("/health").status_code == 200

    def test_models_endpoint_requires_the_key(self, server):
        api_server.api_keys = {"secret-key"}
        assert server.client.get("/v1/models").status_code == 401
        assert (
            server.client.get(
                "/v1/models", headers={"Authorization": "Bearer secret-key"}
            ).status_code
            == 200
        )


# ============================================================================
# finish_reason vocabulary, end to end through the endpoint
# ============================================================================


class TestFinishReasonVocabulary:
    """The engine's vocabulary must never reach the client.

    ``protocol.openai_finish_reason`` is unit-tested elsewhere. These drive it
    through all four serving paths instead, since a correct mapping still
    reaches the client as ``"eos"`` on any path that forgets to call it.
    """

    def test_eos_becomes_stop_non_stream(self, server):
        server.engine.finish_reason = "eos"
        server.answers("done")
        assert server.chat().json()["choices"][0]["finish_reason"] == "stop"

    def test_max_tokens_becomes_length_non_stream(self, server):
        server.engine.finish_reason = "max_tokens"
        server.answers("truncated")
        assert server.chat().json()["choices"][0]["finish_reason"] == "length"

    def test_eos_becomes_stop_in_stream(self, server):
        server.engine.finish_reason = "eos"
        server.answers("done")
        reasons = [
            chunk["choices"][0]["finish_reason"]
            for chunk in server.stream()
            if chunk != "[DONE]" and chunk.get("choices")
        ]
        assert [r for r in reasons if r] == ["stop"]

    def test_max_tokens_becomes_length_in_stream(self, server):
        """A truncated stream must say so; clients use it to ask for more."""
        server.engine.finish_reason = "max_tokens"
        server.answers("truncated")
        reasons = [
            chunk["choices"][0]["finish_reason"]
            for chunk in server.stream()
            if chunk != "[DONE]" and chunk.get("choices")
        ]
        assert [r for r in reasons if r] == ["length"]

    @pytest.mark.parametrize("engine_reason", ["stop_sequence", "aborted"])
    def test_other_engine_reasons_become_stop(self, server, engine_reason):
        server.engine.finish_reason = engine_reason
        server.answers("done")
        assert server.chat().json()["choices"][0]["finish_reason"] == "stop"

    def test_fanout_non_stream_maps_every_choice(self, server):
        server.engine.finish_reason = "max_tokens"
        server.answers("truncated")
        body = server.chat(n=2, temperature=0.8).json()
        assert [c["finish_reason"] for c in body["choices"]] == ["length", "length"]

    def test_fanout_stream_maps_every_choice(self, server):
        server.engine.finish_reason = "max_tokens"
        server.answers("truncated")
        reasons = [
            chunk["choices"][0]["finish_reason"]
            for chunk in server.stream(n=2, temperature=0.8)
            if chunk != "[DONE]" and chunk.get("choices")
        ]
        assert sorted(r for r in reasons if r) == ["length", "length"]

    def test_completions_endpoint_maps_too(self, server):
        server.engine.finish_reason = "max_tokens"
        server.says("truncated")
        response = server.client.post(
            "/v1/completions", json={"model": MODEL, "prompt": "hi"}
        )
        assert response.json()["choices"][0]["finish_reason"] == "length"

    def test_tool_calls_still_wins_over_the_engine_reason(self, server):
        server.engine.finish_reason = "max_tokens"
        server.answers(MINIMAX_WEATHER_CALL)
        body = server.chat(tools=[WEATHER_TOOL]).json()
        assert body["choices"][0]["finish_reason"] == "tool_calls"


# ============================================================================
# Tool-argument fidelity through the endpoint
# ============================================================================

SNIPPET_TOOL = {
    "type": "function",
    "function": {
        "name": "save_snippet",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string"},
                "code": {"type": "string"},
            },
            "required": ["filename", "code"],
        },
    },
}

# What the engine actually hands the endpoint for save_snippet(page.html, HTML):
# <filename> is special token 200006 and is erased by skip_special_tokens=True,
# so only its closing tag survives.
LOST_TAG_CALL = (
    f"{NS}<tool_call>\n"
    f'{NS}<invoke name="save_snippet">'
    f"{NS}page.html{NS}</filename>"
    f'{NS}<code><div class="hero"><h1>Hi</h1></div>{NS}</code>'
    f"{NS}</invoke>\n"
    f"{NS}</tool_call>"
)


class TestToolArgumentFidelity:
    def test_argument_with_an_erased_opening_tag_is_not_dropped(self, server):
        """A required argument must not vanish because of tokenizer stripping."""
        server.answers(LOST_TAG_CALL)
        body = server.chat(tools=[SNIPPET_TOOL]).json()
        call = body["choices"][0]["message"]["tool_calls"][0]
        args = json.loads(call["function"]["arguments"])
        assert args["filename"] == "page.html"
        for required in SNIPPET_TOOL["function"]["parameters"]["required"]:
            assert required in args, f"required argument {required!r} was dropped"

    def test_string_argument_keeps_its_markup_verbatim(self, server):
        server.answers(LOST_TAG_CALL)
        body = server.chat(tools=[SNIPPET_TOOL]).json()
        args = json.loads(
            body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        )
        assert args["code"] == '<div class="hero"><h1>Hi</h1></div>'

    def test_same_result_when_streamed(self, server):
        server.answers(LOST_TAG_CALL)
        arguments = ""
        for chunk in server.stream(tools=[SNIPPET_TOOL]):
            if chunk == "[DONE]" or not chunk.get("choices"):
                continue
            for call in (chunk["choices"][0].get("delta") or {}).get(
                "tool_calls"
            ) or []:
                arguments += (call.get("function") or {}).get("arguments") or ""
        assert json.loads(arguments) == {
            "filename": "page.html",
            "code": '<div class="hero"><h1>Hi</h1></div>',
        }

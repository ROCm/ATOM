# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A PD decode node must not re-render and re-tokenize the prefill's prompt.

Both halves of the handoff are covered here: the producer echoing the ids it
already computed (``return_token_ids``), and the consumer using them instead of
calling the chat template and the tokenizer again.

The assertions are on *whether the work happened*, not on how long it took.
The defect these tests protect against is a duplicated computation, and
"``apply_chat_template`` was called" is exactly observable where a timing is
not. See ``docs/ttft_breakdown_guide.md`` for what the duplication costs.
"""

import asyncio
from types import SimpleNamespace

import pytest
from prometheus_client.parser import text_string_to_metric_families

from atom.entrypoints.openai import api_server
from atom.entrypoints.openai.metrics_setup import create_metrics_exporter
from atom.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
)
from atom.entrypoints.openai.request_timing import RequestTiming
from atom.entrypoints.openai.serving_chat import build_chat_response_multi
from atom.entrypoints.openai.serving_completion import build_completion_response_multi

PROMPT_IDS = [101, 102, 103, 104]


class _FakeSequence:
    """Only what the API process reads off a ``Sequence`` it just built."""

    def __init__(self, token_ids: list[int]) -> None:
        self.id = 7
        self.prompt_token_ids = list(token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.max_tokens = 16


@pytest.fixture
def server(monkeypatch):
    """A chat/completions handler wired to fakes, recording what it called.

    ``generate_async`` is replaced rather than the engine beneath it: these
    tests are about which input the handler hands down, and the real generator
    needs an engine to hand it to.
    """
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


def _chat(**kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="m",
        messages=[ChatMessage(role="user", content="hi")],
        temperature=0.0,
        **kwargs,
    )


class TestDecodeSkipsTheWorkPrefillAlreadyDid:
    def test_ids_in_kv_transfer_params_skip_template_and_tokenizer(self, server):
        # The shape a vLLM-style proxy (and atomesh's ATOM relay) sends.
        request = _chat(
            kv_transfer_params={
                "do_remote_prefill": True,
                "prompt_token_ids": PROMPT_IDS,
            }
        )
        asyncio.run(api_server.chat_completions(request, None))

        assert server.templates == 0, "decode rendered the chat template again"
        sent, _ = server.generate[0]
        assert sent == PROMPT_IDS, "the prefill's ids did not reach the engine"

    def test_ids_at_top_level_skip_template_and_tokenizer(self, server):
        asyncio.run(
            api_server.chat_completions(_chat(prompt_token_ids=PROMPT_IDS), None)
        )

        assert server.templates == 0
        assert server.generate[0][0] == PROMPT_IDS

    def test_without_ids_the_template_still_runs(self, server):
        # The fast path must not become the only path: a direct client sends
        # messages and nothing else, and still has to get a rendered prompt.
        asyncio.run(api_server.chat_completions(_chat(), None))

        assert server.templates == 1
        assert server.generate[0][0] == "<rendered prompt>"

    def test_completions_endpoint_accepts_ids_instead_of_prompt(self, server):
        request = CompletionRequest(model="m", prompt_token_ids=PROMPT_IDS)
        asyncio.run(api_server.completions(request, None))

        assert server.generate[0][0] == PROMPT_IDS

    def test_completions_endpoint_still_accepts_text(self, server):
        asyncio.run(
            api_server.completions(CompletionRequest(model="m", prompt="hi"), None)
        )

        assert server.generate[0][0] == "hi"


class TestTheSkipIsVisibleInTheChatTemplateHistogram:
    """A skipped render observes 0, and a real render observes its own wall.

    The zero matters as much as the measurement: without it the histogram's
    count would not match the request count on a decode node, and "we skipped
    the template" would be indistinguishable from "this build has no such
    metric". These run under a request timing, which is the only state that
    turns the observations on -- the tests above call the handler bare, so they
    cannot see this code at all.
    """

    @staticmethod
    def _timed(monkeypatch):
        exporter, _, _, breakdown = create_metrics_exporter()
        monkeypatch.setattr(api_server, "_ttft_breakdown", breakdown)
        timing = RequestTiming(0.0, lambda _seconds, _streaming: None)
        monkeypatch.setattr(api_server, "get_request_timing", lambda: timing)
        return exporter

    @staticmethod
    def _template_samples(exporter):
        return {
            sample.name: sample.value
            for family in text_string_to_metric_families(exporter.render().decode())
            for sample in family.samples
            if sample.name.startswith("atom:api_chat_template_seconds")
            and sample.name.endswith(("_count", "_sum"))
        }

    def test_reused_ids_observe_a_zero(self, monkeypatch, server):
        exporter = self._timed(monkeypatch)

        asyncio.run(
            api_server.chat_completions(_chat(prompt_token_ids=PROMPT_IDS), None)
        )

        samples = self._template_samples(exporter)
        assert server.templates == 0
        assert samples["atom:api_chat_template_seconds_count"] == 1
        assert samples["atom:api_chat_template_seconds_sum"] == 0

    def test_a_real_render_observes_its_own_wall_time(self, monkeypatch, server):
        exporter = self._timed(monkeypatch)

        asyncio.run(api_server.chat_completions(_chat(), None))

        samples = self._template_samples(exporter)
        assert server.templates == 1
        assert samples["atom:api_chat_template_seconds_count"] == 1
        assert samples["atom:api_chat_template_seconds_sum"] >= 0


class TestPrefillEchoesItsTokenIds:
    def test_chat_response_carries_the_ids(self, server):
        response = asyncio.run(
            api_server.chat_completions(_chat(return_token_ids=True), None)
        )

        assert server.generate[0][1]["return_token_ids"] is True
        assert response.prompt_token_ids == PROMPT_IDS

    def test_completion_response_carries_the_ids(self, server):
        response = asyncio.run(
            api_server.completions(
                CompletionRequest(model="m", prompt="hi", return_token_ids=True), None
            )
        )

        assert response.prompt_token_ids == PROMPT_IDS

    def test_not_asked_means_not_returned(self, server):
        response = asyncio.run(api_server.chat_completions(_chat(), None))

        assert server.generate[0][1]["return_token_ids"] is False
        assert response.prompt_token_ids is None


class TestUnanswerableRequestsForTokenIdsAreRejected:
    """Silently ignoring `return_token_ids` re-hides the cost it removes.

    A proxy that asks for ids and receives none falls back to sending text, so
    the decode node tokenizes again -- the original defect, restored, with no
    error anywhere to attribute it to. So a stream, which has no response body
    to put them on, is refused rather than answered with nothing.
    """

    def test_streaming_is_rejected(self, server):
        with pytest.raises(Exception, match="stream") as excinfo:
            asyncio.run(
                api_server.chat_completions(
                    _chat(return_token_ids=True, stream=True), None
                )
            )
        assert getattr(excinfo.value, "status_code", 400) == 400

    def test_guard_allows_the_answerable_shape(self):
        api_server._validate_return_token_ids(True, False)

    def test_guard_ignores_requests_that_did_not_ask(self):
        api_server._validate_return_token_ids(None, True)
        api_server._validate_return_token_ids(False, True)


class TestFanoutIsAnsweredRatherThanRefused:
    """`n > 1` has one prompt, so it has one answer.

    And refusing it would have been a trap, not a guard: atomesh injects
    `return_token_ids` itself and forwards the client's `n` untouched, so a
    client asking a PD pair for four samples would have received a 400 naming
    a field it never sent.
    """

    @staticmethod
    def _fanout_request() -> ChatCompletionRequest:
        request = _chat(return_token_ids=True, n=4)
        # Greedy collapses n back to 1, so this needs real sampling.
        request.temperature = 1.0
        return request

    def test_fanout_returns_the_shared_prompt_ids(self, monkeypatch, server):
        async def fake_fanout(prompt_or_tokens, _sampling, _rid, **kwargs):
            server.generate.append((prompt_or_tokens, kwargs))
            outputs = [
                {
                    "text": f"s{i}",
                    "finish_reason": "eos",
                    "num_tokens_input": 4,
                    "num_tokens_output": 1,
                }
                for i in range(4)
            ]
            if kwargs.get("return_token_ids"):
                # On the first only: one prompt, one list.
                outputs[0]["prompt_token_ids"] = list(PROMPT_IDS)
            return outputs

        monkeypatch.setattr(api_server, "generate_async_fanout", fake_fanout)
        response = asyncio.run(
            api_server.chat_completions(self._fanout_request(), None)
        )

        assert server.generate[0][1]["return_token_ids"] is True
        assert response.prompt_token_ids == PROMPT_IDS
        assert len(response.choices) == 4

    def test_fanout_builder_reads_the_first_output(self):
        outputs = [
            {
                "text": "a",
                "finish_reason": "eos",
                "num_tokens_input": 4,
                "num_tokens_output": 1,
                "prompt_token_ids": PROMPT_IDS,
            },
            {
                "text": "b",
                "finish_reason": "eos",
                "num_tokens_input": 4,
                "num_tokens_output": 1,
            },
        ]
        chat = build_chat_response_multi("chatcmpl-1", "m", outputs)
        completion = build_completion_response_multi("cmpl-1", "m", outputs)

        assert chat.prompt_token_ids == PROMPT_IDS
        assert completion.prompt_token_ids == PROMPT_IDS

    def test_fanout_builder_omits_ids_nobody_asked_for(self):
        outputs = [
            {
                "text": "a",
                "finish_reason": "eos",
                "num_tokens_input": 4,
                "num_tokens_output": 1,
            }
        ]
        assert build_chat_response_multi("c", "m", outputs).prompt_token_ids is None
        assert (
            build_completion_response_multi("c", "m", outputs).prompt_token_ids is None
        )


class TestGenerateAsyncReadsTheIdsOffTheLocalSequence:
    """The echoed ids come from the API process, not from the engine.

    ``preprocess`` runs here and returns the ``Sequence`` it tokenized, so the
    ids are already in this process. Asking the engine for them instead would
    put a list of tens of thousands of ints on the per-request ZMQ reply for
    the benefit of nobody.
    """

    @staticmethod
    def _engine_that_finishes_immediately(seq: _FakeSequence):
        def preprocess(_prompt, _sampling, stream_callback=None, **_kwargs):
            # The engine calls the callback from its own thread; `preprocess`
            # itself runs in an executor thread here, which is the same
            # arrangement from the event loop's point of view.
            stream_callback(
                SimpleNamespace(
                    output_tokens=[5],
                    finished=True,
                    finish_reason="eos",
                    num_cached_tokens=0,
                )
            )
            return seq

        return SimpleNamespace(
            io_processor=SimpleNamespace(preprocess=preprocess, requests={}),
            core_mgr=SimpleNamespace(add_request=lambda _seqs: None),
        )

    def _run(self, monkeypatch, *, return_token_ids: bool) -> dict:
        seq = _FakeSequence(PROMPT_IDS)
        monkeypatch.setattr(
            api_server, "engine", self._engine_that_finishes_immediately(seq)
        )
        monkeypatch.setattr(
            api_server, "_validate_sequence_context_length", lambda _s: None
        )
        monkeypatch.setattr(api_server, "delivered_text", lambda _ids: "hello")
        monkeypatch.setattr(api_server, "record_nonstream_first_token", lambda: None)

        async def collect() -> dict:
            final = None
            async for output in api_server.generate_async(
                PROMPT_IDS, object(), "req-1", return_token_ids=return_token_ids
            ):
                final = output
            return final

        return asyncio.run(collect())

    def test_ids_are_echoed_when_asked(self, monkeypatch):
        assert self._run(monkeypatch, return_token_ids=True)["prompt_token_ids"] == (
            PROMPT_IDS
        )

    def test_ids_are_absent_when_not_asked(self, monkeypatch):
        assert "prompt_token_ids" not in self._run(monkeypatch, return_token_ids=False)

    def test_prompt_length_does_not_retokenize_a_token_id_input(self, monkeypatch):
        # `num_tokens_input` used to fall back to `tokenizer.encode(prompt)`,
        # which a list of ids cannot be passed to.
        output = self._run(monkeypatch, return_token_ids=False)
        assert output["num_tokens_input"] == len(PROMPT_IDS)

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
import json
from types import SimpleNamespace
from uuid import UUID

import fastapi.routing
import httpx
import pytest
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from atom.entrypoints.openai import api_server
from atom.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
)
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


class TestDecodeDoesNotForwardDuplicateIdsToTheEngine:
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
        kwargs = {
            "kv_transfer_params": {**metadata, "prompt_token_ids": PROMPT_IDS},
            "stream": stream,
            "n": n,
        }
        if endpoint == "chat":
            request = _chat(**kwargs)
            request.temperature = 1.0
            handler = api_server.chat_completions
        else:
            request = CompletionRequest(model="m", temperature=1.0, **kwargs)
            handler = api_server.completions
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
    @pytest.mark.parametrize("nested_ids", [[], [-1], [999]])
    def test_bad_or_conflicting_ids_are_rejected_before_cleanup(
        self, server, endpoint, nested_ids
    ):
        kwargs = {
            "prompt_token_ids": PROMPT_IDS,
            "kv_transfer_params": {"prompt_token_ids": nested_ids},
        }
        if endpoint == "chat":
            request, handler = _chat(**kwargs), api_server.chat_completions
        else:
            request = CompletionRequest(model="m", **kwargs)
            handler = api_server.completions

        with pytest.raises(api_server.HTTPException) as excinfo:
            asyncio.run(handler(request, None))

        assert excinfo.value.status_code == 400
        assert server.generate == []
        assert request.kv_transfer_params["prompt_token_ids"] == nested_ids


class TestPrefillEchoesItsTokenIds:
    def test_chat_response_carries_the_ids(self, server):
        response = asyncio.run(
            api_server.chat_completions(_chat(return_token_ids=True), None)
        )

        assert server.generate[0][1]["return_token_ids"] is True
        assert json.loads(response.body)["prompt_token_ids"] == PROMPT_IDS

    def test_completion_response_carries_the_ids(self, server):
        response = asyncio.run(
            api_server.completions(
                CompletionRequest(model="m", prompt="hi", return_token_ids=True), None
            )
        )

        assert json.loads(response.body)["prompt_token_ids"] == PROMPT_IDS

    def test_not_asked_means_not_returned(self, server):
        response = asyncio.run(api_server.chat_completions(_chat(), None))

        assert server.generate[0][1]["return_token_ids"] is False
        assert json.loads(response.body)["prompt_token_ids"] is None


class TestHttpResponseSerialization:
    @pytest.mark.parametrize("endpoint", ["chat", "completion"])
    @pytest.mark.parametrize("return_ids", [False, True])
    @pytest.mark.parametrize("n", [1, 2])
    def test_wire_response_is_preserved_without_fastapi_recursive_conversion(
        self, monkeypatch, server, endpoint, return_ids, n
    ):
        # Exercise the real ASGI response boundary: returning a BaseModel here
        # would silently reintroduce a Python visit to every prompt token ID.
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
        if return_ids:
            output["prompt_token_ids"] = ids

        async def generate(*_args, **_kwargs):
            yield output

        async def fanout(*_args, **_kwargs):
            return [output] * n

        def build(*args, **kwargs):
            response = original_builder(*args, **kwargs)
            # The previous HTTP path defines the compatibility contract,
            # including nulls, Unicode, and JSON conversion of extension data.
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
        assert json.loads(response.body)["prompt_token_ids"] == PROMPT_IDS
        assert len(json.loads(response.body)["choices"]) == 4

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

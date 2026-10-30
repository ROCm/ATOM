"""Token-byte preservation and SSE probability attachment."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from atom.entrypoints.openai import api_server
from atom.entrypoints.openai.chat_logprobs import (
    LogprobCollector,
    LogprobTracker,
    TokenBytes,
    stream_logprobs,
)
from atom.entrypoints.openai.sse import data_frame, iter_sse_data
from atom.entrypoints.openai.streaming_dispatch import StreamOutputCollector
from atom.model_engine.token_logprobs import TokenLogprob

from .test_api_compatibility import post
from .test_api_compatibility import server as _server_fixture

server = _server_fixture


def codec():
    vocab = {
        0: "hello",
        1: "Ġworld",
        2: "ä",
        3: "¸",
        4: "Ń",
        5: "</think>",
        6: "40",
        7: "<eos>",
    }
    return TokenBytes(
        SimpleNamespace(
            backend_tokenizer=SimpleNamespace(
                decoder=SimpleNamespace(__getstate__=lambda: b'{"type":"ByteLevel"}')
            ),
            all_special_ids=[7],
            convert_ids_to_tokens=lambda i: vocab[i],
        )
    )


def rec(rank=1):
    return TokenLogprob(-0.25, rank, ((0, -0.25), (1, -1.5)))


def test_bytes_across_chinese_tokens_and_special_eos():
    tracker = LogprobTracker(codec(), 1)
    tracker.feed([2, 3, 4, 7], [rec()] * 4)
    rows = tracker.take("中")
    assert len(rows) == 3
    assert bytes(b for row in rows for b in row["bytes"]) == "中".encode()
    assert rows[0]["bytes"] == [0xE4]
    assert all(len(row["top_logprobs"]) == 1 for row in rows)


def test_reasoning_delimiter_and_rank_sentinel():
    tracker = LogprobTracker(codec(), 0)
    tracker.feed([0, 5, 6], [rec(), rec(), rec(21)])
    assert tracker.take("hello")[0]["logprob"] == -0.25
    rows = tracker.take("40")
    assert len(rows) == 1 and rows[0]["logprob"] == -9999.0
    assert rows[0]["top_logprobs"] == []


def test_truncated_utf8_keeps_original_token_bytes():
    tracker = LogprobTracker(codec())
    tracker.feed([2], [rec()])
    rows = tracker.take("\ufffd")
    assert rows[0]["bytes"] == [0xE4]
    assert rows[0]["logprob"] == -0.25


def test_coalescing_keeps_probabilities_and_choice_identity():
    async def run():
        source = StreamOutputCollector()
        source.put_nowait((1, {"token_ids": [0], "logprobs": [rec()], "text": "hello"}))
        source.put_nowait(
            (
                1,
                {
                    "token_ids": [1],
                    "logprobs": [rec()],
                    "text": " world",
                    "finished": True,
                },
            )
        )
        wrapper = LogprobCollector(source, codec(), 2, n=2)
        index, chunk = await wrapper.get()
        assert index == 1 and len(chunk["logprobs"]) == 2
        assert wrapper.trackers[0].spans == []

        async def frames():
            yield data_frame(
                {
                    "choices": [
                        {
                            "index": 1,
                            "delta": {"content": "hello world"},
                            "logprobs": None,
                        }
                    ]
                }
            )
            yield "data: [DONE]\n\n"

        result = [x async for x in stream_logprobs(frames(), wrapper.trackers)]
        value = json.loads(next(iter_sse_data(result[0])))
        rows = value["choices"][0]["logprobs"]["content"]
        assert len(rows) == 2
        assert bytes(b for row in rows for b in row["bytes"]) == b"hello world"
        assert result[-1] == "data: [DONE]\n\n"

    asyncio.run(run())


def test_missing_probability_does_not_succeed_silently():
    tracker = LogprobTracker(codec())
    with pytest.raises(RuntimeError, match="Missing"):
        tracker.feed([0], None)


def test_stream_tracker_releases_consumed_records_and_preserves_partial_tokens():
    tracker = LogprobTracker(codec())
    for _ in range(100):
        tracker.feed([0], [rec()])
        assert tracker.take("hell") == []
        assert len(tracker.take("o")) == 1
        assert not tracker.raw and not tracker.spans


def test_non_bytelevel_tokenizer_is_explicitly_rejected():
    with pytest.raises(ValueError, match="ByteLevel"):
        TokenBytes(SimpleNamespace())


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("top_k", [0, 1, 2])
def test_chat_route_transports_actual_records(server, monkeypatch, stream, top_k):
    from atom.entrypoints.openai import chat_logprobs

    monkeypatch.setattr(chat_logprobs, "token_bytes", lambda tokenizer: codec())
    server.chunks = [
        {
            "text": "hello",
            "token_ids": [0],
            "logprobs": [rec()],
            "finished": True,
            "finish_reason": "eos",
        }
    ]

    async def generate(prompt, params, *args, **kwargs):
        assert params.logprobs and params.top_logprobs == top_k
        yield {
            "text": "hello",
            "token_ids": [0],
            "logprobs": [rec()],
            "finish_reason": "eos",
            "num_tokens_input": 2,
            "num_tokens_output": 1,
        }

    monkeypatch.setattr(api_server, "generate_async", generate)
    response = post(
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "none",
            "logprobs": True,
            "top_logprobs": top_k,
            "stream": stream,
        },
    )
    assert response.status_code == 200, response.text
    if stream:
        chunks = [
            json.loads(data)
            for data in iter_sse_data(response.text)
            if data != "[DONE]"
        ]
        rows = [
            r
            for chunk in chunks
            for choice in chunk.get("choices", [])
            for r in (choice.get("logprobs") or {}).get("content", [])
        ]
    else:
        rows = response.json()["choices"][0]["logprobs"]["content"]
    assert len(rows) == 1 and rows[0]["logprob"] == -0.25
    assert len(rows[0]["top_logprobs"]) == top_k


@pytest.mark.parametrize("stream", [False, True])
def test_top_logprobs_dependency_is_rejected_before_work(server, stream):
    response = post(
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": "hi"}],
            "top_logprobs": 1,
            "stream": stream,
        },
    )
    assert response.status_code == 400
    assert "logprobs=true" in response.json()["error"]["message"]
    assert not server.generations and not server.templates

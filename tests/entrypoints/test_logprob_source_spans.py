"""Probabilities follow parser source spans, including discarded/identical text."""

import json
from types import SimpleNamespace

import pytest

from atom.entrypoints.openai.logprobs import LogprobsProcessor
from atom.entrypoints.openai.reasoning import NO_REASONING
from atom.entrypoints.openai.serving_chat import _build_chat_choice, create_chat_chunk
from atom.entrypoints.openai.sse import iter_sse_data
from atom.entrypoints.openai.tool_parser import ToolCallStreamParser
from atom.entrypoints.openai.tool_parser.deepseekv41_tool_parser import DsmlV41Parser
from atom.model_engine.token_logprobs import TokenLogprob


def response_logprobs(response, outputs, codec, top_k, **kwargs):
    response.choices = [
        _build_chat_choice(
            "",
            "stop",
            index=i,
            logprob_codec=codec,
            token_ids=output["token_ids"],
            token_logprobs=output["logprobs"],
            top_logprobs=top_k,
            **kwargs,
        )
        for i, output in enumerate(outputs)
    ]
    return response


class Codec:
    special = frozenset()

    def raw(self, token):
        return chr(token).encode()

    def candidate(self, token, logprob):
        return {"token": chr(token), "bytes": list(self.raw(token)), "logprob": logprob}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "f",
            "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
        },
    }
]
CALL = '<calls><invoke name="f"><parameter name="x">right</parameter></invoke></calls>'


def records(text):
    return [TokenLogprob(-float(i + 1), 1, ()) for i in range(len(text))]


@pytest.mark.parametrize(
    "text,answer,thought",
    [
        ("before<think>reason</think>after", "beforeafter", "reason"),
        ("left" + CALL + "right", "leftright", None),
        ("<calls>not actually a tool", "<calls>not actually a tool", None),
        ("Hello .", "Hello .", None),
    ],
)
def test_nonstream_joins_source_spans_without_search(text, answer, thought):
    response = SimpleNamespace(choices=[{"message": {"content": "old"}}])
    returned = response_logprobs(
        response,
        [{"token_ids": list(map(ord, text)), "logprobs": records(text)}],
        Codec(),
        0,
        reasoning=NO_REASONING,
        tools=TOOLS,
        tool_parser_cls=DsmlV41Parser,
    )
    assert returned is response
    choice = response.choices[0]
    assert choice["message"]["content"] == answer
    assert choice["message"].get("reasoning_content") == thought
    assert "".join(r["token"] for r in choice["logprobs"]["content"]) == answer
    if text.endswith("right"):
        assert choice["logprobs"]["content"][-1]["logprob"] == -len(text)


@pytest.mark.parametrize("size", [1, 3, 17, 1000])
def test_stream_hidden_repeated_text_uses_visible_occurrence(size):
    text = "left" + CALL + "right"
    tracker = LogprobsProcessor(Codec())
    reader = NO_REASONING.stream()
    parser = ToolCallStreamParser(tools=TOOLS, parser_cls=DsmlV41Parser)
    rows, visible = [], []
    all_records = records(text)
    for start in range(0, len(text), size):
        part = text[start : start + size]
        final = start + size >= len(text)
        source = tracker.feed(
            list(map(ord, part)), all_records[start : start + size], finished=final
        )
        segments = reader.process(source) + (reader.flush() if final else [])
        events = []
        for field, value in segments:
            assert field == "content"
            events.extend(parser.process(value))
        if final:
            events.extend(parser.flush())
        for kind, value in events:
            if kind == "content":
                frame = create_chat_chunk(
                    "req", "model", {"content": value}, with_logprobs=True
                )
                choice = json.loads(next(iter_sse_data(frame)))["choices"][0]
                visible.append(choice["delta"]["content"])
                rows.extend(choice["logprobs"]["content"])
    assert "".join(visible) == "leftright"
    assert [r["logprob"] for r in rows] == [
        -float(i + 1) for i in [*range(4), *range(len(text) - 5, len(text))]
    ]
    tracker.close()


def test_claiming_later_text_keeps_buffered_earlier_records():
    tracker = LogprobsProcessor(Codec(), 20)
    record = TokenLogprob(-1.0, 1, ((97, -1.0),))
    first = tracker.feed([97], [record])
    last = tracker.feed([98], [record])
    assert tracker.take(last)[0]["token"] == "b"
    assert tracker.take(first)[0]["token"] == "a"
    assert tracker.take(first) == []
    tracker.close()
    assert tracker.store.records == []


def test_truncated_multibyte_source_keeps_original_bytes():
    codec = SimpleNamespace(
        special=set(),
        raw=lambda _: b"\xe4",
        candidate=lambda t, p: {"bytes": [228], "token": "\ufffd", "logprob": p},
    )
    tracker = LogprobsProcessor(codec)
    assert tracker.feed([0], [TokenLogprob(-0.25, 1)]) == ""
    source = tracker.feed([], None, finished=True)
    assert source == "\ufffd"
    assert tracker.take(source)[0]["bytes"] == [228]
    tracker.close()


def test_logprob_detokenization_preserves_punctuation_spaces():
    text = "Hello ."
    tracker = LogprobsProcessor(Codec())
    assert tracker.feed(list(map(ord, text)), records(text), finished=True) == text


@pytest.mark.parametrize("traced", [False, True])
@pytest.mark.parametrize("starts_open", [False, True])
def test_schema_literal_answer_preserves_json_and_logprob_source(traced, starts_open):
    from atom.entrypoints.openai.logprobs import LogprobText
    from atom.entrypoints.openai.reasoning import ReasoningChannel

    answer = '{"text":"<think>literal</think>"}'
    prefix = "reason</think>" if starts_open else ""
    text = prefix + answer
    owner = object()
    source = LogprobText([(text, 0, len(text))], owner) if traced else text
    reasoning, content = ReasoningChannel(
        starts_open=starts_open, literal_answer=True
    ).split(source)
    assert reasoning == ("reason" if starts_open else None)
    assert content == answer
    if traced:
        assert isinstance(content, LogprobText)
        assert content.owner is owner
        assert content.source_ranges == [(len(prefix), len(text))]


@pytest.mark.parametrize(
    "pieces",
    [
        [b".</", b"think>", b"ok"],
        [b".</think>ok"],
        [b"a<think>b</think>c"],
        [b"left" + CALL.encode() + b"right"],
    ],
)
@pytest.mark.parametrize("size", [1, 100])
def test_boundary_tokens_owned_once_in_json_and_stream(pieces, size):
    from atom.entrypoints.openai.reasoning import ReasoningChannel

    class ByteCodec:
        special = frozenset()

        def raw(self, token):
            return pieces[token]

        def candidate(self, token, logprob):
            return {
                "token": pieces[token].decode(),
                "bytes": list(pieces[token]),
                "logprob": logprob,
            }

    channel = ReasoningChannel(starts_open=pieces[0].startswith(b"."))
    probabilities = [TokenLogprob(-float(i + 1), 1, ()) for i in range(len(pieces))]
    response = SimpleNamespace(choices=[{"message": {}}])
    response_logprobs(
        response,
        [{"token_ids": list(range(len(pieces))), "logprobs": probabilities}],
        ByteCodec(),
        0,
        reasoning=channel,
        tools=TOOLS,
        tool_parser_cls=DsmlV41Parser,
    )
    expected = response.choices[0]["logprobs"]
    tracker = LogprobsProcessor(ByteCodec())
    reader = channel.stream()
    parser = ToolCallStreamParser(tools=TOOLS, parser_cls=DsmlV41Parser)
    actual = {key: [] for key in expected}
    visible = {key: "" for key in expected}

    def emit(field, text):
        visible[field] += str(text)
        actual[field].extend(tracker.take(text))

    for start in range(0, len(pieces), size):
        stop = min(start + size, len(pieces))
        final = stop == len(pieces)
        text = tracker.feed(
            list(range(start, stop)), probabilities[start:stop], finished=final
        )
        for field, value in reader.process(text) + (reader.flush() if final else []):
            if field == "content":
                for kind, part in parser.process(value):
                    if kind == "content":
                        emit(field, part)
            else:
                emit(field, value)
        if final:
            for kind, part in parser.flush():
                if kind == "content":
                    emit("content", part)
    assert actual == expected
    assert visible == {key: response.choices[0]["message"][key] for key in expected}
    claimed = [entry["logprob"] for entries in actual.values() for entry in entries]
    assert len(claimed) == len(set(claimed))
    assert -1.0 in claimed
    tracker.close()


@pytest.mark.parametrize("tag", [None, 1])
def test_raw_logprob_dispatch_coalesces_without_ordinary_decode(tag):
    import asyncio

    from atom.entrypoints.openai.logprobs import LogprobCollector
    from atom.entrypoints.openai.streaming_dispatch import (
        StreamBatchDispatcher,
        StreamOutputCollector,
    )

    async def run():
        # No tokenizer methods exist: constructing/using an ordinary decoder fails.
        dispatcher = StreamBatchDispatcher(object())
        state = dispatcher.new_state(decode=False)
        assert state.detokenizer is None
        source = StreamOutputCollector()
        wrapper = LogprobCollector(source, Codec(), 0, n=2)
        loop = asyncio.get_running_loop()
        for token in (97, 98):
            dispatcher.enqueue(
                loop=loop,
                collector=source,
                state=state,
                tag=tag,
                chunk={
                    "token_ids": [token],
                    "logprobs": [TokenLogprob(-1, 1, ())],
                    "finished": token == 98,
                },
            )
        dispatcher.flush()
        await asyncio.sleep(0)
        payload = await wrapper.get()
        index, chunk = payload if isinstance(payload, tuple) else (0, payload)
        assert index == (tag or 0)
        assert chunk["text"] == "ab"
        assert chunk["token_ids"] == [97, 98]
        assert chunk["finished"]
        assert len(wrapper.trackers[index].take(chunk["text"])) == 2
        for tracker in wrapper.trackers:
            tracker.close()

    asyncio.run(run())


@pytest.mark.parametrize("fanout", [False, True])
def test_buffered_older_content_owns_shared_token(fanout):
    import asyncio

    from atom.entrypoints.openai.logprobs import LogprobCollector
    from atom.entrypoints.openai.serving_chat import (
        stream_chat_response,
        stream_chat_response_fanout,
    )
    from atom.entrypoints.openai.streaming_dispatch import StreamOutputCollector

    text = "<calls>literal<think>reason</think>end"
    codec = SimpleNamespace(
        special=set(),
        raw=lambda _: text.encode(),
        candidate=lambda t, p: {
            "token": text,
            "bytes": list(text.encode()),
            "logprob": p,
        },
    )
    expected = _build_chat_choice(
        "",
        "stop",
        tools=TOOLS,
        tool_parser_cls=DsmlV41Parser,
        logprob_codec=codec,
        token_ids=[0],
        token_logprobs=[TokenLogprob(-0.25, 1)],
    )

    async def run():
        source = StreamOutputCollector()
        chunk = {
            "token_ids": [0],
            "logprobs": [TokenLogprob(-0.25, 1)],
            "finished": False,
        }
        source.put_nowait((0, chunk) if fanout else chunk)
        final = {"token_ids": [], "finished": True, "finish_reason": "stop"}
        source.put_nowait((0, final) if fanout else final)
        collector = LogprobCollector(source, codec, 0)
        kwargs = {
            "request_id": "req",
            "model": "model",
            "num_prompt_tokens": 1,
            "cleanup_stream": lambda *a, **kw: None,
            "cleanup_request": lambda *a: None,
            "tools": TOOLS,
            "tool_parser_cls": DsmlV41Parser,
        }
        gen = (
            stream_chat_response_fanout(
                shared_collector=collector, seq_ids=[1], **kwargs
            )
            if fanout
            else stream_chat_response(stream_collector=collector, seq_id=1, **kwargs)
        )
        frames = [frame async for frame in gen]
        assert collector.trackers[0].store.records == []
        actual = {"content": [], "reasoning_content": []}
        for frame in frames:
            for data in iter_sse_data(frame):
                if data == "[DONE]":
                    continue
                for choice in json.loads(data).get("choices", []):
                    for field, rows in (choice.get("logprobs") or {}).items():
                        actual[field].extend(rows)
        assert actual == expected["logprobs"]
        assert len(actual["content"]) == 1
        assert not actual["reasoning_content"]

    asyncio.run(run())


def test_bad_logprobs_terminate_only_the_failing_fanout_choice():
    import asyncio

    from atom.entrypoints.openai.logprobs import LogprobCollector
    from atom.entrypoints.openai.serving_chat import stream_chat_response_fanout
    from atom.entrypoints.openai.streaming_dispatch import StreamOutputCollector

    async def run():
        source = StreamOutputCollector()
        source.put_nowait((0, {"token_ids": [97], "logprobs": [], "finished": True}))
        source.put_nowait(
            (
                1,
                {
                    "token_ids": [98],
                    "logprobs": [TokenLogprob(-1, 1)],
                    "finished": True,
                    "finish_reason": "stop",
                },
            )
        )
        collector = LogprobCollector(source, Codec(), 0, 2)
        cleaned = []
        gen = stream_chat_response_fanout(
            "req",
            "model",
            collector,
            [11, 12],
            1,
            lambda seq, **kw: cleaned.append((seq, kw)),
            lambda _: None,
        )
        frames = [frame async for frame in gen]
        payloads = [
            json.loads(data)
            for frame in frames
            for data in iter_sse_data(frame)
            if data != "[DONE]"
        ]
        assert any(p.get("error", {}).get("index") == 0 for p in payloads)
        assert any(
            c["index"] == 1 and c["delta"].get("content") == "b"
            for p in payloads
            for c in p.get("choices", [])
        )
        assert frames[-1].endswith("data: [DONE]\n\n")
        assert (11, {"aborted": True}) in cleaned

    asyncio.run(run())


def test_added_special_tokens_and_padded_vocab_candidates():
    from atom.entrypoints.openai.logprobs import TokenBytes

    tokenizer = SimpleNamespace(
        backend_tokenizer=SimpleNamespace(
            decoder=SimpleNamespace(__getstate__=lambda: b'{"type":"ByteLevel"}')
        ),
        all_special_ids=[7],
        added_tokens_decoder={8: SimpleNamespace(special=True)},
        convert_ids_to_tokens=lambda i: {0: "hello", 7: "eos", 8: "pad"}.get(i),
    )
    codec = TokenBytes(tokenizer)
    tracker = LogprobsProcessor(codec, 1)
    text = tracker.feed(
        [8, 0, 7], [TokenLogprob(-1, 1, ((99, -2),))] * 3, finished=True
    )
    assert text == "hello"
    assert tracker.take(text)[0]["top_logprobs"] == [
        {"token": "", "bytes": [], "logprob": -2.0}
    ]


def test_bad_raw_registration_preserves_other_buffered_chunks():
    import asyncio

    from atom.entrypoints.openai.streaming_dispatch import (
        StreamBatchDispatcher,
        StreamOutputCollector,
    )

    async def run():
        dispatcher = StreamBatchDispatcher(SimpleNamespace())
        source = StreamOutputCollector()
        loop = asyncio.get_running_loop()
        dispatcher.enqueue(
            loop=loop,
            collector=source,
            state=dispatcher.new_state(decode=False),
            chunk={"token_ids": [97]},
        )
        with pytest.raises(ValueError, match="Raw streams"):
            dispatcher.enqueue(
                loop=loop,
                collector=asyncio.Queue(),
                state=dispatcher.new_state(decode=False),
                chunk={},
            )
        dispatcher.flush()
        await asyncio.sleep(0)
        assert (await source.get())["token_ids"] == [97]

    asyncio.run(run())


@pytest.mark.parametrize(
    "option,value,kwargs",
    [
        ("enable_rapidserve", True, {}),
        ("kv_transfer_config", object(), {}),
        ("speculative_config", SimpleNamespace(synthetic_acceptance_rates=[]), {}),
        ("unused", None, {"kv_transfer": {"id": 1}}),
        ("unused", None, {"multimodal": True}),
    ],
)
def test_logprob_execution_admission_is_shared(option, value, kwargs):
    from atom.sampling_params import SamplingParams

    config = SimpleNamespace(**{option: value})
    with pytest.raises(ValueError, match="logprobs"):
        SamplingParams(logprobs=True).validate_logprob_options(config, **kwargs)
    SamplingParams(logprobs=False).validate_logprob_options(config, **kwargs)

"""Probabilities follow parser source spans, including discarded/identical text."""

import json
from types import SimpleNamespace

import pytest

from atom.entrypoints.openai.logprobs import LogprobsProcessor, claim_logprobs
from atom.entrypoints.openai.reasoning import NO_REASONING
from atom.entrypoints.openai.serving_chat import _build_chat_choice, create_chat_chunk
from atom.entrypoints.openai.sse import iter_sse_data
from atom.entrypoints.openai.tool_parser import ToolCallStreamParser
from atom.entrypoints.openai.tool_parser.deepseekv41_tool_parser import DsmlV41Parser
from tests.logprob_helpers import token_logprob


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
    return [token_logprob(-float(i + 1), 1, ()) for i in range(len(text))]


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
        segments = reader.process(source) + (
            reader.flush(stopped=True) if final else []
        )
        events = []
        for field, value in segments:
            assert field == "content"
            events.extend(parser.process(value))
        if final:
            events.extend(parser.flush())
        for kind, value in events:
            if kind == "content":
                frame = create_chat_chunk(
                    "req",
                    "model",
                    {"content": value},
                    with_logprobs=True,
                    logprob_values=claim_logprobs([{"content": value}])[0],
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
    record = token_logprob(-1.0, 1, ((97, -1.0),))
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
    assert tracker.feed([0], [token_logprob(-0.25, 1)]) == ""
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
    ).split(source, stopped=True)
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
    probabilities = [token_logprob(-float(i + 1), 1, ()) for i in range(len(pieces))]
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
        for field, value in reader.process(text) + (
            reader.flush(stopped=True) if final else []
        ):
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
                    "logprobs": [token_logprob(-1, 1, ())],
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
        token_logprobs=[token_logprob(-0.25, 1)],
    )

    async def run():
        source = StreamOutputCollector()
        chunk = {
            "token_ids": [0],
            "logprobs": [token_logprob(-0.25, 1)],
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
                    "logprobs": [token_logprob(-1, 1)],
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
        assert not any(
            c["index"] == 0 and c.get("finish_reason")
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
        get_vocab=lambda: {"hello": 0, "eos": 7, "pad": 8},
        added_tokens_decoder={8: SimpleNamespace(special=True)},
        convert_ids_to_tokens=lambda i: {0: "hello", 7: "eos", 8: "pad"}.get(i),
    )
    codec = TokenBytes(tokenizer)
    tracker = LogprobsProcessor(codec, 1)
    text = tracker.feed(
        [8, 0, 7], [token_logprob(-1, 1, ((99, -2),))] * 3, finished=True
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


@pytest.mark.parametrize("size", [1, 1000, 8192])
@pytest.mark.parametrize("answer", ["Paris.", "a" * 4100])
@pytest.mark.parametrize("finish_reason", ["stop", "max_tokens"])
def test_kimi_replayed_answer_keeps_json_sse_probabilities(size, answer, finish_reason):
    import asyncio

    from atom.entrypoints.openai.logprobs import LogprobCollector
    from atom.entrypoints.openai.reasoning import ReasoningChannel
    from atom.entrypoints.openai.reasoning_dialects import resolve_dialect
    from atom.entrypoints.openai.serving_chat import stream_chat_response

    dialect, _ = resolve_dialect("", "<|open|>think<|sep|>")
    reasoning = ReasoningChannel(dialect=dialect, starts_open=True)
    text = answer + "<|close|>response<|sep|>"
    probabilities = records(text)
    expected = _build_chat_choice(
        "",
        finish_reason,
        reasoning=reasoning,
        logprob_codec=Codec(),
        token_ids=list(map(ord, text)),
        token_logprobs=probabilities,
    )

    async def run():
        class Source:
            def __init__(self):
                self.at = 0

            async def get(self):
                at = self.at
                self.at += size
                return {
                    "token_ids": list(map(ord, text[at : at + size])),
                    "logprobs": probabilities[at : at + size],
                    "finished": self.at >= len(text),
                    "finish_reason": finish_reason if self.at >= len(text) else None,
                }

        collector = LogprobCollector(Source(), Codec(), 0)
        frames = [
            frame
            async for frame in stream_chat_response(
                "req",
                "model",
                collector,
                1,
                1,
                lambda *a, **kw: None,
                lambda *a: None,
                reasoning=reasoning,
            )
        ]
        actual = {"content": [], "reasoning_content": []}
        texts = {"content": "", "reasoning_content": ""}
        for frame in frames:
            for data in iter_sse_data(frame):
                if data == "[DONE]":
                    continue
                obj = json.loads(data)
                assert "error" not in obj, obj
                for choice in obj.get("choices", []):
                    for field in texts:
                        texts[field] += choice["delta"].get(field) or ""
                        actual[field].extend(
                            (choice.get("logprobs") or {}).get(field, [])
                        )
        assert texts["content"] == expected["message"]["content"]
        assert actual == expected["logprobs"]
        assert len(actual["reasoning_content"]) == len(answer)
        assert not actual["content"]

    asyncio.run(run())


def test_lost_source_is_an_internal_error_for_nonstream_too():
    from atom.entrypoints.openai.logprobs import LogprobError

    tracker = LogprobsProcessor(Codec())
    with pytest.raises(LogprobError, match="source spans"):
        tracker.take_fields({"content": "plain text"})


@pytest.mark.parametrize("position", [0, 1, 2])
def test_zero_width_tokens_at_visible_span_boundaries(position):
    class EmptyCodec(Codec):
        def raw(self, token):
            return b"" if token == 0 else super().raw(token)

    tracker = LogprobsProcessor(EmptyCodec())
    ids = [97, 98]
    ids.insert(position, 0)
    text = tracker.feed(ids, [token_logprob(-1, 1)] * 3, finished=True)
    assert text == "ab"
    assert len(tracker.take(text)) == 3


def test_announced_tool_name_is_not_completed_arguments():
    from atom.entrypoints.openai.serving_chat import _chat_deltas

    call = {"function": {"name": "f", "arguments": ""}}
    parser = SimpleNamespace(
        process=lambda text: [("tool_call_start", call)], flush=list
    )
    events = list(_chat_deltas(NO_REASONING.stream(), parser, "x", True, "stop"))
    assert events == [("tool_call_start", {"tool_calls": [call]})]
    assert not any(kind == "tool_call_args" for kind, _ in events)


def test_failed_choice_has_no_normal_finish_frame():
    from atom.entrypoints.openai.serving_chat import chat_finish_frames

    frame = chat_finish_frames(
        "r", "m", [None, "stop"], {"completion_tokens": 1}, usage_on_finish=True
    )
    payloads = [json.loads(data) for data in iter_sse_data(frame) if data != "[DONE]"]
    assert [c["index"] for p in payloads for c in p.get("choices", [])] == [1]
    assert payloads[-1]["usage"] == {"completion_tokens": 1}


def test_open_tool_region_releases_resolved_prefix_and_name():
    from functools import partial

    from atom.entrypoints.openai.serving_chat import _logprob_frames

    create_chunk = partial(create_chat_chunk, with_logprobs=True)
    tracker = LogprobsProcessor(Codec())
    text = tracker.feed(list(map(ord, "abcXYZ")), records("abcXYZ"))
    pending = [
        {"content": text[:3]},
        {"tool_calls": [{"function": {"name": "f", "arguments": ""}}]},
        {"reasoning_content": text[4:]},
    ]
    frames = _logprob_frames(pending, create_chunk, "r", "m", held_text=text[3:4])
    assert len(frames) == 2
    assert pending == [{"reasoning_content": text[4:]}]
    assert len(_logprob_frames(pending, create_chunk, "r", "m")) == 1
    assert pending == []


@pytest.mark.parametrize("cleanup", [False, True])
def test_token_byte_table_is_built_once_without_changing_cleanup(cleanup):
    from atom.entrypoints.openai.logprobs import TokenBytes

    calls = []

    def vocab():
        calls.append(1)
        return {"Ġhello": 0, "!": 1}

    tokenizer = SimpleNamespace(
        backend_tokenizer=SimpleNamespace(
            decoder=SimpleNamespace(__getstate__=lambda: b'{"type":"ByteLevel"}')
        ),
        all_special_ids=[],
        get_vocab=vocab,
        clean_up_tokenization_spaces=cleanup,
    )
    if cleanup:
        with pytest.raises(ValueError, match="whitespace cleanup"):
            TokenBytes(tokenizer)
        assert tokenizer.clean_up_tokenization_spaces
        assert not calls
    else:
        codec = TokenBytes(tokenizer)
        for _ in range(9000):
            assert codec.raw(0) == b" hello"
            assert codec.raw(99) == b""
        assert calls == [1]


@pytest.mark.parametrize("size", [1, 3, 1000])
def test_disabled_tool_parser_never_joins_source_spans(monkeypatch, size):
    from atom.entrypoints.openai.tool_parser import stream

    def forbidden(*args, **kwargs):
        raise AssertionError("disabled requests must not process logprob spans")

    monkeypatch.setattr(stream, "join_text", forbidden)
    text = "left" + CALL + "right"
    parser = ToolCallStreamParser(tools=TOOLS, parser_cls=DsmlV41Parser)
    events = []
    for start in range(0, len(text), size):
        events.extend(parser.process(text[start : start + size]))
    events.extend(parser.flush())
    content, calls = stream.flatten_tool_events(events)
    assert content == "leftright"
    assert len(calls) == 1
    assert calls[0].function["arguments"] == '{"x": "right"}'


@pytest.mark.parametrize("prefix", ["", "x"])
def test_zero_width_token_survives_separate_stream_chunk(prefix):
    class EmptyCodec(Codec):
        def raw(self, token):
            return b"" if token == 0 else super().raw(token)

    tracker = LogprobsProcessor(EmptyCodec())
    if prefix:
        text = tracker.feed([ord(prefix)], [token_logprob(-1, 1)])
        assert len(tracker.take(text)) == 1
        tracker.discard_resolved("")
    assert tracker.feed([0], [token_logprob(-2, 1)]) == ""
    tracker.discard_resolved("")
    text = tracker.feed([97], [token_logprob(-3, 1)], finished=True)
    result = tracker.take(text)
    assert [row["bytes"] for row in result] == [[], [97]]
    assert [row["logprob"] for row in result] == [-2, -3]


def test_slices_and_joins_reuse_byte_indices(monkeypatch):
    from atom.entrypoints.openai.logprobs import LogprobText, _ByteOffsets, join_text

    text = "中a🙂" * 10000
    source = LogprobText([(text, 11, 11 + len(text.encode()))], object())

    def forbidden(*args):
        raise AssertionError("a slice must reuse its original byte index")

    monkeypatch.setattr(_ByteOffsets, "__init__", forbidden)
    for start in range(0, 3000, 3):
        piece = source[start : start + 3][1:]
        assert piece == "a🙂"
        assert piece.source_ranges == [
            (11 + start // 3 * 8 + 3, 11 + start // 3 * 8 + 8)
        ]
    joined = join_text([source[:3], source[-3:]])
    assert joined[2:4] == "🙂中"
    assert joined[2:4].source_ranges == [(15, 19), (11 + 9999 * 8, 14 + 9999 * 8)]
    assert source[4:2].source_ranges == []

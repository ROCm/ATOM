"""Probabilities follow parser source spans, including discarded/identical text."""

import json
from types import SimpleNamespace

import pytest

from atom.entrypoints.openai.chat_logprobs import LogprobTracker, response_logprobs
from atom.entrypoints.openai.reasoning import NO_REASONING
from atom.entrypoints.openai.serving_chat import create_chat_chunk
from atom.entrypoints.openai.sse import iter_sse_data
from atom.entrypoints.openai.streaming_dispatch import IncrementalStreamDetokenizer
from atom.entrypoints.openai.tool_parser import ToolCallStreamParser
from atom.entrypoints.openai.tool_parser.deepseekv41_tool_parser import DsmlV41Parser
from atom.model_engine.token_logprobs import TokenLogprob


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
    response_logprobs(
        response,
        [{"token_ids": list(map(ord, text)), "logprobs": records(text)}],
        Codec(),
        0,
        reasoning=NO_REASONING,
        tools=TOOLS,
        tool_parser_cls=DsmlV41Parser,
    )
    choice = response.choices[0]
    assert choice["message"]["content"] == answer
    assert choice["message"].get("reasoning_content") == thought
    assert "".join(r["token"] for r in choice["logprobs"]["content"]) == answer
    if text.endswith("right"):
        assert choice["logprobs"]["content"][-1]["logprob"] == -len(text)


@pytest.mark.parametrize("size", [1, 3, 17, 1000])
def test_stream_hidden_repeated_text_uses_visible_occurrence(size):
    text = "left" + CALL + "right"
    tracker = LogprobTracker(Codec())
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


def test_long_undecided_tool_records_spill_without_an_in_memory_index():
    tracker = LogprobTracker(Codec(), 20)
    record = TokenLogprob(-1.0, 1, tuple((97, -float(i + 1)) for i in range(20)))
    for _ in range(3000):
        last = tracker.feed([97], [record])
    assert tracker.store.file._rolled
    assert not hasattr(tracker, "spans") and not hasattr(tracker, "raw")
    assert tracker.take(last)[0]["logprob"] == -1.0
    tracker.close()
    assert tracker.store.file.closed


def test_truncated_multibyte_source_keeps_original_bytes():
    codec = SimpleNamespace(
        special=set(),
        raw=lambda _: b"\xe4",
        candidate=lambda t, p: {"bytes": [228], "token": "\ufffd", "logprob": p},
    )
    tracker = LogprobTracker(codec)
    assert tracker.feed([0], [TokenLogprob(-0.25, 1)]) == ""
    source = tracker.feed([], None, finished=True)
    assert source == "\ufffd"
    assert tracker.take(source)[0]["bytes"] == [228]
    tracker.close()


def test_logprob_detokenization_preserves_punctuation_spaces():
    calls = []

    class Tokenizer:
        def decode(
            self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        ):
            calls.append(clean_up_tokenization_spaces)
            text = "".join(map(chr, ids))
            return text.replace(" .", ".") if clean_up_tokenization_spaces else text

    decoder = IncrementalStreamDetokenizer(
        Tokenizer(), clean_up_tokenization_spaces=False
    )
    assert decoder.update(list(map(ord, "Hello .")), True) == "Hello ."
    assert calls and not any(calls)

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""A stop string ends the reply where it appears in the text, and is not part of it."""

import asyncio
from types import SimpleNamespace

import pytest

from atom.entrypoints.openai import api_server
from atom.entrypoints.openai.protocol import ChatCompletionRequest, CompletionRequest
from atom.entrypoints.openai.stop_strings import (
    StopStrings,
    check_stop_strings,
    stop_list,
)
from atom.entrypoints.openai.streaming_dispatch import (
    IncrementalStreamDetokenizer,
    StreamOutputCollector,
)


class _Chars:
    """One token per character, so a stop string can span any token boundary."""

    def decode(self, ids, skip_special_tokens=True):
        return "".join(map(chr, ids))


def _chunk(text, finished=False, reason=None):
    return {
        "token_ids": [ord(c) for c in text],
        "finished": finished,
        "finish_reason": reason,
    }


def test_stop_takes_one_string_or_a_list():
    assert CompletionRequest(prompt="p", stop="```").stop == "```"
    assert (
        ChatCompletionRequest(
            messages=[{"role": "user", "content": "q"}], stop="```"
        ).stop
        == "```"
    )
    assert stop_list(None) == []
    assert stop_list("```") == ["```"]
    assert stop_list(["a", "", "b"]) == ["a", "b"]


def test_text_is_cut_before_the_earliest_stop_string():
    stop = StopStrings(["```", "STOP"])
    assert (stop.feed("a STOP b ``` c"), stop.stopped) == ("a ", True)
    assert stop.feed("more") + stop.flush() == ""
    stop = StopStrings(["```"])
    assert stop.feed("nothing here") + stop.flush() == "nothing here"


def test_stop_strings_are_bounded():
    check_stop_strings(["s"] * 16)
    check_stop_strings("x" * 256)
    with pytest.raises(ValueError, match="at most 16 stop strings"):
        check_stop_strings(["s"] * 17)
    with pytest.raises(ValueError, match="at most 256 characters"):
        check_stop_strings(["ok", "x" * 257])


@pytest.mark.parametrize(
    "deltas, released, stopped",
    [
        (["abc", "def"], "abcdef", False),
        # Split across deltas: nothing of it is released on the way.
        (["ab`", "`", "`tail"], "ab", True),
        # The start of a stop string that never finishes is released in the end.
        (["x``", "y"], "x``y", False),
        (["ends ``"], "ends ``", False),
        (["```first"], "", True),
    ],
)
def test_streamed_text_is_released_once_it_cannot_begin_a_stop(
    deltas, released, stopped
):
    stop = StopStrings(["```", "STOP"])
    out = "".join(stop.feed(delta) for delta in deltas)
    if not stop.stopped:
        out += stop.flush()
    assert (out, stop.stopped) == (released, stopped)


def test_stream_ends_at_a_stop_string_its_tokens_missed():
    aborted = []
    detok = IncrementalStreamDetokenizer(_Chars(), stop=StopStrings(["```"]))
    detok.on_stop = lambda: aborted.append(True)
    collector = StreamOutputCollector("r")
    texts = []

    async def read():
        for piece in ("print(1)\n`", "``\nmore", "even more"):
            chunk = _chunk(piece)
            chunk["_detokenizer"] = detok
            collector.put_nowait(chunk)
            got = await collector.get()
            texts.append(got["text"])
            if got["finished"]:
                return got
        return None

    last = asyncio.run(read())
    assert "".join(texts) == "print(1)\n"
    assert last["finish_reason"] == "stop_sequence"
    assert aborted == [True]


def test_engine_side_stop_is_still_cut_and_not_aborted_again():
    aborted = []
    detok = IncrementalStreamDetokenizer(_Chars(), stop=StopStrings(["END"]))
    detok.on_stop = lambda: aborted.append(True)
    chunk = _chunk("doneEND", finished=True, reason="stop_sequence")
    detok.decode_into(chunk)
    assert (chunk["text"], chunk["finish_reason"], aborted) == (
        "done",
        "stop_sequence",
        [],
    )


def test_text_without_a_stop_string_is_released_in_full():
    detok = IncrementalStreamDetokenizer(_Chars(), stop=StopStrings(["```"]))
    first, last = _chunk("ends with ``"), _chunk("", finished=True, reason="eos")
    detok.decode_into(first)
    detok.decode_into(last)
    assert first["text"] + last["text"] == "ends with ``"
    assert last["finish_reason"] == "eos"


def _nonstream(monkeypatch, pieces, stop):
    """A non-streaming reply of ``pieces``, the last finishing it with eos."""
    aborted = []
    seq = SimpleNamespace(
        id=7, prompt_token_ids=[1], num_prompt_tokens=1, max_tokens=16
    )

    def preprocess(_prompt, _sampling, stream_callback=None, **_kwargs):
        # The engine never matches the stop string's own tokens here.
        for i, text in enumerate(pieces):
            last = i == len(pieces) - 1
            stream_callback(
                SimpleNamespace(
                    output_tokens=[ord(c) for c in text],
                    finished=last,
                    finish_reason="eos" if last else None,
                    num_cached_tokens=0,
                )
            )
        return seq

    engine = SimpleNamespace(
        io_processor=SimpleNamespace(preprocess=preprocess, requests={}),
        core_mgr=SimpleNamespace(
            add_request=lambda _seqs: None, abort_request=aborted.append
        ),
    )
    monkeypatch.setattr(api_server, "engine", engine)
    monkeypatch.setattr(
        api_server, "_validate_sequence_context_length", lambda *_a, **_k: None
    )
    monkeypatch.setattr(api_server, "tokenizer", _Chars())
    monkeypatch.setattr(api_server, "synthetic_token_text", None)
    # The text decoded while watching is the reply; it is not decoded again.
    monkeypatch.setattr(
        api_server, "delivered_text", lambda _ids: pytest.fail("decoded twice")
    )

    async def run():
        params = SimpleNamespace(stop_strings=stop)
        return [out async for out in api_server.generate_async("p", params, "r")]

    [out] = asyncio.run(run())
    return out["text"], out["finish_reason"], aborted


def test_nonstream_reply_ends_before_the_stop_string(monkeypatch):
    assert _nonstream(monkeypatch, ("answer``", "`\nmore", "end"), "```") == (
        "answer",
        "stop_sequence",
        [7],
    )


def test_nonstream_reply_without_its_stop_string_is_whole(monkeypatch):
    assert _nonstream(monkeypatch, ("ends with ", "``"), "```") == (
        "ends with ``",
        "eos",
        [],
    )

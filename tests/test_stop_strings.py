# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Stop strings, matched on text rather than on token ids."""

from atom.model_engine.request import RequestOutput
from atom.model_engine.stop_strings import IncrementalDetokenizer, check_stop_strings
from atom.sampling_params import SamplingParams


class _Tokenizer:
    """Decodes each id to a fixed piece, so tokenization is explicit here."""

    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.pieces[i] for i in ids)


# --- what the token-level matcher used to get wrong -------------------------


def test_a_stop_string_matches_however_the_model_tokenized_it():
    """The bug this whole split exists for.

    A client asks to stop at "five,". Standalone that is one token; mid
    sentence the model emits " five" + ",". Comparing token ids never matches
    and the request runs to `max_tokens`. Comparing text cannot care.
    """
    tok = _Tokenizer({1: " three", 2: ",", 3: " four", 4: ",", 5: " five", 6: ","})
    detok = IncrementalDetokenizer(tok)
    for token in (1, 2, 3, 4, 5):
        delta = detok.update([token], False)
    delta = detok.update([6], False)

    assert detok.text == " three, four, five,"
    assert check_stop_strings(detok.text, len(delta), ["five,"], False) == (
        "five,",
        14,
    )


# --- inclusion and truncation ----------------------------------------------


def test_the_stop_string_is_dropped_by_default():
    """OpenAI, vLLM and TGI all exclude it; ATOM used to return it."""
    assert check_stop_strings("abcSTOPdef", 10, ["STOP"], False) == ("STOP", 3)


def test_include_keeps_it_and_cuts_what_ran_past():
    assert check_stop_strings("abcSTOPdef", 10, ["STOP"], True) == ("STOP", 7)


def test_include_needs_no_cut_when_the_match_ends_the_text():
    assert check_stop_strings("abcSTOP", 7, ["STOP"], True) == ("STOP", -1)


# --- selection -------------------------------------------------------------


def test_the_earliest_completing_stop_wins():
    """One step can append several tokens, so several stops can land at once.

    Picking the earliest completion makes a multi-token step agree with
    feeding the same tokens one at a time.
    """
    assert check_stop_strings("xxAAyyBB", 8, ["BB", "AA"], False) == ("AA", 2)


def test_ties_go_to_stop_list_order():
    assert check_stop_strings("xxAB", 4, ["AB", "B"], False) == ("AB", 2)


def test_a_stop_string_straddling_two_steps_is_found():
    """Only the tail is searched; the window still has to reach back far
    enough that a match split across steps is not missed."""
    assert check_stop_strings("abcSTOP", 2, ["STOP"], False) == ("STOP", 3)


def test_text_older_than_this_step_is_not_rematched():
    """Otherwise a request stops on a string it already streamed past."""
    assert check_stop_strings("STOPabcdef", 3, ["STOP"], False) is None


def test_no_stops_and_no_new_text_are_both_no_ops():
    assert check_stop_strings("abcSTOP", 7, [], False) is None
    assert check_stop_strings("abcSTOP", 0, ["STOP"], False) is None


# --- the detokenizer -------------------------------------------------------


class _Utf8ByteTokenizer:
    """One id per byte, so a multi-byte character spans several tokens."""

    def decode(self, ids, skip_special_tokens=True):
        # `bytes()` of an `array("i")` copies its buffer -- four bytes per id
        # -- where from a list it takes the values. A real tokenizer reads
        # ids, so this double has to as well; keep the `list`.
        return bytes(list(ids)).decode("utf-8", errors="replace")


def test_a_multi_byte_character_is_not_emitted_half_decoded():
    detok = IncrementalDetokenizer(_Utf8ByteTokenizer())
    euro = "€".encode()  # three bytes

    assert detok.update([euro[0]], False) == ""
    assert detok.update([euro[1]], False) == ""
    assert detok.update([euro[2]], False) == "€"
    assert detok.text == "€"


def test_text_accumulates_across_deltas():
    tok = _Tokenizer({1: "he", 2: "ll", 3: "o"})
    detok = IncrementalDetokenizer(tok)
    assert [detok.update([t], False) for t in (1, 2, 3)] == ["he", "ll", "o"]
    assert detok.text == "hello"


# --- the frontend wrapper: where a stop string ends a request ---------------


class _Processor:
    """`InputOutputProcessor`'s stop-string half, without an engine."""

    def __init__(self, tokenizer):
        from atom.model_engine.llm_engine import InputOutputProcessor

        self.aborted = []
        self.impl = InputOutputProcessor.__new__(InputOutputProcessor)
        self.impl.tokenizer = tokenizer
        self.impl._stop_string_hits = {}
        self.impl.abort_request = self.aborted.append

    def wrap(self, params, callback):
        return self.impl._wrap_for_stop_strings(params, callback)


def _output(token_ids, finished=False):
    return RequestOutput(request_id=7, output_tokens=list(token_ids), finished=finished)


def test_a_stop_string_finishes_and_aborts_the_request():
    tok = _Tokenizer({1: "ab", 2: "STOP", 3: "cd"})
    proc = _Processor(tok)
    seen = []
    cb = proc.wrap(SamplingParams(stop_strings=["STOP"]), seen.append)

    cb(_output([1]))
    assert seen[-1].finished is False
    assert proc.aborted == []

    cb(_output([2]))
    assert seen[-1].finished is True
    assert seen[-1].finish_reason == "stop_sequence"
    assert seen[-1].stop_truncate_to == 2  # cut back to "ab"
    # Which stop string matched is deliberately not reported -- `finish_reason`
    # already says one did, and OpenAI's schema has no field for the identity.
    assert not hasattr(seen[-1], "stop_reason")
    assert proc.aborted == [7], "the engine core has to be told to stop"
    assert proc.impl._stop_string_hits[7] == 2


def test_output_arriving_before_the_abort_lands_is_dropped():
    """Abort is fire-and-forget, so more tokens can still show up."""
    tok = _Tokenizer({1: "STOP", 2: "extra"})
    proc = _Processor(tok)
    seen = []
    cb = proc.wrap(SamplingParams(stop_strings=["STOP"]), seen.append)

    cb(_output([1]))
    cb(_output([2]))
    assert len(seen) == 1, "the client already got its finished chunk"
    assert proc.aborted == [7], "and it is aborted once, not twice"


def test_a_request_without_stop_strings_is_not_wrapped_at_all():
    """No detokenizer and no per-step work for the overwhelming majority."""
    proc = _Processor(_Tokenizer({}))

    def cb(_):
        pass

    assert proc.wrap(SamplingParams(), cb) is cb


def test_include_stop_str_in_output_keeps_it():
    tok = _Tokenizer({1: "ab", 2: "STOP"})
    proc = _Processor(tok)
    seen = []
    cb = proc.wrap(
        SamplingParams(stop_strings=["STOP"], include_stop_str_in_output=True),
        seen.append,
    )

    cb(_output([1]))
    cb(_output([2]))
    assert seen[-1].finish_reason == "stop_sequence"
    # The match ends the text, so `check_stop_strings` alone would say -1.
    # The wrapper pins the length anyway: the abort is asynchronous, and
    # tokens emitted before it lands would otherwise be appended past the
    # stop string with nothing left to cut them.
    assert seen[-1].stop_truncate_to == len("abSTOP")


def test_a_tail_arriving_after_the_match_is_still_cut_off():
    """The bug an end-to-end run found: `-1` is only safe if the text cannot
    grow, and between the match and the abort landing it can."""
    tok = _Tokenizer({1: "abSTOP", 2: " tail"})
    proc = _Processor(tok)
    seen = []
    cb = proc.wrap(
        SamplingParams(stop_strings=["STOP"], include_stop_str_in_output=True),
        seen.append,
    )

    cb(_output([1]))
    truncate_to = proc.impl._stop_string_hits[7]
    # Whatever arrives next, the recorded cut still ends at the stop string.
    assert "abSTOP tail"[:truncate_to] == "abSTOP"

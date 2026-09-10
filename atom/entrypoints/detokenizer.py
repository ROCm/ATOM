# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Incremental detokenization, and the stop strings matched on its text.

Stop strings used to be encoded once at admission and matched as token
sequences against `Sequence.token_ids`. That is only correct when the client's
spelling of a stop string tokenizes the same way the model happens to emit it,
and it silently does not:

    "five,"   -> [52670, 11]
    " five,"  -> [ 4236, 11]

A model writing a list emits the second, so a client asking to stop at
``"five,"`` never stops. Nothing reports this -- the request simply runs to
`max_tokens`. Matching the text the client will actually receive has no such
failure mode, and it is what vLLM, TGI and the OpenAI API all specify.

The cost of moving is one boundary: text exists only where a tokenizer does,
which is the frontend, not the engine core. So the split here is vLLM's --
the scheduler keeps every *token*-level stop (EOS, `stop_token_ids`,
`max_tokens`, the `min_tokens` floor) because it can decide those itself, and
string stops live with whoever holds the tokenizer and abort the request from
there.
"""

import array
import logging
import sys
from typing import Any

from atom.model_engine.sequence import new_token_ids

logger = logging.getLogger("atom")


class IncrementalDetokenizer:
    """Decode token deltas without emitting incomplete UTF-8 characters.

    A token is not a character: several tokens can be needed before the text
    they encode is representable, so decoding each one alone yields U+FFFD.
    This keeps a two-mark window over the token stream -- everything before
    `prefix_offset` is settled, and `read_offset` is how far the caller has
    been shown -- and only advances when the newly decoded text is longer than
    the settled prefix and does not end mid-character.

    Callers get the delta; `text` accumulates the whole completion, because
    that is what a stop string has to be searched in (one can straddle any
    number of deltas).
    """

    __slots__ = (
        "prefix_offset",
        "read_offset",
        "text",
        "tokenizer",
        "tokens",
        "track_text",
    )

    def __init__(self, tokenizer: Any, *, track_text: bool = True):
        self.tokenizer = tokenizer
        self.track_text = track_text
        # Only ever sliced into `tokenizer.decode`, which takes an array.
        self.tokens: array.array = new_token_ids()
        self.prefix_offset = 0
        self.read_offset = 0
        self.text = ""

    def update(self, token_ids, finished: bool) -> str:
        """Feed a step's tokens; return the newly readable text."""
        self.tokens.extend(token_ids)
        prefix_text = self.tokenizer.decode(
            self.tokens[self.prefix_offset : self.read_offset],
            skip_special_tokens=True,
        )
        new_text = self.tokenizer.decode(
            self.tokens[self.prefix_offset :],
            skip_special_tokens=True,
        )

        if len(new_text) > len(prefix_text) and not new_text.endswith("�"):
            delta = new_text[len(prefix_text) :]
            self.prefix_offset = self.read_offset
            self.read_offset = len(self.tokens)
        elif finished:
            delta = new_text[len(prefix_text) :]
        else:
            delta = ""

        if self.track_text:
            self.text += delta
        return delta


class StreamingTextState:
    """Detokenize a stream without exposing a possible stop-string prefix.

    A stop string may straddle engine steps.  Holding only the suffix that is
    still a prefix of one of the configured stops prevents a later match from
    requiring text already sent to the client to be retracted.
    """

    __slots__ = ("detokenizer", "emitted_chars", "stops", "synthetic_text")

    def __init__(
        self,
        tokenizer: Any,
        stops: list[str] | None = None,
        *,
        synthetic_text: str | None = None,
    ):
        self.stops = tuple(stop for stop in (stops or ()) if stop)
        self.detokenizer = IncrementalDetokenizer(
            tokenizer, track_text=bool(self.stops)
        )
        self.emitted_chars = 0
        # Emitted once per token in place of the decoded text, for runs whose
        # text is a byproduct rather than an answer. See `SYNTHETIC_TOKEN_TEXT`.
        self.synthetic_text = synthetic_text

    @property
    def tokens(self):
        return self.detokenizer.tokens

    def update(
        self,
        token_ids,
        finished: bool,
        truncate_to: int = -1,
    ) -> str:
        delta = self.detokenizer.update(token_ids, finished)
        if self.synthetic_text is not None:
            # Decoded and thrown away: the run is measuring throughput, and
            # skipping the work would make the server look faster than the one
            # being measured. The stop-string hold-back below is skipped with
            # it -- it holds back text this run does not deliver.
            return self.synthetic_text * len(token_ids)
        if not self.stops:
            return delta

        text = self.detokenizer.text
        if truncate_to >= 0:
            safe_end = min(truncate_to, len(text))
        elif finished:
            safe_end = len(text)
        else:
            held = 0
            for stop in self.stops:
                # A complete stop is handled by the frontend wrapper in this
                # same engine step. Only a proper prefix needs carrying into
                # the next step.
                for prefix_len in range(1, len(stop)):
                    if prefix_len > held and text.endswith(stop[:prefix_len]):
                        held = prefix_len
            safe_end = len(text) - held

        if safe_end <= self.emitted_chars:
            return ""
        delta = text[self.emitted_chars : safe_end]
        self.emitted_chars = safe_end
        return delta


def check_stop_strings(
    output_text: str,
    new_char_count: int,
    stop: list[str],
    include_in_output: bool,
) -> tuple[str, int] | None:
    """Find the stop string that completes earliest in the new text.

    Returns ``(stop_string, truncate_to)`` or ``None``. ``truncate_to`` is the
    length ``output_text`` should be cut to, or -1 to leave it alone.

    Only the tail can contain a *new* match, so the search starts
    ``new_char_count + len(stop_str) - 1`` characters from the end: far enough
    back that a stop string straddling the boundary is still found, no
    further. Without that bound a long completion re-scans itself every step.

    When one step appends several tokens -- speculative decoding does -- more
    than one stop string can land at once. The earliest-completing one wins so
    the result is what appending one token at a time would have given; ties go
    to stop-list order, which is what makes the choice deterministic rather
    than dependent on dict iteration.
    """
    if not new_char_count or not stop:
        return None

    best_stop_str: str | None = None
    best_stop_index = 0
    best_end = sys.maxsize
    for stop_str in stop:
        # The former token-level implementation ignored these because
        # tokenizer.encode("") produced no ids. Preserve that behaviour
        # rather than treating every position as an immediate match.
        if not stop_str:
            continue
        stop_index = output_text.find(stop_str, -new_char_count - len(stop_str) + 1)
        if stop_index == -1:
            continue
        end = stop_index + len(stop_str)
        if end < best_end:
            best_stop_str = stop_str
            best_stop_index = stop_index
            best_end = end

    if best_stop_str is None:
        return None

    if include_in_output:
        # Keep the stop string, drop only what a multi-token step ran past it.
        if best_end >= len(output_text):
            return best_stop_str, -1
        return best_stop_str, best_end

    return best_stop_str, best_stop_index


def wrap_for_stop_strings(
    tokenizer,
    abort_request,
    requests,
    sampling_params,
    callback,
):
    """Wrap a stream callback so a stop string ends the request.

    This is the frontend half of the split described at the top of this
    module: the engine core cannot see text, so the one stop condition
    defined over text is decided here, against the detokenized output the
    client will actually receive, and the request is then aborted.

    `abort_request` tells the engine core to stop, `requests` is the
    frontend's live sequence table -- the cut is written onto the `Sequence`
    so the non-streaming builder can apply it -- and either may be None on a
    path that has neither.

    Requests without stop strings are handed back their own callback
    unchanged -- no detokenizer, no wrapper, no per-step work. That is the
    overwhelming majority of them.
    """
    stops = [stop for stop in (sampling_params.stop_strings or ()) if stop]
    if not stops:
        return callback

    detokenizer = IncrementalDetokenizer(tokenizer)
    include = sampling_params.include_stop_str_in_output
    min_tokens = sampling_params.min_tokens
    # The abort is fire-and-forget and the engine keeps producing until it
    # lands, so more output can arrive for a request already stopped.
    # Everything after the first hit is dropped rather than re-checked.
    state = {
        "stopped": False,
        "abort_sent": False,
        "completion_tokens": 0,
        "truncate_to": -1,
    }

    def request_abort(request_id):
        if state["abort_sent"] or abort_request is None:
            return
        try:
            abort_request(request_id)
            state["abort_sent"] = True
        except Exception:
            # Keep abort_sent false so a later engine chunk retries. The
            # client waits for the genuine terminal output rather than
            # being told the request finished while it still runs.
            logger.warning(
                "Abort after stop string failed for request %s",
                request_id,
                exc_info=True,
            )

    def on_output(request_output):
        state["completion_tokens"] += len(request_output.output_tokens or ())
        if not state["stopped"]:
            delta = detokenizer.update(
                request_output.output_tokens or (), request_output.finished
            )
            # A stop string itself may consume one or more tokens. Waiting
            # until the cumulative count is above the floor guarantees at
            # least min_tokens tokens precede the terminal condition.
            hit = (
                check_stop_strings(detokenizer.text, len(delta), stops, include)
                if state["completion_tokens"] > min_tokens
                else None
            )
            if hit is not None:
                # Only the cut is kept. Which stop string matched is not
                # reported: `finish_reason` already says one did, and
                # OpenAI's schema has no field for the identity.
                _stop_string, truncate_to = hit
                if truncate_to < 0:
                    # `check_stop_strings` says -1 when the match already
                    # ends the text, so there is nothing to cut *yet*. The
                    # abort is asynchronous, though, and whatever the
                    # engine emits before it lands still reaches
                    # `completion_token_ids`. Pinning the length here is
                    # what keeps that tail out of the final text.
                    truncate_to = len(detokenizer.text)
                state["stopped"] = True
                state["truncate_to"] = truncate_to
                request_output.stop_truncate_to = truncate_to
                seq = (requests or {}).get(request_output.request_id)
                if seq is not None:
                    seq.stop_truncate_to = truncate_to
                if request_output.finished:
                    request_output.finish_reason = "stop_sequence"
                else:
                    request_abort(request_output.request_id)
        elif not request_output.finished:
            # Trailing output from before the abort landed.
            request_abort(request_output.request_id)
            return
        else:
            # Do not synthesize a terminal event at the text match. The
            # real one may carry P/D KV-transfer metadata and is what lets
            # EngineCoreManager retire the callback safely.
            request_output.finish_reason = "stop_sequence"
            request_output.stop_truncate_to = state["truncate_to"]

        if callback is not None:
            callback(request_output)

    return on_output

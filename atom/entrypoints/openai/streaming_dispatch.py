# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Cross-thread dispatch and per-request delivery for streaming model output.

Two halves of one hand-off. :class:`StreamBatchDispatcher` runs on the engine
output threads and schedules a whole engine step per event loop.
:class:`StreamOutputCollector` merges pending token deltas before decoding them
when the SSE consumer reads, so a slow consumer does not accumulate decode work.
"""

import array
import logging
import random
import threading
import time
from asyncio import AbstractEventLoop, Event
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, NamedTuple

from atom.model_engine.sequence import new_token_ids

logger = logging.getLogger("atom")

# How long one stream may go without a chunk before it is worth a line. Long
# enough that a slow prefill or a busy engine step never trips it; short
# enough that a wedged response is named while the client is still waiting.
SILENCE_LOG_SECONDS = 30.0

# Every stream currently waiting to hand its client a frame, and since when.
# Keyed by `id()`: entries are added and removed around a single await in one
# place, so an id cannot outlive the watch that owns it.
_WAITING_SINCE: dict[int, float] = {}

# Fields a later chunk overrides on the one it merges into, when it has a value
# of its own. The SSE consumers keep the newest non-empty value they see, so
# merging this way hands them what reading each chunk separately would have.
_LATEST_WINS = ("finish_reason", "kv_transfer_params", "num_cached_tokens")

# Stands in for the decoded text of one token when the run has declared its own
# output meaningless -- forced speculative acceptance, today. It says what it is,
# so a dump of such a run cannot be mistaken for something the model wrote, and
# it carries no marker any dialect or tool-call format intercepts, so it reaches
# the client whatever the model is.
SYNTHETIC_TOKEN_TEXT = "synthetic "

# One audit per thousand reuses, which is what the shipped 1.7x was measured at.
DEFAULT_AUDIT_EVERY = 1000


@dataclass
class _DeltaReuse:
    """Whether `_decode` may skip its first `tokenizer.decode`, process-wide.

    Off until `enable_delta_reuse` has watched this tokenizer agree with it.
    Consulted per call rather than captured per stream, so switching it off
    reaches the four thousand streams already in flight, not only the next one.
    """

    enabled: bool = False
    audit_every: int = DEFAULT_AUDIT_EVERY
    calls: int = 0
    audits: int = 0  # comparisons that had something to compare
    mismatches: int = 0
    # The probe's negative control spoils a delta on purpose, and an ERROR
    # saying reuse is off, followed by the line saying it is on, is a log
    # nobody can read.
    probing: bool = False

    def should_audit(self) -> bool:
        # `calls` starts at zero, so a stream's first update is always checked
        # and a tokenizer the probe let through still meets a comparison at once.
        due = self.calls % self.audit_every == 0
        self.calls += 1
        return due

    def report_mismatch(self, reused: str, decoded: str) -> None:
        self.mismatches += 1
        if self.enabled:
            self.enabled = False
            if self.probing:
                return
            logger.error(
                "[detokenizer] the reused delta disagreed with the tokenizer "
                "(reused %r, decoded %r); delta reuse is now off for this "
                "process. Output stays correct -- this call used the decoded "
                "value -- but the tokenizer decodes a token span differently "
                "depending on where the window starts, which the startup probe "
                "did not reach.",
                reused,
                decoded,
            )


_DELTA_REUSE = _DeltaReuse()


@dataclass
class IncrementalStreamDetokenizer:
    """Decode token deltas without emitting incomplete UTF-8 characters."""

    tokenizer: Any
    # Grows for the whole life of a stream, one entry per token, and only ever
    # sliced into `tokenizer.decode` -- which takes an array. Nothing here is
    # serialized, so this one has no boundary to convert back at.
    tokens: array.array = field(default_factory=new_token_ids)
    prefix_offset: int = 0
    read_offset: int = 0
    # Emitted once per token in place of the decoded text, for runs whose text
    # is a byproduct rather than an answer. See `SYNTHETIC_TOKEN_TEXT`.
    synthetic_text: str | None = None
    # The last delta emitted, which is what the next call's first decode would
    # produce: emitting advances the window to exactly the span that produced
    # it. Maintained whether or not reuse is on, so the two paths never diverge
    # in state -- only in whether they pay for the decode.
    last_delta: str = ""

    def update(self, token_ids: list[int], finished: bool) -> str:
        """The text this stream has not been given yet, from its next tokens.

        One instance per stream, and calls on it must be in token order, one at
        a time. Handing the instance across threads is fine -- `flush` does,
        and `call_soon_threadsafe` supplies the happens-before edge -- but two
        overlapping calls are not.

        That was always true, since `tokens` and the two offsets are mutated
        here. What changed is the failure: `last_delta` now carries a value
        from the previous call, so an interleaved one does not merely reorder
        text, it leaves that value naming a span it no longer describes, and
        every later delta is cut at the wrong length. The audit finds it
        eventually; nothing raises.

        Empty `token_ids` is legal and yields "" without disturbing the stream.
        After `finished=True` there is nothing left to ask for.
        """
        decoded = self._decode(token_ids, finished)
        if self.synthetic_text is None:
            return decoded
        # Decoded and thrown away: the run is measuring throughput, and skipping
        # the work would make the server look faster than the one being measured.
        return self.synthetic_text * len(token_ids)

    def _decode(self, token_ids: list[int], finished: bool) -> str:
        """Both decodes share a window start, so subtracting one from the other
        is what isolates the new text. A character can span several tokens --
        an emoji is two, each decoding alone to U+FFFD -- so the new tokens
        cannot be decoded by themselves, and the window start cannot be the
        stream's or every token would re-decode the whole output.

        The first decode only ever yields a length, and its span is the one the
        previous call already emitted, so a stream that has emitted before
        knows the answer. `_DELTA_REUSE` is whether that shortcut is trusted
        for this tokenizer: measured 1.74x at one token per update and 1.40x
        at sixty-four, on DeepSeek-V4-Pro.
        """
        self.tokens.extend(token_ids)
        reuse = _DELTA_REUSE
        if reuse.enabled and not reuse.should_audit():
            prefix_text = self.last_delta
        else:
            prefix_text = self.tokenizer.decode(
                self.tokens[self.prefix_offset : self.read_offset],
                skip_special_tokens=True,
            )
            if reuse.enabled:
                if prefix_text:
                    reuse.audits += 1
                if prefix_text != self.last_delta:
                    reuse.report_mismatch(self.last_delta, prefix_text)
        new_text = self.tokenizer.decode(
            self.tokens[self.prefix_offset :],
            skip_special_tokens=True,
        )

        if len(new_text) > len(prefix_text) and not new_text.endswith("\ufffd"):
            delta = new_text[len(prefix_text) :]
            self.prefix_offset = self.read_offset
            self.read_offset = len(self.tokens)
            self.last_delta = delta
            return delta
        # Withheld or final: the window did not move, so the last delta is
        # still the text of the span it names.
        if finished:
            return new_text[len(prefix_text) :]
        return ""


# Boundaries, not coverage: characters that span several tokens, runs of
# whitespace, and a leading space -- the shapes that make a span's text depend
# on where its window started. Replayed at several merge depths because the
# premise is about where the window lands.
_PROBE_TEXTS = (
    "The quick brown fox jumps over the lazy dog. " * 3,
    "深度学习模型的推理性能取决于内存带宽和计算密度。" * 2,
    "party 🎉 family 👨‍👩‍👧‍👦 done. " * 2,
    "𝓗𝓮𝓵𝓵𝓸 ∑x²  ≈ ∫f(t)dt " * 2,
    " leading  spaces   and\t\ttabs\n\n ",
    "def merge(a, b):\n    return {**a, **b}\n",
)
_PROBE_CHUNKS = (1, 2, 3, 5, 8, 17, 64)
_PROBE_RANDOM_TOKENS = 128


def _probe_streams(tokenizer) -> list[list[int]]:
    """Token streams to replay, from both sides of the model.

    Encoded text covers what a tokenizer produces; sampled vocabulary covers
    what a *model* produces, which is any id at all -- including the lone byte
    tokens that never appear when encoding prose and are exactly where decoding
    a span depends on its neighbours. Seeded, so a failure is reproducible.
    """
    streams = [tokenizer.encode(t, add_special_tokens=False) for t in _PROBE_TEXTS]
    size = getattr(tokenizer, "vocab_size", 0) or 0
    if size:
        rng = random.Random(0)
        streams += [
            [rng.randrange(size) for _ in range(_PROBE_RANDOM_TOKENS)] for _ in range(2)
        ]
    return [s for s in streams if s]


def _replay(tokenizer, ids: list[int], chunk: int) -> str:
    """One stream through the real detokenizer, `chunk` tokens per update."""
    state = IncrementalStreamDetokenizer(tokenizer)
    return "".join(
        state.update(ids[i : i + chunk], i + chunk >= len(ids))
        for i in range(0, len(ids), chunk)
    )


def _a_wrong_delta_is_caught(tokenizer, ids: list[int]) -> bool:
    """Negative control. Without it, "no mismatches" is also what a comparison
    that never ran would report, and the probe would pass every tokenizer."""
    state = IncrementalStreamDetokenizer(tokenizer)
    for i, tid in enumerate(ids):
        state.update([tid], False)
        if state.last_delta and i + 1 < len(ids):
            state.last_delta += "\0"
            before = _DELTA_REUSE.mismatches
            state.update([ids[i + 1]], False)
            return _DELTA_REUSE.mismatches > before
    return False


def _probe(tokenizer) -> tuple[str | None, int]:
    """Why this tokenizer cannot be trusted with reuse, or None, and how many
    comparisons said so. Runs with reuse on and the audit at every call, so the
    check is the one `_decode` already carries rather than a second copy of it.
    """
    streams = _probe_streams(tokenizer)
    for ids in streams:
        want = tokenizer.decode(ids, skip_special_tokens=True)
        for chunk in _PROBE_CHUNKS:
            reused = _replay(tokenizer, ids, chunk)
            _DELTA_REUSE.enabled = False
            plain = _replay(tokenizer, ids, chunk)
            _DELTA_REUSE.enabled = True
            # Two failures worth telling apart: the first is the one this
            # shortcut causes, the second says the incremental scheme itself
            # does not suit this tokenizer and reuse would only amplify it.
            if reused != plain:
                return "reusing the delta changes the text this tokenizer emits", 0
            if reused != want:
                return (
                    (
                        "this tokenizer disagrees with a whole decode even "
                        "without reuse"
                    ),
                    0,
                )
    if _DELTA_REUSE.mismatches:
        return "the tokenizer disagreed with a reused delta", 0
    if not _DELTA_REUSE.audits:
        return "nothing was ever compared", 0
    if not _a_wrong_delta_is_caught(tokenizer, streams[0]):
        return "a deliberately wrong delta went unnoticed", 0
    return None, _DELTA_REUSE.audits


def _audit_interval(value: int | str) -> int:
    """How often to check a reused delta, from `ATOM_DETOKENIZER_AUDIT_EVERY`.

    Empty means the default. Anything else unusable warns and takes the default
    as well -- including 0, which reads like "never audit" but would divide by
    zero, and is not how reuse is turned off.
    """
    if value is None or value == "":
        return DEFAULT_AUDIT_EVERY
    try:
        interval = int(value)
    except (TypeError, ValueError):
        interval = 0
    if interval < 1:
        logger.warning(
            "[detokenizer] unusable ATOM_DETOKENIZER_AUDIT_EVERY=%r, using %d; "
            "reuse is turned off with ATOM_DETOKENIZER_DELTA_REUSE=off",
            value,
            DEFAULT_AUDIT_EVERY,
        )
        return DEFAULT_AUDIT_EVERY
    return interval


def enable_delta_reuse(
    tokenizer, mode: str = "auto", audit_every: int | str = ""
) -> bool:
    """Let `_decode` skip its first decode, if this tokenizer allows it.

    The shortcut holds where decoding a token span does not depend on where the
    window started -- true for the byte-level BPE tokenizers measured (DeepSeek,
    Qwen3.5, GLM-5.2), and not something to assume from a class name: a
    SentencePiece-style decoder adds or strips a leading space by position.

    So it is verified rather than declared, and any exception is a failure --
    an unverified shortcut is worth less than the decode it saves.

    Unrelated to the KV prefix cache, which is what "cache" means everywhere
    else in this repository.

    Neither setting may end the process or turn reuse on by accident: an
    unreadable mode leaves reuse off, which is what someone spelling this
    setting is reaching for, and an unreadable audit interval falls back to the
    default, since it is read at a callsite that has already loaded the weights.
    """
    setting = str(mode).strip().lower()
    if setting not in ("auto", "on", "off"):
        logger.warning(
            "[detokenizer] unknown delta reuse mode %r, leaving reuse off", mode
        )
        setting = "off"
    _DELTA_REUSE.audit_every = _audit_interval(audit_every)
    if setting != "auto":
        _DELTA_REUSE.enabled = setting == "on"
        logger.info("[detokenizer] delta reuse forced %s", setting)
        return _DELTA_REUSE.enabled

    before = replace(_DELTA_REUSE)
    _DELTA_REUSE.enabled, _DELTA_REUSE.audit_every, _DELTA_REUSE.probing = True, 1, True
    _DELTA_REUSE.audits = _DELTA_REUSE.mismatches = 0
    try:
        why, armed = _probe(tokenizer)
    except Exception:
        logger.exception("[detokenizer] delta reuse probe raised; leaving it off")
        why, armed = "the probe raised", 0
    finally:
        _DELTA_REUSE.audit_every = before.audit_every
        _DELTA_REUSE.calls = before.calls
        _DELTA_REUSE.probing = False
        _DELTA_REUSE.audits = _DELTA_REUSE.mismatches = 0

    _DELTA_REUSE.enabled = why is None
    if why:
        logger.warning("[detokenizer] delta reuse off: %s", why)
    else:
        logger.info(
            "[detokenizer] delta reuse on: %d comparisons agreed across %d merge "
            "depths, and a spoiled delta was caught",
            armed,
            len(_PROBE_CHUNKS),
        )
    return _DELTA_REUSE.enabled


def merge_chunk(into: dict, new: dict) -> None:
    """Fold ``new`` into the chunk already waiting. ``into`` is modified.

    Both deltas extend in place: ``into`` holds the copy ``put_nowait`` took,
    and rebuilding them walked the whole accumulation on every merge.
    """
    into["token_ids"].extend(new.get("token_ids") or ())
    # Popped so the string has one reference and CPython grows it in place;
    # assigning `into["text"] + ...` back reallocates and copies per merge.
    text = into.pop("text", "")
    try:
        text += new.get("text", "")
    finally:
        into["text"] = text
    into["finished"] = bool(into.get("finished") or new.get("finished"))
    for key in _LATEST_WINS:
        if new.get(key):
            into[key] = new[key]


class StreamOutputCollector:
    """Per-request delivery point that merges chunks when the consumer lags.

    A stream holds at most one pending chunk per tag. Raw token deltas from the
    dispatcher merge here before detokenization, coroutine wakeup, JSON encoding
    and socket writing. Detokenizer state is advanced only by ``get()``, on the
    event loop; the engine output threads never decode collector-bound chunks.

    There is no timer or minimum batch size: get() can consume any pending
    delta immediately. Merging covers only unread chunks, but decoding runs
    synchronously on the event loop, so this is not a delivery-latency bound.
    Large backlogs can delay other consumers while their text is decoded.

    ``tag`` is the fan-out sibling index (``SamplingParams.n>1``) or ``None`` for
    a plain single-sequence stream. Chunks merge per tag, so siblings never mix.
    """

    def __init__(self, request_id: str = "") -> None:
        self.request_id = request_id
        self._pending: dict[Any, dict] = {}
        self._ready = Event()

    def put_nowait(self, payload: dict | tuple[int, dict]) -> None:
        """Accept one prepared chunk. Called on the event loop, never off it."""
        if type(payload) is tuple:
            tag, chunk = payload
        else:
            tag, chunk = None, payload
        waiting = self._pending.get(tag)
        if waiting is None:
            # A merge extends this list, so it has to be ours. The scheduler
            # already copies per step, but that is too far away to rely on.
            chunk["token_ids"] = list(chunk.get("token_ids") or ())
            self._pending[tag] = chunk
        else:
            merge_chunk(waiting, chunk)
        self._ready.set()

    async def get(self) -> dict | tuple[int, dict]:
        """Await the next chunk, carrying whatever merged into it."""
        while not self._pending:
            await self._ready.wait()
        tag, chunk = next(iter(self._pending.items()))
        del self._pending[tag]
        if not self._pending:
            self._ready.clear()
        state = chunk.pop("_detokenizer", None)
        if state is not None:
            try:
                chunk["text"] = state.update(
                    chunk["token_ids"], bool(chunk.get("finished"))
                )
            except Exception:
                logger.exception(
                    "Error detokenizing stream %s (tag=%s)", self.request_id, tag
                )
                # Keep token accounting and terminal metadata, and let fan-out
                # siblings continue. Failed tokens remain in the detokenizer;
                # a later update can decode them without appending them twice.
                chunk["text"] = ""
        return chunk if tag is None else (tag, chunk)


class FrameWait:
    """Times one gap between frames the client actually receives.

    Measured here and not at :meth:`StreamOutputCollector.get`, which is where
    it started. That is the one place a stream waits for the *engine*, but two
    stages sit between it and the socket -- the reasoning channel's read-ahead
    and the tool-call format's -- and while either withholds, `get` keeps
    returning on schedule. The gauge read zero while the client received
    nothing, which is the exact symptom it was built for.

    The first frame still has to be excluded, and the docstring here once
    claimed otherwise. Every response generator awaits the collector before
    yielding anything, so the wait for frame one is admission, queueing and
    prefill -- measured, a request 200 ms into a queue with no token yet
    produced put 0.2 s on the gauge, which is `atom:requests_waiting` wearing
    a different name, and at a deep queue would log a line per admitted
    request blaming the read-ahead. `armed` is off for that one wait.

    A timestamp and a dict entry rather than `asyncio.wait_for`: this runs
    once per frame per stream, and arming a timer costs 1.38 us against
    0.07 us for this. No timer also means no background task to own.
    """

    __slots__ = ("_started", "armed", "request_id")

    def __init__(self, request_id: str = "", *, armed: bool = True) -> None:
        self.request_id = request_id
        self.armed = armed
        self._started = 0.0

    def __enter__(self) -> None:
        # No `as`: nothing needs the watch itself, only its lifetime.
        self._started = time.monotonic()
        if self.armed:
            _WAITING_SINCE[id(self)] = self._started

    def __exit__(self, *exc) -> None:
        _WAITING_SINCE.pop(id(self), None)
        silence = time.monotonic() - self._started
        if self.armed and silence >= SILENCE_LOG_SECONDS:
            # After the fact, and free: one comparison on a frame that was
            # going to arrive anyway. Catches a stall that recovered, which
            # the gauge cannot -- by scrape time it is over.
            logger.warning(
                f"request {self.request_id or '<unnamed>'} sent the client "
                f"nothing for {silence:.1f}s before recovering; if this "
                f"repeats, the engine or one of the marker read-aheads is "
                f"holding output back"
            )


def longest_silence_seconds() -> float:
    """How long the most starved in-flight stream has been waiting.

    Zero when nothing is waiting. Exported as a gauge so a stalled response
    shows up while it is stalled, rather than as a support ticket.
    """
    if not _WAITING_SINCE:
        return 0.0
    now = time.monotonic()
    return now - min(_WAITING_SINCE.values())


@dataclass
class StreamDeliveryTiming:
    """Track frontend delivery intervals independently of token decoding."""

    last_output_at: float | None = None

    def record(
        self, num_new_tokens: int, observe: Callable[[float, int], None]
    ) -> None:
        now = time.perf_counter()
        if self.last_output_at is None:
            self.last_output_at = now
        else:
            observe(now - self.last_output_at, num_new_tokens)
            # Keep instrumentation work out of the next interval.
            self.last_output_at = time.perf_counter()


@dataclass
class StreamState:
    detokenizer: IncrementalStreamDetokenizer
    timing: StreamDeliveryTiming = field(default_factory=StreamDeliveryTiming)


class _BufferedChunk(NamedTuple):
    """One stream's chunk, waiting for the end of the current engine step."""

    loop: AbstractEventLoop
    collector: Any
    state: StreamState
    chunk: dict
    tag: int | None


class StreamBatchDispatcher:
    """Collect one engine step per output thread and dispatch it by event loop.

    The dispatcher has no persistent per-stream registry. Each engine callback
    creates one composed stream state and must reuse it for every chunk of its
    (collector, tag); every collector-bound chunk carries that same state.
    Merging keeps the first pending chunk's state, so it must not be replaced
    partway through a stream.

    For StreamOutputCollector, references cross threads but only get() on the
    event loop mutates the detokenizer. Frontend delivery updates the separate
    timing state on that same loop, before merging. Output threads pass the state
    through. An unread chunk or queued delivery keeps the state (including its
    token history) alive after the engine drops the finished callback; consuming
    or discarding those references allows it to be reclaimed. Queue consumers use
    eager decoding on their output thread instead.
    """

    def __init__(
        self,
        tokenizer: Any,
        synthetic_text: str | None = None,
        observe_inter_token_latency: Callable[[float, int], None] | None = None,
    ):
        self.tokenizer = tokenizer
        self.synthetic_text = synthetic_text
        self._observe_inter_token_latency = observe_inter_token_latency
        self._thread_local = threading.local()

    def new_state(self) -> StreamState:
        """Make decoding and delivery timing state for one stream's callback."""
        return StreamState(
            IncrementalStreamDetokenizer(
                self.tokenizer, synthetic_text=self.synthetic_text
            )
        )

    def enqueue(
        self,
        *,
        loop: AbstractEventLoop,
        collector: Any,
        state: StreamState,
        chunk: dict,
        tag: int | None = None,
    ) -> None:
        """Buffer a raw chunk until the current engine step is flushed."""
        buf = getattr(self._thread_local, "buf", None)
        if buf is None:
            buf = self._thread_local.buf = []
        buf.append(_BufferedChunk(loop, collector, state, chunk, tag))

    def flush(self) -> None:
        """Schedule raw chunks, letting each consumer coalesce before decoding."""
        tl = self._thread_local
        buf = getattr(tl, "buf", None)
        if not buf:
            return
        tl.buf = []

        by_loop: dict[AbstractEventLoop, list[_BufferedChunk]] = {}
        for item in buf:
            if isinstance(item.collector, StreamOutputCollector):
                # Keep this state with the pending chunk until get(). Moving
                # only the JSON/socket work downstream still made four output
                # threads decode every token while contending for the GIL.
                item.chunk["_detokenizer"] = item.state.detokenizer
            else:
                # Queue consumers cannot decode on read and still receive a
                # prepared chunk, as before.
                item.chunk["text"] = item.state.detokenizer.update(
                    item.chunk.get("token_ids") or [],
                    bool(item.chunk.get("finished")),
                )
            by_loop.setdefault(item.loop, []).append(item)

        for loop, items in by_loop.items():
            loop.call_soon_threadsafe(self._deliver, items)

    def _deliver(self, items: list[_BufferedChunk]) -> None:
        """Run on the target event loop and hand a whole step to its collectors.

        A step is delivered in one callback, never split across loop iterations.
        Splitting was tried as a fairness measure -- deliver 128, re-arm the rest
        with call_soon -- and it silently corrupts streams: the output thread can
        schedule the next step's delivery in between, so a collector receives
        step N+1's chunk before step N's leftovers. Deltas then merge in the
        wrong order, and an end-of-stream that lands before a straggler is
        overwritten by it, hanging that client for good.
        """
        for item in items:
            num_new_tokens = len(item.chunk.get("token_ids") or ())
            if num_new_tokens and self._observe_inter_token_latency is not None:
                item.state.timing.record(
                    num_new_tokens, self._observe_inter_token_latency
                )
            payload = item.chunk if item.tag is None else (item.tag, item.chunk)
            item.collector.put_nowait(payload)

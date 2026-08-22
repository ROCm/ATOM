# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A stalled response has to be visible while it is stalled.

The symptom this whole line of work started from was ten minutes of silence on
a streaming request with every metric looking healthy. The cause is fixed, and
the next one will be different — so the observation is worth having on its own.

Measured around the *yield to the client*, not around the collector's await.
The collector is the one place a stream waits for the engine, but two stages
sit between it and the socket — the reasoning channel's read-ahead and the
tool-call format's — and while either withholds, the collector keeps returning
on schedule. Watching there reported zero for exactly the stall it was built
to catch; `TestWithholdingIsSilenceToo` is that case.

Deliberately not `asyncio.wait_for`. This runs once per frame per stream;
arming a timer measured 1.38 us against 0.07 us for a timestamp and a dict
entry, and a timestamp needs no background task to own.
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

from atom.entrypoints.openai import api_server
from atom.entrypoints.openai import streaming_dispatch as sd
from atom.entrypoints.openai.streaming_dispatch import (
    FrameWait,
    longest_silence_seconds,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    sd._WAITING_SINCE.clear()
    yield
    sd._WAITING_SINCE.clear()


async def frames(source, request_id: str = "req"):
    """The shape `_client_stream` wraps a response generator in.

    Kept here rather than importing the endpoint's copy: that one also does
    request logging and lives in a module that pulls in the engine. What is
    under test is the timing, and this is the timing verbatim -- including
    `armed`, without which every scenario below would pass while the endpoint
    timed the queue.
    """
    it = source.__aiter__()
    delivered = False
    while True:
        with FrameWait(request_id, armed=delivered):
            try:
                chunk = await it.__anext__()
            except StopAsyncIteration:
                return
        delivered = True
        yield chunk


async def _let_the_loop_run():
    """Yield control so a pending await is actually reached."""
    for _ in range(3):
        await asyncio.sleep(0)


class TestWhileItIsHappening:
    def test_nothing_waiting_reads_as_no_silence(self):
        assert longest_silence_seconds() == 0.0

    def test_a_stream_waiting_for_its_next_frame_is_visible(self):
        gate = asyncio.Event()

        async def source():
            yield "data: one\n\n"
            await gate.wait()
            yield "data: two\n\n"

        async def scenario():
            out = frames(source())
            await out.__anext__()
            task = asyncio.create_task(out.__anext__())
            await _let_the_loop_run()
            seen = longest_silence_seconds()
            gate.set()
            await task
            return seen

        assert asyncio.run(scenario()) > 0.0

    def test_it_clears_once_the_frame_goes_out(self):
        async def source():
            yield "data: one\n\n"
            yield "data: two\n\n"

        async def scenario():
            out = frames(source())
            await out.__anext__()
            await out.__anext__()
            return longest_silence_seconds()

        assert asyncio.run(scenario()) == 0.0

    def test_the_oldest_wait_is_the_one_reported(self):
        g1, g2 = asyncio.Event(), asyncio.Event()

        async def source(gate):
            yield "data: first\n\n"
            await gate.wait()
            yield "data: second\n\n"

        async def scenario():
            a, b = frames(source(g1), "a"), frames(source(g2), "b")
            await a.__anext__()
            await b.__anext__()
            t1 = asyncio.create_task(a.__anext__())
            await _let_the_loop_run()
            await asyncio.sleep(0.02)
            t2 = asyncio.create_task(b.__anext__())
            await _let_the_loop_run()
            seen = longest_silence_seconds()
            g1.set()
            g2.set()
            await t1
            await t2
            return seen

        assert asyncio.run(scenario()) >= 0.02

    def test_a_cancelled_stream_does_not_linger_in_the_registry(self):
        """A client that disconnects unwinds the generator; the entry must go.

        Otherwise one abandoned request pins the gauge high forever and the
        signal is useless from then on.
        """

        async def source():
            yield "data: one\n\n"
            await asyncio.Event().wait()

        async def scenario():
            out = frames(source())
            await out.__anext__()
            task = asyncio.create_task(out.__anext__())
            await _let_the_loop_run()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            return longest_silence_seconds()

        assert asyncio.run(scenario()) == 0.0

    def test_a_finished_stream_leaves_nothing_behind(self):
        """`StopAsyncIteration` unwinds through the watch like any exit."""

        async def source():
            yield "data: only\n\n"

        async def scenario():
            out = frames(source())
            await out.__anext__()
            with pytest.raises(StopAsyncIteration):
                await out.__anext__()
            return longest_silence_seconds()

        assert asyncio.run(scenario()) == 0.0


class TestWithholdingIsSilenceToo:
    """The case that moved the measurement out of the collector.

    A response whose tokens are arriving on time but are being held by a
    marker read-ahead sends the client nothing. Watched at the collector, that
    stream looks perfectly healthy -- it wakes on every token -- and the gauge
    reads zero. Watched at the yield, it is what it is: silence.
    """

    def test_a_generator_that_consumes_without_yielding_reads_as_silent(self):
        released = asyncio.Event()

        async def withholding_source():
            """Tokens arrive; nothing is released until the marker resolves."""
            yield "data: start\n\n"
            for _ in range(20):
                await asyncio.sleep(0)  # a token, consumed and held
            await released.wait()
            yield "data: everything at once\n\n"

        async def scenario():
            out = frames(withholding_source())
            await out.__anext__()
            task = asyncio.create_task(out.__anext__())
            await _let_the_loop_run()
            await asyncio.sleep(0.02)
            seen = longest_silence_seconds()
            released.set()
            await task
            return seen

        assert asyncio.run(scenario()) >= 0.02


class TestAfterItRecovers:
    def test_a_long_silence_is_logged_when_it_ends(self, caplog, monkeypatch):
        """The gauge cannot see a stall that is already over; this can."""
        monkeypatch.setattr(sd, "SILENCE_LOG_SECONDS", 0.01)

        async def source():
            yield "data: one\n\n"
            await asyncio.sleep(0.02)
            yield "data: late\n\n"

        async def scenario():
            out = frames(source(), "req-42")
            await out.__anext__()
            await out.__anext__()

        with caplog.at_level(logging.WARNING, logger="atom"):
            asyncio.run(scenario())
        hits = [r for r in caplog.records if "sent the client nothing" in r.message]
        assert hits and "req-42" in hits[0].message

    def test_an_ordinary_wait_says_nothing(self, caplog):
        """Every frame goes through here, so the quiet case must stay quiet."""

        async def source():
            yield "data: one\n\n"
            yield "data: two\n\n"

        async def scenario():
            out = frames(source())
            await out.__anext__()
            await out.__anext__()

        with caplog.at_level(logging.WARNING, logger="atom"):
            asyncio.run(scenario())
        assert not [r for r in caplog.records if "sent the client" in r.message]


class TestTheEndpointUsesIt:
    """Three streaming responses, and the watchdog has to wrap all three.

    It wrapped two: `_logged_stream` was a logging helper the Anthropic
    endpoint never called. A gauge with an endpoint-shaped hole is worse than
    no gauge, because the zero it reports reads as an answer.
    """

    def test_every_streaming_response_is_wrapped(self):
        import pathlib
        import re

        src = pathlib.Path(api_server.__file__).read_text()
        total = src.count("StreamingResponse(")
        wrapped = len(re.findall(r"StreamingResponse\(\s*_client_stream\(", src))
        assert total == 3, f"the endpoint count changed ({total}); check this test"
        assert wrapped == total, (
            f"{total - wrapped} of {total} StreamingResponse calls are served "
            "from an unwrapped generator"
        )

    def test_the_wrapper_actually_times_the_frame(self):
        """`_client_stream` wrapping every response is half of it; the other
        half is that the wrapper opens a watch around the await. Without this
        the source check passes on a wrapper that only logs."""
        import inspect

        src = inspect.getsource(api_server._client_stream)
        assert "with FrameWait(request_id" in src
        body = src.split("with FrameWait(request_id", 1)[1]
        assert (
            "__anext__()" in body.split("yield")[0]
        ), "the watch does not cover the await that produces the next frame"

    def test_the_logging_only_wrapper_is_gone(self):
        """Its name was the bug: it wrapped what wanted logging, not what
        wanted watching, so the Anthropic endpoint went without either."""
        import pathlib

        src = pathlib.Path(api_server.__file__).read_text()
        assert "_logged_stream(" not in src


class TestQueueingIsNotSilence:
    """The wait for the *first* frame is admission, queueing and prefill.

    Every response generator awaits the collector before yielding anything, so
    timing that wait puts queue depth on the gauge -- which
    `atom:requests_waiting` already reports -- and, past the threshold, logs a
    line per admitted request blaming the marker read-ahead. Moving the watch
    out to the frame did not remove this by itself; the docstring said it did.
    """

    def test_a_queued_request_reads_as_no_silence(self):
        gate = asyncio.Event()

        async def slow_to_start():
            await gate.wait()  # admission + prefill
            yield "data: first\n\n"

        async def scenario():
            out = frames(slow_to_start())
            task = asyncio.create_task(out.__anext__())
            await _let_the_loop_run()
            await asyncio.sleep(0.02)
            queued = longest_silence_seconds()
            gate.set()
            await task
            return queued

        assert asyncio.run(scenario()) == 0.0

    def test_a_slow_first_frame_is_not_logged_however_long(self, caplog, monkeypatch):
        monkeypatch.setattr(sd, "SILENCE_LOG_SECONDS", 0.01)

        async def slow_to_start():
            await asyncio.sleep(0.03)
            yield "data: first\n\n"

        async def scenario():
            out = frames(slow_to_start(), "queued-req")
            await out.__anext__()

        with caplog.at_level(logging.WARNING, logger="atom"):
            asyncio.run(scenario())
        assert not [r for r in caplog.records if "sent the client" in r.message]

    def test_the_gap_after_the_first_frame_is_still_seen(self):
        """The other half: disarming one wait must not disarm the rest."""
        gate = asyncio.Event()

        async def stalls_after_one():
            yield "data: first\n\n"
            await gate.wait()
            yield "data: second\n\n"

        async def scenario():
            out = frames(stalls_after_one())
            await out.__anext__()
            task = asyncio.create_task(out.__anext__())
            await _let_the_loop_run()
            await asyncio.sleep(0.02)
            seen = longest_silence_seconds()
            gate.set()
            await task
            return seen

        assert asyncio.run(scenario()) >= 0.02

    def test_the_endpoint_disarms_the_first_wait(self):
        """Its body is unreachable from a unit test; this reads the source.

        By AST, and for the *shape*: `armed` must be a variable, not a
        constant. `assert "armed=" in src` passed with `armed=True`, which is
        the bug -- and the harness above is a local copy of this loop, so
        mutating the endpoint could not reach it either.
        """
        import ast
        import inspect
        import textwrap

        tree = ast.parse(textwrap.dedent(inspect.getsource(api_server._client_stream)))
        waits = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "FrameWait"
        ]
        assert len(waits) == 1, f"{len(waits)} FrameWait calls; expected one"
        armed = [k for k in waits[0].keywords if k.arg == "armed"]
        assert armed, "the endpoint arms every wait, the queue included"
        assert not isinstance(
            armed[0].value, ast.Constant
        ), "`armed` is a constant, so the first wait is timed like the rest"


class TestRequestLoggingCannotBreakTheStream:
    """`--request-log` is a diagnostic. It was also a way to lose the answer.

    `serving_chat` coalesces finish + usage + `[DONE]` into one send -- a
    deliberate saving of two socket writes per request at a wave boundary --
    and the logger ran `json.loads` over that whole send. `Extra data:` came
    out of the generator, so with the flag on, the last frame of every OpenAI
    stream never reached the client and no `[DONE]` was written. Off by
    default, which is why nothing noticed.

    The Anthropic half is quieter: its frames put `event: NAME` on the line
    above the data, so a `startswith("data: ")` test matched none of them and
    that endpoint logged nothing at all.
    """

    COALESCED = (
        'data: {"id":"x","choices":[{"finish_reason":"stop"}]}\n\n'
        'data: {"usage":{"total_tokens":7}}\n\n'
        "data: [DONE]\n\n"
    )
    ANTHROPIC = 'event: content_block_delta\ndata: {"type":"content_block_delta"}\n\n'

    @staticmethod
    def _logged(chunk):
        """What `_log_sse` writes, without touching the module's real logger."""
        written = []

        class Recorder:
            @staticmethod
            def info(line):
                written.append(json.loads(line))

        original = api_server._request_logger
        api_server._request_logger = Recorder
        try:
            api_server._log_sse(chunk, "req-1")
        finally:
            api_server._request_logger = original
        return written

    def test_a_coalesced_send_does_not_raise(self):
        """The whole bug: this used to come out of the generator."""
        self._logged(self.COALESCED)

    def test_every_frame_in_it_is_logged(self):
        kinds = [e["type"] for e in self._logged(self.COALESCED)]
        assert kinds == ["stream_chunk", "stream_chunk", "stream_done"], kinds

    def test_an_anthropic_frame_is_logged(self):
        events = self._logged(self.ANTHROPIC)
        assert [e["type"] for e in events] == ["stream_chunk"]
        assert events[0]["data"]["type"] == "content_block_delta"

    def test_an_unparsable_payload_is_kept_rather_than_raised(self):
        events = self._logged("data: {not json\n\n")
        assert [e["type"] for e in events] == ["stream_chunk_unparsed"]
        assert events[0]["data"] == "{not json"

    def test_logging_off_writes_nothing_and_still_does_not_raise(self):
        original = api_server._request_logger
        api_server._request_logger = None
        try:
            api_server._log_sse(self.COALESCED, "req-1")
        finally:
            api_server._request_logger = original

    def test_the_client_still_receives_every_frame(self):
        """Logging is a side effect; `_client_stream` must yield regardless."""
        written = []

        class Recorder:
            @staticmethod
            def info(line):
                written.append(line)

        async def source():
            yield "data: {}\n\n"
            yield self.COALESCED

        async def collect():
            return [c async for c in api_server._client_stream(source(), "req-1")]

        original = api_server._request_logger
        api_server._request_logger = Recorder
        try:
            out = asyncio.run(collect())
        finally:
            api_server._request_logger = original
        assert "".join(out).endswith("data: [DONE]\n\n")
        assert len(written) == 4

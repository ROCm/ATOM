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
import logging

import pytest

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
    under test is the timing, and this is the timing verbatim.
    """
    it = source.__aiter__()
    while True:
        with FrameWait(request_id):
            try:
                chunk = await it.__anext__()
            except StopAsyncIteration:
                return
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

        from atom.entrypoints.openai import api_server

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

        from atom.entrypoints.openai import api_server

        src = inspect.getsource(api_server._client_stream)
        assert "with FrameWait(request_id):" in src
        body = src.split("with FrameWait(request_id):", 1)[1]
        assert (
            "__anext__()" in body.split("yield")[0]
        ), "the watch does not cover the await that produces the next frame"

    def test_the_logging_only_wrapper_is_gone(self):
        """Its name was the bug: it wrapped what wanted logging, not what
        wanted watching, so the Anthropic endpoint went without either."""
        import pathlib

        from atom.entrypoints.openai import api_server

        src = pathlib.Path(api_server.__file__).read_text()
        assert "_logged_stream(" not in src

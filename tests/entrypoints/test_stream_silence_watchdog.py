# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A stalled response has to be visible while it is stalled.

The symptom this whole line of work started from was ten minutes of silence on
a streaming request with every metric looking healthy. The cause is fixed, and
the next one will be different — so the observation is worth having on its own:
`StreamOutputCollector.get` is the single point where any stream waits, so the
age of the oldest wait is the age of the most starved response.

Deliberately not `asyncio.wait_for`. This runs once per token per stream;
arming a timer measured 1.38 us against 0.07 us for a timestamp and a dict
entry, and a timestamp needs no background task to own.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from atom.entrypoints.openai import streaming_dispatch as sd
from atom.entrypoints.openai.streaming_dispatch import (
    StreamOutputCollector,
    longest_silence_seconds,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    sd._WAITING_SINCE.clear()
    yield
    sd._WAITING_SINCE.clear()


async def _already_streaming() -> StreamOutputCollector:
    """A collector that has delivered once.

    The watchdog measures silence *between* chunks. A stream that has not
    delivered yet is queued, not stalled -- admission and prefill have not
    happened -- so every scenario about stalling has to get past that first
    chunk before it means anything.
    """
    collector = StreamOutputCollector()
    collector.put_nowait({"text": "first"})
    await collector.get()
    return collector


async def _let_the_loop_run():
    """Yield control so a pending `get()` reaches its await."""
    for _ in range(3):
        await asyncio.sleep(0)


class TestWhileItIsHappening:
    def test_nothing_waiting_reads_as_no_silence(self):
        assert longest_silence_seconds() == 0.0

    def test_a_waiting_stream_is_visible(self):
        async def scenario():
            collector = await _already_streaming()
            task = asyncio.create_task(collector.get())
            await _let_the_loop_run()
            seen = longest_silence_seconds()
            collector.put_nowait({"text": "hi"})
            await task
            return seen

        assert asyncio.run(scenario()) > 0.0

    def test_it_clears_once_the_chunk_arrives(self):
        async def scenario():
            collector = StreamOutputCollector()
            task = asyncio.create_task(collector.get())
            await _let_the_loop_run()
            collector.put_nowait({"text": "hi"})
            await task
            return longest_silence_seconds()

        assert asyncio.run(scenario()) == 0.0

    def test_the_oldest_wait_is_the_one_reported(self):
        async def scenario():
            old, new = await _already_streaming(), await _already_streaming()
            t1 = asyncio.create_task(old.get())
            await _let_the_loop_run()
            await asyncio.sleep(0.02)
            t2 = asyncio.create_task(new.get())
            await _let_the_loop_run()
            seen = longest_silence_seconds()
            old.put_nowait({"a": 1})
            new.put_nowait({"b": 2})
            await t1
            await t2
            return seen

        assert asyncio.run(scenario()) >= 0.02

    def test_a_cancelled_stream_does_not_linger_in_the_registry(self):
        """A client that disconnects unwinds the generator; the entry must go.

        Otherwise one abandoned request pins the gauge high forever and the
        signal is useless from then on.
        """

        async def scenario():
            collector = StreamOutputCollector()
            task = asyncio.create_task(collector.get())
            await _let_the_loop_run()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            return longest_silence_seconds()

        assert asyncio.run(scenario()) == 0.0


class TestQueueingIsNotAStall:
    """A stream that has not started is waiting its turn, not wedged.

    Both SSE generators await `get()` as the first statement of their loop,
    before admission and prefill. At the concurrency this server is
    benchmarked at, that first wait routinely outlives the silence threshold,
    so counting it would log a line per backlogged request onto the event loop
    and turn the gauge into a queue-depth proxy that `atom:requests_waiting`
    already provides.
    """

    def test_the_first_wait_is_not_silence(self):
        async def scenario():
            collector = StreamOutputCollector()
            task = asyncio.create_task(collector.get())
            await _let_the_loop_run()
            seen = longest_silence_seconds()
            collector.put_nowait({"text": "first"})
            await task
            return seen

        assert asyncio.run(scenario()) == 0.0

    def test_the_first_wait_is_not_logged_however_long(self, caplog, monkeypatch):
        monkeypatch.setattr(sd, "SILENCE_LOG_SECONDS", 0.01)

        async def scenario():
            collector = StreamOutputCollector()
            task = asyncio.create_task(collector.get())
            await _let_the_loop_run()
            await asyncio.sleep(0.03)
            collector.put_nowait({"text": "first"})
            await task

        with caplog.at_level(logging.WARNING, logger="atom"):
            asyncio.run(scenario())
        assert not [r for r in caplog.records if "delivered nothing" in r.message]


class TestAfterItRecovers:
    def test_a_long_silence_is_logged_when_it_ends(self, caplog, monkeypatch):
        """The gauge cannot see a stall that is already over; this can."""
        monkeypatch.setattr(sd, "SILENCE_LOG_SECONDS", 0.01)

        async def scenario():
            collector = await _already_streaming()
            task = asyncio.create_task(collector.get())
            await _let_the_loop_run()
            await asyncio.sleep(0.02)
            collector.put_nowait({"text": "late"})
            await task

        with caplog.at_level(logging.WARNING, logger="atom"):
            asyncio.run(scenario())
        assert any("delivered nothing for" in r.message for r in caplog.records)

    def test_an_ordinary_wait_says_nothing(self, caplog):
        """Every token goes through here, so the quiet case must stay quiet."""

        async def scenario():
            collector = StreamOutputCollector()
            task = asyncio.create_task(collector.get())
            await _let_the_loop_run()
            collector.put_nowait({"text": "prompt"})
            await task

        with caplog.at_level(logging.WARNING, logger="atom"):
            asyncio.run(scenario())
        assert not [r for r in caplog.records if "delivered nothing" in r.message]

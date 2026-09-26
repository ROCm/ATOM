# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the per-request stage trace.

The point of the trace is that a finished request carries the wall-clock time
it reached each transition, and that the whole thing costs nothing when it is
off. Both halves are tested here.
"""

import time

import pytest

from atom.model_engine import request_trace
from atom.model_engine.request_trace import STAGES


class FakeSequence:
    """Only the attributes the tracer reads."""

    def __init__(self, seq_id=1, request_id="ext-1"):
        self.id = seq_id
        self.external_request_id = request_id
        self.num_prompt_tokens = 1024
        self.num_cached_tokens = 256
        self.num_completion_tokens = 64
        self.arrive_time = 0.0
        self.first_token_time = 0.0
        self.leave_reason = ""


@pytest.fixture(autouse=True)
def _off_between_tests():
    """A module global that survives a test is a test that passes by accident."""
    request_trace.close()
    yield
    request_trace.close()


def rows(path):
    with open(path) as handle:
        lines = handle.read().splitlines()
    header = lines[0].split(",")
    return [dict(zip(header, line.split(","))) for line in lines[1:]]


def test_stamping_does_nothing_until_configured():
    seq = FakeSequence()
    request_trace.stamp(seq, "queued")
    request_trace.record(seq, "decode", "eos")
    assert not request_trace.enabled()
    # Not just "no row was written": no per-request state was allocated
    # either, which is what makes the off path free.
    assert not hasattr(seq, "_stage_times")


def test_a_finished_request_is_one_row_with_the_stages_it_visited(tmp_path):
    path = tmp_path / "requests.csv"
    request_trace.configure(str(path))

    seq = FakeSequence(seq_id=7, request_id="req-7")
    # Both of these are taken by the engine on the same clock the tracer uses.
    seq.arrive_time = time.time()
    for stage in ("queued", "blocks", "sched", "kv_park", "kv_ready", "first_decode"):
        request_trace.stamp(seq, stage)
    seq.first_token_time = time.time()
    request_trace.record(seq, "decode", "eos")
    request_trace.close()

    (row,) = rows(str(path))
    assert row["role"] == "decode"
    assert row["seq_id"] == "7"
    assert row["request_id"] == "req-7"
    assert row["reason"] == "eos"
    assert row["n_prompt"] == "1024"
    assert row["n_cached"] == "256"
    assert row["n_completion"] == "64"
    # The two timestamps the engine took on its own are carried through, not
    # re-taken: they belong to moments the tracer was not called at.
    # Microsecond resolution: finer than any stage this measures.
    assert float(row["arrive"]) == pytest.approx(seq.arrive_time, abs=1e-6)
    assert float(row["first_token"]) == pytest.approx(seq.first_token_time, abs=1e-6)
    # Every stage the request visited is ordered and non-decreasing.
    visited = [float(row[stage]) for stage in STAGES if row[stage]]
    assert visited == sorted(visited)
    assert len(visited) == len(STAGES)


def test_an_unvisited_stage_is_empty_rather_than_zero(tmp_path):
    path = tmp_path / "requests.csv"
    request_trace.configure(str(path))

    seq = FakeSequence()
    request_trace.stamp(seq, "queued")
    request_trace.record(seq, "prefill", "prefill_done")
    request_trace.close()

    (row,) = rows(str(path))
    assert row["queued"]
    # A monolithic run never parks on a remote KV transfer. Writing 0.0 there
    # would read as "parked at the epoch" and put a 56-year gap in the
    # timeline.
    assert row["kv_park"] == ""
    assert row["kv_ready"] == ""
    assert row["arrive"] == ""


def test_a_stage_keeps_its_first_arrival(tmp_path):
    path = tmp_path / "requests.csv"
    request_trace.configure(str(path))

    seq = FakeSequence()
    request_trace.stamp(seq, "sched")
    first = seq._stage_times["sched"]
    # A preempted request is re-queued and re-scheduled. The question the
    # trace answers is when it first got there.
    request_trace.stamp(seq, "sched")
    assert seq._stage_times["sched"] == first


def test_rows_are_readable_before_the_process_exits(tmp_path):
    path = tmp_path / "requests.csv"
    request_trace.configure(str(path))

    for i in range(request_trace._DRAIN_EVERY):
        request_trace.record(FakeSequence(seq_id=i), "decode", "eos")

    # No close(): a benchmark killed by its scheduler still has its rows.
    assert len(rows(str(path))) == request_trace._DRAIN_EVERY


def test_rows_append_across_drains_under_one_header(tmp_path):
    path = tmp_path / "requests.csv"
    request_trace.configure(str(path))

    request_trace.record(FakeSequence(seq_id=1), "decode", "eos")
    request_trace._tracer.drain()
    request_trace.record(FakeSequence(seq_id=2), "decode", "eos")
    request_trace.close()

    with open(str(path)) as handle:
        text = handle.read()
    assert text.count("role,seq_id") == 1
    assert [row["seq_id"] for row in rows(str(path))] == ["1", "2"]


def test_overflow_is_counted_rather_than_hidden(tmp_path):
    path = tmp_path / "requests.csv"
    # Smaller than the drain threshold, so the buffer wraps before it is
    # written and the loss is real.
    request_trace.configure(str(path), capacity=4)

    tracer = request_trace._tracer
    for i in range(10):
        tracer.record(FakeSequence(seq_id=i), "decode", "eos")
    assert tracer.dropped == 6


def test_a_comma_in_the_finish_reason_does_not_split_the_row(tmp_path):
    path = tmp_path / "requests.csv"
    request_trace.configure(str(path))

    request_trace.record(FakeSequence(), "decode", "stop_a,b")
    request_trace.close()

    (row,) = rows(str(path))
    assert row["reason"] == "stop_a;b"


def test_configure_is_idempotent(tmp_path):
    first = request_trace.configure(str(tmp_path / "a.csv"))
    second = request_trace.configure(str(tmp_path / "b.csv"))
    # Two schedulers in one process must not end up writing to two files,
    # each holding half the requests.
    assert first is second

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the LMCache offload copy-job trace.

The trace exists to measure the half of an offload copy that OFFLOAD_PROFILE
cannot see: the time a job spends in the executor queue before a worker picks
it up. These tests hold that measurement, and hold that a failed job still
produces a row -- a copy that throws is exactly the one worth seeing.
"""

import threading
import time

import pytest

from atom.kv_transfer.offload import offload_trace


class FakeReq:
    def __init__(self, req_id="req-1", load_operation="", save_operation=""):
        self.req_id = req_id
        self.load_operation = load_operation
        self.save_operation = save_operation


@pytest.fixture(autouse=True)
def _off_between_tests():
    offload_trace.close()
    yield
    offload_trace.close()


def rows(path):
    with open(path) as handle:
        lines = handle.read().splitlines()
    header = lines[0].split(",")
    return [dict(zip(header, line.split(","))) for line in lines[1:]]


def run_job(kind, req, ok=True, stats=None):
    job = offload_trace.begin(kind, req)
    if stats is not None:
        offload_trace.annotate(stats)
    offload_trace.end(job, ok)


def test_nothing_happens_until_configured():
    req = FakeReq()
    offload_trace.mark_submit(req)
    run_job("load", req)
    assert not offload_trace.enabled()
    # The submit mark is not even taken, so an unconfigured process does not
    # grow an attribute on every piece of job metadata it routes.
    assert not hasattr(req, offload_trace._SUBMIT_ATTR)


def test_a_job_row_carries_the_queue_wait_and_the_run(tmp_path):
    path = tmp_path / "offload.csv"
    offload_trace.configure(str(path), role="kv_consumer")

    req = FakeReq(req_id="req-9", load_operation="op-9")
    offload_trace.mark_submit(req)
    time.sleep(0.02)  # stand in for the job sitting in the pool
    run_job("load", req)
    offload_trace.close()

    (row,) = rows(str(path))
    assert row["role"] == "kv_consumer"
    assert row["kind"] == "load"
    assert row["req_id"] == "req-9"
    assert row["operation"] == "op-9"
    assert row["ok"] == "1"
    assert float(row["queue_ms"]) >= 20.0
    # The run is the part already profiled, and it is not the wait.
    assert float(row["run_ms"]) < float(row["queue_ms"])
    assert float(row["finish"]) >= float(row["submit"])


def test_a_job_nobody_marked_still_gets_a_row(tmp_path):
    path = tmp_path / "offload.csv"
    offload_trace.configure(str(path))

    run_job("save", FakeReq())
    offload_trace.close()

    (row,) = rows(str(path))
    assert float(row["run_ms"]) >= 0.0
    # Empty, not zero: "we never saw it submitted" is not "it waited 0 ms".
    assert row["queue_ms"] == ""
    assert row["submit"] == ""


def test_a_failed_job_is_recorded_as_failed(tmp_path):
    path = tmp_path / "offload.csv"
    offload_trace.configure(str(path))

    run_job("save", FakeReq(), ok=False)
    offload_trace.close()

    (row,) = rows(str(path))
    assert row["ok"] == "0"


def test_the_job_body_can_attach_what_it_measured(tmp_path):
    path = tmp_path / "offload.csv"
    offload_trace.configure(str(path))

    run_job(
        "load",
        FakeReq(),
        stats={"total_bytes": 4096, "copy_ms": 1.5, "unrelated": 7},
    )
    offload_trace.close()

    (row,) = rows(str(path))
    assert row["total_bytes"] == "4096"
    assert row["copy_ms"] == "1.500"
    # A layout that reports nothing for a column leaves it empty; a zero there
    # would read as a copy that moved no bytes.
    assert row["pack_ms"] == ""


def test_annotate_reaches_only_the_job_on_its_own_thread(tmp_path):
    path = tmp_path / "offload.csv"
    offload_trace.configure(str(path))

    started = threading.Event()

    def other():
        # The save and load pools run concurrently; a body annotating its own
        # job must not land on the other pool's row.
        run_job("save", FakeReq(req_id="save-1"), stats={"copy_ms": 9.0})
        started.set()

    job = offload_trace.begin("load", FakeReq(req_id="load-1"))
    thread = threading.Thread(target=other)
    thread.start()
    started.wait(timeout=5)
    thread.join()
    offload_trace.annotate({"copy_ms": 1.0})
    offload_trace.end(job, True)
    offload_trace.close()

    by_id = {row["req_id"]: row for row in rows(str(path))}
    assert by_id["save-1"]["copy_ms"] == "9.000"
    assert by_id["load-1"]["copy_ms"] == "1.000"


def test_rows_are_readable_before_the_process_exits(tmp_path):
    path = tmp_path / "offload.csv"
    offload_trace.configure(str(path))

    for i in range(offload_trace._DRAIN_EVERY):
        run_job("save", FakeReq(req_id="r%d" % i))

    assert len(rows(str(path))) == offload_trace._DRAIN_EVERY


def test_overflow_is_counted_rather_than_hidden(tmp_path):
    path = tmp_path / "offload.csv"
    tracer = offload_trace.configure(str(path), capacity=4)

    for i in range(10):
        run_job("save", FakeReq(req_id="r%d" % i))
    assert tracer.dropped == 6


def test_configure_is_idempotent(tmp_path):
    first = offload_trace.configure(str(tmp_path / "a.csv"))
    second = offload_trace.configure(str(tmp_path / "b.csv"))
    # Both executors are created on one worker; two files would split the
    # queue being measured across them.
    assert first is second

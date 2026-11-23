# SPDX-License-Identifier: MIT
"""Idle TBO workers must not pin completed forwards' GPU tensors, and a
released wrapper's workers stop."""

import gc
import threading
import time
import weakref

import torch

from atom.utils.tbo.ubatch_wrapper import UBatchWrapper


def test_idle_worker_releases_completed_job():
    class StopWorker(Exception):
        pass

    idle = threading.Event()
    release = threading.Event()

    class Ready:
        waits = 0

        def wait(self):
            self.waits += 1
            if self.waits > 1:
                idle.set()
                release.wait(timeout=5)
                raise StopWorker

        def clear(self):
            pass

    wrapper = UBatchWrapper(torch.nn.Identity())
    payload = torch.zeros(16)
    ref = weakref.ref(payload)
    calls = []

    def make_job(result):
        # Real jobs close over a results list containing microbatch tensors.
        results = [result]

        def job():
            calls.append(results[0].numel())

        return job

    wrapper._worker_jobs[0] = make_job(payload)
    wrapper._worker_job_ready[0] = Ready()
    del payload

    def worker():
        try:
            wrapper._worker_loop(0)
        except StopWorker:
            pass

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        assert wrapper._worker_job_done[0].wait(timeout=5)
        assert idle.wait(timeout=5)
        assert calls == [16]
        assert ref() is None, "idle worker still retains completed job outputs"
    finally:
        release.set()
        thread.join(timeout=5)
        assert not thread.is_alive()


def _live_workers() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith("tbo-ub-")]


def _run_on_workers(wrapper: UBatchWrapper) -> list[int]:
    """Hand each worker one job and wait for both, as a split forward does."""
    ran = []
    for idx in range(wrapper._num_workers):
        wrapper._worker_job_done[idx].clear()
        wrapper._worker_jobs[idx] = lambda idx=idx: ran.append(idx)
        wrapper._worker_job_ready[idx].set()
    for idx in range(wrapper._num_workers):
        assert wrapper._worker_job_done[idx].wait(timeout=5)
    return sorted(ran)


def test_a_released_wrapper_stops_its_workers():
    """Dropping the last reference to a wrapper stops and joins its worker
    pair: the threads return to the count before it was built, and they no
    longer keep the wrapper (and its model) alive."""
    baseline = len(_live_workers())
    wrapper = UBatchWrapper(torch.nn.Identity())
    wrapper._ensure_workers(None)
    workers = list(wrapper._workers)
    assert len(_live_workers()) == baseline + len(workers) == baseline + 2
    assert _run_on_workers(wrapper) == [0, 1]
    ref = weakref.ref(wrapper)

    del wrapper
    gc.collect()
    for worker in workers:
        worker.join(timeout=5)

    assert ref() is None, "the worker threads keep the wrapper alive"
    assert not any(worker.is_alive() for worker in workers)
    assert len(_live_workers()) == baseline


def test_many_released_wrappers_leave_no_workers():
    """Building and releasing wrapper after wrapper (one per test or per
    model) does not pile up parked worker pairs."""
    baseline = len(_live_workers())
    for _ in range(10):
        wrapper = UBatchWrapper(torch.nn.Identity())
        wrapper._ensure_workers(None)
        assert _run_on_workers(wrapper) == [0, 1]
        del wrapper
    gc.collect()
    deadline = time.monotonic() + 5
    while len(_live_workers()) > baseline and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(_live_workers()) == baseline


def test_stop_tbo_workers_joins_the_pair_and_a_later_split_restarts_it():
    """`stop_tbo_workers` (the runner's exit) joins both workers while the
    wrapper lives on; the next split forward starts a fresh pair that runs."""
    baseline = len(_live_workers())
    wrapper = UBatchWrapper(torch.nn.Identity())
    wrapper._ensure_workers(None)
    first = list(wrapper._workers)

    wrapper.stop_tbo_workers()

    assert not any(worker.is_alive() for worker in first)
    assert len(_live_workers()) == baseline
    wrapper.stop_tbo_workers()  # idempotent

    wrapper._ensure_workers(None)
    assert not set(wrapper._workers) & set(first)
    assert _run_on_workers(wrapper) == [0, 1]
    assert len(_live_workers()) == baseline + 2
    wrapper.stop_tbo_workers()
    assert len(_live_workers()) == baseline

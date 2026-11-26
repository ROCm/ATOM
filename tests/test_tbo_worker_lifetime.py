# SPDX-License-Identifier: MIT
"""Idle TBO workers must not pin completed forwards' GPU tensors."""

import threading
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

# SPDX-License-Identifier: MIT

"""A side-stream D2H must keep its default-stream source out of the allocator.

`tokenIDProcessor` copies the sampled ids and the MTP verdict (`num_rejected`,
`num_bonus`) to the host on `async_copy_stream`. The sources are fresh
default-stream allocations that the caller drops right after queuing the copy.
Unless the source is recorded on the copy stream, the caching allocator hands
its block to the next default-stream allocation, and a copy that runs late --
anything already queued on the copy stream -- reads that tensor instead. Under
MTP the stale `num_bonus` turns into `num_accepted_tokens`, which the GDN conv
kernel adds to the state offset unchecked: an out-of-bounds read.

Each case queues a large D2H on the copy stream first, so the copy under test
cannot run before the next allocation is written. The positive control shows
the scenario does race when the hold is disabled; without it a pass would say
nothing.
"""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a real GPU", allow_module_level=True)

from atom.model_engine.model_runner import tokenIDProcessor

DEV = torch.device("cuda:0")
BS = 32
GOOD = 2
STALE = 16385
TRIALS = 20

_BIG = None


def _processor():
    proc = object.__new__(tokenIDProcessor)
    proc.async_copy_stream = torch.cuda.Stream(DEV)
    proc.clean()
    return proc


def _block_copy_stream(proc):
    global _BIG
    if _BIG is None:
        dev = torch.empty(256 << 20, dtype=torch.uint8, device=DEV)
        _BIG = (dev, torch.empty(dev.shape, dtype=torch.uint8).pin_memory())
    dev, host = _BIG
    with torch.cuda.stream(proc.async_copy_stream):
        host.copy_(dev, non_blocking=True)


def _next_step_allocation():
    nxt = torch.empty((3, BS), dtype=torch.int32, device=DEV)
    nxt.fill_(STALE)
    return nxt


def _mtp_status_trial(proc):
    torch.cuda.synchronize()
    _block_copy_stream(proc)
    verdict = torch.empty((3, BS), dtype=torch.int32, device=DEV)
    verdict.fill_(GOOD)
    ready = torch.cuda.Event()
    ready.record()
    proc.send_mtp_status_to_cpu_async(verdict[1], verdict[0], ready)
    del verdict
    _next_step_allocation()
    num_rejected, num_bonus = proc.recv_mtp_status_async()
    return (num_rejected == GOOD).all() and (num_bonus == GOOD).all()


def _sampled_ids_trial(proc):
    torch.cuda.synchronize()
    _block_copy_stream(proc)
    sampled = torch.empty((3, BS), dtype=torch.int32, device=DEV)
    sampled.fill_(GOOD)
    ready = torch.cuda.Event()
    ready.record()
    proc.send_to_cpu_async(sampled, proc.token_ids_cpu, ready)
    del sampled
    _next_step_allocation()
    got = proc.recv_async_output(proc.token_ids_cpu)
    proc.recv_logprobs()
    return bool((got == GOOD).all())


def _draft_ids_trial(proc):
    torch.cuda.synchronize()
    _block_copy_stream(proc)
    draft = torch.empty((3, BS), dtype=torch.int32, device=DEV)
    draft.fill_(GOOD)
    proc.send_to_cpu_async_draft(draft)
    del draft
    _next_step_allocation()
    return bool((proc.recv_async_output_draft() == GOOD).all())


TRIAL_FNS = {
    "mtp_status": _mtp_status_trial,
    "sampled_ids": _sampled_ids_trial,
    "draft_ids": _draft_ids_trial,
}


@pytest.mark.parametrize("path", list(TRIAL_FNS))
def test_a_late_copy_still_reads_its_own_step(path):
    proc = _processor()
    trial = TRIAL_FNS[path]
    bad = sum(not trial(proc) for _ in range(TRIALS))
    assert bad == 0, f"{path}: {bad}/{TRIALS} copies read the next step's tensor"


def test_without_the_hold_the_scenario_races(monkeypatch):
    """Positive control: the copy really does run after the next allocation."""
    monkeypatch.setattr(tokenIDProcessor, "_hold_for_copy_stream", lambda *a: None)
    proc = _processor()
    bad = sum(not _mtp_status_trial(proc) for _ in range(TRIALS))
    assert bad > 0, (
        "no copy read a reused block with the hold disabled, so nothing here "
        "exercises the race and the test above proves nothing"
    )

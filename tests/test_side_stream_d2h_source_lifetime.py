# SPDX-License-Identifier: MIT

"""A side-stream D2H must keep its default-stream source out of the allocator.

`tokenIDProcessor` copies the sampled ids, the MTP verdict (`spec_counts`, a
view of the sampler's `verdict`) and the draft ids to the host on
`async_copy_stream`; DSpark's `VerifyScheduler.record_ell` copies `ell` on the
same stream. The sources are fresh default-stream allocations that the caller
drops right after queuing the copy. Unless the source is recorded on the copy
stream, the caching allocator hands its block to the next default-stream
allocation, and a copy that runs late -- anything already queued on the copy
stream -- reads that tensor instead. Under MTP the stale bonus count turns into
`num_accepted_tokens`, which the GDN conv kernel adds to the state offset
unchecked: an out-of-bounds read.

Each case first parks the copy stream on a spin kernel that outlasts the
case's host work, so the copy under test cannot run before the next
allocation is written. The positive control shows each scenario does race
when `record_stream` is disabled; without it a pass would say nothing.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a real GPU", allow_module_level=True)

from atom.model_engine.model_runner import tokenIDProcessor
from atom.spec_decode import dspark_verify
from atom.spec_decode.dspark_verify import VerifyScheduler

DEV = torch.device("cuda:0")
BS = 32
NUM_SPEC = 3
GOOD = 2
STALE = 16385
TRIALS = 20
# ~20 ms on MI355X; one case's host work takes well under 1 ms.
BLOCK_CYCLES = 50_000_000

_COPY_STREAM = None


def _copy_stream():
    """A stream that does not share a hardware queue with the default stream.

    HIP maps streams onto a few hardware queues. On one that shares the default
    stream's queue, the next step's write waits behind the parked copy, so the
    race cannot happen there and the positive control would fail for that
    reason alone.
    """
    global _COPY_STREAM
    if _COPY_STREAM is not None:
        return _COPY_STREAM
    probe = torch.empty(1, device=DEV)
    for _ in range(64):
        stream = torch.cuda.Stream(DEV)
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(BLOCK_CYCLES)
        probe.fill_(0)
        done = torch.cuda.Event()
        done.record()
        deadline = time.monotonic() + 0.005
        while not done.query() and time.monotonic() < deadline:
            pass
        own_queue = done.query() and not stream.query()
        stream.synchronize()
        if own_queue:
            _COPY_STREAM = stream
            return stream
    pytest.skip("no stream with its own hardware queue")


def _processor():
    proc = object.__new__(tokenIDProcessor)
    proc.async_copy_stream = _copy_stream()
    proc._output_request_capacity = BS
    proc.num_spec_tokens = NUM_SPEC
    proc.clean()
    return proc


def _verify_scheduler(proc):
    vs = object.__new__(VerifyScheduler)
    vs.runner = SimpleNamespace(
        tokenID_processor=proc, config=SimpleNamespace(max_num_seqs=BS)
    )
    vs._last_ell = None
    vs._ell_pending = []
    vs._ell_map = {}
    vs._ell_stage_ring = None
    vs._ell_stage_idx = 0
    vs._ell_slot_event = [None] * dspark_verify._MAX_ELL_INFLIGHT
    vs._ell_last_slot = None
    return vs


def _block_copy_stream(proc):
    with torch.cuda.stream(proc.async_copy_stream):
        torch.cuda._sleep(BLOCK_CYCLES)


def _next_step_allocation():
    nxt = torch.empty((3, BS), dtype=torch.int32, device=DEV)
    nxt.fill_(STALE)
    return nxt


def _spec_status_trial(proc):
    torch.cuda.synchronize()
    _block_copy_stream(proc)
    verdict = torch.empty((3, BS), dtype=torch.int32, device=DEV)
    verdict.fill_(GOOD)
    ready = torch.cuda.Event()
    ready.record()
    proc.spec_status_queue.send(verdict[:2], ready=ready)
    del verdict
    _next_step_allocation()
    num_rejected, num_bonus = proc.recv_spec_status()
    return bool((num_rejected == GOOD).all() and (num_bonus == GOOD).all())


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
    draft = torch.empty((BS, NUM_SPEC), dtype=torch.int32, device=DEV)
    draft.fill_(GOOD)
    proc.send_draft_token_ids_to_cpu(draft)
    del draft
    _next_step_allocation()
    return bool((proc.recv_draft_token_ids() == GOOD).all())


def _ell_trial(proc):
    torch.cuda.synchronize()
    vs = _verify_scheduler(proc)
    _block_copy_stream(proc)
    ell = torch.empty(BS, dtype=torch.int64, device=DEV)
    ell.fill_(GOOD)
    vs.set_last_ell(ell)
    vs.record_ell(list(range(BS)))
    del ell
    vs._last_ell = None
    _next_step_allocation()
    event, cpu_buf, _ = vs._ell_pending[-1]
    event.synchronize()
    return bool((cpu_buf == GOOD).all())


TRIAL_FNS = {
    "spec_status": _spec_status_trial,
    "sampled_ids": _sampled_ids_trial,
    "draft_ids": _draft_ids_trial,
    "dspark_ell": _ell_trial,
}


@pytest.mark.parametrize("path", list(TRIAL_FNS))
def test_a_late_copy_still_reads_its_own_step(path):
    proc = _processor()
    trial = TRIAL_FNS[path]
    bad = sum(not trial(proc) for _ in range(TRIALS))
    assert bad == 0, f"{path}: {bad}/{TRIALS} copies read the next step's tensor"


@pytest.mark.parametrize("path", list(TRIAL_FNS))
def test_without_the_hold_the_scenario_races(monkeypatch, path):
    """Positive control: the copy really does run after the next allocation."""
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *a: None)
    proc = _processor()
    trial = TRIAL_FNS[path]
    bad = sum(not trial(proc) for _ in range(TRIALS))
    assert bad > 0, (
        f"{path}: no copy read a reused block with record_stream disabled, so "
        "nothing here exercises the race and the test above proves nothing"
    )

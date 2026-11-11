# SPDX-License-Identifier: MIT

"""A side-stream D2H must keep its default-stream source alive until it ran.

`tokenIDProcessor` copies the sampled ids and logprobs, the MTP verdict
(`spec_counts`, a view of the sampler's `verdict`) and the draft ids to the
host on `async_copy_stream`; DSpark's `VerifyScheduler.record_ell` copies `ell`
on the same stream. The sources are fresh default-stream allocations that the
caller drops right after queuing the copy. The caching allocator does not see
the side-stream read, so unless something holds the source until the copy is
known to have completed, its block goes to the next default-stream allocation,
and a copy that runs late -- anything already queued on the copy stream --
reads that tensor instead. Under MTP the stale bonus count turns into
`num_accepted_tokens`, which the GDN conv kernel adds to the state offset
unchecked: an out-of-bounds read.

Each case first parks the copy stream on a spin kernel that outlasts the
case's host work, so the copy under test cannot run before the next
allocation is written. The positive controls strip the holds (and the wait in
`clean()`) and show each scenario does race; without them a pass would say
nothing.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a real GPU", allow_module_level=True)

from atom.model_engine.model_runner import tokenIDProcessor
from atom.model_engine.output_transfer import D2HQueue
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
_FORWARD_DONE = None


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


def _forward_done():
    """Stands in for the runner's forward_done_event: one event, re-recorded."""
    global _FORWARD_DONE
    if _FORWARD_DONE is None:
        _FORWARD_DONE = torch.cuda.Event()
    _FORWARD_DONE.record()
    return _FORWARD_DONE


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
    vs._ell_slot_source = [None] * dspark_verify._MAX_ELL_INFLIGHT
    vs._ell_oneoff_inflight = []
    vs._ell_last_slot = None
    return vs


def _block_copy_stream(proc):
    with torch.cuda.stream(proc.async_copy_stream):
        torch.cuda._sleep(BLOCK_CYCLES)


def _fresh(shape, dtype):
    t = torch.empty(shape, dtype=dtype, device=DEV)
    t.fill_(GOOD)
    return t


def _next_step_allocations():
    # Several same-size blocks, kept alive together, so every source freed in
    # this case is reused and overwritten.
    nxt = [torch.empty((3, BS), dtype=torch.int32, device=DEV) for _ in range(4)]
    for t in nxt:
        t.fill_(STALE)
    return nxt


def _spec_status_trial(proc):
    torch.cuda.synchronize()
    _block_copy_stream(proc)
    verdict = _fresh((3, BS), torch.int32)
    ready = _forward_done()
    proc.spec_status_queue.send(verdict[:2], ready=ready)
    del verdict
    _next_step_allocations()
    num_rejected, num_bonus = proc.recv_spec_status()
    return bool((num_rejected == GOOD).all() and (num_bonus == GOOD).all())


def _sampled_ids_trial(proc):
    torch.cuda.synchronize()
    _block_copy_stream(proc)
    sampled = _fresh((3, BS), torch.int32)
    ready = _forward_done()
    proc.send_to_cpu_async(sampled, proc.token_ids_cpu, ready)
    del sampled
    _next_step_allocations()
    got = proc.recv_async_output(proc.token_ids_cpu)
    proc.recv_logprobs()
    return bool((got == GOOD).all())


def _logprobs_trial(proc):
    torch.cuda.synchronize()
    _block_copy_stream(proc)
    sampled = _fresh((3, BS), torch.int32)
    logprobs = _fresh(BS, torch.float32)
    ready = _forward_done()
    proc.send_to_cpu_async(sampled, proc.token_ids_cpu, ready, gpu_logprobs=logprobs)
    del sampled, logprobs
    _next_step_allocations()
    proc.recv_async_output(proc.token_ids_cpu)
    return all(v == GOOD for v in proc.recv_logprobs())


def _draft_ids_trial(proc):
    torch.cuda.synchronize()
    _block_copy_stream(proc)
    draft = _fresh((BS, NUM_SPEC), torch.int32)
    proc.send_draft_token_ids_to_cpu(draft)
    del draft
    _next_step_allocations()
    return bool((proc.recv_draft_token_ids() == GOOD).all())


def _record_fresh_ell(vs):
    ell = _fresh(BS, torch.int64)
    vs.set_last_ell(ell)
    vs.record_ell(list(range(BS)))
    vs.set_last_ell(None)


def _ell_trial(proc):
    torch.cuda.synchronize()
    vs = _verify_scheduler(proc)
    _block_copy_stream(proc)
    _record_fresh_ell(vs)
    assert vs._ell_last_slot == 0
    _next_step_allocations()
    event, cpu_buf, _ = vs._ell_pending[-1]
    event.synchronize()
    return bool((cpu_buf == GOOD).all())


def _ell_oneoff_trial(proc):
    """The ring slot is still busy, so the copy lands in a one-off buffer."""
    torch.cuda.synchronize()
    vs = _verify_scheduler(proc)
    _block_copy_stream(proc)
    _record_fresh_ell(vs)
    vs._ell_stage_idx = 0
    _record_fresh_ell(vs)
    assert vs._ell_last_slot is None
    _next_step_allocations()
    event, cpu_buf, _ = vs._ell_pending[-1]
    event.synchronize()
    return bool((cpu_buf == GOOD).all())


def _clean_trial(proc):
    """clean() drops every queued entry while the copies are still parked."""
    torch.cuda.synchronize()
    _block_copy_stream(proc)
    sampled = _fresh((3, BS), torch.int32)
    logprobs = _fresh(BS, torch.float32)
    verdict = _fresh((3, BS), torch.int32)
    draft = _fresh((BS, NUM_SPEC), torch.int32)
    ready = _forward_done()
    proc.send_to_cpu_async(sampled, proc.token_ids_cpu, ready, gpu_logprobs=logprobs)
    proc.spec_status_queue.send(verdict[:2], ready=ready)
    proc.send_draft_token_ids_to_cpu(draft)
    host = [
        proc.token_ids_cpu[0][0],
        proc.logprobs_cpu[0],
        proc.spec_status_queue._pending[0].tensor_view,
        proc._draft_token_ids_queue._pending[0].tensor_view,
    ]
    del sampled, logprobs, verdict, draft
    proc.clean()
    nxt = _next_step_allocations()
    torch.cuda.synchronize()
    del nxt
    return all(bool((h == GOOD).all()) for h in host)


TRIAL_FNS = {
    "spec_status": _spec_status_trial,
    "sampled_ids": _sampled_ids_trial,
    "logprobs": _logprobs_trial,
    "draft_ids": _draft_ids_trial,
    "dspark_ell": _ell_trial,
    "dspark_ell_oneoff": _ell_oneoff_trial,
    "clean": _clean_trial,
}


def _remove_holds(monkeypatch):
    """Drop every source reference right after its copy is queued."""
    send = D2HQueue.send
    send_to_cpu_async = tokenIDProcessor.send_to_cpu_async
    record_ell = VerifyScheduler.record_ell

    def queue_send(self, source, **kwargs):
        send(self, source, **kwargs)
        self._pending[-1].source = None

    def processor_send(self, gpu_tensor, cpu_tensor_handle, *args, **kwargs):
        send_to_cpu_async(self, gpu_tensor, cpu_tensor_handle, *args, **kwargs)
        cpu_tensor, copy_done, _ = cpu_tensor_handle[-1]
        cpu_tensor_handle[-1] = (cpu_tensor, copy_done, ())

    def verify_record_ell(self, req_ids):
        record_ell(self, req_ids)
        self._ell_slot_source = [None] * dspark_verify._MAX_ELL_INFLIGHT
        self._ell_oneoff_inflight = []

    monkeypatch.setattr(D2HQueue, "send", queue_send)
    monkeypatch.setattr(tokenIDProcessor, "send_to_cpu_async", processor_send)
    monkeypatch.setattr(VerifyScheduler, "record_ell", verify_record_ell)


@pytest.mark.parametrize("path", list(TRIAL_FNS))
def test_a_late_copy_still_reads_its_own_step(path):
    proc = _processor()
    trial = TRIAL_FNS[path]
    bad = sum(not trial(proc) for _ in range(TRIALS))
    assert bad == 0, f"{path}: {bad}/{TRIALS} copies read the next step's tensor"


@pytest.mark.parametrize("path", list(TRIAL_FNS))
def test_without_the_hold_the_scenario_races(monkeypatch, path):
    """Positive control: the copy really does run after the next allocation."""
    if path == "clean":
        # The holds stay; clean() drops them without waiting for the copies.
        monkeypatch.setattr(
            tokenIDProcessor, "_wait_for_pending_copies", lambda self: None
        )
    else:
        _remove_holds(monkeypatch)
    proc = _processor()
    trial = TRIAL_FNS[path]
    bad = sum(not trial(proc) for _ in range(TRIALS))
    assert bad > 0, (
        f"{path}: no copy read a reused block with the holds removed, so "
        "nothing here exercises the race and the test above proves nothing"
    )

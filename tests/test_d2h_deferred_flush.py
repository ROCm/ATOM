# SPDX-License-Identifier: MIT
"""Deferred D2H sends: copied at the flush (behind the event it is given), or at
a receive that comes first, in FIFO order, with the queue's `on_copy` called
once each copy is enqueued (GPU only)."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a real GPU", allow_module_level=True)

from atom.model_engine.model_runner import tokenIDProcessor
from atom.model_engine.output_transfer import D2HQueue


def _queue(on_copy=None):
    return D2HQueue(64, torch.cuda.Stream(), on_copy=on_copy)


def test_flush_behind_the_given_event_keeps_fifo_order():
    copied = []
    q = _queue(on_copy=copied.append)
    a = torch.full((2, 4), 3, dtype=torch.int32, device="cuda")
    b = torch.full((2, 4), 5, dtype=torch.int32, device="cuda")
    q.send(a, defer=True)
    q.send(b, defer=True)
    assert q.has_deferred() and not copied
    a.fill_(7)  # written after the send, before the event the flush waits for
    ready = torch.cuda.Event()
    ready.record()
    q.flush(ready)
    assert not q.has_deferred() and len(copied) == 2
    assert (q.receive() == 7).all() and (q.receive() == 5).all()
    assert q.receive() is None


def test_a_receive_before_the_flush_flushes():
    q = _queue()
    src = torch.full((3,), 9, dtype=torch.int32, device="cuda")
    q.send(src, defer=True)
    assert (q.receive() == 9).all()


def test_processor_defers_ids_status_and_drafts_to_one_event():
    proc = object.__new__(tokenIDProcessor)
    proc.async_copy_stream = torch.cuda.Stream()
    proc._ids_pinned_ring = [None] * 8  # what __init__ sets before clean()
    proc._ids_pinned_next = 0
    proc._output_request_capacity = 8
    proc.num_spec_tokens = 3
    proc.clean()
    ids = torch.full((1, 4), 11, dtype=torch.int32, device="cuda")
    status = torch.full((2, 1), 2, dtype=torch.int32, device="cuda")
    drafts = torch.full((1, 3), 13, dtype=torch.int32, device="cuda")
    proc.send_to_cpu_async(ids, proc.token_ids_cpu, None)
    proc.spec_status_queue.send(status, defer=True)
    proc.send_draft_token_ids_to_cpu(drafts, defer=True)
    assert not proc.token_ids_cpu
    ready = torch.cuda.Event()
    ready.record()
    proc.flush_deferred_sends(ready)
    assert len(proc.token_ids_cpu) == 1
    assert (proc.recv_async_output(proc.token_ids_cpu) == 11).all()
    rejected, bonus = proc.recv_spec_status()
    assert rejected.tolist() == [2] and bonus.tolist() == [2]
    assert (proc.recv_draft_token_ids() == 13).all()
    # A receive before the flush enqueues the copy itself.
    proc.send_to_cpu_async(ids, proc.token_ids_cpu, None)
    assert (proc.recv_async_output(proc.token_ids_cpu) == 11).all()

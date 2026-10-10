# SPDX-License-Identifier: MIT
"""Parent failures must drain TBO workers before metadata is released."""

import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")

from atom.utils.forward_context import _forward_context_local
from atom.utils.tbo import ubatch_wrapper, ubatching
from atom.utils.tbo.ubatch_splitting import UBatchSlice


@pytest.mark.parametrize("phase", ["barrier", "first_turn", "worker_wait"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_parent_failure_drains_workers_before_fencing(monkeypatch, phase, error_type):
    # GPU operations are inert; rendezvous, worker jobs and TBO contexts are real.
    stream = SimpleNamespace(wait_event=lambda event: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: stream)
    monkeypatch.setattr(torch.cuda, "set_stream", lambda value: None)
    monkeypatch.setattr(torch.cuda, "set_device", lambda value: None)
    monkeypatch.setattr(
        torch, "Event", lambda: SimpleNamespace(record=lambda value: None)
    )
    monkeypatch.setattr(ubatching, "_CURRENT_CONTEXTS", [None, None])
    monkeypatch.setattr(ubatching, "_THREAD_ID_TO_CONTEXT", {})
    parent_thread = threading.get_ident()
    failure = error_type("parent interrupted")
    inject = [True]
    release_model = threading.Event()
    model_started = threading.Event()
    captured = []
    fences = []
    children = [SimpleNamespace(index=i) for i in range(2)]
    saved_context = object()
    monkeypatch.setattr(_forward_context_local, "ctx", saved_context, raising=False)

    class Barrier(threading.Barrier):
        def wait(self, timeout=None):
            parent = threading.get_ident() == parent_thread
            if parent and inject[0] and phase == "barrier":
                inject[0] = False
                raise failure
            value = super().wait(timeout)
            if parent and inject[0] and phase == "first_turn":
                inject[0] = False
                raise failure
            return value

    class Model(torch.nn.Module):
        def forward(self, ids, positions, inputs_embeds=None):
            model_started.set()
            assert release_model.wait(5), "test did not release model"
            for _ in range(2):
                ubatching.tbo_yield()
            return ids + 1

    @contextmanager
    def forward_scope(metadata):
        try:
            yield
        finally:
            fences.append(all(context.done for context in captured[-1]))

    wrapper = ubatch_wrapper.UBatchWrapper(
        Model(), SimpleNamespace(ubatch_forward=forward_scope)
    )
    wrapper.comm_stream = stream
    wrapper.ready_barrier = Barrier(3)
    monkeypatch.setattr(wrapper, "_make_ubatch_dp_metadata", lambda ctx, count: None)
    monkeypatch.setattr(wrapper, "_compute_ub_running_tokens", lambda *args: [2, 2])
    monkeypatch.setattr(wrapper, "_ub_tokens_across_dp", lambda *args: None)
    monkeypatch.setattr(
        wrapper,
        "_make_ubatch_context",
        lambda original, part, bs, index, *args, **kwargs: children[index],
    )
    original_make = ubatch_wrapper.make_tbo_contexts

    def make_contexts(**kwargs):
        contexts = original_make(**kwargs)
        captured.append(contexts)
        return contexts

    monkeypatch.setattr(ubatch_wrapper, "make_tbo_contexts", make_contexts)
    original_wait = wrapper._worker_job_done[0].wait

    def wait_for_worker(timeout=None):
        if inject[0] and phase == "worker_wait":
            inject[0] = False
            assert model_started.wait(5)
            release_model.set()
            raise failure
        return original_wait(timeout)

    monkeypatch.setattr(wrapper._worker_job_done[0], "wait", wait_for_worker)
    ctx = SimpleNamespace(
        context=SimpleNamespace(is_prefill=True, running_bs=2),
        attn_metadata=None,
        ubatch_slices=[
            UBatchSlice(slice(0, 1), slice(0, 2)),
            UBatchSlice(slice(1, 2), slice(2, 4)),
        ],
    )
    ids = torch.arange(4)
    try:
        with pytest.raises(error_type, match="parent interrupted") as caught:
            wrapper._run_ubatches(ids, ids, ctx)
        assert caught.value is failure
        assert fences == [True], "parent fenced metadata before workers exited"
        assert all(event.is_set() for event in wrapper._worker_job_done)
        assert _forward_context_local.ctx is saved_context
        assert not ubatching._THREAD_ID_TO_CONTEXT
        assert ubatching._CURRENT_CONTEXTS == [None, None]
        assert all(
            context.partner is None and context.forward_context is None
            for context in captured[-1]
        )
        # The same persistent worker pool and barrier must accept another batch.
        release_model.set()
        actual = wrapper._run_ubatches(ids, ids, ctx)
        torch.testing.assert_close(actual, ids + 1)
        assert fences == [True, True]
        assert _forward_context_local.ctx is saved_context
    finally:
        # Also let the unpatched negative control terminate after its assertion.
        inject[0] = False
        wrapper.ready_barrier.abort()
        release_model.set()
        if captured:
            for i, context in enumerate(captured[-1]):
                context.partner = captured[-1][1 - i]
                context.forward_context = children[i]
                context.cpu_wait_event.set()
            for event in wrapper._worker_job_done:
                event.wait(5)

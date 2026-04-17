# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import threading
from typing import Callable, Optional

import torch

from atom.config import get_current_atom_config


def tbo_overlap_enabled() -> bool:
    return False


def tbo_enabled() -> bool:
    config = get_current_atom_config()
    if config is None:
        return False
    return getattr(config, "enable_tbo", False)


def sync_dp_for_tbo(
    dp_size: int,
    dp_rank: int,
    num_input_tokens: int,
    num_reqs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sync token counts and request counts across DP ranks in one all_reduce.

    Returns (num_tokens_across_dp, reqs_across_dp).
    reqs_across_dp is used to decide TBO on/off (min_reqs >= 2).
    """
    import torch.distributed as dist
    from aiter.dist.parallel_state import get_dp_group

    # Layout: [tokens_r0, ..., tokens_rN, reqs_r0, ..., reqs_rN]
    sync_tensor = torch.zeros(dp_size * 2, device="cpu", dtype=torch.int32)
    sync_tensor[dp_rank] = num_input_tokens
    sync_tensor[dp_size + dp_rank] = num_reqs
    dist.all_reduce(sync_tensor, group=get_dp_group().cpu_group)
    return sync_tensor[:dp_size], sync_tensor[dp_size:]


_THREAD_ID_TO_CONTEXT: dict[int, int] = {}
_CURRENT_CONTEXTS: list["TBOContext | None"] = []
_NUM_UBATCHES: int = 2


class TBOContext:
    """Context manager for micro-batch dual-thread overlap.

    Modelled after vLLM's ``UBatchContext``.  Each ubatch thread enters
    its own ``TBOContext`` as a context manager; synchronisation between
    threads uses threading events arranged in a circular ring:

        cpu_signal_event[i] == cpu_wait_event[(i+1) % N]

    so setting ``self.cpu_signal_event`` wakes the *next* thread while
    ``self.cpu_wait_event.wait()`` sleeps the *current* thread.

    GPU synchronisation uses ``torch.Event`` objects for non-blocking
    stream-to-stream ordering (no CPU-blocking synchronize calls).
    """

    def __init__(
        self,
        ubatch_id: int,
        compute_stream: torch.cuda.Stream,
        comm_stream: torch.cuda.Stream,
        forward_context,  # ForwardContext for this ubatch
        ready_barrier: threading.Barrier,
        cpu_wait_event: threading.Event,
        cpu_signal_event: threading.Event,
        gpu_comm_done_event: torch.Event,
        gpu_compute_done_event: torch.Event,
    ):
        self.ubatch_id = ubatch_id
        self.compute_stream = compute_stream
        self.comm_stream = comm_stream
        self.forward_context = forward_context
        self.ready_barrier = ready_barrier
        self.cpu_wait_event = cpu_wait_event
        self.cpu_signal_event = cpu_signal_event
        self.gpu_comm_done_event = gpu_comm_done_event
        self.gpu_compute_done_event = gpu_compute_done_event
        self.current_stream = compute_stream
        self.recv_hook: Optional[Callable] = None

    # -- context manager protocol ----------------------------------------

    def __enter__(self):
        global _CURRENT_CONTEXTS, _THREAD_ID_TO_CONTEXT
        _THREAD_ID_TO_CONTEXT[threading.get_ident()] = self.ubatch_id
        _CURRENT_CONTEXTS[self.ubatch_id] = self

        # All threads reach the barrier, then the main thread wakes thread 0
        self.ready_barrier.wait()

        # Wait for our turn (thread 0 is woken by the main thread)
        self.cpu_wait_event.wait()
        self.cpu_wait_event.clear()

        self._restore_context()
        self.update_stream(self.compute_stream)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _CURRENT_CONTEXTS, _THREAD_ID_TO_CONTEXT
        _CURRENT_CONTEXTS[self.ubatch_id] = None
        del _THREAD_ID_TO_CONTEXT[threading.get_ident()]
        self.maybe_run_recv_hook()
        # No CPU-blocking synchronize — GPU ordering is handled by
        # torch.Event record/wait in switch_to_comm_sync / switch_to_compute_sync.
        self.cpu_signal_event.set()
        self.cpu_wait_event.clear()
        return False

    # -- stream management ------------------------------------------------

    def update_stream(self, stream):
        self.current_stream = stream
        torch.cuda.set_stream(self.current_stream)

    # -- GPU event sync (non-blocking, no CPU synchronize) ---------------

    def _signal_comm_done(self):
        self.gpu_comm_done_event.record(self.comm_stream)

    def _signal_compute_done(self):
        self.gpu_compute_done_event.record(self.compute_stream)

    def _wait_compute_done(self):
        self.comm_stream.wait_event(self.gpu_compute_done_event)

    def _wait_comm_done(self):
        self.compute_stream.wait_event(self.gpu_comm_done_event)

    def switch_to_comm_sync(self):
        """Switch from compute to comm stream with GPU event ordering."""
        self._signal_compute_done()
        self.update_stream(self.comm_stream)
        self._wait_compute_done()

    def switch_to_compute_sync(self):
        """Switch from comm to compute stream with GPU event ordering."""
        self._signal_comm_done()
        self.update_stream(self.compute_stream)
        self._wait_comm_done()

    # -- forward context --------------------------------------------------

    def _restore_context(self):
        from atom.utils.forward_context import _forward_context_local

        _forward_context_local.ctx = self.forward_context

    # -- CPU yield (ping-pong) -------------------------------------------

    def _cpu_yield(self):
        """Wake the next thread and sleep until woken."""
        self.cpu_signal_event.set()
        self.cpu_wait_event.wait()
        self.cpu_wait_event.clear()
        self._restore_context()

    def yield_(self):
        """Yield CPU, preserving current stream."""
        self._cpu_yield()
        self.update_stream(self.current_stream)

    def yield_and_switch_from_compute_to_comm(self):
        """Record compute-done, yield, switch to comm stream."""
        self._signal_compute_done()
        self._cpu_yield()
        self.update_stream(self.comm_stream)
        self._wait_compute_done()

    def yield_and_switch_from_comm_to_compute(self):
        """Record comm-done, yield, switch to compute stream."""
        self._signal_comm_done()
        self._cpu_yield()
        self.update_stream(self.compute_stream)
        self._wait_comm_done()

    # -- recv hook --------------------------------------------------------

    def maybe_run_recv_hook(self):
        if self.recv_hook is not None:
            self.recv_hook()
            self.recv_hook = None


def tbo_active() -> bool:
    """True if current thread is running inside TBO dual-thread execution."""
    return threading.get_ident() in _THREAD_ID_TO_CONTEXT


def tbo_current_ubatch_id() -> int:
    return _THREAD_ID_TO_CONTEXT.get(threading.get_ident(), 0)


def _get_current_tbo_context() -> "TBOContext":
    ctx_idx = _THREAD_ID_TO_CONTEXT[threading.get_ident()]
    return _CURRENT_CONTEXTS[ctx_idx]


def tbo_yield():
    """Yield CPU to the other ubatch thread."""
    if not tbo_active():
        return
    _get_current_tbo_context().yield_()


def tbo_register_recv_hook(hook: Callable):
    """Register a recv completion hook on the NEXT ubatch's context."""
    ctx_idx = _THREAD_ID_TO_CONTEXT[threading.get_ident()]
    next_ctx = _CURRENT_CONTEXTS[(ctx_idx + 1) % _NUM_UBATCHES]
    next_ctx.recv_hook = hook


def tbo_maybe_run_recv_hook():
    """Run any pending recv hook from the other ubatch."""
    if not tbo_active():
        return
    _get_current_tbo_context().maybe_run_recv_hook()


def tbo_get_comm_stream() -> torch.cuda.Stream:
    return _get_current_tbo_context().comm_stream


def tbo_get_compute_stream() -> torch.cuda.Stream:
    return _get_current_tbo_context().compute_stream


def tbo_yield_and_switch_from_compute_to_comm():
    """Record compute-done event, yield to other thread, switch to comm stream."""
    _get_current_tbo_context().yield_and_switch_from_compute_to_comm()


def tbo_switch_to_compute_sync():
    """Switch from comm stream back to compute stream with GPU event sync."""
    _get_current_tbo_context().switch_to_compute_sync()


def tbo_yield_and_switch_from_comm_to_compute():
    """Record comm-done event, yield to other thread, switch to compute stream."""
    _get_current_tbo_context().yield_and_switch_from_comm_to_compute()


def tbo_switch_to_compute():
    """Switch to compute stream without sync (non-blocking)."""
    _get_current_tbo_context().update_stream(_get_current_tbo_context().compute_stream)


def tbo_switch_to_comm():
    """Switch to comm stream without sync."""
    _get_current_tbo_context().update_stream(_get_current_tbo_context().comm_stream)


def make_tbo_contexts(
    num_micro_batches: int,
    compute_stream: torch.cuda.Stream,
    comm_stream: torch.cuda.Stream,
    forward_contexts: list,
    ready_barrier: threading.Barrier,
) -> list[TBOContext]:
    """Create TBOContext instances for all micro-batches.

    Threading events are arranged in a ring so that each context's
    ``cpu_signal_event`` is the *next* context's ``cpu_wait_event``.
    """
    global _NUM_UBATCHES, _CURRENT_CONTEXTS
    assert num_micro_batches > 1

    _NUM_UBATCHES = num_micro_batches
    # Grow the global context list if needed
    while len(_CURRENT_CONTEXTS) < num_micro_batches:
        _CURRENT_CONTEXTS.append(None)

    cpu_events = [threading.Event() for _ in range(num_micro_batches)]
    gpu_comm_done_events = [torch.Event() for _ in range(num_micro_batches)]
    gpu_compute_done_events = [torch.Event() for _ in range(num_micro_batches)]

    ctxs = []
    for i in range(num_micro_batches):
        ctx = TBOContext(
            ubatch_id=i,
            compute_stream=compute_stream,
            comm_stream=comm_stream,
            forward_context=forward_contexts[i],
            ready_barrier=ready_barrier,
            cpu_wait_event=cpu_events[i],
            cpu_signal_event=cpu_events[(i + 1) % num_micro_batches],
            gpu_comm_done_event=gpu_comm_done_events[i],
            gpu_compute_done_event=gpu_compute_done_events[i],
        )
        ctxs.append(ctx)

    return ctxs

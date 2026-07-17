# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import threading
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from atom.config import get_current_atom_config


def tbo_overlap_enabled() -> bool:
    return False


def tbo_enabled() -> bool:
    config = get_current_atom_config()
    if config is None:
        return False
    return getattr(config, "enable_tbo", False)


# =====================================================================
# DP-side TBO helpers (formerly ModelRunner._local_tbo_precompute and
# the inline sync block inside ModelRunner._preprocess).
# =====================================================================


def _precompute_prefill_token_split(
    num_scheduled_tokens: np.ndarray,
    num_pref_reqs: int,
    min_pref: int,
) -> tuple[bool, bool, int, int]:
    """Prefill, token-midpoint split (ATOM_TBO_PREFILL_TOKEN_SPLIT=1).

    Cut at the exact token midpoint — this can slice THROUGH a request, so
    request count is irrelevant (bs==1 still splits). Only the token total
    matters. MUST mirror `_split_prefill_token_midpoint` in
    ubatch_splitting.py, or the cross-DP MAX-reduced ub0/ub1 disagree with
    the realised slices → all_gather size mismatch → RCCL hang.
    """
    num_pref_tokens = int(
        np.asarray(num_scheduled_tokens[:num_pref_reqs], dtype=np.int64).sum()
    )
    # can_split: need >= 2 tokens to cut into two non-empty halves.
    if num_pref_tokens < 2:
        return False, False, 0, 0
    # meets_min_tokens: this rank's prefill reached the min-token bar
    # (ATOM_TBO_PREFILL_MIN_TOKENS, e.g. 8k) — big enough to be worth TBO.
    meets_min_tokens = not (min_pref > 0 and num_pref_tokens < min_pref)
    ub0 = num_pref_tokens // 2
    ub1 = num_pref_tokens - ub0
    return meets_min_tokens, True, ub0, ub1


def _precompute_prefill_req_split(
    num_scheduled_tokens: np.ndarray,
    num_pref_reqs: int,
    min_pref: int,
) -> tuple[bool, bool, int, int]:
    """Prefill, request-boundary balanced split (ATOM_TBO_PREFILL_TOKEN_SPLIT=0).

    Split on a request boundary closest to the token midpoint, so each ubatch
    gets a whole number of requests. Needs >= 2 requests to put at least one in
    each ubatch.

    Same two-bit contract as the token-split path, only the can_split structural
    gate differs: here it's bs>=2 (request boundaries) instead of tokens>=2.
    The min-token (8k) bar stays the soft OR-reduced `meets_min_tokens`, so one
    rank clearing it turns TBO on for all and under-filled peers force-split.
    """
    # can_split: need >= 2 requests to give each ubatch at least one.
    if num_pref_reqs < 2:
        return False, False, 0, 0
    toks = np.asarray(num_scheduled_tokens[:num_pref_reqs], dtype=np.int64)
    num_pref_tokens = int(toks.sum())
    # meets_min_tokens: this rank's prefill reached the min-token bar
    # (ATOM_TBO_PREFILL_MIN_TOKENS, e.g. 8k) — big enough to be worth TBO.
    meets_min_tokens = not (min_pref > 0 and num_pref_tokens < min_pref)
    target = num_pref_tokens // 2
    cumsum = 0
    split_idx = 1
    for j in range(num_pref_reqs):
        cumsum += int(toks[j])
        if cumsum >= target:
            prev = cumsum - int(toks[j])
            if j > 0 and (target - prev) < (cumsum - target):
                split_idx = j
            else:
                split_idx = j + 1
            break
    split_idx = max(1, min(split_idx, num_pref_reqs - 1))
    ub0 = int(toks[:split_idx].sum())
    ub1 = num_pref_tokens - ub0
    return meets_min_tokens, True, ub0, ub1


def _precompute_decode(batch) -> tuple[bool, bool, int, int]:
    """Decode, split by request count (uniform tokens per request).

    can_split gate is bs > 2: it must be at least as strict as the strictest
    backend's per-ubatch metadata / CUDAGraph-capture threshold, or a rank can
    report "can_split" for a batch size that has no captured graph / no prepared
    ubatch meta, tripping build_ubatch_metadata's assert. Decode ubatch graphs
    are captured only for bs > 2 (model_runner) and V4 prepares meta only for
    scheduled_bs > 2. Decode has no min-token bar, so meets_min_tokens is always
    True here — if it can split, it's worth splitting.
    """
    scheduled_bs = batch.total_seqs_num_decode
    if scheduled_bs <= 2:
        return False, False, 0, 0
    num_tokens = batch.total_tokens_num
    tokens_per_req = num_tokens // scheduled_bs if scheduled_bs else 1
    reqs_per_ub = scheduled_bs // 2
    ub0 = reqs_per_ub * tokens_per_req
    ub1 = num_tokens - ub0
    return True, True, ub0, ub1


def local_tbo_precompute(
    config,
    batch,
    is_prefill: bool,
    num_scheduled_tokens: np.ndarray,
) -> tuple[bool, bool, int, int]:
    """Decide locally this rank's TBO status, split into two bits.

    Dispatches to exactly one of three paths:
      * prefill + ATOM_TBO_PREFILL_TOKEN_SPLIT=1 -> token-midpoint split
      * prefill + ATOM_TBO_PREFILL_TOKEN_SPLIT=0 -> request-boundary split
      * decode                                    -> request-count split

    Returns ``(meets_min_tokens, can_split, ub0_tokens, ub1_tokens)``:

      * ``can_split`` — this rank is *structurally* able to split into 2
        ubatches. AND-reduced across DP: if ANY rank can't split, TBO must
        stay off, else that rank runs 1 ubatch while peers run 2 → per-ubatch
        collective size mismatch → RCCL hang.
      * ``meets_min_tokens`` — this rank's prefill reached the min-token bar
        (ATOM_TBO_PREFILL_MIN_TOKENS, e.g. 8k), i.e. big enough to be worth TBO.
        OR-reduced: one rank clearing the bar turns TBO on for everyone, so
        under-filled-but-splittable ranks are force-split to stay aligned.

    Net: ``collective_active = OR(meets_min_tokens) AND AND(can_split)`` (plus
    the uniform-mode guard in :func:`sync_dp_for_tbo`). The ub0/ub1 counts are
    MAX-reduced so every rank picks the same per-ubatch CUDAGraph buffer size.
    """
    if not config.enable_tbo:
        return False, False, 0, 0

    if is_prefill:
        from atom.utils import envs

        num_pref_reqs = batch.total_seqs_num_prefill
        min_pref = envs.ATOM_TBO_PREFILL_MIN_TOKENS
        if envs.ATOM_TBO_PREFILL_TOKEN_SPLIT:
            return _precompute_prefill_token_split(
                num_scheduled_tokens, num_pref_reqs, min_pref
            )
        return _precompute_prefill_req_split(num_scheduled_tokens, num_pref_reqs, min_pref)

    # Decode path
    if not config.enable_tbo_decode or batch.is_dummy_run:
        return False, False, 0, 0
    return _precompute_decode(batch)


@dataclass
class DPSyncResult:
    """Output of :func:`sync_dp_for_tbo`."""

    # [dp_size] int32 CPU tensor — each rank's input token count.
    num_tokens_across_dp: torch.Tensor
    # True iff ANY rank has at least one prefill seq this step.
    any_rank_has_prefill: bool
    # True iff TBO on AND OR(meets_min_tokens) AND AND(can_split) AND uniform.
    # (One rank clearing the min-token bar turns TBO on for all; under-filled
    # but splittable ranks are force-split to stay collective-aligned. Any rank
    # that structurally can't split, or a mixed prefill/decode step, vetoes it.)
    tbo_collective_active: bool
    # (ub0_max, ub1_max) across DP — only set when tbo_collective_active.
    ub_max_tokens_across_dp: Optional[tuple[int, int]]


def sync_dp_for_tbo(
    *,
    dp_group,
    dp_size: int,
    num_input_tokens: int,
    is_prefill: bool,
    tbo_on: bool,
    local_meets_min_tokens: bool = False,
    local_can_split: bool = False,
    local_ub_tokens: tuple[int, int] = (0, 0),
) -> DPSyncResult:
    """Single packed DP all_gather over the per-rank scalars needed to
    decide DP padding, the prefill fan-out, and the cross-DP TBO gate.

    Pre-Plan-B this required up to 3 separate all_reduces per step
    (``get_dp_padding`` / ``sync_dp_for_tbo`` / a third inside
    ``UBatchWrapper``). Now one all_gather of ``n_fields`` int32 values
    per rank suffices. When TBO is off only the first 2 fields are
    exchanged (saves 60 % payload + skips :func:`local_tbo_precompute`
    at the call site).

    Layout (``sync`` is ``[n_fields, dp_size]``):

      row 0 : num_input_tokens         -> num_tokens_across_dp
      row 1 : is_prefill (0/1)         -> any_rank_has_prefill (OR)
      row 2 : meets_min_tokens (0/1)   -> OR  -> any rank reached the min-token bar [TBO only]
      row 3 : can_split (0/1)          -> AND -> every rank can split              [TBO only]
      row 4 : ub0_tokens               -> ub_max_tokens_across_dp[0]              [TBO only]
      row 5 : ub1_tokens               -> ub_max_tokens_across_dp[1]              [TBO only]

    Gate: ``active = OR(meets_min_tokens) AND AND(can_split) AND uniform``.
    """
    n_fields = 6 if tbo_on else 2
    local = torch.zeros(n_fields, dtype=torch.int32, device="cpu")
    local[0] = num_input_tokens
    local[1] = 1 if is_prefill else 0
    if tbo_on:
        local[2] = 1 if local_meets_min_tokens else 0
        local[3] = 1 if local_can_split else 0
        local[4] = local_ub_tokens[0]
        local[5] = local_ub_tokens[1]

    gathered = [
        torch.empty(n_fields, dtype=torch.int32, device="cpu") for _ in range(dp_size)
    ]
    torch.distributed.all_gather(gathered, local, group=dp_group)
    sync = torch.stack(gathered, dim=1)  # [n_fields, dp_size]

    num_tokens_across_dp = sync[0]
    any_rank_has_prefill = bool(sync[1].any())
    tbo_collective_active = False
    ub_max_tokens_across_dp: Optional[tuple[int, int]] = None
    if tbo_on:
        # OR(meets_min_tokens): one rank reaching the min-token bar turns TBO on
        # for all. AND(can_split): but EVERY rank must be structurally splittable, else
        # that rank would run 1 ubatch while peers run 2 → per-ubatch collective
        # size mismatch → RCCL hang. Under-filled-but-splittable ranks are then
        # force-split (see maybe_create_ubatch_slices force=True) to stay aligned.
        tbo_collective_active = bool(sync[2].any()) and bool(sync[3].all())
        # Mixed-mode guard: ALWAYS require a uniform batch mode (all prefill or
        # all decode) across DP. A prefill rank running 2 ubatches alongside a
        # decode rank running 2 ubatches still issues different collectives per
        # ubatch → hang. (Previously gated behind require_uniform_mode; with the
        # OR-reduce this must be unconditional.)
        if tbo_collective_active:
            prefill_rank_count = int(sync[1].sum())
            uniform_mode = prefill_rank_count == 0 or prefill_rank_count == dp_size
            tbo_collective_active = uniform_mode
        if tbo_collective_active:
            ub_max_tokens_across_dp = (
                int(sync[4].max()),
                int(sync[5].max()),
            )

    return DPSyncResult(
        num_tokens_across_dp=num_tokens_across_dp,
        any_rank_has_prefill=any_rank_has_prefill,
        tbo_collective_active=tbo_collective_active,
        ub_max_tokens_across_dp=ub_max_tokens_across_dp,
    )


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
        # Set True when this ubatch's model.forward returns (or raises).
        # Partner thread checks this in `_cpu_yield` so it doesn't sleep
        # forever if we exited (cleanly or via exception) ahead of it.
        self.done: bool = False
        # Filled by `make_tbo_contexts` — points to the OTHER ubatch's ctx.
        self.partner: Optional["TBOContext"] = None

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
        # Mark this ubatch done BEFORE the final signal so that if the partner
        # is racing into its next `_cpu_yield` between our signal and its wait,
        # it observes `partner.done == True` and skips the wait instead of
        # sleeping forever. Without this, any asymmetry (e.g. partner exits
        # mid-forward via exception) leaves the survivor wedged on the next
        # yield: the dead partner only signals exactly once from __exit__,
        # but the survivor still has ≥1 yield left to do.
        self.done = True
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
        """Wake the next thread and sleep until woken.

        If the partner ubatch has already finished (cleanly or via exception
        — see `__exit__` above), there is nobody to wake us, so we must not
        block. Returning immediately here lets the survivor finish its
        remaining yields and exit normally, so the main thread's `t.join()`
        can complete and any captured `errors[idx]` can be raised.
        """
        self.cpu_signal_event.set()
        if self.partner is not None and self.partner.done:
            self.cpu_wait_event.clear()
            self._restore_context()
            return
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

    # Link each ctx to its neighbour in the ping-pong ring so that `_cpu_yield`
    # can short-circuit when the partner has exited. For N=2 (the only case
    # we run today) `partner` is just the other ctx. For N>2 we point at the
    # NEXT ctx (the one whose `cpu_wait_event` is our `cpu_signal_event`'s
    # target) — that's the only one whose `done` flag can leave us wedged.
    for i, ctx in enumerate(ctxs):
        ctx.partner = ctxs[(i + 1) % num_micro_batches]

    return ctxs

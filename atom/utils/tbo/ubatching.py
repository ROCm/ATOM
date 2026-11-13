# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import threading
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from atom.config import get_current_atom_config
from atom.utils import envs
from atom.utils.tbo.ubatch_splitting import NUM_TBO_UBATCHES, prefill_token_halves


def tbo_overlap_enabled() -> bool:
    return False


def tbo_enabled() -> bool:
    config = get_current_atom_config()
    if config is None:
        return False
    return getattr(config, "enable_tbo", False)


# =====================================================================
# DP-side TBO helpers: a rank's vote and the group's reduction of it.
# =====================================================================


def prefill_tbo_vote(prefill_tokens: int) -> tuple[bool, bool, int, int]:
    """This rank's TBO vote on a prefill of ``prefill_tokens``, cut at the
    token midpoint (`prefill_token_halves`, the cut `cut_prefill_ubatches`
    runs); a decode does not vote (`decide` splits the group's `running_bs`).

    Returns ``(meets_min_tokens, can_split, ub0_tokens, ub1_tokens)``:

      * ``can_split`` — this rank is *structurally* able to split into 2
        ubatches. AND-reduced across DP: if ANY rank can't split, TBO must
        stay off, else that rank runs 1 ubatch while peers run 2 → per-ubatch
        collective size mismatch → RCCL hang.
      * ``meets_min_tokens`` — this rank's prefill reached the min-token bar
        (ATOM_TBO_PREFILL_MIN_TOKENS, e.g. 8k), i.e. big enough to be worth TBO.
        OR-reduced: one rank clearing the bar turns TBO on for everyone, so
        under-filled-but-splittable ranks are force-split to stay aligned.

    Net: ``collective_active = OR(meets_min_tokens) AND AND(can_split)``
    (:func:`sync_dp_metadata`). The ub0/ub1 counts are gathered: each
    micro-batch's MoE gathers every rank's own.
    """
    halves = prefill_token_halves(prefill_tokens)
    if halves is None:
        return False, False, 0, 0
    min_tokens = envs.ATOM_TBO_PREFILL_MIN_TOKENS
    meets_min_tokens = not (min_tokens > 0 and prefill_tokens < min_tokens)
    return meets_min_tokens, True, *halves


@dataclass
class DPSyncResult:
    """Output of :func:`sync_dp_metadata`."""

    # [dp_size] int32 — each rank's scheduled tokens, from which
    # `ForwardMode.decide` settles each rank's running ones.
    scheduled_tokens_per_dp_rank: np.ndarray
    # DP-MAX of the scheduled SEQUENCE count -- the companion to
    # `scheduled_tokens_per_dp_rank` in the other unit. Reduced on EVERY step,
    # not just uniform decode: a captured graph holds its collective at one
    # batch, so a per-rank count is precisely what splits the group into two
    # widths.
    max_bs_across_dp: int
    # Per rank, in DP order: whether its step is a prefill.
    prefill_across_dp: tuple[bool, ...]
    # True iff ANY rank's backend cannot run this step from a graph: a graph
    # replays its collectives, so the group replays or runs eagerly as one.
    any_rank_needs_eager: bool
    # A prefill's split: True iff TBO on AND OR(meets_min_tokens) AND
    # AND(can_split) AND every rank prefills. (One rank clearing the min-token
    # bar turns TBO on for all; under-filled but splittable ranks are
    # force-split to stay collective-aligned. Any rank that structurally can't
    # split vetoes it.) A decode casts no vote: `ForwardMode.decide` splits it.
    tbo_collective_active: bool
    # Per ubatch, every rank's tokens -- only set when tbo_collective_active.
    # Each consumer reduces the row it needs: a gather's sizes are the row, an
    # all2all's receive width its SUM.
    ub_tokens_across_dp: tuple[tuple[int, ...], ...] | None
    # DP-MAX of the per-seq decode length — only set when ``max_seqlen_q`` was
    # passed in (a block drafter under DP). Folded into this packed all_gather so
    # the graph-shape sync no longer needs its own separate all_reduce (halves
    # the per-step DP round-trips).
    max_seqlen_q_across_dp: int | None = None
    # DP-MAX of the requests whose prompt a rank's prefill finishes this step.
    max_final_requests_across_dp: int = 0

    @property
    def any_rank_has_prefill(self) -> bool:
        return any(self.prefill_across_dp)


def sync_dp_metadata(
    *,
    dp_group,
    dp_size: int,
    scheduled_tokens: int,
    scheduled_bs: int,
    is_prefill: bool,
    tbo_on: bool,
    running_tokens_per_dp_rank: np.ndarray,
    needs_eager: bool = False,
    final_requests: int = 0,
    local_meets_min_tokens: bool = False,
    local_can_split: bool = False,
    local_ub_tokens: tuple[int, int] = (0, 0),
    max_seqlen_q: int | None = None,
) -> DPSyncResult:
    """Single packed DP all_gather over all per-rank scalars a decode/prefill
    step needs synchronized: DP token padding, the prefill fan-out, whether the
    group runs eagerly, the cross-DP TBO gate, and (when active) the DSpark
    graph-shape MAX.

    A block drafter folds its per-seq length sync in here too (one extra field),
    so the step issues one collective instead of two.

    One all_gather of ``n_fields`` int32 values per rank. When TBO is off
    only the first 5 fields are exchanged.

    Layout (``sync`` is ``[n_fields, dp_size]``):

      row 0 : scheduled_tokens         -> scheduled_tokens_per_dp_rank
      row 1 : scheduled_bs             -> max_bs_across_dp (MAX)
      row 2 : is_prefill (0/1)         -> prefill_across_dp, any_rank_has_prefill (OR)
      row 3 : needs_eager (0/1)        -> any_rank_needs_eager (OR)
      row 4 : final_requests           -> max_final_requests_across_dp (MAX)
      row 5 : meets_min_tokens (0/1)   -> OR  -> any rank reached the min-token bar [TBO only]
      row 6 : can_split (0/1)          -> AND -> every rank can split              [TBO only]
      row 7 : ub0_tokens               -> ub_tokens_across_dp[0]  [TBO prefill only]
      row 8 : ub1_tokens               -> ub_tokens_across_dp[1]  [TBO prefill only]
      row k+0 : max_seqlen_q           -> max_seqlen_q_across_dp (MAX)  [DSpark only]

    ``final_requests``: the requests whose prompt this rank's prefill
    finishes, for a model whose prefill runs some layers on those requests'
    rows alone and sizes their collectives from the group's
    (`deepseek_v41.bounded_replay`).

    Rows 0 and 1 are the same batch in ATOM's two units; both ride this one
    collective because agreeing on only one of them still leaves two shapes.

    Gate: ``active = OR(meets_min_tokens) AND AND(can_split)``, a prefill's
    only: a decoding rank votes neither.

    ``max_seqlen_q`` folds a block drafter's per-step graph-shape sync into this
    same packed all_gather. It is global-config gated (DSpark on + DP), so every
    rank appends the same field and the payload stays symmetric.

    Neither the batch nor the token total needs a row of its own here: rows 1 and
    0 already carry them, and a step that reaches the reduction with every rank
    decoding has `scheduled_tokens` == its decode total on every rank.

    A rank with no request reports no phase of its own: row 2 reads as the
    group's for it (prefill when any rank prefills).

    ``running_tokens_per_dp_rank`` (the caller's [dp_size] buffer) is filled
    with every rank's running tokens on a ragged step: its scheduled ones, or
    for a rank with none one running token (two under TBO), at most the
    group's smallest nonzero count.
    `ForwardMode.decide` raises them where the step is not ragged.
    """
    tbo_fields = 9 if tbo_on else 5
    dspark_on = max_seqlen_q is not None
    n_fields = tbo_fields + (1 if dspark_on else 0)
    local = torch.zeros(n_fields, dtype=torch.int32, device="cpu")
    local[0] = scheduled_tokens
    local[1] = scheduled_bs
    local[2] = 1 if is_prefill else 0
    local[3] = 1 if needs_eager else 0
    local[4] = final_requests
    if tbo_on:
        local[5] = 1 if local_meets_min_tokens else 0
        local[6] = 1 if local_can_split else 0
        local[7] = local_ub_tokens[0]
        local[8] = local_ub_tokens[1]
    if dspark_on:
        local[tbo_fields + 0] = max_seqlen_q

    # A group of one is its own row, with nothing to exchange.
    gathered = [local]
    if dp_size > 1:
        gathered = [
            torch.empty(n_fields, dtype=torch.int32, device="cpu")
            for _ in range(dp_size)
        ]
        torch.distributed.all_gather(gathered, local, group=dp_group)
    sync = torch.stack(gathered, dim=1).numpy()  # [n_fields, dp_size]

    scheduled_tokens_per_dp_rank = sync[0]
    np.copyto(running_tokens_per_dp_rank, scheduled_tokens_per_dp_rank)
    idle = scheduled_tokens_per_dp_rank == 0
    if idle.any() and not idle.all():
        # A rank with no request runs the group's phase and, on a ragged step,
        # the fewest running tokens a TBO prefill can split (1 + 1), never more
        # than the group's smallest nonzero count. It neither turns TBO on nor
        # vetoes it.
        running_tokens = min(
            scheduled_tokens_per_dp_rank[~idle].min(), 2 if tbo_on else 1
        )
        running_tokens_per_dp_rank[idle] = running_tokens
        prefill = sync[2].any()
        sync[2][idle] = prefill
        if tbo_on:
            sync[5][idle] = 0
            if prefill:
                _, can_split, ub0, ub1 = prefill_tbo_vote(running_tokens)
                sync[6][idle], sync[7][idle], sync[8][idle] = can_split, ub0, ub1
    max_bs_across_dp = int(sync[1].max())
    any_rank_needs_eager = bool(sync[3].any())
    tbo_collective_active = False
    ub_tokens_across_dp: tuple[tuple[int, ...], ...] | None = None
    if tbo_on:
        # OR(meets_min_tokens): one rank reaching the min-token bar turns TBO on
        # for all. AND(can_split): but EVERY rank must be structurally splittable, else
        # that rank would run 1 ubatch while peers run 2 → per-ubatch collective
        # size mismatch → RCCL hang. Under-filled-but-splittable ranks then
        # split too (`cut_prefill_ubatches` has no min-token bar of its own)
        # to stay aligned. A decoding rank votes neither, so a step
        # some rank decodes never splits here.
        tbo_collective_active = bool(sync[5].any()) and bool(sync[6].all())
        if tbo_collective_active:
            ub_tokens_across_dp = (tuple(sync[7].tolist()), tuple(sync[8].tolist()))

    max_seqlen_q_across_dp: int | None = None
    if dspark_on:
        max_seqlen_q_across_dp = int(sync[tbo_fields + 0].max())

    return DPSyncResult(
        scheduled_tokens_per_dp_rank=scheduled_tokens_per_dp_rank,
        max_bs_across_dp=max_bs_across_dp,
        prefill_across_dp=tuple(bool(p) for p in sync[2].tolist()),
        any_rank_needs_eager=any_rank_needs_eager,
        tbo_collective_active=tbo_collective_active,
        ub_tokens_across_dp=ub_tokens_across_dp,
        max_seqlen_q_across_dp=max_seqlen_q_across_dp,
        max_final_requests_across_dp=int(sync[4].max()),
    )


_THREAD_ID_TO_CONTEXT: dict[int, int] = {}
_CURRENT_CONTEXTS: list["TBOContext | None"] = []
_NUM_UBATCHES: int = NUM_TBO_UBATCHES

# Per-ubatch independent pynccl TP communicators for the pure-TP all_reduce
# overlap (see `tbo_all_reduce`).
_TBO_TP_UBATCH_COMMS: "list | None" = None


def tbo_get_ubatch_tp_comm(ubatch_id: int):
    """Return this ubatch's dedicated pynccl TP communicator, building the pair
    lazily on first use. Returns None if TP world size == 1 (no AR needed)."""
    global _TBO_TP_UBATCH_COMMS
    if _TBO_TP_UBATCH_COMMS is not None:
        return _TBO_TP_UBATCH_COMMS[ubatch_id]

    from aiter.dist.device_communicators.communicator_pynccl import (
        PyNcclCommunicator,
    )
    from aiter.dist.parallel_state import get_tp_group

    tp = get_tp_group()
    if tp.world_size == 1:
        _TBO_TP_UBATCH_COMMS = [None] * _NUM_UBATCHES
        return None

    # One independent communicator per ubatch. Construction runs a warmup
    # all_reduce (a collective), so every rank must build the same number in the
    # same order — guaranteed here because all ranks run TBO in lockstep and hit
    # this on the same forward. Bound to the TP cpu_group (non-NCCL backend),
    # exactly like the shared pynccl_comm.
    comms = []
    for _ in range(_NUM_UBATCHES):
        comms.append(PyNcclCommunicator(group=tp.cpu_group, device=tp.device))
    _TBO_TP_UBATCH_COMMS = comms
    return _TBO_TP_UBATCH_COMMS[ubatch_id]


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
        self.recv_hook: Callable | None = None
        # Set True when this ubatch's model.forward returns (or raises).
        # Partner thread checks this in `_cpu_yield` so it doesn't sleep
        # forever if we exited (cleanly or via exception) ahead of it.
        self.done: bool = False
        # Filled by `make_tbo_contexts` — points to the OTHER ubatch's ctx.
        self.partner: TBOContext | None = None

    # -- context manager protocol ----------------------------------------

    def __enter__(self):
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
        # torch.Event record/wait (`switch_to_compute_sync` and the yields).
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
    global _NUM_UBATCHES
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

# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1 DSpark on the ATOM plugin under SGLang 0.5.22's overlap
schedule: resolve a decode round's host ``seq_lens`` after its draft block is
launched instead of before.

Upstream (``FutureMap.resolve_seq_lens_cpu``, called at the top of
``Scheduler.run_batch``) blocks the scheduler thread on the previous round's
publish event (on HIP ``Event.synchronize()``) and then copies the published
lengths to the host, before any of this round's work is enqueued. The GPU
therefore idles from the end of round t's verify until the host has built and
launched round t+1's draft block.

Nothing before the verify needs exact host lengths: SGLang's own
``needs_cpu_seq_lens=False`` contract already runs the draft from device
tensors and sizes host planning off ``DFlashDraftInputV2.nxt_kv_lens_cpu``
(the page-aligned reservation ``prepare_for_decode`` made, an upper bound of
every length the round can reach) whenever ``batch.seq_lens_cpu`` is None
(``alloc_verify_window``, ``DraftBlockProposer._run_forward``). Only this
plugin's verify staging needs the exact committed prefix
(``spec_info.live_seq_lens_cpu``), and it reads it in
``TargetVerifyExecutor.run_non_compact``.

So, for a V4.1 DSpark decode round only, the patched resolve

- orders the device gather ``new_seq_lens_buf[future_indices]`` after the
  publish with a stream wait instead of a host wait (upstream's CUDA path),
- starts the D2H of the gathered lengths on a side stream into pinned memory
  without waiting for it, and leaves ``seq_lens_cpu`` / ``seq_lens_sum`` None
  (upstream's ``needs_cpu_seq_lens=False`` state),

and the verify takes that copy later (it finished when round t's accept did,
well before this round's draft does): the exact values upstream would have
set, the same tensor upstream copies (``batch.seq_lens`` after the gather).
Only the time the host waits for them moves, from before the draft launch to
the verify; when, is below.

The deferred copy is a handle for this ``run_batch`` only: created by the
resolve, consumed (and cleared) by the verify of the same batch, refused if a
second resolve finds one unconsumed. Every other batch (prefill, mixed, idle,
another speculative algorithm) takes upstream's resolve unchanged -- a radix
prefix hit's EXTEND included: the radix seam
(`deepseek_v41_radix.run_v41_prefix_replay`) and its fused replay run on
EXTEND batches only, decide on the mode before reading any length, and so
never see a deferred round. ``_forward_isolation`` restores the scheduler's
``seq_lens_cpu`` to the snapshot (None) after the forward, as for any
``needs_cpu_seq_lens=False`` backend; ``prepare_for_decode`` seeds it again
before the next round.

The verify's host lengths are late: the patched ``run_non_compact`` does not
wait. SGLang runs the verify on its ``needs_cpu_seq_lens=False`` contract
(``seq_lens_cpu`` the reservation ``nxt_kv_lens_cpu``), and the plugin's
verify staging takes the exact committed prefix as its first step
(`take_late_seq_lens`), once, for the batch it was resolved for; one never
taken is materialized (``seq_lens_cpu`` / ``seq_lens_sum`` set) after the
forward. ``run_compact`` is not wrapped: SGLang calls it only for the compact
ragged verify, and the plugin refuses any ``SGLANG_RAGGED_VERIFY_MODE`` but
``static`` at launch (`_dspark_launch_refusals`).

Switch: ``ATOM_SGLANG_V41_DSPARK_DEFERRED_SEQ_LENS=0`` (the plugin registry,
`deepseek_v41_envs`), read at install, keeps upstream's resolve: the path every
other batch takes anyway, so no code stays alive for the switch alone. Shape
guard: the function shapes checked below; where they differ, and under
SGLang's CI consume-once asserts or forward-pass metrics (which read the host
lengths at the forward), upstream's resolve stays and one line logs why. The
plugin runs on SGLang 0.5.22 only (`sglang_version_guard` refuses any other at
launch, before this installs).
"""

from __future__ import annotations

import logging
import weakref
from dataclasses import dataclass
from functools import wraps
from typing import Any

import torch

from atom.plugin.sglang import deepseek_v41_envs
from atom.plugin.sglang.deepseek_v41_bridge import _device_module
from atom.plugin.sglang.patches.sglang_version_guard import (
    missing_source_needles,
    sglang_version,
)

logger = logging.getLogger("atom")


@dataclass
class _Deferred:
    """One decode round's host lengths in flight: the batch they belong to,
    the pinned rows the D2H fills and the event that marks it done."""

    batch: Any  # weakref.ref to the ScheduleBatch
    bs: int
    host: torch.Tensor
    done: Any
    # The batch's STATE slots, which its verify staging must name
    # (`take_late_seq_lens`).
    slots: tuple = ()
    # Left for the verify staging to take (`run_non_compact`, late mode).
    late: bool = False


class _State:
    def __init__(self) -> None:
        self.pending: _Deferred | None = None
        self.stream = None
        self.event = None
        self.pinned: torch.Tensor | None = None


_STATE = _State()


def _upstream_shape_error(future_map_cls, executor_cls) -> str | None:
    """Why the imported SGLang does not look like the one this patch was
    written against, or None."""
    resolve = getattr(future_map_cls, "resolve_seq_lens_cpu", None)
    if resolve is None:
        return "FutureMap.resolve_seq_lens_cpu is missing"
    missing = missing_source_needles(
        resolve, ("publish_ready", "new_seq_lens_buf[fi]", "needs_cpu_seq_lens")
    )
    if missing is None:
        return "FutureMap.resolve_seq_lens_cpu has no source"
    if missing:
        return f"FutureMap.resolve_seq_lens_cpu no longer reads {missing}"
    if not callable(getattr(executor_cls, "run_non_compact", None)):
        return "TargetVerifyExecutor.run_non_compact is missing"
    return None


def _is_v41_dspark_decode(future_map, batch) -> bool:
    """A batch this patch defers: a plain DSpark decode round (the patch is
    installed by the V4.1 DSpark target only, and V4.1 refuses DP attention
    and PP at launch), with everything the deferred path reads."""
    if not future_map.needs_cpu_seq_lens or future_map.publish_ready is None:
        return False
    if not batch.forward_mode.is_decode() or batch.is_extend_in_batch:
        return False
    algorithm = batch.spec_algorithm
    if algorithm is None or not algorithm.is_dspark():
        return False
    spec_info = batch.spec_info
    future_indices = spec_info.future_indices
    # The draft's host bound and the commit's host slots.
    return (
        future_indices is not None
        and int(future_indices.shape[0]) > 0
        and spec_info.nxt_kv_lens_cpu is not None
        and batch.req_pool_indices_cpu is not None
    )


def _pinned(n: int) -> torch.Tensor:
    pinned = _STATE.pinned
    if pinned is None or pinned.numel() < n:
        size = max(n, 64)
        pinned = torch.empty((size,), dtype=torch.int64, pin_memory=True)
        _STATE.pinned = pinned
    return pinned


def resolve_seq_lens_deferred(future_map, batch) -> None:
    """The device half of ``resolve_seq_lens_cpu`` now, the host half later
    (`materialize_seq_lens`)."""
    if _STATE.pending is not None:
        raise RuntimeError(
            "DeepSeek-V4.1 DSpark: a decode round's deferred host seq_lens were "
            "never consumed by its verify"
        )
    device = future_map.new_seq_lens_buf.device
    module = _device_module(device)
    current = module.current_stream()
    # Upstream's CUDA path: a stream wait, not a host wait.
    current.wait_event(future_map.publish_ready)
    future_map._publish_fresh = False
    seq_lens = future_map.new_seq_lens_buf[batch.spec_info.future_indices]
    batch.seq_lens = seq_lens
    if _STATE.stream is None:
        _STATE.stream = module.Stream(device=device)
        _STATE.event = module.Event()
    bs = int(seq_lens.shape[0])
    host = _pinned(bs)[:bs]
    side = _STATE.stream
    side.wait_stream(current)
    with module.stream(side):
        host.copy_(seq_lens, non_blocking=True)
        _STATE.event.record(side)
    if seq_lens.device.type != "cpu":
        # The side stream reads it; the allocator must not hand it out first.
        seq_lens.record_stream(side)
    # Upstream's needs_cpu_seq_lens=False state: the draft sizes its host
    # planning off spec_info.nxt_kv_lens_cpu.
    batch.seq_lens_cpu = None
    batch.seq_lens_sum = None
    slots = tuple(int(slot) for slot in batch.req_pool_indices_cpu[:bs])
    _STATE.pending = _Deferred(weakref.ref(batch), bs, host, _STATE.event, slots=slots)


def materialize_seq_lens(batch) -> bool:
    """Set ``batch.seq_lens_cpu`` / ``seq_lens_sum`` from the deferred copy
    of this batch, if one is pending. Returns whether one was."""
    pending = _STATE.pending
    if pending is None:
        return False
    if pending.batch() is not batch:
        raise RuntimeError(
            "DeepSeek-V4.1 DSpark: the verify ran on a batch other than the one "
            "whose host seq_lens were deferred"
        )
    _STATE.pending = None
    pending.done.synchronize()
    seq_lens_cpu = pending.host[: pending.bs].clone()
    batch.seq_lens_cpu = seq_lens_cpu
    batch.seq_lens_sum = int(seq_lens_cpu.sum())
    return True


def take_late_seq_lens(bs: int, slots=None):
    """For a late-mode round's verify staging: ``(lens, slots)``, the exact
    committed prefix of its ``bs`` requests and the slots it was resolved
    for, or None when none is pending. Another batch's copy is refused."""
    pending = _STATE.pending
    if pending is None or not pending.late:
        return None
    named = None if slots is None else tuple(int(slot) for slot in slots)
    if pending.bs != int(bs) or (named is not None and pending.slots != named):
        raise RuntimeError(
            "DeepSeek-V4.1 DSpark: the verify staged "
            f"{int(bs)} requests in slots {named} but the deferred host "
            f"seq_lens are those of {list(pending.slots)}"
        )
    _STATE.pending = None
    pending.done.synchronize()
    return pending.host[: pending.bs].clone(), pending.slots


def deferred_state_summary() -> str:
    """The pending deferred copy, for an error message."""
    pending = _STATE.pending
    if pending is None:
        return "no copy pending"
    alive = pending.batch() is not None
    return (
        f"a copy pending for {pending.bs} requests in slots "
        f"{list(pending.slots)} late={pending.late} batch_alive={alive}"
    )


def install_dspark_deferred_seq_lens_patch() -> bool:
    """Install the deferred resolve (idempotent). Returns whether it is
    active; False (upstream kept) where the shape guard fails or with the
    switch off."""
    if not deepseek_v41_envs.ATOM_SGLANG_V41_DSPARK_DEFERRED_SEQ_LENS:
        logger.info(
            "ATOM DeepSeek-V4.1 DSpark: host seq_lens resolved before the draft "
            "(upstream; ATOM_SGLANG_V41_DSPARK_DEFERRED_SEQ_LENS=0)"
        )
        return False
    from sglang.srt.managers import overlap_utils
    from sglang.srt.runtime_context import get_observability
    from sglang.srt.speculative.dspark_components import dspark_verify

    future_map_cls = overlap_utils.FutureMap
    executor_cls = dspark_verify.TargetVerifyExecutor
    if getattr(future_map_cls, "_atom_v41_deferred_seq_lens", False):
        return True
    reason = _upstream_shape_error(future_map_cls, executor_cls)
    if reason is None and overlap_utils._DEBUG_ASSERT:
        # SGLang CI's consume-once poison needs the copy to have landed.
        reason = "SGLang's CI asserts are on"
    if reason is None and get_observability().enable_forward_pass_metrics:
        # They read the batch's host lengths at the forward, before the
        # verify would have them.
        reason = "forward-pass metrics are on"
    if reason is not None:
        logger.warning(
            "ATOM DeepSeek-V4.1 DSpark: deferred host seq_lens not installed "
            "(%s); the draft waits for the previous round's publish",
            reason,
        )
        return False

    original_resolve = future_map_cls.resolve_seq_lens_cpu

    @wraps(original_resolve)
    def resolve_seq_lens_cpu(self, batch):
        if _is_v41_dspark_decode(self, batch):
            return resolve_seq_lens_deferred(self, batch)
        return original_resolve(self, batch)

    def _late(original):
        # `run_non_compact`: the verify staging takes the copy
        # (`take_late_seq_lens`); one it never took is settled after.
        @wraps(original)
        def run(self, *, batch, **kwargs):
            pending = _STATE.pending
            if pending is None or pending.batch() is not batch:
                materialize_seq_lens(batch)
                return original(self, batch=batch, **kwargs)
            pending.late = True
            try:
                return original(self, batch=batch, **kwargs)
            finally:
                if _STATE.pending is pending:
                    # No staging asked for it: settle it as upstream would.
                    pending.late = False
                    materialize_seq_lens(batch)

        return run

    future_map_cls.resolve_seq_lens_cpu = resolve_seq_lens_cpu
    executor_cls.run_non_compact = _late(executor_cls.run_non_compact)
    future_map_cls._atom_v41_deferred_seq_lens = True
    logger.info(
        "ATOM DeepSeek-V4.1 DSpark: decode rounds resolve host seq_lens after "
        "the draft launch (FutureMap.resolve_seq_lens_cpu deferred to the "
        "verify staging; sglang %s)",
        sglang_version(),
    )
    return True

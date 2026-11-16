# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1 DSpark on the ATOM plugin under SGLang 0.5.22's overlap
schedule: resolve a decode round's host ``seq_lens`` after its draft block is
launched instead of before.

Upstream (``FutureMap.resolve_seq_lens_cpu``, called at the top of
``Scheduler.run_batch``) blocks the scheduler thread on the previous round's
publish event (on HIP ``Event.synchronize()``) and then copies the published
lengths to the host, before any of this round's work is enqueued. The GPU
therefore idles from the end of round t's verify until the host has built and
launched round t+1's draft block (~1.4 ms a round at c1 on MI308X).

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

and the patched ``run_non_compact`` / ``run_compact`` wait for that copy (it
finished when round t's accept did, well before this round's draft does) and
set ``seq_lens_cpu`` / ``seq_lens_sum`` to the exact values upstream would
have set, before anything reads them. The values are the same tensor upstream
copies (``batch.seq_lens`` after the gather); only the time the host waits for
them moves, from before the draft launch to after it.

The deferred copy is a handle for this ``run_batch`` only: created by the
resolve, consumed (and cleared) by the verify of the same batch, refused if a
second resolve finds one unconsumed. Every other batch (prefill, mixed, idle,
another speculative algorithm or model, SGLang's CI debug asserts, DP
attention, forward-pass metrics) takes upstream's resolve unchanged -- a
radix prefix hit's EXTEND included: the radix seam
(`deepseek_v41_radix.run_v41_prefix_replay`) and its fused or side replay run
on EXTEND batches only, decide on the mode before reading any length, and so
never see a deferred round. ``_forward_isolation`` restores
the scheduler's ``seq_lens_cpu`` to the snapshot (None) after the forward, as
for any ``needs_cpu_seq_lens=False`` backend; ``prepare_for_decode`` seeds it
again before the next round.

Late host lengths (default; ``ATOM_SGLANG_V41_DSPARK_LATE_SEQ_LENS=0`` turns it
off): the patched ``run_non_compact`` does not wait at all. SGLang then runs
the verify on its ``needs_cpu_seq_lens=False`` contract -- ``live_seq_lens_cpu``
None, ``seq_lens_cpu`` the reservation ``nxt_kv_lens_cpu`` (an upper bound),
``candidate_max_seq_len_upper_bound`` set for a DSV4 target -- through
``prepare_for_verify``, ``ForwardBatch.init_new``, the graph runner's checks
and its input fill, and this plugin's verify staging takes the exact
committed prefix itself, as its first step (`take_late_seq_lens`): the host
waits there, after all of SGLang's own preparation, instead of before it.
The copy is the same one and is consumed exactly once, by the verify of the
batch it was resolved for (same request count and STATE slots); a verify
that never stages it (none known) gets it materialized after the forward.
``run_compact`` keeps the wait at its entry.

Kill switch: ``ATOM_SGLANG_V41_DSPARK_DEFERRED_SEQ_LENS=0``. Version guard:
SGLang 0.5.22 (`SUPPORTED_SGLANG_VERSIONS`) with the function shapes checked
below; any other version keeps upstream's resolve and logs why.
"""

from __future__ import annotations

import inspect
import logging
import os
import weakref
from dataclasses import dataclass
from functools import wraps
from typing import Any

import torch

logger = logging.getLogger("atom")

DEFERRED_SEQ_LENS_ENV = "ATOM_SGLANG_V41_DSPARK_DEFERRED_SEQ_LENS"
LATE_SEQ_LENS_ENV = "ATOM_SGLANG_V41_DSPARK_LATE_SEQ_LENS"
SUPPORTED_SGLANG_VERSIONS = ("0.5.22",)


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
        self.deferred_rounds = 0
        # late-mode rounds whose verify staging took the copy, and those
        # whose forward finished without it (materialized after)
        self.late_taken = 0
        self.late_unconsumed = 0


_STATE = _State()


def deferred_seq_lens_enabled() -> bool:
    return os.environ.get(DEFERRED_SEQ_LENS_ENV, "1") != "0"


def late_seq_lens_enabled() -> bool:
    return os.environ.get(LATE_SEQ_LENS_ENV, "1") != "0"


def _sglang_version() -> str:
    try:
        import sglang

        return str(getattr(sglang, "__version__", "?"))
    except Exception:  # noqa: BLE001 - a version string
        return "?"


def _upstream_shape_error(future_map_cls, executor_cls) -> str | None:
    """Why the imported SGLang does not look like the one this patch was
    written against, or None."""
    version = _sglang_version()
    if version not in SUPPORTED_SGLANG_VERSIONS:
        return f"sglang {version} not in {SUPPORTED_SGLANG_VERSIONS}"
    resolve = getattr(future_map_cls, "resolve_seq_lens_cpu", None)
    if resolve is None:
        return "FutureMap.resolve_seq_lens_cpu is missing"
    try:
        source = inspect.getsource(resolve)
    except (OSError, TypeError):
        return "FutureMap.resolve_seq_lens_cpu has no source"
    for needle in ("publish_ready", "new_seq_lens_buf[fi]", "needs_cpu_seq_lens"):
        if needle not in source:
            return f"FutureMap.resolve_seq_lens_cpu no longer reads {needle!r}"
    for name in ("run_non_compact", "run_compact"):
        if not callable(getattr(executor_cls, name, None)):
            return f"TargetVerifyExecutor.{name} is missing"
    return None


def _device_module(device):
    return torch.get_device_module(torch.device(device))


def _is_v41_dspark_decode(future_map, batch) -> bool:
    """A batch this patch defers: a plain DSpark decode round of the V4.1
    target this process serves, with everything the deferred path reads."""
    from atom.plugin.sglang import deepseek_v41_bridge as bridge

    if bridge._V41_BUILDER is None or not deferred_seq_lens_enabled():
        return False
    if not getattr(future_map, "needs_cpu_seq_lens", False):
        return False
    if getattr(future_map, "publish_ready", None) is None:
        return False
    mode = batch.forward_mode
    if not mode.is_decode() or getattr(batch, "is_extend_in_batch", False):
        return False
    algorithm = getattr(batch, "spec_algorithm", None)
    if algorithm is None or not algorithm.is_dspark():
        return False
    spec_info = batch.spec_info
    future_indices = getattr(spec_info, "future_indices", None)
    if future_indices is None or int(future_indices.shape[0]) == 0:
        return False
    # The draft's host bound and the commit's host slots.
    if getattr(spec_info, "nxt_kv_lens_cpu", None) is None:
        return False
    if getattr(batch, "req_pool_indices_cpu", None) is None:
        return False
    try:
        from sglang.srt.managers import overlap_utils

        if overlap_utils._DEBUG_ASSERT:
            # CI's consume-once poison needs the copy to have landed.
            return False
        from sglang.srt.runtime_context import get_observability, get_parallel

        parallel = get_parallel()
        if parallel.attn_dp_enabled or parallel.pp_size != 1:
            return False
        if getattr(get_observability(), "enable_forward_pass_metrics", False):
            # Forward-pass metrics read the batch's host lengths at the
            # forward, before the verify would materialize them.
            return False
    except Exception:  # noqa: BLE001 - unknown runtime: keep upstream
        return False
    return True


def _pinned(n: int) -> torch.Tensor:
    pinned = _STATE.pinned
    if pinned is None or pinned.numel() < n:
        size = max(n, 64)
        pinned = torch.empty(
            (size,), dtype=torch.int64, pin_memory=torch.cuda.is_available()
        )
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
    _STATE.deferred_rounds += 1


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
    """For the verify staging of a late-mode round: ``(lens, slots)``, the
    exact committed prefix of its ``bs`` requests and the STATE slots the
    copy was resolved for (one event wait: the copy finished when the
    previous round's accept did), or None when no late copy is pending. A
    pending copy of another batch -- another request count, or other
    ``slots`` when the staging knows them -- is refused; a staging that
    learns its slots later (the eager verify reads them off the device)
    compares them with the returned ones."""
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
    _STATE.late_taken += 1
    return pending.host[: pending.bs].clone(), pending.slots


def deferred_state_summary() -> str:
    """The pending deferred copy, for an error message."""
    pending = _STATE.pending
    if pending is None:
        return (
            f"no copy pending (deferred_rounds={_STATE.deferred_rounds} "
            f"late_taken={_STATE.late_taken} "
            f"late_unconsumed={_STATE.late_unconsumed})"
        )
    alive = pending.batch() is not None
    return (
        f"a copy pending for {pending.bs} requests in slots "
        f"{list(pending.slots)} late={pending.late} batch_alive={alive}"
    )


def install_dspark_deferred_seq_lens_patch() -> bool:
    """Install the deferred resolve (idempotent). Returns whether it is
    active; False (upstream kept) on another SGLang or with the kill switch."""
    if not deferred_seq_lens_enabled():
        logger.info(
            "ATOM DeepSeek-V4.1 DSpark: host seq_lens resolved before the draft "
            "(upstream; %s=0)",
            DEFERRED_SEQ_LENS_ENV,
        )
        return False
    from sglang.srt.managers import overlap_utils
    from sglang.srt.speculative.dspark_components import dspark_verify

    future_map_cls = overlap_utils.FutureMap
    executor_cls = dspark_verify.TargetVerifyExecutor
    if getattr(future_map_cls, "_atom_v41_deferred_seq_lens", False):
        return True
    reason = _upstream_shape_error(future_map_cls, executor_cls)
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

    def _materializing(original, *, late_ok):
        @wraps(original)
        def run(self, *, batch, **kwargs):
            pending = _STATE.pending
            if not (
                late_ok
                and late_seq_lens_enabled()
                and pending is not None
                and pending.batch() is batch
            ):
                materialize_seq_lens(batch)
                return original(self, batch=batch, **kwargs)
            pending.late = True
            try:
                return original(self, batch=batch, **kwargs)
            finally:
                if _STATE.pending is pending:
                    # No staging asked for it: settle it as upstream would.
                    _STATE.late_unconsumed += 1
                    pending.late = False
                    materialize_seq_lens(batch)

        return run

    future_map_cls.resolve_seq_lens_cpu = resolve_seq_lens_cpu
    executor_cls.run_non_compact = _materializing(
        executor_cls.run_non_compact, late_ok=True
    )
    executor_cls.run_compact = _materializing(executor_cls.run_compact, late_ok=False)
    future_map_cls._atom_v41_deferred_seq_lens = True
    logger.info(
        "ATOM DeepSeek-V4.1 DSpark: decode rounds resolve host seq_lens after "
        "the draft launch (FutureMap.resolve_seq_lens_cpu deferred to %s; "
        "sglang %s; %s=0 to disable)",
        (
            f"the verify staging (late; {LATE_SEQ_LENS_ENV}=0 for run_non_compact)"
            if late_seq_lens_enabled()
            else "TargetVerifyExecutor.run_non_compact"
        ),
        _sglang_version(),
        DEFERRED_SEQ_LENS_ENV,
    )
    return True

# SPDX-License-Identifier: MIT
"""Radix (prefix) cache for DeepSeek-V4.1 on the SGLang plugin: replay first.

A prefix hit starts a request's extend at ``P > 0``: SGLang's tree hands back
the PAGEs below ``P`` (main rows and index planes, shared, read-only), but the
request's STATE slot (window ring, compressor rings, Engram cursor) holds
nothing at ``P``. Native ATOM rebuilds it by default, and SGLang's opt-in
encoder replay the same way, and so does this module: recompute the ``R`` rows
``[P - R, P)`` into the freshly reset slot, write no shared PAGE (ATOM's
compress plan writes no group below ``write_from = P``), and seed the Engram
history from the tokens before ``P - R``. The window rows below
``P - R + 127`` see a truncated window: a bounded approximation, the one
native ATOM defaults to (``Config.enable_encoder_swa_bounded_replay=True``,
``R = 128``, ``REPLAY_ROWS``). SGLang's default is exact instead (its SWA pool
keeps the window behind a hit); this module has no such pool and always
replays. Measured on an earlier head (200 prompts, first-token logits):
hit-vs-cold KL mean 0.036, top-1 agreement 0.925, against a cold-vs-cold floor
of 0.012 / 0.96 on the same server; GSM8K within 1 SE of radix off. Accepted
as matching native ATOM, not as exact.

The ``R`` rows run as native ATOM runs them (``BlockManager.replay_start``):
inside the step's own forward, prepended to the hit request's span. ATOM sees
that request scheduled over ``[P - R, end)`` with
``Replay(until=P, rebuilds=True, history_ids, start=P - R)``: its slot is
reset at ``P - R``, nothing global is written below ``P``, and its last row is
still the request's last row. SGLang's ``ForwardBatch`` is not changed: the
plugin widens only what ATOM executes and maps the output back.
``SGLangPluginRuntime`` takes the widened ``input_ids`` / ``positions`` from
``expand_v41_forward_inputs`` (the adapter's ``expand_forward_inputs``) and
gathers SGLang's rows out of the model's output (``trim_output``) before the
logits processor; the bridge widens the same request's span and adds its
``Replay``. The extra rows need no KV slot: they are positions SGLang's tree
already holds. The builder holds ``R`` rows more than a chunk: SGLang charges
each hit's ``R`` to the step's budget (``_install_admission_accounting``)
after admitting it, so a step runs at most ``R`` rows past its chunk.

Flow per prefill step (target worker, current batch only):

1. ``TpModelWorker.forward_batch_generation`` is wrapped, not copied: before
   the original runs on an ``EXTEND`` batch, ``run_v41_prefix_replay``
   selects the rows that hit (``prefix_lens[i] = P > 0``) on a new tenant of
   their STATE slot and returns a ``FusedReplay`` -- the hits' spans and the
   token ids of ``[P - R - h, P)`` from each request's own
   ``full_untruncated_fill_ids``, the record SGLang builds the step's
   ``input_ids`` from. It rides on the batch for the original call alone
   (set, then removed in ``finally``), on the one ``ForwardBatch``
   ``init_new`` builds from it (``FUSED_ATTR``) and on the view of it the
   eager runner hands the model (``EagerRunner.load_batch``, a
   ``dataclasses.replace`` that keeps dataclass fields only); the call raises
   if its forward never ran the replay.
2. ``deepseek_v41_bridge.build_v41_scheduled_batch`` turns it into ATOM's
   ``ScheduledBatch.replaying``; ATOM's ``_replay_layout`` / ``prepare_state``
   reset and seed the slot. No ATOM core change.
3. The bridge's guard, before any kernel runs, refuses an extend row at
   ``P > 0`` whose slot is neither stamped for this tenant nor rebuilt by the
   record this forward carries; the record is checked against the
   ``ForwardBatch`` it rides on (rows, slots, request ids, ``P``) and only then
   stamps its hits' slots (stamp on consume). The seam stamps every other row
   of the step. A record lost on the way to the model therefore leaves its
   hits unstamped, and the guard fails that forward before it extends from
   ``P`` over the previous tenant's state.

Ownership. Which rows replay is a function of current state alone: the step's
``prefix_lens`` and the proxy pool's per-slot stamp against SGLang's
``req_generation``. ``ReqToTokenPool.alloc_rows`` bumps ``req_generation[row]``
on every allocation (a new or retracted admission), and keeps it monotonic
across ``clear()`` precisely so that "readers detect row reuse by equality with
a generation they stored". The proxy pool owns one such stored value per STATE
slot (``ATOMDeepSeekV41ProxyKVPool.state_gen``), written only here. It is keyed
by the slot SGLang allocated, never by a batch position or request object, so
SGLang's filter / merge / reorder cannot strand it; free needs nothing (the
next tenant's generation differs); no decode graph reads it. The fused record
is a per-forward input like ``input_ids``, derived from the current batch,
consumed by that forward and gone with it; ATOM metadata stays
``f(current ForwardBatch, current pools, config)``.

Overlap scheduling: the seam runs inside ``forward_batch_generation`` on the
forward stream, after SGLang resolved the step's inputs, and reads only host
request state the scheduler already holds. The widened inputs are staged from
pinned memory, so a hit step does not stall the host. The decode staging's host
PAGE rows (``deepseek_v41_bridge.HostPageRows``) stay valid: a radix checkpoint
rewrites a live request's row in place, so every
``UnifiedRadixCache.checkpoint_into_tree`` drops that slot's host copy
(``_install_checkpoint_invalidation``), and the next decode reads it again.

Refused while radix is on (``deepseek_v41_bridge``): mixed chunk (a MIXED
batch skips the replay), session radix cache, unified memory (the only KV
relocation), KV storage backends (LMCache, FlexKV, the external cache linker:
trees that rewrite rows elsewhere), a custom radix backend, and
``--chunked-prefill-size`` below ``R``. Disabled chunked prefill is refused
for every launch. With ``--disable-radix-cache`` nothing here is installed.

Opt-in (``ATOM_V41_RADIX=1``): without it the bridge refuses a launch that
leaves SGLang's radix cache on.
"""

from __future__ import annotations

import logging
from functools import wraps
from typing import NamedTuple

import numpy as np
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardMode

logger = logging.getLogger("atom")

# R: the rows below a hit rebuilt into its slot, native ATOM's replay length
# (one hash block). At least V4.1's sliding window (128): a row at ``P`` reads
# ``window - 1`` ring rows below it, every one of which the replay writes.
REPLAY_ROWS = 128
# The attribute a fused replay rides on: the ScheduleBatch, for the one
# `forward_batch_generation` call it is planned for, the ForwardBatch
# `init_new` builds from it, and the eager runner's view of that.
FUSED_ATTR = "atom_v41_fused_replay"

# True while the replay is installed in this process (radix cache on).
_INSTALLED = False
# This process's metadata builder getter (`deepseek_v41_bridge`), bound at
# install so the seam does no import per step.
_get_builder = None
# Replays run by this process (logged).
_STATS = {"fused_steps": 0, "hits": 0, "rows": 0}


class ReplayHit(NamedTuple):
    """One batch row whose STATE slot is rebuilt below its prefix hit."""

    index: int  # row in the main batch
    slot: int  # req_pool_idx == STATE slot
    start: int  # first replayed position, P - R (floored at 0)
    end: int  # P, where the main extend resumes


def radix_replay_rows() -> int | None:
    """``R`` while the radix replay is installed in this process, else None."""
    return REPLAY_ROWS if _INSTALLED else None


# --- Which rows replay (pure) ---


def plan_prefix_replay(
    prefix_lens, slots, generations, state_gen, page_size: int
) -> list[ReplayHit]:
    """The rows of an extend step that start on a new tenant of their slot
    above position 0, and the span each replays.

    ``slots`` / ``generations``: int64 arrays, one per row. A chunked
    continuation keeps its row and generation, so it never replays; a row at
    ``P == 0`` is a cold prefill ATOM resets on its own. A step without a hit
    costs a few array operations, whatever its batch size.
    """
    prefix = np.asarray(prefix_lens, dtype=np.int64)
    index = np.flatnonzero((prefix > 0) & (state_gen[slots] != generations))
    if not index.size:
        return []
    ends = prefix[index]
    unaligned = ends % page_size != 0
    if unaligned.any():
        raise ValueError(
            f"DeepSeek-V4.1 prefix hits at {ends[unaligned].tolist()} on slots "
            f"{slots[index[unaligned]].tolist()} are not on a {page_size}-token "
            "page boundary"
        )
    starts = np.maximum(ends - REPLAY_ROWS, 0)
    return [
        ReplayHit(int(i), int(slots[i]), int(s), int(e))
        for i, s, e in zip(index, starts, ends)
    ]


def replay_admission_rows(req, prefix_len: int) -> int:
    """Rows an admission's replay costs: ``min(P, R)`` for a new or
    retracted request (the ones a fresh row is allocated for), 0 for others."""
    if int(prefix_len) <= 0:
        return 0
    if req.kv.req_pool_idx is not None and not req.is_retracted:
        return 0
    return min(int(prefix_len), REPLAY_ROWS)


# --- Token ids below a hit ---


def _check_no_cached_prompt_logprobs(req, hit: ReplayHit) -> None:
    """SGLang caps a match at ``logprob_start_len``, so no prompt row whose
    logprob is wanted is ever below a hit; the replay computes none."""
    start = int(req.logprob_start_len)
    if req.return_logprob and 0 <= start < hit.end:
        raise ValueError(
            "DeepSeek-V4.1 radix replay cannot return cached prompt logprobs "
            f"(logprob_start_len {start} < hit {hit.end})"
        )


def _replay_tokens(req, hit: ReplayHit, history_tokens: int):
    """The ids of ``[start, P)`` and the left-padded Engram history before
    ``start``, from the request's own token record -- the one SGLang builds
    the step's ``input_ids`` (and its own encoder replay) from."""
    _check_no_cached_prompt_logprobs(req, hit)
    fill = req.full_untruncated_fill_ids
    if len(fill) < hit.end:
        raise ValueError(
            f"DeepSeek-V4.1 replay to {hit.end} past the {len(fill)} tokens "
            "the request holds"
        )
    ids = [int(token) for token in fill[hit.start : hit.end]]
    before = [
        int(token) for token in fill[max(0, hit.start - history_tokens) : hit.start]
    ]
    # Left-padded as SGLang pads it; the bridge keeps only the real ones.
    return ids, [0] * (history_tokens - len(before)) + before


# --- The record: the rows below each hit ride in the step's own forward ---


class FusedReplay:
    """The rows below this step's prefix hits, for the step's own forward.

    Host data planned off the current batch: per hit (ascending batch row),
    its row, slot, request id, ``start`` (``P - R``) and ``end`` (``P``);
    ``ids``, the token ids of every hit's ``[start, P)`` back to back;
    ``histories``, the ``[hits, h]`` Engram history before each ``start``.
    ``expanded`` is the widened forward input, filled once per forward by
    ``expand_v41_forward_inputs`` for the bridge to stage; ``host``, the
    pinned host tensor its one H2D copy reads, held here for the record's
    lifetime (the forward's ``ForwardBatch``).
    """

    __slots__ = (
        "ends",
        "expanded",
        "histories",
        "host",
        "ids",
        "rids",
        "rows",
        "slots",
        "starts",
    )

    def __init__(self, rows, slots, rids, starts, ends, ids, histories):
        self.rows = np.asarray(rows, dtype=np.int64)
        self.slots = np.asarray(slots, dtype=np.int64)
        self.rids = list(rids)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.ends = np.asarray(ends, dtype=np.int64)
        self.ids = np.asarray(ids, dtype=np.int64)
        self.histories = np.asarray(histories, dtype=np.int64)
        self.expanded = None
        self.host = None
        n = self.rows.size
        if not (
            self.slots.size == n == len(self.rids) == self.starts.size
            and self.ends.size == n
            and self.histories.ndim == 2
            and self.histories.shape[0] == n
        ):
            raise ValueError("DeepSeek-V4.1 fused replay fields disagree in length")
        if n and np.any(np.diff(self.rows) <= 0):
            raise ValueError("DeepSeek-V4.1 fused replay rows must ascend")
        if np.any(self.starts < 0) or np.any(self.starts >= self.ends):
            raise ValueError(
                "DeepSeek-V4.1 fused replay spans must be non-empty: "
                f"starts={self.starts.tolist()} ends={self.ends.tolist()}"
            )
        if int(self.ids.size) != self.num_rows:
            raise ValueError(
                f"DeepSeek-V4.1 fused replay carries {self.ids.size} ids for "
                f"{self.num_rows} rows"
            )

    @property
    def lengths(self) -> np.ndarray:
        return self.ends - self.starts

    @property
    def num_rows(self) -> int:
        return int(self.lengths.sum())

    def positions(self) -> np.ndarray:
        """The positions of ``ids``: every hit's ``[start, P)``."""
        if not self.rows.size:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate(
            [np.arange(s, e, dtype=np.int64) for s, e in zip(self.starts, self.ends)]
        )

    def replay_rows(self, bs: int) -> np.ndarray:
        """``[bs]``: the rows prepended to each request of the step."""
        rows = np.zeros(int(bs), dtype=np.int64)
        if self.rows.size and (self.rows[-1] >= bs or self.rows[0] < 0):
            raise ValueError(
                f"DeepSeek-V4.1 fused replay rows {self.rows.tolist()} outside a "
                f"batch of {bs}"
            )
        rows[self.rows] = self.lengths
        return rows


def build_fused_replay(batch, hits, history_tokens: int) -> FusedReplay:
    """A ``FusedReplay`` of ``hits`` over the current batch's requests."""
    ids, histories = [], []
    for hit in hits:
        tokens, history = _replay_tokens(batch.reqs[hit.index], hit, history_tokens)
        ids.extend(tokens)
        histories.append(history)
    return FusedReplay(
        rows=[hit.index for hit in hits],
        slots=[hit.slot for hit in hits],
        rids=[batch.reqs[hit.index].rid for hit in hits],
        starts=[hit.start for hit in hits],
        ends=[hit.end for hit in hits],
        ids=ids,
        histories=np.asarray(histories, dtype=np.int64).reshape(
            len(hits), history_tokens
        ),
    )


def fused_layout(lengths, replay_rows):
    """Where SGLang's rows and the replay rows sit in the widened forward.

    ``lengths``: SGLang's extend rows per request; ``replay_rows``: the rows
    prepended to each. Request ``i`` occupies ``replay_rows[i]`` replay rows
    and then its ``lengths[i]`` own rows. Returns ``keep`` (``[sum(lengths)]``,
    the widened row of each SGLang row, in order) and ``replay_at``
    (``[sum(replay_rows)]``, the widened row of each replay row, in order).
    """
    lengths = np.asarray(lengths, dtype=np.int64)
    replay_rows = np.asarray(replay_rows, dtype=np.int64)
    wide = lengths + replay_rows
    wide_start = np.cumsum(wide) - wide
    own_start = np.cumsum(lengths) - lengths
    keep = np.repeat(wide_start + replay_rows - own_start, lengths) + np.arange(
        int(lengths.sum()), dtype=np.int64
    )
    replay_start = np.cumsum(replay_rows) - replay_rows
    replay_at = np.repeat(wide_start - replay_start, replay_rows) + np.arange(
        int(replay_rows.sum()), dtype=np.int64
    )
    return keep, replay_at


class ExpandedInputs(NamedTuple):
    """ATOM's widened inputs for one fused forward, and SGLang's rows in it."""

    input_ids: torch.Tensor
    positions: torch.Tensor
    output_rows: torch.Tensor


def expand_v41_forward_inputs(forward_batch, input_ids, positions, input_embeds=None):
    """The adapter's ``expand_forward_inputs``: with a fused replay on this
    ``ForwardBatch``, the ``input_ids`` / ``positions`` ATOM executes (each
    hit's ``[P - R, P)`` prepended to its rows) and the rows of the model's
    output that are SGLang's; None for every other forward.

    One asynchronous H2D of the host layout from a pinned tensor (no host
    stall under overlap); the widened tensors are scattered on the device
    from SGLang's own and are kept on the record for the bridge.
    """
    fused = getattr(forward_batch, FUSED_ATTR, None)
    if fused is None:
        return None
    if fused.expanded is not None:
        return fused.expanded
    if input_ids is None or input_embeds is not None:
        raise NotImplementedError(
            "DeepSeek-V4.1 fused radix replay needs token input_ids, not "
            "input_embeds"
        )
    if positions is None or positions.ndim != 1:
        raise NotImplementedError(
            "DeepSeek-V4.1 fused radix replay needs 1-D positions"
        )
    bs = int(forward_batch.batch_size)
    lengths = np.asarray(forward_batch.extend_seq_lens_cpu[:bs], dtype=np.int64)
    rows = int(lengths.sum())
    if int(input_ids.shape[0]) != rows or int(positions.shape[0]) != rows:
        raise ValueError(
            f"DeepSeek-V4.1 fused replay over {rows} extend rows got input_ids "
            f"{tuple(input_ids.shape)} and positions {tuple(positions.shape)}"
        )
    keep, replay_at = fused_layout(lengths, fused.replay_rows(bs))
    replayed = int(replay_at.size)
    sizes = (rows, replayed, replayed, replayed)
    device = input_ids.device
    # Pinned, so the copy below is asynchronous on the forward's stream.
    # Lifetime: the record holds it as long as this forward's ForwardBatch,
    # and the caching host allocator does not hand a freed pinned block out
    # again before the copies recorded on it have completed.
    host = torch.empty(sum(sizes), dtype=torch.int64, pin_memory=device.type != "cpu")
    np.concatenate((keep, replay_at, fused.ids, fused.positions()), out=host.numpy())
    fused.host = host
    staged = host.to(device, non_blocking=True)
    keep_d, at_d, ids_d, pos_d = torch.split(staged, sizes)
    total = rows + replayed
    wide_ids = input_ids.new_empty(total)
    wide_ids[keep_d] = input_ids
    wide_ids[at_d] = ids_d.to(input_ids.dtype)
    wide_pos = positions.new_empty(total)
    wide_pos[keep_d] = positions
    wide_pos[at_d] = pos_d.to(positions.dtype)
    fused.expanded = ExpandedInputs(wide_ids, wide_pos, keep_d)
    return fused.expanded


# --- The seam ---


def run_v41_prefix_replay(worker, batch) -> FusedReplay | None:
    """Plan the rows below every prefix hit of this step's batch and return
    the ``FusedReplay`` the step's forward must carry, or None without a hit.
    Every row's slot is stamped for its tenant here except a hit's, which the
    bridge stamps when the forward consumes the record (``stamp_fused_hits``).
    """
    if not _INSTALLED:
        return None
    mode = batch.forward_mode
    if not mode.is_extend_without_speculative():
        return None
    runner = worker.model_runner
    pool = runner.token_to_kv_pool
    if not getattr(pool, "is_atom_v41_proxy_pool", False):
        return None
    if mode != ForwardMode.EXTEND:
        raise NotImplementedError(
            f"DeepSeek-V4.1 radix replay runs on EXTEND batches, not {mode}; "
            "--enable-mixed-chunk is refused with radix cache"
        )
    builder = _get_builder()
    slots = batch.req_pool_indices_cpu.numpy()
    generations = runner.req_to_token_pool.req_generation.numpy()[slots]
    hits = plan_prefix_replay(
        batch.prefix_lens, slots, generations, pool.state_gen, builder.block_size
    )
    if not hits:
        # Every row now belongs to its current tenant: it needs no rebuild.
        pool.state_gen[slots] = generations
        return None
    fused = build_fused_replay(batch, hits, int(builder.geometry.history_size))
    widened = int(batch.extend_num_tokens) + fused.num_rows
    if widened > int(builder.max_num_batched_tokens):
        # The builder holds a chunk plus R rows and SGLang charges each hit's
        # R to the step's budget (`_install_admission_accounting`); chunked
        # prefill is always on, so no admitted step reaches this.
        raise RuntimeError(
            f"DeepSeek-V4.1 step widened by {fused.num_rows} replay rows to "
            f"{widened} exceeds the builder's {builder.max_num_batched_tokens}"
        )
    _STATS["fused_steps"] += 1
    _STATS["hits"] += len(hits)
    _STATS["rows"] += fused.num_rows
    if _STATS["fused_steps"] == 1:
        logger.info(
            "V4.1 radix replay: first fused step, %d hits / %d rows (R=%d)",
            len(hits),
            fused.num_rows,
            REPLAY_ROWS,
        )
    # A hit is stamped only when its forward consumes the record
    # (`stamp_fused_hits`), so a forward that lost it fails the bridge's
    # guard before any kernel runs.
    stamped = np.ones(slots.size, dtype=bool)
    stamped[fused.rows] = False
    pool.state_gen[slots[stamped]] = generations[stamped]
    return fused


def check_state_stamps(state_gen, req_generation, slots, prefix, carried=()) -> None:
    """The main forward's guard, before any kernel: an extend row resuming
    above 0 must sit on a slot stamped for its current tenant, or one this
    forward rebuilds (``carried``: the slots of the fused record it carries).

    Without this, a recycled slot whose last tenant's cursor happened to
    equal ``P`` would pass ATOM's cursor check with the wrong state, and a
    fused record lost before the model would extend from ``P`` over it.
    """
    slots = np.asarray(slots, dtype=np.int64)
    resuming = slots[np.asarray(prefix) > 0]
    if len(carried):
        resuming = resuming[~np.isin(resuming, np.asarray(carried, dtype=np.int64))]
    if not resuming.size:
        return
    if torch.is_tensor(req_generation):
        req_generation = req_generation.numpy()
    stale = resuming[
        np.asarray(state_gen)[resuming] != np.asarray(req_generation)[resuming]
    ]
    if stale.size:
        raise RuntimeError(
            "DeepSeek-V4.1 extend resumes slots "
            f"{stale.tolist()} above position 0 whose STATE was not rebuilt "
            "for their current request (radix replay seam skipped)"
        )


def stamp_fused_hits(state_gen, req_generation, fused) -> None:
    """The bridge's stamp on consume: the forward about to run carries
    ``fused``, checked against its ForwardBatch and widened, so its hits'
    slots now belong to their tenants."""
    if fused.expanded is None:
        raise RuntimeError(
            "DeepSeek-V4.1 fused radix replay reached the bridge without its "
            "widened inputs"
        )
    if torch.is_tensor(req_generation):
        req_generation = req_generation.numpy()
    state_gen[fused.slots] = np.asarray(req_generation)[fused.slots]


# --- Install ---


def run_with_fused_replay(original, worker, batch, fused, args, kwargs):
    """``original(worker, batch, ...)`` with ``fused`` on the batch for this
    call alone; ``init_new`` copies it onto the call's ForwardBatch.

    The call must have run the replay. A forward that lost the record on the
    way to the model already failed the bridge's guard before any kernel
    (its hits stay unstamped until the bridge consumes the record); this is
    the second line, for a call that returned without a forward at all."""
    setattr(batch, FUSED_ATTR, fused)
    try:
        result = original(worker, batch, *args, **kwargs)
    finally:
        setattr(batch, FUSED_ATTR, None)
    if fused.expanded is None:
        raise RuntimeError(
            "DeepSeek-V4.1 fused radix replay planned for slots "
            f"{fused.slots.tolist()} never reached the model's forward"
        )
    return result


def _install_worker_seam() -> None:
    from sglang.srt.managers.tp_worker import TpModelWorker

    original = TpModelWorker.forward_batch_generation
    if getattr(original, "_atom_v41_radix_patched", False):
        return

    @wraps(original)
    def forward_batch_generation(self, batch=None, *args, **kwargs):
        fused = None if batch is None else run_v41_prefix_replay(self, batch)
        if fused is None:
            return original(self, batch, *args, **kwargs)
        return run_with_fused_replay(original, self, batch, fused, args, kwargs)

    forward_batch_generation._atom_v41_radix_patched = True
    TpModelWorker.forward_batch_generation = forward_batch_generation


def carry_fused_replay(batch, forward_batch) -> None:
    """``init_new``'s addition: a fused replay planned for ``batch`` rides on
    the ForwardBatch built from it, an EXTEND, and nothing else."""
    fused = getattr(batch, FUSED_ATTR, None)
    if fused is None:
        return
    mode = forward_batch.forward_mode
    if mode != ForwardMode.EXTEND:
        raise NotImplementedError(
            f"DeepSeek-V4.1 fused radix replay rides on an EXTEND, not {mode}"
        )
    setattr(forward_batch, FUSED_ATTR, fused)


def _install_forward_batch_carrier() -> None:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    if getattr(ForwardBatch, "_atom_v41_radix_carrier_patched", False):
        return
    original_init_new = ForwardBatch.init_new.__func__

    @wraps(original_init_new)
    def init_new(cls, batch, model_runner, *args, **kwargs):
        forward_batch = original_init_new(cls, batch, model_runner, *args, **kwargs)
        carry_fused_replay(batch, forward_batch)
        return forward_batch

    ForwardBatch.init_new = classmethod(init_new)
    ForwardBatch._atom_v41_radix_carrier_patched = True


def carry_fused_view(forward_batch, view) -> None:
    """``EagerRunner.load_batch``'s addition: the forward-batch view it hands
    the model (``dataclasses.replace`` of the ForwardBatch over its static
    input buffers, which keeps dataclass fields only) carries the fused
    replay of the ForwardBatch it was made from."""
    fused = getattr(forward_batch, FUSED_ATTR, None)
    if fused is None or view is forward_batch:
        return
    setattr(view, FUSED_ATTR, fused)


def _install_eager_view_carrier() -> None:
    from sglang.srt.model_executor.runner.eager_runner import EagerRunner

    original = EagerRunner.load_batch
    if getattr(original, "_atom_v41_radix_patched", False):
        return

    @wraps(original)
    def load_batch(self, forward_batch, *args, **kwargs):
        view = original(self, forward_batch, *args, **kwargs)
        carry_fused_view(forward_batch, view)
        return view

    load_batch._atom_v41_radix_patched = True
    EagerRunner.load_batch = load_batch


def _install_checkpoint_invalidation() -> None:
    """Drop a slot's host PAGE-row copy whenever the tree rewrites its row.

    ``checkpoint_into_tree`` repoints a live request's ``req_to_token`` row
    at the tree's canonical pages and frees its own duplicates, the one
    in-place rewrite of a live row (sessions and unified memory are refused
    with radix). A decode staged before it read the old pages, which nothing
    reuses before that decode's forward has run (reuse is a later step's
    forward on the same stream); every decode staged after it reads the row
    again (``HostPageRows.lookup`` misses).
    """
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    from atom.plugin.sglang import deepseek_v41_bridge as bridge

    original = UnifiedRadixCache.checkpoint_into_tree
    if getattr(original, "_atom_v41_radix_patched", False):
        return

    @wraps(original)
    def checkpoint_into_tree(self, req, *args, **kwargs):
        slot = req.kv.req_pool_idx
        try:
            return original(self, req, *args, **kwargs)
        finally:
            if slot is not None:
                bridge.forget_host_page_rows(slot)

    checkpoint_into_tree._atom_v41_radix_patched = True
    UnifiedRadixCache.checkpoint_into_tree = checkpoint_into_tree


def _install_admission_accounting() -> None:
    from sglang.srt.managers.schedule_policy import PrefillAdder

    original = PrefillAdder._account_prefill_cache_admission
    if getattr(original, "_atom_v41_radix_patched", False):
        return

    @wraps(original)
    def account(self, req, prefix_len):
        charge = replay_admission_rows(req, prefix_len) if _INSTALLED else 0
        if charge:
            # Budget and metrics only (`#replay-token` in the prefill log).
            self.log_replay_tokens += charge
            self.rem_input_tokens -= charge
            if self.rem_chunk_tokens is not None:
                self.rem_chunk_tokens -= charge
        return original(self, req, prefix_len)

    account._atom_v41_radix_patched = True
    PrefillAdder._account_prefill_cache_admission = account


def install_deepseek_v41_radix(window: int | None = None) -> int:
    """Turn the replay on in this process: the worker seam, the ForwardBatch
    carriers (``init_new`` and the eager runner's view), the host-row
    invalidation and the admission charge. Idempotent; returns ``R``.
    Called only with radix cache on."""
    global _INSTALLED, _get_builder
    from atom.plugin.sglang import deepseek_v41_bridge as bridge

    if window is not None and REPLAY_ROWS < int(window):
        raise ValueError(
            f"DeepSeek-V4.1 radix replay of {REPLAY_ROWS} rows is below the "
            f"sliding window {int(window)}: the rows at a hit would read ring "
            "rows nobody wrote"
        )
    _install_worker_seam()
    _install_forward_batch_carrier()
    _install_eager_view_carrier()
    _install_checkpoint_invalidation()
    _install_admission_accounting()
    _get_builder = bridge.get_deepseek_v41_builder
    _INSTALLED = True
    logger.info(
        "ATOM DeepSeek-V4.1 radix cache: replay-first, R=%d rows below each "
        "hit, fused into the step's forward",
        REPLAY_ROWS,
    )
    return REPLAY_ROWS

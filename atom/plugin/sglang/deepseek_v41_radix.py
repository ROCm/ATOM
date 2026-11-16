# SPDX-License-Identifier: MIT
"""Radix (prefix) cache for DeepSeek-V4.1 on the SGLang plugin: replay first.

A prefix hit starts a request's extend at ``P > 0``: SGLang's tree hands back
the PAGEs below ``P`` (main rows and index planes, shared, read-only), but the
request's STATE slot (window ring, compressor rings, Engram cursor) holds
nothing at ``P``. Native ATOM and native SGLang rebuild it the same way, and so
does this module: recompute the ``R`` rows ``[P - R, P)`` into the freshly
reset slot, write no shared PAGE (ATOM's compress plan writes no group below
``write_from = P``), and seed the Engram history from the tokens before
``P - R``. The main extend then resumes at ``P`` on a slot whose cursor is
``P``. The window rows below ``P - R + 127`` see a truncated window, the same
bounded approximation native ATOM and SGLang default to.

Flow per prefill step (target worker, current batch only):

1. ``TpModelWorker.forward_batch_generation`` is wrapped, not copied: before
   the original runs on an ``EXTEND`` batch, ``run_v41_prefix_replay``
   selects the rows that hit (``prefix_lens[i] = P > 0``) on a new tenant of
   their STATE slot, builds ONE side ``ScheduleBatch`` over all of them (split
   only past the builder's capacity) the way SGLang's own
   ``run_encoder_swa_replay`` builds its single-request one, and runs it
   through ``ModelRunner.forward`` with the official
   ``ForwardBatch.encoder_swa_replay`` and ``engram_history`` fields set.
2. ``deepseek_v41_bridge.build_v41_scheduled_batch`` turns that side forward
   into ATOM's ``ScheduledBatch.replaying`` (``Replay(until=P, rebuilds=True,
   history_ids, start)``); ATOM's ``_replay_layout`` / ``prepare_state``
   reset and seed the slot and advance its cursor to ``P``. No ATOM core
   change.
3. The main forward runs unchanged. Its bridge guard refuses an extend row
   at ``P > 0`` whose slot was not stamped for this tenant.

Which rows are new tenants: ``ReqToTokenPool.alloc_rows`` bumps
``req_generation[row]`` on every allocation (a new or retracted admission),
and keeps it monotonic across ``clear()`` precisely so that "readers detect
row reuse by equality with a generation they stored". The proxy pool owns one
such stored value per STATE slot (``ATOMDeepSeekV41ProxyKVPool.state_gen``),
written only here, after a step's rows are staged. This is pool state beside
the slot's cursor, not private scheduler shadow state: it is keyed by the slot
SGLang allocated, never by a batch position or request object, so SGLang's
filter / merge / reorder cannot strand it; create is SGLang's own generation
bump; free needs nothing (the next tenant's generation differs); flush keeps
the generation monotonic; no decode graph reads it. ATOM metadata stays
``f(current ForwardBatch, current pools, config)``: the side forward's is a
pure function of its own ForwardBatch.

Refused while radix is on (``deepseek_v41_bridge``): mixed chunk (a MIXED
batch skips the replay), overlap scheduling, session radix cache, unified
memory (the only KV relocation), prefill CUDA graphs, a custom radix backend,
``--chunked-prefill-size`` below ``R``, and speculative decoding (the DSpark
interplay is a later phase). With ``--disable-radix-cache`` nothing here is
installed.
"""

from __future__ import annotations

import logging
from copy import copy
from functools import wraps
from typing import NamedTuple

import numpy as np
import torch

logger = logging.getLogger("atom")

REPLAY_ROWS_ENV = "ATOM_V41_RADIX_REPLAY_ROWS"
DEFAULT_REPLAY_ROWS = 128

# The installed replay length R; None while radix cache is off.
_REPLAY_ROWS: int | None = None
# Side forwards run and rows replayed by this process (logged, and read by
# the debug path).
_STATS = {"side_forwards": 0, "hits": 0, "rows": 0, "steps": 0}


class ReplayHit(NamedTuple):
    """One batch row whose STATE slot is rebuilt below its prefix hit."""

    index: int  # row in the main batch
    slot: int  # req_pool_idx == STATE slot
    start: int  # first replayed position, P - R (floored at 0)
    end: int  # P, where the main extend resumes


def radix_replay_rows() -> int | None:
    """``R`` while the radix replay is installed in this process, else None."""
    return _REPLAY_ROWS


def resolve_replay_rows(window: int | None = None) -> int:
    """``ATOM_V41_RADIX_REPLAY_ROWS``, validated: even, positive, and at least
    the sliding window when known (a row at ``P`` reads ``window - 1`` ring
    rows below it, every one of which the replay has to have written)."""
    from atom.plugin.sglang import deepseek_v41_envs as envs

    rows = int(envs.ATOM_V41_RADIX_REPLAY_ROWS)
    if rows <= 0 or rows % 2:
        raise ValueError(
            f"{REPLAY_ROWS_ENV} must be a positive even row count; got {rows}"
        )
    if window is not None and rows < int(window):
        raise ValueError(
            f"{REPLAY_ROWS_ENV}={rows} is below DeepSeek-V4.1's sliding window "
            f"{int(window)}: the rows at a hit would read ring rows nobody wrote"
        )
    return rows


# ---------------------------------------------------------------------------
# Which rows replay (pure)
# ---------------------------------------------------------------------------


def plan_prefix_replay(
    prefix_lens, slots, generations, state_gen, rows: int, page_size: int
) -> list[ReplayHit]:
    """The rows of an extend step that start on a new tenant of their slot
    above position 0, and the span each replays.

    A chunked continuation keeps its row and generation, so it never
    replays; a row at ``P == 0`` is a cold prefill ATOM resets on its own.
    """
    hits = []
    for i, (prefix, slot, generation) in enumerate(
        zip(prefix_lens, slots, generations, strict=True)
    ):
        prefix, slot = int(prefix), int(slot)
        if prefix <= 0 or int(state_gen[slot]) == int(generation):
            continue
        if prefix % page_size:
            raise ValueError(
                f"DeepSeek-V4.1 prefix hit at {prefix} on slot {slot} is not on "
                f"a {page_size}-token page boundary"
            )
        hits.append(ReplayHit(i, slot, max(0, prefix - rows), prefix))
    return hits


def split_replay_groups(hits, max_bs: int, max_tokens: int) -> list[list[ReplayHit]]:
    """Hits packed greedily into side forwards the builder can stage."""
    groups, group, tokens = [], [], 0
    for hit in hits:
        n = hit.end - hit.start
        if n > max_tokens:
            raise ValueError(
                f"DeepSeek-V4.1 replay of {n} rows exceeds the builder's "
                f"{max_tokens} tokens; lower {REPLAY_ROWS_ENV}"
            )
        if group and (len(group) == max_bs or tokens + n > max_tokens):
            groups.append(group)
            group, tokens = [], 0
        group.append(hit)
        tokens += n
    if group:
        groups.append(group)
    return groups


def replay_admission_rows(req, prefix_len: int, rows: int) -> int:
    """Rows an admission's replay costs: ``min(P, R)`` for a new or
    retracted request (the ones a fresh row is allocated for), 0 for others."""
    if int(prefix_len) <= 0:
        return 0
    kv = getattr(req, "kv", None)
    new_row = getattr(kv, "req_pool_idx", None) is None
    if not (new_row or getattr(req, "is_retracted", False)):
        return 0
    return min(int(prefix_len), int(rows))


# ---------------------------------------------------------------------------
# The side forward
# ---------------------------------------------------------------------------


def _host_slots(batch) -> list[int]:
    slots = getattr(batch, "req_pool_indices_cpu", None)
    if slots is None:
        slots = batch.req_pool_indices
    if torch.is_tensor(slots):
        slots = slots.detach().cpu().tolist()
    return [int(slot) for slot in slots]


def _check_no_cached_prompt_logprobs(req, hit: ReplayHit) -> None:
    """SGLang caps a match at ``logprob_start_len``, so no prompt row whose
    logprob is wanted is ever below a hit; the replay computes none."""
    start = getattr(req, "logprob_start_len", -1)
    if getattr(req, "return_logprob", False) and 0 <= int(start) < hit.end:
        raise ValueError(
            "DeepSeek-V4.1 radix replay cannot return cached prompt logprobs "
            f"(logprob_start_len {start} < hit {hit.end})"
        )


def build_side_batch(batch, hits, runner, history_tokens: int):
    """One side ``ScheduleBatch`` replaying ``[start, P)`` of every hit.

    A shallow copy of the step's batch, rewritten the way SGLang's
    ``run_encoder_swa_replay`` rewrites its single-request copy: the hits'
    requests, token ids and slots, extend from ``start`` to ``P`` over the
    token slots ``req_to_token`` already holds, nothing sampled, no logprobs.
    """
    device = runner.device
    req_to_token = runner.req_to_token_pool.req_to_token
    reqs, ids, histories = [], [], []
    for hit in hits:
        req = batch.reqs[hit.index]
        _check_no_cached_prompt_logprobs(req, hit)
        fill = req.full_untruncated_fill_ids
        if len(fill) < hit.end:
            raise ValueError(
                f"DeepSeek-V4.1 replay to {hit.end} past the {len(fill)} tokens "
                "the request holds"
            )
        reqs.append(req)
        ids.extend(int(token) for token in fill[hit.start : hit.end])
        before = [
            int(token) for token in fill[max(0, hit.start - history_tokens) : hit.start]
        ]
        # Left-padded as SGLang pads it; the bridge keeps only the real ones.
        histories.append([0] * (history_tokens - len(before)) + before)
    starts = [hit.start for hit in hits]
    ends = [hit.end for hit in hits]
    lengths = [end - start for start, end in zip(starts, ends)]
    slots = [hit.slot for hit in hits]

    side = copy(batch)
    side.reqs = reqs
    side.input_ids = torch.tensor(ids, dtype=torch.int64, device=device)
    side.prefill_input_ids_cpu = None
    side.req_pool_indices = torch.tensor(slots, dtype=torch.int64, device=device)
    side.req_pool_indices_cpu = torch.tensor(slots, dtype=torch.int64)
    side.prefix_lens = starts
    side.extend_lens = lengths
    side.extend_num_tokens = sum(lengths)
    side.seq_lens = torch.tensor(ends, dtype=torch.int64, device=device)
    side.seq_lens_cpu = torch.tensor(ends, dtype=torch.int64)
    side.seq_lens_sum = sum(ends)
    side.orig_seq_lens = torch.tensor(ends, dtype=torch.int32, device=device)
    side.out_cache_loc = torch.cat(
        [req_to_token[hit.slot, hit.start : hit.end] for hit in hits]
    ).long()
    side.return_logprob = False
    side.top_logprobs_nums = None
    side.token_ids_logprobs = None
    side.extend_logprob_start_lens = list(lengths)
    side.extend_input_logprob_token_ids = None
    side.is_prefill_only = True
    side.spec_info = None
    side.sampling_info = None
    side.has_grammar = False
    side.multimodal_inputs = [None] * len(hits)
    side.engram_history = torch.tensor(
        histories, dtype=torch.int32, device=device
    ).view(len(hits), history_tokens)
    return side


def _run_side_forward(runner, side) -> None:
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardBatch,
    )

    fb = ForwardBatch.init_new(
        side,
        runner,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        return_hidden_states_before_norm=False,
    )
    fb.encoder_swa_replay = True
    runner.forward(fb)


# ---------------------------------------------------------------------------
# Diagnostics (ATOM_V41_RADIX_DEBUG)
# ---------------------------------------------------------------------------


def page_digests(cache, pages: torch.Tensor, chunk: int = 2) -> torch.Tensor:
    """``[len(pages), 2]`` int64 digests of every byte a PAGE owns: its main
    rows (`page_regions`: one region for a BF16 pool, each owner's data and
    scale source for a quantized one) and each owner's index-plane rows.
    Equal before and after a forward iff (up to a checksum collision) that
    forward wrote none of them."""
    num_pages = int(cache.num_pages)
    regions = [view.reshape(num_pages, -1) for _, view in cache.page_regions] + [
        plane.reshape(num_pages, -1)
        for planes in cache.index_planes.values()
        for plane in planes
    ]
    main = regions[0]
    pages = pages.to(device=main.device, dtype=torch.int64)
    out = torch.zeros(int(pages.numel()), 2, dtype=torch.int64, device=main.device)
    for region in regions:
        width = int(region.shape[1])
        weights = torch.arange(width, device=main.device, dtype=torch.int64) % 65521 + 1
        for begin in range(0, int(pages.numel()), chunk):
            rows = region[pages[begin : begin + chunk]].to(torch.int64)
            out[begin : begin + chunk, 0] += rows.sum(dim=1)
            out[begin : begin + chunk, 1] += (rows * weights).sum(dim=1)
    return out


def _hit_pages(req_to_token, hits, page_size: int) -> torch.Tensor:
    pages = [
        req_to_token[hit.slot, : hit.end : page_size].long() // page_size
        for hit in hits
    ]
    return torch.unique(torch.cat(pages))


def _tp_agreement(count: int) -> str:
    try:
        from sglang.srt.distributed.parallel_state import get_tp_group

        group = get_tp_group()
        if group.world_size <= 1:
            return "single"
        values = torch.tensor([count, -count], dtype=torch.int64)
        torch.distributed.all_reduce(
            values, op=torch.distributed.ReduceOp.MAX, group=group.cpu_group
        )
        high, low = int(values[0]), -int(values[1])
        return "yes" if high == low == count else f"NO(min={low},max={high})"
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        return f"unknown({type(exc).__name__})"


# ---------------------------------------------------------------------------
# The seam
# ---------------------------------------------------------------------------


def run_v41_prefix_replay(worker, batch) -> int:
    """Rebuild the STATE below every prefix hit of this step's batch, then
    stamp every row's slot for its tenant. Returns the rows replayed."""
    rows = _REPLAY_ROWS
    if rows is None or batch is None:
        return 0
    mode = batch.forward_mode
    if not mode.is_extend_without_speculative():
        return 0
    runner = worker.model_runner
    pool = runner.token_to_kv_pool
    if not getattr(pool, "is_atom_v41_proxy_pool", False):
        return 0
    if getattr(mode, "name", None) != "EXTEND":
        raise NotImplementedError(
            f"DeepSeek-V4.1 radix replay runs on EXTEND batches, not {mode}; "
            "--enable-mixed-chunk is refused with radix cache"
        )
    from atom.plugin.sglang.deepseek_v41_bridge import get_deepseek_v41_builder

    builder = get_deepseek_v41_builder()
    req_to_token_pool = runner.req_to_token_pool
    slots = _host_slots(batch)
    generations = req_to_token_pool.req_generation[slots].tolist()
    hits = plan_prefix_replay(
        batch.prefix_lens,
        slots,
        generations,
        pool.state_gen,
        rows,
        int(builder.block_size),
    )
    replayed = 0
    from atom.plugin.sglang import deepseek_v41_envs as envs

    debug = bool(envs.ATOM_V41_RADIX_DEBUG)
    agreement = None
    if debug:
        # Every rank, every EXTEND step, hits or not: a rank that disagrees on
        # whether the step has a hit must be reported, not hang the others.
        agreement = _tp_agreement(sum(int(hit.end - hit.start) for hit in hits))
        if agreement not in ("yes", "single"):
            logger.error(
                "V4.1 radix debug: TP ranks disagree on replay rows: %s", agreement
            )
    if hits:
        history = int(builder.geometry.history_size)
        for group in split_replay_groups(
            hits, int(builder.max_bs), int(builder.max_num_batched_tokens)
        ):
            side = build_side_batch(batch, group, runner, history)
            count = int(side.extend_num_tokens)
            if debug:
                pages = _hit_pages(
                    req_to_token_pool.req_to_token, group, int(builder.block_size)
                )
                before = page_digests(builder.cache, pages)
            _run_side_forward(runner, side)
            replayed += count
            _STATS["side_forwards"] += 1
            _STATS["hits"] += len(group)
            _STATS["rows"] += count
            if debug:
                changed = int(
                    (page_digests(builder.cache, pages) != before).any(dim=1).sum()
                )
                log = logger.error if changed else logger.info
                log(
                    "V4.1 radix debug: side_forward=%d hits=%d rows=%d spans=%s "
                    "pages=%d pages_changed=%d tp_agree=%s",
                    _STATS["side_forwards"],
                    len(group),
                    count,
                    [(hit.slot, hit.start, hit.end) for hit in group],
                    int(pages.numel()),
                    changed,
                    agreement,
                )
            elif _STATS["side_forwards"] == 1:
                logger.info(
                    "V4.1 radix replay: first side forward, %d hits / %d rows "
                    "(R=%d)",
                    len(group),
                    count,
                    rows,
                )
    # Every row of the step, P = 0 and continuations included, now belongs
    # to its current tenant.
    pool.state_gen[slots] = generations
    _STATS["steps"] += 1
    return replayed


def check_state_stamps(state_gen, req_generation, slots, prefix) -> None:
    """The main forward's guard: an extend row resuming above 0 must sit on
    a slot stamped for its current tenant, i.e. the replay seam ran for it.

    Without this, a recycled slot whose last tenant's cursor happened to
    equal ``P`` would pass ATOM's cursor check with the wrong state.
    """
    slots = np.asarray(slots, dtype=np.int64)
    resuming = slots[np.asarray(prefix) > 0]
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


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------


def _install_worker_seam() -> None:
    from sglang.srt.managers.tp_worker import TpModelWorker

    original = TpModelWorker.forward_batch_generation
    if getattr(original, "_atom_v41_radix_patched", False):
        return

    @wraps(original)
    def forward_batch_generation(self, batch=None, *args, **kwargs):
        if batch is not None:
            run_v41_prefix_replay(self, batch)
        return original(self, batch, *args, **kwargs)

    forward_batch_generation._atom_v41_radix_patched = True
    TpModelWorker.forward_batch_generation = forward_batch_generation


def _install_admission_accounting() -> None:
    from sglang.srt.managers.schedule_policy import PrefillAdder

    original = PrefillAdder._account_prefill_cache_admission
    if getattr(original, "_atom_v41_radix_patched", False):
        return

    @wraps(original)
    def account(self, req, prefix_len):
        rows = _REPLAY_ROWS
        if rows is not None:
            charge = replay_admission_rows(req, prefix_len, rows)
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
    """Turn the replay on in this process: the worker seam and the admission
    charge. Idempotent; returns ``R``. Called only with radix cache on."""
    global _REPLAY_ROWS
    rows = resolve_replay_rows(window)
    _install_worker_seam()
    _install_admission_accounting()
    _REPLAY_ROWS = rows
    logger.info(
        "ATOM DeepSeek-V4.1 radix cache: replay-first, R=%d rows below each hit",
        rows,
    )
    return rows

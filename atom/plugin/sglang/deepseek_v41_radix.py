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
``R = 128``). SGLang's default is exact instead
(``--enable-encoder-swa-bounded-replay`` is off: its SWA pool keeps the window
behind a hit); this module has no such pool and always replays. Measured cost
(radix EXPERIMENTS E8, 200 prompts, first-token logits): hit-vs-cold KL mean
0.036, top-1 agreement 0.925, against a cold-vs-cold floor of 0.012 / 0.96 on
the same server; GSM8K within 1 SE of radix off. Accepted as matching native
ATOM, not as exact.

Where the ``R`` rows run (``ATOM_V41_RADIX_REPLAY_MODE``):

- ``fused`` (default), as native ATOM runs them (``BlockManager.replay_start``):
  inside the step's own forward, prepended to the hit request's span. ATOM
  sees that request scheduled over ``[P - R, end)`` with
  ``Replay(until=P, rebuilds=True, history_ids, start=P - R)``: its slot is
  reset at ``P - R``, nothing global is written below ``P``, and its last row
  is still the request's last row. SGLang's ``ForwardBatch`` is not changed;
  the plugin widens only what ATOM executes and maps the output back:
  ``SGLangPluginRuntime`` takes the widened ``input_ids`` / ``positions``
  from ``expand_v41_forward_inputs`` (the adapter's
  ``expand_forward_inputs``) and gathers SGLang's rows out of the model's
  output (``trim_output``) before the logits processor; the bridge widens the
  same request's span and adds its ``Replay``. The extra rows need no KV
  slot: they are positions SGLang's tree already holds, and ATOM writes no
  PAGE row below ``P``.
- ``side`` (the first implementation, kept for A/B): one separate EXTEND over
  every hit's ``[P - R, P)`` before the step's forward, run through
  ``ModelRunner.forward`` with the official ``ForwardBatch.encoder_swa_replay``
  and ``engram_history`` fields set, as SGLang's own ``run_encoder_swa_replay``
  builds its single-request one. It costs one more eager forward per step
  with a hit (~0.2 s on MI308X), which made hits below ~4k slower than cold.

Flow per prefill step (target worker, current batch only):

1. ``TpModelWorker.forward_batch_generation`` is wrapped, not copied: before
   the original runs on an ``EXTEND`` batch, ``run_v41_prefix_replay``
   selects the rows that hit (``prefix_lens[i] = P > 0``) on a new tenant of
   their STATE slot. Fused, it returns a ``FusedReplay`` -- the hits' spans
   and the token ids of ``[P - R - h, P)`` from each request's own
   ``full_untruncated_fill_ids``, the record SGLang builds the step's
   ``input_ids`` from -- which rides on the batch for the original call alone
   (set, then removed in ``finally``), on the one ``ForwardBatch``
   ``init_new`` builds from it (``FUSED_ATTR``) and on the view of it the
   eager runner hands the model (``EagerRunner.load_batch``, a
   ``dataclasses.replace`` that keeps dataclass fields only); the call
   raises if its forward never ran the replay. A step whose widened rows
   would not fit the builder falls back to the side forward. Side, it runs
   the side forward(s) there.
2. ``deepseek_v41_bridge.build_v41_scheduled_batch`` turns either into ATOM's
   ``ScheduledBatch.replaying`` (``Replay(until=P, rebuilds=True,
   history_ids, start)``); ATOM's ``_replay_layout`` / ``prepare_state``
   reset and seed the slot. No ATOM core change.
3. The main forward's bridge guard, before any kernel runs, refuses an extend
   row at ``P > 0`` whose slot is neither stamped for this tenant nor
   rebuilt by the fused record this forward carries; the record is checked
   against the ``ForwardBatch`` it rides on (rows, slots, request ids, ``P``)
   and only then stamps its hits' slots (stamp on consume). The seam stamps
   every other row of the step, and a side replay's rows once their side
   forward has run. A record lost on the way to the model therefore leaves
   its hits unstamped, and the guard fails that forward before it extends
   from ``P`` over the previous tenant's state.

Admission: while the replay is installed, a request whose rows are not
determined by its token ids -- ``input_embeds`` (SGLang matches it by
placeholder ids ``[1] * n``), ``positional_embed_overrides`` or multimodal
inputs -- is answered with a 400 at the scheduler's queue door
(``Scheduler._add_request_to_queue``) and never reaches a batch, as SGLang's
own encoder replay refuses it (``TokenizerManager._validate_one_request``,
``ScheduleBatch.prepare_for_extend``). Its tree match would be a placeholder
match, the replay recomputes rows from token ids, and a batch carrying any
of them hands the model batch-wide ``input_embeds`` that a fused widening
cannot extend. With radix cache off nothing here is installed and such
requests are served as before.

Ownership. Which rows replay is a function of current state alone: the step's
``prefix_lens`` and the proxy pool's per-slot stamp against SGLang's
``req_generation``. ``ReqToTokenPool.alloc_rows`` bumps ``req_generation[row]``
on every allocation (a new or retracted admission), and keeps it monotonic
across ``clear()`` precisely so that "readers detect row reuse by equality with
a generation they stored". The proxy pool owns one such stored value per STATE
slot (``ATOMDeepSeekV41ProxyKVPool.state_gen``), written only here, once a
step's rows are planned. This is pool state beside the slot's cursor, not
private scheduler shadow state: it is keyed by the slot SGLang allocated, never
by a batch position or request object, so SGLang's filter / merge / reorder
cannot strand it; create is SGLang's own generation bump; free needs nothing
(the next tenant's generation differs); flush keeps the generation monotonic;
no decode graph reads it. The fused record is a per-forward input like
``input_ids`` (the rows SGLang's ``ForwardBatch`` has no field for), derived
from the current batch, consumed by that forward and gone with it; ATOM
metadata stays ``f(current ForwardBatch, current pools, config)``.

Overlap scheduling: SGLang runs its own encoder replay seam under overlap, and
so does this one -- the seam runs inside ``forward_batch_generation`` on the
forward stream, after SGLang resolved the step's inputs, and reads only host
request state the scheduler already holds. What overlap changes is the decode
staging's host PAGE rows (``deepseek_v41_bridge.HostPageRows``): a radix
checkpoint rewrites a live request's row in place, so every
``UnifiedRadixCache.checkpoint_into_tree`` drops that slot's host copy
(``_install_checkpoint_invalidation``), and the next decode reads it again.

Refused while radix is on (``deepseek_v41_bridge``): mixed chunk (a MIXED
batch skips the replay), session radix cache, unified memory (the only KV
relocation), KV storage backends (LMCache, FlexKV, the external cache linker:
trees that rewrite rows elsewhere), prefill CUDA graphs, a custom radix
backend, ``--chunked-prefill-size`` below ``R``, and speculative decoding (the
DSpark interplay is a later phase). With ``--disable-radix-cache`` nothing
here is installed.

On by default: whenever SGLang's radix cache is on (no
``--disable-radix-cache``). A launch with speculative decoding, or with
``ATOM_V41_RADIX=0``, gets SGLang's radix cache turned off at server-args
resolution with one warning (``v41_radix_launch_override``), so it starts
with radix off instead of failing.
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
REPLAY_MODE_ENV = "ATOM_V41_RADIX_REPLAY_MODE"
DEFAULT_REPLAY_ROWS = 128
REPLAY_MODES = ("fused", "side")
# The attribute a fused replay rides on: the ScheduleBatch, for the one
# `forward_batch_generation` call it is planned for, the ForwardBatch
# `init_new` builds from it, and the eager runner's view of that.
FUSED_ATTR = "atom_v41_fused_replay"
# Request fields whose rows the token ids do not determine (admission).
_NON_TOKEN_INPUTS = (
    ("input_embeds", "input_embeds"),
    ("positional_embed_overrides", "positional_embed_overrides"),
    ("multimodal inputs", "multimodal_inputs"),
)

# The installed replay length R and mode; None while radix cache is off.
_REPLAY_ROWS: int | None = None
_REPLAY_MODE: str | None = None
# Test hook (ATOM_V41_RADIX_TEST_SIDE_STEPS): this many steps that would fuse
# their replay take the side-forward fallback instead. 0 outside tests.
_TEST_SIDE_STEPS = 0
# Replays run by this process (logged, and read by the debug path).
_STATS = {
    "side_forwards": 0,
    "fused_steps": 0,
    "fallbacks": 0,
    "forced_fallbacks": 0,
    "hits": 0,
    "rows": 0,
    "steps": 0,
    "refused_requests": 0,
}


class ReplayHit(NamedTuple):
    """One batch row whose STATE slot is rebuilt below its prefix hit."""

    index: int  # row in the main batch
    slot: int  # req_pool_idx == STATE slot
    start: int  # first replayed position, P - R (floored at 0)
    end: int  # P, where the main extend resumes


def radix_replay_rows() -> int | None:
    """``R`` while the radix replay is installed in this process, else None."""
    return _REPLAY_ROWS


def radix_replay_mode() -> str | None:
    """``fused`` or ``side`` while the radix replay is installed, else None."""
    return _REPLAY_MODE


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


def resolve_replay_mode() -> str:
    """``ATOM_V41_RADIX_REPLAY_MODE``: ``fused`` (default) or ``side``."""
    from atom.plugin.sglang import deepseek_v41_envs as envs

    mode = str(envs.ATOM_V41_RADIX_REPLAY_MODE)
    if mode not in REPLAY_MODES:
        raise ValueError(
            f"{REPLAY_MODE_ENV} must be one of {REPLAY_MODES}; got {mode!r}"
        )
    return mode


# ---------------------------------------------------------------------------
# Launch policy (SGLang server-args resolution)
# ---------------------------------------------------------------------------
# Radix is on whenever SGLang's radix cache is. Where the replay cannot run
# (speculative decoding: the DSpark draft rings and the verify lifecycle are
# not proved under a replay yet) or the operator turned it off
# (ATOM_V41_RADIX=0), SGLang's radix cache is turned off with it, through
# SGLang's own model-override registry: the declaration is resolved once in
# the launching process, before the tokenizer manager and the schedulers are
# spawned, so every reader of ``disable_radix_cache`` (the tokenizer's
# input_embeds check, the scheduler's tree choice, the plugin's
# ``sglang_radix_cache_enabled``) sees the same value. Flipping it later in
# the scheduler would leave the tokenizer manager on the other side.

# SGLang serves V4.1 under V4's architecture name
# (``configs/deepseek_v41.normalize_deepseek_v41_config``); both are keyed,
# the model type decides.
_POLICY_ARCHES = ("DeepseekV4ForCausalLM", "DeepseekV41ForCausalLM")
_V41_MODEL_TYPES = ("deepseek_v41", "deepseek_v41_text")
_POLICY_REGISTERED = False
_POLICY_WARNED = False


def speculation_name(value) -> str:
    """SGLang's speculative algorithm (enum, string or None) as an upper-case
    name, "" when speculation is off."""
    if value is None:
        return ""
    name = str(getattr(value, "name", value)).strip().upper()
    return "" if name in ("", "NONE", "NULL", "FALSE") else name


def radix_off_reason(speculative_algorithm) -> str | None:
    """Why the V4.1 radix path cannot serve a launch whose SGLang radix cache
    is on, or None when it can."""
    import os

    from atom.plugin.sglang import deepseek_v41_envs as envs

    algorithm = speculation_name(speculative_algorithm)
    if algorithm:
        pinned = os.environ.get("ATOM_V41_RADIX", "").strip()
        return (
            f"speculative decoding ({algorithm}) is not supported with the "
            "DeepSeek-V4.1 radix cache yet"
            + (f"; ATOM_V41_RADIX={pinned} is ignored" if pinned else "")
        )
    if not envs.ATOM_V41_RADIX:
        return "ATOM_V41_RADIX is off"
    return None


def _resolving_view(server_args):
    from sglang.srt.arg_groups.model_override_base import resolving_view

    return resolving_view(server_args)


def v41_radix_launch_override(server_args, hf_config) -> dict:
    """SGLang model-override provider: ``disable_radix_cache=True`` for a
    DeepSeek-V4.1 launch the radix path cannot serve (``radix_off_reason``),
    with one warning; nothing otherwise, and nothing for other models."""
    global _POLICY_WARNED
    if getattr(hf_config, "model_type", None) not in _V41_MODEL_TYPES:
        return {}
    cfg = _resolving_view(server_args)
    if getattr(cfg, "disable_radix_cache", False):
        return {}
    reason = radix_off_reason(getattr(cfg, "speculative_algorithm", None))
    if reason is None:
        return {}
    if not _POLICY_WARNED:
        _POLICY_WARNED = True
        logger.warning(
            "ATOM DeepSeek-V4.1: %s; turning SGLang's radix cache off for this "
            "launch (same as --disable-radix-cache). Pass --disable-radix-cache "
            "to silence this.",
            reason,
        )
    return {"disable_radix_cache": True}


def register_v41_radix_launch_policy() -> bool:
    """Register ``v41_radix_launch_override`` with SGLang's model-override
    registry. Runs from the plugin entry point, before SGLang resolves its
    server args; idempotent. Without the registry (older SGLang) the launch
    refusals in ``deepseek_v41_bridge`` stand in: they name
    ``--disable-radix-cache``."""
    global _POLICY_REGISTERED
    if _POLICY_REGISTERED:
        return True
    try:
        from sglang.srt.arg_groups.model_override_base import register_model_override
    except ImportError:
        logger.info(
            "SGLang has no model-override registry; DeepSeek-V4.1 launches the "
            "radix path cannot serve are refused instead"
        )
        return False
    for arch in _POLICY_ARCHES:
        register_model_override(arch)(v41_radix_launch_override)
    _POLICY_REGISTERED = True
    return True


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
# Admission: token-only requests
# ---------------------------------------------------------------------------


def _non_token_inputs(req) -> list[str]:
    return [
        name for name, attr in _NON_TOKEN_INPUTS if getattr(req, attr, None) is not None
    ]


def radix_request_refusal(req) -> str | None:
    """Why ``req`` cannot be admitted while the replay is installed, else None.

    Its rows must be determined by its token ids: the tree matches token ids
    (an ``input_embeds`` request's are SGLang's placeholders), the replay
    below a hit recomputes rows from them, and a batch carrying embeddings
    hands the model batch-wide ``input_embeds`` that a fused widening cannot
    extend. SGLang refuses the same requests under its own encoder replay."""
    if _REPLAY_ROWS is None:
        return None
    carried = _non_token_inputs(req)
    if not carried:
        return None
    return (
        "DeepSeek-V4.1 with radix cache serves token-only requests; this "
        f"request carries {', '.join(carried)} (launch with "
        "--disable-radix-cache to serve it)"
    )


def reject_request(scheduler, req, error: str) -> None:
    """Answer ``req`` with a 400 without queueing it or running the model, as
    SGLang's ``Scheduler._reject_sampling_mask_request`` does."""
    from http import HTTPStatus

    from sglang.srt.disaggregation.utils import prepare_abort

    logger.error("%s, rid=%s", error, getattr(req, "rid", None))
    trace = getattr(getattr(req, "time_stats", None), "trace_ctx", None)
    if trace is not None:
        trace.abort(abort_info={"reason": error})
    if getattr(req, "multimodal_inputs", None) is not None:
        req.release_mm_inputs_on_abort()
    prepare_abort(req, error, status_code=HTTPStatus.BAD_REQUEST)
    scheduler.output_streamer.stream_output([req], req.return_logprob)
    _STATS["refused_requests"] += 1


def admit_or_reject(scheduler, req, is_retracted: bool) -> bool:
    """``Scheduler._add_request_to_queue``'s addition: False (``req``
    answered with a 400) for a new request ``radix_request_refusal``
    refuses. A retracted request re-enters with what it already holds and is
    left to SGLang (none of these requests is ever admitted, so none is ever
    retracted)."""
    if is_retracted or getattr(req, "is_retracted", False):
        return True
    error = radix_request_refusal(req)
    if error is None:
        return True
    reject_request(scheduler, req, error)
    return False


# ---------------------------------------------------------------------------
# Token ids below a hit
# ---------------------------------------------------------------------------


def _check_token_only(req) -> None:
    """Admission refuses these (``admit_or_reject``); a request that got past
    it anyway must not have placeholder ids recomputed as its rows."""
    carried = _non_token_inputs(req)
    if carried:
        raise ValueError(
            "DeepSeek-V4.1 radix replay below a hit recomputes rows from token "
            f"ids; request {getattr(req, 'rid', None)} carries "
            f"{', '.join(carried)} (admission should have refused it)"
        )


def _check_no_cached_prompt_logprobs(req, hit: ReplayHit) -> None:
    """SGLang caps a match at ``logprob_start_len``, so no prompt row whose
    logprob is wanted is ever below a hit; the replay computes none."""
    start = getattr(req, "logprob_start_len", -1)
    if getattr(req, "return_logprob", False) and 0 <= int(start) < hit.end:
        raise ValueError(
            "DeepSeek-V4.1 radix replay cannot return cached prompt logprobs "
            f"(logprob_start_len {start} < hit {hit.end})"
        )


def _replay_tokens(req, hit: ReplayHit, history_tokens: int):
    """The ids of ``[start, P)`` and the left-padded Engram history before
    ``start``, from the request's own token record -- the one SGLang builds
    the step's ``input_ids`` (and its own encoder replay) from."""
    _check_token_only(req)
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


# ---------------------------------------------------------------------------
# Fused: the rows below each hit ride in the step's own forward
# ---------------------------------------------------------------------------


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


def _host_extend_lengths(forward_batch, bs: int) -> np.ndarray:
    lengths = getattr(forward_batch, "extend_seq_lens_cpu", None)
    if lengths is None:
        lengths = forward_batch.extend_seq_lens
    if torch.is_tensor(lengths):
        lengths = lengths[:bs].detach().cpu().numpy()
    return np.asarray(lengths[:bs], dtype=np.int64)


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
        # Admission refuses embedding requests while radix is on
        # (`admit_or_reject`); a batch carrying one cannot be widened.
        raise NotImplementedError(
            "DeepSeek-V4.1 fused radix replay needs token input_ids, not "
            "input_embeds (admission should have refused the request)"
        )
    if positions is None or positions.ndim != 1:
        raise NotImplementedError(
            "DeepSeek-V4.1 fused radix replay needs 1-D positions"
        )
    bs = int(forward_batch.batch_size)
    lengths = _host_extend_lengths(forward_batch, bs)
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


# ---------------------------------------------------------------------------
# Side: one separate forward over the rows below every hit
# ---------------------------------------------------------------------------


def _host_slots(batch) -> list[int]:
    slots = getattr(batch, "req_pool_indices_cpu", None)
    if slots is None:
        slots = batch.req_pool_indices
    if torch.is_tensor(slots):
        slots = slots.detach().cpu().tolist()
    return [int(slot) for slot in slots]


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
        tokens, history = _replay_tokens(req, hit, history_tokens)
        reqs.append(req)
        ids.extend(tokens)
        histories.append(history)
    starts = [hit.start for hit in hits]
    ends = [hit.end for hit in hits]
    lengths = [end - start for start, end in zip(starts, ends)]
    slots = [hit.slot for hit in hits]

    side = copy(batch)
    setattr(side, FUSED_ATTR, None)
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


def _debug_enabled() -> bool:
    from atom.plugin.sglang import deepseek_v41_envs as envs

    return bool(envs.ATOM_V41_RADIX_DEBUG)


class _PageCheck:
    """Debug: the hit pages' digests before a forward, compared after it."""

    def __init__(self, builder, req_to_token, hits, label, agreement):
        self.builder, self.label, self.agreement = builder, label, agreement
        self.spans = [(hit.slot, hit.start, hit.end) for hit in hits]
        self.rows = sum(int(hit.end - hit.start) for hit in hits)
        self.pages = _hit_pages(req_to_token, hits, int(builder.block_size))
        self.before = page_digests(builder.cache, self.pages)

    def report(self) -> None:
        after = page_digests(self.builder.cache, self.pages)
        changed = int((after != self.before).any(dim=1).sum())
        log = logger.error if changed else logger.info
        log(
            "V4.1 radix debug: %s hits=%d rows=%d spans=%s pages=%d "
            "pages_changed=%d tp_agree=%s",
            self.label,
            len(self.spans),
            self.rows,
            self.spans,
            int(self.pages.numel()),
            changed,
            self.agreement,
        )


# ---------------------------------------------------------------------------
# The seam
# ---------------------------------------------------------------------------


def _fits(batch, hits, builder) -> bool:
    """Whether the step widened by its replay rows fits the builder."""
    extend = getattr(batch, "extend_num_tokens", None)
    if extend is None:
        extend = sum(int(n) for n in batch.extend_lens)
    widened = int(extend) + sum(int(hit.end - hit.start) for hit in hits)
    return widened <= int(builder.max_num_batched_tokens)


def run_v41_prefix_replay(worker, batch, debug_checks=None):
    """Plan the rows below every prefix hit of this step's batch and return
    the ``FusedReplay`` the step's forward must carry -- or run the side
    forward(s) and return None. Every row's slot is stamped for its tenant
    here except a fused hit's, which the bridge stamps when the forward
    consumes the record (``stamp_fused_hits``).

    ``debug_checks``: a list a fused step's page checks are appended to, for
    the caller to report after the forward (``ATOM_V41_RADIX_DEBUG``).
    """
    rows = _REPLAY_ROWS
    if rows is None or batch is None:
        return None
    mode = batch.forward_mode
    if not mode.is_extend_without_speculative():
        return None
    runner = worker.model_runner
    pool = runner.token_to_kv_pool
    if not getattr(pool, "is_atom_v41_proxy_pool", False):
        return None
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
    debug = _debug_enabled()
    agreement = None
    if debug:
        # Every rank, every EXTEND step, hits or not: a rank that disagrees on
        # whether the step has a hit must be reported, not hang the others.
        agreement = _tp_agreement(sum(int(hit.end - hit.start) for hit in hits))
        if agreement not in ("yes", "single"):
            logger.error(
                "V4.1 radix debug: TP ranks disagree on replay rows: %s", agreement
            )
    fused = None
    history = int(builder.geometry.history_size)
    fuse = bool(hits) and _REPLAY_MODE == "fused" and _fits(batch, hits, builder)
    forced = fuse and _STATS["forced_fallbacks"] < _TEST_SIDE_STEPS
    if forced:
        fuse = False
        _STATS["forced_fallbacks"] += 1
        logger.warning(
            "V4.1 radix replay: test hook ATOM_V41_RADIX_TEST_SIDE_STEPS=%d sends "
            "fused step %d (%d hits) to the side-forward fallback",
            _TEST_SIDE_STEPS,
            _STATS["forced_fallbacks"],
            len(hits),
        )
    if fuse:
        fused = build_fused_replay(batch, hits, history)
        _STATS["fused_steps"] += 1
        _STATS["hits"] += len(hits)
        _STATS["rows"] += fused.num_rows
        if debug and debug_checks is not None:
            debug_checks.append(
                _PageCheck(
                    builder,
                    req_to_token_pool.req_to_token,
                    hits,
                    f"fused_step={_STATS['fused_steps']}",
                    agreement,
                )
            )
        elif _STATS["fused_steps"] == 1:
            logger.info(
                "V4.1 radix replay: first fused step, %d hits / %d rows (R=%d)",
                len(hits),
                fused.num_rows,
                rows,
            )
    elif hits:
        if _REPLAY_MODE == "fused":
            _STATS["fallbacks"] += 1
            if _STATS["fallbacks"] == 1 and not forced:
                logger.warning(
                    "V4.1 radix replay: a step widened by its %d replay rows "
                    "exceeds the builder's %d tokens; replaying it in a side "
                    "forward (logged once)",
                    sum(int(hit.end - hit.start) for hit in hits),
                    int(builder.max_num_batched_tokens),
                )
        for group in split_replay_groups(
            hits, int(builder.max_bs), int(builder.max_num_batched_tokens)
        ):
            side = build_side_batch(batch, group, runner, history)
            check = None
            if debug:
                check = _PageCheck(
                    builder,
                    req_to_token_pool.req_to_token,
                    group,
                    f"side_forward={_STATS['side_forwards'] + 1}",
                    agreement,
                )
            _run_side_forward(runner, side)
            _STATS["side_forwards"] += 1
            _STATS["hits"] += len(group)
            _STATS["rows"] += int(side.extend_num_tokens)
            if check is not None:
                check.report()
            elif _STATS["side_forwards"] == 1:
                logger.info(
                    "V4.1 radix replay: first side forward, %d hits / %d rows "
                    "(R=%d)",
                    len(group),
                    int(side.extend_num_tokens),
                    rows,
                )
    # Every row of the step, P = 0 and continuations included, now belongs
    # to its current tenant: it needs no rebuild, or its side forward has
    # been issued. A fused hit is stamped only when its forward consumes the
    # record (`stamp_fused_hits`), so a forward that lost it fails the
    # bridge's guard before any kernel runs.
    stamped = np.ones(len(slots), dtype=bool)
    if fused is not None:
        stamped[fused.rows] = False
    pool.state_gen[np.asarray(slots, dtype=np.int64)[stamped]] = np.asarray(
        generations, dtype=np.int64
    )[stamped]
    _STATS["steps"] += 1
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


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------


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
        if batch is None:
            return original(self, batch, *args, **kwargs)
        checks = []
        fused = run_v41_prefix_replay(self, batch, checks)
        if fused is None:
            return original(self, batch, *args, **kwargs)
        result = run_with_fused_replay(original, self, batch, fused, args, kwargs)
        for check in checks:
            check.report()
        return result

    forward_batch_generation._atom_v41_radix_patched = True
    TpModelWorker.forward_batch_generation = forward_batch_generation


def carry_fused_replay(batch, forward_batch) -> None:
    """``init_new``'s addition: a fused replay planned for ``batch`` rides on
    the ForwardBatch built from it, an EXTEND, and nothing else."""
    fused = getattr(batch, FUSED_ATTR, None)
    if fused is None:
        return
    mode = forward_batch.forward_mode
    if getattr(mode, "name", None) != "EXTEND":
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
    in-place rewrite of a live row (sessions, unified memory and storage
    backends are refused with radix). A decode staged before it read the
    old pages, which nothing reuses before that decode's forward has run
    (reuse is a later step's forward on the same stream); every decode
    staged after it reads the row again (``HostPageRows.lookup`` misses).
    """
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    from atom.plugin.sglang import deepseek_v41_bridge as bridge

    original = UnifiedRadixCache.checkpoint_into_tree
    if not getattr(original, "_atom_v41_radix_patched", False):

        @wraps(original)
        def checkpoint_into_tree(self, req, *args, **kwargs):
            slot = getattr(getattr(req, "kv", None), "req_pool_idx", None)
            try:
                return original(self, req, *args, **kwargs)
            finally:
                if slot is not None:
                    bridge.forget_host_page_rows(slot)

        checkpoint_into_tree._atom_v41_radix_patched = True
        UnifiedRadixCache.checkpoint_into_tree = checkpoint_into_tree
    bridge.mark_host_page_rows_radix_safe()


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


def _install_request_admission() -> None:
    """Refuse non-token requests at the scheduler's queue door
    (``admit_or_reject``): every new request -- generate, embedding, a
    grammar request once compiled -- enters the waiting queue through
    ``Scheduler._add_request_to_queue``."""
    from sglang.srt.disaggregation.utils import prepare_abort  # noqa: F401
    from sglang.srt.managers.scheduler import Scheduler

    original = Scheduler._add_request_to_queue
    if getattr(original, "_atom_v41_radix_patched", False):
        return

    @wraps(original)
    def _add_request_to_queue(self, req, is_retracted=False, *args, **kwargs):
        if not admit_or_reject(self, req, is_retracted):
            return None
        return original(self, req, is_retracted, *args, **kwargs)

    _add_request_to_queue._atom_v41_radix_patched = True
    Scheduler._add_request_to_queue = _add_request_to_queue


def install_deepseek_v41_radix(window: int | None = None) -> tuple[int, str]:
    """Turn the replay on in this process: the worker seam, the ForwardBatch
    carriers (``init_new`` and the eager runner's view), the host-row
    invalidation, the admission charge and the token-only admission.
    Idempotent; returns ``(R, mode)``. Called only with radix cache on."""
    global _REPLAY_ROWS, _REPLAY_MODE, _TEST_SIDE_STEPS
    from atom.plugin.sglang import deepseek_v41_envs as envs

    rows = resolve_replay_rows(window)
    mode = resolve_replay_mode()
    _install_worker_seam()
    _install_forward_batch_carrier()
    _install_eager_view_carrier()
    _install_checkpoint_invalidation()
    _install_admission_accounting()
    _install_request_admission()
    _REPLAY_ROWS, _REPLAY_MODE = rows, mode
    _TEST_SIDE_STEPS = max(0, int(envs.ATOM_V41_RADIX_TEST_SIDE_STEPS))
    if _TEST_SIDE_STEPS:
        logger.warning(
            "ATOM_V41_RADIX_TEST_SIDE_STEPS=%d: test hook on, the first %d fused "
            "replay steps take the side-forward fallback",
            _TEST_SIDE_STEPS,
            _TEST_SIDE_STEPS,
        )
    logger.info(
        "ATOM DeepSeek-V4.1 radix cache: replay-first, R=%d rows below each hit, %s",
        rows,
        (
            "fused into the step's forward"
            if mode == "fused"
            else "in a side forward before the step's"
        ),
    )
    return rows, mode

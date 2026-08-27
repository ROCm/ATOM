# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Scheduling logic for batching prefill and decode requests.

This module provides:

- :class:`ScheduledBatch`: A frozen snapshot of sequences selected for the
  next forward pass, together with their block tables and metadata.
- :class:`ScheduledBatchOutput`: Token-level outputs from a completed batch.
- :class:`Scheduler`: The main scheduling loop that manages *waiting* and
  *running* queues, coordinates block allocation, and integrates with the
  KV disaggregation connector for remote prefill/decode.

Every scheduler here owns an :class:`~atom.model_engine.engine_stats.EngineStats`
(``self.engine_stats``); the class itself lives in ``engine_stats.py``.
"""

from __future__ import annotations

import logging
import os
import struct
import threading
import time
from collections import deque
from collections.abc import Iterable

import numpy as np

from atom.config import Config
from atom.kv_transfer.disaggregation import KVConnectorOutput, kv_config_has_producer
from atom.model_engine.block_manager import BlockManager
from atom.model_engine.engine_stats import EngineStats
from atom.model_engine.request import RequestOutput
from atom.model_engine.sequence import (
    Sequence,
    SequenceStatus,
    SequenceType,
    new_token_ids,
)
from atom.model_engine.state_runtime import (
    DEFAULT_STATE_RUNTIME,
    StateMaintenanceOps,
    StateRuntime,
)
from atom.utils import envs

logger = logging.getLogger("atom")


# How often the stalled-save reconciler actually scans deferred_free_blocks. The
# engine polls KV progress every millisecond (KV_IDLE_DRAIN_INTERVAL_S), so the
# reconciler no-ops until this interval has passed.
_SAVE_RECONCILE_INTERVAL_S = 5.0

# Memoised result of `_offload_max_pending_saves`, a process constant likewise.
_MAX_PENDING_OFFLOAD: int | None = None


def _offload_max_pending_saves() -> int:
    """How many offload transfers may be outstanding at once, engine-side.

    Shared with the KV leg's `OFFLOAD_MAX_PENDING_SAVES` rather than given a
    knob of its own: a KV save and a state store both hold the bytes they are
    reading out of the same pool while they run, so the queue depth is one
    question about one resource. Two independent numbers would let an operator
    raise one and unknowingly double what a slow backend can pin.
    """
    global _MAX_PENDING_OFFLOAD
    if _MAX_PENDING_OFFLOAD is not None:
        return _MAX_PENDING_OFFLOAD
    raw = os.environ.get("OFFLOAD_MAX_PENDING_SAVES", "2")
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "invalid OFFLOAD_MAX_PENDING_SAVES=%r; using 2 for the state tier",
            raw,
        )
        value = 2
    _MAX_PENDING_OFFLOAD = max(1, value)
    return _MAX_PENDING_OFFLOAD


def _prompt_tokens_of(result) -> int:
    """Prompt tokens in whatever a `_schedule()` returned.

    The three schedulers disagree on the empty shape — bare `None` from
    `Scheduler` and `DecodeScheduler`, `(None, {})` from `PrefillScheduler` —
    so normalize here rather than at each caller. `total_tokens_num_prefill`
    is 0 on a decode batch, which is what the throughput window wants: its
    generation side is counted in `postprocess`, from tokens actually
    committed.
    """
    batch = result[0] if isinstance(result, tuple) else result
    return 0 if batch is None else batch.total_tokens_num_prefill


def _optimal_cu_fraction(
    decode_batch: int, prefill_waiting_tokens: int
) -> float | None:
    """Return the prefill CU fraction for the current workload, or None for no mask.

    Called by the DecodeScheduler, which has visibility into both the decode
    batch size and the total tokens queued in prefill_waiting.  The chosen
    fraction is written to shared memory so the PrefillScheduler can read it.

    Lookup table derived from empirical benchmarking across CU splits
    (30/50/60/70/80% prefill) on DeepSeek-R1 tp=8.  Prefill latency
    dominates in nearly all cases; decode tolerates CU reduction well
    at typical batch sizes.

    Returns None when CU masking provides no benefit (no pending prefill,
    tiny prefill).
    """
    assert prefill_waiting_tokens >= 0
    if prefill_waiting_tokens == 0 or decode_batch < 64:
        return None
    else:
        return 0.5


class ScheduledBatch:
    """Immutable snapshot of sequences selected for a single forward pass.

    Holds per-sequence metadata (block tables, context lengths, temperatures)
    and the flattened token array ready for the model runner.

    Args:
        seqs: Mapping from request ID to :class:`Sequence`.
        num_scheduled_tokens: Number of new tokens per sequence.
        total_tokens_num: Sum of all scheduled tokens (prefill + decode).
        connector_meta_output: Optional KV connector metadata for this batch.
        num_spec_step: Number of speculative decode steps (0 = disabled).
        scheduled_spec_decode_tokens: Draft token IDs per request for
            speculative decoding (must not use a mutable default).
        state_maintenance_ops: State moves that must execute before this batch.
    """

    def __init__(
        self,
        seqs: dict[int, Sequence],
        num_scheduled_tokens: list[int],
        total_tokens_num: int,
        total_tokens_num_prefill: int = 0,
        total_tokens_num_decode: int = 0,
        total_seqs_num: int = 0,
        total_seqs_num_prefill: int = 0,
        total_seqs_num_decode: int = 0,
        connector_meta_output=None,
        is_dummy_run: bool = False,
        num_spec_step: int = 0,
        scheduled_spec_decode_tokens: dict[int, np.ndarray] | None = None,
        cu_stream_fraction: float | None = None,
        remote_kv_block_ids: list[int] | None = None,
        remote_kv_seq_blocks: dict[int, list[int]] | None = None,
        num_cached_tokens: list[int] | None = None,
        is_final_chunk: list[bool] | None = None,
        next_token_ids: list[int] | None = None,
        state_maintenance_ops: StateMaintenanceOps | None = None,
    ):
        if scheduled_spec_decode_tokens is None:
            scheduled_spec_decode_tokens = {}
        self.remote_kv_block_ids = remote_kv_block_ids or []
        self.remote_kv_seq_blocks = remote_kv_seq_blocks or {}

        self.req_ids = list(seqs.keys())
        self.num_scheduled_tokens = np.asarray(num_scheduled_tokens, dtype=np.int32)
        self.temperatures = np.asarray(
            [seq.temperature for seq in seqs.values()], dtype=np.float32
        )
        self.return_logprobs = [seq.return_logprobs for seq in seqs.values()]
        # `context_lens` is set further down, once `num_cached_tokens` is known:
        # a chunked prefill's context ends at this chunk, not at the whole
        # prompt, so `seq.num_tokens` is only right for decode.
        # Kept as a list too: the offset loop below reads it per sequence, and
        # `arr[i]` builds a numpy scalar where a list index does not. The list
        # is what `np.asarray` consumes anyway, so holding onto it is free.
        num_rejected = [seq.num_rejected for seq in seqs.values()]
        self.num_rejected = np.asarray(num_rejected, dtype=np.int32)
        self.num_bonus = np.asarray(
            [seq.num_bonus_tokens for seq in seqs.values()], dtype=np.int32
        )
        # One entry per state-holding seq: that seq's whole slot set, in
        # allocation order. `[0]` is the committed state and `[1:]` is
        # speculation rollback, one slot per speculated token. The sets are not
        # adjacent and no backend may reconstruct them by arithmetic on a base
        # — see `StateSlotPool`.
        # Gated on `state_slot >= 0`, not on the list being non-empty: a seq
        # whose committed slot was never claimed carries the -1 sentinel in a
        # one-element list, which is truthy, and letting it through would shift
        # every list positionally aligned with this one.
        state_seqs = [
            seq
            for seq in seqs.values()
            if seq.has_per_req_cache and seq.state_slot >= 0
        ]
        self.state_slots = [seq.state_slots for seq in state_seqs]
        # Column 0 broken out, because it is what every non-speculative backend
        # wants and rebuilding it per step in each of them would cost the same
        # Python loop several times over.
        self.state_slots_committed = [seq.state_slot for seq in state_seqs]
        # Read-side twin of the committed column, positionally aligned with it:
        # the slot this forward takes its incoming state from. Differs only on
        # the one forward after a state fork; -1 elsewhere, which attention
        # backends read as "same as the write slot".
        self.state_fork_srcs = [seq.state_fork_src for seq in state_seqs]
        # Midstep checkpoints this forward must write, `[(slot, position)]` per
        # seq, positionally aligned with `state_slots` like the fork sources
        # above. A list per seq, not one entry: a readable backend takes every
        # position the chunk covers rather than only the one it ends on.
        # Positions are absolute prompt offsets; the backend rebases them onto
        # the step's own tokens, which is the only frame its intermediates are
        # in. Empty everywhere except a `readable_midstep` prefill.
        self.state_save_all = [
            [(g, p) for g, p, _h in seq.midstep_reservations] for seq in state_seqs
        ]
        # Physical moves are drained once per real batch.
        self.state_maintenance_ops = (
            state_maintenance_ops
            if state_maintenance_ops is not None
            else StateMaintenanceOps()
        )
        self.top_ks = np.asarray([seq.top_k for seq in seqs.values()], dtype=np.int32)
        self.top_ps = np.asarray([seq.top_p for seq in seqs.values()], dtype=np.float32)
        # True if any seq in the batch is a fan-out child (SamplingParams.n>1)
        # and therefore requires fresh per-row random noise at the sampler
        # rather than the cached shared exponential tensor.
        self.needs_independent_noise = np.asarray(
            [getattr(seq, "needs_independent_noise", False) for seq in seqs.values()],
            dtype=bool,
        )

        self.is_first_decode_without_local_prefill = [
            seq.is_first_decode for seq in seqs.values()
        ]
        self.mrope_positions_by_req = {
            seq.id: seq.mrope_positions
            for seq in seqs.values()
            if getattr(seq, "mrope_positions", None) is not None
        }
        self.mrope_position_deltas = {
            seq.id: getattr(seq, "mrope_position_delta", 0)
            for seq in seqs.values()
            if getattr(seq, "mrope_positions", None) is not None
        }

        # num_cached_tokens for chunked prefill support
        self.num_cached_tokens = (
            num_cached_tokens
            if num_cached_tokens is not None
            else [seq.num_cached_tokens for seq in seqs.values()]
        )

        self.is_final_chunk = is_final_chunk
        # Per seq, the token following this forward where the scheduler knows it
        # (a middle prefill chunk's successor prompt token), -1 where sampling
        # supplies it. A drafter runs one position ahead of the target, so this
        # is its anchor on the chunks that never reach sampling.
        self.next_token_ids = next_token_ids

        # context_lens: for prefill seqs, use num_cached_tokens + num_scheduled_tokens
        self.context_lens = np.asarray(
            [
                (
                    self.num_cached_tokens[i] + num_scheduled_tokens[i]
                    if seq.type == SequenceType.PREFILL
                    else seq.num_tokens
                )
                for i, seq in enumerate(seqs.values())
            ],
            dtype=np.int32,
        )

        # Each sequence's window, staged into one array rather than assigned
        # per sequence: a numpy slice-assign costs ~245ns of dispatch whatever
        # its length, and at decode a window is a single token. `extend`
        # between two `array("i")` is a memcpy. 4x at bs=256.
        staged = new_token_ids()
        for i, (seq, num) in enumerate(zip(seqs.values(), num_scheduled_tokens)):
            if seq.type == SequenceType.PREFILL:
                offset = self.num_cached_tokens[i]
            else:
                offset = seq.num_tokens - num_rejected[i] - num
            staged.extend(seq.token_ids[offset : offset + num])
        # Checked here because nothing downstream will: the array below wraps
        # whatever length was staged, and a sequence too short to fill its
        # window would otherwise leave the batch quietly short.
        if len(staged) != total_tokens_num:
            raise ValueError(
                f"staged {len(staged)} tokens for a batch of {total_tokens_num}: "
                "a sequence is shorter than the window scheduled for it"
            )
        # Wrapped, not copied into a fresh array: consumers only ever read this
        # or rebind it, so the staging buffer can be the buffer.
        self.scheduled_tokens = np.frombuffer(staged, dtype=np.int32)

        if num_spec_step > 0 and scheduled_spec_decode_tokens is not None:
            # One row per sequence, in batch order. The caller's dict is keyed
            # by request id and holds only sequences that DO have drafts, so
            # densifying it with `list(...values())` drops the others and
            # shifts every later row onto the wrong sequence — consumers index
            # this by batch position (`prepare_input_ids`), which turns that
            # shift into one sequence silently reading another's drafts.
            #
            # Rows are collected and written in one assign: a per-row write
            # costs ~245ns of dispatch against a row this short, so the marshal
            # was mostly dispatch (3.7x at bs=256). Rows that are not
            # full width are padded rather than dropping the batch back to the
            # per-row form, because at a large batch a sequence joins decode
            # nearly every step and would take every other sequence with it.
            no_drafts = np.zeros(num_spec_step, dtype=np.int32)
            rows = []
            for req_id in self.req_ids:
                drafts = scheduled_spec_decode_tokens.get(req_id)
                if drafts is not None and drafts.size == num_spec_step:
                    rows.append(drafts)
                elif drafts is None or drafts.size == 0:
                    rows.append(no_drafts)
                else:  # short or long: onto a row of its own, never the shared one
                    padded = no_drafts.copy()
                    padded[: min(drafts.size, num_spec_step)] = drafts[:num_spec_step]
                    rows.append(padded)
            # `concatenate` rejects an empty list, which a warmup batch is.
            flat = (
                np.concatenate(rows, dtype=np.int32)
                if rows
                else np.empty(0, dtype=np.int32)
            )
            self.scheduled_spec_decode_tokens = flat.reshape(len(seqs), num_spec_step)
        self.block_tables = [
            seq.block_table for seq in seqs.values() if seq.block_table
        ]
        self.last_block_num_tokens = [
            _seq.last_block_num_tokens for _seq in seqs.values()
        ]

        # Total number of tokens scheduled for all requests.
        self.total_tokens_num = total_tokens_num
        self.total_tokens_num_prefill = total_tokens_num_prefill
        self.total_tokens_num_decode = total_tokens_num_decode

        # Total number of reqs scheduled for all requests.
        self.total_seqs_num = total_seqs_num
        self.total_seqs_num_prefill = total_seqs_num_prefill
        self.total_seqs_num_decode = total_seqs_num_decode

        self.connector_meta_output = connector_meta_output
        self.finished_recving_kv_req_ids: list[int] = []

        self.is_dummy_run = is_dummy_run
        self.num_spec_step = num_spec_step
        # DSpark RAGGED (paper §5.2): per-request decode query lengths [bs]
        # (ell_r + 1). None unless _dspark_apply_ragged set it this step; when
        # set, consumers use it (per-seq) instead of the scalar above.
        self.dynamic_spec_query_tokens_per_req = None

        # Detailed attention aggregates (set by Scheduler.compute_detailed_aggregates
        # when profiling is active and ATOM_ENABLE_DETAILED_ANNOTATION is set).
        # None on the normal path; consumed by ModelRunner.run_model to extend
        # the prefill[]/decode[] trace labels.
        self.detailed_sqsq: int | None = None  # sum N_Q^2
        self.detailed_sqsk: int | None = None  # sum N_Q*N_KV
        self.detailed_sk: int | None = None  # sum N_KV

        # Key into ModelRunner's stream pool for CU-masked disagg streams.
        # None means full-CU fallback (no mask).
        self.cu_stream_fraction = cu_stream_fraction
        # Collect multimodal data from prefill sequences
        self.multimodal_data = {}
        for seq in seqs.values():
            if getattr(seq, "multimodal_data", None) is not None:
                self.multimodal_data[seq.id] = seq.multimodal_data
                # Clear after first use to avoid re-sending on decode steps
                seq.multimodal_data = None
        self.external_request_ids = [seq.external_request_id for seq in seqs.values()]

        # logger.info(f"{[el for el in scheduled_spec_decode_tokens.keys()]=}")
        # logger.info(f"{self.num_scheduled_tokens=}")
        # logger.info(f"{self.context_lens=}")
        # logger.info(f"{[len(blk)*16 for blk in self.block_tables]=}")
        # logger.info(f"{self.block_tables=}")

    def produces_output(self) -> bool:
        """True if this batch yields a token the head must consume.

        Decode batches always do. A pure-prefill batch yields a token only
        when at least one seq is on its final chunk; a batch of middle chunks
        produces nothing.

        A DP-sync dummy answers True and must: reporting lags a step, so the
        dummy is what flushes the last real one. Skip its `postprocess` and
        the request waking up after it has an anchor on neither side.
        """
        if self.total_seqs_num_decode > 0:
            return True
        if self.is_final_chunk is None:
            return True
        return any(self.is_final_chunk)


class ScheduledBatchOutput:
    """Token-level results from a single forward pass.

    Attributes:
        token_ids: Mapping of request ID -> accepted token IDs.
        draft_token_ids: Speculative draft tokens (one row per request).
        num_rejected: Per-request count of rejected speculative tokens.
        num_bonus: Per-request count of bonus accepted tokens.
        is_deferred_out: Whether output was deferred from a previous step.
    """

    def __init__(
        self,
        req_ids: list[int],
        token_ids: list[tuple[int, ...]],
        num_rejected: np.ndarray | None,
        num_bonus: np.ndarray | None,
        draft_token_ids: np.ndarray | None,
        is_deferred_out: bool = False,
        is_prev_prefill=False,
        logprobs=None,
        dspark_ell: np.ndarray | None = None,
    ):
        self.req_ids = req_ids
        self.token_ids = token_ids
        self.draft_token_ids = draft_token_ids
        self.num_rejected = num_rejected
        self.num_bonus = num_bonus
        self.is_deferred_out = is_deferred_out
        self.is_prev_prefill = is_prev_prefill
        self.logprobs = logprobs  # Optional[dict[int, float]]
        # DSpark Phase 2: {req_id: ell_r} from this step's propose() — the
        # scheduler-chosen verify length per request. Rides back to the
        # (main-process) scheduler so the NEXT step can size each request's
        # verification to ell_r+1. None when DSpark scheduling is off.
        self.dspark_ell = dspark_ell
        # O(1) lookup: req_id -> index (lazy-built on first access)
        self._req_id_to_idx: dict[int, int] | None = None

    def get_idx(self, req_id: int) -> int | None:
        """O(1) lookup of request index by id."""
        if self._req_id_to_idx is None:
            self._req_id_to_idx = {rid: i for i, rid in enumerate(self.req_ids)}
        return self._req_id_to_idx.get(req_id)


class Scheduler:
    """Manages the lifecycle of inference requests through prefill and decode.

    The scheduler maintains two primary queues:

    - **waiting**: Newly arrived requests pending their first prefill.
    - **running**: Active requests that have completed prefill and are
      being decoded token-by-token.

    On each :meth:`schedule` call it selects a batch of sequences that
    fit within the token and sequence budget, allocates KV cache blocks
    via :class:`BlockManager`, and returns a :class:`ScheduledBatch`.

    Integration with the KV disaggregation connector is handled through
    :meth:`_update_waiting_for_remote_kv` (decode side) and
    :meth:`_update_from_kv_xfer_finished` (both sides).
    """

    _ENGINE_LABEL = ""
    _METRICS_ROLE = ""

    def __init__(
        self,
        config: Config,
        *,
        state_runtime: StateRuntime = DEFAULT_STATE_RUNTIME,
    ):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.long_prefill_token_threshold = config.long_prefill_token_threshold
        self.max_model_len = config.max_model_len
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.stop_token_ids = config.stop_token_ids
        self.block_manager = BlockManager(
            config,
            state_runtime=state_runtime,
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.config = config

        # Admit-rejected seqs (those `_unschedulable_reason` flags). Drained
        # by `take_rejected` each EngineCore step; routed through the same
        # output_queue path as forward-finished seqs.
        self._rejected: list[Sequence] = []

        # KV transfer bookkeeping
        self.finished_recving_kv_req_ids: list[int] = []
        self.failed_recving_kv_req_ids: list[int] = []
        self.deferred_free_blocks: dict[int, Sequence] = {}
        # Reclamation bookkeeping for offload saves whose completion is never
        # reported (see `_reconcile_stalled_deferred_saves`).
        self._abandoned_saves: int = 0
        self._next_save_reconcile_at: float = 0.0

        # Scheduling delay for batching efficiency
        self.prev_time = 0.0
        # Did we schedule a prompt at previous step?
        self.prev_prompt = False
        # Latency of the last prompt step
        self.last_prompt_latency = 0.0
        self.delay_factor = config.scheduler_delay_factor

        # Speculative decoding. A P/D producer is excluded for the same reason
        # `PrefillScheduler` is: it prefills, samples T0 and hands the request
        # off, so it never proposes. ModelRunner disables deferred output on a
        # producer (the consumer must consume T0 exactly once, or KDA state
        # advances twice), and `propose()` only runs on the deferred path.
        # Leaving speculation on here would still pad every sequence with
        # `mtp_k` placeholders and size its decode window to `mtp_k + 1`, and
        # the padding holds `num_tokens` `mtp_k` short of `max_tokens` so the
        # handoff never fires -- the producer keeps decoding a request whose
        # drafts do not exist. The drafter itself stays loaded: the producer
        # still writes the draft's context KV into the shared pool at prefill.
        is_pd_producer = kv_config_has_producer(
            getattr(config, "kv_transfer_config", {}) or {}
        )
        self.use_spec = config.speculative_config is not None and not is_pd_producer
        self.mtp_k: int = (
            config.speculative_config.num_speculative_tokens if self.use_spec else 0
        )  # type: ignore
        # EAGLE/MTP needs the successor token; DSpark does not.
        self.drafter_needs_next_token = self.use_spec and not (
            config.speculative_config.use_dspark()
        )
        # True when this engine both drafts and verifies; False under PP
        # (which only drafts for handoff to the decode node).
        pp_size = getattr(config, "pipeline_parallel_size", 1)
        self.spec_decode_local = self.use_spec and pp_size == 1
        if self.use_spec and not self.spec_decode_local:
            logger.info(
                "Speculative decoding: drafting only (pipeline_parallel_size=%d). "
                "Drafts are produced for handoff; this engine verifies none.",
                pp_size,
            )
        # `engine_stats` (spec / cache / throughput sections) is constructed
        # below, once `dp_rank` — the throughput section's engine index — is
        # known.
        # Dashboard counters update only at request lifecycle boundaries.
        self.total_prompt_tokens = 0
        self.total_generation_tokens = 0
        self.total_finished_requests = 0
        self.total_preemptions = 0
        self.profile_active = False
        # Cache the env flag once (env vars are fixed at process start) so the
        # per-iteration compute_detailed_aggregates never pays an os.getenv.
        self._detailed_annotation_enabled = envs.ATOM_ENABLE_DETAILED_ANNOTATION

        self.enable_chunked_prefill = config.enable_chunked_prefill
        # Running seqs currently mid-prefill; counter lets schedule() skip the
        # running-queue scan on pure-decode steps.
        self._partial_prefill_count: int = 0
        self._schedule_tick: int = 0

        self._num_parked_remote_kv: int = 0

        # Under PP the head keeps pp_size batches in flight, so schedule() must
        # advance chunked-prefill progress itself rather than defer it to
        # postprocess, else back-to-back schedules re-issue the same chunk.
        self.advance_on_schedule: bool = (
            getattr(config, "pipeline_parallel_size", 1) > 1
        )
        # Seq ids whose sampled token is in flight; the decode scheduler skips
        # them until the head releases the id after postprocess, so a seq is
        # never decoded against a token not yet appended.
        self._pp_inflight_token_block: set[int] = set()

        from atom.utils.forward_context import get_kvconnector

        self.kv_connector = get_kvconnector("scheduler", config)

        from atom.distributed.kv_events import (
            EventPublisher as _EventPublisher,
        )
        from atom.distributed.kv_events import (
            make_publisher as _make_publisher,
        )

        kv_events_cfg = getattr(config, "kv_events_config", None)
        parallel_cfg = getattr(config, "parallel_config", None)
        dp_rank = (
            getattr(parallel_cfg, "data_parallel_rank", None)
            if parallel_cfg is not None
            else None
        )
        self.engine_stats = EngineStats(
            engine_index=dp_rank or 0,
            label=self._ENGINE_LABEL,
            use_spec=self.use_spec,
            mtp_k=self.mtp_k,
            enable_prefix_caching=config.enable_prefix_caching,
            enable_log_stats=config.enable_log_stats,
            throughput_log_interval_s=config.throughput_log_interval,
            cache_hit_rate_window=config.cache_hit_rate_window,
            pool_pressure=self.block_manager.pool_pressure,
        )
        if config.enable_prefix_caching:
            self.engine_stats.block_manager = self.block_manager
        if kv_events_cfg is not None and kv_events_cfg.enable:
            self.kv_event_publisher: _EventPublisher = _make_publisher(
                enabled=True,
                publisher_kind=kv_events_cfg.publisher,
                endpoint=kv_events_cfg.endpoint,
                topic=kv_events_cfg.topic,
                hwm=kv_events_cfg.hwm,
                buffer_steps=kv_events_cfg.buffer_steps,
                data_parallel_rank=dp_rank,
            )
            logger.info(
                "KV event publisher enabled: kind=%s endpoint=%s dp_rank=%s",
                kv_events_cfg.publisher,
                kv_events_cfg.endpoint,
                dp_rank,
            )
        else:
            self.kv_event_publisher = _make_publisher(
                enabled=False,
                publisher_kind="null",
                endpoint="",
            )

        # Cross-DP prefill alignment. Set by DPEngineCoreProc after
        # dp_group is available. See `prefill_delayer.py` for rationale.
        from atom.model_engine.prefill_delayer import PrefillDelayer

        self.prefill_delayer: PrefillDelayer | None = None

    def set_prefill_delayer(self, delayer) -> None:
        self.prefill_delayer = delayer

    def _can_admit_head_prefill(self) -> bool:
        """Match SGL's `local_prefillable=True` semantics: report True iff
        this rank would *actually* admit a new prefill this tick.

        Just having `self.waiting` non-empty is too coarse — during a
        concurrent-burst workload (e.g. 1k/1k @ high concurrency) every
        DP rank has a full waiting queue, so `bool(self.waiting)` is
        ALWAYS True on all ranks → status="all" → delayer never engages.
        But only the 1-2 ranks with free KV blocks actually admit a
        prefill that tick; the other 6-7 ranks decode. That's the real
        "mixed" we need to delay.

        We peek the front of `waiting` (skipping a few unschedulable
        entries) and check `can_allocate` + token-budget, mirroring the
        same checks the admission while-loop runs below.
        """
        if self._partial_prefill_count > 0:
            return True
        if not self.waiting:
            return False
        for i, seq in enumerate(self.waiting):
            if i >= 4:
                break
            if self._unschedulable_reason(seq) is not None:
                continue
            if seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS:
                continue
            num_new_tokens = seq.num_tokens - seq.num_cached_tokens
            if (
                not self.enable_chunked_prefill
                and num_new_tokens > self.max_num_batched_tokens
            ):
                continue
            # KV-pressured requests definitely cannot prefill. A fit probe, not
            # an admission -- `record=False` so it does not move the joint-
            # boundary funnel counters for a seq it may never schedule.
            return self.block_manager.can_allocate(seq, record=False) >= 0
        return False

    def _kv_usage(self) -> float:
        """Fraction of KV-cache blocks currently in use ∈ [0, 1].

        The `kv_usage` signal for PrefillDelayer's KV watermark bounds — both the
        high watermark (near-full → release, can't accumulate more) and the
        optional low watermark (GPU starving → release, feed it). Derived from
        BlockManager bookkeeping; cheap (no traversal of seq tables).
        """
        bm = self.block_manager
        total = bm.kv.num_blocks
        if total <= 0:
            return 0.0
        return bm.kv.num_used / total

    def _record_throughput(
        self, num_prompt_tokens: int = 0, num_generation_tokens: int = 0
    ) -> None:
        """Feed this tick's token counts into the throughput section of
        `engine_stats` and emit the periodic engine-status log line once
        `throughput_log_interval_s` has elapsed.

        No-op when `--no-enable-log-stats` disabled the section.

        The token counts must be accumulated on every call — they are what the
        line reports — but the three arguments below it are read fresh and then
        discarded on all but one call in `interval / step_time`, which at a 10s
        interval is about one in ten thousand. `window_expired` is a subtract
        and a compare, so gating on it keeps the per-step cost to the counter
        update.
        """
        stats = self.engine_stats
        if not stats.throughput_enabled:
            return
        stats.update_throughput(num_prompt_tokens, num_generation_tokens)
        if not stats.window_expired(time.monotonic()):
            return
        num_running_reqs, num_waiting_reqs = self.get_request_counts()
        stats.maybe_log_throughput(
            num_running_reqs=num_running_reqs,
            num_waiting_reqs=num_waiting_reqs,
            kv_usage=self._kv_usage(),
        )

    def heartbeat_throughput(self, now: float) -> None:
        """Close the throughput window on time while the engine sits idle."""
        if self.engine_stats.window_expired(now):
            self._record_throughput()

    def _waiting_new_token_count(self) -> int:
        """Sum of new (uncached) tokens across the ADMITTABLE waiting queue,
        saturated at `max_num_batched_tokens`.

        Feeds PrefillDelayer's coalescer fill signal (accumulate a full prefill
        batch before releasing, instead of firing many tiny fragments). The cap
        early-exits the scan: one batch's worth is all the coalescer compares
        against, so there's no point summing a deep queue.

        Skips the same non-admittable seqs as `_can_admit_head_prefill` —
        unschedulable, WAITING_FOR_REMOTE_KVS, and oversized-when-chunking-off —
        so the "queued work" signal counts only tokens this rank could actually
        prefill this step. Counting remote-KV / unschedulable tokens here would
        inflate the cross-rank aggregate and reach the fill target before a real
        batch has accumulated. The `num_cached_tokens` discount is best-effort:
        an un-admitted seq has not been probed against the prefix cache yet, so
        this is an upper bound on new tokens for cache-hit prompts.
        """
        cap = self.max_num_batched_tokens
        total = 0
        for seq in self.waiting:
            if self._unschedulable_reason(seq) is not None:
                continue
            if seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS:
                continue
            num_new_tokens = seq.num_tokens - seq.num_cached_tokens
            if (
                not self.enable_chunked_prefill
                and num_new_tokens > self.max_num_batched_tokens
            ):
                continue
            total += max(0, num_new_tokens)
            if total >= cap:
                return cap
        return total

    def _partial_prefill_remaining_tokens(self) -> int:
        """Sum of remaining (not-yet-computed) tokens across mid-chunked-prefill
        seqs in `running`, saturated at `max_num_batched_tokens`.

        Folded into the coalescer's pending-token signal so a small partial tail
        chunk does not force its own tiny prefill forward — it accumulates with
        fresh prefills instead, bounded by the coalescer's partial deadline.
        Skipped entirely in the common steady-state (no partial) via the
        `_partial_prefill_count` counter, and stops scanning once all
        `_partial_prefill_count` partials have been summed.

        This is an UPPER BOUND on what a partial can actually contribute in one
        forward: `remaining = num_tokens - num_cached_tokens` does NOT apply the
        per-step `long_prefill_token_threshold` chunk clamp that Phase-1
        scheduling uses, so when that threshold is set the coalescer fill signal
        may read slightly high. Acceptable for a coalescing heuristic; consistent
        with `_waiting_new_token_count`'s best-effort estimate.
        """
        if self._partial_prefill_count == 0:
            return 0
        cap = self.max_num_batched_tokens
        total = 0
        seen = 0
        for seq in self.running:
            if not seq.is_partial_prefill:
                continue
            total += max(0, seq.num_tokens - seq.num_cached_tokens)
            if total >= cap:
                return cap
            seen += 1
            if seen >= self._partial_prefill_count:
                break  # all partials summed; skip the rest of the decode tail
        return total

    def _oldest_waiting_prefill_age_ms(self) -> float:
        """Age in ms (since arrival) of the oldest ADMITTABLE waiting prefill,
        or 0.0 if none.

        Feeds PrefillDelayer's TTFT SLA guard: if this exceeds max_queue_ms the
        coalescer force-releases so a request never starves in the queue. Uses
        `seq.arrive_time` (wall-clock seconds, stamped at engine entry) — the
        true end-to-end wait, including backlog and coalescer holds. Skips the
        same non-admittable seqs as `_can_admit_head_prefill` (unschedulable,
        WAITING_FOR_REMOTE_KVS) so a permanently-stuck seq can't peg the guard.
        """
        oldest_arrive = None
        for seq in self.waiting:
            if self._unschedulable_reason(seq) is not None:
                continue
            if seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS:
                continue
            if oldest_arrive is None or seq.arrive_time < oldest_arrive:
                oldest_arrive = seq.arrive_time
        if oldest_arrive is None:
            return 0.0
        return max(0.0, (time.time() - oldest_arrive) * 1000.0)

    def publish_kv_events(self) -> None:
        """Drain BlockManager's event log and publish as one EventBatch. Called
        by EngineCore at the end of each scheduler step. No-op when events are
        disabled (NullEventPublisher swallows the publish call)."""
        events = self.block_manager.take_events()
        if events:
            self.kv_event_publisher.publish(events)

    def shutdown_kv_events(self) -> None:
        """Tear down the publisher background thread and ZMQ socket. Called
        by EngineCore on engine shutdown."""
        try:
            self.kv_event_publisher.shutdown()
        except Exception:
            logger.exception("KV event publisher shutdown failed")

    def is_finished(self):
        # `_rejected` must be considered too: if a batch of seqs is all
        # oversized, schedule() moves them straight from `waiting` to
        # `_rejected`, leaving both `waiting` and `running` empty. Without
        # this check, busy_loop's `is_finished()` short-circuits to True
        # before EngineCore drains `_rejected` via take_rejected(), and
        # llm.generate() blocks forever.
        return (
            not self.waiting
            and not self.running
            and not self._rejected
            and not self.deferred_free_blocks
        )

    def add(self, seq: Sequence):
        self._warn_if_unschedulable(seq)
        self.waiting.append(seq)

    def extend(self, seqs: list[Sequence]):
        for seq in seqs:
            self._warn_if_unschedulable(seq)
        self.waiting.extend(seqs)

    def _deferred_sequence(self, req_id) -> Sequence | None:
        seq = self.deferred_free_blocks.get(req_id)
        if seq is not None:
            return seq
        try:
            return self.deferred_free_blocks.get(int(req_id))
        except (TypeError, ValueError):
            return None

    def _connector_should_defer_free(self, seq: Sequence) -> bool:
        callback = getattr(self.kv_connector, "should_defer_free", None)
        return bool(callable(callback) and callback(seq))

    def _connector_release_stalled_save(self, seq: Sequence) -> None:
        """Let the connector drop a stall-escaped save this free surrenders.

        `preempt` frees blocks with `block_manager.deallocate` and no
        `request_finished`, so a K3 request whose `should_defer_free` returned
        False via its stall escape would keep its `_save_tracker` entry -- the
        base save loop could later emit a save reading those now-freed blocks.
        The connector's `should_defer_free` is a pure predicate, so this mutator
        does the cleanup exactly at the free. Guarded because not every connector
        offloads saves.
        """
        callback = getattr(self.kv_connector, "release_stalled_save", None)
        if callable(callback):
            callback(seq)

    def _connector_abandon_save(self, seq: Sequence) -> None:
        """Tell the connector to drop a save this reclaim just abandoned.

        Without it the connector's `_save_inflight` (and, on K3, the stall
        latch) keeps the reclaimed request forever: `should_defer_free` stays
        True and `has_pending_kv_work()` never goes False, so the engine
        busy-loops. Guarded because not every connector implements offload
        saves.
        """
        callback = getattr(self.kv_connector, "abandon_save", None)
        if callable(callback):
            callback(str(seq.id))

    def _maybe_release_deferred(self, seq: Sequence) -> None:
        if (
            seq.id not in self.deferred_free_blocks
            or getattr(seq, "_awaiting_aborted_load_cleanup", False)
            or self._connector_should_defer_free(seq)
        ):
            return

        callback = getattr(self.kv_connector, "request_finished", None)
        if callable(callback):
            callback(seq)
        self.deferred_free_blocks.pop(seq.id, None)
        self.block_manager.deallocate(seq)

    def _save_abandon_timeout_s(self) -> float:
        """Seconds a deferred save may sit before reclamation, or 0 to disable.

        Sourced from the offload connector (`save_abandon_timeout_s`), not
        re-derived here: the window is a function of LMCache's own pin timeout,
        which is the connector's knowledge, and keeping the derivation in one
        place is what stops the safety ordering (reclaim strictly after LMCache
        force-unpins) from silently drifting. Returns 0 when no offload connector
        is attached, which disables reclamation -- there is nothing to reclaim.
        """
        callback = getattr(self.kv_connector, "save_abandon_timeout_s", None)
        return callback() if callable(callback) else 0.0

    def _reconcile_stalled_deferred_saves(self) -> int:
        """Reclaim blocks whose offload save has stalled past the abandon timeout.

        `_maybe_release_deferred` cannot do this: the connector still reports the
        save as pending (`should_defer_free` stays True), which is exactly why
        the blocks were never released. Once the save has been deferred longer
        than `_save_abandon_timeout_s()` -- set above LMCache's pin
        timeout, so upstream has force-unpinned and released the source bytes --
        the blocks are safe to return to the pool. Without this, a single
        never-reported save keeps `has_pending_kv_work()` True forever and the
        engine busy-loops with every GPU idle.

        Mirrors the producer `finished_sending` reclaim (pop + deallocate),
        deliberately not re-invoking `request_finished`: it was already called
        when the request finished, before the block free was deferred. It does
        notify the connector via `abandon_save`, though -- freeing the blocks
        here is not enough on its own: the connector still holds the save in
        `_save_inflight` (and, on K3, in the stall latch), so `should_defer_free`
        would stay True and `has_pending_kv_work()` would never clear without
        that drop.

        The complement of the K3 connector's stall escape
        (`kimi_k3.connector.save_stall_seconds()`), not a duplicate of it: that
        one releases the blocks of a save the backend never took, on a shorter
        clock, and leaves a save already handed out alone -- precisely the case
        this reclaims once the report is not coming. Between them every deferred
        save has a way out.
        """
        timeout = self._save_abandon_timeout_s()
        if timeout <= 0 or not self.deferred_free_blocks:
            return 0
        now = time.monotonic()
        if now < self._next_save_reconcile_at:
            return 0
        self._next_save_reconcile_at = now + _SAVE_RECONCILE_INTERVAL_S
        stalled = [
            seq
            for seq in list(self.deferred_free_blocks.values())
            if getattr(seq, "_deferred_save_at", None) is not None
            and now - seq._deferred_save_at >= timeout
        ]
        for seq in stalled:
            self.deferred_free_blocks.pop(seq.id, None)
            self._connector_abandon_save(seq)
            self.block_manager.deallocate(seq)
            self._abandoned_saves += 1
        if stalled:
            logger.warning(
                "Reclaimed %d offload save(s) still deferred after %.0fs with no "
                "completion report (LMCache force-unpins a stalled save without "
                "reporting it); freed their blocks so the engine does not stall. "
                "total abandoned_saves=%d",
                len(stalled),
                timeout,
                self._abandoned_saves,
            )
        return len(stalled)

    def _unschedulable_reason(self, seq: Sequence) -> str | None:
        """Return a human-readable reason if `seq` is permanently unschedulable.

        Only checks static (configuration-time) capacity. Dynamic conditions
        that can clear up as other seqs finish (e.g. transiently full
        per-req-cache pool) are NOT checked here — they're warned at submit
        time (`_warn_if_unschedulable`) but not eligible for permanent drop
        at schedule time, since the prefill loop's existing `can_allocate`
        check will retry them later.

        Permanent failure modes (each leaves the seq stuck in `waiting`
        forever and would head-of-line block the prefill loop, which
        `break`s on the first oversized seq):
          - prompt longer than `max_model_len` → exceeds per-seq KV cache
            geometry; attention backends size `block_tables` as
            `max_model_len // block_size` cols and would crash with a
            broadcast error at prepare-time. (Checked first since it's the
            usual actionable cause.)
          - prompt longer than `max_num_batched_tokens` AND chunked prefill
            disabled → no single prefill forward can ever fit it (with chunked
            prefill enabled, the prompt is split across steps and this is fine)
          - prompt's KV blocks exceed the total pool size → never fits even on
            a fully empty pool. Counted per-rank (dcp-local, via
            `BlockManager.num_pool_blocks`) to match how the pool is sized and
            drawn, so a dcp>1 deployment admits prompts `dcp_world_size` times
            longer than the global block count alone would suggest.

        Called at submit time (`_warn_if_unschedulable`, which logs the
        reason and adds extra dynamic warnings) and at schedule time
        (drops the seq before it reaches the attention backend).
        """
        num_tokens = seq.num_tokens
        if num_tokens > self.max_model_len:
            return (
                f"input tokens={num_tokens} > max_model_len={self.max_model_len}. "
                f"Increase --max-model-len or shorten the prompt."
            )
        # Multimodal prefills are never chunked (the vision embeddings cover the
        # whole prompt), so for them the batched-token budget is a hard cap even
        # when chunked prefill is on.
        is_multimodal = getattr(seq, "multimodal_data", None) is not None
        if (
            not self.enable_chunked_prefill or is_multimodal
        ) and num_tokens > self.max_num_batched_tokens:
            remedy = (
                "Increase --max-num-batched-tokens or shorten the prompt "
                "(multimodal prompts are never chunked)."
                if is_multimodal
                else "Increase --max-num-batched-tokens, enable chunked "
                "prefill, or shorten the prompt."
            )
            return (
                f"input tokens={num_tokens} > max_num_batched_tokens="
                f"{self.max_num_batched_tokens}. {remedy}"
            )
        bm = self.block_manager
        total_blocks = bm.kv.num_blocks
        # `num_pool_blocks`, not `seq.num_blocks`: the latter counts global
        # blocks, while the pool is sized and drawn per-rank. Under dcp>1 the
        # two differ by `dcp_world_size`.
        pool_blocks = bm.num_pool_blocks(num_tokens)
        if pool_blocks > total_blocks:
            scope = (
                f" (per-rank, dcp={bm.dcp_world_size})" if bm.dcp_world_size > 1 else ""
            )
            return (
                f"needs {pool_blocks} KV blocks{scope} for {num_tokens} input "
                f"tokens > total pool blocks={total_blocks}. Reduce prompt length "
                f"or raise --gpu-memory-utilization. (Per-req state cache lives in "
                f"its own pre-allocated tensor and does not consume pool blocks.)"
            )
        return None

    def _warn_if_unschedulable(self, seq: Sequence) -> None:
        """Log a single warning at submit time for permanently-unschedulable
        sequences. The seq still enters `waiting`; the prefill scheduler drops
        it later (see `schedule`).

        Also surfaces a dynamic configuration-time-only warning when the
        model was started with zero per-req-cache slots (max_num_seqs=0) —
        this is permanent if it holds at submit time, but is NOT eligible
        for schedule-time drop (a future config change could create slots).
        """
        reason = self._unschedulable_reason(seq)
        if reason is not None:
            logger.warning("Request %s will never be scheduled: %s", seq.id, reason)
            return
        bm = self.block_manager
        # No slots ever allocated (max_num_seqs=0 effectively) AND no slots
        # currently in use → seq with has_per_req_cache=True can never enter.
        # "No slots ever existed" is the permanent case, distinguished from
        # "all busy" by the pool's total capacity.
        # An empty free list means either "all slots in use" — which the
        # schedule loop handles by waiting — or "the pool can never hold one
        # request", the permanent case, and only that one is worth warning
        # about.
        #
        # Measured against what one request needs, not against zero. A pool of
        # 3 slots where a request takes `1 + num_spec` = 4 is as permanently
        # unschedulable as an empty one; `== 0` would let it wait forever in
        # silence. The group-shaped predicate this replaced divided before
        # comparing, so it caught that case for free.
        if (
            seq.has_per_req_cache
            and not bm.state.has_free(bm.state_slots_per_req)
            and bm.num_state_slots < bm.state_slots_per_req
        ):
            logger.warning(
                "Request %s will never be scheduled: needs %d per-req cache "
                "slot(s) but the pool holds %d (max_num_seqs=0 for this model "
                "type, or --num-speculative-tokens too high for it).",
                seq.id,
                bm.state_slots_per_req,
                bm.num_state_slots,
            )

    def take_rejected(self) -> list[Sequence]:
        """Pop and return any seqs the prefill scheduler dropped because
        `_unschedulable_reason` flagged them (oversized prompt, exhausted
        pool, etc.). Caller (EngineCore) pushes them onto the same
        output_queue as forward-finished seqs so `llm.generate()` returns
        an output for them instead of blocking forever.
        """
        if not self._rejected:
            return []
        out = self._rejected
        self._rejected = []
        return out

    def schedule(self) -> tuple[ScheduledBatch, dict[int, Sequence]] | None:
        """Run a scheduling pass and close the throughput window.

        **Override `_schedule`, not this.** The window's 10s cadence is a
        whole-program invariant, and putting the tick here makes it depend on
        one call rather than on a `_record_throughput()` hand-placed at every
        early return inside `_schedule`. There are already several such
        returns; the next one added would otherwise stall the status line for
        as long as it fires, and nothing would fail — the log would just go
        quiet, which is indistinguishable from an idle engine.
        """
        result = self._schedule()
        self._record_throughput(num_prompt_tokens=_prompt_tokens_of(result))
        return result

    def _schedule(self) -> tuple[ScheduledBatch, dict[int, Sequence]] | None:
        """Select the next batch of sequences for a forward pass.

        Tries prefill first; if no new prefills are ready, falls back to
        decoding already-running sequences.
        """
        self._schedule_tick += 1
        # Sources borrowed by the previous batch: its forward has been issued,
        # so they can go back on the free list.
        self.block_manager.complete_previous_state_batch()
        scheduled_seqs = {}
        num_seqs_prefill = 0
        num_batched_tokens = 0
        skipped_waiting_requests: deque[Sequence] = deque()
        num_scheduled_tokens: list[int] = []
        scheduled_spec_decode_tokens: dict[int, np.ndarray] = {}

        self._promote_ready_remote_kv_requests()
        self._park_ready_offload_partial_prefills()

        # should_allow_prefill() runs a cross-DP all_reduce and MUST be called
        # every tick on every rank for lockstep — hence before the early-return.
        if self.prefill_delayer is not None:
            # pending = fresh waiting new-tokens + resumable partials' remaining,
            # capped at the batch budget: the coalescer's accumulation signal.
            pending_tokens = min(
                self._waiting_new_token_count()
                + self._partial_prefill_remaining_tokens(),
                self.max_num_batched_tokens,
            )
            delayer_allows = self.prefill_delayer.should_allow_prefill(
                prefillable=self._can_admit_head_prefill(),
                pending_tokens=pending_tokens,
                # decode-only: self.running also holds mid-chunked-prefill seqs,
                # which are NOT decode load — counting them would defeat the
                # coalescer's "no decode → fire" fast path.
                running_decode_batch=max(
                    0, len(self.running) - self._partial_prefill_count
                ),
                kv_usage=self._kv_usage(),
                has_partial=self._partial_prefill_count > 0,
                oldest_waiting_age_ms=self._oldest_waiting_prefill_age_ms(),
            )
        else:
            delayer_allows = True

        if not self.running and not self.waiting:
            return None

        # ---- Phase 1: resume partial prefills from running ----
        # Gated by `delayer_allows` so cross-DP alignment still holds when one
        # rank is mid-chunked-prefill: a delayer veto skips both Phase 1 and
        # Phase 2 in lockstep. Inside that, skip the running-queue scan entirely
        # when no seq is mid-prefill — the common steady-state decode case —
        # using the counter maintained by postprocess / preempt / finished-removal.
        if delayer_allows and self._partial_prefill_count > 0:
            for seq in self.running:
                if num_seqs_prefill >= self.max_num_seqs:
                    break
                if not seq.is_partial_prefill:
                    continue
                remaining = seq.num_tokens - seq.num_cached_tokens
                if 0 < self.long_prefill_token_threshold < remaining:
                    remaining = self.long_prefill_token_threshold
                budget_remaining = self.max_num_batched_tokens - num_batched_tokens
                chunk = self._chunked_prefill_size(
                    remaining, budget_remaining, num_batched_tokens
                )
                if chunk:
                    chunk = self._finalize_prefill_chunk(
                        seq, seq.num_cached_tokens, chunk
                    )
                if chunk <= 0:
                    break
                num_batched_tokens += chunk
                num_seqs_prefill += 1
                seq.type = SequenceType.PREFILL
                scheduled_seqs[seq.id] = seq
                num_scheduled_tokens.append(chunk)

        # ---- Phase 2: new requests from waiting ----
        while (
            delayer_allows
            and (self.delay_factor <= 0 or self._passed_delay(time.time()))
            and self.waiting
            and num_seqs_prefill < self.max_num_seqs
            and num_batched_tokens < self.max_num_batched_tokens
        ):
            seq = self.waiting.popleft()

            # Client disconnected before this seq ever ran: it holds no KV yet
            # and needs no forward pass, so finish it outright and route via
            # `_rejected` (emits a finished RequestOutput through the same
            # output_queue). Must intercept here BEFORE the waiting->running
            # promotion below, which would overwrite ABORTED with RUNNING and
            # lose the abort intent.
            #
            # It does not follow that it holds nothing: a sequence parked for
            # a transfer was allocated before it parked, and holds a full block
            # table plus, for a hybrid, a state group. Leaking the group wedges
            # every hybrid request once enough disconnects accumulate, since
            # `can_allocate` refuses one outright with the pool empty.
            # `deallocate` is a no-op for a seq that really held nothing.
            if seq.status == SequenceStatus.ABORTED:
                self._reject_aborted_waiting(seq)
                continue

            # Drop seqs the static-capacity check at submit-time flagged as
            # permanently unschedulable (oversized prompt, exhausted pool,
            # etc.). They've already been warned; mark FINISHED + record the
            # rejection reason and route them to `_rejected` so EngineCore
            # surfaces them through the same output_queue as forward-finished
            # seqs. Without this they'd reach the attention backend (where an
            # oversized prompt crashes with a broadcast error) AND
            # `llm.generate()` would block forever waiting for an output.
            # Re-check here (not just at submit) since pool state may change.
            unschedulable = self._unschedulable_reason(seq)
            if unschedulable is not None:
                seq.status = SequenceStatus.FINISHED
                seq.leave_reason = f"unschedulable: {unschedulable}"
                self._rejected.append(seq)
                continue

            if len(self.running) >= self.max_num_seqs:
                self.waiting.appendleft(seq)
                break

            remote_ready_for_decode = self._resolve_waiting_remote_kv(
                seq, skipped_waiting_requests
            )
            if remote_ready_for_decode is None:
                continue

            offload_resume = self._is_offload_prefill_resume(seq)
            needs_remote_load = self._query_connector_prefill_match(
                seq,
                skip=remote_ready_for_decode or offload_resume,
            )

            if remote_ready_for_decode:
                self._schedule_first_decode_after_remote_kv(seq)
                continue

            if offload_resume:
                # Blocks already held from the pre-park allocate; only re-check
                # the batch budget. No re-match / re-allocate / re-park.
                num_new_tokens = seq.num_prompt_tokens - seq.num_cached_tokens
                budget_remaining = self.max_num_batched_tokens - num_batched_tokens
                chunk = self._prefill_chunk_for_budget(
                    num_new_tokens, budget_remaining, num_batched_tokens
                )
                if chunk is None:
                    self.waiting.appendleft(seq)
                    break
                chunk = self._finalize_prefill_chunk(seq, seq.num_cached_tokens, chunk)
                self._assert_positive_prefill_chunk(
                    chunk, num_new_tokens, budget_remaining
                )
                # This branch `continue`s past both assignments below, so the
                # refresh happens here too: `_mark_offload_load_ready` raised
                # `num_cached_tokens` to the post-load hit, and the stale field
                # is what reaches the API as `cached_tokens` -- the tier would
                # look like it returned nothing. Unconditional, because a seq
                # re-admitted after `preempt()` keeps the field.
                seq.prefix_cache_hit_tokens = seq.num_cached_tokens
                num_seqs_prefill, num_batched_tokens = self._schedule_prefill_seq(
                    seq,
                    chunk,
                    scheduled_seqs,
                    num_scheduled_tokens,
                    num_seqs_prefill,
                    num_batched_tokens,
                )
                self._uncount_inflight_load(seq)
                continue

            if (
                needs_remote_load
                and len(self.running) + self._num_parked_remote_kv >= self.max_num_seqs
            ):
                self.waiting.appendleft(seq)
                break

            # Probe cache hits FIRST so budget check sees the real
            # (post-prefix-cache) remaining token count. V4 SWA correctness is
            # enforced inside can_allocate (_swa_bounded_hit bounds the hit to
            # where the trailing-window SWA is present); no post-hoc warmup trim.
            num_cached_blocks = self.block_manager.can_allocate(seq)
            if num_cached_blocks < 0:
                self.waiting.appendleft(seq)
                break

            # Use num_tokens (not num_prompt_tokens) so preempted seqs re-forward
            # their decoded tokens — preempt() frees their KV blocks but keeps
            # the token_ids, so num_tokens > num_prompt_tokens and those tokens
            # still need KV recomputed.
            num_new_tokens = (
                seq.num_tokens - num_cached_blocks * self.block_manager.hash_block_size
            )
            # Vision embeddings are computed for the whole prompt in one shot
            # and scattered onto the placeholder positions of the tokens in the
            # batch, so a multimodal prefill must not be split: a partial chunk
            # would either miss the placeholders entirely or land them at the
            # wrong offsets. Take the prompt whole or wait for a step with
            # enough budget.
            #
            # TODO: support chunked multimodal prefill. Needs the vision
            # embeddings computed once and cached per request, then sliced by
            # the chunk's token offset at scatter time (see the merge site in
            # ModelRunner.run_model). Today the scheduler also clears
            # `seq.multimodal_data` after the first batch, so later chunks would
            # silently embed raw `<|media_pad|>` tokens into the KV cache. Until
            # that lands, `max_num_batched_tokens` caps multimodal prompt length
            # even with chunked prefill enabled.
            atomic_prefill = getattr(seq, "multimodal_data", None) is not None
            if (
                not atomic_prefill
                and self.enable_chunked_prefill
                and 0 < self.long_prefill_token_threshold < num_new_tokens
            ):
                num_new_tokens = self.long_prefill_token_threshold
            budget_remaining = self.max_num_batched_tokens - num_batched_tokens
            chunk = self._prefill_chunk_for_budget(
                num_new_tokens, budget_remaining, num_batched_tokens
            )
            if chunk is None or (atomic_prefill and chunk < num_new_tokens):
                self.waiting.appendleft(seq)
                break
            if not self.block_manager.allocate(seq, num_cached_blocks):
                # A state-less joint boundary could not be privatised without
                # risking a shared decoding sequence's blocks (finding #2).
                # Return the seq to waiting for a clean recompute once the pool
                # has room, and stop admitting this pass.
                self.block_manager.deallocate(seq)
                self.waiting.appendleft(seq)
                break

            # The hit the request kept, not the one `can_allocate` offered:
            # `allocate` may disown the boundary, and `num_cached_blocks` is the
            # pre-disown count. `seq.num_cached_tokens` also gets the unit right
            # under DCP, and is what EngineStats uses, so the user-visible number
            # and the internal hit rate cannot disagree.
            #
            # Guard: PD decode consumer inherits hit from prefill node;
            # don't clobber with the local hit (always 0 on consumer).
            if not seq.prefix_cache_hit_tokens:
                seq.prefix_cache_hit_tokens = seq.num_cached_tokens

            self._notify_connector_after_prefill_alloc(seq)

            needs_remote_load = self._confirm_remote_load_after_alloc(
                seq, needs_remote_load
            )

            if needs_remote_load:
                oj = seq.offload_joint
                if oj.load_hash != -1 and not oj.boundary_tokens:
                    # Two transfers but one report: the first completion would
                    # unpark the request while the other is still writing. Only
                    # reachable when the legs were decided separately -- a joint
                    # load arrives as one event via `_JointPark`.
                    logger.warning(
                        "seq %s has both a remote KV load and a state load "
                        "pending, with no joint boundary; dropping the state "
                        "load.",
                        seq.id,
                    )
                    if not self.block_manager.cancel_state_load(seq):
                        # Disown could not be backed (finding #2); requeue for a
                        # clean recompute instead of parking a load into shared
                        # blocks.
                        self.block_manager.deallocate(seq)
                        self.waiting.appendleft(seq)
                        break
                self._park_for_remote_load(seq, skipped_waiting_requests)
                continue

            if seq.offload_joint.boundary_tokens:
                # The KV leg was refused after the state leg was aimed at the
                # joint boundary, so the state claims a prefix whose KV nobody
                # will fetch. Disown it; `_decide_load_after_alloc` logs why.
                logger.debug(
                    "[JOINT-DISOWN] seq %s: KV leg refused at boundary %d; "
                    "recomputing from %d",
                    seq.id,
                    seq.offload_joint.boundary_tokens,
                    seq.num_cached_tokens,
                )
                # Privatise the claimed prefix (finding #2) so recompute-from-0
                # cannot tear a shared decode. cancel_state_load does it when a
                # CPU state load was pending; otherwise disown directly.
                if seq.offload_joint.load_hash != -1:
                    disowned = self.block_manager.cancel_state_load(seq)
                else:
                    disowned = self.block_manager.disown_claimed_prefix(seq)
                if not disowned:
                    self.block_manager.deallocate(seq)
                    self.waiting.appendleft(seq)
                    break
                seq.num_cached_tokens = 0
                # Partial reset by design: the KV/claim spans are left as-is,
                # matching the pre-record code -- not a full `reset_joint`.
                seq.offload_joint.boundary_tokens = 0
                seq.offload_joint.boundary_hash = -1

            if seq.offload_joint.load_hash != -1:
                # The kept state boundary is in LMCache, not HBM. It parks
                # exactly as a KV load does -- same queue, same backpressure,
                # same wake-up -- because to the scheduler both are one event:
                # a transfer into blocks this request already holds.
                self._park_for_remote_load(seq, skipped_waiting_requests)
                continue

            # Refresh, not a duplicate of the set above: that one is guarded
            # on the field being empty, so a seq re-admitted after `preempt()`
            # would keep its previous admission's hit. Unconditional here, where
            # the request is certain to prefill locally. The only path reaching
            # this with an inherited PD hit is a *failed* remote recv, and there
            # the local hit is the truthful number.
            seq.prefix_cache_hit_tokens = seq.num_cached_tokens

            chunk = self._adjust_prefill_chunk_after_alloc(seq, chunk)
            chunk = self._finalize_prefill_chunk(seq, seq.num_cached_tokens, chunk)

            # Re-assert the atomic-prefill invariant against the *final* chunk.
            # The guard above ran on the pre-allocation size, but the two
            # adjusters just above can both shorten the chunk after the fact --
            # `_adjust_prefill_chunk_after_alloc` defers part of a prefill for an
            # offload load, and `_finalize_prefill_chunk` lands the chunk on a
            # state-checkpoint rung. Either can leave a multimodal prompt split
            # even though it passed the pre-alloc check, which would scatter the
            # vision embeddings against the wrong positions (or, on a later
            # chunk once `multimodal_data` is cleared, embed raw `<|media_pad|>`
            # into the KV cache). Take it whole or not at all: undo this
            # admission and wait for a step that can, exactly as the joint-disown
            # path above requeues. `num_cached_tokens` is post-allocation, so
            # this stays correct even if the prefix was disowned.
            if atomic_prefill and chunk < seq.num_tokens - seq.num_cached_tokens:
                self.block_manager.deallocate(seq)
                self.waiting.appendleft(seq)
                break

            self._assert_positive_prefill_chunk(chunk, num_new_tokens, budget_remaining)
            num_seqs_prefill, num_batched_tokens = self._schedule_prefill_seq(
                seq,
                chunk,
                scheduled_seqs,
                num_scheduled_tokens,
                num_seqs_prefill,
                num_batched_tokens,
            )

        if skipped_waiting_requests:
            logger.debug(
                "Re-adding %d skipped requests back to waiting queue.",
                len(skipped_waiting_requests),
            )
            self.waiting.extend(skipped_waiting_requests)

        if self._num_parked_remote_kv > 0 and self._schedule_tick % 1000 == 0:
            logger.info(
                "PD backpressure: parked=%d, waiting=%d, running=%d, "
                "resident=%d/%d, kv_usage=%.2f",
                self._num_parked_remote_kv,
                len(self.waiting),
                len(self.running),
                len(self.running) + self._num_parked_remote_kv,
                self.max_num_seqs,
                self._kv_usage(),
            )

        if self._schedule_tick % 100 == 0:
            fates = self.block_manager.state_checkpoint_fates()
            if any(fates.values()):
                logger.info(
                    "state checkpoints: %s",
                    " ".join(f"{k}={v}" for k, v in sorted(fates.items())),
                )
            # Its own line rather than a key in the one above: those are what
            # became of a checkpoint, this is whether a prefix in LMCache could
            # be paired with one at all. Silent unless the joint load is on,
            # since both counters stay 0 without it.
            chosen = self.block_manager.joint_boundaries
            skips = self.block_manager.joint_skips
            if chosen or skips:
                # The hbm/tier split is the actionable half: both are reuse,
                # but only `tier` costs an image-sized H2D and a park. Under
                # #2045 a ratio dominated by `tier` says the PAGED pool is too
                # small for this concurrency -- checkpoints are being squeezed
                # out by the KV write stream they now share a pool with -- and
                # that is a diagnosis the fork-era split could not make.
                logger.info(
                    "joint kv: boundaries=%d (state_hbm=%d state_tier=%d) | %s",
                    chosen,
                    self.block_manager.state_hbm_boundaries,
                    self.block_manager.state_tier_boundaries,
                    " ".join(f"{k}={v}" for k, v in sorted(skips.items())),
                )

        total_tokens_num_prefill = sum(num_scheduled_tokens)

        if num_seqs_prefill > 0:
            # A cursor, not a hit count: it starts at the prefix-cache hit and
            # then advances by each finished chunk, so a chunked prompt logs the
            # same req_id repeatedly with this climbing by the previous `new`.
            # Logged as "done" so those repeats don't read as a growing hit.
            num_cached_tokens_list = [
                seq.num_cached_tokens for seq in scheduled_seqs.values()
            ]
            logger.info(
                f"Scheduled prefill batch: {num_seqs_prefill} reqs, "
                f"{total_tokens_num_prefill} new tokens "
                f"(done: {num_cached_tokens_list}, new: {num_scheduled_tokens}), "
                f"req_ids: {tuple(scheduled_seqs.keys())}"
            )
            self.prev_prompt = True
            # lip: TODO for prefill/decode mixed batch

            connector_meta_output = None
            if self.kv_connector is not None:
                self._publish_state_loads()
                self._publish_state_stores()
                connector_meta_output = self.kv_connector.build_connector_meta()

            # Freeze, per seq, whether this chunk finishes the prompt. Uses the
            # pre-advance offsets so it is correct whether or not schedule-time
            # advancement runs below.
            is_final_chunk = [
                (num_cached_tokens_list[i] + int(num_scheduled_tokens[i]))
                >= seq.num_prompt_tokens
                for i, seq in enumerate(scheduled_seqs.values())
            ]
            # Bound on num_tokens (not num_prompt_tokens): preempted seqs
            # re-forward generated tokens past the prompt boundary.
            next_token_ids = None
            if self.drafter_needs_next_token:
                next_token_ids = []
                for i, seq in enumerate(scheduled_seqs.values()):
                    end = num_cached_tokens_list[i] + int(num_scheduled_tokens[i])
                    next_token_ids.append(
                        -1 if end >= seq.num_tokens else int(seq.token_ids[end])
                    )

            # Reserve midstep checkpoint destinations for the chunks just
            # settled. Here rather than inside `_finalize_prefill_chunk`
            # because a reservation takes a slot off the free list, and
            # admission for this pass only finishes above — planning any
            # earlier would let a checkpoint's destination compete with a
            # request still to be let in. The batch below snapshots what this
            # leaves on each seq.
            for i, seq in enumerate(scheduled_seqs.values()):
                start = num_cached_tokens_list[i]
                self.block_manager.plan_midstep(
                    seq, start, start + int(num_scheduled_tokens[i])
                )

            prefill_batch = ScheduledBatch(
                seqs=scheduled_seqs,
                num_scheduled_tokens=num_scheduled_tokens,
                total_tokens_num=total_tokens_num_prefill,
                total_tokens_num_prefill=total_tokens_num_prefill,
                total_seqs_num=num_seqs_prefill,
                total_seqs_num_prefill=num_seqs_prefill,
                connector_meta_output=connector_meta_output,
                num_cached_tokens=num_cached_tokens_list,
                is_final_chunk=is_final_chunk,
                next_token_ids=next_token_ids,
                state_maintenance_ops=self.block_manager.take_state_maintenance_ops(),
            )
            self._consume_state_forks(scheduled_seqs)

            if self.advance_on_schedule:
                # Advance after batch build (so the batch keeps pre-advance
                # offsets) so the next schedule() issues the following chunk.
                self._advance_prefill_on_schedule(
                    scheduled_seqs, num_scheduled_tokens, is_final_chunk
                )

            return (prefill_batch, scheduled_seqs)

        # --- Decode scheduling ---
        num_seqs_decode = 0
        num_decode_tokens = 0
        # anchor + drafts if verifying locally, anchor alone otherwise.
        spec_width = self.mtp_k if self.spec_decode_local else 0
        tokens_per_decode_seq = spec_width + 1
        num_new_tokens = spec_width + 1
        remote_kv_blocks: set[int] = set()
        remote_kv_seq_blocks: dict[int, list[int]] = {}
        skipped_partial_prefills: list[Sequence] = []
        # Pipeline-parallel: seqs whose sampled token is still in flight cannot
        # be decoded yet. Re-queue them at the tail (like partial prefills) so
        # they are reconsidered once the head releases them post-postprocess.
        skipped_pp_inflight: list[Sequence] = []
        _pp_block = self._pp_inflight_token_block
        while self.running and num_seqs_decode < self.max_num_seqs:
            if num_decode_tokens + tokens_per_decode_seq > self.max_num_batched_tokens:
                break
            seq = self.running.popleft()
            if seq.is_partial_prefill:
                skipped_partial_prefills.append(seq)
                continue
            if _pp_block and seq.id in _pp_block:
                skipped_pp_inflight.append(seq)
                continue
            blocked_by_pinned_save = False
            preempted_current = False
            while not self.block_manager.can_append(seq, num_new_tokens):
                if self._preempt_one_running():
                    continue
                if self.preempt(seq):
                    preempted_current = True
                    break
                self.running.appendleft(seq)
                blocked_by_pinned_save = True
                break
            if blocked_by_pinned_save:
                break
            if not preempted_current:
                if self.spec_decode_local and seq.spec_token_ids.size > 0:
                    scheduled_spec_decode_tokens[seq.id] = seq.spec_token_ids
                num_seqs_decode += 1
                num_decode_tokens += num_new_tokens
                # For PD first-decode: if T0 was injected, may_append is
                # needed for the new position N. Without T0 injection,
                # blocks were already allocated during prefill.
                is_first = getattr(seq, "is_first_decode", False)
                if is_first and seq.block_table:
                    remote_kv_blocks.update(seq.block_table)
                    remote_kv_seq_blocks[seq.id] = list(seq.block_table)
                has_injected_t0 = (
                    is_first
                    and (seq.kv_transfer_params or {}).get("first_token_id") is not None
                )
                if not is_first or has_injected_t0:
                    self.block_manager.may_append(seq, num_new_tokens)
                if is_first:
                    logger.debug(
                        "[PD-FIRST-DECODE] seq %s: num_tokens=%d, "
                        "blocks=%d, injected_t0=%s, "
                        "last_block_num=%d, context_will_be=%d",
                        seq.id,
                        seq.num_tokens,
                        len(seq.block_table),
                        has_injected_t0,
                        seq.last_block_num_tokens,
                        seq.num_tokens,
                    )
                scheduled_seqs[seq.id] = seq
                seq.type = SequenceType.DECODE
                num_scheduled_tokens.append(num_new_tokens)
                seq.is_first_decode = False

        total_tokens_num_decode = sum(num_scheduled_tokens)

        if scheduled_seqs:
            self.running.extendleft(reversed(scheduled_seqs.values()))
        if skipped_partial_prefills:
            # Re-queue skipped partial prefills at the TAIL, not the head.
            #
            # A partial (chunked, prompt-not-done) prefill can land in this
            # decode loop when the cross-DP PrefillDelayer vetoes prefill for a
            # tick (Phase 1 skipped, so num_prefill==0 and the prefill-only
            # early-return doesn't fire). Re-inserting it at the head pins it
            # at running[0]; once it finishes prefill it becomes the batch's
            # position-0 *deferred* seq, which pushes the fresh decode seqs to
            # positions 1..N. TokenIDProcessor.prepare_input_ids then takes the
            # [deferred | new] path and indexes the (compacted)
            # scheduled_spec_decode_tokens array by those shifted positions —
            # running off the end (IndexError: index N out of bounds size N).
            #
            # Appending at the tail keeps the partial out of position 0 (its
            # prefill still resumes: Phase 1 scans all of `running`), so new
            # decode seqs stay contiguous from position 0 and the safe
            # [new | deferred] slice path is used.
            self.running.extend(skipped_partial_prefills)
        if skipped_pp_inflight:
            self.running.extend(skipped_pp_inflight)

        connector_meta_output = None
        if self.kv_connector is not None:
            self._publish_state_loads()
            self._publish_state_stores()
            connector_meta_output = self.kv_connector.build_connector_meta()

        decode_batch = ScheduledBatch(
            seqs=scheduled_seqs,
            num_scheduled_tokens=num_scheduled_tokens,
            total_tokens_num=total_tokens_num_decode,
            total_tokens_num_decode=total_tokens_num_decode,
            total_seqs_num=num_seqs_prefill + num_seqs_decode,
            total_seqs_num_prefill=num_seqs_prefill,
            total_seqs_num_decode=num_seqs_decode,
            connector_meta_output=connector_meta_output,
            num_spec_step=self.mtp_k if self.spec_decode_local else 0,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            remote_kv_block_ids=sorted(remote_kv_blocks) if remote_kv_blocks else [],
            remote_kv_seq_blocks=remote_kv_seq_blocks,
            # An empty batch cannot execute queued maintenance.
            # For spills that is not merely wasted work: a drained spill is
            # released only once its copy reaches a forward, so draining into a
            # batch that is never forwarded strands every slot it drained.
            state_maintenance_ops=(
                self.block_manager.take_state_maintenance_ops()
                if scheduled_seqs
                else None
            ),
        )
        self._consume_state_forks(scheduled_seqs)
        return (decode_batch, scheduled_seqs)

    @staticmethod
    def _consume_state_forks(scheduled_seqs: dict[int, Sequence]) -> None:
        """Clear the fork flags the batch just snapshotted.

        A fork describes one forward: the batch carries `state_fork_src`, and
        every later batch for the same seq must read and write the same slot
        again. Cleared here rather than in the batch constructor so the snapshot
        stays free of side effects on Sequence.
        """
        for seq in scheduled_seqs.values():
            seq.state_fork_src = -1

    # -- Remote KV / offload admission helpers ------------------------------
    def _resolve_waiting_remote_kv(
        self, seq: Sequence, skipped_waiting_requests: deque[Sequence]
    ) -> bool | None:
        """Resolve a ``WAITING_FOR_REMOTE_KVS`` request before admission.

        Returns:
          - ``None`` when the request is still blocked and has been requeued.
          - ``True`` when a P/D consumer should jump to first decode.
          - ``False`` when normal prefill admission should continue.
        """
        if seq.status != SequenceStatus.WAITING_FOR_REMOTE_KVS:
            return False

        if self._consume_failed_remote_kv(seq):
            return False

        if not self._update_waiting_for_remote_kv(seq):
            skipped_waiting_requests.append(seq)
            return None

        seq.status = SequenceStatus.WAITING
        if self._connector_flag("is_offload"):
            self._mark_offload_load_ready(seq)
            return False
        self._uncount_inflight_load(seq)
        return True

    def _consume_failed_remote_kv(self, seq: Sequence) -> bool:
        if not self._pop_req_id(self.failed_recving_kv_req_ids, seq.id):
            return False

        seq.status = SequenceStatus.WAITING
        if not self._connector_flag("is_offload"):
            self._uncount_inflight_load(seq)
        if seq.offload_joint.load_hash != -1 or seq.offload_joint.boundary_tokens:
            # The state never arrived, so the boundary is not this request's
            # history. Disown it exactly as `BlockManager.allocate` does at
            # admission, otherwise the resume runs with `num_cached_tokens > 0`
            # over a group nobody filled -- `has_initial_state` reads that as
            # "the recurrence continues". Before `offload_loaded_tokens` is set
            # below, so that field records the disowned figure. Privatise the
            # claimed prefix (finding #2): the offload resume path reuses this
            # exact block_table without re-allocating, so recompute-from-0 would
            # otherwise tear a shared decode.
            disowned = self.block_manager.disown_claimed_prefix(seq)
            seq.num_cached_tokens = 0
            seq.offload_joint.load_hash = -1
            # A joint boundary that failed on either leg is not this request's
            # history any more than a state-only one is, and leaving it set
            # would let the connector clamp a later lookup to it. Partial reset
            # by design: the KV/claim spans are left as-is, as before.
            seq.offload_joint.boundary_tokens = 0
            seq.offload_joint.boundary_hash = -1
            if not disowned:
                # Cannot back private copies; drop the whole table so the
                # re-admission rebuilds a clean one via can_allocate/allocate
                # (empty block_table disqualifies the offload-resume shortcut).
                self.block_manager.deallocate(seq)
        seq.offload_loaded = False
        seq.offload_loaded_tokens = seq.num_cached_tokens
        seq.offload_load_start_tokens = None
        seq.offload_load_failed = True
        return True

    def _reject_aborted_waiting(self, seq: Sequence) -> None:
        has_inflight_load = bool(getattr(seq, "_counted_as_inflight_load", False))
        seq.status = SequenceStatus.FINISHED
        seq.leave_reason = "aborted"
        self._rejected.append(seq)
        if not has_inflight_load or not self._connector_flag("is_offload"):
            self._uncount_inflight_load(seq)
            return

        self.deferred_free_blocks[seq.id] = seq
        terminal_queued = self._pop_req_id(
            self.finished_recving_kv_req_ids, seq.id
        ) or self._pop_req_id(self.failed_recving_kv_req_ids, seq.id)
        terminal_consumed = bool(
            getattr(seq, "offload_loaded", False)
            or getattr(seq, "offload_load_failed", False)
        )
        if terminal_queued or terminal_consumed:
            self._cleanup_aborted_load(seq)
            return
        seq._awaiting_aborted_load_cleanup = True

    def _cleanup_aborted_load(self, seq: Sequence) -> None:
        if hasattr(seq, "_awaiting_aborted_load_cleanup"):
            delattr(seq, "_awaiting_aborted_load_cleanup")
        self._uncount_inflight_load(seq)
        self._maybe_release_deferred(seq)

    def _finish_aborted_load_cleanup(self, req_id) -> bool:
        seq = self._deferred_sequence(req_id)
        if seq is None or not getattr(seq, "_awaiting_aborted_load_cleanup", False):
            return False
        self._cleanup_aborted_load(seq)
        return True

    def _mark_offload_load_ready(self, seq: Sequence) -> None:
        """Turn a completed offload load into a suffix-prefill resume."""
        loaded = getattr(seq, "offload_loaded_tokens", None)
        load_start = getattr(seq, "offload_load_start_tokens", None)
        logger.debug(
            "[OFFLOAD-WAKE] seq %s: loaded=%s prev_cached=%d num_tokens=%d",
            seq.id,
            loaded,
            seq.num_cached_tokens,
            seq.num_tokens,
        )
        if loaded is not None and loaded > seq.num_cached_tokens:
            promoted = 0
            if load_start is not None and load_start < loaded:
                promoted = self.block_manager.publish_loaded_prefix(
                    seq,
                    start_token=load_start,
                    end_token=loaded,
                )
                logger.info(
                    "[OFFLOAD-PROMOTE] seq=%s loaded_range=%d:%d "
                    "gpu_indexed_tokens=%d",
                    seq.id,
                    load_start,
                    loaded,
                    promoted,
                )
            seq.offload_promoted_tokens = promoted
            seq.num_cached_tokens = loaded
            # Report the extended CPU-offload hit without reducing any
            # prefix-cache hit inherited from an upstream prefill node.
            seq.prefix_cache_hit_tokens = max(seq.prefix_cache_hit_tokens, loaded)
        seq.offload_load_start_tokens = None
        seq.offload_loaded = True
        # A state load moves no KV, so the block above does nothing for it --
        # `num_cached_tokens` is already the boundary the state covers, and all
        # that is left is to stop calling the load pending. A joint load is the
        # one case where both halves fire, and `_JointPark` held the wake until
        # both legs reported.
        # Partial reset by design: the KV/claim spans are left as-is, as the
        # pre-record code did -- not a full `reset_joint`.
        seq.offload_joint.load_hash = -1
        seq.offload_joint.boundary_tokens = 0
        seq.offload_joint.boundary_hash = -1

    def _is_offload_prefill_resume(self, seq: Sequence) -> bool:
        """True when offload already owns blocks and should resume suffix prefill.

        This avoids a second prefix lookup and, more importantly, avoids calling
        ``BlockManager.allocate`` again for a sequence whose block table was
        allocated before it parked for the LMCache load.
        """
        return (
            self._connector_flag("is_offload")
            and (
                getattr(seq, "offload_loaded", False)
                or getattr(seq, "offload_load_failed", False)
            )
            and len(seq.block_table) > 0
        )

    def _query_connector_prefill_match(self, seq: Sequence, *, skip: bool) -> bool:
        """Ask the connector whether this prefill should park for remote KV."""
        if skip or self.kv_connector is None:
            return False
        ext_tokens, needs_remote_load = self.kv_connector.get_num_new_matched_tokens(
            seq
        )
        # Keep the lookup's answer, not just the park flag: `can_allocate` runs
        # next and a hybrid's two legs have to be aimed at one boundary, for
        # which this is the KV leg's ceiling. In tokens, and absolute -- the
        # connector reports what it found *beyond* what the seq already has.
        seq.offload_joint.kv_prefix_tokens = int(ext_tokens) + int(
            seq.num_cached_tokens
        )
        return needs_remote_load

    def _schedule_first_decode_after_remote_kv(self, seq: Sequence) -> None:
        """P/D path: a remote prefill completed, so schedule first decode."""
        seq.status = SequenceStatus.RUNNING
        seq.is_first_decode = True
        first_token_id = (seq.kv_transfer_params or {}).get("first_token_id")
        if first_token_id is not None:
            seq.append_token(first_token_id)
            seq._injected_t0 = first_token_id
            if self.mtp_k > 0:
                drafts = list(
                    (seq.kv_transfer_params or {}).get("draft_token_ids") or []
                )[: self.mtp_k]
                for d in drafts:
                    seq.append_token(int(d))
                # Pad to the full verify window regardless of what the producer
                # sent -- normally nothing, since a producer does not
                # speculate. The first decode slices the trailing `mtp_k + 1`
                # tokens, so a short seq makes that slice reach back into the
                # prompt: T0 lands at the end of the window instead of at its
                # head and the consumer verifies prompt tokens as drafts.
                # Placeholders are what postprocess leaves on every other
                # running seq, and the target rejects any the drafter's absence
                # left it disagreeing with.
                for _ in range(self.mtp_k - len(drafts)):
                    seq.append_token(self.eos_token_id)
                seq.spec_token_ids = np.asarray(drafts, dtype=np.int32)
        logger.debug(
            "[PD-TRANSITION] seq %s: num_tokens=%d, "
            "num_prompt=%d, blocks=%d, first_token=%s, "
            "last_5_tids=%s",
            seq.id,
            seq.num_tokens,
            seq.num_prompt_tokens,
            len(seq.block_table),
            first_token_id,
            seq.token_ids[-5:],
        )
        self.running.append(seq)

    def _chunked_prefill_size(
        self, num_new_tokens: int, budget_remaining: int, num_batched_tokens: int
    ) -> int:
        """Tokens to forward this step, or 0 to leave the request for the next.

        A chunk cut short by the budget is floored onto the block grid. When
        what's left is under one aligned unit the answer is 0: that sliver buys
        the request nothing — it needs a later step to finish either way — so
        all it does is split the prefill into an extra forward, off the grid.
        Flooring is also what frees the remainder the sliver is made of, so
        without the 0 case the alignment manufactures its own tail; a
        16384-token budget was going out as `..., 640, 10`.

        Never 0 for an empty batch: something has to go out each step, or a
        `max_num_batched_tokens` below the alignment would stall forever.
        """
        chunk = min(num_new_tokens, budget_remaining)
        if chunk >= num_new_tokens:
            return chunk
        aligned = chunk - chunk % max(self.block_manager.block_size, 64)
        return aligned or (0 if num_batched_tokens else chunk)

    def _prefill_chunk_for_budget(
        self, num_new_tokens: int, budget_remaining: int, num_batched_tokens: int
    ) -> int | None:
        if self.enable_chunked_prefill:
            return (
                self._chunked_prefill_size(
                    num_new_tokens, budget_remaining, num_batched_tokens
                )
                or None
            )
        if num_new_tokens > budget_remaining and num_batched_tokens > 0:
            return None
        return num_new_tokens

    def _finalize_prefill_chunk(self, seq: Sequence, start: int, chunk: int) -> int:
        """Align a prefill chunk to the state checkpoint boundary and vet forks.

        Two jobs, both about `BlockManager`'s state checkpoints:

        1. A checkpoint can only be kept where a forward ends exactly on a rung
           of the ladder — otherwise the state is ahead of the hash it would be
           filed under. So land chunks on that grid (plus, at most once, a
           position this seq itself was seen to want), shortening to the
           previous rung; `BlockManager.checkpoint_cut` owns the arithmetic, so
           that it cannot drift from the rule deciding what actually gets kept.
        2. The forward carrying a fork has to fill the request's new slot by
           itself. If the budget left a chunk too short for that, drop the fork
           rather than the request — unless the source is shared with another
           request forking off it this step, which rules out taking it over. Then
           the fork stays and the chunk is held at `min_fork_tokens` instead;
           `can_allocate` only offered a resumable boundary with that many
           prompt tokens behind it, so the tokens are there to forward. Only the
           rolling state class forks, so this job asks it directly.

        No-op for models without per-request state, and for any prompt shorter
        than one checkpoint interval.
        """
        bm = self.block_manager
        # A multimodal prompt is prefilled whole -- chunked multimodal prefill is
        # unsupported (see the `atomic_prefill` guard in `_schedule_prefills`), so
        # it is either admitted entire or requeued entire. Landing it on a state
        # checkpoint rung shortens it deterministically, and the post-alloc atomic
        # re-assert then requeues it whole; the very next pass produces the same
        # shortened chunk and requeues again -- a livelock with idle GPUs and
        # head-of-line blocking. Skip the rung cut here: the checkpoint machinery
        # keeps a checkpoint only where a forward ends exactly on a rung, so a
        # whole prompt that ends off-grid simply keeps no mid-prompt checkpoint,
        # which is correct for a single-shot prefill (there is no later chunk to
        # resume from anyway). Forks do not arise on a fresh multimodal prompt
        # (`state_fork_src < 0`), so the fork-vetting below is likewise moot.
        if getattr(seq, "multimodal_data", None) is not None:
            return chunk
        target = bm.checkpoint_cut(seq, start, start + chunk)
        if target:
            chunk = target - start
        # `cancel_state_fork` runs only when the first two hold, and returning
        # False is what leaves the fork in place — hence holding the chunk open.
        if (
            seq.state_fork_src >= 0
            and chunk < bm.state.min_fork_tokens
            and not bm.cancel_state_fork(seq)
        ):
            chunk = bm.state.min_fork_tokens
        return chunk

    def _checkpoint_room(self, seq: Sequence, finished: bool) -> int:
        """Tokens the forward after this one carries, for `hash_decode_blocks`.

        0 means "do not checkpoint here", for any of three reasons:

        - the request stops on this step, so there is nothing after it: no
          forward to fork into the slot a checkpoint would hand away, and no
          batch to issue a copy on either;
        - the seq is still on its prompt, where the prefill call site has
          already decided using the prompt's own remainder;
        - speculative decode, for a class that checkpoints by forking. Two
          reasons, and either is enough. The spec path's state index has no
          read-side counterpart (see
          `GDNAttentionMetadataBuilder.prepare_state_indices`), so a fork must
          never reach it. And a spec step commits `1 + accepted_drafts` tokens,
          which is what a fork's successor actually gets — the rest is rolled
          back and re-forwarded — so no promise made here can be kept.
          Checkpointing during *prefill* stays safe on the same models:
          `min_fork_tokens` guarantees prompt is left over, and prompt always
          forwards down the non-spec path.

        A class that checkpoints by copying is bound by none of that: the
        destination is complete when the copy lands, so any decode step will do.

        Otherwise plain decode carries exactly one token, and whether that is
        enough to fill a fresh slot is the backend's `min_fork_tokens` to say.
        """
        if finished or seq.type != SequenceType.DECODE:
            return 0
        if self.mtp_k and self.block_manager.state.transfer.forks:
            return 0
        return 1

    @staticmethod
    def _assert_positive_prefill_chunk(
        chunk: int, num_new_tokens: int, budget_remaining: int
    ) -> None:
        assert chunk > 0, (
            f"chunk must be positive: {chunk=}, "
            f"{num_new_tokens=}, {budget_remaining=}"
        )

    def _schedule_prefill_seq(
        self,
        seq: Sequence,
        chunk: int,
        scheduled_seqs: dict[int, Sequence],
        num_scheduled_tokens: list[int],
        num_seqs_prefill: int,
        num_batched_tokens: int,
    ) -> tuple[int, int]:
        num_seqs_prefill += 1
        if self.engine_stats.cache_enabled:
            # Hit counts are in hash blocks — one block_table entry spans
            # `block_size * dcp_world_size` tokens — so scaling by block_size
            # would under-report by the DCP factor.
            hbs = self.block_manager.hash_block_size
            # The reuse ceiling, mirroring `can_allocate`'s match loop, which
            # runs over `range(n_hash_blocks - 1)`: prefill must forward at
            # least one block to produce sampler logits, so the trailing block
            # is never a reuse candidate and no cache can be charged for it.
            # Floored at 0 for a sequence shorter than one hash block, whose
            # ceiling is genuinely zero — nothing about it is reusable.
            n_hash_blocks = (seq.num_tokens + hbs - 1) // hbs
            num_reusable_tokens = min(max(n_hash_blocks - 1, 0) * hbs, seq.num_tokens)
            # `num_cached_tokens` sits above the HBM walk once a load landed
            # -- that is the feature. Those tokens are not paged-pool reuse, so
            # they go in their own series rather than inflating `cached` and
            # making `wanted - cached` negative. `seq.num_cached_tokens` is
            # deliberately NOT clamped: the forward must skip what was loaded.
            wanted_tokens = seq.num_wanted_hit_blocks * hbs
            hbm_cached_tokens = min(seq.num_cached_tokens, wanted_tokens)
            offload_tokens = seq.num_cached_tokens - hbm_cached_tokens
            self.engine_stats.update_cache(
                hbm_cached_tokens,
                seq.num_tokens,
                seq.num_compressed_hit_blocks * hbs,
                wanted_tokens,
                num_reusable_tokens,
                num_offload_tokens=offload_tokens,
            )
        num_batched_tokens += chunk
        seq.status = SequenceStatus.RUNNING
        seq.type = SequenceType.PREFILL
        self.running.append(seq)
        scheduled_seqs[seq.id] = seq
        num_scheduled_tokens.append(chunk)
        return num_seqs_prefill, num_batched_tokens

    def _notify_connector_after_prefill_alloc(self, seq: Sequence) -> None:
        if self.kv_connector is not None:
            self.kv_connector.update_state_after_alloc(seq)

    def _confirm_remote_load_after_alloc(
        self, seq: Sequence, needs_remote_load: bool
    ) -> bool:
        if not needs_remote_load:
            return False
        if hasattr(self.kv_connector, "should_park_for_load_after_alloc"):
            return self.kv_connector.should_park_for_load_after_alloc(seq)
        return True

    def _settle_state_load(self, req_id, ok: bool) -> None:
        """Release the state group a load reserved. No-op for an id the state
        index never issued, and for a scheduler built without a block manager
        (the connector test doubles)."""
        settle = getattr(
            getattr(self, "block_manager", None), "settle_state_load", None
        )
        if settle is not None:
            settle(req_id, ok=ok)

    def _abandon_state_load(self, req_id) -> None:
        """Release a state group a load reserved *without* voting the outcome.

        The 'neither outcome' release, for a joint load whose state H2D landed
        intact but whose KV leg failed (an LMCache LRU miss on the KV chunk).
        Settling that as a failure would call `StateOffloadIndex.fail_load` and
        `forget(h)`, permanently un-advertising a state image whose bytes are
        still present -- the next request over that prefix could have reloaded
        it. `abandon_load` pops the pending reservation and releases the orphan
        slot but keeps the hash loadable. Same guards as `_settle_state_load`:
        no-op for an id the state index never issued and for a scheduler built
        without a block manager (the connector test doubles)."""
        abandon = getattr(
            getattr(self, "block_manager", None), "abandon_state_load", None
        )
        if abandon is not None:
            abandon(req_id)

    def _state_store_pending_cap(self) -> int:
        """The running-plus-queued cap for state-tier stores.

        Read off the connector's public `max_pending_saves` -- the canonical
        `_offload_common.max_pending_saves` value it computed from
        `kv_connector_extra_config` and `OFFLOAD_COPY_WORKERS`, the exact bound
        the KV leg's `_may_emit_save` enforces. Sharing that number rather than
        re-reading the bare `OFFLOAD_MAX_PENDING_SAVES` default of 2 keeps the
        two legs from pinning different amounts of the same pool, and honours a
        per-connector `"max_pending_saves"` override the env reader never sees.
        Falls back to the env reader for a connector (or test double) that never
        bounded its save queue (`max_pending_saves` is None, as on dense) and,
        under `kv_connector: multi`, reaches the bound through the state-tier sub
        -- `MultiConnectorScheduler` bounds nothing of its own, so without the
        sub probe the composite silently caps the state leg at the env default of
        2 while the KV leg keeps the sub's real bound. The public accessor means
        neither read reaches through the delegating shell's `_impl` any more.
        """
        conn = self.kv_connector
        cap = getattr(conn, "max_pending_saves", None)
        if cap is None:
            # Under `multi` the bound lives on the state-tier sub (itself a
            # delegating shell), reached the same way the state face is routed.
            sub = getattr(conn, "_state_tier_sub", None)
            tier = sub() if callable(sub) else None
            if tier is not None:
                cap = getattr(tier, "max_pending_saves", None)
        if isinstance(cap, int) and not isinstance(cap, bool) and cap > 0:
            return cap
        return _offload_max_pending_saves()

    def _publish_state_stores(self) -> None:
        """Hand this pass's ready checkpoints to the connector for the CPU tier.

        Beside `_publish_state_loads` and on the same schedule, so one pass
        makes one decision about the tier. The cap is the connector's public
        `max_pending_saves` (see `_state_store_pending_cap`) -- the same bound
        the KV leg's `_may_emit_save` enforces, and for the same reason: each
        outstanding transfer holds the bytes it is reading out of the pool, so
        the queue depth is also how much of the pool a slow backend can pin.

        `take_state_stores` pins as it hands over, so anything the connector
        will not carry has to be settled here rather than left to the
        reconciler's full timeout -- the units are already held.
        """
        if self.kv_connector is None:
            return
        take = getattr(getattr(self, "block_manager", None), "take_state_stores", None)
        if take is None:
            return
        stores = take(self._state_store_pending_cap())
        if not stores:
            return
        accepted = False
        enqueue = getattr(self.kv_connector, "enqueue_state_stores", None)
        if enqueue is not None:
            accepted = bool(enqueue(stores))
        if accepted:
            return
        offload = getattr(getattr(self, "block_manager", None), "state_offload", None)
        if offload is not None:
            offload.stores_refused += len(stores)
        logger.warning(
            "state offload: %s did not carry %d state store(s); releasing "
            "their units now. Either this connector has no state tier (the "
            "index should not have been installed against it), or the "
            "delegating shell is missing an `enqueue_state_stores` forwarder "
            "to the implementation that does -- check both before assuming "
            "the configuration is wrong.",
            type(self.kv_connector).__name__,
            len(stores),
        )
        for op, _units in stores:
            # `attempted=False`: this refusal is already counted once above as
            # `stores_refused`; the units must still be released, but the store
            # never reached a worker, so it is not a `stores_failed`.
            self.block_manager.settle_state_store(op, ok=False, attempted=False)

    def _publish_state_loads(self) -> None:
        """Hand this pass's state-tier loads to the connector, before it builds
        its metadata.

        Drained here rather than at the park, because the connector publishes
        once per pass and a load handed over twice would write the same entry
        into a group the first transfer is already filling.
        """
        if self.kv_connector is None:
            return
        # Guarded like the `_settle_state_load` / `_publish_state_stores`
        # siblings: a scheduler built without a block manager (the connector
        # test doubles) would otherwise AttributeError on the bare deref.
        take = getattr(getattr(self, "block_manager", None), "take_state_loads", None)
        if take is None:
            return
        loads = take()
        if not loads:
            return
        accepted = False
        if hasattr(self.kv_connector, "enqueue_state_loads"):
            accepted = bool(self.kv_connector.enqueue_state_loads(loads))
        if accepted:
            return
        # Nothing will carry them and the requests are parked against a report
        # that would never come, so wake them for recompute -- `failed_loading`
        # means "recompute over the blocks already allocated".
        #
        # `abandon_state_load`, not `settle(ok=False)`: nothing was attempted,
        # so this is not a miss to count against the index.
        logger.warning(
            "state offload: %s did not carry %d state load(s); waking those "
            "requests to recompute. Either this connector has no state tier, "
            "or the delegating shell is missing an `enqueue_state_loads` "
            "forwarder -- the class named here is the shell, not the "
            "implementation behind it.",
            type(self.kv_connector).__name__,
            len(loads),
        )
        for req_id, _h, _group in loads:
            self.block_manager.abandon_state_load(req_id)
            self.failed_recving_kv_req_ids.append(req_id)

    def _park_for_remote_load(
        self, seq: Sequence, skipped_waiting_requests: deque[Sequence]
    ) -> None:
        skipped_waiting_requests.append(seq)
        seq.status = SequenceStatus.WAITING_FOR_REMOTE_KVS
        self._count_inflight_load(seq)

    def _count_inflight_load(self, seq: Sequence) -> None:
        if not getattr(seq, "_counted_as_inflight_load", False):
            self._num_parked_remote_kv += 1
            seq._counted_as_inflight_load = True

    def _uncount_inflight_load(self, seq: Sequence) -> None:
        if getattr(seq, "_counted_as_inflight_load", False):
            self._num_parked_remote_kv -= 1
            seq._counted_as_inflight_load = False

    def _adjust_prefill_chunk_after_alloc(self, seq: Sequence, chunk: int) -> int:
        if self.kv_connector is not None and hasattr(
            self.kv_connector, "adjust_prefill_chunk_after_alloc"
        ):
            return self.kv_connector.adjust_prefill_chunk_after_alloc(seq, chunk)
        return chunk

    def _is_preemptable(self, seq: Sequence) -> bool:
        return not self._connector_should_defer_free(seq)

    def _preempt_one_running(self) -> bool:
        for index in range(len(self.running) - 1, -1, -1):
            candidate = self.running[index]
            if not self._is_preemptable(candidate):
                continue
            del self.running[index]
            if self.preempt(candidate):
                return True
            self.running.insert(index, candidate)
        return False

    def preempt(self, seq: Sequence) -> bool:
        if not self._is_preemptable(seq):
            return False
        self.total_preemptions += 1
        seq.status = SequenceStatus.WAITING
        # Strip placeholder + rejected draft tokens added by postprocess.
        # Real token count = seq.num_tokens - mtp_k - num_rejected
        # (same formula as postprocess line: num_tokens = seq.num_tokens - self.mtp_k - num_rejected)
        if self.spec_decode_local and self.mtp_k > 0:
            strip = self.mtp_k + seq.num_rejected
            if strip > 0:
                del seq.token_ids[-strip:]
                del seq.output_tokens[-strip:]
                seq.num_tokens -= strip
        seq.num_rejected = 0
        seq.num_bonus_tokens = 0
        seq.num_placeholder_tokens = 0
        seq.spec_token_ids = np.array([], dtype=np.int32)
        seq.is_first_decode = False
        if seq.is_partial_prefill:
            seq.is_partial_prefill = False
            self._partial_prefill_count -= 1
        # This free runs no `request_finished`; drop any stall-escaped save
        # tracker now so the connector's save loop cannot later read these
        # surrendered blocks (the mutator half of `should_defer_free`'s escape).
        self._connector_release_stalled_save(seq)
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
        return True

    def _advance_prefill_on_schedule(
        self,
        scheduled_seqs: dict[int, Sequence],
        num_scheduled_tokens: list[int],
        is_final_chunk: list[bool],
    ) -> None:
        """Advance chunked-prefill progress at schedule time (pipeline-parallel).

        Applies the num_cached_tokens / is_partial_prefill bookkeeping that
        postprocess() otherwise does, so the head can schedule the next chunk
        before this one's output returns. Hash registration is NOT done here — it
        stays in postprocess where the forward has computed the KV.
        """
        for i, seq in enumerate(scheduled_seqs.values()):
            seq.num_cached_tokens += int(num_scheduled_tokens[i])
            now_partial = not is_final_chunk[i]
            if now_partial != seq.is_partial_prefill:
                self._partial_prefill_count += 1 if now_partial else -1
                seq.is_partial_prefill = now_partial

    @staticmethod
    def _pp_inflight_req_ids(batch: ScheduledBatch):
        """Req_ids that will produce a not-yet-appended token in this batch.

        A whole decode batch, or the final chunk of a chunked prefill, yields a
        token the head has not appended yet. mark_pp_inflight() and
        release_pp_inflight() MUST use this same predicate so every add() has a
        matching discard(); a seq spanning two batches (middle then final chunk)
        is only blocked/released by its final-chunk batch.
        """
        final = batch.is_final_chunk
        if batch.total_seqs_num_decode > 0:
            yield from batch.req_ids
        elif final is not None:
            for i, req_id in enumerate(batch.req_ids):
                if final[i]:
                    yield req_id

    def mark_pp_inflight(self, batch: ScheduledBatch) -> None:
        """Head: block re-scheduling of seqs whose token is now in flight.

        Blocking them until release_pp_inflight() prevents decoding against a
        stale token while the pipeline is filled.
        """
        for req_id in self._pp_inflight_req_ids(batch):
            self._pp_inflight_token_block.add(req_id)

    def release_pp_inflight(self, batch: ScheduledBatch) -> None:
        """Head: release seqs blocked by mark_pp_inflight after postprocess.

        Discards exactly the set mark_pp_inflight() added (see
        _pp_inflight_req_ids) so a middle-chunk batch cannot clear a block that a
        later final-chunk batch set for the same seq.
        """
        for req_id in self._pp_inflight_req_ids(batch):
            self._pp_inflight_token_block.discard(req_id)

    def _batch_seq_lookup(self, seqs: Iterable[Sequence] | None) -> dict[int, Sequence]:
        """Look up batch seqs directly; falls back to ``running`` if None."""
        if seqs is not None:
            return {seq.id: seq for seq in seqs}
        return {seq.id: seq for seq in self.running}

    def register_prefill_hashes(
        self, batch: ScheduledBatch, seqs: Iterable[Sequence] | None = None
    ) -> None:
        """Hash blocks for middle chunked-prefill chunks that skip postprocess."""
        if not self.block_manager.enable_prefix_caching:
            return
        if batch.is_final_chunk is None:
            return
        seq_by_id = self._batch_seq_lookup(seqs)
        for i, req_id in enumerate(batch.req_ids):
            seq = seq_by_id.get(req_id)
            if seq is None or not seq.block_table:
                logger.warning(
                    "register_prefill_hashes: seq %s unavailable "
                    "(possible preemption leak under PP)",
                    req_id,
                )
                continue
            chunk = int(batch.num_scheduled_tokens[i])
            start_tokens = int(batch.num_cached_tokens[i])
            self.block_manager.hash_blocks(seq, chunk, start_tokens=start_tokens)

    def postprocess(
        self,
        seqs: list[Sequence],
        fwd_output: ScheduledBatchOutput,
        stream_output_queue=None,
        batch: ScheduledBatch = None,
    ) -> list[Sequence]:
        """Process model outputs: update tokens, check stop conditions, free blocks.

        Also updates num_cached_tokens for prefill seqs and tracks which seqs
        are still mid-prefill (partial chunks) so their sampled tokens can be
        discarded.
        """
        # Remember which seqs were already in the middle of chunked prefill
        # before this postprocess call mutates seq.is_partial_prefill below.
        #
        # In deferred-output mode, fwd_output.token_ids is one step late. If a
        # seq finishes its final prompt chunk in this call, the token we see
        # here is still from the previous partial chunk, not the real first
        # generated token. Keep the old partial state so we can drop that stale
        # token later in this loop.
        prev_partial_ids: set[int] = set()
        # Middle-chunk req_ids whose sampled token must be dropped, frozen from
        # batch.is_final_chunk (a later schedule() may have already flipped the
        # live seq.is_partial_prefill while this batch was in flight).
        pp_middle_chunk_ids: set[int] = set()
        running_by_id = {seq.id: seq for seq in self.running} if batch else {}
        num_prefill = int(getattr(batch, "total_seqs_num_prefill", 0))
        if self._connector_flag("is_offload") and num_prefill:
            for req_id in batch.req_ids[:num_prefill]:
                seq = running_by_id.get(req_id)
                if seq is not None and seq.has_per_req_cache:
                    # StateTransfer.copy runs inside this prefill forward. Mark
                    # the live destination ready only after that forward has
                    # returned, before connector metadata for a later batch.
                    seq._state_initialized_after_alloc = True
        if batch is not None and self.advance_on_schedule:
            # Progress already advanced at schedule time; publish prefix-cache
            # hashes at the chunk's pre-advance offset and record non-final chunks.
            # See _batch_seq_lookup: seq may have left running.
            seq_by_id = self._batch_seq_lookup(seqs)
            final = batch.is_final_chunk
            for i, req_id in enumerate(batch.req_ids):
                seq = seq_by_id.get(req_id)
                if seq is None or final is None or not seq.block_table:
                    continue
                is_final = final[i]
                chunk = int(batch.num_scheduled_tokens[i])
                start_tokens = int(batch.num_cached_tokens[i])
                self.block_manager.hash_blocks(seq, chunk, start_tokens=start_tokens)
                if not is_final:
                    pp_middle_chunk_ids.add(req_id)
        elif batch is not None:
            for i, req_id in enumerate(batch.req_ids):
                seq = running_by_id.get(req_id)
                if seq is None or seq.type != SequenceType.PREFILL:
                    continue
                if seq.is_partial_prefill:
                    prev_partial_ids.add(seq.id)
                chunk = int(batch.num_scheduled_tokens[i])
                # Register prefix-cache hashes for blocks the prefill step
                # just finalized BEFORE advancing num_cached_tokens. Deferred
                # from BlockManager.allocate() so a hash is only published
                # once the block's KV has been computed by the forward —
                # correct under chunked-prefill where one block may span
                # multiple steps (hash_blocks clips to fully-filled blocks).
                self.block_manager.hash_blocks(seq, chunk)
                seq.num_cached_tokens += chunk
                # Prefill is partial until the whole PROMPT's KV is computed.
                # Compare against num_prompt_tokens, not num_tokens: once a
                # completion token is appended (this step's sampled token, or an
                # externally-appended EOS), num_tokens > num_prompt_tokens and
                # comparing against it would wrongly keep a finished prefill
                # flagged partial — which makes the EOS/finish loop below skip it.
                now_partial = seq.num_cached_tokens < seq.num_prompt_tokens
                if now_partial != seq.is_partial_prefill:
                    self._partial_prefill_count += 1 if now_partial else -1
                    seq.is_partial_prefill = now_partial

        prev_token_ids = fwd_output.token_ids
        draft_token_ids = fwd_output.draft_token_ids
        is_deferred_out = fwd_output.is_deferred_out
        token_logprobs = fwd_output.logprobs  # Optional[dict[int, float]]
        # update token_ids with the actual sampled token ids

        finished_seqs = []
        stream_outputs = []
        num_new_generation_tokens = 0

        need_placeholder = is_deferred_out or self.spec_decode_local
        # Drafts occupy trailing slots only on an engine that verifies them; a
        # drafting-only engine's tokens are all real.
        num_placeholder_width = self.mtp_k if self.spec_decode_local else 0
        num_placeholder = self.mtp_k
        if is_deferred_out:
            num_placeholder += 1

        for seq in self.running:
            # Update the running status
            idx = fwd_output.get_idx(seq.id)
            if idx is None:
                continue
            # Partial prefill: KV written but prefill not complete — discard
            # the sampled token. Prefix hashes are also deferred since
            # num_tokens < num_prompt_tokens until the prompt finishes.
            #
            # Under schedule-time advancement seq.is_partial_prefill reflects a
            # possibly-later schedule() and cannot gate THIS batch's output;
            # use the frozen middle-chunk set instead.
            if self.advance_on_schedule:
                if seq.id in pp_middle_chunk_ids:
                    continue
            elif seq.is_partial_prefill:
                continue
            # With deferred output, a token visible after a normal chunked-
            # prefill step belongs to the previous partial chunk and must be
            # dropped. An offload handoff is different: parking removes the
            # request from at least one intervening model-runner batch, which
            # already discards its deferred partial output. The first output
            # after the request resumes is fresh and must be kept.
            if seq.id in prev_partial_ids:
                continue
            # Register prefix-cache hashes for blocks the prefill step just
            # finalized. Deferred from BlockManager.allocate() so a hash is
            # only published after the block's KV has actually been computed
            # by the forward — keeps the block manager correct under chunked
            # prefill where one block may span multiple steps. Must run before
            # any seq state update so num_cached_tokens and block_table still
            # reflect the pre-step view.
            #
            # Gate is `not prefix_hashes_published`, not `seq.type ==
            # PREFILL`: ModelRunner runs in deferred-output mode by default
            # (tokenIDProcessor.is_deferred_out), so the prefill step's
            # postprocess sees idx=None and skips this seq (above). By the
            # time the prefill output surfaces, the next step's schedule has
            # already flipped seq.type to DECODE — the old PREFILL gate never
            # fires and the content index stays empty for prompt blocks (HBM
            # prefix cache silently dead). The flag gate fires once per seq
            # at the first postprocess with idx.
            #
            # `num_new` subtracts `num_placeholder` when deferred-output is
            # active: those slots are filled with the real prefill output
            # later in this loop, so they're not part of the prompt hash
            # chain — leaving them in would mint a stale partial-block hash.
            if not seq.prefix_hashes_published:
                if batch is None:
                    _num_new = seq.num_tokens - seq.num_cached_tokens
                    if need_placeholder:
                        _num_new -= num_placeholder
                    self.block_manager.hash_blocks(seq, max(0, _num_new))
                seq.prefix_hashes_published = True
            token_ids = prev_token_ids[idx]
            num_new_token = len(token_ids)
            token_logprob = None
            if token_logprobs is not None and seq.return_logprobs:
                token_logprob = token_logprobs.get(seq.id)

            # In-place overwrite only when placeholders already exist.
            if is_deferred_out or (
                self.spec_decode_local and seq.num_placeholder_tokens > 0
            ):
                # int() casts strip the np.int32 wrapper coming out of
                # fwd_output's np.ndarray indexing. Without these, the values
                # propagate into seq.num_rejected / seq.num_bonus_tokens, then
                # into seq.num_tokens via `preempt()`'s `-= mtp_k + num_rejected`,
                # contaminating downstream logs and arithmetic with np.int32.
                num_rejected = int(fwd_output.num_rejected[idx])
                num_bonus = int(fwd_output.num_bonus[idx])
                offset = 0 if (num_new_token + num_rejected) == 1 else self.mtp_k
                # Align stats with vLLM: only count steps that actually ran
                # speculation (drafts proposed and validated). Skip the
                # prefill-only step where no draft tokens were scored against
                # the target — vLLM gates this via
                # `if scheduled_spec_token_ids and generated_token_ids`.
                if (
                    self.engine_stats.spec_enabled
                    and num_new_token > 0
                    and (num_new_token + num_rejected) > 1
                ):
                    self.engine_stats.update_spec(num_new_token)
                seq.num_rejected = num_rejected
                seq.num_bonus_tokens = num_bonus
                # DSpark Phase 2: stash this step's scheduler-chosen ell on the
                # seq so the NEXT decode schedule can size its verification to
                # ell+1. Keyed by req_id (order-safe). Missing -> leave default
                # (mtp_k), so a request never gets under-verified.
                if fwd_output.dspark_ell is not None:
                    ell_r = fwd_output.dspark_ell.get(seq.id)
                    if ell_r is not None:
                        seq.dspark_next_ell = int(ell_r)
                required_placeholders = num_placeholder + offset
                missing_placeholders = required_placeholders - len(seq.output_tokens)
                if missing_placeholders > 0:
                    logger.warning(
                        "Repairing missing deferred-output placeholders for seq %s: "
                        "required=%d, available=%d",
                        seq.id,
                        required_placeholders,
                        len(seq.output_tokens),
                    )
                    for _ in range(missing_placeholders):
                        seq.append_token(self.eos_token_id)
                for i, el in enumerate(token_ids):
                    seq.token_ids[-required_placeholders + i] = el
                    seq.output_tokens[-required_placeholders + i] = el
                if seq.return_logprobs and token_logprob is not None:
                    if seq.logprobs:
                        seq.logprobs[-1] = token_logprob
                    else:
                        seq.logprobs.append(token_logprob)
            else:
                num_rejected = 0
                num_bonus = 0
                for token_id in token_ids:
                    seq.append_token(token_id)
                if seq.return_logprobs and token_logprob is not None:
                    seq.logprobs.append(token_logprob)
            new_tokens = token_ids

            injected_t0 = getattr(seq, "_injected_t0", None)
            if injected_t0 is not None:
                new_tokens = [injected_t0] + list(new_tokens)
                seq._injected_t0 = None

            # `draft_token_ids` is None whenever the runner skipped propose(),
            # which a P/D producer does on every step: it disables deferred
            # output so the consumer consumes T0 exactly once (KDA state would
            # otherwise advance twice). The seq is handed off rather than
            # decoded here, so it keeps its empty spec_token_ids and the
            # consumer proposes for itself.
            if self.mtp_k > 0 and draft_token_ids is not None:
                seq.spec_token_ids = draft_token_ids[idx]

            if seq.num_completion_tokens <= 3 and seq.kv_transfer_params:
                logger.debug(
                    "[PD-DECODE] seq %s: comp_tokens=%d, "
                    "new_token=%s, num_tokens=%d, blocks=%d",
                    seq.id,
                    seq.num_completion_tokens,
                    token_ids,
                    seq.num_tokens,
                    len(seq.block_table),
                )
            num_tokens = seq.num_tokens - num_placeholder_width - num_rejected
            leave_reason = None
            # Client disconnected -> finish now via the normal stop path (frees
            # KV blocks, emits a finished RequestOutput). A natural stop below
            # may still overwrite the reason; either way the seq terminates.
            if seq.status == SequenceStatus.ABORTED:
                leave_reason = "aborted"
            # MTP edge case: `rejection_sampler` does NOT inspect EOS — it
            # only compares draft vs target_argmax for acceptance. So when
            # the verified token is EOS the kernel still emits 1+ accepted
            # bonus tokens after EOS (often BOS, since the model naturally
            # starts a new sentence). Without truncating, those post-EOS
            # tokens leak into the detokenized output (e.g. "...6.<EOS><BOS>").
            # Empirically confirmed via DIAG: `token_ids=[EOS=1, BOS=0]`,
            # `eos_idx=0`, `num_new=2`, `num_rejected=0` for V4-Pro MTP-1.
            # Track the earliest stop position so `num_tokens` can drop the
            # spurious tail below.
            stop_at_idx: int | None = None
            # Check if sequence ends with any stop sequence
            for stop_seq in seq.stop_token_sequences:
                stop_len = len(stop_seq)
                if num_tokens >= stop_len:
                    is_stop = False
                    for i in range(num_new_token):
                        offset = num_tokens - i
                        if seq.token_ids[offset - stop_len : offset] == stop_seq:
                            is_stop = True
                            # `i` counts back from the last sampled token
                            # (i=0 = last). Truncate to include this stop
                            # sequence (drop everything after it).
                            stop_at_idx = num_new_token - 1 - i
                            break
                    if is_stop:
                        leave_reason = "stop_sequence"
                        break
            else:
                # Check the last token in the list for EOS
                if token_ids and not seq.ignore_eos and self.eos_token_id in token_ids:
                    leave_reason = "eos"
                    stop_at_idx = token_ids.index(self.eos_token_id)
                elif not seq.ignore_eos and any(
                    t in self.stop_token_ids for t in token_ids
                ):
                    stop_at_idx = next(
                        i for i, t in enumerate(token_ids) if t in self.stop_token_ids
                    )
                    leave_reason = f"stop_{token_ids[stop_at_idx]}"

            # ``num_tokens`` is the real post-verification length. One MTP
            # forward can accept multiple tokens, so the final batch can cross
            # max_tokens even though the request was below the cap when it was
            # scheduled. Select the earlier boundary between a natural stop
            # above and the output cap, then reuse the common truncation path
            # for both internal and client-visible tokens.
            completion_tokens = num_tokens - seq.num_prompt_tokens
            if completion_tokens >= seq.max_tokens:
                overflow = completion_tokens - seq.max_tokens
                # ``stop_at_idx`` indexes this step's model output, excluding
                # an injected P/D T0.  -1 therefore keeps T0 but no model
                # output; when the cap retains no tokens, -2 is the boundary
                # before T0.
                min_stop_at_idx = -1 - (1 if injected_t0 is not None else 0)
                max_stop_at_idx = max(min_stop_at_idx, num_new_token - 1 - overflow)
                if stop_at_idx is None or max_stop_at_idx < stop_at_idx:
                    stop_at_idx = max_stop_at_idx
                    leave_reason = "max_tokens"

            # Drop accepted-draft tokens past the stop position (MTP only —
            # for non-spec the sampler emits exactly 1 token so this is a
            # no-op).
            if stop_at_idx is not None and stop_at_idx < num_new_token - 1:
                num_tokens -= (num_new_token - 1) - stop_at_idx
                # The same truncation MUST apply to the EMITTED tokens, not just
                # the internal seq length. The client-visible text is built from
                # RequestOutput.output_tokens (an accumulation of `new_tokens`) by
                # generate_async / the streaming callback — NOT from
                # completion_token_ids (which the `seq.num_tokens` write above
                # governs). Without trimming `new_tokens` here, the post-stop
                # tokens the rejection sampler emits past EOS (it does not inspect
                # EOS) leak into the response: strict-match still finds the answer,
                # but flexible-extract's last-number picks up the leaked trailing
                # digit. `injected_t0` (if present) prepends one slot not counted
                # in stop_at_idx / num_new_token, so offset the cut by it.
                keep = stop_at_idx + 1 + (1 if injected_t0 is not None else 0)
                new_tokens = new_tokens[:keep]
                if seq.return_logprobs:
                    # LLMEngine.postprocess returns the complete logprobs
                    # array rather than slicing it through num_tokens, so keep
                    # it aligned explicitly with the cropped completion.
                    del seq.logprobs[num_tokens - seq.num_prompt_tokens :]

            # Record TTFT from the finalized retained length, after rejected
            # speculative tokens and cap/stop overflow have been removed. A
            # terminal response with no completion tokens must keep TTFT zero.
            if num_tokens - seq.num_prompt_tokens >= 1 and seq.first_token_time == 0.0:
                seq.first_token_time = time.time()

            # Counted here, not from `len(token_ids)` above: `new_tokens` is
            # what reaches RequestOutput, and it differs from the forward's
            # raw output in both directions — the truncation just above drops
            # accepted drafts past EOS/stop, and `injected_t0` prepends the
            # token the prefill process sampled. Counting the raw output made
            # the status line disagree with what the client received and with
            # `total_generation_tokens` below, which this same call derives
            # from the post-truncation length.
            num_new_generation_tokens += len(new_tokens)

            # Hash generated blocks. Deferred output: all tokens forwarded;
            # undeferred: last token not yet forwarded, so exclude it.
            self.block_manager.hash_decode_blocks(
                seq,
                num_tokens - (0 if is_deferred_out else 1),
                next_forward_tokens=self._checkpoint_room(
                    seq, leave_reason is not None
                ),
            )

            # Prepare stream output
            # A terminal event is required even when truncation leaves no
            # tokens (for example max_tokens <= 0). Async consumers wait for
            # this finished RequestOutput and would otherwise block forever.
            if stream_output_queue is not None and (
                new_tokens or leave_reason is not None
            ):
                if self.kv_connector is not None and leave_reason is not None:
                    self.kv_connector.request_finished(seq)
                output_tokens_list = (
                    list(new_tokens)
                    if isinstance(new_tokens, tuple)
                    else new_tokens.copy()
                )
                request_output = RequestOutput(
                    request_id=seq.id,
                    output_tokens=output_tokens_list,
                    finished=(leave_reason is not None),
                    finish_reason=leave_reason,
                    kv_transfer_params_output=getattr(
                        seq, "kv_transfer_params_output", None
                    ),
                    num_cached_tokens=getattr(seq, "prefix_cache_hit_tokens", 0),
                )

                if request_output.kv_transfer_params_output is not None:
                    logger.debug("KV transfer output present in stream output.")

                stream_outputs.append((seq.id, request_output))
                logger.debug(
                    f"Scheduler: Created stream output for seq_id={seq.id}, "
                    f"tokens={new_tokens}, finished={leave_reason is not None}"
                )

            if leave_reason is not None:
                # logger.info(
                #     f"Sequence {seq.id} finished with reason: {leave_reason}, {seq.token_ids[-8:]=}"
                # )
                seq.num_tokens = num_tokens
                seq.leave_reason = leave_reason
                seq.status = SequenceStatus.FINISHED
                self.total_finished_requests += 1
                self.total_prompt_tokens += int(seq.num_prompt_tokens)
                self.total_generation_tokens += max(
                    0, int(num_tokens) - int(seq.num_prompt_tokens)
                )
                finished_seqs.append(seq)

        if stream_output_queue is not None and stream_outputs:
            stream_output_queue.put_nowait(stream_outputs)

        if need_placeholder:
            # placeholder for the each decode step
            for seq in seqs:
                if seq.status == SequenceStatus.RUNNING and not seq.is_partial_prefill:
                    num = num_placeholder - seq.num_rejected
                    for _ in range(num):
                        seq.append_token(self.eos_token_id)
                        if seq.return_logprobs:
                            seq.logprobs.append(0.0)
                    seq.num_placeholder_tokens = num
        for seq in finished_seqs:
            logger.debug("Freeing blocks for finished seq %s", seq.id)
            if seq.is_partial_prefill:
                seq.is_partial_prefill = False
                self._partial_prefill_count -= 1
            if self.kv_connector is not None:
                if hasattr(self.kv_connector, "request_finished"):
                    self.kv_connector.request_finished(seq)
                if self._connector_flag("is_producer"):
                    logger.debug(
                        "Deferring block free for seq %s until KV send completes.",
                        seq.id,
                    )
                    self.deferred_free_blocks[seq.id] = seq
                elif self._connector_should_defer_free(seq):
                    logger.debug(
                        "Deferring block free for seq %s until KV save completes.",
                        seq.id,
                    )
                    # Stamp when the save was deferred so the reconciler can
                    # reclaim it if the completion report never arrives.
                    seq._deferred_save_at = time.monotonic()
                    self.deferred_free_blocks[seq.id] = seq
                else:
                    self.block_manager.deallocate(seq)
            else:
                self.block_manager.deallocate(seq)
            self.running.remove(seq)

        self.engine_stats.update_throughput(
            num_generation_tokens=num_new_generation_tokens
        )
        return finished_seqs

    def compute_detailed_aggregates(
        self,
        scheduled_batch: ScheduledBatch,
        seqs: dict[int, Sequence],
    ) -> None:
        """Attach detailed attention aggregates to *scheduled_batch* in place.

        Only the quadratic terms genuinely needed for a downstream attention-FLOP
        estimate are computed here. The request counts and total query tokens are
        already emitted by the ``prefill[]``/``decode[]`` labels in
        :meth:`ModelRunner.run_model`, so this avoids duplicating them.

        The following batch-level sums are stored on the batch and appended to
        those labels by the runner:

            sqsq  — sum of N_Q^2      (per request)
            sqsk  — sum of N_Q*N_KV   (per request)
            sk    — sum of N_KV       (per request)

        where ``N_Q`` is the number of query tokens scheduled for a request and
        ``N_KV`` is its KV length (cached + new tokens for prefill, full
        sequence length for decode). Aggregating over every request in the
        batch gives the total for that single forward, which is exactly the
        quantity a per-iteration attention-FLOP estimate needs. For MTP/spec-decode a
        decode step schedules ``mtp_k + 1`` query tokens, so the scheduled
        token count is used as ``N_Q`` for both branches (rather than a
        hardcoded 1) to avoid undercounting.

        This is a no-op (leaves the fields ``None``) unless profiling is active
        and ``ATOM_ENABLE_DETAILED_ANNOTATION`` is set.
        """
        if not self.profile_active or not self._detailed_annotation_enabled:
            return

        sqsq = 0  # sum N_Q^2
        sqsk = 0  # sum N_Q*N_KV
        sk = 0  # sum N_KV
        for seq, num_tokens in zip(seqs.values(), scheduled_batch.num_scheduled_tokens):
            # Cast to Python int: num_scheduled_tokens is np.int32, so nq*nq /
            # nq*nkv would overflow once a prefill/chunk exceeds ~46341 tokens
            # (e.g. np.int32(65536)**2 == 0), silently corrupting the estimate.
            nq = int(num_tokens)  # query tokens scheduled this forward
            if seq.type == SequenceType.DECODE:
                nkv = int(seq.num_tokens)  # full sequence length
            else:
                # PREFILL: KV length = cached + new tokens.
                nkv = int(seq.num_cached_tokens) + nq
            sqsq += nq * nq
            sqsk += nq * nkv
            sk += nkv

        scheduled_batch.detailed_sqsq = sqsq
        scheduled_batch.detailed_sqsk = sqsk
        scheduled_batch.detailed_sk = sk

    def _connector_flag(self, name: str) -> bool:
        return bool(getattr(self.kv_connector, name, False))

    @staticmethod
    def _has_req_id(req_ids: list, seq_id) -> bool:
        candidates = (seq_id, str(seq_id))
        for candidate in candidates:
            if candidate in req_ids:
                return True
        try:
            int_id = int(seq_id)
        except (TypeError, ValueError):
            return False
        return int_id in req_ids

    @staticmethod
    def _pop_req_id(req_ids: list, seq_id) -> bool:
        candidates = (seq_id, str(seq_id))
        for candidate in candidates:
            if candidate in req_ids:
                req_ids.remove(candidate)
                return True
        try:
            int_id = int(seq_id)
        except (TypeError, ValueError):
            return False
        if int_id in req_ids:
            req_ids.remove(int_id)
            return True
        return False

    def _update_waiting_for_remote_kv(self, seq: Sequence) -> bool:
        """Check whether a remote KV transfer for *seq* has completed.

        The ``finished_recving_kv_req_ids`` list is populated by
        :meth:`_update_from_kv_xfer_finished` during the previous
        scheduling step.  When ready, the sequence transitions back
        from ``WAITING_FOR_REMOTE_KVS`` to ``WAITING``.
        """
        if not self._pop_req_id(self.finished_recving_kv_req_ids, seq.id):
            return False

        logger.debug("KV transfer finished for seq %s, ready for scheduling.", seq.id)

        # Hash received prompt blocks into prefix cache so the next turn
        # transfers only the delta. Decode never runs prefill forward, so
        # this is the only place these blocks get hashed.
        bm = self.block_manager
        prefix_caching = getattr(bm, "enable_prefix_caching", False)
        kv_events = bm.kv_events_enabled
        if not (prefix_caching or kv_events):
            return True

        # num_cached_tokens is a global-token count. Under DCP each block-table
        # entry spans one virtual hash block, not one rank-local physical block.
        num_cached_blocks = seq.num_cached_tokens // bm.hash_block_size
        # PD consumer only: full prompt KV arrived via RDMA, safe to hash all.
        # Offload/LMCache path skipped — suffix KV not yet computed.
        if prefix_caching and not self._connector_flag("is_offload"):
            bm.register_received_prefix(seq)

        # Emit BlockStored(REMOTE) for delta blocks so external consumers
        # can track remote-resident KV.
        if kv_events:
            remote_hashes: list[int] = []
            remote_tokens: list[int] = []
            parent_block_hash: int | None = None
            prev_hash: int | None = None
            for i, block_id in enumerate(seq.block_table):
                blk = bm.kv.block(block_id)
                if blk.hash == -1:
                    continue
                if i < num_cached_blocks:
                    prev_hash = blk.hash
                    continue
                if not remote_hashes:
                    parent_block_hash = prev_hash
                remote_hashes.append(blk.hash)
                remote_tokens.extend(blk.token_ids)
                prev_hash = blk.hash
            if remote_hashes:
                bm.record_remote_store(
                    block_hashes=remote_hashes,
                    token_ids=remote_tokens,
                    parent_block_hash=parent_block_hash,
                )
        return True

    def _promote_ready_remote_kv_requests(self) -> None:
        """Move completed remote-KV waiters ahead of fresh admissions.

        Offload waiters already own allocated blocks. If a fresh request at the
        head cannot allocate while a completed waiter sits behind it, the waiter
        cannot finish and free blocks. Preserve FIFO order within the ready and
        blocked slots.
        """
        if not self.waiting or not (
            self.finished_recving_kv_req_ids or self.failed_recving_kv_req_ids
        ):
            return

        ready: deque[Sequence] = deque()
        blocked: deque[Sequence] = deque()
        while self.waiting:
            seq = self.waiting.popleft()
            if seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS and (
                self._has_req_id(self.finished_recving_kv_req_ids, seq.id)
                or self._has_req_id(self.failed_recving_kv_req_ids, seq.id)
            ):
                ready.append(seq)
            else:
                blocked.append(seq)

        if ready:
            self.waiting.extend(ready)
            self.waiting.extend(blocked)
        else:
            self.waiting.extend(blocked)

    def _park_ready_offload_partial_prefills(self) -> None:
        if (
            not self.running
            or self.kv_connector is None
            or not hasattr(self.kv_connector, "should_park_partial_prefill_for_load")
        ):
            return

        parked: deque[Sequence] = deque()
        keep_running: deque[Sequence] = deque()
        while self.running:
            seq = self.running.popleft()
            should_park = self.kv_connector.should_park_partial_prefill_for_load(seq)
            if should_park:
                if seq.is_partial_prefill:
                    seq.is_partial_prefill = False
                    self._partial_prefill_count -= 1
                seq.status = SequenceStatus.WAITING_FOR_REMOTE_KVS
                self._count_inflight_load(seq)
                parked.append(seq)
            else:
                keep_running.append(seq)

        self.running = keep_running
        if parked:
            self.waiting.extendleft(reversed(parked))

    def _update_from_kv_xfer_finished(self, kv_connector_output: KVConnectorOutput):
        """Reconcile scheduler state with completed KV transfers.

        * ``finished_recving``: marks requests as ready for decode scheduling.
        * ``finished_sending``: releases deferred block allocations on the
          producer side.
        """
        if kv_connector_output is None:
            return

        is_producer = self._connector_flag("is_producer")
        is_offload = self._connector_flag("is_offload")

        process_completions = getattr(self.kv_connector, "process_completions", None)
        if callable(process_completions):
            kv_connector_output = process_completions(kv_connector_output)

        for req_id in kv_connector_output.finished_recving or ():
            assert not is_producer, "Only consumer should update recving KV status"
            logger.debug("Finished recving KV transfer for request %s", req_id)
            self.finished_recving_kv_req_ids.append(req_id)

        for req_id in kv_connector_output.failed_recving or ():
            assert not is_producer, "Only consumer should update failed KV recv status"
            logger.warning(
                "KV receive failed for request %s; falling back to prefill.", req_id
            )
            self.failed_recving_kv_req_ids.append(req_id)

        # The two loading channels carry state-tier loads as well as KV ones,
        # which buys the state leg the aggregator's per-request quorum for free.
        # They cannot be confused: a KV load is refused for any sequence with a
        # per-request cache, and a state load exists only for such a sequence.
        for req_id in kv_connector_output.finished_loading or ():
            assert is_offload, "Only offload connector should update loading KV status"
            logger.debug("Finished offload KV load for request %s", req_id)
            # Ahead of the abort guard: an aborted request's state group has to
            # be released too, and the guard takes the `continue`.
            self._settle_state_load(req_id, ok=True)
            if self._finish_aborted_load_cleanup(req_id):
                continue
            self.finished_recving_kv_req_ids.append(req_id)

        # A joint load can fail on its KV leg alone while its state H2D landed
        # intact (an LMCache LRU miss dropping the KV chunk). The connector
        # advertises those survivors on a failure-dominant, TP-quorumed
        # disposition channel, drained here on the same step the KV failure
        # report arrives: a survivor is *abandoned* (pending released, hash
        # kept loadable) rather than *failed* (hash forgotten). Everything else
        # -- a genuine state-leg miss, or a connector that never advertises the
        # channel -- still settles as a failure.
        state_survived = getattr(self.kv_connector, "take_state_load_survived", None)
        state_survived = state_survived() if state_survived is not None else set()
        for req_id in kv_connector_output.failed_loading or ():
            assert (
                is_offload
            ), "Only offload connector should update failed KV load status"
            logger.warning(
                "Offload KV load failed for request %s; falling back to prefill.",
                req_id,
            )
            if req_id in state_survived:
                self._abandon_state_load(req_id)
            else:
                self._settle_state_load(req_id, ok=False)
            if self._finish_aborted_load_cleanup(req_id):
                continue
            self.failed_recving_kv_req_ids.append(req_id)

        finished_saving = kv_connector_output.finished_saving or ()
        for req_id in kv_connector_output.finished_sending or ():
            assert (
                self.kv_connector.is_producer
            ), "Only producer should free blocks after sending KV"
            logger.debug("Finished sending KV transfer for request %s", req_id)
            seq = self._deferred_sequence(req_id)
            if seq is None:
                # Already reclaimed by `_reconcile_stalled_deferred_saves` after
                # a stall; a late completion report has nothing left to free.
                continue
            self.deferred_free_blocks.pop(seq.id, None)
            self.block_manager.deallocate(seq)

        if not is_producer:
            for req_id in finished_saving:
                seq = self._deferred_sequence(req_id)
                if seq is not None:
                    self._maybe_release_deferred(seq)

        # The state-offload store reports, not keyed by request: by the time a
        # store lands its owner is long gone and only the hash remains. They
        # ride the connector's completion channels, so they are drained from the
        # connector with TP quorum already taken.
        #
        # Indexed only on the report, never at submission: a hash advertised
        # before its bytes exist parks the next request over that prefix
        # against a `get` that must miss.
        bm = getattr(self, "block_manager", None)
        offload = getattr(bm, "state_offload", None)
        take = getattr(self.kv_connector, "take_state_reports", None)
        if offload is not None and take is not None:
            indexed, failed = take()
            # Both settle the pin; only success indexes the hash. A hash whose
            # bytes are not really there must never be voted for, and a pin
            # exists to keep the source still during the copy -- which is over
            # either way.
            # The source release first, and separately: it hands the PAGE
            # units back the moment the GPU stopped reading them, which is
            # before the CPU put has been decided. Waiting for the store
            # report would keep a whole image out of the pool across an
            # operation that cannot touch it.
            release = getattr(self.kv_connector, "take_state_source_releases", None)
            if release is not None:
                for op in release():
                    bm.release_state_store_source(op)
            for op in indexed:
                bm.settle_state_store(op, ok=True)
            for op in failed:
                bm.settle_state_store(op, ok=False)
            # Unit-side twin of `_reconcile_stalled_deferred_saves`, on the same
            # window and for the same reason: the pin lives in this process
            # while the D2H runs in the worker, so a report lost to a crashed
            # worker or a dropped completion would hold a whole image out of
            # the pool forever. Recovery is total -- a leaked pin breaks no
            # `BlockPool` invariant -- so zeroing the count restores the record
            # exactly.
            bm.reclaim_stale_state_store_pins(self._save_abandon_timeout_s())
            # Load-side twin of the same defence: `deallocate` parks a slot when
            # it tears down a request whose state load is still in flight, on the
            # promise that `settle_state_load` will hand it back. A crashed
            # worker or dropped completion breaks that promise and the slot sits
            # off the free list forever, wedging `can_allocate`'s state gate.
            # Same window, same "cannot tell lost from slow" caveat as the pins.
            bm.reconcile_orphan_load_slots(self._save_abandon_timeout_s())

    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        return len(self.running), len(self.waiting)

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running)

    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests in the scheduler's
        internal queue."""
        return self.get_num_unfinished_requests() > 0

    def has_requests(self) -> bool:
        """Returns True if there are unfinished requests, or finished requests
        not yet returned in SchedulerOutputs."""
        return self.has_unfinished_requests()

    def get_next_batch_info(self) -> tuple[bool, int, int]:
        # Check for partial prefills in running (chunked prefill resume)
        for seq in self.running:
            if seq.num_cached_tokens < seq.num_tokens:
                remaining = seq.num_tokens - seq.num_cached_tokens
                chunk = min(remaining, self.max_num_batched_tokens)
                return (True, chunk, 1)
        # Only consider waiting seqs that are not blocked on a remote KV
        # transfer (P/D disaggregation) when deciding if we can prefill.
        eligible_waiting = [
            seq
            for seq in self.waiting
            if seq.status != SequenceStatus.WAITING_FOR_REMOTE_KVS
        ]
        if eligible_waiting:
            # new request is waiting, will do prefill
            num_reqs = 0
            total_tokens = 0
            for seq in eligible_waiting:
                tokens = seq.num_tokens - seq.num_cached_tokens
                if self.enable_chunked_prefill:
                    tokens = min(tokens, self.max_num_batched_tokens - total_tokens)
                if total_tokens + tokens > self.max_num_batched_tokens:
                    break
                if num_reqs >= self.max_num_seqs:
                    break
                total_tokens += tokens
                num_reqs += 1
            return (True, total_tokens, num_reqs)
        elif self.running:
            # decode
            num_tokens = len(self.running)
            return (False, num_tokens, num_tokens)
        else:
            # No requests
            return (False, 0, 0)

    def _passed_delay(self, now: float) -> bool:
        # borrowed from https://github.com/vllm-project/vllm/pull/3279
        # if the earliest arrived request has waited long enough,
        # i.e., > delay_factor * last_prompt_latency (the latency of last prefill in unit of seconds),
        # new prefill should be scheduled immediately
        if self.prev_prompt:
            self.last_prompt_latency = now - self.prev_time
        self.prev_time, self.prev_prompt = now, False
        # Delay scheduling prompts to let waiting queue fill up
        if self.delay_factor > 0 and self.waiting:
            earliest_arrival_time = min([seq.arrive_time for seq in self.waiting])
            passed_delay = (now - earliest_arrival_time) > (
                self.delay_factor * self.last_prompt_latency
            ) or not self.running
        else:
            passed_delay = True
        return passed_delay


class PrefillScheduler:
    """Scheduler for the disaggregated prefill process.

    Key differences from the base Scheduler:
    - No BlockManager: KV blocks are pre-assigned by DecodeEngineCore and
      written into seq.block_table before schedule() is called.
    - schedule() only runs sequences that already have a non-empty block_table.
      Sequences still waiting for a BlockAssignment message stay in waiting.
    - postprocess() is a no-op: prefill produces no sampled tokens.
    - Decode scheduling is never performed.
    """

    # Every request here is also held by the decode side, whose queues span
    # the full lifetime; the aggregator drops this rank's counts so an
    # in-flight prefill is not counted on both. See `_METRICS_ROLE` on
    # `Scheduler`.
    _METRICS_ROLE = "prefill"

    def __init__(self, config: Config, disagg_cu_shm_name: str = ""):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.block_manager = None  # blocks managed by decode process
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        # spec decode not used on prefill side
        self.use_spec = False
        self.spec_decode_local = False
        self.mtp_k = 0
        # Only the throughput section applies to the prefill side: it does not
        # speculate, and it has no BlockManager to source prefix-cache hits.
        # Throughput follows config.enable_log_stats (default True), matching
        # the aggregated Scheduler and the decode side.
        parallel_cfg = getattr(config, "parallel_config", None)
        dp_rank = (
            getattr(parallel_cfg, "data_parallel_rank", None)
            if parallel_cfg is not None
            else None
        )
        self.engine_stats = EngineStats(
            engine_index=dp_rank or 0,
            label="Prefill ",
            enable_log_stats=config.enable_log_stats,
            throughput_log_interval_s=config.throughput_log_interval,
        )
        self.total_prompt_tokens = 0
        self.total_generation_tokens = 0
        self.total_finished_requests = 0
        self.total_preemptions = 0

        # Shared memory for dynamic CU partitioning.
        # Layout: [0:4] = decode_tokens (uint32)
        self._cu_shm = None
        if disagg_cu_shm_name:
            import multiprocessing.shared_memory

            self._cu_shm = multiprocessing.shared_memory.SharedMemory(
                name=disagg_cu_shm_name, create=False
            )
            logger.info("initialized shared memory")
        self._pending_lock = threading.Lock()

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def has_requests(self) -> bool:
        return bool(self.waiting) or bool(self.running)

    def publish_kv_events(self) -> None:
        # No-op: disagg prefill has no BlockManager — the decode process owns the
        # KV blocks and emits KV events. Defined so EngineCore.busy_loop teardown
        # (which calls this on every scheduler) works for PrefillScheduler.
        pass

    def shutdown_kv_events(self) -> None:
        # No-op: see publish_kv_events.
        pass

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def extend(self, seqs: list):
        self.waiting.extend(seqs)

    def schedule(self):
        """Run a scheduling pass and close the throughput window.

        Override `_schedule`, not this — see `Scheduler.schedule` for why the
        tick lives at the one entry point instead of at each early return.
        """
        result = self._schedule()
        self._record_throughput(num_prompt_tokens=_prompt_tokens_of(result))
        return result

    def _schedule(self):
        """Schedule only sequences whose block_table has been populated.

        Sequences that do not yet have a block assignment (block_table is
        empty) remain in the waiting queue and will be reconsidered on the
        next call.

        Returns (ScheduledBatch, dict[seq_id, Sequence]) or (None, {}) when
        no sequence is ready.
        """
        scheduled_seqs = {}
        num_scheduled_tokens = []
        num_batched_tokens = 0
        num_seqs = 0

        with self._pending_lock:
            # Collect ready sequences (have received BlockAssignment from decode)
            ready = [s for s in self.waiting if s.block_table]

            for seq in ready:
                if num_seqs >= self.max_num_seqs:
                    break
                num_new_tokens = seq.num_tokens - seq.num_cached_tokens
                if num_batched_tokens + num_new_tokens > self.max_num_batched_tokens:
                    break
                self.waiting.remove(seq)
                seq.status = SequenceStatus.RUNNING
                seq.type = SequenceType.PREFILL
                self.running.append(seq)
                scheduled_seqs[seq.id] = seq
                num_scheduled_tokens.append(num_new_tokens)
                num_batched_tokens += num_new_tokens
                num_seqs += 1

        if not scheduled_seqs:
            return None, {}

        # Read the decode tokens from decode process via shared memory.

        cu_fraction = None

        if self._cu_shm is not None:
            decode_tokens = struct.unpack_from("I", self._cu_shm.buf, 0)[0]
            cu_fraction = _optimal_cu_fraction(decode_tokens, num_batched_tokens)

        return (
            ScheduledBatch(
                seqs=scheduled_seqs,
                num_scheduled_tokens=num_scheduled_tokens,
                total_tokens_num=num_batched_tokens,
                total_tokens_num_prefill=num_batched_tokens,
                total_seqs_num=num_seqs,
                total_seqs_num_prefill=num_seqs,
                cu_stream_fraction=cu_fraction,
            ),
            scheduled_seqs,
        )

    def _record_throughput(
        self, num_prompt_tokens: int = 0, num_generation_tokens: int = 0
    ) -> None:
        """`Scheduler._record_throughput`'s counterpart for the prefill side.

        This process schedules no decode, so generation stays 0, and the
        decode process owns the KV blocks — no local BlockManager means
        `kv_usage=None`, which the line reports as `n/a` rather than as a
        real, empty pool. Same `window_expired` gate as the base class, and
        for the same reason.
        """
        stats = self.engine_stats
        if not stats.throughput_enabled:
            return
        stats.update_throughput(num_prompt_tokens, num_generation_tokens)
        if not stats.window_expired(time.monotonic()):
            return
        num_running_reqs, num_waiting_reqs = self.get_request_counts()
        stats.maybe_log_throughput(
            num_running_reqs=num_running_reqs,
            num_waiting_reqs=num_waiting_reqs,
            kv_usage=None,
        )

    def heartbeat_throughput(self, now: float) -> None:
        """Idle-pass counterpart of `Scheduler.heartbeat_throughput`."""
        if self.engine_stats.window_expired(now):
            self._record_throughput()

    def postprocess(self, seqs, fwd_output, stream_output_queue=None) -> list:
        """No-op: prefill produces no sampled tokens."""
        return []

    def get_next_batch_info(self) -> tuple:
        if self.waiting:
            seq = self.waiting[0]
            return (True, seq.num_tokens - seq.num_cached_tokens)
        return (False, 0)

    def get_request_counts(self) -> tuple[int, int]:
        """(running, waiting). Not a `Scheduler` subclass, so this is declared
        rather than inherited — the engine-status line and any metrics reader
        expect every scheduler to answer it."""
        return len(self.running), len(self.waiting)

    def get_num_unfinished_requests(self) -> int:
        return sum(self.get_request_counts())


class DecodeScheduler(Scheduler):
    """Scheduler for the disaggregated decode process.

    Manages 3 queues:
    - waiting:         new requests pending block allocation
    - prefill_waiting: blocks allocated, BlockAssignment sent, awaiting PrefillDone
    - running:         ongoing decode sequences

    Block allocation is separated from scheduling: allocate_waiting() is called
    by DecodeEngineCore after draining the input queue, and returns newly
    allocated sequences so the engine can send BlockAssignment to prefill.

    on_prefill_done() promotes sequences directly from prefill_waiting to
    running.  schedule() only schedules the running queue as decode batches.
    """

    _ENGINE_LABEL = "Decode "
    _METRICS_ROLE = "decode"

    def get_request_counts(self) -> tuple[int, int]:
        """Fold in the two queues this scheduler adds.

        `allocate_waiting()` drains `waiting` almost immediately, so a request
        spends most of its life in `prefill_waiting` (blocks assigned, awaiting
        PrefillDone) and lands in `prefill_done` before `schedule()` promotes
        it. Counting only the base pair reports `Running: 0, Waiting: 0` on an
        engine holding a full load of in-flight requests — on the status line
        and, because `/metrics` reads this same method, on the dashboard too.
        """
        return (
            len(self.running) + len(self.prefill_done),
            len(self.waiting) + len(self.prefill_waiting),
        )

    def __init__(
        self,
        config: Config,
        disagg_cu_shm_name: str = "",
        *,
        state_runtime: StateRuntime = DEFAULT_STATE_RUNTIME,
    ):
        super().__init__(
            config,
            state_runtime=state_runtime,
        )
        # seq_id → Sequence; blocks allocated, BlockAssignment sent, awaiting PrefillDone.
        self.prefill_waiting: dict[int, Sequence] = {}
        self.prefill_done: deque[Sequence] = deque()
        # Shared memory for dynamic CU partitioning.
        self._cu_shm = None
        if disagg_cu_shm_name:
            import multiprocessing.shared_memory

            self._cu_shm = multiprocessing.shared_memory.SharedMemory(
                name=disagg_cu_shm_name, create=False
            )
            struct.pack_into("I", self._cu_shm.buf, 0, 0)

        # Protects prefill_waiting and running: on_prefill_done is called
        # from the _recv_prefill_done background thread.
        self._prefill_lock = threading.Lock()
        self.cu_fraction: float | None = None

    def is_finished(self) -> bool:
        # Kept explicit rather than derived from get_request_counts: unlike the
        # base it deliberately ignores `_rejected` and `deferred_free_blocks`.
        # If a queue is ever added to this scheduler it has to be listed here
        # *and* in get_request_counts.
        return (
            not self.waiting
            and not self.prefill_waiting
            and not self.running
            and not self.prefill_done
        )

    def get_num_unfinished_requests(self) -> int:
        # Derived, so the queue inventory lives in get_request_counts alone.
        # `has_requests` / `has_unfinished_requests` come off the base's chain
        # through this method and need no override of their own.
        return sum(self.get_request_counts())

    def allocate_waiting(self) -> list[Sequence]:
        """Allocate KV blocks for sequences in waiting; move them to prefill_waiting.

        Returns newly allocated sequences so DecodeEngineCore can send a
        BlockAssignment message to the prefill process for each one.
        Called from the main busy_loop thread only.
        """
        newly_allocated = []
        while self.waiting:
            seq = self.waiting[0]
            with self._prefill_lock:
                if self.block_manager.can_allocate(seq) < 0:
                    logger.warning("Cannot allocate prefill")
                    break
                if not self.block_manager.allocate(seq):
                    # Disown could not be backed (finding #2); leave the seq at
                    # the front of waiting to retry once the pool has room.
                    self.block_manager.deallocate(seq)
                    break
            self.waiting.popleft()

            self.prefill_waiting[seq.id] = seq
            newly_allocated.append(seq)
        return newly_allocated

    def on_prefill_done(
        self, seq_id: int, num_tokens_computed: int, sampled_token_id: int
    ) -> None:
        """Promote a sequence from prefill_waiting directly to running.

        Called from the _recv_prefill_done background thread.
        sampled_token_id is the first generated token sampled by the prefill
        process; it is appended here so that context_lens and slot_mapping
        match the non-disagg postprocess state before the first decode step.
        """

        seq = self.prefill_waiting.pop(seq_id, None)
        if seq is not None:
            seq.num_cached_tokens = num_tokens_computed
            seq.append_token(sampled_token_id)
            seq.first_token_time = time.time()
            self.prefill_done.append(seq)

    def _schedule(self):
        """Schedule decode-only batches.

        Sequences are promoted directly from prefill_waiting to running by
        on_prefill_done(); this method only schedules the running queue.

        Overrides the base `_schedule`, so the inherited `schedule()` still
        closes the throughput window on every one of the returns below.
        """

        # This override does not call `super()._schedule()`, but it does route
        # through the same `block_manager.allocate` and the same `postprocess`,
        # so it owes the state pool the same two hooks. Without this one the
        # pins taken by every resume accumulate forever and admission starves.
        self.block_manager.complete_previous_state_batch()

        prefill_finished = False
        while self.prefill_done:
            seq = self.prefill_done.popleft()
            seq.status = SequenceStatus.RUNNING
            seq.type = SequenceType.DECODE
            # Append the first generated token sampled by the prefill process.
            # In non-disagg mode, Scheduler.postprocess() does this after the
            # prefill forward (is_deferred_out=True always appends one placeholder
            # that is later overwritten with the real token from the async queue).
            # In disagg mode the prefill process ran sampling and sent us the real
            # token; appending it here puts num_tokens, context_lens, and
            # slot_mapping in the same state as non-disagg before the first decode
            # step.
            self.running.append(seq)
            prefill_finished = True

        if not self.running:
            self.cu_fraction = None
            if self._cu_shm is not None:
                struct.pack_into("I", self._cu_shm.buf, 0, 0)
            return None

        scheduled_seqs: dict[int, Sequence] = {}
        num_scheduled_tokens: list[int] = []
        scheduled_spec_decode_tokens: dict[int, np.ndarray] = {}

        with self._prefill_lock:
            while self.running and len(scheduled_seqs) < self.max_num_seqs:
                seq = self.running.popleft()
                # logger.warning("decode state: waiting=%d prefill_waiting=%d prefill_done=%d running=%d free_blocks=%d",
                #     len(self.waiting), len(self.prefill_waiting), len(self.prefill_done),
                #     len(self.running), self.block_manager.kv.num_free)
                while not self.block_manager.can_append(seq, self.mtp_k + 1):
                    logger.warning("Cannot allocate")
                    if self.running:
                        self.preempt(self.running.pop())
                    else:
                        self.preempt(seq)
                        break
                else:
                    if self.spec_decode_local and seq.spec_token_ids.size > 0:
                        scheduled_spec_decode_tokens[seq.id] = seq.spec_token_ids
                    num_new_tokens = self.mtp_k + 1
                    self.block_manager.may_append(seq, num_new_tokens)
                    scheduled_seqs[seq.id] = seq
                    seq.type = SequenceType.DECODE
                    num_scheduled_tokens.append(num_new_tokens)

        if not scheduled_seqs:
            self.cu_fraction = None
            if self._cu_shm is not None:
                struct.pack_into("I", self._cu_shm.buf, 0, 0)
            return None

        self.running.extendleft(reversed(scheduled_seqs.values()))

        total_tokens_num_decode = sum(num_scheduled_tokens)

        # Dynamic CU partitioning: decode decides the fraction based on its
        # batch size and the total tokens queued for prefill, then writes the
        # decode tokens to shared memory for PrefillScheduler to read.

        if self._cu_shm is not None:
            struct.pack_into("I", self._cu_shm.buf, 0, total_tokens_num_decode)
            if prefill_finished:
                pwait = sum(seq.num_tokens for seq in self.prefill_waiting.values())
                self.cu_fraction = _optimal_cu_fraction(total_tokens_num_decode, pwait)

        return (
            ScheduledBatch(
                seqs=scheduled_seqs,
                num_scheduled_tokens=num_scheduled_tokens,
                total_tokens_num=total_tokens_num_decode,
                total_tokens_num_decode=total_tokens_num_decode,
                total_seqs_num=len(scheduled_seqs),
                total_seqs_num_decode=len(scheduled_seqs),
                num_spec_step=self.mtp_k if self.spec_decode_local else 0,
                scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
                cu_stream_fraction=self.cu_fraction,
                state_maintenance_ops=self.block_manager.take_state_maintenance_ops(),
            ),
            scheduled_seqs,
        )

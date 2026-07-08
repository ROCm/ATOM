# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Scheduling logic for batching prefill and decode requests.

This module provides:

- :class:`SpecStats`: Tracks speculative-decoding acceptance rates.
- :class:`ScheduledBatch`: A frozen snapshot of sequences selected for the
  next forward pass, together with their block tables and metadata.
- :class:`ScheduledBatchOutput`: Token-level outputs from a completed batch.
- :class:`Scheduler`: The main scheduling loop that manages *waiting* and
  *running* queues, coordinates block allocation, and integrates with the
  KV disaggregation connector for remote prefill/decode.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Optional

import numpy as np

from atom.config import Config
from atom.kv_transfer.disaggregation import KVConnectorOutput
from atom.model_engine.block_manager import BlockManager
from atom.model_engine.request import RequestOutput
from atom.model_engine.sequence import Sequence, SequenceStatus, SequenceType

logger = logging.getLogger("atom")


class SpecStats:
    """Tracks speculative decoding acceptance statistics."""

    __slots__ = (
        "mtp_k",
        "total_draft_tokens",
        "distribution",
        "_log_interval",
        "_interval_draft_tokens",
        "_interval_distribution",
    )

    def __init__(self, mtp_k: int, log_interval: int = 1000):
        self.mtp_k = mtp_k
        # Log every log_interval decode steps (in terms of draft tokens)
        self._log_interval = log_interval * mtp_k
        self.total_draft_tokens: int = 0
        self.distribution: dict[int, int] = {k: 0 for k in range(mtp_k + 1)}
        # Per-interval tracking
        self._interval_draft_tokens: int = 0
        self._interval_distribution: dict[int, int] = {k: 0 for k in range(mtp_k + 1)}

    def update(self, num_accepted_tokens: int) -> None:
        """Record acceptance result for one sequence in one decode step."""
        self.total_draft_tokens += self.mtp_k
        self._interval_draft_tokens += self.mtp_k
        num_bonus = num_accepted_tokens - 1
        self.distribution[num_bonus] += 1
        self._interval_distribution[num_bonus] += 1

        if self.total_draft_tokens % self._log_interval == 0:
            self._log()
            self._reset_interval()

    @property
    def total_accepted(self) -> int:
        """Total number of accepted bonus tokens across all steps."""
        return sum(k * v for k, v in self.distribution.items())

    @property
    def total_steps(self) -> int:
        """Total number of decode steps recorded."""
        return sum(self.distribution.values())

    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accepted / self.total_draft_tokens

    def get_statistics(self) -> dict:
        """Return a summary dict compatible with engine_core reporting."""
        return {
            "total_draft_tokens": self.total_draft_tokens,
            "total_accepted_tokens": self.total_accepted,
            "acceptance_rate": self.acceptance_rate,
            "distribution": dict(self.distribution),
        }

    def reset(self) -> None:
        self.total_draft_tokens = 0
        self.distribution = {k: 0 for k in range(self.mtp_k + 1)}
        self._reset_interval()

    def _reset_interval(self) -> None:
        self._interval_draft_tokens = 0
        self._interval_distribution = {k: 0 for k in range(self.mtp_k + 1)}

    def _log(self) -> None:
        ts = self.total_steps
        if ts == 0:
            return
        # Interval stats
        iv_steps = sum(self._interval_distribution.values())
        if iv_steps == 0:
            self._reset_interval()
            return
        iv_accepted = sum(k * v for k, v in self._interval_distribution.items())
        iv_rate = (
            iv_accepted / self._interval_draft_tokens
            if self._interval_draft_tokens > 0
            else 0.0
        )
        logger.info(
            f"[MTP Stats Interval] Average toks/fwd: {1 + iv_accepted / iv_steps:.2f}, "
            f"Accepted/Total Draft tokens: {iv_accepted}/{self._interval_draft_tokens}, "
            f"Acceptance rate: {iv_rate:.2%}, "
            f"Accepted tokens distribution: { {k: f'{v / iv_steps:.2%}' for k, v in self._interval_distribution.items()} }"
        )
        logger.info(
            f"[MTP Stats         ] Average toks/fwd: {1 + self.total_accepted / ts:.2f}, "
            f"Accepted/Total Draft tokens: {self.total_accepted}/{self.total_draft_tokens}, "
            f"Acceptance rate: {self.acceptance_rate:.2%}, "
            f"Accepted tokens distribution: { {k: f'{v / ts:.2%}' for k, v in self.distribution.items()} }"
        )


class CacheStats:
    """Tracks prefix caching hit statistics."""

    __slots__ = (
        "_log_interval",
        "total_requests",
        "total_cached_tokens",
        "total_full_tokens",
        "_interval_requests",
        "_interval_cached_tokens",
        "_interval_full_tokens",
    )

    def __init__(self, log_interval: int = 100):
        self._log_interval = log_interval
        self.total_requests: int = 0
        self.total_cached_tokens: int = 0
        self.total_full_tokens: int = 0
        self._interval_requests: int = 0
        self._interval_cached_tokens: int = 0
        self._interval_full_tokens: int = 0

    def update(self, num_cached_tokens: int, num_full_tokens: int) -> None:
        """Record cache stats for one prefill sequence."""
        self.total_requests += 1
        self.total_cached_tokens += num_cached_tokens
        self.total_full_tokens += num_full_tokens
        self._interval_requests += 1
        self._interval_cached_tokens += num_cached_tokens
        self._interval_full_tokens += num_full_tokens

        if self.total_requests % self._log_interval == 0:
            self._log()
            self._reset_interval()

    @property
    def hit_rate(self) -> float:
        if self.total_full_tokens == 0:
            return 0.0
        return self.total_cached_tokens / self.total_full_tokens

    def _reset_interval(self) -> None:
        self._interval_requests = 0
        self._interval_cached_tokens = 0
        self._interval_full_tokens = 0

    def _log(self) -> None:
        iv_rate = (
            self._interval_cached_tokens / self._interval_full_tokens
            if self._interval_full_tokens > 0
            else 0.0
        )
        logger.info(
            f"[Cache Stats Interval] Reqs: {self._interval_requests}, "
            f"Cached/Total tokens: {self._interval_cached_tokens}/{self._interval_full_tokens}, "
            f"Hit rate: {iv_rate:.2%}"
        )
        logger.info(
            f"[Cache Stats         ] Reqs: {self.total_requests}, "
            f"Cached/Total tokens: {self.total_cached_tokens}/{self.total_full_tokens}, "
            f"Hit rate: {self.hit_rate:.2%}"
        )


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
        remote_kv_block_ids: list[int] | None = None,
        remote_kv_seq_blocks: dict[int, list[int]] | None = None,
        num_cached_tokens: list[int] | None = None,
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
        self.context_lens = np.asarray(
            [seq.num_tokens for seq in seqs.values()], dtype=np.int32
        )
        self.num_rejected = np.asarray(
            [seq.num_rejected for seq in seqs.values()], dtype=np.int32
        )
        self.num_bonus = np.asarray(
            [seq.num_bonus_tokens for seq in seqs.values()], dtype=np.int32
        )
        self.per_req_cache_groups = [
            seq.per_req_cache_group
            for seq in seqs.values()
            if seq.has_per_req_cache and seq.per_req_cache_group >= 0
        ]
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
        self.has_mrope = bool(self.mrope_positions_by_req)

        # num_cached_tokens for chunked prefill support
        self.num_cached_tokens = (
            num_cached_tokens
            if num_cached_tokens is not None
            else [seq.num_cached_tokens for seq in seqs.values()]
        )

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

        # Compute token offsets: prefill uses num_cached_tokens, decode uses existing formula
        self.scheduled_tokens = np.empty(total_tokens_num, dtype=np.int32)
        pos = 0
        for i, (seq, num) in enumerate(zip(seqs.values(), num_scheduled_tokens)):
            if seq.type == SequenceType.PREFILL:
                offset = self.num_cached_tokens[i]
            else:
                offset = seq.num_tokens - self.num_rejected[i] - num
            self.scheduled_tokens[pos : pos + num] = seq.token_ids[
                offset : offset + num
            ]
            pos += num

        if num_spec_step > 0 and scheduled_spec_decode_tokens is not None:
            self.scheduled_spec_decode_tokens = np.asarray(
                list(scheduled_spec_decode_tokens.values()), dtype=np.int32
            )
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
        num_rejected: Optional[np.ndarray],
        num_bonus: Optional[np.ndarray],
        draft_token_ids: Optional[np.ndarray],
        is_deferred_out: bool = False,
        is_prev_prefill=False,
        logprobs=None,
    ):
        self.req_ids = req_ids
        self.token_ids = token_ids
        self.draft_token_ids = draft_token_ids
        self.num_rejected = num_rejected
        self.num_bonus = num_bonus
        self.is_deferred_out = is_deferred_out
        self.is_prev_prefill = is_prev_prefill
        self.logprobs = logprobs  # Optional[dict[int, float]]
        # O(1) lookup: req_id -> index (lazy-built on first access)
        self._req_id_to_idx: Optional[dict[int, int]] = None

    def get_idx(self, req_id: int) -> Optional[int]:
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

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.long_prefill_token_threshold = config.long_prefill_token_threshold
        self.max_model_len = config.max_model_len
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.stop_token_ids = config.stop_token_ids
        self.block_manager = BlockManager(config)
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

        # Scheduling delay for batching efficiency
        self.prev_time = 0.0
        # Did we schedule a prompt at previous step?
        self.prev_prompt = False
        # Latency of the last prompt step
        self.last_prompt_latency = 0.0
        self.delay_factor = config.scheduler_delay_factor
        self._prefill_hold_passes = 0
        self._prefill_hold_max_passes = 30
        _pc = getattr(config, "parallel_config", None)
        self._prefill_gate_enabled = getattr(_pc, "data_parallel_size", 1) > 1

        # Speculative decoding
        self.use_spec = config.speculative_config is not None
        self.mtp_k: int = (
            config.speculative_config.num_speculative_tokens if self.use_spec else 0
        )  # type: ignore
        self.spec_stats: Optional[SpecStats] = (
            SpecStats(mtp_k=self.mtp_k) if self.use_spec else None
        )
        self.cache_stats: Optional[CacheStats] = (
            CacheStats() if config.enable_prefix_caching else None
        )
        self.enable_chunked_prefill = config.enable_chunked_prefill
        # V4 SWA correctness on a prefix-cache hit. V4's sliding-window state is
        # a per-request ring (NOT shared across blocks), so on a hit the new
        # request's ring is empty and a tail token whose SWA window reaches back
        # into the cached region would read garbage. Fix: pull the forward start
        # back by enough whole blocks that the last `window` cached tokens are
        # re-forwarded, repopulating the ring. The compressed-KV hit (n_committed
        # = context_len // ratio) is UNAFFECTED because context_lens =
        # num_cached + num_scheduled is invariant under this shift. Only V4 needs
        # this; 0 disables (e.g. once SWA gets per-block shared storage, "fix C").
        # See docs / plan: V4 prefix cache fix B'.
        # NOTE: V4 detection must use `architectures`, not `model_type` — the
        # config registry maps "deepseek_v4" -> "deepseek_v3" so model_type
        # reads as v3 (same reason config.py:1118 uses architectures).
        self._v4_swa_warmup_blocks = 0
        _hf = getattr(config, "hf_config", None)
        _arches = getattr(_hf, "architectures", None) or []
        _is_v4 = any("DeepseekV4" in str(a) for a in _arches)
        if config.enable_prefix_caching and _is_v4:
            import math as _math

            window = int(getattr(_hf, "sliding_window", 128) or 128)
            # The SWA ring's physical stride is `win_with_spec = window + mtp_k`
            # (MTP draft tokens get their own ring slots). A tail token's window
            # can reach back `win_with_spec - 1` ring slots, so the re-forwarded
            # region must cover that many tokens — not just `window`. Verified:
            # with mtp_k=1, rolling back only `window`(128) leaves the deepest
            # reach-back (r=4) reading one stale slot -> DIVERGE.
            mtp_k = (
                config.speculative_config.num_speculative_tokens
                if config.speculative_config is not None
                else 0
            )
            win_with_spec = window + int(mtp_k or 0)
            self._v4_swa_warmup_blocks = _math.ceil(
                win_with_spec / self.block_manager.block_size
            )
        # Number of running seqs currently mid-prefill (per-seq state lives in
        # `Sequence.is_partial_prefill`). Maintained as a counter so Phase 1
        # of `schedule()` can skip the running-queue scan entirely on
        # pure-decode steps (the common case).
        self._partial_prefill_count: int = 0

        from atom.utils.forward_context import get_kvconnector

        self.kv_connector = get_kvconnector("scheduler", config)

        from atom.distributed.kv_events import (
            EventPublisher as _EventPublisher,
            make_publisher as _make_publisher,
        )

        kv_events_cfg = getattr(config, "kv_events_config", None)
        parallel_cfg = getattr(config, "parallel_config", None)
        dp_rank = (
            getattr(parallel_cfg, "data_parallel_rank", None)
            if parallel_cfg is not None
            else None
        )
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

        self.prefill_delayer: Optional[PrefillDelayer] = None

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
            if self.block_manager.can_allocate(seq) < 0:
                return False  # KV-pressured: definitely cannot prefill
            return True
        return False

    def _waiting_prefill_tokens(self) -> int:
        """Sum of admissible new-prefill tokens sitting in the waiting queue,
        capped at max_num_batched_tokens (we only care whether a *dense* batch
        can be filled). Skips unschedulable / remote-KV-waiting entries and
        clamps each seq to the chunked-prefill / budget limit, mirroring the
        Phase-2 admission math so the count reflects what would actually pack
        into one prefill step."""
        total = 0
        cap = self.max_num_batched_tokens
        for seq in self.waiting:
            if self._unschedulable_reason(seq) is not None:
                continue
            if seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS:
                continue
            n = seq.num_tokens - seq.num_cached_tokens
            if (
                self.enable_chunked_prefill
                and 0 < self.long_prefill_token_threshold < n
            ):
                n = self.long_prefill_token_threshold
            if n > cap:
                n = cap
            total += n
            if total >= cap:
                return cap
        return total

    def _waiting_is_dense(self) -> bool:
        """Shared density predicate: the waiting queue holds a full dense batch's
        worth of prefill tokens (>= max_num_batched_tokens). Single source of
        truth for both `_local_prefillable` and `_prefill_batch_ready` so their
        thresholds can never drift apart."""
        return self._waiting_prefill_tokens() >= self.max_num_batched_tokens

    def _local_prefillable(self, is_dense: bool) -> bool:
        """This rank's `local_prefillable` signal for the PrefillDelayer: it can
        admit a prefill AND has a full dense batch's worth of waiting tokens
        (`is_dense`, precomputed by the caller to avoid a second waiting-queue
        scan — see `_waiting_is_dense`). Ranks report this so the delayer can
        align dense prefills cross-DP instead of letting stragglers fire
        under-full."""
        return self._can_admit_head_prefill() and is_dense

    def _prefill_batch_ready(self, is_dense: Optional[bool] = None) -> bool:
        """Gate for *starting* a new dense prefill from a pure-decode tick.

        NOTE: this is NOT the sole gate on whether a new prefill fires. Once
        Phase 1 has resumed a partial (making the step a prefill anyway),
        Phase 2 may still pack new waiting requests even when this returns
        False — see the `step_is_prefill` branch in `schedule()`. This method
        only decides whether an under-full waiting batch is allowed to preempt
        decode and START a prefill (#1437).

        The density threshold is max_num_batched_tokens (a full prefill batch);
        `is_dense` (from `_waiting_is_dense`) may be passed in to reuse a
        per-tick computation, else it is computed on demand.

        Returns True (allow prefill) when:
          - the waiting queue can fill a dense batch (>= threshold tokens), or
          - there is nothing left to decode (running empty) — tail escape so a
            partial final batch still goes out, or
          - we've suppressed prefill for too many consecutive passes
            (_prefill_hold_max_passes) — starvation bound.

        Otherwise returns False (keep decoding) and advances the hold counter.
        The counter resets whenever this gate allows prefill to fire (the True
        branches above). NOTE: it is only advanced/read when the delayer permits
        (the call site short-circuits behind `delayer_allows`), so it does not
        tick during a cross-DP delayer veto.
        """
        # Gate only applies under DP (>1); otherwise never hold (legacy path).
        if not self._prefill_gate_enabled:
            return True
        # Tail escape: no decode work left — never hold, or we'd deadlock.
        if not self.running:
            self._prefill_hold_passes = 0
            return True
        if is_dense is None:
            is_dense = self._waiting_is_dense()
        if is_dense:
            self._prefill_hold_passes = 0
            return True
        # Under-full: hold prefill (keep decoding) up to the pass budget.
        self._prefill_hold_passes += 1
        if self._prefill_hold_passes >= self._prefill_hold_max_passes:
            self._prefill_hold_passes = 0
            return True
        return False

    def _kv_usage(self) -> float:
        """Fraction of KV-cache blocks currently in use ∈ [0, 1].

        Used as the `token_usage` signal for PrefillDelayer's low-watermark
        safety valve. Derived from BlockManager bookkeeping; cheap (no
        traversal of seq tables).
        """
        bm = self.block_manager
        total = len(bm.blocks)
        if total <= 0:
            return 0.0
        return len(bm.used_block_ids) / total

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
        return not self.waiting and not self.running and not self._rejected

    def add(self, seq: Sequence):
        self._warn_if_unschedulable(seq)
        self.waiting.append(seq)

    def extend(self, seqs: list[Sequence]):
        for seq in seqs:
            self._warn_if_unschedulable(seq)
        self.waiting.extend(seqs)

    def _unschedulable_reason(self, seq: Sequence) -> Optional[str]:
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
          - prompt's KV blocks (+ per-req cache reservation) exceed the total
            pool size → never fits even on a fully empty pool

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
        if not self.enable_chunked_prefill and num_tokens > self.max_num_batched_tokens:
            return (
                f"input tokens={num_tokens} > max_num_batched_tokens="
                f"{self.max_num_batched_tokens}. Increase --max-num-batched-tokens, "
                f"enable chunked prefill, or shorten the prompt."
            )
        bm = self.block_manager
        total_blocks = len(bm.blocks)
        if seq.num_blocks > total_blocks:
            return (
                f"needs {seq.num_blocks} KV blocks for {num_tokens} input tokens "
                f"> total pool blocks={total_blocks}. Reduce prompt length or "
                f"raise --gpu-memory-utilization. (Per-req state cache lives in "
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
        # We check the slot list length below; without the accounting dict we
        # infer "no slots ever existed" from `num_per_req_cache_groups == 0`,
        # exposed via the free list at init time (slot ids 0..N-1).
        if seq.has_per_req_cache and len(bm.free_per_req_cache_groups) == 0:
            # All slots are currently in-use OR no slots were ever created.
            # The schedule loop handles "currently full" by waiting; only
            # warn for the permanent "never created" case, identified by
            # `num_per_req_cache_groups` being 0 in the config.
            if getattr(self.config, "num_per_req_cache_groups", 0) == 0:
                logger.warning(
                    "Request %s will never be scheduled: needs per-req cache "
                    "slot but no slots were allocated (max_num_seqs=0 for "
                    "this model type).",
                    seq.id,
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

    def schedule(self) -> tuple[ScheduledBatch, dict[int, Sequence]]:
        """Select the next batch of sequences for a forward pass.

        Tries prefill first; if no new prefills are ready, falls back to
        decoding already-running sequences.
        """
        scheduled_seqs = {}
        num_seqs_prefill = 0
        num_batched_tokens = 0
        skipped_waiting_requests: deque[Sequence] = deque()
        num_scheduled_tokens: list[int] = []
        scheduled_spec_decode_tokens: dict[int, np.ndarray] = {}

        self._promote_ready_remote_kv_requests()
        self._park_ready_offload_partial_prefills()

        # ─── Prefill gate ──────────────────────────────────────────────
        # Two flags drive prefill admission below:
        #   delayer_allows       — cross-DP alignment permits prefill (also gates
        #                          Phase-1 partial resume). True when no delayer.
        #   new_prefill_allowed  — a *new* dense prefill (Phase 2) may START.
        #
        # A new prefill needs cross-DP alignment AND local density. These are
        # COMPLEMENTARY, not redundant — each is the operative gate in a
        # different regime:
        #   • "mixed" (some ranks have a full batch, some don't): the delayer
        #     vetoes so a lone rank doesn't prefill while siblings decode
        #     (unbalanced MoE all-to-all).
        #   • "none" (every rank under-full): the delayer permits vacuously, so
        #     the LOCAL `_prefill_batch_ready` is what holds back an under-full
        #     prefill that would preempt decode out of cudagraph (#1437).
        #   • "all" (every rank has a full batch): both trivially pass — the
        #     only regime where the two overlap.
        #
        # `is_dense` (the waiting-queue density) feeds BOTH the delayer's
        # local_prefillable signal and `_prefill_batch_ready`, so compute it once
        # per tick. Only compute it when a consumer reads it (the delayer always
        # does; `_prefill_batch_ready` only under the DP gate) — this keeps the
        # non-DP path from scanning the waiting queue; None means "not computed".
        need_density = self.prefill_delayer is not None or self._prefill_gate_enabled
        is_dense = self._waiting_is_dense() if need_density else None

        # should_allow_prefill() runs a cross-DP all_reduce and MUST be called
        # every tick on every rank for lockstep — hence before the early-return.
        if self.prefill_delayer is not None:
            delayer_allows = self.prefill_delayer.should_allow_prefill(
                local_prefillable=self._local_prefillable(is_dense),
                token_usage=self._kv_usage(),
            )
        else:
            delayer_allows = True

        if not self.running and not self.waiting:
            return None

        # Keep the `and` short-circuit: `_prefill_batch_ready` mutates the
        # starvation hold-counter, so it must NOT run when the delayer vetoes
        # (delayer_allows=False), or the counter would advance on vetoed ticks.
        new_prefill_allowed = delayer_allows and self._prefill_batch_ready(is_dense)

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
                chunk = min(remaining, budget_remaining)
                if chunk <= 0:
                    break
                num_batched_tokens += chunk
                num_seqs_prefill += 1
                seq.type = SequenceType.PREFILL
                scheduled_seqs[seq.id] = seq
                num_scheduled_tokens.append(chunk)

        # ---- Phase 2: new requests from waiting ----
        # Admit new waiting requests when a new dense prefill may START
        # (`new_prefill_allowed`) OR Phase 1 already made this a prefill step
        # (`step_is_prefill`): in the latter case decode is already interrupted,
        # so packing fresh requests into the remaining token budget is free and
        # avoids emitting a partial-only under-full prefill. Both disjuncts are
        # gated by `delayer_allows` (the first embeds it, the second ANDs it) so
        # a delayer veto skips both phases in cross-DP lockstep.
        step_is_prefill = num_seqs_prefill > 0
        while (
            (new_prefill_allowed or (delayer_allows and step_is_prefill))
            and (self.delay_factor <= 0 or self._passed_delay(time.time()))
            and self.waiting
            and num_seqs_prefill < self.max_num_seqs
            and num_batched_tokens < self.max_num_batched_tokens
        ):
            seq = self.waiting.popleft()

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
                self._assert_positive_prefill_chunk(
                    chunk, num_new_tokens, budget_remaining
                )
                num_seqs_prefill, num_batched_tokens = self._schedule_prefill_seq(
                    seq,
                    chunk,
                    scheduled_seqs,
                    num_scheduled_tokens,
                    num_seqs_prefill,
                    num_batched_tokens,
                )
                continue

            # Probe cache hits FIRST so budget check sees the real
            # (post-prefix-cache) remaining token count.
            num_cached_blocks = self.block_manager.can_allocate(seq)
            if num_cached_blocks < 0:
                self.waiting.appendleft(seq)
                break

            # V4 SWA fix (B'): drop the last `_v4_swa_warmup_blocks` hit blocks so
            # those tokens are re-forwarded, repopulating the per-request SWA
            # ring (see __init__). Compressed-KV n_committed is unaffected
            # (context_lens = cached + scheduled stays = prompt length). Only
            # fires on a real hit (>0); a full miss is untouched.
            if self._v4_swa_warmup_blocks and num_cached_blocks > 0:
                num_cached_blocks = max(
                    0, num_cached_blocks - self._v4_swa_warmup_blocks
                )

            # Use num_tokens (not num_prompt_tokens) so preempted seqs re-forward
            # their decoded tokens — preempt() frees their KV blocks but keeps
            # the token_ids, so num_tokens > num_prompt_tokens and those tokens
            # still need KV recomputed.
            num_new_tokens = (
                seq.num_tokens - num_cached_blocks * self.block_manager.block_size
            )
            if (
                self.enable_chunked_prefill
                and 0 < self.long_prefill_token_threshold < num_new_tokens
            ):
                num_new_tokens = self.long_prefill_token_threshold
            budget_remaining = self.max_num_batched_tokens - num_batched_tokens
            chunk = self._prefill_chunk_for_budget(
                num_new_tokens, budget_remaining, num_batched_tokens
            )
            if chunk is None:
                self.waiting.appendleft(seq)
                break
            self.block_manager.allocate(seq, num_cached_blocks)

            # Snapshot the genuine prefix-cache hit at admission. After this,
            # num_cached_tokens is repurposed to track chunked-prefill progress
            # (it grows to the full prompt length in postprocess), so it can't be
            # used to report the cache hit. Set once per seq (Phase-2 admission
            # only); Phase-1 resume doesn't recompute num_cached_blocks.
            seq.prefix_cache_hit_tokens = (
                num_cached_blocks * self.block_manager.block_size
            )

            self._notify_connector_after_prefill_alloc(seq)

            needs_remote_load = self._confirm_remote_load_after_alloc(
                seq, needs_remote_load
            )

            if needs_remote_load:
                self._park_for_remote_load(seq, skipped_waiting_requests)
                continue

            chunk = self._adjust_prefill_chunk_after_alloc(seq, chunk)

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

        total_tokens_num_prefill = sum(num_scheduled_tokens)

        if num_seqs_prefill > 0:
            num_cached_tokens_list = [
                seq.num_cached_tokens for seq in scheduled_seqs.values()
            ]
            cached_per_req = [s.num_cached_tokens for s in scheduled_seqs.values()]
            logger.info(
                f"Scheduled prefill batch: {num_seqs_prefill} reqs, "
                f"{total_tokens_num_prefill} new tokens "
                f"(cached: {cached_per_req}, new: {num_scheduled_tokens}), "
                f"req_ids: {tuple(scheduled_seqs.keys())}"
            )
            self.prev_prompt = True
            # lip: TODO for prefill/decode mixed batch

            connector_meta_output = None
            if self.kv_connector is not None:
                connector_meta_output = self.kv_connector.build_connector_meta()
            return (
                ScheduledBatch(
                    seqs=scheduled_seqs,
                    num_scheduled_tokens=num_scheduled_tokens,
                    total_tokens_num=total_tokens_num_prefill,
                    total_tokens_num_prefill=total_tokens_num_prefill,
                    total_seqs_num=num_seqs_prefill,
                    total_seqs_num_prefill=num_seqs_prefill,
                    connector_meta_output=connector_meta_output,
                    num_cached_tokens=num_cached_tokens_list,
                ),
                scheduled_seqs,
            )

        # --- Decode scheduling ---
        num_seqs_decode = 0
        num_decode_tokens = 0
        tokens_per_decode_seq = self.mtp_k + 1
        num_new_tokens = self.mtp_k + 1
        remote_kv_blocks: set[int] = set()
        remote_kv_seq_blocks: dict[int, list[int]] = {}
        skipped_partial_prefills: list[Sequence] = []
        while self.running and num_seqs_decode < self.max_num_seqs:
            if num_decode_tokens + tokens_per_decode_seq > self.max_num_batched_tokens:
                break
            seq = self.running.popleft()
            if seq.is_partial_prefill:
                skipped_partial_prefills.append(seq)
                continue
            while not self.block_manager.can_append(seq, num_new_tokens):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                if seq.spec_token_ids.size > 0:
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
                    logger.info(
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

        connector_meta_output = None
        if self.kv_connector is not None:
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
            num_spec_step=self.mtp_k,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            remote_kv_block_ids=sorted(remote_kv_blocks) if remote_kv_blocks else [],
            remote_kv_seq_blocks=remote_kv_seq_blocks,
        )
        return (decode_batch, scheduled_seqs)

    # -- Remote KV / offload admission helpers ------------------------------
    def _resolve_waiting_remote_kv(
        self, seq: Sequence, skipped_waiting_requests: deque[Sequence]
    ) -> Optional[bool]:
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
        if self._is_offload_connector():
            self._mark_offload_load_ready(seq)
            return False
        return True

    def _consume_failed_remote_kv(self, seq: Sequence) -> bool:
        if not self._pop_req_id(self.failed_recving_kv_req_ids, seq.id):
            return False

        if self.kv_connector is not None and hasattr(self.kv_connector, "load_failed"):
            self.kv_connector.load_failed(seq.id)
        seq.status = SequenceStatus.WAITING
        seq.offload_loaded = False
        seq.offload_loaded_tokens = seq.num_cached_tokens
        seq.offload_load_failed = True
        return True

    def _mark_offload_load_ready(self, seq: Sequence) -> None:
        """Turn a completed offload load into a suffix-prefill resume."""
        loaded = getattr(seq, "offload_loaded_tokens", None)
        logger.debug(
            "[OFFLOAD-WAKE] seq %s: loaded=%s prev_cached=%d num_tokens=%d",
            seq.id,
            loaded,
            seq.num_cached_tokens,
            seq.num_tokens,
        )
        if loaded is not None and loaded > seq.num_cached_tokens:
            seq.num_cached_tokens = loaded
        seq.offload_loaded = True

    def _is_offload_prefill_resume(self, seq: Sequence) -> bool:
        """True when offload already owns blocks and should resume suffix prefill.

        This avoids a second prefix lookup and, more importantly, avoids calling
        ``BlockManager.allocate`` again for a sequence whose block table was
        allocated before it parked for the LMCache load.
        """
        return (
            self._is_offload_connector()
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
        _ext_tokens, needs_remote_load = self.kv_connector.get_num_new_matched_tokens(
            seq
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
                seq.spec_token_ids = np.asarray(drafts, dtype=np.int32)
        logger.info(
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

    def _prefill_chunk_for_budget(
        self, num_new_tokens: int, budget_remaining: int, num_batched_tokens: int
    ) -> Optional[int]:
        if self.enable_chunked_prefill:
            return min(num_new_tokens, budget_remaining)
        if num_new_tokens > budget_remaining and num_batched_tokens > 0:
            return None
        return num_new_tokens

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
        if self.cache_stats:
            self.cache_stats.update(seq.num_cached_tokens, seq.num_tokens)
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

    def _park_for_remote_load(
        self, seq: Sequence, skipped_waiting_requests: deque[Sequence]
    ) -> None:
        skipped_waiting_requests.append(seq)
        seq.status = SequenceStatus.WAITING_FOR_REMOTE_KVS

    def _adjust_prefill_chunk_after_alloc(self, seq: Sequence, chunk: int) -> int:
        if self.kv_connector is not None and hasattr(
            self.kv_connector, "adjust_prefill_chunk_after_alloc"
        ):
            return self.kv_connector.adjust_prefill_chunk_after_alloc(seq, chunk)
        return chunk

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        # Strip placeholder + rejected draft tokens added by postprocess.
        # Real token count = seq.num_tokens - mtp_k - num_rejected
        # (same formula as postprocess line: num_tokens = seq.num_tokens - self.mtp_k - num_rejected)
        if self.mtp_k > 0:
            strip = self.mtp_k + seq.num_rejected
            if strip > 0:
                del seq.token_ids[-strip:]
                del seq.output_tokens[-strip:]
                seq.num_tokens -= strip
        seq.num_rejected = 0
        seq.num_bonus_tokens = 0
        seq.spec_token_ids = np.array([], dtype=np.int32)
        seq.is_first_decode = False
        if seq.is_partial_prefill:
            seq.is_partial_prefill = False
            self._partial_prefill_count -= 1
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

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
        if batch is not None:
            running_by_id = {seq.id: seq for seq in self.running}
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

        need_placeholder = is_deferred_out or self.use_spec
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
            if seq.is_partial_prefill:
                continue
            # Drop stale tokens produced by chunked-prefill steps.
            #
            # There are two ways a stale token reaches this point:
            # 1. Normal chunked prefill: the seq was partial at the start of
            #    this postprocess call. With deferred output, the visible token
            #    is one step late, so it belongs to the previous partial chunk.
            # 2. Offload/remote-KV handoff: a partial seq can be parked out of
            #    running and have seq.is_partial_prefill cleared. In that case
            #    prev_partial_ids can no longer see it, so the park path sets
            #    _discard_next_deferred_output to carry "drop one old token"
            #    across the park/resume boundary.
            was_partial_prefill_at_step_start = seq.id in prev_partial_ids
            drop_old_token_after_offload_park = False
            if is_deferred_out and getattr(seq, "_discard_next_deferred_output", False):
                seq._discard_next_deferred_output = False
                drop_old_token_after_offload_park = True
            if was_partial_prefill_at_step_start or drop_old_token_after_offload_park:
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
            # fires and `hash_to_block_id` stays empty for prompt blocks (HBM
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

            if is_deferred_out or self.use_spec:
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
                    self.spec_stats
                    and num_new_token > 0
                    and (num_new_token + num_rejected) > 1
                ):
                    self.spec_stats.update(num_new_token)
                seq.num_rejected = num_rejected
                seq.num_bonus_tokens = num_bonus
                for i, el in enumerate(token_ids):
                    seq.token_ids[-num_placeholder - offset + i] = el
                    seq.output_tokens[-num_placeholder - offset + i] = el
                # logger.info(
                #     f"{seq.id=}, {num_new_token=} {num_rejected=} {self.mtp_k} {token_ids=} {seq.token_ids[-8:]=}"
                # )
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

            if self.mtp_k > 0:
                # idx already resolved above via get_idx
                seq.spec_token_ids = draft_token_ids[idx]

            if seq.num_completion_tokens <= 3 and seq.kv_transfer_params:
                logger.info(
                    "[PD-DECODE] seq %s: comp_tokens=%d, "
                    "new_token=%s, num_tokens=%d, blocks=%d",
                    seq.id,
                    seq.num_completion_tokens,
                    token_ids,
                    seq.num_tokens,
                    len(seq.block_table),
                )
            if seq.num_completion_tokens >= 1 and seq.first_token_time == 0.0:
                seq.first_token_time = time.time()

            num_tokens = seq.num_tokens - self.mtp_k - num_rejected
            leave_reason = None
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
            stop_at_idx: Optional[int] = None
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
                elif (num_tokens - seq.num_prompt_tokens) >= seq.max_tokens:
                    # Use the local `num_tokens` (= seq.num_tokens - mtp_k -
                    # num_rejected, set at line 716) instead of the property
                    # `seq.num_completion_tokens` which still reflects the
                    # raw mtp_k+1 placeholder bump from `prepare_decode`. The
                    # property over-counts by `mtp_k + num_rejected`, causing
                    # max_tokens to trip that many tokens early (visible as
                    # `output tokens=95` for max_tokens=100, mtp_k=3). Non-MTP
                    # path: mtp_k = num_rejected = 0 → behavior unchanged.
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

            # Prepare stream output
            if stream_output_queue is not None and new_tokens:
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
                    logger.info("KV transfer output present in stream output.")

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
        for seq in finished_seqs:
            logger.debug("Freeing blocks for finished seq %s", seq.id)
            if seq.is_partial_prefill:
                seq.is_partial_prefill = False
                self._partial_prefill_count -= 1
            if self.kv_connector is not None:
                if hasattr(self.kv_connector, "request_finished"):
                    self.kv_connector.request_finished(seq)
                if not self.kv_connector.is_producer:
                    if hasattr(self.kv_connector, "should_defer_free") and (
                        self.kv_connector.should_defer_free(seq)
                    ):
                        logger.debug(
                            "Deferring block free for seq %s until KV save completes.",
                            seq.id,
                        )
                        self.deferred_free_blocks[seq.id] = seq
                    else:
                        self.block_manager.deallocate(seq)
                else:
                    logger.debug(
                        "Deferring block free for seq %s until KV send completes.",
                        seq.id,
                    )
                    self.deferred_free_blocks[seq.id] = seq
            else:
                self.block_manager.deallocate(seq)
            self.running.remove(seq)
        return finished_seqs

    def _is_offload_connector(self) -> bool:
        """True when the active KV connector is the CPU/NVMe offload backend.

        Offload wakes a parked seq into a SUFFIX prefill (not the P/D decode
        jump). Connectors set ``is_offload = True`` to opt into this path.
        """
        return getattr(self.kv_connector, "is_offload", False)

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

        if self.block_manager.kv_events_enabled:
            bm = self.block_manager
            num_cached_blocks = seq.num_cached_tokens // bm.block_size
            remote_hashes: list[int] = []
            remote_tokens: list[int] = []
            parent_block_hash: int | None = None
            prev_hash: int | None = None
            for i, block_id in enumerate(seq.block_table):
                blk = bm.blocks[block_id]
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
                self.block_manager.record_remote_store(
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
        blocked groups.
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
                    seq._discard_next_deferred_output = True
                    seq.is_partial_prefill = False
                    self._partial_prefill_count -= 1
                seq.status = SequenceStatus.WAITING_FOR_REMOTE_KVS
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

        def _pop_deferred(req_id):
            seq = self.deferred_free_blocks.pop(req_id, None)
            if seq is not None:
                return seq
            try:
                return self.deferred_free_blocks.pop(int(req_id), None)
            except (TypeError, ValueError):
                return None

        for req_id in kv_connector_output.finished_recving or ():
            assert (
                not self.kv_connector.is_producer
            ), "Only consumer should update recving KV status"
            logger.debug("Finished recving KV transfer for request %s", req_id)
            self.finished_recving_kv_req_ids.append(req_id)

        for req_id in kv_connector_output.failed_recving or ():
            assert (
                not self.kv_connector.is_producer
            ), "Only consumer should update failed KV recv status"
            logger.warning(
                "KV receive failed for request %s; falling back to prefill.", req_id
            )
            self.failed_recving_kv_req_ids.append(req_id)

        for req_id in kv_connector_output.finished_sending or ():
            assert (
                self.kv_connector.is_producer
            ), "Only producer should free blocks after sending KV"
            logger.debug("Finished sending KV transfer for request %s", req_id)
            seq = _pop_deferred(req_id)
            assert seq is not None, f"req_id={req_id} not found in deferred_free_blocks"
            self.block_manager.deallocate(seq)

        for req_id in kv_connector_output.finished_saving or ():
            if hasattr(self.kv_connector, "save_finished"):
                self.kv_connector.save_finished(req_id)
            seq = self.deferred_free_blocks.get(req_id)
            if seq is None:
                try:
                    seq = self.deferred_free_blocks.get(int(req_id))
                except (TypeError, ValueError):
                    seq = None
            if seq is not None and not (
                hasattr(self.kv_connector, "should_defer_free")
                and self.kv_connector.should_defer_free(seq)
            ):
                seq = _pop_deferred(req_id)
                if seq is not None and hasattr(self.kv_connector, "request_finished"):
                    self.kv_connector.request_finished(seq)
                if seq is not None:
                    self.block_manager.deallocate(seq)

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

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
            f"[MTP Stats         ] Average toks/fwd: {1+self.total_accepted / ts:.2f}, "
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
    ):
        if scheduled_spec_decode_tokens is None:
            scheduled_spec_decode_tokens = {}

        self.req_ids = list(seqs.keys())
        # self.scheduled_tokens = [
        #     seq.token_ids[-num_tokens:]
        #     for seq, num_tokens in zip(seqs.values(), num_scheduled_tokens)
        # ]
        # logger.info(f"{num_scheduled_tokens=}")
        # logger.info(f"{self.scheduled_tokens=}")
        # num_scheduled_tokens for each sequence in the batch
        self.num_scheduled_tokens = np.asarray(num_scheduled_tokens, dtype=np.int32)
        self.temperatures = np.asarray(
            [seq.temperature for seq in seqs.values()], dtype=np.float32
        )
        self.context_lens = np.asarray(
            [seq.num_tokens for seq in seqs.values()], dtype=np.int32
        )
        self.num_rejected = np.asarray(
            [seq.num_rejected for seq in seqs.values()], dtype=np.int32
        )
        self.num_bonus = np.asarray(
            [seq.num_bonus_tokens for seq in seqs.values()], dtype=np.int32
        )
        self.mamba_state_slots = [
            seq.mamba_state_slot
            for seq in seqs.values()
            if seq.mamba_enabled and seq.mamba_state_slot >= 0
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

        offs = self.context_lens - self.num_rejected - self.num_scheduled_tokens
        self.scheduled_tokens = np.empty(total_tokens_num, dtype=np.int32)
        pos = 0
        for seq, num, offset in zip(seqs.values(), num_scheduled_tokens, offs):
            self.scheduled_tokens[pos : pos + num] = seq.token_ids[
                offset : offset + num
            ]
            pos += num

        if num_spec_step > 0:
            self.scheduled_spec_decode_tokens = np.asarray(
                list(scheduled_spec_decode_tokens.values()), dtype=np.int32
            )
        self.block_tables = [
            seq.block_table for seq in seqs.values() if seq.block_table
        ]
        self.last_block_num_tokens = [
            seq.last_block_num_tokens for seq in seqs.values()
        ]
        self.num_cached_tokens = [seq.num_cached_tokens for seq in seqs.values()]

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
    ):
        self.req_ids = req_ids
        self.token_ids = token_ids
        self.draft_token_ids = draft_token_ids
        self.num_rejected = num_rejected
        self.num_bonus = num_bonus
        self.is_deferred_out = is_deferred_out
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
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.stop_token_ids = config.stop_token_ids
        self.block_manager = BlockManager(config)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.config = config

        # KV transfer bookkeeping
        self.finished_recving_kv_req_ids: list[int] = []
        self.deferred_free_blocks: dict[int, Sequence] = {}

        # Scheduling delay for batching efficiency
        self.prev_time = 0.0
        # Did we schedule a prompt at previous step?
        self.prev_prompt = False
        # Latency of the last prompt step
        self.last_prompt_latency = 0.0
        self.delay_factor = config.scheduler_delay_factor

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

        from atom.utils.forward_context import get_kvconnector

        self.kv_connector = get_kvconnector("scheduler", config)

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def extend(self, seqs: list[Sequence]):
        self.waiting.extend(seqs)

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

        if not self.running and not self.waiting:
            return None

        # --- Prefill scheduling ---
        while self.waiting and num_seqs_prefill < self.max_num_seqs:
            seq = self.waiting.popleft()

            # KV Transfer: skip request if still waiting for remote KVs
            waiting_remote_to_waiting_ready = False
            if seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS:
                waiting_remote_to_waiting_ready = self._update_waiting_for_remote_kv(
                    seq
                )
                if waiting_remote_to_waiting_ready:
                    seq.status = SequenceStatus.WAITING
                else:
                    skipped_waiting_requests.append(seq)
                    continue

            need_to_remove_to_load_kv_async_queue = False
            if self.kv_connector is not None and not waiting_remote_to_waiting_ready:
                _ext_tokens, need_to_remove_to_load_kv_async_queue = (
                    self.kv_connector.get_num_new_matched_tokens(seq)
                )

            num_new_tokens = seq.num_tokens - seq.num_cached_tokens
            if (
                num_batched_tokens + num_new_tokens > self.max_num_batched_tokens
                or not self.block_manager.can_allocate(seq)
            ):
                self.waiting.appendleft(seq)
                break

            if not waiting_remote_to_waiting_ready:
                self.block_manager.allocate(seq)

            if self.kv_connector is not None:
                self.kv_connector.update_state_after_alloc(seq)

            if need_to_remove_to_load_kv_async_queue:
                skipped_waiting_requests.append(seq)
                seq.status = SequenceStatus.WAITING_FOR_REMOTE_KVS
                continue

            if waiting_remote_to_waiting_ready:
                seq.status = SequenceStatus.RUNNING
                seq.is_first_decode = True
                self.running.append(seq)
                continue

            num_seqs_prefill += 1
            num_new_tokens = seq.num_tokens - seq.num_cached_tokens
            if self.cache_stats:
                self.cache_stats.update(seq.num_cached_tokens, seq.num_tokens)
            num_batched_tokens += num_new_tokens
            seq.status = SequenceStatus.RUNNING
            seq.type = SequenceType.PREFILL
            self.running.append(seq)
            scheduled_seqs[seq.id] = seq
            num_scheduled_tokens.append(num_new_tokens)

        if skipped_waiting_requests:
            logger.debug(
                "Re-adding %d skipped requests back to waiting queue.",
                len(skipped_waiting_requests),
            )
            self.waiting.extend(skipped_waiting_requests)

        total_tokens_num_prefill = sum(num_scheduled_tokens)

        if num_seqs_prefill > 0:
            logger.info(
                f"Scheduled prefill batch: {num_seqs_prefill} reqs, "
                f"{total_tokens_num_prefill} tokens, "
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
                ),
                scheduled_seqs,
            )

        # --- Decode scheduling ---
        num_seqs_decode = 0
        num_new_tokens = self.mtp_k + 1
        while self.running and num_seqs_decode < self.max_num_seqs:
            seq = self.running.popleft()
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
                # Skip block append for the first decode step after remote
                # prefill — blocks were already allocated during prefill.
                if not getattr(seq, "is_first_decode", False):
                    self.block_manager.may_append(seq, num_new_tokens)
                scheduled_seqs[seq.id] = seq
                seq.type = SequenceType.DECODE
                num_scheduled_tokens.append(num_new_tokens)
                seq.is_first_decode = False

        total_tokens_num_decode = sum(num_scheduled_tokens)

        if scheduled_seqs:
            self.running.extendleft(reversed(scheduled_seqs.values()))

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
        )
        return (decode_batch, scheduled_seqs)

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
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(
        self,
        seqs: list[Sequence],
        fwd_output: ScheduledBatchOutput,
        stream_output_queue=None,
    ) -> list[Sequence]:
        """Process model outputs: update tokens, check stop conditions, free blocks."""
        prev_token_ids = fwd_output.token_ids
        draft_token_ids = fwd_output.draft_token_ids
        is_deferred_out = fwd_output.is_deferred_out
        # logger.info(
        #     f"Scheduler postprocess: received output for req_ids={fwd_output.req_ids}, draft_token_ids shape={fwd_output.draft_token_ids.shape}, accepted token ids: {prev_token_ids}"
        # )
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
            token_ids = prev_token_ids[idx]
            num_new_token = len(token_ids)
            if self.spec_stats:
                self.spec_stats.update(num_new_token)
            if is_deferred_out or self.use_spec:
                num_rejected = fwd_output.num_rejected[idx]
                num_bonus = fwd_output.num_bonus[idx]
                offset = 0 if (num_new_token + num_rejected) == 1 else self.mtp_k
                seq.num_rejected = num_rejected
                seq.num_bonus_tokens = num_bonus
                for i, el in enumerate(token_ids):
                    seq.token_ids[-num_placeholder - offset + i] = el
                    seq.output_tokens[-num_placeholder - offset + i] = el
                # logger.info(
                #     f"{seq.id=}, {num_new_token=} {num_rejected=} {self.mtp_k} {token_ids=} {seq.token_ids[-8:]=}"
                # )
            else:
                num_rejected = 0
                num_bonus = 0
                for token_id in token_ids:
                    seq.append_token(token_id)
            new_tokens = token_ids

            if self.mtp_k > 0:
                # idx already resolved above via get_idx
                seq.spec_token_ids = draft_token_ids[idx]

            if seq.num_completion_tokens == 1 and seq.first_token_time == 0.0:
                seq.first_token_time = time.time()

            num_tokens = seq.num_tokens - self.mtp_k - num_rejected
            leave_reason = None
            # Check if sequence ends with any stop sequence
            for stop_seq in seq.stop_token_sequences:
                stop_len = len(stop_seq)
                if num_tokens >= stop_len:
                    is_stop = False
                    for i in range(num_new_token):
                        offset = num_tokens - i
                        if seq.token_ids[offset - stop_len : offset] == stop_seq:
                            is_stop = True
                            break
                    if is_stop:
                        leave_reason = "stop_sequence"
                        break
            else:
                # Check the last token in the list for EOS
                if token_ids and not seq.ignore_eos and self.eos_token_id in token_ids:
                    leave_reason = "eos"
                elif not seq.ignore_eos and any(
                    t in self.stop_token_ids for t in token_ids
                ):
                    first_stop_token = next(
                        t for t in token_ids if t in self.stop_token_ids
                    )
                    leave_reason = f"stop_{first_stop_token}"
                elif seq.num_completion_tokens >= seq.max_tokens:
                    leave_reason = "max_tokens"

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
                if seq.status == SequenceStatus.RUNNING:
                    num = num_placeholder - seq.num_rejected
                    for _ in range(num):
                        seq.append_token(self.eos_token_id)
                    # logger.info(
                    #     f"{seq.id=}, added {num}, total tokens now: {seq.num_tokens}"
                    # )
        for seq in finished_seqs:
            logger.debug("Freeing blocks for finished seq %s", seq.id)
            if self.kv_connector is not None:
                if not self.kv_connector.is_producer:
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

    def _update_waiting_for_remote_kv(self, seq: Sequence) -> bool:
        """Check whether a remote KV transfer for *seq* has completed.

        The ``finished_recving_kv_req_ids`` list is populated by
        :meth:`_update_from_kv_xfer_finished` during the previous
        scheduling step.  When ready, the sequence transitions back
        from ``WAITING_FOR_REMOTE_KVS`` to ``WAITING``.
        """
        if seq.id not in self.finished_recving_kv_req_ids:
            return False

        self.finished_recving_kv_req_ids.remove(seq.id)
        logger.debug("KV transfer finished for seq %s, ready for scheduling.", seq.id)
        return True

    def _update_from_kv_xfer_finished(self, kv_connector_output: KVConnectorOutput):
        """Reconcile scheduler state with completed KV transfers.

        * ``finished_recving``: marks requests as ready for decode scheduling.
        * ``finished_sending``: releases deferred block allocations on the
          producer side.
        """
        if kv_connector_output is None:
            return

        for req_id in kv_connector_output.finished_recving or ():
            assert (
                not self.kv_connector.is_producer
            ), "Only consumer should update recving KV status"
            logger.debug("Finished recving KV transfer for request %s", req_id)
            self.finished_recving_kv_req_ids.append(req_id)

        for req_id in kv_connector_output.finished_sending or ():
            assert (
                self.kv_connector.is_producer
            ), "Only producer should free blocks after sending KV"
            logger.debug("Finished sending KV transfer for request %s", req_id)
            assert (
                req_id in self.deferred_free_blocks
            ), f"req_id={req_id} not found in deferred_free_blocks"
            self.block_manager.deallocate(self.deferred_free_blocks.pop(req_id))

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

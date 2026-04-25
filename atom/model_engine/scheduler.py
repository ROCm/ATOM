# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
import threading
import time
from collections import deque
from typing import Optional

import numpy as np
from atom.config import Config
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


def _optimal_cu_fraction(
    decode_batch: int, prefill_waiting_tokens: int
) -> Optional[float]:
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
    return None
    # if prefill_waiting_tokens==0 or decode_batch<64:
    #     logger.info("returning none")
    #     return None
    # else:
    #     logger.info("returning 0.5")
    #     return 0.5
    # if prefill_waiting_tokens == 0:
    #     return None
    # if prefill_waiting_tokens < 512:
    #     return None
    # if prefill_waiting_tokens >= 8192:
    #     return None
    # if prefill_waiting_tokens >= 4096:
    #     return 0.7
    # # 512 <= prefill_waiting_tokens < 4096
    # if decode_batch > 96:
    #     return 0.7
    # return 0.8


class ScheduledBatch:
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
        is_dummy_run: bool = False,
        num_spec_step: int = 0,
        scheduled_spec_decode_tokens: dict[int, np.ndarray] = {},
        cu_stream_fraction: Optional[float] = None,
    ):
        # len(seqs) == total_seqs_num == total_seqs_num_prefill + total_seqs_num_decode
        # self.seqs = seqs
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
        self.mamba_block_tables = [
            seq.mamba_block_table for seq in seqs.values() if seq.mamba_block_table
        ]
        self.top_ks = np.asarray([seq.top_k for seq in seqs.values()], dtype=np.int32)
        self.top_ps = np.asarray([seq.top_p for seq in seqs.values()], dtype=np.float32)

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

        self.is_dummy_run = is_dummy_run

        self.num_spec_step = num_spec_step

        # Key into ModelRunner's stream pool for CU-masked disagg streams.
        # None means full-CU fallback (no mask).
        self.cu_stream_fraction = cu_stream_fraction

        # logger.info(f"{[el for el in scheduled_spec_decode_tokens.keys()]=}")
        # logger.info(f"{self.num_scheduled_tokens=}")
        # logger.info(f"{self.context_lens=}")
        # logger.info(f"{[len(blk)*16 for blk in self.block_tables]=}")
        # logger.info(f"{self.block_tables=}")


class ScheduledBatchOutput:

    def __init__(
        self,
        req_ids: list[int],
        token_ids: list[tuple[int, ...]],
        num_rejected: Optional[np.ndarray],
        num_bonus: Optional[np.ndarray],
        draft_token_ids: Optional[np.ndarray],
        is_deferred_out=False,
    ):
        self.req_ids = req_ids
        self.token_ids = token_ids
        self.draft_token_ids = draft_token_ids
        self.num_rejected = num_rejected
        self.num_bonus = num_bonus
        self.is_deferred_out = is_deferred_out
        # O(1) lookup: req_id -> index (lazy-built on first access)
        self._req_id_to_idx: Optional[dict[int, int]] = None

    def get_idx(self, req_id: int) -> Optional[int]:
        """O(1) lookup of request index by id."""
        if self._req_id_to_idx is None:
            self._req_id_to_idx = {rid: i for i, rid in enumerate(self.req_ids)}
        return self._req_id_to_idx.get(req_id)


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.stop_token_ids = config.stop_token_ids
        self.block_manager = BlockManager(config)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        # Time at previous scheduling step
        self.prev_time = 0.0
        # Did we schedule a prompt at previous step?
        self.prev_prompt = False
        # Latency of the last prompt step
        self.last_prompt_latency = 0.0
        self.delay_factor = config.scheduler_delay_factor

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

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def extend(self, seqs: list[Sequence]):
        self.waiting.extend(seqs)

    def schedule(self) -> tuple[ScheduledBatch, dict[int, Sequence]]:
        # prefill
        scheduled_seqs = {}
        num_seqs_prefill = 0
        num_batched_tokens = 0

        num_scheduled_tokens: list[int] = []
        scheduled_spec_decode_tokens: dict[int, np.ndarray] = {}

        if not self.running and not self.waiting:
            # self.block_manager.reset()
            return None

        while (
            (self.delay_factor <= 0 or self._passed_delay(time.time()))
            and self.waiting
            and num_seqs_prefill < self.max_num_seqs
        ):
            seq = self.waiting[0]
            num_new_tokens = seq.num_tokens - seq.num_cached_tokens
            if (
                num_batched_tokens + num_new_tokens > self.max_num_batched_tokens
                or not self.block_manager.can_allocate(seq)
            ):
                break
            num_seqs_prefill += 1
            self.block_manager.allocate(seq)
            # Recalculate after allocation: prefix caching may have updated
            # seq.num_cached_tokens, reducing the actual number of new tokens.
            num_new_tokens = seq.num_tokens - seq.num_cached_tokens
            if self.cache_stats:
                self.cache_stats.update(seq.num_cached_tokens, seq.num_tokens)
            num_batched_tokens += num_new_tokens
            seq.status = SequenceStatus.RUNNING
            seq.type = SequenceType.PREFILL
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs[seq.id] = seq
            num_scheduled_tokens.append(num_new_tokens)

        num_scheduled_tokens_np = num_scheduled_tokens
        total_tokens_num_prefill = sum(num_scheduled_tokens_np)

        if num_seqs_prefill > 0:
            logger.info(
                f"Scheduled prefill batch: {num_seqs_prefill} reqs, {total_tokens_num_prefill} token_nums: {num_scheduled_tokens}, req_ids: {tuple(scheduled_seqs.keys())}"
            )
            self.prev_prompt = True
            # lip: TODO for prefill/decode mixed batch
            return (
                ScheduledBatch(
                    seqs=scheduled_seqs,
                    num_scheduled_tokens=num_scheduled_tokens_np,
                    total_tokens_num=total_tokens_num_prefill,
                    total_tokens_num_prefill=total_tokens_num_prefill,
                    total_seqs_num=num_seqs_prefill,
                    total_seqs_num_prefill=num_seqs_prefill,
                ),
                scheduled_seqs,
            )

        # decode
        num_seqs_decode = 0
        while self.running and num_seqs_decode < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq, self.mtp_k + 1):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                if seq.spec_token_ids.size > 0:
                    scheduled_spec_decode_tokens[seq.id] = seq.spec_token_ids
                num_seqs_decode += 1
                num_new_tokens = self.mtp_k + 1
                self.block_manager.may_append(seq, num_new_tokens)
                scheduled_seqs[seq.id] = seq
                seq.type = SequenceType.DECODE
                num_scheduled_tokens.append(num_new_tokens)

        num_scheduled_tokens_np = num_scheduled_tokens
        total_tokens_num_decode = sum(num_scheduled_tokens_np)

        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs.values()))
        # logger.info(
        #     f"Scheduled decode batch: {num_seqs_decode} reqs, {total_tokens_num_decode} tokens, req_ids: {tuple(scheduled_seqs.keys())}"
        # )
        return (
            ScheduledBatch(
                seqs=scheduled_seqs,
                num_scheduled_tokens=num_scheduled_tokens_np,
                total_tokens_num=total_tokens_num_decode,
                total_tokens_num_decode=total_tokens_num_decode,
                total_seqs_num=num_seqs_prefill + num_seqs_decode,
                total_seqs_num_prefill=num_seqs_prefill,
                total_seqs_num_decode=num_seqs_decode,
                num_spec_step=self.mtp_k,
                scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            ),
            scheduled_seqs,
        )

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(
        self,
        seqs: list[Sequence],
        fwd_output: ScheduledBatchOutput,
        stream_output_queue=None,
    ) -> list[Sequence]:
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
                )
                # Store sequence ID instead of sequence object to avoid pickling issues
                stream_outputs.append((seq.id, request_output))
                logger.debug(
                    f"Scheduler: Created stream output for seq_id={seq.id}, tokens={new_tokens}, finished={leave_reason is not None}"
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
            self.block_manager.deallocate(seq)
            self.running.remove(seq)
        return finished_seqs

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

    def get_next_batch_info(self) -> tuple[bool, int]:
        if self.waiting:
            # new request is waiting, will do prefill
            seq = self.waiting[0]
            num_tokens = seq.num_tokens - seq.num_cached_tokens
            return (True, num_tokens)
        elif self.running:
            # decode
            num_tokens = len(self.running)
            return (False, num_tokens)
        else:
            # No requests
            return (False, 0)

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

    def __init__(self, config: Config, disagg_cu_shm_name: str = ""):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.block_manager = None  # blocks managed by decode process
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        # spec decode not used on prefill side
        self.use_spec = False
        self.mtp_k = 0
        self.spec_stats = None
        self.cache_stats = None

        # Shared memory for dynamic CU partitioning.
        # Layout: [0:4] = prefill_tokens (uint32), [4:8] = decode_batch (uint32).
        self._cu_shm = None
        if disagg_cu_shm_name:
            import multiprocessing.shared_memory

            self._cu_shm = multiprocessing.shared_memory.SharedMemory(
                name=disagg_cu_shm_name, create=False
            )

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def has_requests(self) -> bool:
        return bool(self.waiting) or bool(self.running)

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def extend(self, seqs: list):
        self.waiting.extend(seqs)

    def schedule(self):
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

        # Read the CU fraction chosen by the DecodeScheduler via shared memory.
        # 0.0 in shm means "no mask" → map to None (the stream pool key).

        cu_fraction = None
        # if self._cu_shm is not None:
        #     raw = struct.unpack_from("f", self._cu_shm.buf, 0)[0]
        #     if raw > 0.0:
        #         cu_fraction = raw

        # logger.info(
        #     f"[PrefillScheduler] scheduled {num_seqs} seqs, "
        #     f"{num_batched_tokens} tokens, req_ids: {tuple(scheduled_seqs.keys())}"
        # )
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

    def postprocess(self, seqs, fwd_output, stream_output_queue=None) -> list:
        """No-op: prefill produces no sampled tokens."""
        return []

    def get_next_batch_info(self) -> tuple:
        if self.waiting:
            seq = self.waiting[0]
            return (True, seq.num_tokens - seq.num_cached_tokens)
        return (False, 0)

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running)


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

    def __init__(self, config: Config, disagg_cu_shm_name: str = ""):
        super().__init__(config)
        # seq_id → Sequence; blocks allocated, BlockAssignment sent, awaiting PrefillDone.
        self.prefill_waiting: dict[int, Sequence] = {}

        # Shared memory for dynamic CU partitioning.
        self._cu_shm = None
        # if disagg_cu_shm_name:
        #     import multiprocessing.shared_memory

        #     self._cu_shm = multiprocessing.shared_memory.SharedMemory(
        #         name=disagg_cu_shm_name, create=False
        #     )

        # Protects prefill_waiting and running: on_prefill_done is called
        # from the _recv_prefill_done background thread.
        self._prefill_lock = threading.Lock()

    def is_finished(self) -> bool:
        return not self.waiting and not self.prefill_waiting and not self.running

    def has_requests(self) -> bool:
        return bool(self.waiting or self.prefill_waiting or self.running)

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.prefill_waiting) + len(self.running)

    def allocate_waiting(self) -> list[Sequence]:
        """Allocate KV blocks for sequences in waiting; move them to prefill_waiting.

        Returns newly allocated sequences so DecodeEngineCore can send a
        BlockAssignment message to the prefill process for each one.
        Called from the main busy_loop thread only.
        """
        newly_allocated = []
        while self.waiting:
            seq = self.waiting[0]
            if not self.block_manager.can_allocate(seq):
                break
            self.block_manager.allocate(seq)
            self.waiting.popleft()
            with self._prefill_lock:
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
        with self._prefill_lock:
            seq = self.prefill_waiting.pop(seq_id, None)
            if seq is not None:
                seq.num_cached_tokens = num_tokens_computed
                seq.status = SequenceStatus.RUNNING
                seq.type = SequenceType.DECODE
                seq.append_token(sampled_token_id)
                self.running.append(seq)

    def schedule(self):
        """Schedule decode-only batches.

        Sequences are promoted directly from prefill_waiting to running by
        on_prefill_done(); this method only schedules the running queue.
        """
        if not self.running:
            return None

        scheduled_seqs: dict[int, Sequence] = {}
        num_scheduled_tokens: list[int] = []
        scheduled_spec_decode_tokens: dict[int, np.ndarray] = {}

        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq, self.mtp_k + 1):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                if seq.spec_token_ids.size > 0:
                    scheduled_spec_decode_tokens[seq.id] = seq.spec_token_ids
                num_new_tokens = self.mtp_k + 1
                self.block_manager.may_append(seq, num_new_tokens)
                scheduled_seqs[seq.id] = seq
                seq.type = SequenceType.DECODE
                num_scheduled_tokens.append(num_new_tokens)

        if not scheduled_seqs:
            return None

        total_tokens_num_decode = sum(num_scheduled_tokens)

        # Dynamic CU partitioning: decode decides the fraction based on its
        # batch size and the total tokens queued for prefill, then writes the
        # fraction to shared memory for PrefillScheduler to read.

        cu_fraction = None
        # if self._cu_shm is not None:
        #     prefill_waiting_tokens = sum(
        #         seq.num_tokens for seq in self.prefill_waiting.values()
        #     )
        #     raw = _optimal_cu_fraction(len(scheduled_seqs), prefill_waiting_tokens)
        #     # Write to shm for prefill to read. 0.0 means "no mask".
        #     struct.pack_into("f", self._cu_shm.buf, 0, raw or 0.0)
        #     # Stream pool is keyed by None (no mask) or a positive float.
        #     cu_fraction = raw if raw and raw > 0.0 else None

        self.running.extendleft(reversed(scheduled_seqs.values()))
        return (
            ScheduledBatch(
                seqs=scheduled_seqs,
                num_scheduled_tokens=num_scheduled_tokens,
                total_tokens_num=total_tokens_num_decode,
                total_tokens_num_decode=total_tokens_num_decode,
                total_seqs_num=len(scheduled_seqs),
                total_seqs_num_decode=len(scheduled_seqs),
                num_spec_step=self.mtp_k,
                scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
                cu_stream_fraction=cu_fraction,
            ),
            scheduled_seqs,
        )

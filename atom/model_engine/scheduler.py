# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
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
        self._log_interval = log_interval
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

        if self.total_draft_tokens % self._log_interval < self.mtp_k:
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
        # Interval stats
        iv_steps = sum(self._interval_distribution.values())
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

        # logger.info(f"{[el for el in scheduled_spec_decode_tokens.keys()]=}")
        # logger.info(f"{self.num_scheduled_tokens=}")
        # logger.info(f"{self.context_lens=}")
        # logger.info(f"{[len(blk)*16 for blk in self.block_tables]=}")
        # logger.info(f"{self.block_tables=}")


class ScheduledBatchOutput:

    def __init__(
        self,
        token_ids: dict[int, tuple[int, ...]],
        num_rejected: np.ndarray,
        num_bonus: np.ndarray,
        draft_token_ids: Optional[np.ndarray],
        # num_bonus_tokens
        is_deferred_out=False,
    ):
        # TODO need refine
        self.is_deferred_out = is_deferred_out
        self.req_ids = list(token_ids.keys())
        self.token_ids = token_ids
        self.draft_token_ids = draft_token_ids
        self.num_rejected = num_rejected
        self.num_bonus = num_bonus
        # logger.info(f"ScheduledBatchOutput: req_ids={self.req_ids}")
        # assert len(self.req_ids) - 1 == len(draft_token_ids)
        # self.num_bonus_tokens = num_bonus_tokens  # num per req


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
        scheduled_spec_decode_tokens: dict[int, list[int]] = {}

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
            # print("schedule seq with token_ids: ", seq.token_ids, flush=True)
            # print("schedule seq with block tables: ", seq.block_table, flush=True)
            while not self.block_manager.can_append(seq):
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
        # print("is prev prefill: ", is_prev_prefill, flush=True)
        # print("fwd_output: ", fwd_output.req_ids, flush=True)
        for seq in self.running:
            # Update the running status
            if seq.id not in fwd_output.req_ids:
                continue
            token_ids = prev_token_ids[seq.id]
            num_new_token = len(token_ids)
            if self.spec_stats:
                self.spec_stats.update(num_new_token)
            idx = fwd_output.req_ids.index(seq.id)
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
                for token_id in token_ids:
                    seq.append_token(token_id)
            new_tokens = token_ids

            if self.mtp_k > 0:
                idx = fwd_output.req_ids.index(seq.id)
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

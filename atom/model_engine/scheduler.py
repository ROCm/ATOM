# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
import time
from collections import deque

import numpy as np
from atom.config import Config
from atom.model_engine.block_manager import BlockManager
from atom.model_engine.request import RequestOutput
from atom.model_engine.sequence import Sequence, SequenceStatus, SequenceType

logger = logging.getLogger("atom")


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
    ):
        # len(seqs) == total_seqs_num == total_seqs_num_prefill + total_seqs_num_decode
        # self.seqs = seqs
        self.req_ids = list(seqs.keys())
        self.scheduled_tokens = [
            seq.token_ids[-num_tokens:]
            for seq, num_tokens in zip(seqs.values(), num_scheduled_tokens)
        ]
        # print(f"{num_scheduled_tokens=}")
        # print(f"{self.scheduled_tokens=}")
        self.temperatures = [seq.temperature for seq in seqs.values()]
        self.context_lens = [seq.num_tokens for seq in seqs.values()]
        self.block_tables = [
            seq.block_table for seq in seqs.values() if seq.block_table
        ]
        self.last_block_num_tokens = [
            seq.last_block_num_tokens for seq in seqs.values()
        ]
        self.num_cached_tokens = [seq.num_cached_tokens for seq in seqs.values()]

        # num_scheduled_tokens for each sequence in the batch
        self.num_scheduled_tokens = num_scheduled_tokens

        # Total number of tokens scheduled for all requests.
        self.total_tokens_num = total_tokens_num
        self.total_tokens_num_prefill = total_tokens_num_prefill
        self.total_tokens_num_decode = total_tokens_num_decode

        # Total number of reqs scheduled for all requests.
        self.total_seqs_num = total_seqs_num
        self.total_seqs_num_prefill = total_seqs_num_prefill
        self.total_seqs_num_decode = total_seqs_num_decode


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

        if not self.running and not self.waiting:
            # self.block_manager.reset()
            return None

        while self.waiting and num_seqs_prefill < self.max_num_seqs:
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
                f"scheduled prefill batch: {num_seqs_prefill} reqs, {total_tokens_num_prefill} tokens"
            )
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
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs_decode += 1
                self.block_manager.may_append(seq)
                num_new_tokens = 1
                scheduled_seqs[seq.id] = seq
                seq.type = SequenceType.DECODE
                num_scheduled_tokens.append(num_new_tokens)

        num_scheduled_tokens_np = num_scheduled_tokens
        total_tokens_num_decode = sum(num_scheduled_tokens_np)

        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs.values()))
        # logger.info(
        #     f"Scheduled decode batch: {num_seqs_decode} reqs, {total_tokens_num_decode} tokens"
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
        prev_token_ids: dict[int, int],
        stream_output_queue=None,
    ) -> list[Sequence]:
        is_deferred_out = prev_token_ids.get(-1, False)
        # update token_ids with the actual sampled token ids
        finished_seqs = []
        stream_outputs = []

        for seq in self.running:
            if seq.id not in prev_token_ids:
                continue
            token_id = prev_token_ids[seq.id]
            new_tokens = []
            if is_deferred_out:
                seq.token_ids[-1] = token_id

                if seq.output_tokens:
                    seq.output_tokens[-1] = token_id
                    new_tokens = [token_id]
                else:
                    seq.output_tokens.append(token_id)
                    new_tokens = [token_id]
            else:
                seq.append_token(token_id)
                new_tokens = [token_id]

            if seq.num_completion_tokens == 1 and seq.first_token_time == 0.0:
                seq.first_token_time = time.time()

            leave_reason = None
            # Check if sequence ends with any stop sequence
            for stop_seq in seq.stop_token_sequences:
                if len(seq.token_ids) >= len(stop_seq):
                    if seq.token_ids[-len(stop_seq) :] == stop_seq:
                        leave_reason = "stop_sequence"
                        break
            else:
                if not seq.ignore_eos and token_id == self.eos_token_id:
                    leave_reason = "eos"
                elif not seq.ignore_eos and token_id in self.stop_token_ids:
                    leave_reason = str(token_id)
                elif seq.num_completion_tokens >= seq.max_tokens:
                    leave_reason = "max_tokens"
            # Prepare stream output
            if stream_output_queue is not None and new_tokens:
                request_output = RequestOutput(
                    request_id=seq.id,
                    output_tokens=new_tokens.copy(),
                    finished=(leave_reason is not None),
                    finish_reason=leave_reason,
                )
                # Store sequence ID instead of sequence object to avoid pickling issues
                stream_outputs.append((seq.id, request_output))
                logger.debug(
                    f"Scheduler: Created stream output for seq_id={seq.id}, tokens={new_tokens}, finished={leave_reason is not None}"
                )

            if leave_reason is not None:
                seq.leave_reason = leave_reason
                seq.status = SequenceStatus.FINISHED
                finished_seqs.append(seq)

        if stream_output_queue is not None and stream_outputs:
            stream_output_queue.put_nowait(stream_outputs)

        if is_deferred_out:
            # placeholder for the each decode step
            for seq in seqs:
                if seq.status == SequenceStatus.RUNNING:
                    seq.append_token(self.eos_token_id)
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

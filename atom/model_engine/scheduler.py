from collections import deque
from dataclasses import dataclass

from atom.config import Config
from atom.model_engine.block_manager import BlockManager
from atom.model_engine.sequence import Sequence, SequenceStatus


@dataclass
class ScheduledBatchs:
    seqs: list[Sequence]
    is_prefill: bool
    has_seq_out: bool


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos_token_id = config.eos_token_id
        self.block_manager = BlockManager(config)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.has_seq_out = False

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> ScheduledBatchs:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        if not self.running and not self.waiting:
            self.block_manager.reset()

        has_seq_out = self.has_seq_out
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(
                seq
            ) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return ScheduledBatchs(
                seqs=scheduled_seqs,
                is_prefill=True,
                has_seq_out=has_seq_out,
            )

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        self.has_seq_out = False
        return ScheduledBatchs(
            seqs=scheduled_seqs,
            is_prefill=False,
            has_seq_out=has_seq_out,
        )

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            leave_reason = None
            if not seq.ignore_eos and token_id == self.eos_token_id:
                leave_reason = "eos"
            elif seq.num_completion_tokens == seq.max_tokens:
                leave_reason = "max_tokens"
            if leave_reason is not None:
                seq.leave_reason = leave_reason
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                self.has_seq_out = True

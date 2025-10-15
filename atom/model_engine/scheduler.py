from collections import deque
from dataclasses import dataclass, field
from typing import Optional, cast, Any

from atom.config import Config
from atom.model_engine.block_manager import BlockManager
from atom.model_engine.sequence import Sequence, SequenceStatus

import torch
import numpy as np

@dataclass
class ScheduledBatchs:
    seqs: list[Sequence]
    is_prefill: bool
    deferred_reqID: list[int]

    unfinished_prev_req: list[Sequence]

    # req_id -> num_scheduled_tokens
    # Number of tokens scheduled for each request.
    num_scheduled_tokens: dict[str, int]
    # Total number of tokens scheduled for all requests.
    # Equal to sum(num_scheduled_tokens.values())
    total_num_scheduled_tokens: int
        # Cached reference to the GPU tensor of previously sampled tokens
    # prev_sampled_token_ids: Optional[torch.Tensor] = None
    # prev_sampled_token_ids_invalid_indices: Optional[set[int]] = None
    # prev_req_id_to_index: Optional[dict[str, int]] = None


class InputBatch:

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int
    ):
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.token_ids_cpu_tensor = torch.zeros(
            (max_num_reqs, max_model_len),
            device="cpu",
            dtype=torch.int32,
            pin_memory=False,
        )
        self.token_ids_cpu = self.token_ids_cpu_tensor.numpy()
        self.num_tokens = np.zeros(max_num_reqs, dtype=np.int32)
        self.req_id_to_index: dict[str, int] = {}

        # Cached reference to the GPU tensor of previously sampled tokens
        self.prev_sampled_token_ids: Optional[torch.Tensor] = None
        self.prev_sampled_token_ids_invalid_indices: Optional[set[int]] = None
        self.prev_req_id_to_index: Optional[dict[str, int]] = None

        self._req_ids: list[Optional[str]] = []
        self.req_output_token_ids: list[Optional[list[int]]] = []

        self.finished_req_ids: set[str] = set()

    @property
    def num_reqs(self) -> int:
        return len(self.req_id_to_index)

    def _register_add_request(self, request: "Sequence") -> int:
        """Track add-request operations for logits processors.
        Not applicable to pooling models.
        """

        new_req_index = self.num_reqs

        assert new_req_index < self.max_num_reqs
        return new_req_index


    def add_request(
        self,
        request: "Sequence",
    ) -> int:
        req_index = self._register_add_request(request)

        req_id = request.id

        if req_index == len(self._req_ids):
            self._req_ids.append(req_id)
            # self.req_output_token_ids.append(request.output_token_ids)
        else:
            self._req_ids[req_index] = req_id
            # self.req_output_token_ids[req_index] = request.output_token_ids
        self.req_id_to_index[req_id] = req_index

        # new_req_index = request.last_token
        # self.req_id_to_index[req_id] = new_req_index

        # self._req_ids.append(req_id)
        # print("req_id:", req_id)

    def remove_request(self, req_id: str) -> Optional[int]:
        """This method must always be followed by a call to condense().

        Args:
          req_id: request to remove

        Returns:
          Removed request index, or `None` if `req_id` not recognized
        """
        req_index = self.req_id_to_index.pop(req_id, None)
        if req_index is None:
            return None

        self._req_ids[req_index] = None
        # self.req_output_token_ids[req_index] = None


    @property
    def req_ids(self) -> list[str]:
        # None elements should only be present transiently
        # while performing state updates to the batch.
        return cast(list[str], self._req_ids)


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.block_manager = BlockManager(config)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.deferred_reqID = []

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def extend(self, seqs: list[Sequence]):
        self.waiting.extend(seqs)

    def schedule(self) -> ScheduledBatchs:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0

        num_scheduled_tokens: dict[str, int] = {}

        if not self.running and not self.waiting:
            self.block_manager.reset()

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
            # in vllm, using num_new_tokens
            num_scheduled_tokens[seq.id] = seq.num_tokens

        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())

        if scheduled_seqs:
            return ScheduledBatchs(
                seqs=scheduled_seqs, is_prefill=True, deferred_reqID=[], unfinished_prev_req=scheduled_seqs,
                num_scheduled_tokens=num_scheduled_tokens,
                total_num_scheduled_tokens=total_num_scheduled_tokens,
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
            num_scheduled_tokens[seq.id] = seq.num_tokens
        
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())

        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return ScheduledBatchs(
            seqs=scheduled_seqs,
            is_prefill=False,
            deferred_reqID=self.deferred_reqID,
            unfinished_prev_req=scheduled_seqs,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
        )

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], prev_token_ids: list[int], prev_seqs: list[Sequence]):
        # # placeholder for the each decode step
        # token_ids = [self.eos_token_id] * len(seqs)
        # for seq, token_id in zip(seqs, prev_token_ids):
        #     seq.append_token(token_id)

        # print('--------------------')
        # for i, seq in enumerate(prev_seqs): 
        #     print("prev_seqs", seq.id)
        #     print(seq.token_ids)
        # for i, seq in enumerate(seqs): 
        #     print("seqs", seq.id)
        #     print(seq.token_ids)
        # print("prev_token_ids", prev_token_ids)

        if not prev_token_ids or not prev_seqs:
            return
        
        # self.unfinished_req = []
        # processed_seqs = set()
        
        # for i, prev_seq in enumerate(prev_seqs):
        #     token_id = prev_token_ids[i]
            
        #     current_seq = None
        #     for seq in seqs:
        #         if seq.id == prev_seq.id:
        #             current_seq = seq
            
        #             if current_seq and current_seq.id not in processed_seqs:
        #                 processed_seqs.add(current_seq.id)
                        
        #                 # Check if this token has been used before (to avoid outputting a word repeatedly)
        #                 if (len(current_seq.token_ids) > prev_seq.num_tokens and
        #                     current_seq.token_ids[-1] == token_id):
        #                     print(f"Token {token_id} already applied to sequence {current_seq.id}")
        #                     continue
                            
        #                 current_seq.append_token(token_id)
        #                 print(f"Applied token {token_id} to sequence {current_seq.id}")
                        
        #                 leave_reason = None
        #                 if not current_seq.ignore_eos and token_id == self.eos_token_id:
        #                     leave_reason = "eos"
        #                 elif current_seq.num_completion_tokens == current_seq.max_tokens:
        #                     leave_reason = "max_tokens"
                            
        #                 if leave_reason is not None:
        #                     current_seq.leave_reason = leave_reason
        #                     current_seq.status = SequenceStatus.FINISHED
        #                     self.block_manager.deallocate(current_seq)
        #                     self.running.remove(current_seq)
        #                 else:
        #                     self.unfinished_req.append(i)

        seq_id_to_index = {seq.id: i for i, seq in enumerate(prev_seqs)}
        self.unfinished_req = []
        # prev_token_ids = out.get_output()
        # worker_response_mq.dequeue(
        token_ids = prev_token_ids
        for i, (seq, token_id) in enumerate(zip(seqs, token_ids)):
            # if seq.id in prev_seqs:
            #     seq.append_token(token_id)
            seq.token_ids[-2] = token_id
            leave_reason = None
            
            # Check if sequence ends with any stop sequence
            stop_matched = False
            for stop_seq in seq.stop_token_sequences:
                if len(seq.token_ids) >= len(stop_seq):
                    if seq.token_ids[-len(stop_seq):] == stop_seq:
                        stop_matched = True
                        break
            
            if (not seq.ignore_eos and token_id == self.eos_token_id) or stop_matched:
                leave_reason = "eos"
            elif seq.num_completion_tokens == seq.max_tokens:
                leave_reason = "max_tokens"
            if leave_reason is not None:
                seq.leave_reason = leave_reason
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
            else:
                self.unfinished_req.append(i)

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from copy import copy
from enum import Enum, auto
from itertools import count
from typing import Any, Callable, Optional

from atom.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()
    EXIT_ENGINE = auto()


class SequenceType(Enum):
    DUMMY = auto()
    PREFILL = auto()
    DECODE = auto()


def get_exit_sequence():
    exit_seq = Sequence([-1], 1)
    exit_seq.status = SequenceStatus.EXIT_ENGINE
    return exit_seq


class Sequence:
    counter = count()

    def __init__(
        self,
        token_ids: list[int],
        block_size: int,
        sampling_params=SamplingParams(),
        stop_token_sequences: list[list[int]] = None,
        stream_callback: Optional[Callable[[Any], None]] = None,
        id=None,
    ):
        self.block_size = block_size
        self.id = id or next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.type = SequenceType.DUMMY
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.stop_strings = sampling_params.stop_strings
        self.stop_token_sequences = stop_token_sequences or []

        # stream callback
        self.stream_callback = stream_callback
        self.output_tokens = []  # cache for newly generate tokens

        # save speculative tokens if is_deferred_output = False or prefill is inter
        self.spec_token_ids: list[int] = []

        # statistics fields
        self.arrive_time = 0.0
        self.first_token_time = 0.0
        self.leave_time = 0.0
        self.leave_reason = ""

    def __len__(self):
        return self._num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def num_tokens(self):
        """The total number of tokens in the sequence. i.e. prompt + completion"""
        return self._num_tokens

    @num_tokens.setter
    def num_tokens(self, value):
        self._num_tokens = value
        self.num_blocks = (value + self.block_size - 1) // self.block_size
        self.last_block_num_tokens = (
            self._num_tokens - (self.num_blocks - 1) * self.block_size
        )

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[: self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    # @property
    # def num_blocks(self):
    #     return (self.num_tokens + self.block_size - 1) // self.block_size

    # @property
    # def last_block_num_tokens(self):
    #     return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size : (i + 1) * self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.output_tokens.append(token_id)
        self.num_tokens += 1

    # def __getstate__(self):
    #     return (
    #         self.num_tokens,
    #         self.num_prompt_tokens,
    #         self.num_cached_tokens,
    #         self.block_table,
    #         self.token_ids if self.num_completion_tokens == 0 else self.last_token,
    #     )

    # def __setstate__(self, state):
    #     (
    #         self.num_tokens,
    #         self.num_prompt_tokens,
    #         self.num_cached_tokens,
    #         self.block_table,
    #     ) = state[:-1]
    #     if self.num_completion_tokens == 0:
    #         self.token_ids = state[-1]
    #     else:
    #         self.last_token = state[-1]

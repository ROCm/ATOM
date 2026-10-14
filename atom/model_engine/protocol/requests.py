# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""What the client side holds instead of a `Sequence`.

`EngineRequest` is a request as the frontend built it: the prompt it tokenized
and the arguments it chose, kept as-is until they are written to the wire. The
`Sequence` it describes is built by the EngineCore that admits it.

`CompletionResult` is a finished sequence as the frontend reads it.

Both answer to the attribute names the frontend already used on `Sequence`
(`id`, `num_prompt_tokens`, `max_tokens`, `completion_token_ids`, ...), so the
code that hands them around did not have to learn a second vocabulary.
"""

import array

import numpy as np

from atom.sampling_params import SamplingParams


def prompt_token_array(tokens) -> array.array:
    """`tokens` as an `array("i")`, without a copy when it already is one."""
    if isinstance(tokens, array.array) and tokens.typecode == "i":
        return tokens
    if isinstance(tokens, np.ndarray):
        out = array.array("i")
        out.frombytes(
            memoryview(np.ascontiguousarray(tokens, dtype=np.int32)).cast("B")
        )
        return out
    return array.array("i", tokens)


class EngineRequest:
    """One sequence to submit; mirrors the `Sequence` constructor."""

    def __init__(
        self,
        *,
        id: int,
        prompt_token_ids,
        sampling_params: SamplingParams,
        stop_token_sequences: list[list[int]] | None = None,
        stream_callback=None,
        kv_transfer_params: dict | None = None,
        multimodal_data: dict | None = None,
        cache_seed: int | None = None,
        mrope_positions=None,
        mrope_position_delta: int = 0,
        needs_independent_noise: bool = False,
        parent_request_id: str | None = None,
        sibling_index: int = 0,
        external_request_id: str | None = None,
        data_parallel_rank: int | None = None,
        dp_session_id: str | None = None,
        dp_parent_session_id: str | None = None,
        arrive_time: float = 0.0,
    ):
        self.id = id
        # Siblings of one fan-out share this array; nothing writes to it.
        self.prompt_token_ids = prompt_token_array(prompt_token_ids)
        self.sampling_params = sampling_params
        # Its own attribute, as on `Sequence`: what goes on the wire, and what
        # DisaggCoreManager lowers to 1 for the prefill half.
        self.max_tokens = sampling_params.max_tokens
        self.stop_token_sequences = stop_token_sequences or []
        self.stream_callback = stream_callback
        self.kv_transfer_params = kv_transfer_params
        self.multimodal_data = multimodal_data
        self.cache_seed = cache_seed
        self.mrope_positions = mrope_positions
        self.mrope_position_delta = mrope_position_delta
        self.needs_independent_noise = needs_independent_noise
        self.parent_request_id = parent_request_id
        self.sibling_index = sibling_index
        self.external_request_id = external_request_id
        self.data_parallel_rank = data_parallel_rank
        self.dp_session_id = dp_session_id
        self.dp_parent_session_id = dp_parent_session_id
        self.arrive_time = arrive_time

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    def __repr__(self) -> str:
        return (
            f"EngineRequest(id={self.id}, prompt_tokens={self.num_prompt_tokens}, "
            f"max_tokens={self.max_tokens})"
        )


class CompletionResult:
    """A finished sequence: the fields of it a client reads."""

    def __init__(
        self,
        *,
        id: int,
        completion_token_ids: array.array | None = None,
        logprobs: array.array | None = None,
        return_logprobs: bool = False,
        num_prompt_tokens: int = 0,
        num_cached_tokens: int = 0,
        arrive_time: float = 0.0,
        first_token_time: float = 0.0,
        leave_time: float = 0.0,
        leave_reason: str = "",
        kv_transfer_params_output: dict | None = None,
        external_request_id: str | None = None,
        parent_request_id: str | None = None,
        sibling_index: int = 0,
    ):
        self.id = id
        self.completion_token_ids = (
            array.array("i") if completion_token_ids is None else completion_token_ids
        )
        self.logprobs = array.array("d") if logprobs is None else logprobs
        self.return_logprobs = return_logprobs
        self.num_prompt_tokens = num_prompt_tokens
        self.num_cached_tokens = num_cached_tokens
        self.arrive_time = arrive_time
        self.first_token_time = first_token_time
        self.leave_time = leave_time
        self.leave_reason = leave_reason
        self.kv_transfer_params_output = kv_transfer_params_output
        self.external_request_id = external_request_id
        self.parent_request_id = parent_request_id
        self.sibling_index = sibling_index

    @property
    def num_completion_tokens(self) -> int:
        return len(self.completion_token_ids)

    @property
    def num_tokens(self) -> int:
        return self.num_prompt_tokens + self.num_completion_tokens

    def __repr__(self) -> str:
        return (
            f"CompletionResult(id={self.id}, tokens={self.num_completion_tokens}, "
            f"leave_reason={self.leave_reason!r})"
        )

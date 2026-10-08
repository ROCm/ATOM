# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Small request and result objects for the Python engine boundary."""

import array
import weakref
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from atom.sampling_params import SamplingParams

from .buffer_types import PromptHandle


@dataclass(slots=True)
class AdmissionRequest:
    id: int
    prompt: PromptHandle
    sampling_params: SamplingParams
    stop_token_sequences: list[list[int]] = field(default_factory=list)
    external_request_id: str | None = None
    kv_transfer_params: dict | None = None
    needs_independent_noise: bool = False
    parent_request_id: str | None = None
    sibling_index: int = 0
    multimodal_data: dict | None = None
    mrope_positions: Any = None
    mrope_position_delta: int = 0
    data_parallel_rank: int | None = None
    dp_session_id: str | None = None
    dp_parent_session_id: str | None = None
    arrive_time: float = 0.0
    received_at: float = 0.0
    admission_id: str = ""
    submission_sequence: int = 0
    # `prefix_hash.ShippedChain` over the prompt's full blocks, or None.
    prompt_hash_chain: Any = field(default=None, repr=False)
    # The request span this admission's engine spans hang under, as
    # `tracing.bind_sequences` names it on a sequence.
    _atom_trace_carrier: dict | None = field(default=None, repr=False)
    _atom_trace_model: str | None = field(default=None, repr=False)
    _atom_trace_request_id: str | None = field(default=None, repr=False)
    _atom_trace_timing: Any = field(default=None, repr=False, compare=False)
    _state_ref: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if not self.admission_id:
            self.admission_id = str(self.id)

    @property
    def num_prompt_tokens(self):
        return self.prompt.token_count

    @property
    def num_tokens(self):
        return self.num_prompt_tokens

    def __len__(self):
        return self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.prompt.view

    @property
    def cache_seed(self):
        """`Sequence.cache_seed`, which the engine derives the same way.

        Kept on `multimodal_data`, so the media is hashed once and the engine
        receives the seed instead of hashing it again.
        """
        if self.multimodal_data is None:
            return -1
        seed = self.multimodal_data.get("cache_seed")
        if seed is None:
            from atom.model_engine.multimodal_runtime import multimodal_cache_seed

            seed = self.multimodal_data["cache_seed"] = multimodal_cache_seed(
                self.multimodal_data
            )
        return seed

    @property
    def max_tokens(self):
        return self.sampling_params.max_tokens

    @max_tokens.setter
    def max_tokens(self, value):
        self.sampling_params.max_tokens = value

    @property
    def request_state(self):
        return self._state_ref() if self._state_ref is not None else None

    @property
    def stream_callback(self):
        state = self.request_state
        return state.callback if state is not None else None

    @property
    def callback_only(self):
        state = self.request_state
        return state.callback_only if state is not None else False

    @callback_only.setter
    def callback_only(self, value):
        state = self.request_state
        if state is not None:
            state.callback_only = bool(value)

    def release_prompt(self):
        self.prompt.release()


IncomingAdmission = AdmissionRequest


@dataclass
class RequestState:
    admission: AdmissionRequest
    callback: Any = None
    callback_only: bool = False
    actual_target_ranks: tuple[int, ...] = ()
    submission_state: str = "pending"
    prepared_submission: Any = None
    frontend_prompt_lease: PromptHandle | None = None
    prefill_load_released: bool = False
    terminal_callback_delivered: bool = False
    result_delivered: bool = False

    def __post_init__(self):
        self.admission._state_ref = weakref.ref(self)
        if self.frontend_prompt_lease is None:
            self.frontend_prompt_lease = self.admission.prompt.acquire()

    @property
    def id(self):
        return self.admission.id

    def release_frontend(self):
        lease, self.frontend_prompt_lease = self.frontend_prompt_lease, None
        if lease is not None:
            lease.release()

    def __del__(self):
        self.release_frontend()


@dataclass(slots=True)
class CompletionResult:
    id: int
    completion_token_ids: Any = field(default_factory=lambda: array.array("i"))
    logprobs: Any = field(default_factory=lambda: array.array("d"))
    return_logprobs: bool = False
    num_prompt_tokens: int = 0
    num_cached_tokens: int = 0
    arrive_time: float = 0.0
    first_token_time: float = 0.0
    leave_time: float = 0.0
    leave_reason: str | None = ""
    kv_transfer_params_output: dict | None = None
    external_request_id: str | None = None
    parent_request_id: str | None = None
    sibling_index: int = 0
    prefill_start_time: float = 0.0
    prefill_end_time: float = 0.0

    @classmethod
    def from_sequence(cls, seq):
        tokens = seq.completion_token_ids
        if not isinstance(tokens, array.array):
            tokens = array.array("i", tokens)
        return cls(
            id=seq.id,
            completion_token_ids=tokens,
            logprobs=seq.logprobs[: len(tokens)],
            return_logprobs=seq.return_logprobs,
            num_prompt_tokens=seq.num_prompt_tokens,
            num_cached_tokens=seq.num_cached_tokens,
            arrive_time=seq.arrive_time,
            first_token_time=seq.first_token_time,
            leave_time=seq.leave_time,
            leave_reason=seq.leave_reason,
            kv_transfer_params_output=deepcopy(seq.kv_transfer_params_output),
            external_request_id=seq.external_request_id,
            parent_request_id=seq.parent_request_id,
            sibling_index=seq.sibling_index,
            prefill_start_time=getattr(seq, "prefill_start_time", 0.0),
            prefill_end_time=getattr(seq, "prefill_end_time", 0.0),
        )

    @property
    def num_completion_tokens(self):
        return len(self.completion_token_ids)

    @property
    def num_tokens(self):
        return self.num_prompt_tokens + self.num_completion_tokens

    @property
    def output_tokens(self):
        return self.completion_token_ids

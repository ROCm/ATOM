# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from collections.abc import Callable
from copy import copy
from enum import Enum, auto
from itertools import count
from typing import Any

import numpy as np

from atom.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING_FOR_REMOTE_KVS = auto()
    WAITING = auto()
    RUNNING = auto()
    # Client disconnected: the seq is still live (its KV must be freed via the
    # normal stop path). The scheduler finishes it at the next step (running) or
    # drops it when popped from `waiting`. Distinct from FINISHED so it still
    # rides one cleanup pass; is_finished() stays False until then.
    ABORTED = auto()
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
        sampling_params: SamplingParams | None = None,
        stop_token_sequences: list[list[int]] | None = None,
        stream_callback: Callable[[Any], None] | None = None,
        id=None,
        kv_transfer_params: dict | None = None,
        num_draft_tokens: int = 0,
        has_per_req_cache: bool = False,
        needs_independent_noise: bool = False,
        parent_request_id: str | None = None,
        sibling_index: int = 0,
        request_id: str | None = None,
        multimodal_data: dict | None = None,
        mrope_positions: np.ndarray | None = None,
        mrope_position_delta: int = 0,
        data_parallel_rank: int | None = None,
    ):
        # Built here rather than as a default argument: one instance shared by
        # every defaulting Sequence would be a mutable default in all but name.
        if sampling_params is None:
            sampling_params = SamplingParams()
        self.block_size = block_size
        self.id = id or next(Sequence.counter)
        self.external_request_id = request_id
        self.status = SequenceStatus.WAITING
        self.type = SequenceType.DUMMY
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_draft_tokens = num_draft_tokens
        # `has_per_req_cache=True` means this seq's attention type maintains
        # a per-request stateful buffer outside the paged KV pool (e.g. GDN
        # recurrent state, future DeepseekV4 ring-buffer + compressor state).
        # Triggers BlockManager to allocate a per-req cache slot in
        # allocate() / free it in deallocate().
        self.has_per_req_cache = has_per_req_cache
        self.multimodal_data = multimodal_data
        self.mrope_positions = mrope_positions
        self.mrope_position_delta = mrope_position_delta
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_rejected = 0
        self.num_cached_tokens = 0
        # Tokens whose blocks are registered in the prefix cache: through the
        # prompt as chunks finalize, then on through decode as generated blocks
        # fill (BlockManager.hash_decode_blocks). Distinct from
        # `num_cached_tokens`, which means "KV computed" and stops at the
        # prompt; this means "content hash published". They part ways the moment
        # generation starts.
        self.num_hashed_tokens = 0
        self.num_compressed_hit_blocks = 0
        # The same hit asked counterfactually: how far it would have reached
        # with a state checkpoint at every boundary. Equal to the admitted hit
        # when nothing was lost to a missing checkpoint, so the difference is
        # the reuse a checkpoint would have delivered — what CacheStats reports
        # as recoverable.
        self.num_wanted_hit_blocks = 0
        # That gap as a prompt position, once it is worth a forward: the one
        # place off the checkpoint grid where this seq's prefill is cut so a
        # checkpoint can be kept. 0 = nowhere. Both written by
        # `BlockManager._record_checkpoint_demand` at admission; this one is
        # read by `checkpoint_cut` and `checkpointers_at`, which must agree.
        self.checkpoint_demand_pos = 0
        # Where this seq last kept a checkpoint. Prefill lands on the grid so
        # this tracks it, but a speculative decode step lands wherever
        # `1 + accepted` puts it, and there the grid is unreachable — see
        # `BlockManager.checkpointers_at`, which measures spacing from here.
        self.last_checkpoint_pos = 0
        self.prefix_cache_hit_tokens = 0
        # True iff this seq is mid-prefill (chunked prefill produced KV for
        # some prompt tokens but not all). Maintained by the scheduler:
        # set in postprocess when an advance leaves prompt tokens remaining,
        # cleared when prefill completes or seq is preempted. Used to discard
        # garbage sampled tokens from intermediate chunks and to skip the
        # scheduler's Phase 1 scan when no partials exist.
        self.is_partial_prefill = False
        self.block_table = []
        # Per-request cache slot index (filled by BlockManager.allocate()).
        # -1 = unallocated. The slot indexes into the per-req cache tensors
        # owned by ModelRunner (e.g. mamba_k_cache for GDN).
        self.per_req_cache_group = -1
        # Group the NEXT forward reads its incoming state from, when that is not
        # the group it writes (`per_req_cache_group`). Set by BlockManager on a
        # state fork — resuming from a checkpoint, or taking one — and
        # cleared by the scheduler once a batch has carried it, so it describes
        # exactly one forward. -1 = read and write the same group, the case for
        # every step in between.
        self.state_fork_src = -1
        # Content hash of a boundary this seq's last forward landed on and whose
        # state is worth keeping, for state classes that checkpoint by copying
        # (`StateGroupPool.checkpoint`). The copy needs a forward to issue it, so
        # the intent outlives the step that formed it; `StateGroupPool.take_copies`
        # turns it into a destination group and a copy pair when the next batch
        # is built.
        # -1 = nothing pending, which is also what `deallocate` restores.
        self.pending_checkpoint = -1
        # Content hash of a checkpoint the offload tier has been asked to fetch
        # back into `per_req_cache_group`, or -1. Set by
        # `BlockManager._attach_state_group` when the boundary this seq hit
        # lives in LMCache rather than HBM; it is what tells the scheduler to
        # park the request, and what tells the failure path that
        # `num_cached_tokens` is claiming state nobody delivered.
        self.state_load_hash = -1
        self.temperature = sampling_params.temperature
        self.top_k = sampling_params.top_k
        self.top_p = sampling_params.top_p
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.stop_strings = sampling_params.stop_strings
        self.stop_token_sequences = stop_token_sequences or []
        self.is_first_decode = False
        # Set to True by Scheduler.postprocess after BlockManager.hash_blocks
        # has registered the prompt blocks for prefix caching. The trigger has
        # to be per-seq because in deferred-output mode the prefill step's
        # postprocess has no fwd_output entry for the seq (idx is None) — the
        # prefill output surfaces one step later, at which point seq.type has
        # already been flipped to DECODE. A seq.type / len(output_tokens) gate
        # would never fire for the prefill blocks; this flag does.
        self.prefix_hashes_published = False
        self.return_logprobs = bool(getattr(sampling_params, "logprobs", False))
        self.logprobs: list[float] = []
        # stream callback
        self.stream_callback = stream_callback
        self.output_tokens = []  # cache for newly generate tokens

        # save speculative tokens if is_deferred_output = False or prefill is inter
        self.spec_token_ids: np.ndarray = np.array([], dtype=np.int32)

        # DSpark Phase 2: scheduler-chosen verify length from the previous
        # decode step's propose(). None = no schedule yet -> verify mtp_k (full).
        # Next decode step sizes this seq's verification to dspark_next_ell+1.
        self.dspark_next_ell: int | None = None

        # statistics fields
        self.arrive_time = 0.0
        self.first_token_time = 0.0
        self.leave_time = 0.0
        self.leave_reason = ""

        # kv_transfer params
        self.kv_transfer_params = kv_transfer_params
        self.kv_transfer_params_output = None
        if kv_transfer_params:
            self.prefix_cache_hit_tokens = kv_transfer_params.get(
                "prefix_cache_hit_tokens", 0
            )

        self.prefix_cache_hit_tokens = (kv_transfer_params or {}).get(
            "prefix_cache_hit_tokens", 0
        )

        # accepted tokens for spec decode
        self.num_bonus_tokens = 0

        # Fan-out bookkeeping for SamplingParams.n > 1. When True, the sampler
        # must produce fresh, per-row random noise for this sequence instead
        # of reusing the cached shared exponential tensor, otherwise sibling
        # sequences with identical logits would emit identical tokens.
        self.needs_independent_noise = needs_independent_noise
        # Parent request id (user-facing id from the API layer) and this
        # sequence's index within the fan-out group [0, n). Both default
        # to safe values for single-sample requests.
        self.parent_request_id = parent_request_id
        self.sibling_index = sibling_index
        # Explicitly requested DP rank, e.g. for cache aware DP routing.
        # Consumed by CoreManager._dispatch_to_dp_ranks as a routing hint.
        self.data_parallel_rank = data_parallel_rank

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
        return self.token_ids[self.num_prompt_tokens : self.num_tokens]

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

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import array
from collections.abc import Callable
from enum import Enum, auto
from itertools import count
from typing import Any

import numpy as np

from atom.sampling_params import SamplingParams


def new_token_ids(token_ids=()) -> array.array:
    """A sequence's token ids.

    An `array("i")` rather than a list for two reasons that both scale with
    context. The scheduler copies a slice of this into the flat
    `scheduled_tokens` buffer every step -- a whole chunk of it on a prefill --
    and from a list that is one CPython int unboxing per token, 0.28 ms for a
    16k chunk against 0.001 ms from an array. And a list of 100k ids costs
    3.4 MiB of boxed ints where the array costs 0.38.

    Behaves as a list for append/pop/index/len/iterate/slice-delete, and holds
    negative ids (the exit sentinel is -1). It does NOT compare equal to a
    list, which is why `stop_token_sequences` below is converted too, and why
    `BlockManager.compute_hash` pins its dtype.
    """
    return array.array("i", token_ids)


def new_block_table(block_ids=()) -> array.array:
    """A sequence's physical block ids.

    An `array("i")` rather than a list because every forward marshals these
    into the int32 `block_tables` buffer, where a list costs one CPython int
    unboxing per block (~17k per step at 50 seqs x 100k ctx) and an array is a
    memcpy. It behaves as a list for append/pop/index/len/iterate; it has no
    `.clear()` (use `del bt[:]`) and no `.copy()` (use `list(bt)`).
    """
    return array.array("i", block_ids)


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
        dp_session_id: str | None = None,
        dp_parent_session_id: str | None = None,
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
        self.token_ids = new_token_ids(token_ids)
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
        # Which of the two demand counters this seq has already been put
        # against. `can_allocate` re-runs for a sequence the queue keeps
        # deferring, and the position above cannot serve as the marker for
        # either one: a declined demand writes 0 back, so it does not remember
        # the decline, and a decline retracts a demand the recorded counter had
        # already taken. Both are cleared by `deallocate`, so a re-admitted
        # request counts again, which it should.
        self.checkpoint_demand_counted = False
        self.checkpoint_demand_declined = False
        # The demand's sibling: this prompt's own end, floored to the hash
        # grid. 0 = nowhere.
        #
        # The demand is reactive — it only exists once a hit has already been
        # refused for want of a checkpoint, which is one request too late for
        # the position that serves the *next* turn of a conversation. On
        # agentic traffic that position is where nearly all the reuse is (see
        # `BlockManager._record_checkpoint_end`), so it is reserved up front
        # rather than waited for.
        #
        # Written by `BlockManager._record_checkpoint_end` at admission, read
        # by `checkpoint_cut` and `checkpointers_at` — which must agree, the
        # same contract `checkpoint_demand_pos` is held to.
        self.checkpoint_end_pos = 0
        # The chained content hash of every block of this prompt, not just the
        # ones that hit. Empty unless the state backend reserves checkpoints
        # midstep (`StateTransfer.readable_midstep`), which is the only caller
        # that needs to name a position the forward has not reached yet — see
        # `BlockManager._extend_hash_chain` for why it cannot simply be the
        # `block_hashes` the admission scan built.
        self.block_hashes: list[int] = []
        # Slots taken for midstep checkpoints of the forward now in flight,
        # as `(slot, position, hash)` — one slot each, since a checkpoint never
        # carries speculation scratch. Filled by `BlockManager.plan_midstep`
        # before the batch is built, drained by `commit_midstep` after it, and
        # handed back by `cancel_midstep` if that forward never runs. Non-empty
        # only between those two points.
        self.midstep_reservations: list[tuple] = []
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
        # `new_block_table` is main's: an array("i") rather than a list,
        # because every forward marshals these into the int32 buffer.
        self.block_table = new_block_table()
        # Per-request state slots (filled by BlockManager.allocate()), indexing
        # the per-req cache tensors owned by ModelRunner (e.g. mamba_k_cache for
        # GDN). Empty = unallocated.
        #
        # `[0]` is the committed state, which every path reads and writes;
        # `[1:]` is one rollback slot per speculated token, which only the
        # spec-decode path touches. Held as a list rather than a base index
        # because the slots are allocated one at a time and need not be
        # adjacent — see `StateSlotPool`.
        self.state_slots: list[int] = []
        # Slot the NEXT forward reads its incoming state from, when that is not
        # the slot it writes (`state_slot`). Set by BlockManager on a state fork
        # — resuming from a checkpoint, or taking one — and cleared by the
        # scheduler once a batch has carried it, so it describes exactly one
        # forward. -1 = read and write the same slot, the case for every step in
        # between. Always a single slot: a checkpoint is one slot wide.
        self.state_fork_src = -1
        self.temperature = sampling_params.temperature
        self.top_k = sampling_params.top_k
        self.top_p = sampling_params.top_p
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.stop_strings = sampling_params.stop_strings
        # Same type as `token_ids`, because the stop check compares a slice of
        # that against these and an `array("i")` never equals a list.
        self.stop_token_sequences = [
            new_token_ids(s) for s in (stop_token_sequences or [])
        ]
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
        # One entry per completion token, so the same reason `token_ids` is an
        # array applies: a list would box a PyFloat per token and hand the
        # collector a slot to walk for each one. `json.dumps` is the only
        # consumer that needs a list, and it converts at its own boundary.
        self.logprobs: array.array = array.array("d")
        # stream callback
        self.stream_callback = stream_callback
        # The completion half of `token_ids`, kept in step with it by every
        # writer, so it is the same array type for the same reasons.
        self.output_tokens = new_token_ids()
        # Placeholders from previous postprocess; overwritten in place.
        self.num_placeholder_tokens: int = 0

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
        # Optional client session lineage used by CoreManager's cache-aware,
        # token-load-aware DPA router.  These are routing metadata only; they
        # are deliberately kept out of the model-facing request payload.
        self.dp_session_id = dp_session_id
        self.dp_parent_session_id = dp_parent_session_id

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
    def state_slot(self) -> int:
        """The committed state slot, or -1 if this seq holds none.

        What every non-speculative path means by "the" slot: the one the
        forward reads and writes, the one a fork gives away, the one a
        checkpoint is. The rollback slots are `state_slots[1:]` and only the
        spec-decode path has any use for them.
        """
        return self.state_slots[0] if self.state_slots else -1

    @state_slot.setter
    def state_slot(self, slot: int) -> None:
        """Re-point the committed slot, keeping the rollback set.

        A fork moves where the request writes without disturbing its scratch:
        the speculation slots persist across forwards (step N's accepted slot
        is step N+1's initial state), so they belong to the request rather than
        to whichever slot it currently commits into.
        """
        if self.state_slots:
            self.state_slots[0] = slot
        else:
            self.state_slots = [slot]

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

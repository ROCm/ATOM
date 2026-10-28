# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Wire types, message enum and topology mapping for the prefill↔decode
disaggregation channel.

All messages are pickle-serialized as (DisaggMsgType, payload) tuples and
sent over dedicated ZMQ PUSH/PULL sockets between PrefillEngineCore and
DecodeEngineCore.

Deliberately depends on the standard library only: it is imported from both
engine processes and from ModelRunner, and it is the one place the
prefill↔decode rank pairing is written down.
"""

import enum
from dataclasses import dataclass, field


def disagg_pair_rank(decode_rank: int, worker_rank: int, stage_span: int) -> int:
    """The prefill TP rank a decode worker shares its GPU with.

    Prefill runs one worker per GPU with TP rank == GPU index, and every IPC
    handle it exports is only openable on the GPU that produced it — so a decode
    worker has to name the prefill rank on its OWN GPU, which is a GPU index and
    not a rank index. The two rapidserve topologies spread decode over GPUs
    differently and this is the sum of both terms:

    - symmetric: ONE decode process at TP=N, so `worker_rank` (0..N-1) is the
      GPU and `decode_rank` is 0 for every worker;
    - paired: N processes per side at TP=1, so `worker_rank` is always 0 and
      the process's DP rank is the GPU.

    Dropping either term collapses one topology onto GPU 0. `stage_span` is
    tp*pcp, matching ModelRunner._setup_device_and_distributed.
    """
    return decode_rank * stage_span + worker_rank


class DisaggMsgType(enum.Enum):
    """Message types for the direct prefill↔decode ZMQ channel.

    Byte values are chosen to not overlap with EngineCoreRequestType
    (which uses 0x00–0x07).
    """

    PREFILL_BATCH = b"\xa0"  # decode → prefill: one batch to execute, as planned
    PREFILL_BATCH_DONE = b"\xa1"  # prefill → decode: that batch's forward retired
    ABORT = b"\xa2"  # decode → prefill: cancel a pending sequence


@dataclass
class PrefillChunk:
    """One sequence's share of a prefill batch decode has planned.

    Decode owns the BlockManager, so it is the only half that can allocate,
    find a prefix-cache hit, walk the checkpoint ladder or measure KV pressure.
    It therefore decides EVERYTHING about this chunk -- which sequence, which
    span, which rung -- and prefill only executes it. That is the inversion
    this type encodes: it used to ship an *assignment* (blocks, and a
    speculative ladder of rungs prefill would choose from) because prefill did
    its own scheduling; it now ships a *decision*.

    Carries no token ids. The prefill process already holds the whole
    `Sequence`, prompt included, from the ADD that `DisaggCoreManager`
    fans out to both members of a pair -- so repeating the prompt here would
    re-send, per chunk, bytes the other side already has.
    """

    seq_id: int
    # ABSOLUTE offset this chunk begins at: the `num_cached_tokens` prefill
    # must adopt before the forward. Absolute rather than a delta because
    # decode is the only owner of the offset now -- a delta would make
    # prefill's local copy authoritative again, which is the thing being
    # removed.
    start: int
    num_tokens: int  # chunk width; `start + num_tokens` is where it ends
    # Whether this chunk finishes the prompt. Drives `is_partial_prefill` on
    # the prefill side and `ScheduledBatch.is_final_chunk`, which is what makes
    # `produces_output()` answer correctly for a middle chunk.
    is_final_chunk: bool
    # Physical block IDs owned by decode. Sent on a sequence's FIRST chunk only
    # and None afterwards, meaning "unchanged": decode allocates the whole
    # prompt at admission, so the table is stable for the rest of its life, and
    # re-sending it per chunk is most of a long prompt's message cost.
    block_table: list | None = None
    # Per-request STATE slot indices, in `Sequence.state_slots` order (index 0
    # is the committed slot; the rest are speculative rollback slots). Decode
    # allocates them; prefill writes this request's compressor/recurrent state
    # into them. An index means the same bytes in both processes only because
    # they share one per-request pool over CUDA IPC (`shares_per_req_cache`).
    # Empty for a model with no per-request state.
    state_slots: list = field(default_factory=list)
    # The ONE `CheckpointStoreOp` whose image this chunk's forward produces, or
    # None. A checkpoint holds the state as of a forward's LAST token, so an op
    # is only real if the forward ends exactly at its rung -- which is why
    # decode shortens the chunk to that rung before filling this in
    # (`_finalize_prefill_chunk` -> `checkpoint_cut`).
    #
    # Singular, where the old `BlockAssignment.prefill_cuts` was a whole ladder:
    # that list existed so a BlockManager-less prefill could pick a rung for
    # itself. Decode picks it now, so it ships the one it chose.
    store_op: object | None = None
    # The token at `start + num_tokens`, for drafters that need the next token
    # to propose from; -1 when there is none or the drafter does not want one.
    # Decode holds the token ids, so it reads this rather than prefill.
    next_token_id: int = -1


@dataclass
class PrefillBatchMsg:
    """Sent from DecodeEngineCore to PrefillEngineCore: run exactly this.

    The unit of work is a BATCH, not a request. Decode schedules prefill the
    same way it schedules itself -- pick the sequences, size the chunks against
    one token budget -- and a batch is what that decision produces.
    """

    batch_id: int
    # Which decode rank planned this. Under DP decode each rank has its own
    # BlockManager and KV pool, so the block IDs inside are meaningful only on
    # that rank, and its paired prefill rank is the only one that may run them.
    target_rank: int
    # Fraction of CUs the prefill forward may use, or None for no mask. Decode
    # computes it because decode is now the only half that knows both operands
    # (its own batch's tokens and the prefill tokens it just planned) -- which
    # is what retired the shared-memory channel that used to carry its decode
    # token count across to prefill.
    cu_stream_fraction: float | None
    chunks: list  # list[PrefillChunk]


@dataclass
class PrefillChunkResult:
    """What one chunk of a batch produced."""

    seq_id: int
    # Absolute position prefill computed to. Decode already knows this -- it
    # planned the span -- so this is a CROSS-CHECK, not the source of truth:
    # decode asserts it against the chunk it dispatched. It was authoritative
    # while prefill owned the offset and reported only the chunks that landed
    # on a rung; now that every dispatched chunk is acknowledged, a mismatch
    # means the two sides have diverged and is worth failing on.
    prefill_pos: int
    sampled_token_id: int  # first generated token sampled from prefill logits
    # Speculative drafts for the token(s) AFTER sampled_token_id, proposed by
    # the drafter in the prefill process. Empty when spec decode is off, or for
    # drafters prefill does not run (see RapidServeModelRunner.prefill_forward),
    # in which case decode falls back to placeholder drafts.
    draft_token_ids: list = field(default_factory=list)
    # False while this prompt has chunks left. A partial chunk carries real KV
    # -- decode publishes its hashes -- but the request is not decodable yet,
    # so it must not be promoted to `running` and carries no sampled token.
    is_final: bool = True


@dataclass
class PrefillBatchDone:
    """Sent from PrefillEngineCore to DecodeEngineCore once a batch retires.

    ONE message per batch, not one per sequence, because that is the real
    granularity of the GPU guarantee: `prefill_forward` ends in a blocking
    `stream.synchronize()` and is called with `wait_out=True`, so every
    sequence in the batch has its KV committed at the same instant. Decode also
    needs the whole batch retired before it may dispatch the next one, so a
    per-sequence message would only ever be reassembled into this.
    """

    batch_id: int
    results: list  # list[PrefillChunkResult]

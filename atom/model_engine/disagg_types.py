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

    BLOCK_ASSIGNMENT = b"\xa0"  # decode → prefill: assign KV blocks for a new seq
    PREFILL_DONE = b"\xa1"  # prefill → decode: prefill forward pass complete
    ABORT = b"\xa2"  # decode → prefill: cancel a pending sequence


@dataclass
class BlockAssignment:
    """Sent from DecodeEngineCore to PrefillEngineCore when a new request arrives.

    Decode allocates KV blocks via its BlockManager and notifies prefill so
    prefill can write the prompt's K/V values into the correct physical blocks.
    """

    seq_id: int
    block_table: list  # list[int] — physical block IDs owned by decode
    num_cached_tokens: int  # prefix-cache hits; prefill skips these blocks
    context_len: int  # total token count (prompt length)
    # Which decode rank owns this sequence's KV. Under DP decode each rank has
    # its own BlockManager and KV pool, so the block IDs above are meaningful
    # only on THIS rank — prefill must route the matching PrefillDone back to it
    # and must write the KV from the prefill TP rank sharing its GPU. Always 0
    # for symmetric rapidserve (single decode rank).
    target_rank: int = 0
    # Paged-SWA (DeepSeek-V4) parallel block table, positionally aligned with
    # block_table. The PREFILL forward is what writes the sliding-window KV, but
    # only decode owns the SlidingWindowPool — so decode materializes the
    # trailing window and ships the resulting slots here. Empty list for models
    # without paged SWA.
    swa_block_table: list = field(default_factory=list)


@dataclass
class PrefillDone:
    """Sent from PrefillEngineCore to DecodeEngineCore after the forward pass.

    Decode uses this signal to move the sequence from its prefill-pending
    holding area into the active decode scheduler queue.
    """

    seq_id: int
    num_tokens_computed: int  # tokens written into the KV cache
    sampled_token_id: int  # first generated token sampled from prefill logits
    # Speculative drafts for the token(s) AFTER sampled_token_id, proposed by
    # the drafter in the prefill process. Empty when spec decode is off, or for
    # drafters prefill does not run (see RapidServeModelRunner.prefill_forward),
    # in which case decode falls back to placeholder drafts.
    draft_token_ids: list = field(default_factory=list)

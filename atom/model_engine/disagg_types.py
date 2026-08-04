# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Wire types and message enum for the prefill↔decode disaggregation channel.

All messages are pickle-serialized as (DisaggMsgType, payload) tuples and
sent over dedicated ZMQ PUSH/PULL sockets between PrefillEngineCore and
DecodeEngineCore.
"""

import enum
from dataclasses import dataclass


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


@dataclass
class PrefillDone:
    """Sent from PrefillEngineCore to DecodeEngineCore after the forward pass.

    Decode uses this signal to move the sequence from its prefill-pending
    holding area into the active decode scheduler queue.
    """

    seq_id: int
    num_tokens_computed: int  # tokens written into the KV cache
    sampled_token_id: int  # first generated token sampled from prefill logits

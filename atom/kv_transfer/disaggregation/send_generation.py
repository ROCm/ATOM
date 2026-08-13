# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Signed-64 wire encoding for exact producer send generations.

Legacy MoRIIO transfer IDs are nonnegative request IDs. Generation-aware IDs
use the negative half of the signed-64 domain and pack a bounded request ID and
generation without quadratic growth. Keeping this helper outside the MoRIIO
package also lets protocol tests import it without loading optional RDMA/msgpack
dependencies from ``moriio.__init__``.
"""

from __future__ import annotations

from atom.kv_transfer.disaggregation.types import SendOperationId

_GENERATION_BITS = 32
_REQUEST_BITS = 31
_GENERATION_MASK = (1 << _GENERATION_BITS) - 1
_MAX_REQUEST_ID = (1 << _REQUEST_BITS) - 1
_MAX_GENERATION = _GENERATION_MASK
_MIN_SIGNED_64 = -(1 << 63)
_MAX_SIGNED_64 = (1 << 63) - 1


def encode_send_operation(operation: SendOperationId) -> int:
    """Pack one exact send identity into a negative signed-64 integer.

    Request IDs are limited to 31 bits and the scheduler-lifetime generation to
    32 bits. A process would need billions of sends to exhaust either bound;
    rejecting overflow is safer than emitting an integer that JSON/msgpack
    transports cannot represent consistently.
    """

    if not isinstance(operation, SendOperationId):
        raise TypeError("operation must be a SendOperationId")
    req_id = operation.req_id
    generation = operation.generation
    if isinstance(req_id, bool) or not isinstance(req_id, int):
        # Preserve the wire codec's historical validation contract.
        raise ValueError(  # noqa: TRY004
            "generation-aware request IDs must be integers"
        )
    if req_id < 0 or req_id > _MAX_REQUEST_ID:
        raise ValueError(
            "generation-aware request ID exceeds the 31-bit wire range: " f"{req_id}"
        )
    if isinstance(generation, bool) or not isinstance(generation, int):
        # Preserve the wire codec's historical validation contract.
        raise ValueError("send operation generation must be an integer")  # noqa: TRY004
    if generation < 0 or generation > _MAX_GENERATION:
        raise ValueError(
            "send operation generation exceeds the 32-bit wire range: " f"{generation}"
        )

    payload = (req_id << _GENERATION_BITS) | generation
    return -payload - 1


def decode_send_completion(transfer_id: int) -> int | SendOperationId:
    """Decode a signed-64 wire ID, preserving legacy nonnegative IDs."""

    if isinstance(transfer_id, bool) or not isinstance(transfer_id, int):
        # Preserve the wire codec's historical validation contract.
        raise ValueError("send completion wire ID must be an integer")  # noqa: TRY004
    if transfer_id < _MIN_SIGNED_64 or transfer_id > _MAX_SIGNED_64:
        raise ValueError("send completion wire ID must fit signed 64-bit")
    if transfer_id >= 0:
        return transfer_id

    payload = -transfer_id - 1
    req_id = payload >> _GENERATION_BITS
    generation = payload & _GENERATION_MASK
    return SendOperationId(req_id=req_id, generation=generation)


__all__ = ["decode_send_completion", "encode_send_operation"]

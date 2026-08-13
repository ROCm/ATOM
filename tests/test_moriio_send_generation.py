# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest

from atom.kv_transfer.disaggregation.send_generation import (
    decode_send_completion,
    encode_send_operation,
)
from atom.kv_transfer.disaggregation.types import SendOperationId


@pytest.mark.parametrize(
    "operation",
    [
        SendOperationId(0, 0),
        SendOperationId(7, 3),
        SendOperationId(2**31 - 1, 2**32 - 1),
    ],
)
def test_moriio_send_generation_wire_round_trip(operation):
    transfer_id = encode_send_operation(operation)

    assert transfer_id < 0
    assert decode_send_completion(transfer_id) == operation


def test_moriio_legacy_nonnegative_completion_is_preserved():
    assert decode_send_completion(17) == 17


@pytest.mark.parametrize("req_id", [-1, 2**31, "request", True])
def test_moriio_generation_wire_requires_bounded_integer_request(req_id):
    with pytest.raises(ValueError, match="request ID"):
        encode_send_operation(SendOperationId(req_id, 0))


@pytest.mark.parametrize("generation", [2**32, True])
def test_moriio_generation_wire_rejects_generation_overflow(generation):
    operation = SendOperationId.__new__(SendOperationId)
    object.__setattr__(operation, "req_id", 0)
    object.__setattr__(operation, "generation", generation)

    with pytest.raises(ValueError, match="generation"):
        encode_send_operation(operation)


@pytest.mark.parametrize("transfer_id", [-(2**63) - 1, 2**63])
def test_moriio_generation_wire_rejects_values_outside_signed_64(transfer_id):
    with pytest.raises(ValueError, match="signed 64-bit"):
        decode_send_completion(transfer_id)

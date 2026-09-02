# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for protobuf prefill/decode runtime payloads."""


def test_block_assignment_protobuf_roundtrip():
    from atom.model_engine.ipc_utils import DisaggIpcCodec
    from atom.proto.engine import disagg_proto as disagg_pb2

    original = disagg_pb2.BlockAssignment(
        seq_id=42,
        block_table=[0, 1, 5, 9],
        num_cached_tokens=16,
        context_len=64,
    )
    msg_type, restored = DisaggIpcCodec.decode_assignment_or_abort(
        DisaggIpcCodec.encode_block_assignment(original)
    )

    assert msg_type == "block_assignment"
    assert restored.seq_id == 42
    assert restored.block_table == [0, 1, 5, 9]
    assert restored.num_cached_tokens == 16
    assert restored.context_len == 64


def test_prefill_done_protobuf_roundtrip():
    from atom.model_engine.ipc_utils import DisaggIpcCodec
    from atom.proto.engine import disagg_proto as disagg_pb2

    original = disagg_pb2.PrefillDone(
        seq_id=7, num_tokens_computed=128, sampled_token_id=16
    )
    restored = DisaggIpcCodec.decode_prefill_done(
        DisaggIpcCodec.encode_prefill_done(original)
    )

    assert restored.seq_id == 7
    assert restored.num_tokens_computed == 128
    assert restored.sampled_token_id == 16


def test_abort_protobuf_roundtrip():
    from atom.model_engine.ipc_utils import DisaggIpcCodec

    seq_id = 99
    msg_type, restored = DisaggIpcCodec.decode_assignment_or_abort(
        DisaggIpcCodec.encode_abort(seq_id)
    )

    assert msg_type == "abort"
    assert restored == seq_id


def test_bootstrap_protobuf_roundtrip():
    from atom.model_engine.ipc_utils import DisaggIpcCodec

    paths = ["/tmp/rank-0.pkl", "/tmp/rank-1.pkl"]
    assert DisaggIpcCodec.decode_weight_handles(
        DisaggIpcCodec.encode_weight_handles(paths)
    ) == paths
    assert DisaggIpcCodec.decode_kv_cache_handles(
        DisaggIpcCodec.encode_kv_cache_handles(paths, 128)
    ) == (paths, 128)
    DisaggIpcCodec.decode_acknowledgement(
        DisaggIpcCodec.encode_acknowledgement()
    )

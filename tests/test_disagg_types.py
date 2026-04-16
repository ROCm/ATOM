# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for disagg_types.py — no GPU required.

Note: test_mxfp4_moe_has_bias.py purges all atom.* entries from sys.modules at
collection time (lines 23-25 of that file).  To avoid stale class identity
issues when pickle resolves the module, we import inside each test function so
the import always reflects the current sys.modules state.
"""

import pickle

# Known byte values from EngineCoreRequestType (engine_core.py lines 32-43).
# Hardcoded here to avoid importing engine_core (pulls in atom.config.ParallelConfig).
_ENGINE_CORE_BYTES = {
    b"\x00",
    b"\x01",
    b"\x02",
    b"\x03",
    b"\x04",
    b"\x05",
    b"\x06",
    b"\x07",
}


def test_disagg_msg_type_no_overlap_with_engine_core_request_type():
    """DisaggMsgType byte values must not collide with EngineCoreRequestType."""
    from atom.model_engine.disagg_types import DisaggMsgType

    disagg_bytes = {t.value for t in DisaggMsgType}
    assert _ENGINE_CORE_BYTES.isdisjoint(disagg_bytes), (
        f"Byte value collision between EngineCoreRequestType and DisaggMsgType: "
        f"{_ENGINE_CORE_BYTES & disagg_bytes}"
    )


def test_block_assignment_pickle_roundtrip():
    from atom.model_engine.disagg_types import BlockAssignment, DisaggMsgType

    original = BlockAssignment(
        seq_id=42,
        block_table=[0, 1, 5, 9],
        num_cached_tokens=16,
        context_len=64,
    )
    msg = pickle.dumps((DisaggMsgType.BLOCK_ASSIGNMENT, original))
    msg_type, restored = pickle.loads(msg)

    assert msg_type == DisaggMsgType.BLOCK_ASSIGNMENT
    assert restored.seq_id == 42
    assert restored.block_table == [0, 1, 5, 9]
    assert restored.num_cached_tokens == 16
    assert restored.context_len == 64


def test_prefill_done_pickle_roundtrip():
    from atom.model_engine.disagg_types import DisaggMsgType, PrefillDone

    original = PrefillDone(seq_id=7, num_tokens_computed=128)
    msg = pickle.dumps((DisaggMsgType.PREFILL_DONE, original))
    msg_type, restored = pickle.loads(msg)

    assert msg_type == DisaggMsgType.PREFILL_DONE
    assert restored.seq_id == 7
    assert restored.num_tokens_computed == 128


def test_abort_msg_type_pickle_roundtrip():
    from atom.model_engine.disagg_types import DisaggMsgType

    seq_id = 99
    msg = pickle.dumps((DisaggMsgType.ABORT, seq_id))
    msg_type, restored = pickle.loads(msg)

    assert msg_type == DisaggMsgType.ABORT
    assert restored == seq_id


def test_all_disagg_msg_types_have_unique_bytes():
    from atom.model_engine.disagg_types import DisaggMsgType

    values = [t.value for t in DisaggMsgType]
    assert len(values) == len(set(values)), "DisaggMsgType has duplicate byte values"

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


def test_prefill_batch_pickle_roundtrip():
    """The batch message is what decode plans and prefill executes."""
    from atom.model_engine.disagg_types import (
        DisaggMsgType,
        PrefillBatchMsg,
        PrefillChunk,
    )

    original = PrefillBatchMsg(
        batch_id=3,
        target_rank=2,
        cu_stream_fraction=0.5,
        chunks=[
            PrefillChunk(
                seq_id=42,
                start=16,
                num_tokens=48,
                is_final_chunk=False,
                block_table=[0, 1, 5, 9],
                state_slots=[7],
                next_token_id=1234,
            ),
            PrefillChunk(seq_id=43, start=0, num_tokens=8, is_final_chunk=True),
        ],
    )
    msg = pickle.dumps((DisaggMsgType.PREFILL_BATCH, original))
    msg_type, restored = pickle.loads(msg)

    assert msg_type == DisaggMsgType.PREFILL_BATCH
    assert restored.batch_id == 3
    assert restored.target_rank == 2
    assert restored.cu_stream_fraction == 0.5
    assert [c.seq_id for c in restored.chunks] == [42, 43]
    first = restored.chunks[0]
    assert (first.start, first.num_tokens) == (16, 48)
    assert first.block_table == [0, 1, 5, 9]
    assert first.state_slots == [7]
    assert first.next_token_id == 1234
    assert first.is_final_chunk is False


def test_chunk_carries_no_token_ids():
    """A batch message never repeats the prompt.

    Prefill already holds the whole `Sequence` -- `DisaggCoreManager` fans the
    ADD out to both members of a pair -- so shipping token ids per chunk would
    re-send, every chunk, bytes the other side already has. Guarded because the
    temptation to "just include the tokens" is what the old per-request
    assignment shape invited.
    """
    from atom.model_engine.disagg_types import PrefillChunk

    fields = set(PrefillChunk.__dataclass_fields__)
    assert not fields & {"token_ids", "tokens", "prompt", "prompt_token_ids"}


def test_prefill_batch_done_is_batch_granular():
    """One reply per batch, carrying one result per chunk.

    Batch-granular because the GPU guarantee is: `prefill_forward` is
    synchronous and ends in `stream.synchronize()`, so every sequence in the
    batch retires together.
    """
    from atom.model_engine.disagg_types import (
        DisaggMsgType,
        PrefillBatchDone,
        PrefillChunkResult,
    )

    original = PrefillBatchDone(
        batch_id=3,
        results=[
            PrefillChunkResult(
                seq_id=7,
                prefill_pos=128,
                sampled_token_id=16,
                draft_token_ids=[1, 2],
            ),
            PrefillChunkResult(
                seq_id=8, prefill_pos=64, sampled_token_id=-1, is_final=False
            ),
        ],
    )
    msg = pickle.dumps((DisaggMsgType.PREFILL_BATCH_DONE, original))
    msg_type, restored = pickle.loads(msg)

    assert msg_type == DisaggMsgType.PREFILL_BATCH_DONE
    assert restored.batch_id == 3
    assert [r.seq_id for r in restored.results] == [7, 8]
    assert restored.results[0].prefill_pos == 128
    assert restored.results[0].draft_token_ids == [1, 2]
    assert restored.results[1].is_final is False


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

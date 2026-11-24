# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""A continued prompt must not displace older decodes when their KV grows."""

from types import SimpleNamespace

import numpy as np
import pytest
from conftest import MockConfig
from test_scheduler import _ConnectorNamespace

from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence, SequenceStatus, SequenceType


def _scheduler(mixed, block_size, pages, width, prefix):
    return Scheduler(
        MockConfig(
            enable_mixed_prefill_decode=mixed,
            enable_prefix_caching=prefix,
            kv_cache_block_size=block_size,
            num_kvcache_blocks=pages,
            max_num_seqs=4,
            max_num_batched_tokens=16 if block_size == 16 else 128,
            max_model_len=4096,
            speculative_config=(
                SimpleNamespace(
                    num_speculative_tokens=width - 1, use_dspark=lambda: True
                )
                if width > 1
                else None
            ),
        )
    )


def _older_decode(sched, block_size, width, remote):
    seq = Sequence(
        [11] * (block_size - 3 * width),
        block_size=block_size,
        kv_transfer_params={"first_token_id": 42} if remote else None,
    )
    assert sched.block_manager.allocate(seq, 0)
    if remote:
        sched.kv_connector = _ConnectorNamespace(is_producer=False, is_offload=False)
        seq.status = SequenceStatus.WAITING_FOR_REMOTE_KVS
        sched.add(seq)
        sched._count_inflight_load(seq)
        sched.finished_recving_kv_req_ids.append(str(seq.id))
        batch, _ = sched.schedule()
        assert batch.req_ids == [seq.id] and batch.total_seqs_num_decode == 1
    else:
        seq.num_cached_tokens = len(seq)
        for _ in range(width):
            seq.append_token(42)
        seq.type = SequenceType.DECODE
        seq.status = SequenceStatus.RUNNING
        seq.prefill_completed = True
        sched.running.append(seq)
    assert len(seq) == block_size - 2 * width
    assert len(seq.block_table) == 1
    return seq


def _finish_step(sched, batch, scheduled, width):
    # Use real postprocess for chunk completion. Model generation is simulated
    # by accepting the entire decode window; there is no GPU in this test.
    lengths = {seq.id: len(seq) for seq in scheduled.values()}
    for seq in scheduled.values():
        seq.num_placeholder_tokens = 0
    sched.postprocess(
        list(scheduled.values()),
        ScheduledBatchOutput(
            req_ids=[],
            token_ids=[],
            num_rejected=None,
            num_bonus=None,
            draft_token_ids=None,
        ),
        batch=batch,
    )
    for seq in scheduled.values():
        if seq.is_partial_prefill:
            continue
        seq.num_cached_tokens = lengths[seq.id]
        for _ in range(lengths[seq.id] + width - len(seq)):
            seq.append_token(42)
        seq.spec_token_ids = np.full(width - 1, 42, dtype=np.int32)


@pytest.mark.parametrize(
    "block_size,pages,prompt,width",
    [(16, 5, 64, 1), (256, 9, 2000, 1), (256, 9, 2000, 8)],
)
@pytest.mark.parametrize("prefix", [False, True])
@pytest.mark.parametrize("remote", [False, True])
def test_decode_growth_preempts_newer_prompt_like_mixed_off(
    block_size, pages, prompt, width, prefix, remote
):
    for mixed in (False, True):
        sched = _scheduler(mixed, block_size, pages, width, prefix)
        older = _older_decode(sched, block_size, width, remote)
        received_pages = list(older.block_table)
        newer = Sequence([22] * prompt, block_size=block_size)
        sched.add(newer)
        partial_ticks = 0
        for _ in range(64):
            result = sched.schedule()
            if sched.total_preemptions:
                break
            assert result is not None
            batch, scheduled = result
            _finish_step(sched, batch, scheduled, width)
            partial_ticks += newer.is_partial_prefill
        else:
            pytest.fail("decode did not reach the expected KV boundary")
        assert partial_ticks >= 2
        assert older.status == SequenceStatus.RUNNING
        assert list(older.block_table[:1]) == received_pages
        assert newer in sched.waiting and not newer.block_table
        assert sched.total_preemptions == 1
        assert sched._partial_prefill_count == 0
        assert result[0].req_ids == [older.id]


def test_pressured_partial_is_not_reported_as_prefill_work():
    sched = _scheduler(True, 16, 5, 1, False)
    older = _older_decode(sched, 16, 1, False)
    newer = Sequence([22] * 64, block_size=16)
    sched.add(newer)
    for _ in range(2):
        batch, scheduled = sched.schedule()
        assert batch.is_mixed
        _finish_step(sched, batch, scheduled, 1)
    assert older in sched.running and newer.is_partial_prefill
    assert sched.block_manager.kv.num_free == 0
    limits = sched._prefill_limits()
    assert limits.reserved_decode_blocks == 1
    assert limits.tokens == 0  # caller suppresses admission probes under pressure
    assert sched._probe_local_prefill_work(limits) == (False, 0)
    assert sched._estimate_prefill_chunk_cost(limits) == 0

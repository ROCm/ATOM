# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Same-tick P/D promotion must retain its received KV under mixed admission."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from conftest import MockConfig
from test_scheduler import _ConnectorNamespace

from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence, SequenceStatus, SequenceType


@pytest.mark.parametrize("prefix", [False, True])
@pytest.mark.parametrize("width", [1, 8])
@pytest.mark.parametrize("receives", [1, 2])
def test_promoted_remote_decodes_keep_their_append_pages(prefix, width, receives):
    sched = Scheduler(
        MockConfig(
            enable_mixed_prefill_decode=True,
            enable_prefix_caching=prefix,
            kv_cache_block_size=16,
            num_kvcache_blocks=2 * receives,
            max_num_seqs=4,
            max_num_batched_tokens=64,
            max_model_len=128,
            speculative_config=(
                SimpleNamespace(
                    num_speculative_tokens=width - 1, use_dspark=lambda: True
                )
                if width > 1
                else None
            ),
        )
    )
    sched.kv_connector = _ConnectorNamespace(is_producer=False, is_offload=False)
    # Deliberately put the local request first. The real ready-receive ordering
    # and promotion paths must protect all newly promoted rows before admission.
    prompt_len = 16 - sched.draft_lookahead
    local = Sequence([99] * (16 * receives - sched.draft_lookahead), block_size=16)
    sched.add(local)
    remote = []
    for i in range(receives):
        seq = Sequence(
            [10 + i] * prompt_len,
            block_size=16,
            kv_transfer_params={"first_token_id": 42},
        )
        sched.add(seq)
        assert sched.block_manager.allocate(seq, sched.block_manager.can_allocate(seq))
        assert len(seq.block_table) == 1
        seq.status = SequenceStatus.WAITING_FOR_REMOTE_KVS
        sched._count_inflight_load(seq)
        sched.finished_recving_kv_req_ids.append(str(seq.id))
        remote.append(seq)
    received_pages = {seq.id: list(seq.block_table) for seq in remote}

    batch, scheduled = sched.schedule()

    assert sched.total_preemptions == 0
    assert set(scheduled) == {seq.id for seq in remote}
    assert batch.total_seqs_num_decode == receives
    assert batch.total_seqs_num_prefill == 0
    assert local in sched.waiting and not local.block_table
    for seq in remote:
        assert seq.status == SequenceStatus.RUNNING
        assert list(seq.block_table[:1]) == received_pages[seq.id]
        assert len(seq.block_table) == 2
        assert seq.prefill_completed
        assert seq.token_ids[prompt_len] == 42


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("receiving", [False, True])
def test_pure_decode_skips_prefill_reservation(monkeypatch, mixed, receiving):
    sched = Scheduler(MockConfig(enable_mixed_prefill_decode=mixed))
    seq = Sequence([3] * 4, block_size=4)
    assert sched.block_manager.allocate(seq, 0)
    seq.num_cached_tokens = 4
    seq.append_token(4)
    seq.type = SequenceType.DECODE
    seq.status = SequenceStatus.RUNNING
    sched.running.append(seq)
    if receiving:
        sched.kv_connector = _ConnectorNamespace(is_producer=False, is_offload=False)
        remote = Sequence([9] * 4, block_size=4)
        sched.add(remote)
        assert sched.block_manager.allocate(remote, 0)
        remote.status = SequenceStatus.WAITING_FOR_REMOTE_KVS
        sched._count_inflight_load(remote)
    reserve = Mock(
        side_effect=AssertionError("pure decode has no prefill to reserve for")
    )
    monkeypatch.setattr(sched, "_prefill_limits", reserve)

    batch, _ = sched.schedule()

    assert batch.total_seqs_num_decode == 1
    reserve.assert_not_called()

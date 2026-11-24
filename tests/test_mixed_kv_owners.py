# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""KV ownership must survive admission, transfer completion and preemption."""

from types import SimpleNamespace

import pytest
from conftest import MockConfig
from test_mixed_chunk_decode_pressure import _finish_step
from test_scheduler import _ConnectorNamespace, _OffloadMixinStub

from atom.model_engine.scheduler import PrefillLimits, Scheduler
from atom.model_engine.sequence import Sequence, SequenceStatus, SequenceType


class OffloadConnector(_OffloadMixinStub):
    _do_load = False


def scheduler(mixed=True, *, pages=20, width=1, prefix=False, offload=False):
    sched = Scheduler(
        MockConfig(
            enable_mixed_prefill_decode=mixed,
            enable_prefix_caching=prefix,
            kv_cache_block_size=16,
            num_kvcache_blocks=pages,
            max_num_seqs=8,
            max_num_batched_tokens=32,
            max_model_len=256,
            speculative_config=(
                SimpleNamespace(
                    num_speculative_tokens=width - 1, use_dspark=lambda: True
                )
                if width > 1
                else None
            ),
        )
    )
    sched.kv_connector = (
        OffloadConnector()
        if offload
        else _ConnectorNamespace(is_producer=False, is_offload=False)
    )
    return sched


def decode(sched, *, length=16, width=1):
    seq = Sequence([11] * (length - width), block_size=16)
    assert sched.block_manager.allocate(seq, 0)
    seq.num_cached_tokens = len(seq)
    for _ in range(width):
        seq.append_token(42)
    seq.type = SequenceType.DECODE
    seq.status = SequenceStatus.RUNNING
    seq.prefill_completed = True
    sched.running.append(seq)
    return seq


def receive(sched, *, ready=True, token=22):
    seq = Sequence(
        [token] * (16 - sched.draft_lookahead),
        block_size=16,
        kv_transfer_params={"first_token_id": 42},
    )
    sched.add(seq)
    assert sched.block_manager.allocate(seq, sched.block_manager.can_allocate(seq))
    seq.status = SequenceStatus.WAITING_FOR_REMOTE_KVS
    sched._count_inflight_load(seq)
    if ready:
        sched.finished_recving_kv_req_ids.append(str(seq.id))
    return seq


def resume(sched, *, length=64, cached=32, failed=False):
    seq = Sequence([33] * length, block_size=16)
    sched.add(seq)
    assert sched.block_manager.allocate(seq, 0)
    sched.block_manager.may_append(seq, 0, lookahead=sched.draft_lookahead)
    seq.num_cached_tokens = cached
    seq.offload_loaded = not failed
    seq.offload_load_failed = failed
    seq.offload_loaded_tokens = cached
    sched._count_inflight_load(seq)
    sched._early_load.note_submitted(seq)
    return seq


def partial(sched):
    seq = Sequence([55] * 32, block_size=16)
    assert sched.block_manager.allocate(seq, 0)
    sched.block_manager.may_append(seq, 0, lookahead=sched.draft_lookahead)
    seq.num_cached_tokens = 16
    seq.is_partial_prefill = True
    seq.status = SequenceStatus.RUNNING
    sched.running.append(seq)
    sched._partial_prefill_count += 1
    return seq


@pytest.mark.parametrize("width", [1, 8])
@pytest.mark.parametrize("prefix", [False, True])
@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("waiter_newer", [False, True])
def test_pressure_compares_waiting_and_running_owner_age(
    width, prefix, offload, waiter_newer
):
    extra = int(width > 1)
    pages = 4 + extra + (1 + extra if offload else 0)
    sched = scheduler(pages=pages, width=width, prefix=prefix, offload=offload)
    older = decode(sched, width=width)
    waiting = resume(sched, length=32, cached=16) if offload else receive(sched)
    prompt = partial(sched)
    # Arrival order, not request-ID order or the current queue, defines age.
    older.arrive_time = 1.0
    waiting.arrive_time = 3.0 if waiter_newer else 2.0
    prompt.arrive_time = 2.0 if waiter_newer else 3.0
    sched.max_num_seqs = 2
    assert sched.block_manager.kv.num_free == 0
    original = {s.id: list(s.block_table) for s in (older, waiting, prompt)}

    batch, _ = sched.schedule()

    victim, retained = (waiting, prompt) if waiter_newer else (prompt, waiting)
    assert older.id in batch.req_ids
    assert not victim.block_table
    assert list(retained.block_table) == original[retained.id]
    assert list(older.block_table[:1]) == original[older.id]
    assert sched.total_preemptions == 1


def test_older_waiting_owner_survives_younger_current_decode():
    sched = scheduler(pages=3, offload=True)
    waiting = resume(sched, length=32, cached=16)
    current = decode(sched)
    waiting.arrive_time, current.arrive_time = 1.0, 2.0
    sched.max_num_seqs = 1
    original = list(waiting.block_table)

    sched.schedule()

    assert list(waiting.block_table) == original
    assert not current.block_table
    assert sched._num_parked_remote_kv == 1


def test_waiting_reclaim_uses_age_after_queue_prioritization():
    sched = scheduler(pages=5, offload=True)
    older = decode(sched)
    first = resume(sched, length=32, cached=16)
    second = resume(sched, length=32, cached=16)
    older.arrive_time, first.arrive_time, second.arrive_time = 1.0, 2.0, 3.0
    # Completion prioritization can move the newest owner ahead of an older one.
    sched.waiting.rotate(1)
    original = list(first.block_table)

    sched.schedule()

    assert list(first.block_table) == original
    assert not second.block_table


def test_zero_budget_ticks_preserve_older_partial_until_decode_growth():
    outcomes = []
    for mixed in (False, True):
        sched = scheduler(mixed, pages=4)
        prompt = partial(sched)
        first = decode(sched, length=8)
        second = decode(sched, length=17)
        prompt.arrive_time, first.arrive_time, second.arrive_time = 1.0, 2.0, 3.0
        original = list(prompt.block_table)
        for tick in range(12):
            batch, scheduled = sched.schedule()
            if mixed and tick == 0:
                assert batch.total_seqs_num_prefill == 0
                assert list(sched.running) == [prompt, first]
            assert list(prompt.block_table[: len(original)]) == original
            _finish_step(sched, batch, scheduled, 1)
        outcomes.append([seq.id == prompt.id for seq in sched.running])
    assert outcomes[0] == outcomes[1] == [True]


@pytest.mark.parametrize("width", [1, 8])
@pytest.mark.parametrize("parked", ["settled", "aborted"])
def test_parked_count_does_not_hide_admissible_prefill(width, parked):
    sched = scheduler(width=width, offload=parked == "settled")
    older = decode(sched, width=width)
    if parked == "settled":
        prompt = resume(sched)
    else:
        aborted = receive(sched, ready=False)
        held = list(aborted.block_table)
        assert sched.abort_request(aborted.id)
        assert aborted not in sched.waiting
        prompt = Sequence([33] * 64, block_size=16)
        sched.add(prompt)
    assert len(sched.waiting) == sched._num_parked_remote_kv == 1

    batch, _ = sched.schedule()

    assert batch.is_mixed
    assert batch.req_ids == [prompt.id, older.id]
    assert batch.total_tokens_num_decode == width
    if parked == "aborted":
        assert list(aborted.block_table) == held
        assert sched._num_parked_remote_kv == 1
        assert sched._finish_aborted_load_cleanup(aborted.id)
        assert sched._num_parked_remote_kv == 0


@pytest.mark.parametrize("prefix", [False, True])
@pytest.mark.parametrize("width", [1, 8])
@pytest.mark.parametrize("receives", [1, 2])
def test_ready_receive_at_zero_pages_matches_flag_off(prefix, width, receives):
    victims = []
    for mixed in (False, True):
        sched = scheduler(mixed, pages=1 + receives, width=width, prefix=prefix)
        older = decode(sched, width=width)
        incoming = [receive(sched, token=22 + i) for i in range(receives)]
        original = list(older.block_table)
        assert sched.block_manager.kv.num_free == 0

        batch, _ = sched.schedule()

        assert older.id in batch.req_ids
        assert list(older.block_table[: len(original)]) == original
        assert sched._num_parked_remote_kv == 0
        assert not sched.finished_recving_kv_req_ids
        victims.append([i for i, seq in enumerate(incoming) if not seq.block_table])
        assert len({s.id for s in sched.waiting}) == len(sched.waiting)
    assert victims[0] and victims[0] == victims[1]


@pytest.mark.parametrize("prefix", [False, True])
@pytest.mark.parametrize("failed", [False, True])
def test_settled_resume_is_victim_before_older_decode(prefix, failed):
    for mixed in (False, True):
        sched = scheduler(mixed, pages=5, prefix=prefix, offload=True)
        older = decode(sched)
        restored = resume(sched, failed=failed)
        # Also exercise the non-empty admissible tail, independently of the
        # parked-count shortcut addressed by the first regression above.
        sched.add(Sequence([44] * 16, block_size=16))
        original = list(older.block_table)
        for _ in range(8):
            batch, scheduled = sched.schedule()
            if sched.total_preemptions:
                break
            _finish_step(sched, batch, scheduled, 1)
        else:
            pytest.fail("decode never encountered the full pool")
        assert list(older.block_table[: len(original)]) == original
        assert older.id in batch.req_ids
        assert restored in sched.waiting and not restored.block_table
        assert sched.total_preemptions == 1
        assert sched._num_parked_remote_kv == 0
        if mixed:
            assert not sched._is_offload_prefill_resume(restored)
            assert restored.id not in sched._early_load.entries


def test_receive_reservation_precedes_partial_chunk_and_dynamic_accounting():
    sched = scheduler(pages=4)
    older = decode(sched, length=8)
    partial = Sequence([33] * 32, block_size=16)
    assert sched.block_manager.allocate(partial, 0)
    partial.num_cached_tokens = 16
    partial.is_partial_prefill = True
    partial.status = SequenceStatus.RUNNING
    sched.running.append(partial)
    sched._partial_prefill_count = 1
    incoming = receive(sched)
    sched.enable_dynamic_chunking = True
    sched._calibrating = True
    before = sched._dynamic_chunking_counts.copy()
    batch, _ = sched.schedule()

    assert sched._dynamic_chunking_counts == before
    assert batch.total_seqs_num_prefill == 0
    assert older.id in batch.req_ids
    assert incoming.token_ids.count(42) == 1
    assert sched._num_parked_remote_kv == 0


@pytest.mark.parametrize("width", [1, 8])
def test_ready_receive_can_release_pages_when_running_slots_are_full(width):
    sched = scheduler(pages=2, width=width)
    older = decode(sched, width=width)
    incoming = receive(sched)
    sched.max_num_seqs = 1
    batch, _ = sched.schedule()
    assert batch.req_ids == [older.id]
    assert incoming in sched.waiting and not incoming.block_table
    assert sched._num_parked_remote_kv == 0
    assert incoming.token_ids.count(42) == 1
    assert incoming.num_placeholder_tokens == 0
    assert incoming.spec_token_ids.size == 0


def test_prefill_delay_does_not_block_receive_completion(monkeypatch):
    sched = scheduler(pages=2)
    older = decode(sched)
    incoming = receive(sched)
    sched.delay_factor = 1
    monkeypatch.setattr(sched, "_passed_delay", lambda now: False)
    batch, _ = sched.schedule()
    assert batch.req_ids == [older.id]
    assert incoming in sched.waiting and not incoming.block_table
    assert sched._num_parked_remote_kv == 0


@pytest.mark.parametrize("width", [1, 8])
def test_promoted_decodes_reduce_prefill_token_budget(width):
    sched = scheduler(pages=100, width=width)
    older = decode(sched, width=width)
    incoming = receive(sched)
    prompt = Sequence([55] * 64, block_size=16)
    sched.add(prompt)
    batch, _ = sched.schedule()
    assert batch.req_ids == [prompt.id, older.id, incoming.id]
    assert batch.total_tokens_num_prefill == 32 - 2 * width
    assert batch.num_scheduled_tokens.tolist() == [32 - 2 * width, width, width]


@pytest.mark.parametrize("width", [1, 8])
def test_receive_promotion_respects_running_capacity(width):
    sched = scheduler(pages=100, width=width)
    older = decode(sched, width=width)
    first, second = receive(sched), receive(sched, token=23)
    sched.max_num_seqs = 2
    batch, _ = sched.schedule()
    assert batch.req_ids == [older.id, first.id]
    assert list(sched.running) == [older, first]
    assert list(sched.waiting) == [second]
    assert second.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert second.token_ids.count(42) == 0


@pytest.mark.parametrize("width,expected_budget", [(1, 1), (8, 4)])
def test_receive_reservation_stops_at_mixed_decode_seats(width, expected_budget):
    sched = scheduler(pages=10, width=width)
    sched.max_num_batched_tokens = 4 * width
    incoming = [receive(sched, token=30 + i) for i in range(7)]
    assert sched._mixed_decode_seats() == 3
    assert sched.block_manager.kv.num_free == 3

    limits = sched._promote_mixed_remote_decodes(PrefillLimits(4 * width, 0, 0))

    assert list(sched.running) == incoming
    assert limits.num_decodes == 7
    assert limits.reserved_decode_blocks == 3
    assert (
        sched._budget_for_decodes(limits.num_decodes, limits.reserved_decode_blocks)
        == expected_budget
    )


@pytest.mark.parametrize("protection", ["receiving", "connector", "pipeline"])
def test_waiting_owner_protection_survives_pressure(protection, monkeypatch):
    sched = scheduler(pages=3, offload=True)
    older = decode(sched)
    restored = resume(sched, length=32, cached=16)
    if protection == "receiving":
        # A second load may retain stale completion fields from the first.
        restored.status = SequenceStatus.WAITING_FOR_REMOTE_KVS
    elif protection == "connector":
        monkeypatch.setattr(
            sched.kv_connector, "should_defer_free", lambda s: s is restored
        )
    else:
        sched._pp_inflight_token_block.add(restored.id)
    sched.add(Sequence([44] * 16, block_size=16))
    original = list(restored.block_table)
    sched.schedule()
    assert list(restored.block_table) == original
    assert sched._num_parked_remote_kv == 1
    assert restored in sched.waiting
    assert not older.block_table


@pytest.mark.parametrize("failed", [False, True])
def test_waiting_victim_order_and_readmission(failed):
    sched = scheduler(pages=5, offload=True)
    older = decode(sched)
    first = resume(sched, length=32, cached=16)
    second = resume(sched, length=32, cached=16, failed=failed)
    sched.add(Sequence([44] * 16, block_size=16))
    first_pages = list(first.block_table)
    batch, _ = sched.schedule()
    assert batch.req_ids == [older.id]
    assert list(first.block_table) == first_pages
    assert second in sched.waiting and not second.block_table
    assert sched._num_parked_remote_kv == 1
    assert not sched._is_offload_prefill_resume(second)
    assert second.id not in sched._early_load.entries
    # Retire the decode's allocation; both resumes must admit normally again.
    sched.running.remove(older)
    sched.block_manager.deallocate(older)
    for _ in range(8):
        batch, scheduled = sched.schedule()
        if second.id in scheduled:
            assert not second.offload_loaded and not second.offload_load_failed
            assert second.offload_loaded_tokens == 0
            assert batch.num_cached_tokens[batch.req_ids.index(second.id)] == 0
            break
        _finish_step(sched, batch, scheduled, 1)
    else:
        pytest.fail("reclaimed resume never re-entered prefill")
    assert sched._num_parked_remote_kv == 0
    ids = [s.id for s in sched.running] + [s.id for s in sched.waiting]
    assert len(ids) == len(set(ids))

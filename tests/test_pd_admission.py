# SPDX-License-Identifier: MIT
"""Admission bounds must protect KV ownership without stopping decode progress."""

from types import SimpleNamespace

import pytest
from aiter_stub import stubbed_aiter
from conftest import MockConfig

from atom.kv_transfer.disaggregation.types import KVConnectorOutput
from atom.model_engine.pd_admission import PdAdmissionController
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import SequenceStatus


@pytest.fixture
def mooncake_module(monkeypatch):
    with stubbed_aiter():
        from atom.kv_transfer.disaggregation.mooncake import mooncake_connector
    monkeypatch.setattr(mooncake_connector, "get_open_port", lambda: 30000)
    monkeypatch.setattr(mooncake_connector, "get_ip", lambda: "127.0.0.1")
    return mooncake_connector


@pytest.fixture
def build(monkeypatch, mooncake_module, seq_factory):
    def make(*, options=None, prefix=False, connector_wrapper=None, **overrides):
        config = MockConfig(
            num_kvcache_blocks=64,
            pd_admission_block_bytes=1024,
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
            parallel_config=SimpleNamespace(
                data_parallel_size=1,
                data_parallel_rank=0,
            ),
            enable_prefix_caching=prefix,
            kv_transfer_config={
                "kv_connector": "mooncake",
                "kv_role": "kv_consumer",
                "pd_admission": (
                    options
                    if options is not None
                    else {
                        "enabled": True,
                        "max_pending_requests": 1,
                        "max_pending_kv_fraction": None,
                    }
                ),
            },
        )
        for k, v in overrides.items():
            setattr(config, k, v)
        connector = mooncake_module.MooncakeConnectorScheduler(config)
        if connector_wrapper:
            connector = connector_wrapper(connector)
        monkeypatch.setattr(
            "atom.utils.forward_context.get_kvconnector", lambda *args: connector
        )
        scheduler = Scheduler(config)

        def request(tokens=8):
            seq = seq_factory(list(range(tokens)))
            seq.kv_transfer_params = {
                "do_remote_prefill": True,
                "transfer_id": f"p-{seq.id}",
                "remote_block_ids": list(range((tokens + 3) // 4)),
                "block_size": 4,
                "dcp_size": 1,
                "remote_host": "127.0.0.1",
                "remote_port": 30000,
                "tp_size": 1,
                "remote_pp_size": 4,
                "first_token_id": 42,
            }
            return seq

        return scheduler, request

    return make


def test_count_limit_holds_ready_credit_until_first_batch(build):
    sched, request = build()
    a, b = request(), request()
    sched.extend([a, b])
    batch, _ = sched.schedule()
    assert set(batch.connector_meta_output.reqs_to_recv) == {a.id}
    assert a.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert not b.block_table
    assert len(sched.pd_admission.pending) == 1
    assert sched.pd_admission.pending_bytes == 2048

    sched._update_from_kv_xfer_finished(KVConnectorOutput(finished_recving={str(a.id)}))
    assert len(sched.pd_admission.pending) == 1
    assert sched.pd_admission.pending[str(a.id)].ready_at is not None
    batch, _ = sched.schedule()
    assert a.id in batch.req_ids
    assert not b.block_table  # A held its credit during this admission pass.
    assert not sched.pd_admission.pending
    assert a.block_table  # Credit release never frees a running request's KV.
    batch, _ = sched.schedule()
    assert set(batch.connector_meta_output.reqs_to_recv) == {b.id}
    assert a.id in batch.req_ids  # Decode progresses while B receives.


def test_bytes_limit_and_isolated_large_request(build):
    sched, request = build(
        options={
            "enabled": True,
            "max_pending_requests": 4,
            "max_pending_kv_bytes": 3072,
        }
    )
    a, b = request(), request(16)
    sched.extend([a, b])
    sched.schedule()
    assert a.block_table and not b.block_table
    assert sched.pd_admission.rejected["bytes"] > 0
    sched._update_from_kv_xfer_finished(KVConnectorOutput(finished_recving={a.id}))
    sched.schedule()
    sched.schedule()
    assert b.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert sched.pd_admission.pending_bytes == 4096
    assert sched.pd_admission.oversized_admissions == 1


@pytest.mark.parametrize("keep_referenced", [False, True])
def test_prefix_hits_charge_only_receive_targets(build, seq_factory, keep_referenced):
    sched, request = build(
        prefix=True,
        options={
            "enabled": True,
            "max_pending_requests": 4,
            "max_pending_kv_bytes": 2048,
        },
    )
    seed = seq_factory(list(range(8)))
    bm = sched.block_manager
    bm.allocate(seed, bm.can_allocate(seed))
    bm.hash_blocks(seed, seed.num_tokens)
    if not keep_referenced:
        bm.deallocate(seed)
    free_before = bm.kv.num_free
    a, b = request(), request()
    sched.extend([a, b])
    batch, _ = sched.schedule()
    assert set(batch.connector_meta_output.reqs_to_recv) == {a.id, b.id}
    assert sched.pd_admission.pending_bytes == 2048
    assert a.block_table[0] == b.block_table[0]
    assert free_before - bm.kv.num_free == (2 if keep_referenced else 3)
    for meta in batch.connector_meta_output.reqs_to_recv.values():
        assert meta.num_computed_blocks == 1


def test_incompatible_geometry_charges_full_receive(build, seq_factory):
    sched, request = build(prefix=True)
    seed = seq_factory(list(range(8)))
    bm = sched.block_manager
    bm.allocate(seed, bm.can_allocate(seed))
    bm.hash_blocks(seed, seed.num_tokens)
    bm.deallocate(seed)
    a = request()
    a.kv_transfer_params["block_size"] = 8
    sched.add(a)
    batch, _ = sched.schedule()
    assert batch.connector_meta_output.reqs_to_recv[a.id].num_computed_blocks == 0
    assert sched.pd_admission.pending_bytes == 2048


def test_cancel_keeps_credit_until_safe_terminal_and_ignores_duplicate(build):
    sched, request = build()
    a, b = request(), request()
    sched.extend([a, b])
    sched.schedule()
    free_before = sched.block_manager.kv.num_free
    assert sched.abort_request(a.id)
    assert sched.pd_admission.pending[str(a.id)].cancelled
    sched.schedule()
    assert not b.block_table
    assert sched.block_manager.kv.num_free == free_before
    # The worker has already aggregated all producers/ranks and drained writes.
    sched._update_from_kv_xfer_finished(KVConnectorOutput(failed_recving={str(a.id)}))
    assert not sched.pd_admission.pending
    assert not a.block_table
    sched.schedule()
    assert b.block_table
    # A late success for A must not retire B's newly acquired credit.
    sched._update_from_kv_xfer_finished(KVConnectorOutput(finished_recving={str(a.id)}))
    assert set(sched.pd_admission.pending) == {str(b.id)}


def test_ready_without_compute_room_keeps_credit(build):
    sched, request = build()
    a = request()
    sched.add(a)
    sched.schedule()
    sched._update_from_kv_xfer_finished(KVConnectorOutput(finished_recving={a.id}))
    # Admission can promote A, but no token fits the speculative decode width.
    sched.mtp_k = 4
    sched.spec_decode_local = True
    sched.max_num_batched_tokens = 4
    batch, _ = sched.schedule()
    assert a.status == SequenceStatus.RUNNING
    assert not batch.req_ids
    assert str(a.id) in sched.pd_admission.pending
    sched.max_num_batched_tokens = 64
    batch, _ = sched.schedule()
    assert a.id in batch.req_ids
    assert not sched.pd_admission.pending


def test_disabled_preserves_admission(build):
    sched, request = build(options={"enabled": False})
    a, b = request(), request()
    sched.extend([a, b])
    batch, _ = sched.schedule()
    assert set(batch.connector_meta_output.reqs_to_recv) == {a.id, b.id}
    assert sched.pd_admission is None


def test_speculative_first_decode_uses_existing_lookahead(build):
    sched, request = build(
        speculative_config=SimpleNamespace(
            num_speculative_tokens=3, use_dspark=lambda: True
        )
    )
    a, b = request(), request()
    sched.extend([a, b])
    sched.schedule()
    assert len(a.block_table) == 2  # Only the prompt is a receive target.
    assert sched.pd_admission.pending_bytes == 2048
    sched._update_from_kv_xfer_finished(KVConnectorOutput(finished_recving={a.id}))
    batch, _ = sched.schedule()
    assert batch.req_ids == [a.id]
    assert batch.num_spec_step == 3
    assert sched.block_manager.draft_writes_allocated(a)
    assert not sched.pd_admission.pending
    assert not b.block_table


def test_failed_receive_releases_credits_without_local_recompute(build):
    sched, request = build()
    a, b = request(), request()
    sched.extend([a, b])
    sched.schedule()
    sched._update_from_kv_xfer_finished(KVConnectorOutput(failed_recving={a.id}))
    assert sched.take_rejected() == [a]
    assert not a.block_table
    assert not sched.pd_admission.pending
    batch, _ = sched.schedule()
    assert a.id not in batch.req_ids
    assert set(batch.connector_meta_output.reqs_to_recv) == {b.id}


def test_multi_pd_ready_resume_releases_only_on_batch(build, monkeypatch):
    from test_multi_connector import FakeSchedSub

    from atom.kv_transfer.disaggregation.multi import multi_connector

    store = FakeSchedSub(is_offload=True, offload_methods=True, has_state_tier=False)
    # Complete the base P/D no-op face on the offload test double.
    store.admission_ready = lambda seq: True
    store.process_pd_completions = lambda output: output

    def wrap(pd):
        monkeypatch.setattr(
            multi_connector, "_build_subconnectors", lambda *args, **kwargs: [pd, store]
        )
        return multi_connector.MultiConnectorScheduler(SimpleNamespace())

    sched, request = build(connector_wrapper=wrap)
    a, b = request(), request()
    sched.extend([a, b])
    batch, _ = sched.schedule()
    assert set(batch.connector_meta_output.metas[0].reqs_to_recv) == {a.id}
    assert not b.block_table
    sched._update_from_kv_xfer_finished(KVConnectorOutput(finished_recving={a.id}))
    assert sched.pd_admission.pending[str(a.id)].ready_at is not None
    batch, _ = sched.schedule()
    assert a.id in batch.req_ids  # The existing Multi suffix-resume path runs.
    assert a.block_table
    assert not sched.pd_admission.pending


def test_ready_preempted_before_first_forward_releases_credit(build):
    sched, request = build(num_kvcache_blocks=2)
    a = request()
    sched.add(a)
    sched.schedule()
    sched._update_from_kv_xfer_finished(KVConnectorOutput(finished_recving={a.id}))
    # The inherited decode capacity check needs another block for T0. There is
    # no G reservation in this change, so it may safely preempt the ready row.
    batch, _ = sched.schedule()
    assert not batch.req_ids
    assert not a.block_table
    assert sched.total_preemptions == 1
    assert not sched.pd_admission.pending


def test_fraction_uses_physical_pool_bytes(build):
    sched, _ = build(
        options={
            "enabled": True,
            "max_pending_requests": None,
            "max_pending_kv_fraction": 0.25,
        }
    )
    assert sched.pd_admission.max_bytes == 64 * 1024 // 4


def test_hard_capacity_still_rejects_oversized_request(build):
    sched, request = build(
        num_kvcache_blocks=2,
        options={
            "enabled": True,
            "max_pending_kv_bytes": 1024,
        },
    )
    a = request(16)
    sched.add(a)
    sched.schedule()
    assert sched.take_rejected() == [a]
    assert not a.block_table
    assert not sched.pd_admission.pending


def test_multiconnector_uses_load_winner(build):
    from atom.kv_transfer.disaggregation.multi.multi_connector import (
        MultiConnectorScheduler,
    )

    def wrap(pd):
        class Store:
            is_offload = True
            is_producer = False

            def get_num_new_matched_tokens(self, seq):
                return (8, True)

            def cancel_pending_load(self, seq):
                pass

        multi = MultiConnectorScheduler.__new__(MultiConnectorScheduler)
        multi._connectors = [pd, Store()]
        multi._load_winner = {}
        return multi

    # Exercise the real first-hit selection without invoking an offload backend.
    sched, request = build()
    multi = wrap(sched.kv_connector)
    assert multi.supports_pd_admission
    a = request()
    multi.get_num_new_matched_tokens(a)
    assert multi.pd_receive_block_offset(a, 2, 0) == 0
    a.kv_transfer_params = {}
    multi.get_num_new_matched_tokens(a)
    assert multi.pd_receive_block_offset(a, 2, 0) is None
    multi._connectors.reverse()
    assert not multi.supports_pd_admission


@pytest.mark.parametrize(
    "options",
    [
        {"enabled": "false"},
        {"enabled": True, "max_pending_requests": 0},
        {"enabled": True, "max_pending_requests": True},
        {"enabled": True, "max_pending_kv_bytes": -1},
        {"enabled": True, "max_pending_kv_fraction": float("nan")},
        {"enabled": True, "max_pending_kv_fraction": 1.1},
        {"enabled": True, "max_pending_kv_fraction": 0.1, "max_pending_kv_bytes": 10},
        {
            "enabled": True,
            "max_pending_requests": None,
            "max_pending_kv_fraction": None,
        },
        {"enabled": True, "extra_growth_blocks_per_request": 1},
    ],
)
def test_invalid_config_rejected(build, options):
    with pytest.raises(ValueError):
        build(options=options)


@pytest.mark.parametrize(
    "override",
    [
        {"pd_admission_block_bytes": 0},
        {"pipeline_parallel_size": 2},
        {"decode_context_parallel_size": 2},
    ],
)
def test_unsupported_layout_rejected(build, override):
    with pytest.raises(ValueError):
        build(**override)


def test_physical_targets_are_deduplicated_and_release_is_idempotent():
    policy = PdAdmissionController(4, 4096, 1024)
    a, b = SimpleNamespace(id=1), SimpleNamespace(id=2)
    policy.admit(a, [1, 2])
    policy.admit(b, [2, 3])
    assert policy.pending_bytes == 3072
    policy.release(a)
    policy.release(a)
    assert policy.pending_bytes == 2048
    # Same numeric ID on a different sequence cannot release an existing entry.
    policy.release(SimpleNamespace(id=2))
    assert policy.pending_bytes == 2048
    policy.release(b)
    assert policy.pending_bytes == 0


def test_admission_metrics_follow_credit_lifecycle():
    from prometheus_client import CollectorRegistry

    from atom.metrics.scheduler import SchedulerMetrics

    registry = CollectorRegistry()
    metrics = SchedulerMetrics(registry=registry)
    policy = PdAdmissionController(1, 4096, 1024, metrics)
    seq = SimpleNamespace(id=1)
    labels = {"dp_rank": "0", "engine_role": "default"}
    policy.admit(seq, [1, 2])
    assert (
        registry.get_sample_value(
            "atom:pd_admission_pending_requests", {**labels, "state": "receiving"}
        )
        == 1
    )
    assert not policy.can_admit(1)
    assert (
        registry.get_sample_value(
            "atom:pd_admission_deferred_total", {**labels, "reason": "requests"}
        )
        == 1
    )
    policy.received(seq.id)
    assert (
        registry.get_sample_value(
            "atom:pd_admission_pending_requests", {**labels, "state": "ready"}
        )
        == 1
    )
    policy.scheduled([seq])
    assert registry.get_sample_value("atom:pd_admission_pending_kv_bytes", labels) == 0
    assert (
        registry.get_sample_value("atom:pd_admission_ready_wait_seconds_count", labels)
        == 1
    )

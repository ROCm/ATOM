# SPDX-License-Identifier: MIT
"""Producer-side PD admission: KV held by finished requests' sends bounds new prefills.

A legacy (non-chunked) producer parks a finished request whole until the decode
has pulled its KV. When prefill outpaces the transfer -- an offload tier turns
long contexts into short prefills -- those parked tables fill the pool. With
`pd_admission` enabled on the producer, a new prefill waits once the parked
sends hold the configured bytes.
"""

from types import SimpleNamespace

import pytest
from aiter_stub import stubbed_aiter
from conftest import MockConfig
from metrics_helpers import histogram_values
from test_multi_connector import FakeSchedSub

from atom.kv_transfer.disaggregation.types import KVConnectorOutput
from atom.model_engine.engine_utility import deferred_free_blocks_by_holder
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.sampling_params import SamplingParams

# 4-token blocks of 1024 bytes: an 8-token prompt holds 2 blocks, 2048 bytes.
BLOCK_BYTES = 1024


@pytest.fixture
def build(monkeypatch, seq_factory):
    with stubbed_aiter():
        from atom.kv_transfer.disaggregation.mooncake import mooncake_connector
        from atom.kv_transfer.disaggregation.multi import multi_connector
    monkeypatch.setattr(mooncake_connector, "get_open_port", lambda: 30000)
    monkeypatch.setattr(mooncake_connector, "get_ip", lambda: "127.0.0.1")

    def make(*, options=None, prefix=False, store=None, **overrides):
        config = MockConfig(
            num_kvcache_blocks=64,
            pool_entry_bytes={"kv": BLOCK_BYTES},
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
            parallel_config=SimpleNamespace(data_parallel_size=1, data_parallel_rank=0),
            enable_prefix_caching=prefix,
            kv_transfer_config={
                "kv_connector": "mooncake",
                "kv_role": "kv_producer",
                "enable_chunked_transfer": True,
                "pd_admission": (
                    options
                    if options is not None
                    else {"enabled": True, "max_pending_kv_bytes": 2048}
                ),
            },
        )
        for key, value in overrides.items():
            setattr(config, key, value)
        connector = mooncake_connector.MooncakeConnectorScheduler(config)
        if store is not None:
            # `multi`: the Mooncake producer first, then an offload store.
            subs = [connector, store]
            monkeypatch.setattr(
                multi_connector, "_build_subconnectors", lambda *args, **kw: subs
            )
            connector = multi_connector.MultiConnectorScheduler(SimpleNamespace())
        monkeypatch.setattr(
            "atom.utils.forward_context.get_kvconnector", lambda *args: connector
        )
        sched = Scheduler(config)

        def request(tokens=8, *, first=0, chunked=False):
            seq = seq_factory(
                list(range(first, first + tokens)),
                sampling_params=SamplingParams(max_tokens=1),
            )
            seq.kv_transfer_params = {"do_remote_decode": True}
            if chunked:
                seq.kv_transfer_params |= {
                    "chunked_transfer": True,
                    "transfer_id": f"t-{seq.id}",
                }
            return seq

        return sched, request

    return make


def prefill(sched, *seqs):
    """One forward that finishes `seqs`' prefill; each emits its only token."""
    batch, _ = sched.schedule()
    assert set(batch.req_ids) == {seq.id for seq in seqs}
    sched.postprocess(
        list(sched.running),
        ScheduledBatchOutput(
            req_ids=[seq.id for seq in seqs],
            token_ids=[(7,)] * len(seqs),
            num_rejected=None,
            num_bonus=None,
            draft_token_ids=None,
        ),
        batch=batch,
    )


def sent(sched, seq):
    sched._update_from_kv_xfer_finished(KVConnectorOutput(finished_sending={seq.id}))


def holders(sched):
    return deferred_free_blocks_by_holder(
        sched.deferred_free_blocks.values(), sched.kv_connector, 64
    )


def test_finished_sends_hold_the_budget_until_sent(build):
    sched, request = build()
    a, b = request(), request(first=100)
    sched.add(a)
    prefill(sched, a)
    # Parked whole for its send, and charged after the finish-time release.
    assert a.id in sched.deferred_free_blocks
    assert sched.pd_admission.pending_bytes == 2048
    assert sched.pd_admission.pending_states == {"sending": 1}
    # A forward retires no send credit.
    sched.pd_admission.scheduled([a])
    assert sched.pd_admission.pending_states == {"sending": 1}

    sched.add(b)
    # Every probe of the waiting work agrees with Phase 2.
    assert not sched._can_admit_head_prefill()
    assert sched._local_prefill_pending_work() == (False, 0)
    for _ in range(2):
        sched.schedule()
        assert not b.block_table
    # One deferral per admission wait, however many passes it takes.
    assert sched.pd_admission.deferred["bytes"] == 1

    sent(sched, a)
    assert a.id not in sched.deferred_free_blocks
    assert not sched.pd_admission.pending
    assert sched._can_admit_head_prefill()
    prefill(sched, b)


def test_shared_prefix_blocks_are_charged_once(build):
    sched, request = build(
        prefix=True, options={"enabled": True, "max_pending_kv_bytes": 8192}
    )
    a, b = request(12), request(12)  # same tokens: b hits a's two full blocks
    sched.add(a)
    prefill(sched, a)
    sched.add(b)
    prefill(sched, b)
    assert {a.id, b.id} <= set(sched.deferred_free_blocks)
    shared = set(a.block_table) & set(b.block_table)
    assert shared
    union = set(a.block_table) | set(b.block_table)
    assert sched.pd_admission.pending_bytes == len(union) * BLOCK_BYTES


def test_budget_is_soft_by_what_was_admitted_when_it_filled(build):
    sched, request = build()
    a, b, c = request(), request(first=100), request(first=200)
    sched.extend([a, b])
    # Nothing is parked yet, so both start; each holds the whole budget alone.
    prefill(sched, a, b)
    assert sched.pd_admission.pending_bytes == 4096  # 2048 over, bounded by a+b
    sched.add(c)
    sched.schedule()
    assert not c.block_table
    sent(sched, a)
    sched.schedule()
    # b alone still holds the whole budget.
    assert not c.block_table
    sent(sched, b)
    prefill(sched, c)


def test_chunked_transfers_are_neither_gated_nor_charged(build):
    sched, request = build()
    a = request()
    sched.add(a)
    prefill(sched, a)
    chunked = request(first=100, chunked=True)
    legacy = request(first=200)
    sched.extend([chunked, legacy])
    # D's readiness report, as `process_pd_completions` records it.
    sched.kv_connector._chunked_ready.add(chunked.id)
    # The budget is full: the chunked request starts, the legacy one waits.
    prefill(sched, chunked)
    assert not legacy.block_table
    # Parked for its send, but D's readiness paced it: no credit.
    assert chunked.id in sched.deferred_free_blocks
    assert sched.kv_connector.pd_send_pending(chunked)
    assert set(sched.pd_admission.pending) == {str(a.id)}


def test_send_credit_retires_with_the_send_while_a_save_parks_the_blocks(
    build, monkeypatch
):
    now = [100.0]
    monkeypatch.setattr("atom.metrics.scheduler.time.perf_counter", lambda: now[0])
    store = FakeSchedSub(is_offload=True, offload_methods=True, has_state_tier=False)
    # Complete the base P/D face on the offload test double. Its save claims
    # every finished request until it reports.
    store.admission_ready = lambda seq: True
    store.process_pd_completions = lambda output: output
    store.pd_send_pending = lambda seq: False
    store.defer = True
    sched, request = build(store=store)
    a, local, b = request(), request(first=100), request(first=200)
    local.kv_transfer_params = None  # served on P alone: no send
    sched.extend([a, local])
    prefill(sched, a, local)
    # The save parks both, and a also waits on its send: only a takes a credit.
    assert {a.id, local.id} <= set(sched.deferred_free_blocks)
    assert set(sched.pd_admission.pending) == {str(a.id)}
    assert holders(sched) == {"send": 2, "store": 2, "aborted_load": 0}
    sched.add(b)
    sched.schedule()
    assert not b.block_table

    now[0] += 3
    sent(sched, a)
    # The send is done; the save still parks a's blocks, under `store` now.
    assert a.id in sched.deferred_free_blocks
    assert not sched.pd_admission.pending
    assert holders(sched) == {"send": 0, "store": 4, "aborted_load": 0}
    assert histogram_values(sched.metrics.pd_send_hold)["sum"] == 3
    sent(sched, a)  # a repeated report observes nothing
    assert histogram_values(sched.metrics.pd_send_hold)["sum"] == 3
    prefill(sched, b)

    now[0] += 5
    store.defer = False
    sched._update_from_kv_xfer_finished(
        KVConnectorOutput(finished_saving={a.id, local.id})
    )
    assert not {a.id, local.id} & set(sched.deferred_free_blocks)
    # b's send is still pending; the saves added no hold time.
    assert set(sched.pd_admission.pending) == {str(b.id)}
    assert histogram_values(sched.metrics.pd_send_hold)["sum"] == 3


def test_disabled_producer_admission_keeps_admitting(build):
    sched, request = build(options={"enabled": False})
    assert sched.pd_admission is None
    a, b = request(), request(first=100)
    sched.add(a)
    prefill(sched, a)
    assert a.id in sched.deferred_free_blocks
    sched.add(b)
    prefill(sched, b)


@pytest.mark.parametrize(
    "options,match",
    [
        # A producer has no default budget.
        ({"enabled": True}, "needs max_pending_kv_bytes or max_pending_kv_fraction"),
        # The decode growth reserve is a consumer's.
        (
            {
                "enabled": True,
                "max_pending_kv_bytes": 2048,
                "num_reserved_decode_tokens": 8,
            },
            "applies to a decode",
        ),
    ],
)
def test_invalid_producer_config_rejected(build, options, match):
    with pytest.raises(ValueError, match=match):
        build(options=options)


def test_producer_fraction_uses_physical_pool_bytes(build):
    sched, _ = build(options={"enabled": True, "max_pending_kv_fraction": 0.5})
    assert sched.pd_admission.producer
    assert sched.pd_admission.max_bytes == int(64 * BLOCK_BYTES * 0.5)

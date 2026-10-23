"""DPScheduleSync — the scheduler's one cross-DP collective per tick.

It sums the PrefillDelayer's decision counts and hands every rank each rank's
prefill chunk cost (rounded up), carrying only what is on. Real gloo groups —
one thread per rank, stateless like the engine's DP group — check the layout
and two schedulers running in lockstep; the engine wiring is checked on its
own.
"""

import threading
from types import SimpleNamespace

import pytest
import torch.distributed as dist
from aiter_stub import stubbed_aiter
from conftest import MockConfig

from atom.model_engine.dp_schedule_sync import DPScheduleSync
from atom.model_engine.prefill_delayer import PrefillDelayer
from atom.model_engine.scheduler import Scheduler

with stubbed_aiter():
    from atom.model_engine.engine_core import DPEngineCoreProc

SLOTS = PrefillDelayer.DECISION_SLOTS


def _on_gloo_ranks(dp_size, body):
    """Run `body(group)` on `dp_size` gloo ranks, one thread each, and return
    what each rank returned."""
    store = dist.HashStore()
    results = [None] * dp_size
    errors = []

    def run(rank):
        try:
            results[rank] = body(dist.ProcessGroupGloo(store, rank, dp_size))
        except Exception as e:  # noqa: BLE001 - re-raised on the test thread
            errors.append(e)

    threads = [threading.Thread(target=run, args=(r,)) for r in range(dp_size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not any(t.is_alive() for t in threads), "a rank is stuck in the collective"
    if errors:
        raise errors[0]
    return results


def _rank(r):
    return SimpleNamespace(rank=lambda: r)


def test_sums_counts_and_hands_every_rank_every_cost():
    def body(group):
        r = group.rank()
        sync = DPScheduleSync(group, 3, decision_slots=SLOTS, prefill_costs=True)
        return sync.exchange([1, 100 * (r + 1), r, 0, 0, 0, 0, 0], 10.0 * r + 0.5)

    for counts, costs in _on_gloo_ranks(3, body):
        assert counts == [3, 600, 3, 0, 0, 0, 0, 0]
        assert costs == [1.0, 11.0, 21.0]


def test_carries_only_what_is_on():
    def body(group):
        costs_only = DPScheduleSync(group, 2, decision_slots=0, prefill_costs=True)
        counts_only = DPScheduleSync(
            group, 2, decision_slots=SLOTS, prefill_costs=False
        )
        return costs_only.exchange(None, 2.0), counts_only.exchange([1] * SLOTS, 2.0)

    for costs_only, counts_only in _on_gloo_ranks(2, body):
        assert costs_only == (None, [2.0, 2.0])
        assert counts_only == ([2] * SLOTS, [])


@pytest.mark.parametrize(
    "cost, carried",
    [
        (0.0, 0.0),
        (2048.0, 2048.0),
        (1984.5, 1985.0),
        (2_869_681_200.5, 2_869_681_201.0),
    ],
)
def test_rounds_costs_up(cost, carried):
    group = dist.ProcessGroupGloo(dist.HashStore(), 0, 1)
    sync = DPScheduleSync(group, 1, decision_slots=0, prefill_costs=True)
    assert sync.exchange(None, cost) == (None, [carried])


def test_needs_something_to_carry():
    with pytest.raises(ValueError, match="neither on"):
        DPScheduleSync(_rank(0), 2, decision_slots=0, prefill_costs=False)


def test_rank_must_be_inside_the_group():
    with pytest.raises(ValueError, match="outside"):
        DPScheduleSync(_rank(2), 2, decision_slots=SLOTS, prefill_costs=True)


def _peer_down(self, buf):
    raise RuntimeError("Connection closed by peer")


def test_failed_collective_returns_nothing(monkeypatch):
    monkeypatch.setattr(DPScheduleSync, "_sum_over_ranks", _peer_down)
    sync = DPScheduleSync(_rank(0), 2, decision_slots=SLOTS, prefill_costs=True)
    assert sync.exchange([1] * SLOTS, 2048.0) == (None, [])


def _dp_sched(monkeypatch, dp_size=2):
    monkeypatch.setenv("ATOM_DP_PREFILL_BALANCE_TOKEN_WEIGHT", "0")
    monkeypatch.setenv("ATOM_DP_PREFILL_BALANCE_SLACK", "0.1")
    monkeypatch.setenv("ATOM_DP_PREFILL_BALANCE_MIN_TOKENS", "8")
    return Scheduler(
        MockConfig(
            max_num_seqs=8,
            num_kvcache_blocks=64,
            kv_cache_block_size=4,
            max_model_len=256,
            max_num_batched_tokens=64,
            enable_dp_attention=True,
            parallel_config=SimpleNamespace(data_parallel_size=dp_size),
        )
    )


def test_scheduler_admits_unbalanced_when_the_collective_fails(
    monkeypatch, seq_factory
):
    """A peer going down must not wedge schedule(): the delayer admits (here a
    tick it would otherwise hold, the other rank having no prefill) and
    nothing balances."""
    monkeypatch.setattr(DPScheduleSync, "_sum_over_ranks", _peer_down)
    sched = _dp_sched(monkeypatch)
    delayer = PrefillDelayer(dp_size=2, max_num_batched_tokens=64)
    delayer._first = False
    sched.set_prefill_delayer(delayer)
    sched.set_dp_sync(
        DPScheduleSync(_rank(0), 2, decision_slots=SLOTS, prefill_costs=True)
    )
    sched.add(seq_factory(list(range(100, 164))))
    sched.add(seq_factory([1, 2, 3, 4]))
    batch, _ = sched.schedule()
    assert list(batch.num_scheduled_tokens) == [64]
    assert sched._balance_stats[0] == 0


def test_two_ranks_balance_in_lockstep(monkeypatch, seq_factory):
    """Delayer and balancing on, one real collective per tick: the rank whose
    chunk costs more shrinks to the target, the cheaper one keeps its chunk."""
    # Costs 64 * 32 = 2048 and 16 * 8 = 128: target = 1.1 * 1088 = 1196.8,
    # i.e. 48 tokens for rank 0.
    prompts = [list(range(100, 164)), list(range(300, 316))]

    def body(group):
        sched = _dp_sched(monkeypatch)
        sched.set_prefill_delayer(PrefillDelayer(dp_size=2, max_num_batched_tokens=64))
        sched.set_dp_sync(
            DPScheduleSync(group, 2, decision_slots=SLOTS, prefill_costs=True)
        )
        sched.add(seq_factory(prompts[group.rank()]))
        batch, _ = sched.schedule()
        return list(batch.num_scheduled_tokens)

    assert _on_gloo_ranks(2, body) == [[48], [16]]


@pytest.mark.parametrize("delayer_on", [True, False])
def test_a_lone_rank_keeps_an_odd_chunk_across_real_ranks(
    monkeypatch, seq_factory, delayer_on
):
    """Rank 1 is idle but still joins the collective; rank 0 alone sets the
    target with an odd 63-token chunk (cost 1984.5) and keeps all of it."""

    def body(group):
        sched = _dp_sched(monkeypatch)
        if delayer_on:
            sched.set_prefill_delayer(
                PrefillDelayer(dp_size=2, max_num_batched_tokens=64)
            )
        sched.set_dp_sync(
            DPScheduleSync(
                group,
                2,
                decision_slots=SLOTS if delayer_on else 0,
                prefill_costs=True,
            )
        )
        if group.rank() == 0:
            sched.add(seq_factory(list(range(100, 163))))
        result = sched.schedule()
        return None if result is None else list(result[0].num_scheduled_tokens)

    assert _on_gloo_ranks(2, body) == [[63], None]


def _engine(delayer_on, balance_on, *, rapidserve=False, scheduler=True):
    proc = DPEngineCoreProc.__new__(DPEngineCoreProc)
    attached = []
    proc.scheduler = (
        SimpleNamespace(
            prefill_delayer=(
                PrefillDelayer(dp_size=2, max_num_batched_tokens=64)
                if delayer_on
                else None
            ),
            cost_balance=balance_on,
            set_dp_sync=attached.append,
        )
        if scheduler
        else None
    )
    proc.dp_group = _rank(1)
    config = SimpleNamespace(
        enable_rapidserve=rapidserve,
        parallel_config=SimpleNamespace(data_parallel_size=2),
    )
    proc._init_dp_schedule_sync(config)
    return attached


@pytest.mark.parametrize(
    "delayer_on, balance_on, decision_slots",
    [(True, True, SLOTS), (True, False, SLOTS), (False, True, 0)],
)
def test_engine_wires_one_sync_for_what_is_on(delayer_on, balance_on, decision_slots):
    (sync,) = _engine(delayer_on, balance_on)
    assert sync.decision_slots == decision_slots
    assert sync.prefill_costs is balance_on
    assert sync.dp_size == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"delayer_on": False, "balance_on": False},
        {"delayer_on": True, "balance_on": True, "rapidserve": True},
        {"delayer_on": False, "balance_on": True, "scheduler": False},
    ],
)
def test_engine_wires_no_sync_when_nothing_needs_it(kwargs):
    assert _engine(**kwargs) == []

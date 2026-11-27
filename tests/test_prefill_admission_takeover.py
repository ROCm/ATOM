"""CPU-only admission-port regressions; no real distributed or GPU work.

For the pinned default-off differential test set GPU3_ADMISSION_BASE to the
preserved r3_fix1 base_inputs directory delivered alongside the patch.
"""

import importlib.util
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from conftest import MockConfig
from test_prefill_short_first import _Delayer, _prompt, _sched, _shorts, _step

from atom.model_engine.prefill_delayer import _FIXED_SLOTS, PrefillDelayer
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import SequenceStatus, SequenceType
from atom.utils import envs

KNOBS = {
    "ATOM_PREFILL_SHORT_FIRST_ROWS": 0,
    "ATOM_PREFILL_SHORT_FIRST_LONG_ROWS": 4096,
    "ATOM_PREFILL_SHORT_FIRST_MAX_SKIPS": 32,
    "ATOM_PREFILL_COST_CEIL": False,
    "ATOM_PREFILL_DECODE_INTERVAL_BUSY": 0,
    "ATOM_PREFILL_DECODE_INTERVAL_BUSY_SEQS": 48,
    "ATOM_PREFILL_LONG_CAP_BUSY": 0,
    "ATOM_PREFILL_LONG_CAP_BUSY_SEQS": 48,
}


def test_all_admission_defaults(monkeypatch):
    for key, value in KNOBS.items():
        monkeypatch.delenv(key, raising=False)
        assert getattr(envs, key) == value
    sched = Scheduler(MockConfig())
    assert sched._short_first_rows == sched._busy_long_prefill_cap == 0
    d = PrefillDelayer(dp_size=1, cpu_group=None, max_num_batched_tokens=16)
    assert not d._cost_ceil
    assert d.busy_decode_interval == d.group_running_decode == 0


@pytest.mark.parametrize("where", ["running", "waiting"])
def test_repeated_short_arrivals_do_not_starve_long(monkeypatch, seq_factory, where):
    sched = _sched(monkeypatch, rows=4, max_skips=2)
    long = _prompt(seq_factory, 1000, 64)
    sched.add(long)
    if where == "running":
        assert _step(sched) == [(long, 8)]
    history = []
    for tick in range(12):
        if not long.is_partial_prefill and long not in sched.waiting:
            break
        sched.extend(_shorts(seq_factory, tick))
        taken = _step(sched)
        history.append({s.id: k for s, k in taken}.get(long.id, 0))
        assert sched._partial_prefill_count == sum(
            s.is_partial_prefill for s in sched.running
        )
    assert long.num_cached_tokens == 64
    if where == "running":
        assert all(k == 8 for k in history)
    else:
        assert history[:2] == [0, 0]
        assert all(k == 8 for k in history[2:])
        assert long.num_skipped_steps == 2


def test_skips_do_not_age_on_empty_or_blocked_ticks(monkeypatch, seq_factory):
    sched = _sched(monkeypatch, rows=4, max_skips=2)
    long = _prompt(seq_factory, 1000, 40)
    sched.add(long)
    real = sched.block_manager.can_allocate_memoized
    exact = sched.block_manager.can_allocate
    monkeypatch.setattr(sched.block_manager, "can_allocate_memoized", lambda seq: -1)
    monkeypatch.setattr(sched.block_manager, "can_allocate", lambda *args, **kw: -1)
    for _ in range(4):
        _batch, selected = sched.schedule()
        assert not selected
    assert long.num_skipped_steps == 0
    assert long in sched.waiting
    monkeypatch.setattr(sched.block_manager, "can_allocate_memoized", real)
    monkeypatch.setattr(sched.block_manager, "can_allocate", exact)
    for tick in range(2):
        sched.extend(_shorts(seq_factory, tick))
        assert long not in dict(_step(sched))
    assert long.num_skipped_steps == 2
    assert _step(sched)[0] == (long, 8)


@pytest.mark.parametrize("kind", ["aborted", "remote", "unschedulable", "kv"])
def test_all_ordering_barriers_and_prefix_probe_are_respected(
    monkeypatch, seq_factory, kind
):
    sched = _sched(monkeypatch, rows=4)
    a, short, barrier, tail = [
        _prompt(seq_factory, 1000 + i * 1000, n) for i, n in enumerate((40, 4, 40, 4))
    ]
    sched.extend([a, short, barrier, tail])
    probe = sched.block_manager.can_allocate_memoized
    seen = []

    def checked(seq):
        seen.append(seq)
        return -1 if kind == "kv" and seq is barrier else probe(seq)

    monkeypatch.setattr(sched.block_manager, "can_allocate_memoized", checked)
    if kind == "aborted":
        barrier.status = SequenceStatus.ABORTED
    elif kind == "remote":
        barrier.status = SequenceStatus.WAITING_FOR_REMOTE_KVS
    elif kind == "unschedulable":
        reason = sched._unschedulable_reason
        monkeypatch.setattr(
            sched,
            "_unschedulable_reason",
            lambda seq: "blocked for test" if seq is barrier else reason(seq),
        )
    before = (
        sched.block_manager.num_free_blocks
        if hasattr(sched.block_manager, "num_free_blocks")
        else None
    )
    sched._short_first_order()
    assert list(sched.waiting) == [short, a, barrier, tail]
    assert tail not in seen
    assert before == (
        sched.block_manager.num_free_blocks
        if hasattr(sched.block_manager, "num_free_blocks")
        else None
    )
    assert sched._partial_prefill_count == 0


@pytest.mark.parametrize("fail", [False, True])
def test_cost_limit_restored_even_if_partial_resume_raises(
    monkeypatch, seq_factory, fail
):
    sched = _sched(monkeypatch, rows=4)
    long = _prompt(seq_factory, 1000, 40)
    sched.add(long)
    assert _step(sched) == [(long, 8)]
    sched._tick_cost = 900.0
    sched._tick_cost_limit = 1.0
    calls = []

    def resume(seqs, scheduled, tokens, count, batched, budget):
        calls.append((list(seqs), sched._tick_cost_limit))
        if len(calls) == 1:
            assert sched._tick_cost_limit > 900
            if fail:
                raise RuntimeError("injected finalize/resume failure")
        else:
            assert sched._tick_cost_limit == 1.0
        return count, batched

    monkeypatch.setattr(sched, "_resume_partial_prefills", resume)
    if fail:
        with pytest.raises(RuntimeError, match="injected"):
            sched._resume_long_partials(
                [long], {}, [], 0, 0, sched.max_num_batched_tokens
            )
    else:
        assert sched._resume_long_partials(
            [long], {}, [], 0, 0, sched.max_num_batched_tokens
        ) == (0, 0)
    assert sched._tick_cost_limit == 1.0


class _Group:
    def __init__(self, rank):
        self._rank = rank

    def rank(self):
        return self._rank


@pytest.mark.parametrize("ceil", [False, True])
def test_dp_cost_ceil_rounds_per_rank_and_preserves_target_room(monkeypatch, ceil):
    monkeypatch.setenv("ATOM_PREFILL_COST_CEIL", str(int(ceil)))
    costs = [40.5, 40.5, 0.0]
    rounded = [math.ceil(x) if ceil else int(x) for x in costs]
    observed = []

    def reduce(tensor, op=None, group=None):
        rank = group.rank()
        observed.append((rank, tensor[_FIXED_SLOTS + rank].item()))
        assert observed[-1][1] == rounded[rank]
        for i, value in enumerate(rounded):
            tensor[_FIXED_SLOTS + i] = value

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    for rank in range(3):
        d = PrefillDelayer(
            dp_size=3,
            cpu_group=_Group(rank),
            max_num_batched_tokens=16,
            exchange_prefill_costs=True,
        )
        d.should_allow_prefill(
            prefill_budget=16,
            prefillable=rank < 2,
            pending_tokens=9,
            running_decode_batch=0,
            kv_usage=0.0,
            prefill_cost=costs[rank],
        )
        assert d.rank_prefill_costs == rounded
        sched = Scheduler(MockConfig(max_num_batched_tokens=16))
        sched.prefill_delayer = d
        sched._balance_token_weight = 0
        sched._balance_slack = 0.1
        sched._balance_min_tokens = 0
        sched._balance_prefill_step(costs[rank])
        assert sched._tick_cost_limit == rounded[0]
        assert sched._prefill_room(0, 0, 16) == (9 if ceil else 8)
    assert len(observed) == 3


@pytest.mark.parametrize(
    "busy,decode,interval", [(0, (1, 49), 2), (5, (1, 49), 5), (5, (1, 2), 2)]
)
@pytest.mark.parametrize("prefill_only", [False, True])
def test_busy_interval_same_across_uneven_ranks(
    monkeypatch, busy, decode, interval, prefill_only
):
    ds = [
        PrefillDelayer(
            dp_size=2,
            cpu_group=_Group(r),
            max_num_batched_tokens=16,
            prefill_decode_interval=2,
            busy_decode_interval=busy,
            busy_decode_seqs=48,
        )
        for r in range(2)
    ]
    # Drive the real encoding/readback path with a deterministic SUM collective.
    execution = [0]

    def reduce(tensor, op=None, group=None):
        tensor[0] = 2
        tensor[1] = 32
        tensor[2] = sum(decode)
        tensor[7] = execution[0]  # only rank 0 really prefills; both see its report
        tensor[10] = 0
        tensor[11] = execution[0] if prefill_only else 0

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)

    def tick():
        return [
            d.should_allow_prefill(
                prefill_budget=16,
                prefillable=True,
                pending_tokens=16,
                running_decode_batch=decode[r],
                kv_usage=0.1,
            ).allow
            for r, d in enumerate(ds)
        ]

    assert tick() == [True, True]
    execution[0] = 1
    if not prefill_only:
        tick()
        assert [d._decode_interval_remaining for d in ds] == [0, 0]
        assert [d.group_running_decode for d in ds] == [sum(decode)] * 2
        return
    assert tick() == [False, False]
    assert [d._decode_interval_remaining for d in ds] == [interval - 1] * 2
    assert [d.group_running_decode for d in ds] == [sum(decode)] * 2
    execution[0] = 0
    for _ in range(interval - 1):
        assert tick() == [False, False]
    assert tick() == [True, True]


def test_busy_ignored_by_adaptive_policy():
    d = PrefillDelayer(
        dp_size=1,
        cpu_group=None,
        max_num_batched_tokens=16,
        admit_policy="adaptive",
        busy_decode_interval=5,
    )
    assert d.busy_decode_interval == 0


@pytest.mark.parametrize(
    "base_cap,busy_cap,load,expected",
    [(8, 4, 47, 8), (8, 4, 48, 4), (8, 12, 48, 8), (0, 4, 48, 4), (8, 0, 100, 8)],
)
def test_busy_long_cap_and_replay_semantics(
    monkeypatch, seq_factory, base_cap, busy_cap, load, expected
):
    monkeypatch.setenv("ATOM_PREFILL_LONG_CAP_BUSY", str(busy_cap))
    sched = _sched(monkeypatch, rows=0, long_prefill_token_threshold=base_cap)
    assert sched._long_prefill_cap() == base_cap  # no delayer: no busy signal
    sched.prefill_delayer = SimpleNamespace(group_running_decode=load)
    assert sched._long_prefill_cap() == expected
    seq = _prompt(seq_factory, 1000, 40)
    # Replay rows from 4 to 8 are in addition to the new-token cap.
    assert sched._new_prefill_tokens(seq, 4, resume=8) == expected + 4
    sched.enable_chunked_prefill = False
    assert sched._new_prefill_tokens(seq, 4, resume=8) == 36


def _base_class(rel, name):
    base = os.environ.get("GPU3_ADMISSION_BASE")
    if not base:
        pytest.skip("pinned differential requires GPU3_ADMISSION_BASE")
    path = Path(base) / rel
    spec = importlib.util.spec_from_file_location("_gpu3_pinned_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


@pytest.mark.parametrize("explicit_zero", [False, True])
@pytest.mark.parametrize("prefix", [False, True])
@pytest.mark.parametrize("balance", [False, True])
def test_default_off_exact_pinned_schedule_trace(
    monkeypatch, seq_factory, explicit_zero, prefix, balance
):
    for key in KNOBS:
        monkeypatch.delenv(key, raising=False)
    if explicit_zero:
        for key in (
            "ATOM_PREFILL_SHORT_FIRST_ROWS",
            "ATOM_PREFILL_COST_CEIL",
            "ATOM_PREFILL_DECODE_INTERVAL_BUSY",
            "ATOM_PREFILL_LONG_CAP_BUSY",
        ):
            monkeypatch.setenv(key, "0")
    base_cls = _base_class("atom/model_engine/scheduler.py", "Scheduler")
    config = {
        "max_num_seqs": 64,
        "num_kvcache_blocks": 400,
        "kv_cache_block_size": 4,
        "max_model_len": 256,
        "max_num_batched_tokens": 16,
        "long_prefill_token_threshold": 8,
        "enable_prefix_caching": prefix,
        "enable_dp_attention": balance,
        "parallel_config": SimpleNamespace(data_parallel_size=2),
    }
    old, new = [cls(MockConfig(**config)) for cls in (base_cls, Scheduler)]
    if balance:
        for sched in (old, new):
            sched.set_prefill_delayer(_Delayer([100.0]))

    def forbidden(*args):
        pytest.fail("default-off called short-first/skip path")

    monkeypatch.setattr(new, "_short_first_order", forbidden)
    monkeypatch.setattr(new, "_count_passed_over", forbidden)
    requests = [[], []]
    for i, sched in enumerate((old, new)):
        if prefix:
            seed = seq_factory(list(range(1000, 1036)))
            sched.block_manager.allocate(seed, 0)
            sched.block_manager.hash_blocks(seed, 36)
            sched.block_manager.deallocate(seed)
        for first, length in [(2000, 40), (1000, 40), (3000, 4), (4000, 28)]:
            seq = _prompt(seq_factory, first, length)
            requests[i].append(seq)
            sched.add(seq)

    def snapshot(sched, seqs):
        batch, selected = sched.schedule()
        by_id = {s.id: i for i, s in enumerate(seqs)}
        order = [
            (by_id[s.id], int(k), s.type.name)
            for s, k in zip(selected.values(), batch.num_scheduled_tokens)
        ]
        state = [
            (
                s.num_cached_tokens,
                s.is_partial_prefill,
                s.status.name,
                list(s.block_table),
            )
            for s in seqs
        ]
        queues = (
            [by_id[s.id] for s in sched.waiting],
            [by_id[s.id] for s in sched.running],
        )
        for s, k in zip(selected.values(), batch.num_scheduled_tokens):
            if s.type == SequenceType.PREFILL:
                s.num_cached_tokens += int(k)
                partial = s.num_cached_tokens < s.num_prompt_tokens
                if not partial:
                    s.append_token(99)
                if partial != s.is_partial_prefill:
                    sched._partial_prefill_count += 1 if partial else -1
                    s.is_partial_prefill = partial
        assert sched._partial_prefill_count == sum(
            s.is_partial_prefill for s in sched.running
        )
        return (
            order,
            state,
            queues,
            sched._tick_cost,
            sched._tick_cost_limit,
            sched._partial_prefill_count,
        )

    for tick in range(12):
        if balance:
            old.prefill_delayer.allow = new.prefill_delayer.allow = tick not in (2, 5)
        assert snapshot(old, requests[0]) == snapshot(new, requests[1])
    assert all(s.num_skipped_steps == 0 for s in requests[1])


def test_default_off_delayer_matches_pinned_trace(monkeypatch):
    monkeypatch.delenv("ATOM_PREFILL_COST_CEIL", raising=False)
    cls = _base_class("atom/model_engine/prefill_delayer.py", "PrefillDelayer")
    kwargs = {
        "dp_size": 1,
        "cpu_group": None,
        "max_num_batched_tokens": 16,
        "prefill_decode_interval": 2,
        "exchange_prefill_costs": True,
    }
    old, new = cls(**kwargs), PrefillDelayer(**kwargs)
    for tick in range(20):
        for d in (old, new):
            d._prefill_executed_since_last_decision = tick in (1, 9)
        kwargs = {
            "prefillable": tick % 5 != 0,
            "pending_tokens": (tick * 3) % 17,
            "running_decode_batch": 0 if tick % 7 == 0 else 8,
            "kv_usage": 0.95 if tick == 13 else 0.1,
            "has_partial": tick % 3 == 0,
            "prefill_cost": 20.5,
            "waiting_requests": 3,
            "prefill_budget": 16,
        }
        assert tuple(old.should_allow_prefill(**kwargs)) == tuple(
            new.should_allow_prefill(**kwargs)
        )
        for attr in (
            "_decode_interval_remaining",
            "_hold_ticks",
            "_stall_count",
            "rank_prefill_costs",
        ):
            assert getattr(old, attr) == getattr(new, attr)


@pytest.mark.parametrize(
    "helper", ["_resume_partial_prefills", "_resume_long_partials"]
)
@pytest.mark.parametrize("budget", [0, 4])
def test_partial_resume_honors_reduced_mixed_budget(
    monkeypatch, seq_factory, helper, budget
):
    sched = _sched(monkeypatch, rows=4)
    seq = _prompt(seq_factory, 1000, 64)
    sched.add(seq)
    assert _step(sched) == [(seq, 8)]
    scheduled, tokens = {}, []
    counts = getattr(sched, helper)([seq], scheduled, tokens, 0, 0, budget)
    assert counts == (int(budget > 0), budget)
    assert tokens == ([budget] if budget else [])
    assert len(scheduled) == int(budget > 0)

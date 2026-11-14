# SPDX-License-Identifier: MIT
"""Phase semantics and integration with ATOM's real CPU batch allocator."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from threading import Barrier, Lock

import pytest
import torch

from atom.model_engine.deepinfer_scheduler import DeepInferDPPolicy


class MinCollective:
    """Rendezvous actual rank calls, failing on a missing/misordered vote."""

    def __init__(self, size):
        self.barrier = Barrier(size, timeout=10)
        self.lock = Lock()
        self.rows = []
        self.rounds = []

    def all_reduce(self, tensor, op, group):
        assert group is self
        assert op == torch.distributed.ReduceOp.MIN
        with self.lock:
            self.rows.append(tensor)
        leader = self.barrier.wait() == 0
        if leader:
            values = [int(row.item()) for row in self.rows]
            self.rounds.append(values)
            for row in self.rows:
                row.fill_(min(values))
        self.barrier.wait()
        if leader:
            self.rows.clear()
        self.barrier.wait()


@pytest.fixture
def group(monkeypatch):
    collective = MinCollective(2)
    monkeypatch.setattr(torch.distributed, "all_reduce", collective.all_reduce)
    with ThreadPoolExecutor(max_workers=2) as executor:
        yield collective, executor


def policies(group, **kwargs):
    collective, _ = group
    return [DeepInferDPPolicy(2, collective, **kwargs) for _ in range(2)]


def step(group, ranks, ready, admitted=None, usage=(0.0, 0.0)):
    if admitted is None:
        admitted = ready

    def run(rank):
        policy = ranks[rank]
        allow = policy.begin_step(has_prefill=ready[rank], kv_usage=usage[rank])
        prefill = policy.finish_step(prefill_scheduled=allow and admitted[rank])
        return allow, prefill, policy.decode_step

    result = list(group[1].map(run, range(2)))
    assert result[0][1:] == result[1][1:], "DP phase/clock diverged"
    return result


def test_all_at_zero_allows_consecutive_prefills(group):
    ranks = policies(group)
    for _ in range(3):
        assert step(group, ranks, (True, True)) == [(True, True, 0)] * 2
    assert len(group[0].rounds) == 6  # admission + actual output, every time


def test_isolated_rank_waits_for_any_even_without_running_decode(group):
    ranks = policies(group)
    # No ATOM skip-first, no 'no decode -> prefill' shortcut, no interval reset.
    for tick in range(100):
        assert step(group, ranks, (True, False)) == [(False, False, tick + 1)] * 2
    assert len(group[0].rounds) == 2  # ALL at 0 and 70; no per-tick vote
    assert step(group, ranks, (True, False)) == [
        (True, True, 0),
        (False, True, 0),
    ]
    assert len(group[0].rounds) == 4


def test_any_wins_coincident_all_point(group):
    ranks = policies(group)
    for policy in ranks:
        policy.decode_step = 700
    assert step(group, ranks, (False, True)) == [
        (False, True, 0),
        (True, True, 0),
    ]
    assert all(policy.last_vote == "any" for policy in ranks)


def test_failed_allocation_falls_back_to_decode_without_reset(group):
    ranks = policies(group)
    assert step(group, ranks, (True, True), (False, False)) == [
        (True, False, 1),
        (True, False, 1),
    ]
    assert step(group, ranks, (True, True)) == [(False, False, 2)] * 2
    assert len(group[0].rounds) == 2


def test_one_successful_admission_commits_global_prefill(group):
    assert step(group, policies(group), (True, True), (True, False)) == [
        (True, True, 0),
        (True, True, 0),
    ]


def test_kv_watermark_is_inclusive_and_high_rank_does_not_admit(group):
    ranks = policies(group)
    assert step(group, ranks, (True, True), usage=(0.9, 0.9)) == [(True, True, 0)] * 2
    assert (
        step(group, ranks, (True, True), usage=(0.9, 0.90001))
        == [(False, False, 1)] * 2
    )
    for policy in ranks:
        policy.decode_step = 100
    assert step(group, ranks, (True, True), usage=(0.9, 0.90001)) == [
        (True, True, 0),
        (False, True, 0),
    ]


def test_single_rank_has_same_clock_without_collectives(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("DP=1 must not require a process group")

    monkeypatch.setattr(torch.distributed, "all_reduce", unexpected)
    policy = DeepInferDPPolicy(1, prefill_interval=2, prefill_any_interval=3)
    assert policy.begin_step(has_prefill=True, kv_usage=0)
    assert not policy.finish_step(prefill_scheduled=False)
    assert not policy.begin_step(has_prefill=True, kv_usage=0)
    assert not policy.finish_step(prefill_scheduled=False)
    assert policy.begin_step(has_prefill=True, kv_usage=0)
    assert policy.finish_step(prefill_scheduled=True)
    assert policy.decode_step == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dp_size": 0},
        {"dp_size": 2},
        {"prefill_interval": 0},
        {"prefill_any_interval": -1},
    ],
)
def test_invalid_policy_configuration(kwargs):
    with pytest.raises(ValueError):
        DeepInferDPPolicy(**{"dp_size": 1, **kwargs})


def test_collective_failure_does_not_fail_open(monkeypatch):
    def failed(*args, **kwargs):
        raise RuntimeError("peer disconnected")

    monkeypatch.setattr(torch.distributed, "all_reduce", failed)
    policy = DeepInferDPPolicy(2, object())
    with pytest.raises(RuntimeError, match="peer disconnected"):
        policy.begin_step(has_prefill=True, kv_usage=0)
    assert policy.decode_step == 0


def make_scheduler(group, **overrides):
    from conftest import MockConfig

    from atom.model_engine.scheduler import Scheduler

    config = MockConfig(
        **{
            "num_kvcache_blocks": 100,
            "max_num_batched_tokens": 8,
            "enable_dp_attention": True,
            **overrides,
        }
    )
    scheduler = Scheduler(config)
    scheduler.set_dp_phase_policy(DeepInferDPPolicy(2, group[0]))
    assert scheduler.prefill_delayer is None
    assert not scheduler.cost_balance
    return scheduler


def schedule_group(group, schedulers):
    results = list(group[1].map(lambda scheduler: scheduler.schedule(), schedulers))
    assert len({s.dp_phase_policy.decode_step for s in schedulers}) == 1
    return results


def finish_prefill(scheduler, result):
    from atom.model_engine.scheduler import ScheduledBatchOutput

    batch, seqs = result
    # Partial chunks produce no sampled token; final chunks produce one.
    final = [seq.id for seq, last in zip(seqs.values(), batch.is_final_chunk) if last]
    scheduler.postprocess(
        list(seqs.values()),
        ScheduledBatchOutput(final, [(99,)] * len(final), None, None, None),
        batch=batch,
    )


def test_actual_schedulers_keep_idle_rank_in_any_vote_without_graphs(
    group, seq_factory
):
    schedulers = [make_scheduler(group) for _ in range(2)]
    schedulers[0].add(seq_factory([10, 11, 12, 13]))
    for scheduler in schedulers:
        scheduler.dp_phase_policy.decode_step = 100
    real, idle = schedule_group(group, schedulers)
    assert real[0].total_seqs_num_prefill == 1
    assert idle is None
    assert not schedulers[1].prefill_token_graphs
    assert schedulers[0].dp_phase_policy.decode_step == 0


@pytest.mark.parametrize("prefill_sizes", [(), (16, 32)])
@pytest.mark.parametrize("query_width", [1, 8])
def test_empty_rank_borrows_the_admitted_prefills_execution_phase(
    group, seq_factory, monkeypatch, prefill_sizes, query_width
):
    import numpy as np

    from atom.model_engine.scheduler import ScheduledBatch
    from atom.utils.forward_context import ForwardMode
    from atom.utils.region_graph import TokenGraph

    schedulers = [make_scheduler(group) for _ in range(2)]
    schedulers[0].add(seq_factory([10, 11, 12, 13]))
    for scheduler in schedulers:
        scheduler.dp_phase_policy.decode_step = 100
    real, idle = schedule_group(group, schedulers)
    assert idle is None
    peer = real[0]

    # The runner now schedules no fabricated request. Its DP metadata exchange
    # sees the actual prefill selected by the scheduler's second vote.
    def all_gather(outputs, local, group=None):
        outputs[0].copy_(local)
        outputs[1].copy_(local)
        outputs[1][0] = peer.total_tokens_num
        outputs[1][1] = peer.total_seqs_num
        outputs[1][2] = int(peer.has_prefill)

    monkeypatch.setattr(torch.distributed, "all_gather", all_gather)
    monkeypatch.setattr(
        TokenGraph,
        "captured_sizes",
        staticmethod(lambda phase=None: prefill_sizes if phase == "prefill" else ()),
    )
    batch = ScheduledBatch(
        seqs={},
        num_scheduled_tokens=[],
        total_tokens_num=0,
        num_spec_step=query_width - 1,
    )
    mode = ForwardMode.decide(
        batch=batch,
        dp_size=2,
        dp_group=object(),
        enforce_eager=False,
        step_needs_eager=False,
        capture_sizes=np.array([1, 2, 4, 8], dtype=np.int32),
        is_block_drafter=False,
        tbo_on=False,
        local_tbo=(False, False, 0, 0),
        max_seqlen_q=query_width,
        running_tokens_per_dp_rank=np.zeros(2, dtype=np.int32),
    )
    assert mode.is_prefill
    assert mode.scheduled_bs == mode.scheduled_tokens == 0
    assert mode.running_tokens == (prefill_sizes[0] if prefill_sizes else 1)
    assert mode.prefill_token_graph == bool(prefill_sizes)
    assert len(batch.req_ids) == 0


def test_partial_prefill_in_running_participates_in_next_all_vote(group, seq_factory):
    schedulers = [make_scheduler(group) for _ in range(2)]
    for scheduler in schedulers:
        scheduler.add(seq_factory(list(range(12))))
    first = schedule_group(group, schedulers)
    for scheduler, result in zip(schedulers, first):
        assert result[0].total_tokens_num_prefill == 8
        finish_prefill(scheduler, result)
        assert not scheduler.waiting
        assert scheduler._partial_prefill_count == 1
    second = schedule_group(group, schedulers)
    for scheduler, result in zip(schedulers, second):
        assert result[0].total_tokens_num_prefill == 4
        assert scheduler.dp_phase_policy.decode_step == 0


def test_full_sequence_slots_use_second_vote_then_decode(group, seq_factory):
    schedulers = [make_scheduler(group, max_num_seqs=1) for _ in range(2)]
    for scheduler in schedulers:
        scheduler.add(seq_factory([10, 11, 12, 13]))
    for scheduler, result in zip(schedulers, schedule_group(group, schedulers)):
        finish_prefill(scheduler, result)
        scheduler.add(seq_factory([20, 21, 22, 23]))
    before = len(group[0].rounds)
    results = schedule_group(group, schedulers)
    assert len(group[0].rounds) == before + 2
    for scheduler, result in zip(schedulers, results):
        assert result[0].total_seqs_num_decode == 1
        assert scheduler.dp_phase_policy.decode_step == 1
        assert len(scheduler.waiting) == 1


def test_kv_allocation_failure_does_not_spin_at_step_zero(
    group, seq_factory, monkeypatch
):
    schedulers = [make_scheduler(group) for _ in range(2)]
    for scheduler in schedulers:
        scheduler.add(seq_factory([10, 11, 12, 13]))
        monkeypatch.setattr(scheduler.block_manager, "can_allocate", lambda *a, **k: -1)
    results = schedule_group(group, schedulers)
    for scheduler, result in zip(schedulers, results):
        assert result[0].total_seqs_num_prefill == 0
        assert scheduler.dp_phase_policy.decode_step == 1


@pytest.mark.parametrize("prefill_token_graphs", [False, True])
def test_non_prefilling_rank_pauses_its_decode_on_group_prefill(
    group, seq_factory, prefill_token_graphs
):
    schedulers = [make_scheduler(group) for _ in range(2)]
    for scheduler in schedulers:
        scheduler.prefill_token_graphs = prefill_token_graphs
        scheduler.add(seq_factory([10, 11, 12, 13]))
    for scheduler, result in zip(schedulers, schedule_group(group, schedulers)):
        finish_prefill(scheduler, result)
        scheduler.dp_phase_policy.decode_step = 100
    schedulers[0].add(seq_factory([20, 21, 22, 23]))
    results = schedule_group(group, schedulers)
    assert results[0][0].total_seqs_num_prefill == 1
    assert results[1] is None
    assert len(schedulers[1].running) == 1
    finish_prefill(schedulers[0], results[0])
    # The group decision belongs to this tick, not a latched dummy flag.
    following = schedule_group(group, schedulers)
    assert following[1][0].total_seqs_num_decode == 1
    assert schedulers[1].dp_phase_policy.decode_step == 1


def test_consecutive_prefill_is_allowed_while_decodes_are_running(group, seq_factory):
    schedulers = [make_scheduler(group) for _ in range(2)]
    for _ in range(3):
        for scheduler in schedulers:
            scheduler.add(seq_factory([10, 11, 12, 13]))
        for scheduler, result in zip(schedulers, schedule_group(group, schedulers)):
            assert result[0].total_seqs_num_prefill == 1
            assert result[0].total_seqs_num_decode == 0
            finish_prefill(scheduler, result)
            assert scheduler.dp_phase_policy.decode_step == 0
    assert all(len(s.running) == 3 for s in schedulers)


def test_rejection_only_prefill_is_a_global_prompt_step(group, seq_factory):
    schedulers = [make_scheduler(group) for _ in range(2)]
    schedulers[0].add(seq_factory(list(range(80))))  # exceeds max_model_len=64
    for scheduler in schedulers:
        scheduler.dp_phase_policy.decode_step = 100
    results = schedule_group(group, schedulers)
    assert len(schedulers[0].take_rejected()) == 1
    assert results == [None, None]
    assert all(s.dp_phase_policy.decode_step == 0 for s in schedulers)


def test_native_delayer_cannot_be_combined_with_deepinfer(group):
    from atom.model_engine.prefill_delayer import PrefillDelayer

    scheduler = make_scheduler(group)
    with pytest.raises(ValueError, match="exclusive"):
        scheduler.set_prefill_delayer(PrefillDelayer(1, None, 8))


@pytest.mark.parametrize("admit_policy", ["interval", "adaptive"])
def test_engine_selects_deepinfer_independently_of_native_delayer_switch(
    monkeypatch, admit_policy
):
    pytest.importorskip(
        "aiter", reason="EngineCore imports AITER's worker message queue"
    )
    from types import SimpleNamespace

    from conftest import MockConfig, atom_config_double

    from atom.model_engine.engine_core import EngineCore
    from atom.model_engine.scheduler import Scheduler

    monkeypatch.setenv("ATOM_ENABLE_PREFILL_DELAYER", "0")
    monkeypatch.setenv("ATOM_PREFILL_DECODE_INTERVAL", "16")
    config = atom_config_double(
        dp_scheduler="deepinfer",
        prefill_admit_policy=admit_policy,
        enable_dp_attention=True,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        deepinfer_prefill_interval=7,
        deepinfer_prefill_any_interval=11,
    )
    core = EngineCore.__new__(EngineCore)
    core.scheduler = Scheduler(MockConfig())
    core._init_prefill_scheduler(config)
    policy = core.scheduler.dp_phase_policy
    assert isinstance(policy, DeepInferDPPolicy)
    assert (policy.prefill_interval, policy.prefill_any_interval) == (7, 11)
    assert core.scheduler.prefill_delayer is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"dp_scheduler": "typo"},
        {"enable_dp_attention": False},
        {"pipeline_parallel_size": 2},
        {"enable_rapidserve": True},
        {"disagg_is_decode": True},
        {"kv_transfer_config": {"kv_connector": "mooncake", "kv_role": "kv_producer"}},
        {"deepinfer_prefill_interval": 0},
        {"deepinfer_prefill_any_interval": -1},
    ],
)
def test_configuration_rejects_unsupported_modes(overrides):
    from conftest import atom_config_double

    from atom.config import Config

    config = atom_config_double(
        **{"dp_scheduler": "deepinfer", "enable_dp_attention": True, **overrides}
    )
    with pytest.raises(ValueError):
        Config._validate_dp_scheduler(config)


def test_cli_passes_policy_and_intervals_to_config():
    from atom.model_engine.arg_utils import EngineArgs

    parser = EngineArgs.add_cli_args(argparse.ArgumentParser())
    args = EngineArgs.from_cli_args(
        parser.parse_args(
            [
                "--enable-dp-attention",
                "--dp-scheduler",
                "deepinfer",
                "--deepinfer-prefill-interval",
                "7",
                "--deepinfer-prefill-any-interval",
                "11",
            ]
        )
    )
    kwargs = args._get_engine_kwargs()
    assert kwargs["dp_scheduler"] == "deepinfer"
    assert kwargs["deepinfer_prefill_interval"] == 7
    assert kwargs["deepinfer_prefill_any_interval"] == 11
    assert EngineArgs.from_cli_args(parser.parse_args([])).dp_scheduler == "atom"


@pytest.mark.parametrize("mask", [0, 1, 0x55, 0x7F, 0xFF])
@pytest.mark.parametrize("tick", [0, 70, 100, 700])
def test_eight_rank_votes_stay_in_phase(monkeypatch, mask, tick):
    collective = MinCollective(8)
    monkeypatch.setattr(torch.distributed, "all_reduce", collective.all_reduce)

    def run(rank):
        policy = DeepInferDPPolicy(8, collective)
        policy.decode_step = tick
        allow = policy.begin_step(has_prefill=bool(mask & (1 << rank)), kv_usage=0)
        selected = policy.finish_step(prefill_scheduled=allow)
        return selected, policy.decode_step

    expected = bool(mask) if tick in (100, 700) else mask == 0xFF
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(run, range(8)))
    assert results == [(expected, 0 if expected else tick + 1)] * 8
    assert len(collective.rounds) == (2 if expected else 1)


def _gloo_worker(rank, rendezvous):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        policy = DeepInferDPPolicy(2, dist.group.WORLD, 2, 3)
        for tick in range(4):
            allow = policy.begin_step(has_prefill=rank == 0, kv_usage=0)
            selected = policy.finish_step(prefill_scheduled=allow)
            assert selected == (tick == 3)
            assert policy.decode_step == (0 if selected else tick + 1)
        assert policy.begin_step(has_prefill=True, kv_usage=0)
        assert policy.finish_step(prefill_scheduled=True)
        assert policy.begin_step(has_prefill=True, kv_usage=0)
        assert not policy.finish_step(prefill_scheduled=False)
        assert policy.decode_step == 1
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_gloo_available(), reason="Gloo unavailable"
)
def test_real_two_process_gloo_votes(tmp_path):
    # A real transport check complements the deterministic threaded rank tests.
    torch.multiprocessing.spawn(
        _gloo_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True
    )

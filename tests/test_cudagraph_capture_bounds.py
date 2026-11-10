# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CUDAGraph decode buckets must stay inside what the scheduler can hand them.

`capture_cudagraph` used to filter capture sizes by `max_num_seqs` alone. That
is only one of the two bounds `Scheduler.schedule_decode` applies, and under
speculation it is not the one that binds: `mtp_k=3` turns 256 sequences into
1024 tokens, past a 512-token budget. Capturing there wrote past the per-token
forward buffers (sized `max_num_batched_tokens`) and surfaced as a bare
`could not broadcast input array from shape (1024,) into shape (512,)`.
"""

from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_without_aiter


@pytest.fixture(scope="module")
def load():
    with atom_without_aiter() as load:
        yield load


@pytest.fixture(scope="module")
def model_runner(load):
    return load("atom.model_engine.model_runner")


@pytest.fixture
def max_schedulable_decode_bs(model_runner):
    return model_runner.max_schedulable_decode_bs


class TestMaxSchedulableDecodeBs:
    def test_sequence_count_binds_when_the_token_budget_is_ample(
        self, max_schedulable_decode_bs
    ):
        # Default serving shape: 16384 // 4 = 4096 sequences' worth of budget,
        # far past max_num_seqs, so the sequence cap is what's left.
        assert max_schedulable_decode_bs(256, 16384, 4) == 256

    def test_token_budget_binds_under_speculation(self, max_schedulable_decode_bs):
        # The regression: 256 sequences * (mtp_k=3 + 1) = 1024 > 512.
        assert max_schedulable_decode_bs(256, 512, 4) == 128

    def test_no_speculation_charges_one_token_per_sequence(
        self, max_schedulable_decode_bs
    ):
        assert max_schedulable_decode_bs(256, 512, 1) == 256

    @pytest.mark.parametrize("full_q_len", [1, 2, 4, 8])
    def test_bound_is_never_exceeded_by_the_bucket_it_admits(
        self, max_schedulable_decode_bs, full_q_len
    ):
        # The property the capture loop relies on: every admitted bs, at the
        # full speculative width, fits the token budget. Smaller q buckets fit
        # a fortiori, which is why one filter covers them all.
        budget = 512
        bs = max_schedulable_decode_bs(1024, budget, full_q_len)
        assert bs * full_q_len <= budget

    def test_budget_below_one_sequence_yields_zero_not_a_partial_bucket(
        self, max_schedulable_decode_bs
    ):
        # Caller turns this into an actionable assert rather than capturing a
        # bucket it cannot fill.
        assert max_schedulable_decode_bs(256, 2, 4) == 0

    def test_matches_the_scheduler_admission_loop(self, max_schedulable_decode_bs):
        # Mirrors `schedule_decode`: charge `tokens_per_decode_seq` per seq and
        # stop at either bound. Pins the two to each other rather than to a
        # comment — if the scheduler ever charges the replayed q instead of the
        # full width, this is what should fail.
        for max_num_seqs in (1, 7, 128, 256):
            for budget in (1, 4, 500, 512, 16384):
                for full_q_len in (1, 4, 8):
                    admitted, tokens = 0, 0
                    while admitted < max_num_seqs:
                        if tokens + full_q_len > budget:
                            break
                        tokens += full_q_len
                        admitted += 1
                    assert (
                        max_schedulable_decode_bs(max_num_seqs, budget, full_q_len)
                        == admitted
                    )


@pytest.mark.parametrize("sp", [2, 4, 8])
@pytest.mark.parametrize("max_bs", [1, 2, 6, 12, 250, 512])
def test_split_rows_graph_ladder_covers_every_schedulable_batch(
    model_runner, sp, max_bs
):
    # Rounding to SP multiples used to run past `max_bs` and get filtered out,
    # leaving the top batch sizes eager (or no ladder at all).
    ladder = model_runner.split_rows_graph_ladder(
        list(range(1, 9)) + list(range(16, 513, 16)), sp, max_bs
    )
    assert max(ladder) == max_bs
    assert len({-(-r // sp) for r in ladder}) == len(ladder)


def test_sp_token_graphs_are_bucketed_by_step_rows(monkeypatch, load, model_runner):
    # Under SP the recording unit runs this rank's shard of the step; its bucket
    # is the step's rows, which capture dedups on and a decode step looks up.
    region_graph = load("atom.utils.region_graph")
    forward_context = load("atom.utils.forward_context")
    sp_wrapper = load("atom.utils.sp_wrapper")
    ulysses_sp = load("atom.distributed.ulysses_sp")
    monkeypatch.setattr(ulysses_sp, "get_sp_world_size", lambda: 4)
    monkeypatch.setattr(
        load("aiter.dist.parallel_state"),
        "get_pcp_group",
        lambda: SimpleNamespace(rank_in_group=3),
    )
    state = SimpleNamespace(tr_graph=None, context=None)
    monkeypatch.setattr(forward_context, "get_forward_context", lambda: state)
    monkeypatch.setattr(sp_wrapper, "get_forward_context", lambda: state)
    monkeypatch.setattr(sp_wrapper, "sp_gather_tokens", lambda x, rows: x)
    monkeypatch.setattr(region_graph, "graph_pool", lambda: None)
    monkeypatch.setattr(region_graph, "drain_nccl_works", lambda: None)

    def record(forward, *args, pool=None, **kwargs):
        output = forward(*args, **kwargs)
        return SimpleNamespace(replay=output.clone), output

    monkeypatch.setattr(region_graph.TokenGraph, "_record", staticmethod(record))

    @region_graph.token_graph
    class Shard(torch.nn.Module):
        runs = 0

        def forward(self, ids, positions):
            Shard.runs += 1
            return torch.full((ids.shape[0], 1), positions.shape[0])

    model = sp_wrapper.SPModelWrapper(Shard())
    ids = torch.zeros(64, dtype=torch.int64)

    def step(rows, mode):
        state.tr_graph = mode
        state.context = forward_context.Context(positions=ids[:rows])
        return model(ids[:rows], ids[:rows])

    def recorded(rows):
        return forward_context.ForwardMode._running_tokens(
            is_prefill=False,
            unified=True,
            running_bs=rows,
            scheduled_tokens=rows,
            running_tokens_per_dp_rank=None,
            packed_decode=False,
            max_seqlen_q=1,
        )

    ladder = model_runner.split_rows_graph_ladder([1, 2, 4, 8, 16, 32], 4, 30)
    for rows in ladder:
        step(rows, "record")
    assert region_graph.TokenGraph.captured_sizes("decode") == (4, 8, 16, 30)
    runs = Shard.runs
    for rows in ladder:
        assert recorded(rows) == (rows, True)
        assert step(rows, "replay").unique().tolist() == [rows]
    assert Shard.runs == runs
    # 29 rows shard like 30 but no graph was recorded at 29.
    assert recorded(29) == (29, False)
    model.model.__dict__["_token_graph"].release_graphs()


def test_memory_guard_skips_a_bucket_any_pcp_rank_cannot_fit(monkeypatch, model_runner):
    # A rank with ample free memory must still skip the bucket a peer skips,
    # or the peers' captured collectives pair mismatched buckets.
    groups = {"get_dp_group": 1, "get_pcp_group": 4}
    for name, size in groups.items():
        group = SimpleNamespace(world_size=size, cpu_group=name)
        monkeypatch.setattr(model_runner, name, lambda group=group: group)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (40 << 30, 80 << 30))
    reduced = []

    def all_reduce(free, op, group):
        reduced.append(group)
        free.fill_(5 << 30)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    runner = SimpleNamespace(
        rank=0,
        config=SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=1),
            hf_config=SimpleNamespace(hidden_size=6144),
        ),
    )
    assert model_runner.ModelRunner._skip_token_bucket(runner, 1024)
    assert reduced == ["get_pcp_group"]

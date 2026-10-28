# SPDX-License-Identifier: MIT
import itertools
from types import SimpleNamespace as NS

import numpy as np
import pytest

from atom.config import CUDAGraphMode, DSparkConfig
from atom.spec_decode.dspark_dp import (
    DPVerifyPlanner,
    choose_dp_candidate,
    executable_graph,
    local_frontier,
)
from atom.spec_decode.dspark_runtime import validate_online_runtime


@pytest.mark.parametrize("seed", range(5))
def test_joint_policy_matches_exhaustive_prefix_search(seed):
    rng = np.random.default_rng(seed)
    qs = [1, 2, 3, 4]  # deliberately non-nested divisibility
    captured = [1, 2, 3, 4, 6, 8, 9, 12]
    candidates = [(q, g) for q in qs for g in captured if g % q == 0]
    survival = [rng.uniform(0.01, 0.99, (n, 3)).cumprod(1) for n in [1, 2]]
    floors = [rng.integers(1, 3, len(a)) for a in survival]
    front, lengths = zip(
        *[
            local_frontier(a, lo, np.full(len(a), 4), candidates, qs)
            for a, lo in zip(survival, floors)
        ]
    )
    cost = lambda g: 0.3 + g * 0.4
    selected = choose_dp_candidate(np.array(front), captured, 8, 4, cost)
    best = 0.0
    for values in itertools.product(*[range(int(lo), 5) for f in floors for lo in f]):
        local = [values[:1], values[1:]]
        tau = sum(
            len(a) + sum(a[i, : length - 1].sum() for i, length in enumerate(ls))
            for a, ls in zip(survival, local)
        )
        for q in qs:
            if q < max(values):
                continue
            graph = executable_graph(max(map(sum, local)), q, captured)
            if graph is not None:
                best = max(best, tau / cost(graph))
    if selected is None:
        score = sum(len(a) + a.sum() for a in survival) / cost(8)
    else:
        q = candidates[selected][0]
        chosen = [x[selected] for x in lengths]
        graph = executable_graph(max(int(x.sum()) for x in chosen), q, captured)
        score = sum(x[selected, 0] for x in front) / cost(graph)
        for lo, ls in zip(floors, chosen):
            assert (ls >= lo).all() and (ls <= q).all()
    assert score == pytest.approx(best)


def test_dummy_consumes_graph_rows_but_contributes_no_tokens():
    qs, candidates = [1, 3, 6], [(1, 6), (3, 6), (6, 6), (6, 12)]
    frontier, _ = local_frontier(np.ones((1, 5)), [6], [6], candidates, qs, dummy=True)
    assert frontier[:, 0].tolist() == [0, 0, 0, 0]
    assert frontier[:, 3].tolist() == [0, 0, 1, 1]
    assert frontier[2, 1:3].tolist() == [6, 6]


def test_busy_peer_and_bonus_floors_can_erase_local_graph_saving():
    captured = [6, 12, 24]
    # Local work shrinks, but the peer forces the same 24-row graph.
    front = np.array([[[3, 6, 6, 1]], [[4, 18, 6, 1]]], dtype=float)
    assert choose_dp_candidate(front, captured, 24, 6, lambda g: g) is None
    front[1, 0, 1] = 12
    assert choose_dp_candidate(front, captured, 24, 6, lambda g: g) == 0
    # With a fallback graph the plan only fills it: same step cost, more
    # expected output; a different graph is never chosen.
    two = np.array(
        [[[3, 6, 6, 1], [5, 12, 6, 1]], [[4, 12, 6, 1], [4, 12, 6, 1]]], dtype=float
    )
    assert choose_dp_candidate(two, captured, 24, 6, lambda g: g, 12) == 1
    assert choose_dp_candidate(two, captured, 24, 6, lambda g: g, 6) is None
    front[1, 0, 3] = 0
    assert choose_dp_candidate(front, captured, 24, 6, lambda g: g) is None


def test_uncovered_shapes_do_not_borrow_smaller_graph_cost():
    front = np.array([[[3, 20, 6, 1]], [[4, 12, 6, 1]]], dtype=float)
    assert choose_dp_candidate(front, [6, 12], 24, 6, lambda g: g) is None


def test_delayed_confidence_is_exact_epoch_and_exact_request_set():
    import torch

    class Event:
        calls = 0

        def synchronize(self):
            self.calls += 1

    planner = DPVerifyPlanner.__new__(DPVerifyPlanner)
    planner.epoch = 12
    event = Event()
    planner.pending = [
        (9, event, torch.zeros((1, 3)), (8,), None),
        (10, event, torch.ones((1, 3)), (9,), None),
        (11, event, torch.ones((1, 3)) * 2, (9,), None),
    ]
    assert planner._delayed((8,)) is None
    assert event.calls == 0
    assert planner._delayed((9,)).tolist() == [[1, 1, 1]]
    assert event.calls == 1
    planner.invalidate()
    assert planner._delayed((9,)) is None
    assert len(planner.pending) == 3  # pending DMA still owns its buffers


def test_confidence_copies_keep_sources_and_do_not_reuse_busy_destinations(monkeypatch):
    from contextlib import nullcontext

    import torch

    class Event:
        ready = False

        def query(self):
            return self.ready

        def record(self, _):
            pass

    empty = torch.empty
    monkeypatch.setattr(
        torch,
        "empty",
        lambda *a, **kw: empty(
            *a, **{k: v for k, v in kw.items() if k != "pin_memory"}
        ),
    )
    stream = NS(wait_stream=lambda _: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", Event)
    p = DPVerifyPlanner.__new__(DPVerifyPlanner)
    p.runner = NS(
        config=NS(max_num_seqs=2), tokenID_processor=NS(async_copy_stream=stream)
    )
    p.gamma, p.pending, p.free, p.skipped_busy = 3, [], [], 0
    for epoch in range(1, 7):
        p.epoch, p.survival = epoch, torch.full((1, 3), epoch, dtype=torch.float32)
        p.record([1])
    assert len(p.pending) == 4 and p.skipped_busy == 2
    assert len({x[2].data_ptr() for x in p.pending}) == 4
    assert [int(x[4][0, 0]) for x in p.pending] == [1, 2, 3, 4]
    for x in p.pending:
        x[1].ready = True
    p.epoch, p.survival = 8, torch.ones((1, 3))
    p.record([1])
    assert len(p.pending) == 1 and len(p.free) == 3
    # Warmup drafts two requests even at max_num_seqs=1.
    p.runner.config.max_num_seqs, p.free = 1, []
    p.epoch, p.survival = 20, torch.ones((2, 3))
    p.record([1, 2])
    assert p.pending[-1][3] == (1, 2)


@pytest.mark.parametrize("ep", [False, True])
def test_online_admits_dp_and_ep_with_supported_execution(ep):
    config = NS(
        dspark=DSparkConfig(),
        parallel_config=NS(data_parallel_size=2),
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        enable_expert_parallel=ep,
        enable_tbo=False,
        enforce_eager=False,
        compilation_config=NS(cudagraph_mode=CUDAGraphMode.TR_HYBRID),
    )
    config.dspark.online_sps = True
    validate_online_runtime(config)
    config.enable_tbo = True
    with pytest.raises(ValueError, match="TBO disabled"):
        validate_online_runtime(config)


def test_observe_only_keeps_the_joint_static_budget(tmp_path, monkeypatch):
    import json

    import torch

    from atom.spec_decode.online_sps import OnlineSPS
    from atom.utils.region_graph import TokenGraph

    baseline = [1000 / (1 + 0.5 * i) for i in range(7)]
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"sps_table": baseline}))
    monkeypatch.setattr(
        TokenGraph, "captured_sizes", classmethod(lambda cls, phase: (1, 3, 6))
    )
    model = OnlineSPS(baseline, 1)
    for step in range(1, 11):
        model.observe(1, 6, 4.0, 20.0, step=step)
    model.publish()
    selected = []
    for observe_only in [True, False]:
        cfg = NS(
            dspark=DSparkConfig(
                online_sps=True,
                confidence_schedule=True,
                ragged=True,
                ragged_graph_sizes="1,3,6",
                calibration_profile=str(path),
                online_sps_observe_only=observe_only,
            ),
            speculative_config=NS(num_speculative_tokens=5),
            max_num_seqs=1,
        )
        runner = NS(config=cfg, tokenID_processor=NS(prev_batch=NS(req_ids=(7,))))
        scheduler = NS(runner=runner, runtime=NS(model=model), sts_temperatures=None)
        mesh = NS(
            tp_rank=0,
            dp_size=2,
            agree=lambda _: None,
            broadcast_tp=lambda _: None,
            gather_dp=lambda x: torch.stack([x, x]),
        )
        planner = DPVerifyPlanner(scheduler, mesh=mesh)
        planner.bind(baseline)
        planner.epoch = 2
        survival = torch.tensor([[0.35**i for i in range(1, 6)]])
        planner.pending = [(1, NS(synchronize=lambda: None), survival, (7,), survival)]
        wire = planner.prepare(
            NS(
                total_seqs_num=1,
                req_ids=(7,),
                total_tokens_num_prefill=0,
                is_dummy_run=False,
                num_spec_step=5,
                num_scheduled_tokens=np.array([6]),
                num_bonus=np.array([0]),
            )
        )
        # Two identical ranks: the payload is what sync_dp_metadata gathers.
        plan = planner.resolve(np.stack([wire, wire]))
        selected.append(plan[2][7] if plan is not None else 5)
        if plan is not None:
            q, tokens, lengths = plan
            assert q == planner.selected_q
            assert tokens.tolist() == [lengths[7] + 1] * 2
    # Static: q=1 wins (1/1.5). Adding 20 ms shifts the optimum to
    # q=3 (1.4725/22.5), ahead of q=6 (~1.5356/24).
    assert selected == [0, 2]

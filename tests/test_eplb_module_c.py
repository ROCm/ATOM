# SPDX-License-Identifier: MIT
# Tests for atom/model_ops/eplb.py (Module-C rebalancing algorithms)

import pytest
from import_guard import skip_if_dependency_missing

torch = pytest.importorskip("torch")

try:
    from atom.model_ops.eplb import (
        _build_logical_to_physical_map,
        balanced_packing,
        rebalance_experts,
        replicate_experts,
        step_peak_rebalance,
    )
except ImportError as _e:  # aiter/triton absent under bare non-GPU pytest
    skip_if_dependency_missing(_e, "requires full atom import env")


def test_balanced_packing_equal_cardinality():
    weight = torch.tensor([[9, 8, 7, 6]], dtype=torch.int32)
    pack_idx, rank_in_pack = balanced_packing(weight, num_packs=2)
    assert pack_idx.shape == (1, 4)
    assert rank_in_pack.shape == (1, 4)

    counts = [(pack_idx[0] == p).sum().item() for p in range(2)]
    assert counts == [2, 2]
    assert sorted(rank_in_pack[0, pack_idx[0] == 0].tolist()) == [0, 1]
    assert sorted(rank_in_pack[0, pack_idx[0] == 1].tolist()) == [0, 1]


def test_replicate_experts_conservation_and_rank():
    weight = torch.tensor([[10, 1, 1]], dtype=torch.int32)
    phy2log, phyrank, logcnt = replicate_experts(weight, num_physical=5)
    assert phy2log.shape == (1, 5)
    assert phyrank.shape == (1, 5)
    assert logcnt.shape == (1, 3)
    assert int(logcnt.sum().item()) == 5

    for logical_id in range(3):
        mask = phy2log[0] == logical_id
        count = int(mask.sum().item())
        assert count == int(logcnt[0, logical_id].item())
        if count > 0:
            ranks = sorted(phyrank[0, mask].tolist())
            assert ranks == list(range(count))


def test_build_logical_to_physical_map_uses_phyrank():
    p2l = torch.tensor([[0, 0, 0, 1]], dtype=torch.int32)
    phyrank = torch.tensor([[2, 0, 1, 0]], dtype=torch.int32)
    logcnt = torch.tensor([[3, 1]], dtype=torch.int32)
    l2p = _build_logical_to_physical_map(p2l, phyrank, logcnt)
    assert l2p.shape == (1, 2, 3)
    assert l2p[0, 0].tolist() == [1, 2, 0]
    assert l2p[0, 1].tolist() == [3, -1, -1]


def test_build_logical_to_physical_map_rejects_rank_out_of_logical_range():
    p2l = torch.tensor([[0, 1, 1]], dtype=torch.int32)
    phyrank = torch.tensor([[0, 2, 0]], dtype=torch.int32)
    logcnt = torch.tensor([[1, 2]], dtype=torch.int32)
    with pytest.raises(
        AssertionError, match="physical rank out of logical expert range"
    ):
        _build_logical_to_physical_map(p2l, phyrank, logcnt)


def test_rebalance_experts_global_invariants():
    weight = torch.tensor([[8, 6, 2, 1], [1, 2, 6, 8]], dtype=torch.int32)
    p2l, l2p, logcnt = rebalance_experts(
        weight,
        num_physical=8,
        num_groups=1,
        num_nodes=1,
        num_gpus=4,
        enable_hierarchical=False,
    )
    assert p2l.shape == (2, 8)
    assert logcnt.shape == (2, 4)
    assert l2p.shape[0] == 2 and l2p.shape[1] == 4
    assert l2p.shape[2] == int(logcnt.max().item())

    for layer in range(2):
        for logical_id in range(4):
            expected = int(logcnt[layer, logical_id].item())
            physical_ids = l2p[layer, logical_id]
            valid = physical_ids[physical_ids >= 0]
            assert valid.numel() == expected
            assert torch.all(p2l[layer, valid.to(torch.int64)] == logical_id)


@pytest.mark.parametrize(
    "kwargs,err",
    [
        (
            {
                "num_physical": 8,
                "num_groups": 3,
                "num_nodes": 2,
                "num_gpus": 4,
                "enable_hierarchical": True,
            },
            "num_logical must be divisible by num_groups",
        ),
        (
            {
                "num_physical": 12,
                "num_groups": 4,
                "num_nodes": 3,
                "num_gpus": 6,
                "enable_hierarchical": True,
            },
            "num_groups must be divisible by num_nodes",
        ),
        (
            {
                "num_physical": 12,
                "num_groups": 4,
                "num_nodes": 2,
                "num_gpus": 3,
                "enable_hierarchical": True,
            },
            "num_gpus must be divisible by num_nodes",
        ),
        (
            {
                "num_physical": 10,
                "num_groups": 4,
                "num_nodes": 2,
                "num_gpus": 4,
                "enable_hierarchical": True,
            },
            "num_physical must be divisible by num_gpus",
        ),
    ],
)
def test_rebalance_experts_constraints(kwargs, err):
    weight = torch.ones((1, 8), dtype=torch.int32)
    with pytest.raises(AssertionError, match=err):
        rebalance_experts(weight, **kwargs)


def test_rebalance_experts_flat_accepts_dsr1_shared_expert_tail():
    # DSR1 has 256 routed experts in 8 router groups. EPLB also routes its
    # fused shared expert, yielding 257 logical experts. Router-group
    # divisibility is irrelevant to flat single-node placement.
    weight = torch.ones((1, 257), dtype=torch.int32)
    p2l, l2p, logcnt = rebalance_experts(
        weight,
        num_physical=328,
        num_groups=8,
        num_nodes=1,
        num_gpus=8,
        enable_hierarchical=False,
    )
    assert p2l.shape == (1, 328)
    assert l2p.shape[:2] == (1, 257)
    assert logcnt.shape == (1, 257)
    assert int(logcnt.sum().item()) == 328


def test_rebalance_experts_hierarchical_invariants():
    weight = torch.tensor([[100, 80, 1, 1, 60, 40, 1, 1]], dtype=torch.int32)
    p2l, l2p, logcnt = rebalance_experts(
        weight,
        num_physical=8,
        num_groups=4,
        num_nodes=2,
        num_gpus=4,
        enable_hierarchical=True,
    )
    assert p2l.shape == (1, 8)
    assert logcnt.shape == (1, 8)
    assert l2p.shape[0] == 1 and l2p.shape[1] == 8
    assert int(logcnt.sum().item()) == 8
    assert sorted(p2l[0].tolist()) == list(range(8))


def _step_peak_gpu_loads(step_load, p2l, num_gpus):
    """[S, num_gpus] per-step GPU load of one layer, replicas sharing evenly."""
    counts = torch.bincount(p2l, minlength=step_load.shape[-1]).to(torch.float64)
    per_replica = step_load[:, 0].to(torch.float64) / counts
    return per_replica[:, p2l].view(step_load.shape[0], num_gpus, -1).sum(-1)


@pytest.mark.parametrize("num_physical", [6, 8])
def test_step_peak_never_copies_idle_experts_on_a_tied_peak(num_physical):
    # After expert 2's first copy, experts 1 and 2 tie at the peak (10 per
    # replica): no single copy lowers it, so every candidate costs the same.
    # The copy must go to a loaded expert, not to idle expert 0.
    step_load = torch.tensor([[[0, 10, 20, 0]]])
    p2l = step_peak_rebalance(step_load, num_physical=num_physical, num_gpus=2)[0]
    counts = torch.bincount(p2l, minlength=4).tolist()
    assert counts[0] == 1 and counts[3] == 1
    assert _step_peak_gpu_loads(step_load, p2l, 2).tolist() == [[15.0, 15.0]]


def test_step_peak_packs_alternating_hotspots_together():
    # A and B are hot on alternate steps, C and D steady; the window sums are
    # all equal. Pairing A with B keeps every step at 20 per GPU, where any
    # pairing of a hot expert with a steady one peaks at 30.
    step_load = torch.tensor([[[20, 0, 10, 10]], [[0, 20, 10, 10]]])
    p2l = step_peak_rebalance(step_load, num_physical=4, num_gpus=2)[0]
    assert {frozenset(p2l[:2].tolist()), frozenset(p2l[2:].tolist())} == {
        frozenset({0, 1}),
        frozenset({2, 3}),
    }
    assert _step_peak_gpu_loads(step_load, p2l, 2).max().item() == 20.0


@pytest.mark.parametrize("num_gpus,num_physical", [(2, 8), (4, 12), (4, 16)])
def test_step_peak_invariants(num_gpus, num_physical):
    gen = torch.Generator().manual_seed(num_physical)
    step_load = torch.randint(0, 50, (6, 3, 8), generator=gen)
    p2l = step_peak_rebalance(step_load, num_physical=num_physical, num_gpus=num_gpus)
    assert tuple(p2l.shape) == (3, num_physical)
    for row in p2l:
        counts = torch.bincount(row, minlength=8)
        assert int(counts.min()) >= 1 and int(counts.sum()) == num_physical


def _steady_peak(p2l, weight, num_gpus):
    """Busiest GPU's load under a steady (every step equal) load, no replicas."""
    return int(weight[p2l.long()].view(num_gpus, -1).sum(-1).max())


def _steady_loads():
    """Steady loads on which a peak tie broken by GPU id lost to naive (found
    by search before the fix): 128 experts, 8 GPUs, no redundant experts."""
    cases = []
    for seed in (180, 360, 570):  # integer loads 0..20
        g = torch.Generator().manual_seed(seed)
        cases.append(torch.randint(0, 21, (1, 128), generator=g)[0])
    g = torch.Generator().manual_seed(67)  # sparse: 30% of experts carry load
    mask = torch.rand(1, 128, generator=g) < 0.3
    cases.append((mask.long() * torch.randint(1, 9, (1, 128), generator=g))[0])
    g = torch.Generator().manual_seed(3898)  # 384 tokens: a mean GPU load of 48
    popularity = torch.rand(128, generator=g) ** 6 + 0.001
    picks = torch.multinomial(popularity, 384, replacement=True, generator=g)
    cases.append(torch.bincount(picks, minlength=128))
    return cases


@pytest.mark.parametrize("case", range(5))
def test_step_peak_steady_load_is_not_worse_than_naive(case):
    # With every step equal the per-step peak is the window peak, which naive's
    # balanced packing already minimizes. A replica that raises no step's peak
    # ties on every GPU that can absorb it; breaking that tie by GPU id stacked
    # such replicas on GPU 0 and lost to naive by one token here.
    weight = _steady_loads()[case]
    step_load = weight.view(1, 1, -1).expand(4, 1, -1).contiguous()
    p2l = step_peak_rebalance(step_load, num_physical=128, num_gpus=8)[0]
    naive, _, _ = rebalance_experts(
        weight.view(1, -1),
        num_physical=128,
        num_groups=1,
        num_nodes=1,
        num_gpus=8,
        enable_hierarchical=False,
    )
    assert _steady_peak(p2l, weight, 8) <= _steady_peak(naive[0], weight, 8)


@pytest.mark.parametrize("num_logical", [128, 256])
@pytest.mark.parametrize("seed", range(6))
def test_step_peak_steady_random_loads_match_naive_peak(num_logical, seed):
    g = torch.Generator().manual_seed(seed)
    weight = torch.randint(0, 21, (num_logical,), generator=g)
    step_load = weight.view(1, 1, -1).expand(2, 1, -1).contiguous()
    p2l = step_peak_rebalance(step_load, num_physical=num_logical, num_gpus=8)[0]
    naive, _, _ = rebalance_experts(
        weight.view(1, -1),
        num_physical=num_logical,
        num_groups=1,
        num_nodes=1,
        num_gpus=8,
        enable_hierarchical=False,
    )
    assert _steady_peak(p2l, weight, 8) <= _steady_peak(naive[0], weight, 8)


def test_step_peak_spreads_the_copies_of_an_expert():
    # Uniform load, 256 logical / 320 physical / EP8: 64 experts get a second
    # copy. Every pair of copies ties everywhere on load, so only a
    # replica-aware choice keeps both copies off one GPU, where they would
    # split nothing.
    step_load = torch.full((4, 1, 256), 5, dtype=torch.int64)
    p2l = step_peak_rebalance(step_load, num_physical=320, num_gpus=8)[0]
    counts = torch.bincount(p2l, minlength=256)
    gpu_of = torch.arange(320) // 40
    replicated = (counts > 1).nonzero().flatten().tolist()
    assert len(replicated) == 64
    coverage = [len(set(gpu_of[p2l == e].tolist())) for e in replicated]
    assert coverage == [2] * 64


def test_placement_from_p2l_matches_per_layer_rebuild():
    from atom.model_ops.eplb import _placement_from_p2l, _rebuild_placement_from_p2l

    g = torch.Generator().manual_seed(0)
    num_logical, num_physical = 16, 24
    rows = []
    for _ in range(5):  # every logical once, then 8 random copies, shuffled
        extra = torch.randint(
            0, num_logical, (num_physical - num_logical,), generator=g
        )
        row = torch.cat((torch.arange(num_logical), extra))
        rows.append(row[torch.randperm(num_physical, generator=g)])
    p2l = torch.stack(rows)
    got_p2l, got_l2p, got_cnt = _placement_from_p2l(p2l, num_logical)
    rebuilt = [_rebuild_placement_from_p2l(row, num_logical) for row in p2l]
    want_p2l = torch.stack([r[0] for r in rebuilt]).to(torch.int32)
    want_rank = torch.stack([r[1] for r in rebuilt])
    want_cnt = torch.stack([r[2] for r in rebuilt])
    assert torch.equal(got_p2l, want_p2l)
    assert torch.equal(got_cnt, want_cnt)
    assert torch.equal(
        got_l2p, _build_logical_to_physical_map(want_p2l, want_rank, want_cnt)
    )

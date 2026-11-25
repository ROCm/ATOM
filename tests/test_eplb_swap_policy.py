# SPDX-License-Identifier: MIT
# Tests for the EPLB "swap" placement policy
# (atom/model_ops/eplb_gate_permute.py: swap_rebalance / greedy_swaps).

import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from import_guard import skip_if_dependency_missing

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

try:
    import atom.config  # noqa: F401
    from atom.model_ops import eplb
    from atom.model_ops import eplb_gate_permute as gp
except ImportError as _e:  # aiter/triton absent under bare non-GPU pytest
    skip_if_dependency_missing(_e, "requires full atom import env")

COUNT1 = Path("/mnt/m2m_nobackup/zhimding/mega_dep8_work/runs/routes/count1")


def reference_greedy(values, rk, num_ranks, budget, eps, w=gp.SWAP_WEIGHTS, lam=1.0):
    """The policy written as plain loops over (heavy rank, other rank)."""
    w = np.asarray(w, float)
    loads = gp.rank_loads(values, rk, num_ranks)
    swaps = []
    while budget is None or len(swaps) < budget:
        best = (eps, None)
        top = loads.max(0)
        heavy = sorted({int(loads[:, m].argmax()) for m in range(len(w)) if w[m] > 0})
        for h in heavy:
            eh = np.flatnonzero(rk == h)
            for low in range(num_ranks):
                if low == h:
                    continue
                el = np.flatnonzero(rk == low)
                d = values[el][None, :, :] - values[eh][:, None, :]
                rest = np.delete(loads, [h, low], axis=0).max(0)
                gain = top - np.maximum(np.maximum(loads[h] + d, loads[low] - d), rest)
                gain = gain + lam * 2 * d * (loads[low] - loads[h] - d)
                gain = (gain * w).sum(-1)
                i, j = np.unravel_index(gain.argmax(), gain.shape)
                if gain[i, j] > best[0]:
                    best = (gain[i, j], (h, low, eh[i], el[j]))
        if best[1] is None:
            break
        h, low, a, b = best[1]
        loads[h] += values[b] - values[a]
        loads[low] += values[a] - values[b]
        rk[a], rk[b] = low, h
        swaps.append((int(a), int(b)))
    return swaps


def smoothed_objective(values, rk, num_ranks, w=gp.SWAP_WEIGHTS, lam=1.0):
    loads = gp.rank_loads(values, rk, num_ranks)
    return float(((loads.max(0) + lam * (loads**2).sum(0)) * np.asarray(w)).sum())


def skewed_loads(layers, experts, seed, ties=False):
    """Prefill and decode pick counts with a few hot experts a layer."""
    rng = np.random.default_rng(seed)
    pf = rng.lognormal(0.0, 1.2, size=(layers, experts)) * 1e4
    dec = pf * rng.lognormal(0.0, 0.5, size=(layers, experts)) * 0.05
    if ties:
        pf, dec = np.round(pf / 4000), np.round(dec / 200)
    return (
        torch.from_numpy(np.round(pf).astype(np.int64)),
        torch.from_numpy(np.round(dec).astype(np.int64)),
    )


def identity_p2l(layers, experts):
    return torch.arange(experts, dtype=torch.int32).repeat(layers, 1)


def run_policy(pf, dec, p2l, *, budget=16, eps=0.005, rows=144.0, ranks=8, **kw):
    return gp.swap_rebalance(
        pf, dec, rows, p2l, num_gpus=ranks, top_k=6, budget=budget, eps=eps, **kw
    )


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("experts,ranks", [(64, 8), (48, 4), (384, 8)])
def test_greedy_matches_the_loop_form(experts, ranks, ties):
    pf, dec = skewed_loads(1, experts, seed=experts + ranks + ties, ties=ties)
    values = gp.swap_terms(pf[0].numpy(), dec[0].numpy(), 144.0, 6, ranks)
    start = np.arange(experts) // (experts // ranks)
    for budget, eps in ((4, 0.0), (16, 0.005), (None, 0.02)):
        mine, ref = start.copy(), start.copy()
        got = gp.greedy_swaps(values, mine, ranks, budget=budget, eps=eps)
        want = reference_greedy(values, ref, ranks, budget, eps)
        assert got == want
        assert np.array_equal(mine, ref)


def test_every_swap_lowers_the_objective_by_more_than_eps():
    pf, dec = skewed_loads(1, 384, seed=3)
    values = gp.swap_terms(pf[0].numpy(), dec[0].numpy(), 144.0, 6, 8)
    rk = np.arange(384) // 48
    replay = rk.copy()
    eps = 0.005
    swaps = gp.greedy_swaps(values, rk, 8, budget=None, eps=eps)
    assert swaps
    for a, b in swaps:
        before = smoothed_objective(values, replay, 8)
        replay[a], replay[b] = replay[b], replay[a]
        assert before - smoothed_objective(values, replay, 8) > eps * (1 - 1e-9)
    assert np.array_equal(replay, rk)
    # every rank keeps its share of the experts
    assert np.bincount(rk, minlength=8).tolist() == [48] * 8
    # and nothing more is worth eps
    assert gp.greedy_swaps(values, rk, 8, budget=None, eps=eps) == []


def test_budget_bounds_the_swaps_of_a_layer():
    pf, dec = skewed_loads(3, 384, seed=5)
    p2l = identity_p2l(3, 384)
    for budget in (0, 1, 5, 16):
        new, _l2p, _logcnt, swaps = run_policy(pf, dec, p2l, budget=budget)
        assert all(n <= budget for n in swaps.values())
        moved = (new != p2l).sum(dim=1)
        assert moved.tolist() == [2 * swaps[i] for i in range(3)]
        if budget == 0:
            assert torch.equal(new, p2l)
    unbounded = run_policy(pf, dec, p2l, budget=None, eps=0.0005)[3]
    assert max(unbounded.values()) > 16


def test_eps_is_the_gain_a_swap_needs():
    pf, dec = skewed_loads(2, 384, seed=7)
    p2l = identity_p2l(2, 384)
    counts = [
        sum(run_policy(pf, dec, p2l, budget=None, eps=eps)[3].values())
        for eps in (0.0, 0.005, 0.05, 1e9)
    ]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] > counts[2] > 0
    assert counts[-1] == 0


def test_result_is_what_rebalance_experts_returns():
    pf, dec = skewed_loads(4, 384, seed=11)
    p2l = identity_p2l(4, 384)
    new, l2p, logcnt, _ = run_policy(pf, dec, p2l)
    assert new.shape == (4, 384) and new.dtype == torch.int32
    assert l2p.shape == (4, 384, 1) and l2p.dtype == torch.int32
    assert logcnt.shape == (4, 384) and bool((logcnt == 1).all())
    slots = torch.arange(384, dtype=torch.int32)
    for layer in range(4):
        assert torch.equal(new[layer].sort().values, slots)
        assert torch.equal(new[layer][l2p[layer, :, 0].long()], slots)
    # the existing metadata assembles from it and commits over the live one
    live = eplb.ExpertLocationMetadata.from_trivial(
        num_layers=4, num_logical_experts=384, ep_size=8, ep_rank=3
    )
    meta = eplb.ExpertLocationMetadata.from_rebalance_result(
        physical_to_logical_map=new,
        logical_to_physical_map=l2p,
        logical_replica_count=logcnt,
        ep_size=8,
        ep_rank=3,
        max_num_replicas=1,
    )
    live.update(meta, [0, 1, 2, 3])
    assert torch.equal(live.physical_to_logical_map, new)


def test_layer_without_a_swap_plans_no_migration():
    # Layer 0 is balanced to the last pick, layer 1 has nothing counted, layer
    # 2 is skewed: only layer 2 may move.
    pf, dec = skewed_loads(3, 384, seed=13)
    pf[0], dec[0] = 1000, 50
    pf[1], dec[1] = 0, 0
    p2l = identity_p2l(3, 384)
    new, _, _, swaps = run_policy(pf, dec, p2l)
    assert swaps[0] == 0 and swaps[1] == 0 and swaps[2] > 0
    for layer in (0, 1):
        assert torch.equal(new[layer], p2l[layer])
        for rank in range(8):
            plans = eplb._plan_single_layer_migration(
                old_p2l_layer=p2l[layer],
                new_p2l_layer=new[layer],
                num_local_physical_experts=48,
                num_gpu_per_node=8,
                rank=rank,
                world_size=8,
            )
            assert all(len(part) == 0 for part in plans)


def test_only_the_named_layers_are_placed():
    pf, dec = skewed_loads(4, 384, seed=17)
    p2l = identity_p2l(4, 384)
    new, _, _, swaps = run_policy(pf, dec, p2l, layer_ids=[1, 3])
    assert sorted(swaps) == [1, 3]
    assert torch.equal(new[0], p2l[0]) and torch.equal(new[2], p2l[2])
    assert not torch.equal(new[1], p2l[1])


def test_swaps_start_from_the_live_placement():
    pf, dec = skewed_loads(1, 384, seed=19)
    first = run_policy(pf, dec, identity_p2l(1, 384), budget=8)[0]
    second, _, _, swaps = run_policy(pf, dec, first, budget=8)
    # the second round continues where the first stopped: at most 8 more swaps
    assert (second != first).sum().item() == 2 * swaps[0] <= 16
    both = run_policy(pf, dec, identity_p2l(1, 384), budget=16)[0]
    assert torch.equal(second, both)


def test_prefill_only_load_still_places():
    pf, _ = skewed_loads(1, 384, seed=23)
    new, _, _, swaps = run_policy(pf, None, identity_p2l(1, 384), rows=0.0)
    assert swaps[0] > 0
    rk_new = gp.ranks_of(new.numpy().astype(np.int64), 48)[0]
    rk_old = np.arange(384) // 48
    load = pf[0].numpy().astype(float)
    assert gp.placement_metrics(load, load * 0, 0.0, rk_new, 8, 6)[0] < (
        gp.placement_metrics(load, load * 0, 0.0, rk_old, 8, 6)[0]
    )


def _digest(seed):
    pf, dec = skewed_loads(6, 384, seed=seed, ties=True)
    new, l2p, _, swaps = run_policy(pf, dec, identity_p2l(6, 384))
    blob = new.numpy().tobytes() + l2p.numpy().tobytes() + repr(swaps).encode()
    return hashlib.sha256(blob).hexdigest()


def test_same_plan_in_every_process():
    here = _digest(29)
    assert here == _digest(29)
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "import test_eplb_swap_policy as t; print(t._digest(29))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, str(Path(__file__).parent)],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "OMP_NUM_THREADS": "7"},
    )
    assert out.stdout.strip().splitlines()[-1] == here


def _count1():
    files = [COUNT1 / f"r{r}_counts.pt" for r in range(8)]
    if not all(f.is_file() for f in files):
        pytest.skip(f"no count1 route counts under {COUNT1}")
    total = sum(torch.load(f)[384][:, :40, :384] for f in files)
    return total[1], total[0]  # prefill, decode: [40, 384] picks


def test_count1_load_is_balanced_from_the_identity():
    pf, dec = _count1()
    p2l = identity_p2l(40, 384)
    start = time.perf_counter()
    new, _, _, swaps = run_policy(pf, dec, p2l, budget=None, eps=0.005)
    one_shot_s = time.perf_counter() - start
    pfn, decn = pf.numpy().astype(float), dec.numpy().astype(float)
    before = gp.ranks_of(p2l.numpy().astype(np.int64), 48)
    after = gp.ranks_of(new.numpy().astype(np.int64), 48)

    def table(rk):
        rows = [
            gp.placement_metrics(pfn[i], decn[i], 144.0, rk[i], 8, 6) for i in range(40)
        ]
        return np.array(rows)

    was, now = table(before), table(after)
    # prefill rows of the decoder layers (39 runs one row a request)
    assert now[20:39, 0].mean() < 1.05 < was[20:39, 0].mean()
    assert now[:20, 0].mean() < was[:20, 0].mean()
    # the active-expert proxy's heaviest rank, and the decode rows
    assert now[:, 1].mean() < was[:, 1].mean()
    assert now[:, 3].mean() < was[:, 3].mean()
    # the mean over the ranks does not depend on the placement
    assert np.allclose(now[:, 2], was[:, 2])
    # a budgeted round is a prefix of the same greedy
    budgeted = run_policy(pf, dec, p2l, budget=16, eps=0.005)[3]
    assert all(budgeted[i] == min(16, swaps[i]) for i in range(40))
    print(
        f"\ncount1 one-shot: {sum(swaps.values())} swaps in {one_shot_s:.2f}s; "
        f"prefill max/mean L20-38 {was[20:39, 0].mean():.3f} -> {now[20:39, 0].mean():.3f}, "
        f"proxy max {was[:, 1].mean():.2f} -> {now[:, 1].mean():.2f}, "
        f"decode rows {was[:, 3].mean():.3f} -> {now[:, 3].mean():.3f}"
    )
    line = gp.metrics_line(pfn, decn, 144.0, after, list(range(40)), 8, 6)
    assert "L0-19" in line and "L20-38" in line


def test_count1_placement_depends_on_the_operating_point():
    """Different concurrency, different placement: solved from the identity
    for 64 and for 1344 decode rows a step of the group, each placement
    carries fewer active experts on its heaviest rank at its own operating
    point than the other does there."""
    pf, dec = _count1()
    p2l = identity_p2l(40, 384)
    pfn, decn = pf.numpy().astype(float), dec.numpy().astype(float)
    solved = {
        rows: gp.ranks_of(
            run_policy(pf, dec, p2l, budget=None, eps=0.005, rows=rows)[0]
            .numpy()
            .astype(np.int64),
            48,
        )
        for rows in (64.0, 1344.0)
    }

    def proxy_max(rk, rows):
        return np.mean(
            [
                gp.placement_metrics(pfn[i], decn[i], rows, rk[i], 8, 6)[1]
                for i in range(40)
            ]
        )

    low, high = solved[64.0], solved[1344.0]
    assert (low != high).any()
    assert proxy_max(low, 64.0) < proxy_max(high, 64.0)
    assert proxy_max(high, 1344.0) < proxy_max(low, 1344.0)
    # and both beat the identity at either point
    identity = gp.ranks_of(p2l.numpy().astype(np.int64), 48)
    for rows in (64.0, 1344.0):
        assert max(proxy_max(low, rows), proxy_max(high, rows)) < proxy_max(
            identity, rows
        )

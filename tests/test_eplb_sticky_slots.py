# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Slot stability: `_keep_slots_stable`.

Placement decides WHICH experts a rank owns; the slot they land in inside that
rank is arbitrary. Rebuilding placement from scratch every rebalance therefore
shuffles experts between slots of the same rank for no reason, and the migration
planner turns each such shuffle into a local weight copy. `_keep_slots_stable`
permutes each rank's slots so those experts stay put.

CPU-only; no GPU or distributed setup.
"""

from collections import Counter

import torch

from atom.model_ops.eplb import (
    _keep_slots_stable,
    _placement_biased,
    rebalance_experts,
)


def _ref(p2l, phyrank, old_p2l, num_gpus):
    """Straightforward reference: per rank, greedily hand each old slot back the
    expert it held. Only used to pin the optimum the vectorized version targets.
    """
    num_layers, num_physical = p2l.shape
    n = num_physical // num_gpus
    old, new, rk = old_p2l.tolist(), p2l.tolist(), phyrank.tolist()
    out_p2l = [row[:] for row in new]
    out_rk = [row[:] for row in rk]
    for layer in range(num_layers):
        for g in range(num_gpus):
            base = g * n
            by_expert = {}
            for j in range(n):
                by_expert.setdefault(new[layer][base + j], []).append(j)
            perm = [None] * n
            taken = [False] * n
            for i in range(n):
                slots = by_expert.get(old[layer][base + i])
                if slots:
                    j = slots.pop()
                    perm[i], taken[j] = j, True
            free = [j for j in range(n) if not taken[j]]
            k = 0
            for i in range(n):
                if perm[i] is None:
                    perm[i], k = free[k], k + 1
            for i in range(n):
                out_p2l[layer][base + i] = new[layer][base + perm[i]]
                out_rk[layer][base + i] = rk[layer][base + perm[i]]
    return (
        torch.tensor(out_p2l, dtype=p2l.dtype),
        torch.tensor(out_rk, dtype=phyrank.dtype),
    )


def _upper_bound(old_p2l, p2l, num_gpus):
    """Most fixed points any within-rank permutation can reach."""
    n = p2l.shape[1] // num_gpus
    total = 0
    for layer in range(p2l.shape[0]):
        for g in range(num_gpus):
            lo, hi = g * n, (g + 1) * n
            oc = Counter(old_p2l[layer][lo:hi].tolist())
            nc = Counter(p2l[layer][lo:hi].tolist())
            total += sum(min(oc[e], nc[e]) for e in nc)
    return total


def _placement(weight, *, num_physical, num_gpus):
    p2l, _, logcnt = rebalance_experts(
        weight,
        num_physical=num_physical,
        num_groups=1,
        num_nodes=1,
        num_gpus=num_gpus,
        enable_hierarchical=False,
    )
    return p2l, logcnt


def _two_placements(*, num_logical=64, num_physical=72, num_gpus=8, num_layers=4):
    g = torch.Generator().manual_seed(0)
    base = torch.rand((num_layers, num_logical), generator=g) + 0.1
    old, _ = _placement(base, num_physical=num_physical, num_gpus=num_gpus)
    jitter = 0.7 + 0.6 * torch.rand((num_layers, num_logical), generator=g)
    new, _ = _placement(base * jitter, num_physical=num_physical, num_gpus=num_gpus)
    phyrank = (
        torch.arange(num_physical, dtype=torch.int32)
        .unsqueeze(0)
        .expand(num_layers, -1)
        .contiguous()
    )
    return old, new, phyrank


def _per_rank_experts(t, num_gpus):
    n = t.shape[1] // num_gpus
    return [
        [Counter(t[layer][g * n : (g + 1) * n].tolist()) for g in range(num_gpus)]
        for layer in range(t.shape[0])
    ]


class TestKeepSlotsStable:
    def test_rank_ownership_is_untouched(self):
        """The permutation must not move an expert to another rank -- that would
        silently undo the placement policy's balancing decision."""
        old, new, phyrank = _two_placements()
        out, _ = _keep_slots_stable(new, phyrank, old, 8)
        assert _per_rank_experts(out, 8) == _per_rank_experts(new, 8)

    def test_result_is_a_permutation_within_each_rank(self):
        old, new, phyrank = _two_placements()
        out, out_rank = _keep_slots_stable(new, phyrank, old, 8)
        n = new.shape[1] // 8
        for layer in range(new.shape[0]):
            for g in range(8):
                lo, hi = g * n, (g + 1) * n
                before = Counter(
                    zip(new[layer][lo:hi].tolist(), phyrank[layer][lo:hi].tolist())
                )
                after = Counter(
                    zip(out[layer][lo:hi].tolist(), out_rank[layer][lo:hi].tolist())
                )
                assert before == after, "phyrank must travel with its expert"

    def test_increases_fixed_points_towards_the_optimum(self):
        old, new, phyrank = _two_placements()
        out, _ = _keep_slots_stable(new, phyrank, old, 8)
        before = int((old == new).sum())
        after = int((old == out).sum())
        bound = _upper_bound(old, new, 8)
        assert after > before, f"no slots saved ({before} -> {after})"
        assert after == bound, f"{after} short of the reachable {bound}"

    def test_matches_the_reference(self):
        old, new, phyrank = _two_placements()
        out, _ = _keep_slots_stable(new, phyrank, old, 8)
        ref, _ = _ref(new, phyrank, old, 8)
        assert int((old == out).sum()) == int((old == ref).sum())

    def test_is_deterministic(self):
        """Every rank runs this independently and their migration plans have to
        pair up, so the same maps must always give the same permutation."""
        old, new, phyrank = _two_placements()
        first, first_rank = _keep_slots_stable(new, phyrank, old, 8)
        for _ in range(20):
            again, again_rank = _keep_slots_stable(new, phyrank, old, 8)
            assert torch.equal(again, first)
            assert torch.equal(again_rank, first_rank)

    def test_identical_placement_is_a_noop(self):
        _, new, phyrank = _two_placements()
        out, out_rank = _keep_slots_stable(new, phyrank, new, 8)
        assert torch.equal(out, new)
        assert torch.equal(out_rank, phyrank)

    def test_no_old_placement_passes_through(self):
        """First rebalance after startup has nothing to stay stable against."""
        _, new, phyrank = _two_placements()
        out, out_rank = _keep_slots_stable(new, phyrank, None, 8)
        assert out is new and out_rank is phyrank

    def test_shape_mismatch_passes_through(self):
        """num_redundant changed between rebalances -> old map is unusable."""
        _, new, phyrank = _two_placements()
        stale = new[:, : new.shape[1] - 8]
        out, out_rank = _keep_slots_stable(new, phyrank, stale, 8)
        assert out is new and out_rank is phyrank


class TestRebalanceExpertsUsesIt:
    def test_old_p2l_makes_placement_stickier(self):
        """End to end through rebalance_experts: passing the live placement must
        leave more slots untouched than not passing it."""
        num_logical, num_physical, num_gpus, num_layers = 64, 72, 8, 4
        g = torch.Generator().manual_seed(1)
        base = torch.rand((num_layers, num_logical), generator=g) + 0.1
        old, _, _ = rebalance_experts(
            base,
            num_physical=num_physical,
            num_groups=1,
            num_nodes=1,
            num_gpus=num_gpus,
            enable_hierarchical=False,
        )
        jitter = 0.7 + 0.6 * torch.rand((num_layers, num_logical), generator=g)

        def run(old_p2l):
            p2l, _, _ = rebalance_experts(
                base * jitter,
                num_physical=num_physical,
                num_groups=1,
                num_nodes=1,
                num_gpus=num_gpus,
                enable_hierarchical=False,
                old_p2l=old_p2l,
            )
            return int((old == p2l).sum())

        assert run(old) > run(None)

    def test_only_naive_gets_sticky_slots(self, monkeypatch):
        """`biased` pins hot experts to the front of each GPU block on purpose,
        in expert-id order, so repeated rebalances reproduce the same placement
        and its fast path can reuse the live map verbatim. Permuting its slots
        would fight that, so the pass must run for naive only.

        Asserted on the call itself: comparing placements cannot tell the two
        apart, because `_placement_biased` reads `old_p2l` for its own fast path.
        """
        from atom.model_ops import eplb

        calls = []
        real = eplb._keep_slots_stable

        def spy(*args, **kwargs):
            calls.append(True)
            return real(*args, **kwargs)

        monkeypatch.setattr(eplb, "_keep_slots_stable", spy)

        num_logical, num_physical, num_gpus, num_layers = 64, 72, 8, 4
        g = torch.Generator().manual_seed(2)
        weight = torch.rand((num_layers, num_logical), generator=g) + 0.1
        old, _, _ = rebalance_experts(
            weight,
            num_physical=num_physical,
            num_groups=1,
            num_nodes=1,
            num_gpus=num_gpus,
            enable_hierarchical=False,
        )

        def run(policy):
            rebalance_experts(
                weight * 1.1,
                num_physical=num_physical,
                num_groups=1,
                num_nodes=1,
                num_gpus=num_gpus,
                enable_hierarchical=False,
                policy=policy,
                old_p2l=old,
            )

        calls.clear()  # building `old` above already went through naive
        run(_placement_biased)
        assert not calls, "slot stability must not touch biased placement"
        run(None)
        assert calls, "naive placement must get slot stability"

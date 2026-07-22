"""EPLB module-A runtime helpers (statistics only)."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from functools import wraps
from typing import Any, Optional

import torch
from aiter.dist.parallel_state import get_tp_group

import logging

logger = logging.getLogger("atom")

BufferCopyPlan = list[tuple[int, int]]


@dataclass(frozen=True)
class _LocalCopyAction:
    src_slot: int
    dst_slot: int


@dataclass(frozen=True)
class _P2PAction:
    logical_expert_id: int
    peer_rank: int
    local_slot: int


def balanced_packing(
    weight: torch.Tensor, num_packs: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack weighted items into equal-size packs with greedy LPT.

    Args:
        weight: [num_layers, num_items], non-negative.
        num_packs: number of packs.

    Returns:
        pack_index: [num_layers, num_items] int32
        rank_in_pack: [num_layers, num_items] int32
    """
    assert weight.dim() == 2, "weight must be rank-2 [num_layers, num_items]"
    assert num_packs > 0, "num_packs must be > 0"
    num_layers, num_items = weight.shape
    assert (
        num_items % num_packs == 0
    ), "num_items must be divisible by num_packs for equal-cardinality packing"
    cap = num_items // num_packs
    # Do all bookkeeping in Python lists (per-element tensor index/.item() carries
    # dispatch overhead even on CPU); materialize tensors once at the end.
    weight_rows = weight.cpu().tolist()
    pack_index_rows: list[list[int]] = []
    rank_in_pack_rows: list[list[int]] = []
    for layer in range(num_layers):
        wl = weight_rows[layer]
        # Descending by weight, tie-break by original index (stable).
        order = sorted(range(num_items), key=lambda i: (-wl[i], i))
        loads = [0.0] * num_packs
        counts = [0] * num_packs
        pi = [0] * num_items
        rip = [0] * num_items
        for item in order:
            # Deterministic tie-break: lower load, then lower count, then lower pack id.
            best = min(
                (p for p in range(num_packs) if counts[p] < cap),
                key=lambda p: (loads[p], counts[p], p),
            )
            pi[item] = best
            rip[item] = counts[best]
            counts[best] += 1
            loads[best] += wl[item]
        pack_index_rows.append(pi)
        rank_in_pack_rows.append(rip)
    pack_index = torch.tensor(pack_index_rows, dtype=torch.int32, device=weight.device)
    rank_in_pack = torch.tensor(
        rank_in_pack_rows, dtype=torch.int32, device=weight.device
    )
    return pack_index, rank_in_pack


def replicate_experts(
    weight: torch.Tensor, num_physical: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Greedy replication by max(weight / replica_count).

    Args:
        weight: [num_layers, num_logical]
        num_physical: total physical experts per layer.

    Returns:
        physical_to_logical: [num_layers, num_physical] int32
        physical_rank: [num_layers, num_physical] int32
        logical_replica_count(logcnt): [num_layers, num_logical] int32
    """
    assert weight.dim() == 2, "weight must be rank-2 [num_layers, num_logical]"
    num_layers, num_logical = weight.shape
    assert num_logical > 0, "num_logical must be > 0"
    assert (
        num_physical >= num_logical
    ), "num_physical must be >= num_logical for replication"
    extra = num_physical - num_logical
    weight_rows = weight.to(torch.float32).cpu().tolist()
    logcnt_rows: list[list[int]] = []
    phy2log_rows: list[list[int]] = []
    phyrank_rows: list[list[int]] = []
    for layer in range(num_layers):
        wl = weight_rows[layer]
        cnt_l = [1] * num_logical
        for _ in range(extra):
            # Greedy: replicate the expert with the highest per-replica load.
            target = max(range(num_logical), key=lambda e: wl[e] / cnt_l[e])
            cnt_l[target] += 1
        logcnt_rows.append(cnt_l)
        p2l = [0] * num_physical
        prank = [0] * num_physical
        k = 0
        for e in range(num_logical):
            for r in range(cnt_l[e]):
                p2l[k] = e
                prank[k] = r
                k += 1
        assert k == num_physical
        phy2log_rows.append(p2l)
        phyrank_rows.append(prank)
    dev = weight.device
    logcnt = torch.tensor(logcnt_rows, dtype=torch.int32, device=dev)
    phy2log = torch.tensor(phy2log_rows, dtype=torch.int32, device=dev)
    phyrank = torch.tensor(phyrank_rows, dtype=torch.int32, device=dev)
    return phy2log, phyrank, logcnt


def _build_logical_to_physical_map(
    physical_to_logical: torch.Tensor,
    physical_rank: torch.Tensor,
    logcnt: torch.Tensor,
) -> torch.Tensor:
    """Build padded logical_to_physical map from p2l + rank + logcnt."""
    num_layers, num_physical = physical_to_logical.shape
    assert (
        physical_rank.shape == physical_to_logical.shape
    ), "physical_rank shape must match physical_to_logical"
    _, num_logical = logcnt.shape
    cur = int(logcnt.max().item())
    p2l_rows = physical_to_logical.cpu().tolist()
    prank_rows = physical_rank.cpu().tolist()
    logcnt_rows = logcnt.cpu().tolist()
    out_rows: list[list[list[int]]] = []
    for layer in range(num_layers):
        p2l_l = p2l_rows[layer]
        prank_l = prank_rows[layer]
        cnt_l = logcnt_rows[layer]
        row = [[-1] * cur for _ in range(num_logical)]
        for p in range(num_physical):
            e = p2l_l[p]
            rank = prank_l[p]
            assert 0 <= rank < cnt_l[e], "physical rank out of logical expert range"
            assert row[e][rank] == -1, "duplicate physical rank for logical expert"
            row[e][rank] = p
        for e in range(num_logical):
            need = cnt_l[e]
            if need == 0:
                continue
            got = sum(1 for r in range(need) if row[e][r] >= 0)
            assert got == need, "logical expert has missing physical ranks"
        out_rows.append(row)
    return torch.tensor(out_rows, dtype=torch.int32, device=physical_to_logical.device)


def _rebuild_placement_from_p2l(
    p2l_layer: torch.Tensor, num_logical: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reconstruct (p2l, phyrank, logcnt) from an existing physical->logical map,
    unchanged. Used by the biased sticky fast-path: returning the old p2l verbatim
    makes the new metadata identical to the live one, so migration planning sees a
    zero diff (no P2P). phyrank[slot] = the 0-based replica index of that slot's
    logical among its slots (order of appearance); logcnt[e] = #slots holding e."""
    lst = p2l_layer.tolist()
    dev = p2l_layer.device
    cnt = [0] * num_logical
    prank = [0] * len(lst)
    seen = [0] * num_logical
    for slot, lg in enumerate(lst):
        if lg < 0:
            prank[slot] = 0
            continue
        prank[slot] = seen[lg]
        seen[lg] += 1
        cnt[lg] += 1
    return (
        p2l_layer.clone(),
        torch.tensor(prank, dtype=torch.int32, device=dev),
        torch.tensor(cnt, dtype=torch.int32, device=dev),
    )


def _placement_biased(
    weight_l: torch.Tensor,
    num_physical: int,
    num_gpus: int,
    old_p2l_layer: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Biased placement policy: spend the whole redundant-expert budget on FULLY
    replicating the top-K hottest logical experts onto ALL `num_gpus` GPUs (one
    replica per GPU), then fill remaining slots with cold experts via greedy LPT
    packing. K is derived from the redundant budget of THIS call context:
        K = (num_physical - num_logical) // num_gpus
    so the hierarchical path (called per-node with phy_per_node/gpus_per_node)
    naturally does WITHIN-NODE full replication (intra-node locality; inter-node
    traffic for hot experts unchanged). K<=0 (no budget) -> naive fallback.
    Top-K is read from the live per-layer load `weight_l` (online, no history).
    Returns (p2l[num_physical], phyrank[num_physical], logcnt[num_logical]).
    """
    num_logical = weight_l.numel()
    assert num_physical % num_gpus == 0
    phy_per_gpu = num_physical // num_gpus
    force_n = (num_physical - num_logical) // num_gpus
    if force_n <= 0:
        return _placement_naive(weight_l, num_physical, num_gpus)
    assert (
        0 < force_n <= phy_per_gpu
    ), f"force_n={force_n} must be in (0, {phy_per_gpu}]"
    w = weight_l.to(torch.float32).cpu().tolist()
    order = sorted(range(num_logical), key=lambda e: (-w[e], e))
    _new_hot_set = set(order[:force_n])
    # SHORTEST-PATH (sticky) fast-path: if the previous placement already fully
    # replicates EXACTLY this hot set, keep the entire old placement -- hot AND
    # cold -- untouched, so the migration diff is zero. Once the hot set is placed,
    # per-GPU balance is already ~optimal; re-packing cold on every rebalance churns
    # cold replicas for no real balancedness gain (that churn was the ~1.4-2.2s of
    # wasted migration per interval). We only rebuild when the hot set changes.
    if old_p2l_layer is not None:
        old_list = old_p2l_layer.tolist()
        old_counts: dict[int, int] = {}
        for _lg in old_list:
            if _lg >= 0:
                old_counts[_lg] = old_counts.get(_lg, 0) + 1
        old_hot_set = {e for e in old_list[:force_n] if e >= 0}
        if old_hot_set == _new_hot_set and all(
            old_counts.get(e, 0) == num_gpus for e in _new_hot_set
        ):
            return _rebuild_placement_from_p2l(old_p2l_layer, num_logical)
    # Select the top-K hot set by load, but place them into the hot slot block in
    # STABLE (expert-id) order rather than load-rank order. The hot block is fully
    # replicated to every GPU, so when the hot SET is unchanged between rebalances
    # (the common steady state), stable ordering keeps each hot expert in the same
    # physical slot on every rank -> the per-slot old==new check in migration
    # planning skips it -> no redundant re-migration of the hottest (highest-traffic)
    # experts. Load-rank ordering would permute the block on every load wiggle and
    # re-migrate all K replicas across all GPUs for nothing.
    hot_set = _new_hot_set
    hot = sorted(hot_set)
    cold = [e for e in range(num_logical) if e not in hot_set]

    cnt = [0] * num_logical
    for e in hot:
        cnt[e] = num_gpus
    cold_slots_total = (
        num_physical - force_n * num_gpus
    )  # = num_gpus*(phy_per_gpu-force_n)
    assert cold_slots_total >= len(cold), (
        f"not enough slots for cold experts: {cold_slots_total} < {len(cold)} "
        f"(reduce force_n or raise num_redundant)"
    )
    for e in cold:
        cnt[e] = 1
    extra = cold_slots_total - len(cold)
    for _ in range(extra):
        target = max(cold, key=lambda e: w[e] / cnt[e])
        cnt[target] += 1

    # Greedy LPT pack cold replicas onto GPUs, capacity = phy_per_gpu - force_n each.
    cap_cold = phy_per_gpu - force_n
    cold_items = []  # (per_replica_load, expert, replica_rank)
    for e in cold:
        per = w[e] / cnt[e] if cnt[e] > 0 else 0.0
        for r in range(cnt[e]):
            cold_items.append((per, e, r))
    cold_items.sort(key=lambda x: (-x[0], x[1], x[2]))
    gpu_load = [0.0] * num_gpus
    gpu_cnt = [0] * num_gpus
    gpu_cold: list[list[tuple[int, int]]] = [[] for _ in range(num_gpus)]
    for load, e, r in cold_items:
        best = min(
            (g for g in range(num_gpus) if gpu_cnt[g] < cap_cold),
            key=lambda g: (gpu_load[g], gpu_cnt[g], g),
        )
        gpu_cold[best].append((e, r))
        gpu_load[best] += load
        gpu_cnt[best] += 1

    p2l = [0] * num_physical
    prank = [0] * num_physical
    for g in range(num_gpus):
        base = g * phy_per_gpu
        for i, e in enumerate(hot):
            p2l[base + i] = e
            prank[base + i] = g  # g-th replica; cnt[e]=num_gpus so rank in [0,8)
        for j, (e, r) in enumerate(gpu_cold[g]):
            p2l[base + force_n + j] = e
            prank[base + force_n + j] = r

    dev = weight_l.device
    return (
        torch.tensor(p2l, dtype=torch.int32, device=dev),
        torch.tensor(prank, dtype=torch.int32, device=dev),
        torch.tensor(cnt, dtype=torch.int32, device=dev),
    )


def _placement_naive(
    weight_l: torch.Tensor,
    num_physical: int,
    num_gpus: int,
    old_p2l_layer: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Naive placement policy (default): greedy per-replica-load replication
    (replicate_experts) + balanced_packing spread across GPUs — i.e. the same
    redundant budget spread thinly over the hottest experts. Contrast with
    `_placement_biased`. `old_p2l_layer` is accepted for a uniform policy
    signature but unused (naive has no sticky fast-path). Returns (p2l, phyrank,
    logcnt)."""
    _ = old_p2l_layer
    num_logical = weight_l.numel()
    phy2log, phyrank, logcnt = replicate_experts(
        weight_l.view(1, num_logical), num_physical
    )
    phy2log_l = phy2log[0].clone()
    phyrank_l = phyrank[0].clone()
    logcnt_l = logcnt[0].clone()
    # Step-3 pack physical slots onto GPUs (equal cardinality per GPU).
    per_phy_load = weight_l.to(torch.float32)[phy2log_l] / logcnt_l[phy2log_l].to(
        torch.float32
    )
    pack_idx, rank_in_pack = balanced_packing(per_phy_load.view(1, -1), num_gpus)
    pack_idx = pack_idx[0].to(torch.int64)
    rank_in_pack = rank_in_pack[0].to(torch.int64)
    phy_per_gpu = num_physical // num_gpus
    new_phy_index = pack_idx * phy_per_gpu + rank_in_pack
    reordered = torch.empty_like(phy2log_l)
    reordered_rank = torch.empty_like(phyrank_l)
    reordered[new_phy_index] = phy2log_l
    reordered_rank[new_phy_index] = phyrank_l
    return reordered, reordered_rank, logcnt_l


# Pluggable per-layer placement policies. A policy is a callable
#   (weight_l[num_logical], num_physical, num_gpus) -> (p2l, phyrank, logcnt).
# Policy name comes from EPLBConfig; `biased` derives its top-K from the
# redundant budget (num_redundant // num_gpus), no extra param.
_PLACEMENT_POLICIES = {"naive": _placement_naive, "biased": _placement_biased}


def resolve_placement_policy(name: Optional[str]):
    """Return the per-layer placement callable for `name` (default naive)."""
    key = (name or "naive").lower().strip()
    if key not in _PLACEMENT_POLICIES:
        raise ValueError(
            f"unknown eplb placement_policy={name!r}; "
            f"choices={sorted(_PLACEMENT_POLICIES)}"
        )
    return _PLACEMENT_POLICIES[key]


def _rebalance_single_layer_global(
    weight_l: torch.Tensor,
    num_physical: int,
    num_gpus: int,
    policy=None,
    old_p2l_layer: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (physical_to_logical, physical_rank, logcnt) for one layer via the
    given placement policy callable (default: naive). `old_p2l_layer` is the layer's
    current placement, passed to policies with a sticky/shortest-path fast-path."""
    return (policy or _placement_naive)(weight_l, num_physical, num_gpus, old_p2l_layer)


def rebalance_experts(
    weight: torch.Tensor,
    *,
    num_physical: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
    enable_hierarchical: bool,
    policy=None,
    old_p2l: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Module-C entrypoint. `policy` is the per-layer placement callable
    (default naive); in the hierarchical path it is applied per-node so `biased`
    replicates within-node. `old_p2l` [num_layers, num_physical] is the current
    live placement; passed per-layer to policies for a sticky/shortest-path
    fast-path (flat/non-hierarchical path only; hierarchical recomputes fresh).

    Returns:
        physical_to_logical_map: [num_layers, num_physical] int32
        logical_to_physical_map: [num_layers, num_logical, cur] int32 (cur=max(logcnt))
        logcnt: [num_layers, num_logical] int32
    """
    assert weight.dim() == 2, "weight must be rank-2 [num_layers, num_logical]"
    num_layers, num_logical = weight.shape
    assert num_layers > 0 and num_logical > 0
    assert num_groups > 0 and num_nodes > 0 and num_gpus > 0
    assert num_logical % num_groups == 0, "num_logical must be divisible by num_groups"
    assert num_groups % num_nodes == 0, "num_groups must be divisible by num_nodes"
    assert num_gpus % num_nodes == 0, "num_gpus must be divisible by num_nodes"
    assert num_physical % num_gpus == 0, "num_physical must be divisible by num_gpus"
    assert num_physical >= num_logical

    p2l = torch.empty(
        (num_layers, num_physical), dtype=torch.int32, device=weight.device
    )
    phyrank = torch.empty(
        (num_layers, num_physical), dtype=torch.int32, device=weight.device
    )
    logcnt = torch.zeros(
        (num_layers, num_logical), dtype=torch.int32, device=weight.device
    )

    if not enable_hierarchical or num_groups == 1 or num_nodes == 1:
        for layer in range(num_layers):
            old_p2l_l = None if old_p2l is None else old_p2l[layer]
            p2l_l, rank_l, cnt_l = _rebalance_single_layer_global(
                weight[layer],
                num_physical,
                num_gpus,
                policy=policy,
                old_p2l_layer=old_p2l_l,
            )
            p2l[layer] = p2l_l
            phyrank[layer] = rank_l
            logcnt[layer] = cnt_l
        l2p = _build_logical_to_physical_map(p2l, phyrank, logcnt)
        return p2l, l2p, logcnt

    # Hierarchical path: group->node assignment, then node-local rebalance.
    group_size = num_logical // num_groups
    groups_per_node = num_groups // num_nodes
    gpus_per_node = num_gpus // num_nodes
    phy_per_node = num_physical // num_nodes
    phy_per_gpu = num_physical // num_gpus

    for layer in range(num_layers):
        group_weight = weight[layer].view(num_groups, group_size).sum(dim=1).view(1, -1)
        group_to_node, _ = balanced_packing(group_weight, num_nodes)
        group_to_node = group_to_node[0]

        logical_ids_per_node = []
        for n in range(num_nodes):
            node_groups = [
                g for g in range(num_groups) if int(group_to_node[g].item()) == n
            ]
            # determinism in case of equal packing loads
            node_groups.sort()
            assert len(node_groups) == groups_per_node
            node_logical = []
            for g in node_groups:
                start = g * group_size
                node_logical.extend(range(start, start + group_size))
            logical_ids_per_node.append(node_logical)

        p2l_l = torch.empty((num_physical,), dtype=torch.int32, device=weight.device)
        phyrank_l = torch.empty(
            (num_physical,), dtype=torch.int32, device=weight.device
        )
        cnt_l = torch.zeros((num_logical,), dtype=torch.int32, device=weight.device)

        for node_id in range(num_nodes):
            node_logical_ids = logical_ids_per_node[node_id]
            node_weight = weight[layer, node_logical_ids]
            node_p2l_local, node_rank_local, node_cnt_local = (
                _rebalance_single_layer_global(
                    node_weight, phy_per_node, gpus_per_node, policy=policy
                )
            )
            node_global_logical = torch.tensor(
                node_logical_ids, dtype=torch.int64, device=weight.device
            )[node_p2l_local.to(torch.int64)].to(torch.int32)
            for e_local, e_global in enumerate(node_logical_ids):
                cnt_l[e_global] = node_cnt_local[e_local]

            # Map node-local physical index to global physical index.
            local_gpu = torch.div(
                torch.arange(phy_per_node, device=weight.device),
                phy_per_gpu,
                rounding_mode="floor",
            )
            local_rank = torch.remainder(
                torch.arange(phy_per_node, device=weight.device), phy_per_gpu
            )
            global_gpu = node_id * gpus_per_node + local_gpu
            global_phy = global_gpu * phy_per_gpu + local_rank
            p2l_l[global_phy.to(torch.int64)] = node_global_logical
            phyrank_l[global_phy.to(torch.int64)] = node_rank_local

        p2l[layer] = p2l_l
        phyrank[layer] = phyrank_l
        logcnt[layer] = cnt_l

    l2p = _build_logical_to_physical_map(p2l, phyrank, logcnt)
    return p2l, l2p, logcnt


def _pad_logical_to_physical(
    l2p_var: torch.Tensor, max_num_replicas: int
) -> torch.Tensor:
    """Pad module-C's variable-width [L, Lg, cur] map to fixed [L, Lg, R]."""
    num_layers, num_logical, cur = l2p_var.shape
    assert (
        cur <= max_num_replicas
    ), f"cur={cur} exceeds max_num_replicas={max_num_replicas}"
    out = torch.full(
        (num_layers, num_logical, max_num_replicas),
        -1,
        dtype=torch.int32,
        device=l2p_var.device,
    )
    out[:, :, :cur] = l2p_var.to(torch.int32)
    return out


def _build_expert_map(
    *,
    num_layers: int,
    num_physical: int,
    num_local_physical: int,
    ep_rank: int,
    device: torch.device,
) -> torch.Tensor:
    """Per-rank physical-slot -> local index map ([L, P], -1 if not on this rank).

    Physical slots are packed contiguously onto GPUs by module-C, so a rank owns
    the contiguous block [ep_rank*num_local, (ep_rank+1)*num_local); this map is
    layer-invariant but stored per-layer for uniform in-place commit.
    """
    expert_map = torch.full(
        (num_layers, num_physical), -1, dtype=torch.int32, device=device
    )
    base = ep_rank * num_local_physical
    local_ids = torch.arange(num_local_physical, dtype=torch.int32, device=device)
    expert_map[:, base : base + num_local_physical] = local_ids.unsqueeze(0).expand(
        num_layers, -1
    )
    return expert_map


def _build_rank_dispatch_map(
    *,
    logical_to_physical_map: torch.Tensor,
    logical_replica_count: torch.Tensor,
    num_local_physical: int,
    ep_rank: int,
) -> torch.Tensor:
    """Per-rank locality-aware replica choice ([L, Lg] -> physical slot id).

    For each logical expert this rank picks ONE physical replica to dispatch to:
    prefer a replica owned by this rank (local, no cross-GPU cost); otherwise
    spread deterministically across replicas by `ep_rank % replica_count`.
    """
    num_layers, num_logical, _ = logical_to_physical_map.shape
    device = logical_to_physical_map.device
    l2p_rows = logical_to_physical_map.cpu().tolist()
    cnt_rows = logical_replica_count.cpu().tolist()
    out_rows: list[list[int]] = []
    for layer in range(num_layers):
        l2p_l = l2p_rows[layer]
        cnt_l = cnt_rows[layer]
        row = [0] * num_logical
        for e in range(num_logical):
            cnt = cnt_l[e]
            assert cnt >= 1, "every logical expert must have >= 1 replica"
            reps = l2p_l[e]
            chosen = -1
            for i in range(cnt):
                p = reps[i]
                if p // num_local_physical == ep_rank:
                    chosen = p
                    break
            if chosen < 0:
                chosen = reps[ep_rank % cnt]
            row[e] = chosen
        out_rows.append(row)
    return torch.tensor(out_rows, dtype=torch.int32, device=device)


@dataclass
class ExpertLocationMetadata:
    """Central EPLB shared state: physical/logical placement maps.

    Base maps (module-C output, deterministic & identical across ranks):
      - physical_to_logical_map [L, P]        physical slot -> logical expert
      - logical_to_physical_map [L, Lg, R]    logical -> physical replicas (-1 padded to R)
      - logical_replica_count   [L, Lg]       replicas per logical (<= R)
    Per-rank derived maps (base maps + EP topology):
      - expert_map                            [L, P]  physical slot -> local index (-1 non-local)
      - logical_to_rank_dispatch_physical_map [L, Lg] this rank's chosen replica per logical

    Not all per-rank maps are dynamic. `logical_to_rank_dispatch_physical_map`
    changes every rebalance (which replica each logical dispatches to). But
    `expert_map` encodes physical-slot OWNERSHIP -- a function of ep_rank +
    budget (see `_build_expert_map`), fixed at init and independent of the load
    placement. It is therefore INVARIANT across rebalances: built once, consumed
    by `_bind_layer_expert_maps` (which seeds the layer's runtime expert_map /
    expert_mask, mask == `expert_map > -1`) and the loader consistency check, and
    NOT re-committed by `update()` (see its docstring). The layer's derived
    expert_mask lives only on the layer; it is likewise invariant.

    R = max_num_replicas is the init-fixed budget (num_redundant + 1); tensors keep
    fixed addresses so `update` can write in place (copy_) under cudagraph capture.
    """

    num_layers: int
    num_logical_experts: int
    num_physical_experts: int
    max_num_replicas: int
    ep_size: int
    ep_rank: int
    num_local_physical_experts: int
    physical_to_logical_map: torch.Tensor
    logical_to_physical_map: torch.Tensor
    logical_replica_count: torch.Tensor
    expert_map: torch.Tensor
    logical_to_rank_dispatch_physical_map: torch.Tensor

    def __post_init__(self) -> None:
        L, P = self.num_layers, self.num_physical_experts
        Lg, R = self.num_logical_experts, self.max_num_replicas
        assert self.physical_to_logical_map.shape == (L, P)
        assert self.logical_to_physical_map.shape == (L, Lg, R)
        assert self.logical_replica_count.shape == (L, Lg)
        assert self.expert_map.shape == (L, P)
        assert self.logical_to_rank_dispatch_physical_map.shape == (L, Lg)
        assert P == self.ep_size * self.num_local_physical_experts

    @classmethod
    def from_rebalance_result(
        cls,
        *,
        physical_to_logical_map: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
        ep_size: int,
        ep_rank: int,
        max_num_replicas: int,
    ) -> "ExpertLocationMetadata":
        """Assemble metadata from module-C output (pad + derive per-rank maps)."""
        num_layers, num_physical = physical_to_logical_map.shape
        _, num_logical = logical_replica_count.shape
        assert num_physical % ep_size == 0, "num_physical must be divisible by ep_size"
        num_local = num_physical // ep_size
        l2p_padded = _pad_logical_to_physical(logical_to_physical_map, max_num_replicas)
        expert_map = _build_expert_map(
            num_layers=num_layers,
            num_physical=num_physical,
            num_local_physical=num_local,
            ep_rank=ep_rank,
            device=physical_to_logical_map.device,
        )
        dispatch = _build_rank_dispatch_map(
            logical_to_physical_map=l2p_padded,
            logical_replica_count=logical_replica_count,
            num_local_physical=num_local,
            ep_rank=ep_rank,
        )
        return cls(
            num_layers=num_layers,
            num_logical_experts=num_logical,
            num_physical_experts=num_physical,
            max_num_replicas=max_num_replicas,
            ep_size=ep_size,
            ep_rank=ep_rank,
            num_local_physical_experts=num_local,
            physical_to_logical_map=physical_to_logical_map.contiguous().to(
                torch.int32
            ),
            logical_to_physical_map=l2p_padded,
            logical_replica_count=logical_replica_count.contiguous().to(torch.int32),
            expert_map=expert_map,
            logical_to_rank_dispatch_physical_map=dispatch,
        )

    @classmethod
    def from_trivial(
        cls,
        *,
        num_layers: int,
        num_logical_experts: int,
        num_physical_experts: Optional[int] = None,
        ep_size: int,
        ep_rank: int,
        device: Optional[torch.device] = None,
    ) -> "ExpertLocationMetadata":
        """Initial placement following the SGLang/vllm convention.

        Physical slot i maps to logical expert i % num_logical_experts.
        Redundant slots are assigned valid logical experts immediately (no -1
        cold-start gap), matching the round-robin used by both SGLang
        (init_trivial) and vllm (build_initial_global_physical_to_logical_map).
        """
        dev = device if device is not None else torch.device("cpu")
        num_physical = (
            num_logical_experts
            if num_physical_experts is None
            else int(num_physical_experts)
        )
        assert num_physical >= num_logical_experts
        num_redundant = num_physical - num_logical_experts

        # p2l: slot i → logical i % num_logical (round-robin)
        p2l = (
            torch.arange(num_physical, dtype=torch.int32, device=dev)
            .remainder(num_logical_experts)
            .unsqueeze(0)
            .expand(num_layers, -1)
            .contiguous()
        )

        # logcnt: first num_redundant experts get 2 replicas, rest get 1
        logcnt = torch.ones(
            (num_layers, num_logical_experts), dtype=torch.int32, device=dev
        )
        if num_redundant > 0:
            logcnt[:, :num_redundant] = 2

        # l2p: [num_layers, num_logical, actual_max_replicas]
        # slot e → expert e (primary); slot e+num_logical → expert e (redundant, if e < num_redundant)
        actual_max = 2 if num_redundant > 0 else 1
        l2p = torch.full(
            (num_layers, num_logical_experts, actual_max),
            -1,
            dtype=torch.int32,
            device=dev,
        )
        l2p[:, :, 0] = (
            torch.arange(num_logical_experts, dtype=torch.int32, device=dev)
            .unsqueeze(0)
            .expand(num_layers, -1)
        )
        if num_redundant > 0:
            extra = torch.arange(
                num_logical_experts,
                num_logical_experts + num_redundant,
                dtype=torch.int32,
                device=dev,
            )
            l2p[:, :num_redundant, 1] = extra.unsqueeze(0).expand(num_layers, -1)

        return cls.from_rebalance_result(
            physical_to_logical_map=p2l,
            logical_to_physical_map=l2p,
            logical_replica_count=logcnt,
            ep_size=ep_size,
            ep_rank=ep_rank,
            max_num_replicas=num_redundant + 1,
        )

    def update(self, other: "ExpertLocationMetadata", layer_ids: list[int]) -> None:
        """In-place atomic commit of the placement-dependent maps for the given
        layers (module-E §5).

        Same-shape copy_ into fixed-address live tensors (cudagraph-safe). Both
        metas must share num_logical / num_physical / max_num_replicas / ep_rank.

        `expert_map` is deliberately NOT committed here. It encodes physical-slot
        OWNERSHIP (which contiguous slot block this rank owns + its local buffer
        index), a function of ep_rank + budget that is fixed at init and
        independent of the load placement -- so it never changes across a
        rebalance (see `_build_expert_map`). It is built once and consumed by
        `_bind_layer_expert_maps` / the loader check; the layer runtime buffers
        (`layer.expert_map` / `layer.expert_mask`, mask == `expert_map > -1`) are
        seeded there once and need no per-rebalance refresh. The assert below
        pins that invariant: if a future placement policy ever reassigns slot
        ownership, it fires here instead of silently serving a stale map/mask --
        at which point the layer-side refresh (removed for this reason) must come
        back and `expert_map` must be committed again.
        """
        assert (
            self.max_num_replicas == other.max_num_replicas
        ), "max_num_replicas budget mismatch (live vs new)"
        assert self.physical_to_logical_map.shape == other.physical_to_logical_map.shape
        assert self.logical_to_physical_map.shape == other.logical_to_physical_map.shape
        assert self.ep_rank == other.ep_rank, "per-rank maps must be same rank"
        for layer_id in layer_ids:
            assert torch.equal(self.expert_map[layer_id], other.expert_map[layer_id]), (
                "expert_map changed across rebalance, but slot ownership is "
                "assumed invariant. Placement likely became dynamic -- re-enable "
                "the layer expert_map/mask refresh and commit expert_map here."
            )
            self.physical_to_logical_map[layer_id].copy_(
                other.physical_to_logical_map[layer_id]
            )
            self.logical_to_physical_map[layer_id].copy_(
                other.logical_to_physical_map[layer_id]
            )
            self.logical_replica_count[layer_id].copy_(
                other.logical_replica_count[layer_id]
            )
            self.logical_to_rank_dispatch_physical_map[layer_id].copy_(
                other.logical_to_rank_dispatch_physical_map[layer_id]
            )


def physical_load_to_logical_load(
    physical_load: torch.Tensor,
    physical_to_logical_map: torch.Tensor,
    num_logical_experts: int,
) -> torch.Tensor:
    """Fold [layers, physical] load into [layers, logical] by live placement."""
    assert physical_load.dim() == 2
    assert physical_to_logical_map.shape == physical_load.shape
    out = torch.zeros(
        (physical_load.shape[0], num_logical_experts),
        dtype=physical_load.dtype,
        device=physical_load.device,
    )
    idx = physical_to_logical_map.to(torch.int64)
    valid = idx >= 0
    safe_idx = torch.where(valid, idx, torch.zeros_like(idx))
    out.scatter_add_(
        1, safe_idx, torch.where(valid, physical_load, torch.zeros_like(physical_load))
    )
    return out


def _assign_sender_for_receiver(
    ranks_to_send: list[int], ranks_to_recv: list[int], recv_rank: int
) -> int:
    """Deterministic recv->sender mapping aligned with the design doc contract."""
    assert len(ranks_to_send) > 0, "ranks_to_send must be non-empty"
    assert recv_rank in ranks_to_recv, "recv_rank must exist in ranks_to_recv"
    n_send = len(ranks_to_send)
    n_recv = len(ranks_to_recv)
    base = n_recv // n_send
    recv_pos = ranks_to_recv.index(recv_rank)
    if base > 0:
        # Block-assign the first base*n_send receivers (base each), then hand the
        # remaining receivers to senders 0, 1, ... one apiece (max-min <= 1).
        cut = base * n_send
        if recv_pos < cut:
            return ranks_to_send[recv_pos // base]
        return ranks_to_send[recv_pos - cut]
    # n_recv < n_send: each receiver takes a distinct sender.
    return ranks_to_send[recv_pos]


def _select_source_rank_for_receiver(
    *,
    ranks_to_send: list[int],
    ranks_to_recv: list[int],
    recv_rank: int,
    num_gpu_per_node: int,
) -> int:
    """Node-aware deterministic source selection.

    Prefer same-node senders when available; otherwise fallback to global senders.
    Send/recv sides both call this function to keep pairwise symmetry.
    """
    assert num_gpu_per_node > 0
    recv_node = recv_rank // num_gpu_per_node
    same_node_senders = [
        r for r in ranks_to_send if (r // num_gpu_per_node) == recv_node
    ]
    if len(same_node_senders) > 0:
        same_node_recvs = [
            r for r in ranks_to_recv if (r // num_gpu_per_node) == recv_node
        ]
        return _assign_sender_for_receiver(
            same_node_senders, same_node_recvs, recv_rank
        )
    return _assign_sender_for_receiver(ranks_to_send, ranks_to_recv, recv_rank)


def _effective_p2p_chunk_size(*, requested: int, num_logical_experts: int) -> int:
    if num_logical_experts <= 1:
        return 1
    req = max(1, int(requested))
    max_allowed = num_logical_experts - 1
    if req > max_allowed:
        logger.warning(
            "EPLB ROCm P2P chunk size clamped from %d to %d "
            "(must be < num_logical_experts=%d).",
            req,
            max_allowed,
            num_logical_experts,
        )
        return max_allowed
    return req


def _as_p2p_bytes(t: torch.Tensor) -> torch.Tensor:
    """Reinterpret a tensor as uint8 for NCCL/RCCL P2P.

    NCCL/RCCL send/recv rejects packed low-bit dtypes (e.g. float4_e2m1fn_x2)
    and some fp8 dtypes. Weight migration is a raw byte copy, so a uint8 view is
    dtype-agnostic and correct. Both send and recv apply this, so element counts
    stay matched. The slice must be contiguous (true for empty_like temp buffers
    and contiguous expert-weight rows) so the uint8 view aliases the same storage
    — required for irecv to write into the real buffer rather than a copy.
    """
    if t.dtype == torch.uint8:
        return t
    if not t.is_contiguous():
        t = t.contiguous()
    return t.view(torch.uint8)


def _execute_batched_p2p_ops(
    *,
    ops_by_logical: dict[int, list[Any]],
    num_logical_experts: int,
    p2p_batch_chunk_size: int,
    cuda_stream: Optional[torch.cuda.Stream] = None,
) -> None:
    total_ops = sum(len(v) for v in ops_by_logical.values())
    if total_ops == 0:
        return
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError(
            "torch.distributed must be initialized for EPLB migration P2P"
        )

    chunk = _effective_p2p_chunk_size(
        requested=p2p_batch_chunk_size, num_logical_experts=num_logical_experts
    )

    stream_ctx = (
        torch.cuda.stream(cuda_stream) if cuda_stream is not None else nullcontext()
    )
    with stream_ctx:
        for start in range(0, num_logical_experts, chunk):
            end = min(start + chunk, num_logical_experts)
            batch = []
            for logical_id in range(start, end):
                batch.extend(ops_by_logical.get(logical_id, []))
            if not batch:
                continue
            reqs = torch.distributed.batch_isend_irecv(batch)
            for req in reqs:
                req.wait()


def _plan_single_layer_migration(
    *,
    old_p2l_layer: torch.Tensor,
    new_p2l_layer: torch.Tensor,
    num_local_physical_experts: int,
    num_gpu_per_node: int,
    rank: int,
    world_size: int,
) -> tuple[BufferCopyPlan, list[_LocalCopyAction], list[_P2PAction], list[_P2PAction]]:
    """Plan local copies + send/recv P2P actions for one layer."""
    assert old_p2l_layer.dim() == 1 and new_p2l_layer.dim() == 1
    assert old_p2l_layer.shape == new_p2l_layer.shape
    num_physical = old_p2l_layer.numel()
    assert world_size > 0
    assert num_local_physical_experts > 0
    assert num_physical == world_size * num_local_physical_experts
    assert num_gpu_per_node > 0 and world_size % num_gpu_per_node == 0
    # Convert index tensors to Python lists once: planning does many per-element
    # reads; each tensor.item() carries dispatch overhead (a GPU sync if on device).
    # vllm does the same with .cpu().numpy().
    old_list = old_p2l_layer.tolist()
    new_list = new_p2l_layer.tolist()
    base = rank * num_local_physical_experts
    old_local = old_list[base : base + num_local_physical_experts]
    new_local = new_list[base : base + num_local_physical_experts]

    # old holders: logical -> ranks that currently own it
    holders_by_logical: dict[int, list[int]] = {}
    holder_seen_by_logical: dict[int, set[int]] = {}
    src_slot_by_rank_logical: dict[tuple[int, int], int] = {}
    for gslot in range(num_physical):
        logical = old_list[gslot]
        if logical < 0:
            continue
        r = gslot // num_local_physical_experts
        if logical not in holders_by_logical:
            holders_by_logical[logical] = []
            holder_seen_by_logical[logical] = set()
        if r not in holder_seen_by_logical[logical]:
            holders_by_logical[logical].append(r)
            holder_seen_by_logical[logical].add(r)
        key = (r, logical)
        if key not in src_slot_by_rank_logical:
            src_slot_by_rank_logical[key] = gslot % num_local_physical_experts

    # local old slots by logical
    local_old_slots_by_logical: dict[int, list[int]] = {}
    for lslot in range(num_local_physical_experts):
        logical = old_local[lslot]
        if logical < 0:
            continue
        local_old_slots_by_logical.setdefault(logical, []).append(lslot)

    # ranks that need remote receive for each logical
    recv_ranks_by_logical: dict[int, list[int]] = {}
    for r in range(world_size):
        rbase = r * num_local_physical_experts
        old_r = old_list[rbase : rbase + num_local_physical_experts]
        new_r = new_list[rbase : rbase + num_local_physical_experts]
        old_set = set(x for x in old_r if x >= 0)
        new_set = set(x for x in new_r if x >= 0)
        for logical in sorted(new_set):
            if logical not in old_set:
                recv_ranks_by_logical.setdefault(logical, []).append(r)

    buffer_copy_plan: BufferCopyPlan = []
    local_copy_actions: list[_LocalCopyAction] = []
    recv_actions: list[_P2PAction] = []
    send_actions: list[_P2PAction] = []
    primary_dst_by_logical: dict[int, int] = {}

    for dst in range(num_local_physical_experts):
        old_logical = old_local[dst]
        new_logical = new_local[dst]
        if old_logical == new_logical:
            continue  # unchanged

        if new_logical in primary_dst_by_logical:
            buffer_copy_plan.append((primary_dst_by_logical[new_logical], dst))
            continue  # free-rider

        local_sources = local_old_slots_by_logical.get(new_logical, [])
        if len(local_sources) > 0:
            src = local_sources[0]
            local_copy_actions.append(_LocalCopyAction(src_slot=src, dst_slot=dst))
            buffer_copy_plan.append((dst, dst))
            primary_dst_by_logical[new_logical] = dst
            continue

        ranks_to_send = holders_by_logical.get(new_logical, [])
        ranks_to_recv = recv_ranks_by_logical.get(new_logical, [])
        assert (
            len(ranks_to_send) > 0
        ), f"no sender rank found for logical expert {new_logical}"
        assert (
            rank in ranks_to_recv
        ), f"rank={rank} expects remote logical expert {new_logical}, but not in recv set"
        src_rank = _select_source_rank_for_receiver(
            ranks_to_send=ranks_to_send,
            ranks_to_recv=ranks_to_recv,
            recv_rank=rank,
            num_gpu_per_node=num_gpu_per_node,
        )

        recv_actions.append(
            _P2PAction(
                logical_expert_id=new_logical,
                peer_rank=src_rank,
                local_slot=dst,
            )
        )
        buffer_copy_plan.append((dst, dst))
        primary_dst_by_logical[new_logical] = dst

    # Add send actions for any recv rank this rank is assigned to serve.
    for logical, ranks_to_recv in recv_ranks_by_logical.items():
        ranks_to_send = holders_by_logical.get(logical, [])
        if rank not in ranks_to_send:
            continue
        src_slot = src_slot_by_rank_logical[(rank, logical)]
        for recv_rank in ranks_to_recv:
            assigned = _select_source_rank_for_receiver(
                ranks_to_send=ranks_to_send,
                ranks_to_recv=ranks_to_recv,
                recv_rank=recv_rank,
                num_gpu_per_node=num_gpu_per_node,
            )
            if assigned != rank:
                continue
            # same-rank "send" is impossible here because recv ranks are remote-only.
            if recv_rank == rank:
                continue
            send_actions.append(
                _P2PAction(
                    logical_expert_id=logical,
                    peer_rank=recv_rank,
                    local_slot=src_slot,
                )
            )

    return buffer_copy_plan, local_copy_actions, send_actions, recv_actions


def _migrate_single_layer(
    routed_experts_weights: list[torch.Tensor],
    temp_buffers: list[torch.Tensor],
    old_p2l_layer: torch.Tensor,
    new_p2l_layer: torch.Tensor,
    num_local_physical_experts: int,
    num_gpu_per_node: int,
    rank: int,
    world_size: int,
    ep_group: Any,
    num_logical_experts: int,
    p2p_batch_chunk_size: int = 32,
    cuda_stream: Optional[torch.cuda.Stream] = None,
) -> BufferCopyPlan:
    """Migrate one layer into temp buffers and return BufferCopyPlan for module-E."""
    assert len(routed_experts_weights) == len(temp_buffers)
    if len(routed_experts_weights) == 0:
        return []
    for w, b in zip(routed_experts_weights, temp_buffers):
        assert w.shape[0] == num_local_physical_experts
        assert b.shape[0] == num_local_physical_experts

    buffer_copy_plan, local_copy_actions, send_actions, recv_actions = (
        _plan_single_layer_migration(
            old_p2l_layer=old_p2l_layer,
            new_p2l_layer=new_p2l_layer,
            num_local_physical_experts=num_local_physical_experts,
            num_gpu_per_node=num_gpu_per_node,
            rank=rank,
            world_size=world_size,
        )
    )

    stream_ctx = (
        torch.cuda.stream(cuda_stream) if cuda_stream is not None else nullcontext()
    )
    # Local copy path (case-2) must run on the same stream as P2P + commit.
    with stream_ctx:
        for action in local_copy_actions:
            for w, b in zip(routed_experts_weights, temp_buffers):
                b[action.dst_slot].copy_(w[action.src_slot])

    # Build and execute P2P ops for case-4/5.
    ops_by_logical: dict[int, list[Any]] = {}
    for action in send_actions:
        for w in routed_experts_weights:
            op = torch.distributed.P2POp(
                torch.distributed.isend,
                _as_p2p_bytes(w[action.local_slot]),
                action.peer_rank,
                ep_group,
            )
            ops_by_logical.setdefault(action.logical_expert_id, []).append(op)
    for action in recv_actions:
        for b in temp_buffers:
            op = torch.distributed.P2POp(
                torch.distributed.irecv,
                _as_p2p_bytes(b[action.local_slot]),
                action.peer_rank,
                ep_group,
            )
            ops_by_logical.setdefault(action.logical_expert_id, []).append(op)

    _execute_batched_p2p_ops(
        ops_by_logical=ops_by_logical,
        num_logical_experts=num_logical_experts,
        p2p_batch_chunk_size=p2p_batch_chunk_size,
        cuda_stream=cuda_stream,
    )
    return buffer_copy_plan


def migrate_experts_chunk(
    layer_ids: list[int],
    old_meta: Any,
    new_meta: Any,
    expert_weights_of_layer: dict[int, list[torch.Tensor]],
    temp_buffers: list[torch.Tensor],
    ep_group: Any,
    nnodes: int,
    rank: int,
    p2p_batch_chunk_size: Optional[int] = None,
    cuda_stream: Optional[torch.cuda.Stream] = None,
) -> dict[int, BufferCopyPlan]:
    """Chunk-level D entrypoint: fill temp buffers and return per-layer plans."""
    assert nnodes > 0
    if p2p_batch_chunk_size is None:
        # Lazy import to avoid module-load circular dependencies.
        from atom.config import get_current_atom_config

        cfg = get_current_atom_config()
        p2p_batch_chunk_size = int(getattr(cfg.eplb_config, "p2p_batch_chunk_size", 32))
    # num_logical is uniform across layers; read it once from metadata instead of
    # a per-layer `old_p2l_layer.max().item()` (a GPU->CPU sync that, on a busy
    # default stream, blocks until the whole forward backlog drains).
    num_logical_experts = int(old_meta.num_logical_experts)
    plans: dict[int, BufferCopyPlan] = {}
    for layer_id in layer_ids:
        old_p2l_layer = old_meta.physical_to_logical_map[layer_id]
        new_p2l_layer = new_meta.physical_to_logical_map[layer_id]
        num_physical = old_p2l_layer.numel()
        assert num_physical == new_p2l_layer.numel()
        assert num_physical % nnodes == 0
        routed_weights = expert_weights_of_layer[layer_id]
        assert len(routed_weights) == len(temp_buffers)
        num_local_physical = int(routed_weights[0].shape[0])
        world_size = num_physical // num_local_physical
        assert world_size % nnodes == 0
        num_gpu_per_node = world_size // nnodes
        plans[layer_id] = _migrate_single_layer(
            routed_experts_weights=routed_weights,
            temp_buffers=temp_buffers,
            old_p2l_layer=old_p2l_layer,
            new_p2l_layer=new_p2l_layer,
            num_local_physical_experts=num_local_physical,
            num_gpu_per_node=num_gpu_per_node,
            rank=rank,
            world_size=world_size,
            ep_group=ep_group,
            num_logical_experts=num_logical_experts,
            p2p_batch_chunk_size=p2p_batch_chunk_size,
            cuda_stream=cuda_stream,
        )
    return plans


def move_from_buffer(
    plan: BufferCopyPlan,
    temp_buffers: list[torch.Tensor],
    expert_weights: list[torch.Tensor],
    cuda_stream: Optional[torch.cuda.Stream] = None,
) -> None:
    assert len(temp_buffers) == len(expert_weights)
    if len(temp_buffers) == 0:
        return
    num_slots = int(temp_buffers[0].shape[0])
    for b, w in zip(temp_buffers, expert_weights):
        assert b.shape[0] == num_slots
        assert w.shape[0] == num_slots
    stream_ctx = (
        torch.cuda.stream(cuda_stream) if cuda_stream is not None else nullcontext()
    )
    with stream_ctx:
        for src_slot, dst_slot in plan:
            assert 0 <= src_slot < num_slots
            assert 0 <= dst_slot < num_slots
            for b, w in zip(temp_buffers, expert_weights):
                # Keep fixed tensor addresses for cudagraph compatibility.
                w[dst_slot].copy_(b[src_slot])


def commit_layer(
    plan: BufferCopyPlan,
    temp_buffers: list[torch.Tensor],
    expert_weights: list[torch.Tensor],
    live_meta: Any,
    new_meta: Any,
    layer_id: int,
    cuda_stream: Optional[torch.cuda.Stream] = None,
) -> None:
    """Module-E single-layer atomic commit: temp->weight then metadata update."""
    stream_ctx = (
        torch.cuda.stream(cuda_stream) if cuda_stream is not None else nullcontext()
    )
    with stream_ctx:
        move_from_buffer(plan, temp_buffers, expert_weights, cuda_stream=None)
        # Metadata owns in-place update of all coupled maps for this layer.
        live_meta.update(new_meta, [layer_id])


def commit_experts_chunk(
    *,
    layer_ids: list[int],
    plans: dict[int, BufferCopyPlan],
    temp_buffers: list[torch.Tensor],
    expert_weights_of_layer: dict[int, list[torch.Tensor]],
    live_meta: Any,
    new_meta: Any,
    cuda_stream: Optional[torch.cuda.Stream] = None,
) -> None:
    """Module-E explicit chunk orchestration driven by upper-layer plans."""
    for layer_id in layer_ids:
        plan = plans[layer_id]
        expert_weights = expert_weights_of_layer[layer_id]
        commit_layer(
            plan=plan,
            temp_buffers=temp_buffers,
            expert_weights=expert_weights,
            live_meta=live_meta,
            new_meta=new_meta,
            layer_id=layer_id,
            cuda_stream=cuda_stream,
        )


def migrate_and_commit_chunk(
    *,
    layer_ids: list[int],
    old_meta: Any,
    new_meta: Any,
    expert_weights_of_layer: dict[int, list[torch.Tensor]],
    temp_buffers: list[torch.Tensor],
    ep_group: Any,
    nnodes: int,
    rank: int,
    live_meta: Any,
    p2p_batch_chunk_size: Optional[int] = None,
    cuda_stream: Optional[torch.cuda.Stream] = None,
) -> None:
    """Explicit upper-layer orchestration: per-layer D migrate -> E commit.

    temp_buffers are sized for one layer and intentionally reused.  Do not
    migrate the whole chunk before committing, or earlier layers would read the
    last migrated layer's staged data.
    """
    for layer_id in layer_ids:
        plans = migrate_experts_chunk(
            layer_ids=[layer_id],
            old_meta=old_meta,
            new_meta=new_meta,
            expert_weights_of_layer=expert_weights_of_layer,
            temp_buffers=temp_buffers,
            ep_group=ep_group,
            nnodes=nnodes,
            rank=rank,
            p2p_batch_chunk_size=p2p_batch_chunk_size,
            cuda_stream=cuda_stream,
        )
        commit_experts_chunk(
            layer_ids=[layer_id],
            plans=plans,
            temp_buffers=temp_buffers,
            expert_weights_of_layer=expert_weights_of_layer,
            live_meta=live_meta,
            new_meta=new_meta,
            cuda_stream=cuda_stream,
        )


def count_physical_load(topk_physical: torch.Tensor, num_physical: int) -> torch.Tensor:
    """Count per-physical expert load for one pass.

    Invalid ids (`<0` or `>= num_physical`) are ignored.

    Capture-safe: uses only fixed-shape elementwise ops + scatter_add_, so it
    can run inside a hip/cuda graph capture (decode path). Avoids torch.bincount,
    boolean-mask indexing, and `.any()` host-syncs -- all of which raise
    "operation not permitted when stream is capturing".
    """
    assert topk_physical.dtype in (
        torch.int32,
        torch.int64,
    ), f"topk_physical must be int32 or int64, got {topk_physical.dtype}"
    counts = torch.zeros(num_physical, dtype=torch.int32, device=topk_physical.device)
    # numel() reads static shape metadata (a host int), safe during capture.
    if topk_physical.numel() == 0:
        return counts

    flat = topk_physical.reshape(-1).to(torch.int64)
    valid = (flat >= 0) & (flat < num_physical)
    # Route invalid ids to slot 0 but contribute 0 so they don't affect counts.
    safe_idx = torch.where(valid, flat, torch.zeros_like(flat))
    contrib = valid.to(torch.int32)
    counts.scatter_add_(0, safe_idx, contrib)
    return counts


class ExpertLoadMonitor:
    def __init__(self, *, enabled: bool, window_size: int):
        self.enabled = enabled
        self.window_size = max(1, int(window_size))
        self._slot = 0
        self._filled = 0
        self._num_layers = 0
        self._num_physical = 0
        self._device: Optional[torch.device] = None
        self._cur_pass_count: Optional[torch.Tensor] = None
        self._expert_load_window: Optional[torch.Tensor] = None
        self._logged_first_record: bool = False
        self._logged_logical_without_metadata: bool = False
        self._load_group: Optional[Any] = None

    def set_load_group(self, group: Any) -> None:
        self._load_group = group

    def initialize(
        self, *, num_layers: int, num_physical: int, device: torch.device
    ) -> None:
        """Allocate fixed-address load tensors once during EPLB runtime init."""
        if not self.enabled:
            return
        num_layers = int(num_layers)
        num_physical = int(num_physical)
        device = torch.device(device)
        if num_layers <= 0 or num_physical <= 0:
            raise ValueError(
                "ExpertLoadMonitor requires positive dimensions: "
                f"num_layers={num_layers}, num_physical={num_physical}"
            )
        if self._cur_pass_count is not None or self._expert_load_window is not None:
            if (
                self._cur_pass_count is not None
                and self._expert_load_window is not None
                and self._num_layers == num_layers
                and self._num_physical == num_physical
                and self._device == device
            ):
                return
            raise RuntimeError(
                "ExpertLoadMonitor is already initialized; runtime resizing is "
                "not allowed because CUDA graphs may capture these tensor "
                "addresses. Existing capacity/device="
                f"({self._num_layers}, {self._num_physical}, {self._device}), "
                f"requested=({num_layers}, {num_physical}, {device})."
            )

        self._cur_pass_count = torch.zeros(
            (num_layers, num_physical), dtype=torch.int32, device=device
        )
        self._expert_load_window = torch.zeros(
            (self.window_size, num_layers, num_physical),
            dtype=torch.int32,
            device=device,
        )
        self._num_layers = num_layers
        self._num_physical = num_physical
        self._device = device

    def initialize_for_metadata(self, meta: "ExpertLocationMetadata") -> None:
        """Preallocate fixed-address load tensors for all EPLB layers."""
        self.initialize(
            num_layers=meta.num_layers,
            num_physical=meta.num_physical_experts,
            device=meta.expert_map.device,
        )

    def ensure_capacity_for_metadata(self, meta: "ExpertLocationMetadata") -> None:
        """Compatibility wrapper for the old lazy-capacity API."""
        self.initialize_for_metadata(meta)

    def _validate_record_shape(
        self, *, layer_id: int, num_physical: int, device: torch.device
    ) -> None:
        assert (
            self._cur_pass_count is not None and self._expert_load_window is not None
        ), (
            "ExpertLoadMonitor.record() called before initialization; "
            "initialize the EPLB runtime before warmup/cudagraph capture."
        )
        assert (
            self._device == device
        ), f"ExpertLoadMonitor.record() device mismatch: initialized on {self._device}, got {device}."
        assert layer_id < self._num_layers and num_physical == self._num_physical, (
            f"ExpertLoadMonitor.record() is outside initialized capacity: "
            f"layer_id={layer_id}, num_physical={num_physical}; "
            f"capacity=({self._num_layers}, {self._num_physical})."
        )

    def on_forward_start(self) -> None:
        if not self.enabled or self._cur_pass_count is None:
            return
        self._cur_pass_count.zero_()

    def record(
        self, *, layer_id: int, topk_physical: torch.Tensor, num_physical: int
    ) -> None:
        if not self.enabled or layer_id < 0:
            return
        self._validate_record_shape(
            layer_id=layer_id,
            num_physical=num_physical,
            device=topk_physical.device,
        )
        assert self._cur_pass_count is not None
        load = count_physical_load(topk_physical, self._num_physical)
        self._cur_pass_count[layer_id].add_(load)
        if not self._logged_first_record:
            self._logged_first_record = True
            logger.info(
                "EPLB monitor first record: layer_id=%d num_physical=%d "
                "topk_shape=%s (stats hook is live)",
                layer_id,
                self._num_physical,
                tuple(topk_physical.shape),
            )

    def on_forward_end(self, is_dummy_run: bool, is_pure_prefill: bool = True) -> None:
        # Commit this pass's load to the sliding window ONLY on a real
        # (non-dummy) pure-prefill forward -- EPLB balances on prefill load
        # only, so decode / DP-mixed / dummy passes are dropped here. Committing
        # is a purely local op, so gating it per-rank never desyncs the
        # (step-driven, collective) rebalance -- lockstep is enforced separately
        # by on_forward_pass_end, which advances on a group-uniform prefill flag.
        if (
            not self.enabled
            or is_dummy_run
            or not is_pure_prefill
            or self._cur_pass_count is None
            or self._expert_load_window is None
        ):
            return
        self._expert_load_window[self._slot].copy_(self._cur_pass_count)
        self._slot = (self._slot + 1) % self.window_size
        self._filled = min(self._filled + 1, self.window_size)

    def dump_global_physical_load(self) -> Optional[torch.Tensor]:
        if self._expert_load_window is None or self._cur_pass_count is None:
            return None
        if self._filled == 0:
            local = torch.zeros_like(self._cur_pass_count)
        else:
            local = self._expert_load_window[: self._filled].sum(dim=0)

        group = self._load_group if self._load_group is not None else get_tp_group()
        world_size = int(getattr(group, "world_size", 1))
        if world_size > 1:
            # Reduce the integer token counts with a standard integer all-reduce.
            # Integer SUM is exactly associative/commutative, so the reduced
            # tensor is bit-identical on every rank regardless of the reduction
            # order -- which is what lets each rank derive the SAME rebalance
            # plan locally without broadcasting it (see SGLang, which reduces the
            # load as int32 for the same reason). The previous path routed the
            # counts through aiter's float custom all-reduce + round, which relied
            # on FP determinism to stay bit-identical across ranks.
            device_group = getattr(group, "device_group", None)
            if device_group is not None:
                global_load = local.to(torch.int64)
                torch.distributed.all_reduce(
                    global_load,
                    op=torch.distributed.ReduceOp.SUM,
                    group=device_group,
                )
                return global_load.to(torch.int32)
            # Fallback: no torch process group exposed -- keep the float path.
            global_load = group.all_reduce(local.to(torch.float32), ca_fp8_quant=False)
            return global_load.round().to(torch.int32)
        return local

    def dump_global_logical_load(self) -> Optional[torch.Tensor]:
        """Fold the observed physical load into per-logical-expert load.

        Single source of truth for the physical -> logical folding: both the
        rebalance path (``EPLBManager._execute_runtime_rebalance``) and external
        observers call this. Returns ``None`` when no load has been recorded
        yet. Before runtime metadata exists it returns the raw physical load as
        a compatibility fallback (shape ``[layers, physical]``) with a one-time
        warning. The physical/live-placement shapes are fixed at startup and can
        only agree, so a mismatch is asserted (not pad/truncated) to surface any
        future divergence instead of silently corrupting the statistics.
        """
        physical = self.dump_global_physical_load()
        if physical is None:
            return None
        meta = get_live_expert_location_metadata()
        if meta is None:
            if not self._logged_logical_without_metadata and self._num_physical > 0:
                self._logged_logical_without_metadata = True
                logger.warning(
                    "EPLB logical load requested before runtime metadata is "
                    "available; returning physical load as a compatibility fallback"
                )
            return physical
        assert tuple(physical.shape) == tuple(meta.physical_to_logical_map.shape), (
            f"EPLB physical_load shape {tuple(physical.shape)} != live placement "
            f"shape {tuple(meta.physical_to_logical_map.shape)}"
        )
        return physical_load_to_logical_load(
            physical,
            meta.physical_to_logical_map,
            meta.num_logical_experts,
        )


_MONITOR: Optional[ExpertLoadMonitor] = None
_MANAGER: Optional["EPLBManager"] = None


def get_expert_load_monitor(*, enabled: bool, window_size: int) -> ExpertLoadMonitor:
    global _MONITOR
    if (
        _MONITOR is None
        or _MONITOR.enabled != enabled
        or _MONITOR.window_size != max(1, int(window_size))
    ):
        _MONITOR = ExpertLoadMonitor(enabled=enabled, window_size=window_size)
    return _MONITOR


def get_live_expert_location_metadata() -> Optional[ExpertLocationMetadata]:
    return _MANAGER.live_metadata if _MANAGER is not None else None


class EPLBManager:
    """Module-B scheduler/trigger manager.

    Scope for now:
    - periodic step progression on every forward (including dummy)
    - balancedness gate on module-A physical load
    - state-machine trigger for owner-provided C/D/E execution
    """

    def __init__(
        self,
        *,
        enabled: bool,
        monitor: ExpertLoadMonitor,
        rebalance_interval: int,
        rebalance_min_balancedness: float,
        rebalance_balancedness_agg: str,
        placement_policy: str = "naive",
    ):
        self.enabled = enabled
        self.monitor = monitor
        self.rebalance_interval = int(rebalance_interval)
        self.rebalance_min_balancedness = float(rebalance_min_balancedness)
        self.rebalance_balancedness_agg = str(rebalance_balancedness_agg).lower()
        self.placement_policy = str(placement_policy).lower().strip()
        assert self.rebalance_interval > 0, "eplb_rebalance_interval must be > 0"
        assert (
            self.rebalance_interval >= self.monitor.window_size
        ), "eplb_rebalance_interval must be >= eplb_load_window_size"
        assert self.rebalance_balancedness_agg in (
            "min",
            "mean",
        ), "eplb_rebalance_balancedness_agg must be one of {'min','mean'}"
        self._gen = self._entrypoint()
        self._rebalance_count = 0
        self._last_balancedness: Optional[float] = None
        self.live_metadata: Optional[ExpertLocationMetadata] = None
        self._moe_layers: dict[int, Any] = {}
        self._expert_weights_of_layer: dict[int, list[torch.Tensor]] = {}
        self._reusable_temp_buffers: Optional[list[torch.Tensor]] = None
        self._expert_map_tails: dict[int, torch.Tensor] = {}
        self._ep_group: Optional[Any] = None
        # Dedicated process group for EPLB weight-migration P2P, isolated from the
        # EP group that the forward pass (MoE all-to-all etc.) runs on. NCCL/RCCL
        # matches P2P per (peer, per-communicator op order); if migration isend/
        # irecv shared the EP communicator with in-flight forward P2P/collectives,
        # a migration send could cross-match a forward op to the same peer and hang.
        # SGLang avoids this by issuing migration P2P on a separate (default) group;
        # we mirror that with an EP-membership subgroup used only for migration.
        self._migration_group: Optional[Any] = None
        self._ep_rank: int = 0
        self._nnodes: int = 1
        self._rebalance_layers_per_chunk: int = 64
        self._p2p_batch_chunk_size: int = 32
        # True iff the DP group == the migration (EP) group. When True the DP
        # sync's `any_rank_has_prefill` already spans exactly the migration
        # collective, so the prefill gate reuses it for free (no extra collective).
        # When False we OR the local prefill flag over the migration group
        # ourselves. Resolved once in bind_runtime_owner (fail-safe default False
        # => self-compute, which is always correct).
        self._dp_is_migration_group: bool = False

    def bind_runtime_owner(self, owner: Any) -> None:
        """Scan the owner's model for EP MoE layers and build runtime metadata.

        Idempotent: once ``live_metadata`` is built the call is a no-op, so it is
        safe to invoke once at model-runner init. Fail-loud on misconfiguration
        (missing model / EP MoE layers / EP group) — EPLB is only reached when
        explicitly enabled, so a broken wiring should crash at startup rather
        than silently no-op through the whole run.
        """
        self._maybe_initialize_runtime(owner)

    def _maybe_initialize_runtime(self, owner: Any) -> None:
        if self.live_metadata is not None:
            return
        model = getattr(owner, "model", None)
        if model is None or not hasattr(model, "modules"):
            raise RuntimeError(
                "EPLB is enabled but the runtime owner has no model.modules(); "
                "cannot initialize manager-owned ExpertLocationMetadata"
            )

        layers: dict[int, Any] = {}
        for module in model.modules():
            layer_id = getattr(module, "layer_id", None)
            if not isinstance(layer_id, int):
                continue
            if not bool(getattr(module, "use_ep", False)):
                continue
            if not all(hasattr(module, name) for name in ("w13_weight", "w2_weight")):
                continue
            layers[layer_id] = module
        if not layers:
            raise RuntimeError(
                "EPLB is enabled but no EP MoE layers with expert weights "
                "were found; check enable_expert_parallel and model wiring"
            )

        first_layer = layers[min(layers)]
        num_physical = int(getattr(first_layer, "global_num_experts", 0))
        num_redundant = int(getattr(first_layer, "num_redundant_experts", 0))
        assert num_physical > num_redundant
        num_logical = num_physical - num_redundant
        ep_size = int(getattr(first_layer, "ep_size"))
        ep_rank = int(getattr(first_layer, "ep_rank"))
        if num_physical % ep_size != 0:
            raise RuntimeError(
                "EPLB physical experts must be divisible by ep_size: "
                f"num_physical={num_physical}, ep_size={ep_size}"
            )
        for layer_id, layer in layers.items():
            layer_physical = int(getattr(layer, "global_num_experts", 0))
            layer_redundant = int(getattr(layer, "num_redundant_experts", 0))
            assert layer_physical > layer_redundant
            layer_logical = layer_physical - layer_redundant
            layer_ep_size = int(getattr(layer, "ep_size"))
            if (
                layer_logical != num_logical
                or layer_physical != num_physical
                or layer_ep_size != ep_size
            ):
                raise RuntimeError(
                    "EPLB requires a uniform MoE layout across managed layers: "
                    f"layer_id={layer_id}, logical={layer_logical}, "
                    f"physical={layer_physical}, ep_size={layer_ep_size}; "
                    f"expected logical={num_logical}, physical={num_physical}, "
                    f"ep_size={ep_size}"
                )
        device = first_layer.w13_weight.device
        self.live_metadata = ExpertLocationMetadata.from_trivial(
            num_layers=max(layers) + 1,
            num_logical_experts=num_logical,
            num_physical_experts=num_physical,
            ep_size=ep_size,
            ep_rank=ep_rank,
            device=device,
        )
        self._moe_layers = layers
        self._expert_weights_of_layer = {
            layer_id: self._collect_expert_weight_tensors(layer)
            for layer_id, layer in layers.items()
        }
        # Pre-allocate ONE layer's temp buffers, reused across every layer and
        # every rebalance. Allocating per-layer per-rebalance (torch.empty_like)
        # under a loaded server (KV budget full) spikes memory and OOMs. Reserved
        # here — before KV-cache sizing (model_runner initializes EPLB first) — so
        # the KV budget accounts for it. All DSv4 MoE layers share expert shape.
        first_id = min(layers)
        self._reusable_temp_buffers = [
            torch.empty_like(w) for w in self._expert_weights_of_layer[first_id]
        ]
        # Guard the trivial placement against the checkpoint loader BEFORE
        # _bind_layer_expert_maps overwrites layer.expert_map with our copy.
        self._assert_placement_matches_loaded()
        self._bind_layer_expert_maps()

        try:
            from aiter.dist.parallel_state import get_ep_group

            ep = get_ep_group()
            self.monitor.set_load_group(ep)
            self._ep_group = ep.device_group
            self._ep_rank = int(ep.rank_in_group)
            # Dedicated communicator for migration P2P (same members/order as the
            # EP group, so the ep-relative peer ranks in the migration plan stay
            # valid). Isolating migration off the EP communicator prevents its
            # isend/irecv from cross-matching in-flight forward ops on that group.
            # new_group is collective over the default group; every rank calls it.
            ep_global_ranks = torch.distributed.get_process_group_ranks(self._ep_group)
            self._migration_group = torch.distributed.new_group(ranks=ep_global_ranks)
            # If the DP group is exactly the migration group, the DP sync's
            # any_rank_has_prefill already spans it -> reuse it (free). Otherwise
            # keep False and OR the prefill flag over the migration group ourselves.
            try:
                from aiter.dist.parallel_state import get_dp_group

                dp_ranks = torch.distributed.get_process_group_ranks(
                    get_dp_group().device_group
                )
                self._dp_is_migration_group = sorted(dp_ranks) == sorted(
                    ep_global_ranks
                )
            except Exception:
                self._dp_is_migration_group = False
            logger.info("EPLB dp_is_migration_group=%s", self._dp_is_migration_group)
        except Exception as exc:
            raise RuntimeError(
                "EPLB is enabled but EP process group is unavailable; "
                "manager-owned runtime metadata cannot safely rebalance"
            ) from exc

        try:
            from atom.config import get_current_atom_config

            cfg = get_current_atom_config().eplb_config
            self._rebalance_layers_per_chunk = int(
                getattr(cfg, "rebalance_layers_per_chunk", 64)
            )
            self._p2p_batch_chunk_size = int(getattr(cfg, "p2p_batch_chunk_size", 32))
        except Exception:
            self._rebalance_layers_per_chunk = 64
            self._p2p_batch_chunk_size = 32

        logger.info(
            "EPLB runtime initialized: layers=%d num_logical=%d "
            "num_physical=%d ep_size=%d ep_rank=%d",
            len(layers),
            num_logical,
            num_physical,
            ep_size,
            ep_rank,
        )
        self.monitor.initialize_for_metadata(self.live_metadata)
        # Initialize redundant physical slots the checkpoint loader left empty.
        # At init the weights are loaded, the EP group is up, and no forward has
        # run yet -- an idle window for the one-time trivial copy.
        # PyTorch requires all ranks to participate when batch_isend_irecv is
        # the first collective on an NCCL process group. Initialize the dedicated
        # migration group with an all-rank barrier because fill_redundant issues
        # asymmetric P2P operations involving only a subset of EP ranks.
        torch.distributed.barrier(group=self._migration_group)
        self.fill_redundant()

    def _collect_expert_weight_tensors(self, layer: Any) -> list[torch.Tensor]:
        assert self.live_metadata is not None
        num_local = self.live_metadata.num_local_physical_experts
        names = (
            "w13_weight",
            "w2_weight",
            "w13_weight_scale",
            "w2_weight_scale",
            "w13_input_scale",
            "w2_input_scale",
            "w13_bias",
            "w2_bias",
        )
        # Infer the per-rank expert count from a reliable per-expert weight tensor
        # (w13/w2_weight have dim0 == #experts on this rank). Shuffled FP4 scales
        # are FLATTENED to [#experts * per_expert_rows, K] (expert-contiguous), so
        # they must be reshaped back to per-expert [#experts, N, K] before per-slot
        # migration — otherwise `tensor[:num_local]` slices raw rows (a fraction of
        # one expert) and migration moves the wrong scale bytes → relocated experts
        # get mismatched scales → wrong FP4 dequant → accuracy loss (no crash).
        ref_experts = None
        for _nm in ("w13_weight", "w2_weight"):
            _t = getattr(layer, _nm, None)
            if isinstance(_t, torch.Tensor) and _t.dim() > 0:
                ref_experts = int(_t.shape[0])
                break
        tensors: list[torch.Tensor] = []
        for name in names:
            tensor = getattr(layer, name, None)
            if not (isinstance(tensor, torch.Tensor) and tensor.dim() > 0):
                continue
            d0 = int(tensor.shape[0])
            if ref_experts is not None and d0 == ref_experts:
                per_expert = tensor
            elif ref_experts is not None and d0 > ref_experts and d0 % ref_experts == 0:
                if not tensor.is_contiguous():
                    raise RuntimeError(
                        "EPLB cannot migrate flattened per-expert tensor "
                        f"{name!r}: expected contiguous storage so the "
                        "per-expert view aliases the live weight buffer, got "
                        f"shape={tuple(tensor.shape)} stride={tuple(tensor.stride())}."
                    )
                # Flattened [#experts * N, ...] -> per-expert view
                # [#experts, N, ...]. view() is intentional: migration copy_
                # must update the live tensor storage the kernel reads, not a
                # reshape-created temporary copy.
                per_expert = tensor.view(
                    ref_experts, d0 // ref_experts, *tensor.shape[1:]
                )
            else:
                # Not per-expert mappable (e.g. global/per-tensor scalar). Skip —
                # such tensors are not relocated with experts.
                continue
            if int(per_expert.shape[0]) >= num_local:
                tensors.append(per_expert[:num_local])
        return tensors

    def _bind_layer_expert_maps(self) -> None:
        assert self.live_metadata is not None
        num_physical = self.live_metadata.num_physical_experts
        for layer_id, layer in self._moe_layers.items():
            old_map = getattr(layer, "expert_map", None)
            if isinstance(old_map, torch.Tensor) and old_map.numel() > int(
                getattr(layer, "global_num_experts", num_physical)
            ):
                tail = old_map[int(getattr(layer, "global_num_experts")) :].clone()
            else:
                tail = torch.empty(
                    0, dtype=torch.int32, device=self.live_metadata.expert_map.device
                )
            self._expert_map_tails[layer_id] = tail
            runtime_map = torch.empty(
                num_physical + int(tail.numel()),
                dtype=torch.int32,
                device=self.live_metadata.expert_map.device,
            )
            runtime_map[:num_physical].copy_(self.live_metadata.expert_map[layer_id])
            if tail.numel() > 0:
                runtime_map[num_physical:].copy_(tail.to(runtime_map.device))
            layer.expert_map = runtime_map
            if getattr(layer, "expert_mask", None) is not None:
                layer.expert_mask = (runtime_map > -1).to(torch.int32)

    def _assert_placement_matches_loaded(self) -> None:
        """Sanity-check the trivial placement against the checkpoint loader.

        `from_trivial` declares the initial physical->logical placement and
        derives expert_map independently; the model loader (determine_expert_map +
        weight_loader) established its own. They MUST agree, else migration and
        dispatch operate on a wrong picture. Both are layer-invariant, so checking
        one representative MoE layer suffices. Assert instead of trusting blindly:
          - base p2l is identity (loader convention: logical e -> physical slot e);
          - our expert_map equals the loader's layer.expert_map.
        """
        assert self.live_metadata is not None
        m = self.live_metadata
        num_logical = m.num_logical_experts
        num_physical = m.num_physical_experts
        layer_id = min(self._moe_layers)
        base = m.physical_to_logical_map[layer_id, :num_logical]
        expected = torch.arange(num_logical, dtype=base.dtype, device=base.device)
        assert torch.equal(base, expected), (
            "EPLB initial placement base p2l is not identity; incompatible with "
            "the checkpoint loader (logical e must load into physical slot e)"
        )
        loaded = getattr(self._moe_layers[layer_id], "expert_map", None)
        if isinstance(loaded, torch.Tensor):
            ours = m.expert_map[layer_id].to(loaded.device)
            assert torch.equal(ours, loaded[:num_physical]), (
                "EPLB expert_map disagrees with the loader's determine_expert_map; "
                "per-rank physical->local ownership mismatch"
            )

    def _base_only_metadata(self) -> "ExpertLocationMetadata":
        """Copy of the live placement with redundant slots emptied (-1).

        Used as the `old` side of fill_redundant's migration: its only diff vs
        the live (trivial) placement is the redundant slots, so migrating old->new
        copies each logical expert into its redundant replicas.
        """
        assert self.live_metadata is not None
        m = self.live_metadata
        num_layers = m.num_layers
        num_logical = m.num_logical_experts
        num_physical = m.num_physical_experts
        dev = m.physical_to_logical_map.device
        ident = torch.arange(num_logical, dtype=torch.int32, device=dev)
        p2l = torch.full((num_layers, num_physical), -1, dtype=torch.int32, device=dev)
        p2l[:, :num_logical] = ident.unsqueeze(0).expand(num_layers, -1)
        logcnt = torch.ones((num_layers, num_logical), dtype=torch.int32, device=dev)
        l2p = torch.full(
            (num_layers, num_logical, 1), -1, dtype=torch.int32, device=dev
        )
        l2p[:, :, 0] = ident.unsqueeze(0).expand(num_layers, -1)
        return ExpertLocationMetadata.from_rebalance_result(
            physical_to_logical_map=p2l,
            logical_to_physical_map=l2p,
            logical_replica_count=logcnt,
            ep_size=m.ep_size,
            ep_rank=m.ep_rank,
            max_num_replicas=1,
        )

    def fill_redundant(self) -> None:
        """One-time init copy of each logical expert into its redundant physical
        replica slots.

        The checkpoint loader only fills base slots (physical id < num_logical);
        the redundant slots declared by `from_trivial` are uninitialized after
        load, so dispatch to them before the first rebalance would read garbage.
        This makes the loaded weights match the trivial placement by reusing the
        migrate+commit path (old = base-only, new = live) -- a plain P2P copy from
        the base holders. No-op when there are no redundant experts.
        """
        assert self.live_metadata is not None
        if self._ep_group is None or not self._moe_layers:
            return
        m = self.live_metadata
        if m.num_physical_experts <= m.num_logical_experts:
            return  # no redundant slots to fill
        layer_ids = sorted(self._moe_layers)
        migrate_and_commit_chunk(
            layer_ids=layer_ids,
            old_meta=self._base_only_metadata(),
            new_meta=m,
            expert_weights_of_layer=self._expert_weights_of_layer,
            temp_buffers=self._reusable_temp_buffers,
            ep_group=self._migration_group,
            nnodes=self._nnodes,
            rank=self._ep_rank,
            live_meta=m,
            p2p_batch_chunk_size=self._p2p_batch_chunk_size,
            cuda_stream=None,
        )
        # fill_redundant's migration is highly asymmetric: only base-holder ranks
        # send and only redundant-holder ranks receive (trivial placement clusters
        # redundant slots on the last ranks), so ranks with no migration work return
        # from migrate_and_commit_chunk immediately. Without a barrier they race
        # ahead into the warmup forward while the migrating ranks are still running
        # P2P. Barrier (on the migration group) so all ranks finish migration before
        # any proceeds.
        torch.distributed.barrier(group=self._migration_group)
        logger.info(
            "EPLB fill_redundant: initialized %d redundant slots/layer across "
            "%d layers",
            m.num_physical_experts - m.num_logical_experts,
            len(layer_ids),
        )

    @property
    def rebalance_count(self) -> int:
        return self._rebalance_count

    @property
    def last_balancedness(self) -> Optional[float]:
        return self._last_balancedness

    def on_forward_pass_end(
        self,
        local_has_prefill: bool,
        dp_any_has_prefill: Optional[bool] = None,
    ) -> None:
        # Advance the rebalance generator ONLY on steps with prefill activity, so
        # a pure-decode step never advances the counter nor triggers the (sync,
        # default-stream) migration -- keeping decode off the EPLB critical path
        # and tying the interval to prefill activity. The flag is reduced to a
        # migration-group-uniform signal, so all ranks advance/freeze in lockstep.
        # This has-prefill gate coincides with the recording gate's pure-prefill
        # commit under ATOM's no-mixed-batch invariant (a pass is all-prefill or
        # all-decode); revisit both together if mixed batches land.
        if not self.enabled:
            return

        # Resolve a migration-group-uniform has-prefill flag.
        if self._dp_is_migration_group and dp_any_has_prefill is not None:
            # DP group == migration group: reuse the DP sync's has-prefill OR for
            # free (no extra collective).
            has_prefill = dp_any_has_prefill
        else:
            # OR the local has-prefill flag over exactly the migration collective
            # (the group the rebalance actually runs on).
            flag = torch.tensor(
                [1 if local_has_prefill else 0], device="cuda", dtype=torch.int32
            )
            torch.distributed.all_reduce(
                flag, op=torch.distributed.ReduceOp.MAX, group=self._migration_group
            )
            has_prefill = bool(flag.item())

        if not has_prefill:
            # Pure-decode step across the migration group: do not advance.
            return
        next(self._gen)

    def trigger_offline_rebalance(self, reason: str = "manual") -> None:
        if not self.enabled:
            return
        logger.info("EPLB offline rebalance triggered: reason=%s", reason)
        # Update balancedness state even on the force path for observability.
        physical_load = self.monitor.dump_global_physical_load()
        if physical_load is not None:
            self._compute_balancedness_and_update(physical_load)
        for _ in self._execute_rebalance():
            pass  # drain generator synchronously

    def _entrypoint(self):
        # vllm-style warm start: the first LIVE rebalance fires after a quarter
        # of the interval (equivalent to vllm initializing its rearrangement
        # step counter to 3/4 of the interval), so balancing on real traffic
        # kicks in early. Initial placement stays trivial (round-robin) until
        # this first real-load rebalance
        first_window = max(1, self.rebalance_interval // 4)
        for _ in range(first_window):
            yield
        yield from self._rebalance()
        while True:
            for _ in range(self.rebalance_interval):
                yield
            yield from self._rebalance()

    def _rebalance(self):
        """Periodic rebalance generator (with balancedness gate).

        Yields 0 times in Phase 1 (C/D/E not yet implemented). When chunked
        migration is added, this will yield between chunks so a forward pass
        can run in between:
            for chunk in self._chunk_layers(...):
                yield
                migrate_and_commit(new_meta, layer_ids=chunk)
        """
        physical_load = self.monitor.dump_global_physical_load()
        if physical_load is None:
            return
        if not self._need_rebalance(physical_load):
            return
        yield from self._execute_rebalance()

    def _execute_rebalance(self):
        """Generator: run one rebalance (rearrange + chunked migrate/commit),
        yielding between chunks so a forward pass can run in between.

        Migration runs on the default/current stream (aligned with SGLang &
        vllm). A dedicated stream would run the P2P + weight copies CONCURRENTLY
        with the just-submitted (but not yet GPU-complete) forward pass on the
        default stream, overwriting expert weights mid-read → HSA hardware
        exception. Same-stream ordering makes migration naturally queue after
        the last forward pass's kernels, no synchronize needed.
        """
        self._rebalance_count += 1
        yield from self._execute_runtime_rebalance()

    def _execute_runtime_rebalance(self):
        import time as _time

        assert self.live_metadata is not None
        if self._ep_group is None:
            logger.warning("EPLB rebalance skipped: EP process group is unavailable")
            return

        logical_load = self.monitor.dump_global_logical_load()
        if logical_load is None:
            return

        first_layer = self._moe_layers[min(self._moe_layers)]
        num_groups = int(getattr(first_layer, "num_expert_group", None) or 1)
        num_nodes = self._nnodes
        ep_size = self.live_metadata.ep_size
        _ep_rank = self.live_metadata.ep_rank
        _rc = self._rebalance_count

        _t0 = _time.perf_counter()
        p2l, l2p, logcnt = rebalance_experts(
            logical_load,
            num_physical=self.live_metadata.num_physical_experts,
            num_groups=num_groups,
            num_nodes=num_nodes,
            num_gpus=ep_size,
            enable_hierarchical=(num_groups > 1 and num_nodes > 1),
            policy=resolve_placement_policy(self.placement_policy),
            # Current live placement -> enables the biased sticky/shortest-path
            # fast-path: unchanged hot set => reuse old placement => zero migration.
            old_p2l=self.live_metadata.physical_to_logical_map,
        )
        new_meta = ExpertLocationMetadata.from_rebalance_result(
            physical_to_logical_map=p2l,
            logical_to_physical_map=l2p,
            logical_replica_count=logcnt,
            ep_size=self.live_metadata.ep_size,
            ep_rank=self.live_metadata.ep_rank,
            max_num_replicas=self.live_metadata.max_num_replicas,
        )

        chunk_size = max(1, int(self._rebalance_layers_per_chunk))
        layer_ids = sorted(self._moe_layers)
        num_chunks = (len(layer_ids) + chunk_size - 1) // chunk_size
        _t_rearrange_ms = (_time.perf_counter() - _t0) * 1000.0

        logger.info(
            "EPLB rebalance #%d ep_rank=%d: rearrange=%.1fms; migrating %d layers "
            "in %d chunks (size=%d)",
            _rc,
            _ep_rank,
            _t_rearrange_ms,
            len(layer_ids),
            num_chunks,
            chunk_size,
        )

        self._log_rebalance_metrics(rc=_rc, logical_load=logical_load, logcnt=logcnt)

        for start in range(0, len(layer_ids), chunk_size):
            chunk = layer_ids[start : start + chunk_size]
            yield
            _tc = _time.perf_counter()
            for layer_id in chunk:
                # Reuse the pre-allocated single-layer temp buffers (allocated once
                # at init) across every layer/rebalance — no serving-time alloc, no
                # OOM spike. Each layer's migrate+commit finishes before the next
                # reuses them. Same approach as vllm's shared weights_buffer.
                temp_buffers = self._reusable_temp_buffers
                migrate_and_commit_chunk(
                    layer_ids=[layer_id],
                    old_meta=self.live_metadata,
                    new_meta=new_meta,
                    expert_weights_of_layer=self._expert_weights_of_layer,
                    temp_buffers=temp_buffers,
                    ep_group=self._migration_group,
                    nnodes=self._nnodes,
                    rank=self._ep_rank,
                    live_meta=self.live_metadata,
                    p2p_batch_chunk_size=self._p2p_batch_chunk_size,
                    cuda_stream=None,
                )
                # if needed, just refresh relayer expert_map/mask here
            _chunk_ms = (_time.perf_counter() - _tc) * 1000.0
            logger.info(
                "EPLB rebalance #%d ep_rank=%d chunk %d/%d layers=%s migrate=%.1fms",
                _rc,
                _ep_rank,
                start // chunk_size + 1,
                num_chunks,
                chunk,
                _chunk_ms,
            )

    def _log_rebalance_metrics(
        self, *, rc: int, logical_load: torch.Tensor, logcnt: torch.Tensor
    ) -> None:
        """Log the NEW plan's characteristics (ep_rank 0 only; identical across
        ranks) for sizing num_redundant / choosing placement policy:
          - replicated_experts: how many (layer, logical) slots got >1 replica.
          - traffic_to_replicated: fraction of load hitting replicated (logcnt>1)
            experts. The CEILING of any locality-first dispatch benefit, since
            non-replicated experts have a single slot and no dispatch choice.

        The live placement's per-GPU balancedness (the gate metric) is logged
        by the gate (_need_rebalance) for the same window.

        Diagnostic only: never raises into the rebalance path.
        """
        assert self.live_metadata is not None
        if self.live_metadata.ep_rank != 0:
            return
        try:
            ll = logical_load.detach().to("cpu", torch.float32)
            logcnt_cpu = logcnt.detach().to("cpu")
            redundant_mask = logcnt_cpu > 1
            total = float(ll.sum().item())
            frac = float(ll[redundant_mask].sum().item()) / total if total > 0 else 0.0
            num_redundant = int(redundant_mask.sum().item())

            # New-plan characteristics only. The live placement's per-GPU
            # balancedness is logged by the gate (_need_rebalance).
            logger.info(
                "EPLB rebalance #%d metrics: replicated_experts=%d/%d "
                "(%.1f/layer), traffic_to_replicated=%.4f",
                rc,
                num_redundant,
                logcnt_cpu.numel(),
                num_redundant / logcnt_cpu.shape[0],
                frac,
            )
        except Exception as exc:  # pragma: no cover - diagnostic only
            logger.warning("EPLB rebalance #%d metrics calc failed: %s", rc, exc)

    def _need_rebalance(self, physical_load: torch.Tensor) -> bool:
        balancedness = self._compute_balancedness_and_update(physical_load)
        if balancedness >= self.rebalance_min_balancedness:
            logger.info(
                "EPLB gate @interval: balancedness=%.3f >= threshold=%.3f -> SKIP",
                balancedness,
                self.rebalance_min_balancedness,
            )
            return False
        logger.info(
            "EPLB gate @interval: balancedness=%.3f < threshold=%.3f -> REBALANCE",
            balancedness,
            self.rebalance_min_balancedness,
        )
        return True

    def _compute_balancedness_and_update(self, physical_load: torch.Tensor) -> float:
        balancedness = self._compute_balancedness(physical_load)
        self._last_balancedness = balancedness
        return balancedness

    def _compute_balancedness(self, physical_load: torch.Tensor) -> float:
        # Per-GPU balancedness of the live placement on THIS window, from the
        # MEASURED per-physical-slot load: sum each GPU's contiguous slot block
        # into a per-(layer, GPU) load, take per-layer mean/max over GPUs, then
        # aggregate over layers (agg='min' = worst layer, 'mean' = average).
        # Measured load means replica traffic is already split across GPUs -- no
        # logical->physical projection, no replica double-counting. GPU is the
        # bottleneck unit (ranks sync at combine), so this is the balancedness
        # that actually reflects whether a rebalance would help.
        load_f = physical_load.to(torch.float32)
        num_l, num_p = load_f.shape
        meta = self.live_metadata
        ep = int(meta.ep_size) if meta is not None else 0
        if ep <= 0 or num_p % ep != 0:
            return 0.0
        perg = load_f.view(num_l, ep, num_p // ep).sum(dim=2)
        per_layer_max = perg.max(dim=1).values
        per_layer_mean = perg.mean(dim=1)
        per_layer_bal = torch.ones_like(per_layer_mean)
        nonzero = per_layer_max > 0
        per_layer_bal[nonzero] = per_layer_mean[nonzero] / per_layer_max[nonzero]
        if self.rebalance_balancedness_agg == "mean":
            return float(per_layer_bal.mean().item())
        return float(per_layer_bal.min().item())


def get_eplb_manager(
    *,
    enabled: bool,
    monitor: ExpertLoadMonitor,
    rebalance_interval: int,
    rebalance_min_balancedness: float,
    rebalance_balancedness_agg: str,
    placement_policy: str = "naive",
) -> EPLBManager:
    global _MANAGER
    if (
        _MANAGER is None
        or _MANAGER.enabled != enabled
        or _MANAGER.monitor is not monitor
        or _MANAGER.monitor.window_size != monitor.window_size
        or _MANAGER.rebalance_interval != int(rebalance_interval)
        or _MANAGER.rebalance_min_balancedness != float(rebalance_min_balancedness)
        or _MANAGER.rebalance_balancedness_agg
        != str(rebalance_balancedness_agg).lower()
        or _MANAGER.placement_policy != str(placement_policy).lower().strip()
    ):
        _MANAGER = EPLBManager(
            enabled=enabled,
            monitor=monitor,
            rebalance_interval=rebalance_interval,
            rebalance_min_balancedness=rebalance_min_balancedness,
            rebalance_balancedness_agg=rebalance_balancedness_agg,
            placement_policy=placement_policy,
        )
    return _MANAGER


def _get_configured_eplb_manager() -> Optional[EPLBManager]:
    from atom.config import get_current_atom_config

    cfg = get_current_atom_config()
    if not getattr(cfg, "eplb_enable", False):
        return None
    monitor = get_expert_load_monitor(
        enabled=True, window_size=cfg.eplb_config.load_window_size
    )
    return get_eplb_manager(
        enabled=True,
        monitor=monitor,
        rebalance_interval=cfg.eplb_config.rebalance_interval,
        rebalance_min_balancedness=cfg.eplb_config.rebalance_min_balancedness,
        rebalance_balancedness_agg=cfg.eplb_config.rebalance_balancedness_agg,
        placement_policy=cfg.eplb_config.placement_policy,
    )


def initialize_eplb_runtime(owner: Any) -> Optional[EPLBManager]:
    """Initialize manager-owned EPLB runtime state before warmup/capture.

    Returns None when EPLB is disabled. When enabled, binds the runtime owner
    and builds ExpertLocationMetadata fail-loud: a misconfiguration (no EP MoE
    layers / no EP group) raises here at startup rather than silently no-op'ing.
    """
    manager = _get_configured_eplb_manager()
    if manager is None:
        return None
    manager.bind_runtime_owner(owner)
    return manager


def trigger_eplb_profile_rearrange() -> None:
    """Force one full rebalance (migrate + commit) immediately, off the periodic
    schedule and ignoring the balancedness gate.

    NOTE: This is NOT wired into the serving path anymore (see model_runner).
    Kept for manual/debug use only; real load-aware
    balancing happens at the first live rebalance (interval/4 forwards in).
    """
    manager = _get_configured_eplb_manager()
    if manager is None:
        return
    manager.trigger_offline_rebalance(reason="profile")


def with_eplb_forward_monitor(fn):
    # Resolve once on the first call: if EPLB is disabled at that point the
    # inner function is replaced with a direct pass-through so subsequent calls
    # pay no overhead. _MANAGER and its live_metadata are set by
    # initialize_eplb_runtime() during model runner init, which always precedes
    # the first forward pass, so the hot path never re-binds the owner.
    resolved_manager: list[EPLBManager | None] = []

    @wraps(fn)
    def wrapper(self, batch, *args, **kwargs):
        if not resolved_manager:
            resolved_manager.append(_MANAGER)
        manager = resolved_manager[0]
        # Pass through when EPLB is off, or defensively if init never bound
        # metadata (should not happen — init is fail-loud).
        if manager is None or manager.live_metadata is None:
            return fn(self, batch, *args, **kwargs)
        monitor = manager.monitor
        monitor.on_forward_start()
        try:
            return fn(self, batch, *args, **kwargs)
        finally:
            is_dummy_run = getattr(batch, "is_dummy_run", False)
            # Pure-prefill = has prefill tokens and no decode tokens (batches are
            # not mixed today; the decode==0 check also future-proofs the TODO
            # mixed batch). Local per-rank signal -- no DP sync needed, since only
            # the (non-collective) window commit is gated; the step still advances.
            is_pure_prefill = (
                getattr(batch, "total_tokens_num_prefill", 0) > 0
                and getattr(batch, "total_tokens_num_decode", 0) == 0
            )
            # Recording gate commits clean load on pure-prefill only (local op).
            monitor.on_forward_end(is_dummy_run, is_pure_prefill)
            # Step gate advances on prefill ACTIVITY (has-prefill), reduced to a
            # group-uniform flag. Coincides with the pure-prefill commit above
            # under the no-mixed-batch invariant (see on_forward_pass_end).
            local_has_prefill = getattr(batch, "total_tokens_num_prefill", 0) > 0
            dp_any_has_prefill = getattr(self, "_eplb_any_rank_has_prefill", None)
            manager.on_forward_pass_end(local_has_prefill, dp_any_has_prefill)

    return wrapper


def eplb_map_logical_to_physical(
    layer: Any, topk_ids: "torch.Tensor"
) -> "torch.Tensor":
    """Remap router logical expert ids to physical slot ids for EP dispatch.

    Returns topk_ids unchanged when EPLB metadata is unavailable (non-EP or
    pre-rebalance), so callers need no EPLB-awareness guard.
    """
    meta = get_live_expert_location_metadata()
    layer_id = getattr(layer, "layer_id", None)
    if meta is None or not isinstance(layer_id, int):
        return topk_ids
    dispatch = meta.logical_to_rank_dispatch_physical_map[layer_id].to(
        device=topk_ids.device
    )
    num_logical = int(dispatch.numel())
    id_delta = int(meta.num_physical_experts - meta.num_logical_experts)
    topk_i64 = topk_ids.to(torch.int64)
    valid = (topk_i64 >= 0) & (topk_i64 < num_logical)
    safe_logical = torch.where(valid, topk_i64, torch.zeros_like(topk_i64))
    mapped = dispatch[safe_logical].to(topk_ids.dtype)
    shifted_tail = (topk_i64 + id_delta).to(topk_ids.dtype)
    tail_or_invalid = torch.where(topk_i64 >= num_logical, shifted_tail, topk_ids)
    return torch.where(valid, mapped, tail_or_invalid)


def record_eplb_expert_load(layer: Any, topk_physical: "torch.Tensor") -> None:
    """Record per-physical-slot token counts for EPLB load monitoring."""
    from atom.config import get_current_atom_config

    atom_cfg = get_current_atom_config()
    if not getattr(atom_cfg, "eplb_enable", False):
        return
    layer_id = getattr(layer, "layer_id", None)
    if not isinstance(layer_id, int):
        return
    meta = get_live_expert_location_metadata()
    num_physical = (
        int(meta.num_physical_experts)
        if meta is not None
        else int(getattr(layer, "global_num_experts", -1))
    )
    if num_physical <= 0:
        return
    monitor = get_expert_load_monitor(
        enabled=True, window_size=atom_cfg.eplb_config.load_window_size
    )
    monitor.record(
        layer_id=layer_id, topk_physical=topk_physical, num_physical=num_physical
    )

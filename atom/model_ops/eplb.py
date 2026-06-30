"""EPLB module-A runtime helpers (statistics only)."""

from __future__ import annotations

from functools import wraps
from typing import Callable, Optional

import torch
from aiter.dist.parallel_state import get_tp_group

import logging

logger = logging.getLogger("atom")


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
    pack_index = torch.empty_like(weight, dtype=torch.int32)
    rank_in_pack = torch.empty_like(weight, dtype=torch.int32)
    for l in range(num_layers):
        # Descending by weight, tie-break by original index (stable argsort).
        order = torch.argsort(weight[l], descending=True, stable=True).tolist()
        loads = [0.0] * num_packs
        counts = [0] * num_packs
        for item in order:
            candidates = [p for p in range(num_packs) if counts[p] < cap]
            # Deterministic tie-break: lower load, then lower count, then lower pack id.
            best = min(candidates, key=lambda p: (loads[p], counts[p], p))
            pack_index[l, item] = best
            rank_in_pack[l, item] = counts[best]
            counts[best] += 1
            loads[best] += float(weight[l, item].item())
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
    logcnt = torch.ones((num_layers, num_logical), dtype=torch.int32, device=weight.device)
    extra = num_physical - num_logical
    if extra > 0:
        weight_f = weight.to(torch.float32)
        for l in range(num_layers):
            for _ in range(extra):
                score = weight_f[l] / logcnt[l].to(torch.float32)
                target = int(torch.argmax(score).item())
                logcnt[l, target] += 1
    phy2log = torch.empty((num_layers, num_physical), dtype=torch.int32, device=weight.device)
    phyrank = torch.empty((num_layers, num_physical), dtype=torch.int32, device=weight.device)
    for l in range(num_layers):
        k = 0
        for e in range(num_logical):
            cnt = int(logcnt[l, e].item())
            for r in range(cnt):
                phy2log[l, k] = e
                phyrank[l, k] = r
                k += 1
        assert k == num_physical
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
    out = torch.full(
        (num_layers, num_logical, cur),
        -1,
        dtype=torch.int32,
        device=physical_to_logical.device,
    )
    for l in range(num_layers):
        for p in range(num_physical):
            e = int(physical_to_logical[l, p].item())
            rank = int(physical_rank[l, p].item())
            max_rank = int(logcnt[l, e].item())
            assert 0 <= rank < max_rank, "physical rank out of logical expert range"
            assert out[l, e, rank] == -1, "duplicate physical rank for logical expert"
            out[l, e, rank] = p
        for e in range(num_logical):
            need = int(logcnt[l, e].item())
            if need == 0:
                continue
            got = int((out[l, e, :need] >= 0).sum().item())
            assert got == need, "logical expert has missing physical ranks"
    return out


def _rebalance_single_layer_global(
    weight_l: torch.Tensor, num_physical: int, num_gpus: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (physical_to_logical[num_physical], phyrank[num_physical], logcnt[num_logical])."""
    num_logical = weight_l.numel()
    phy2log, phyrank, logcnt = replicate_experts(
        weight_l.view(1, num_logical), num_physical
    )
    phy2log_l = phy2log[0].clone()
    phyrank_l = phyrank[0].clone()
    logcnt_l = logcnt[0].clone()
    # Step-3 pack physical slots onto GPUs (equal cardinality per GPU).
    per_phy_load = weight_l.to(torch.float32)[phy2log_l] / logcnt_l[phy2log_l].to(torch.float32)
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


def rebalance_experts(
    weight: torch.Tensor,
    *,
    num_physical: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
    enable_hierarchical: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Module-C entrypoint.

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

    p2l = torch.empty((num_layers, num_physical), dtype=torch.int32, device=weight.device)
    phyrank = torch.empty((num_layers, num_physical), dtype=torch.int32, device=weight.device)
    logcnt = torch.zeros((num_layers, num_logical), dtype=torch.int32, device=weight.device)

    if not enable_hierarchical or num_groups == 1 or num_nodes == 1:
        for l in range(num_layers):
            p2l_l, rank_l, cnt_l = _rebalance_single_layer_global(
                weight[l], num_physical, num_gpus
            )
            p2l[l] = p2l_l
            phyrank[l] = rank_l
            logcnt[l] = cnt_l
        l2p = _build_logical_to_physical_map(p2l, phyrank, logcnt)
        return p2l, l2p, logcnt

    # Hierarchical path: group->node assignment, then node-local rebalance.
    group_size = num_logical // num_groups
    groups_per_node = num_groups // num_nodes
    gpus_per_node = num_gpus // num_nodes
    phy_per_node = num_physical // num_nodes
    phy_per_gpu = num_physical // num_gpus

    for l in range(num_layers):
        group_weight = weight[l].view(num_groups, group_size).sum(dim=1).view(1, -1)
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
        phyrank_l = torch.empty((num_physical,), dtype=torch.int32, device=weight.device)
        cnt_l = torch.zeros((num_logical,), dtype=torch.int32, device=weight.device)

        for node_id in range(num_nodes):
            node_logical_ids = logical_ids_per_node[node_id]
            node_weight = weight[l, node_logical_ids]
            node_p2l_local, node_rank_local, node_cnt_local = _rebalance_single_layer_global(
                node_weight, phy_per_node, gpus_per_node
            )
            node_global_logical = torch.tensor(
                node_logical_ids, dtype=torch.int64, device=weight.device
            )[node_p2l_local.to(torch.int64)].to(torch.int32)
            for e_local, e_global in enumerate(node_logical_ids):
                cnt_l[e_global] = node_cnt_local[e_local]

            # Map node-local physical index to global physical index.
            local_gpu = torch.div(
                torch.arange(phy_per_node, device=weight.device), phy_per_gpu, rounding_mode="floor"
            )
            local_rank = torch.remainder(
                torch.arange(phy_per_node, device=weight.device), phy_per_gpu
            )
            global_gpu = node_id * gpus_per_node + local_gpu
            global_phy = global_gpu * phy_per_gpu + local_rank
            p2l_l[global_phy.to(torch.int64)] = node_global_logical
            phyrank_l[global_phy.to(torch.int64)] = node_rank_local

        p2l[l] = p2l_l
        phyrank[l] = phyrank_l
        logcnt[l] = cnt_l

    l2p = _build_logical_to_physical_map(p2l, phyrank, logcnt)
    return p2l, l2p, logcnt

def count_physical_load(
    topk_physical: torch.Tensor, num_physical: int
) -> torch.Tensor:
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
    counts = torch.zeros(
        num_physical, dtype=torch.int32, device=topk_physical.device
    )
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
        self._is_frozen: bool = False
        self._logged_first_record: bool = False

    def freeze(self) -> None:
        """Lock tensor addresses before cudagraph capture.

        After this call, _ensure_capacity will raise if any new layer_id or
        num_physical is seen — prevents silent stale-address writes inside a
        captured graph.
        """
        self._is_frozen = True

    def _ensure_capacity(
        self, *, layer_id: int, num_physical: int, device: torch.device
    ) -> None:
        need_layers = max(self._num_layers, layer_id + 1)
        need_physical = max(self._num_physical, num_physical)
        need_alloc = (
            self._cur_pass_count is None
            or self._expert_load_window is None
            or self._device != device
            or need_layers != self._num_layers
            or need_physical != self._num_physical
        )
        if not need_alloc:
            return
        if self._is_frozen:
            raise RuntimeError(
                f"ExpertLoadMonitor is frozen (post-cudagraph-capture) but "
                f"_ensure_capacity was triggered: layer_id={layer_id}, "
                f"num_physical={num_physical} vs current "
                f"({self._num_layers}, {self._num_physical}). "
                "Call monitor.freeze() only after all layers have been seen."
            )

        new_cur = torch.zeros(
            (need_layers, need_physical), dtype=torch.int32, device=device
        )
        new_window = torch.zeros(
            (self.window_size, need_layers, need_physical),
            dtype=torch.int32,
            device=device,
        )

        if (
            self._cur_pass_count is not None
            and self._expert_load_window is not None
            and self._device == device
            and self._num_layers > 0
            and self._num_physical > 0
        ):
            old_l = self._num_layers
            old_p = self._num_physical
            new_cur[:old_l, :old_p].copy_(self._cur_pass_count)
            new_window[:, :old_l, :old_p].copy_(self._expert_load_window)

        self._cur_pass_count = new_cur
        self._expert_load_window = new_window
        self._num_layers = need_layers
        self._num_physical = need_physical
        self._device = device

    def on_forward_start(self) -> None:
        if not self.enabled or self._cur_pass_count is None:
            return
        self._cur_pass_count.zero_()

    def record(
        self, *, layer_id: int, topk_physical: torch.Tensor, num_physical: int
    ) -> None:
        if not self.enabled or layer_id < 0:
            return
        self._ensure_capacity(
            layer_id=layer_id, num_physical=num_physical, device=topk_physical.device
        )
        assert self._cur_pass_count is not None
        load = count_physical_load(topk_physical, self._num_physical)
        self._cur_pass_count[layer_id].add_(load)
        if not self._logged_first_record:
            self._logged_first_record = True
            logger.info(
                "EPLB monitor first record: layer_id=%d num_physical=%d "
                "topk_shape=%s nonzero_experts=%d (stats hook is live)",
                layer_id,
                self._num_physical,
                tuple(topk_physical.shape),
                int((load > 0).sum().item()),
            )

    def on_forward_end(self, is_dummy_run: bool) -> None:
        if (
            not self.enabled
            or is_dummy_run
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

        tp_group = get_tp_group()
        if tp_group.world_size > 1:
            # Group all_reduce path is float-oriented in this stack.
            global_load = tp_group.all_reduce(local.to(torch.float32), ca_fp8_quant=False)
            return global_load.round().to(torch.int32)
        return local

    def dump_global_logical_load(self) -> Optional[torch.Tensor]:
        # First integration stage keeps physical==logical.
        return self.dump_global_physical_load()


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


class EPLBManager:
    """Module-B scheduler/trigger manager.

    Scope for now:
    - periodic step progression on every forward (including dummy)
    - balancedness gate on module-A physical load
    - trigger callback skeleton for future rebalance execution
    """

    def __init__(
        self,
        *,
        enabled: bool,
        monitor: ExpertLoadMonitor,
        rebalance_interval: int,
        rebalance_min_balancedness: float,
        rebalance_balancedness_agg: str,
        on_rebalance: Optional[Callable[[], None]] = None,
    ):
        self.enabled = enabled
        self.monitor = monitor
        self.rebalance_interval = int(rebalance_interval)
        self.rebalance_min_balancedness = float(rebalance_min_balancedness)
        self.rebalance_balancedness_agg = str(rebalance_balancedness_agg).lower()
        self.on_rebalance = on_rebalance
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

    @property
    def rebalance_count(self) -> int:
        return self._rebalance_count

    @property
    def last_balancedness(self) -> Optional[float]:
        return self._last_balancedness

    def on_forward_pass_end(self, is_dummy_run: bool) -> None:
        # Keep scheduler lockstep regardless of dummy/non-dummy.
        _ = is_dummy_run
        if not self.enabled:
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
        """Generator: the actual rebalance work, chunked across forwards.

        Phase 1: no chunks (C/D/E not implemented), yields nothing.
        When D/E are added, replace the body with:
            for chunk in self._chunk_layers(all_moe_layer_ids):
                yield
                migrate_and_commit(new_meta, layer_ids=chunk)
        """
        self._rebalance_count += 1
        if self.on_rebalance is not None:
            self.on_rebalance()
        # Marks this as a generator function so `yield from _execute_rebalance()`
        # works today; the real yields will be added with D/E.
        if False:  # pragma: no cover
            yield

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
        # per-layer balancedness = mean / max over physical experts
        load_f = physical_load.to(torch.float32)
        per_layer_max = load_f.max(dim=1).values
        per_layer_mean = load_f.mean(dim=1)
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
    ):
        _MANAGER = EPLBManager(
            enabled=enabled,
            monitor=monitor,
            rebalance_interval=rebalance_interval,
            rebalance_min_balancedness=rebalance_min_balancedness,
            rebalance_balancedness_agg=rebalance_balancedness_agg,
        )
    return _MANAGER


def with_eplb_forward_monitor(fn):
    @wraps(fn)
    def wrapper(self, batch, *args, **kwargs):
        # Lazy import to avoid a circular import at module load time
        # (atom.config <-> atom.model_ops).
        from atom.config import get_current_atom_config

        cfg = get_current_atom_config()
        if not getattr(cfg, "eplb_enable", False):
            return fn(self, batch, *args, **kwargs)
        monitor = get_expert_load_monitor(
            enabled=True, window_size=cfg.eplb_load_window_size
        )
        manager = get_eplb_manager(
            enabled=True,
            monitor=monitor,
            rebalance_interval=cfg.eplb_rebalance_interval,
            rebalance_min_balancedness=cfg.eplb_rebalance_min_balancedness,
            rebalance_balancedness_agg=cfg.eplb_rebalance_balancedness_agg,
        )
        monitor.on_forward_start()
        try:
            return fn(self, batch, *args, **kwargs)
        finally:
            is_dummy_run = getattr(batch, "is_dummy_run", False)
            monitor.on_forward_end(is_dummy_run)
            manager.on_forward_pass_end(is_dummy_run)

    return wrapper

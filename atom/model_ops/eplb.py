"""EPLB module-A runtime helpers (statistics only)."""

from __future__ import annotations

from functools import wraps
from typing import Callable, Optional

import torch
from aiter.dist.parallel_state import get_tp_group

import logging

logger = logging.getLogger("atom")

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

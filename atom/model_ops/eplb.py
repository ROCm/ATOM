"""EPLB module-A runtime helpers (statistics only)."""

from __future__ import annotations

from functools import wraps
from typing import Optional

import torch
from aiter.dist.parallel_state import get_tp_group
from atom.config import get_current_atom_config

def count_physical_load(
    topk_physical: torch.Tensor, num_physical: int
) -> torch.Tensor:
    """Count per-physical expert load for one pass.

    Invalid ids (`<0` or `>= num_physical`) are ignored.
    """
    if topk_physical.numel() == 0:
        return torch.zeros(
            num_physical, dtype=torch.int32, device=topk_physical.device
        )

    flat = topk_physical.reshape(-1).to(torch.int64)
    valid = (flat >= 0) & (flat < num_physical)
    if valid.any():
        counts = torch.bincount(flat[valid], minlength=num_physical)
    else:
        counts = torch.zeros(num_physical, dtype=torch.int64, device=flat.device)
    return counts.to(torch.int32)


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


def get_expert_load_monitor(*, enabled: bool, window_size: int) -> ExpertLoadMonitor:
    global _MONITOR
    if (
        _MONITOR is None
        or _MONITOR.enabled != enabled
        or _MONITOR.window_size != max(1, int(window_size))
    ):
        _MONITOR = ExpertLoadMonitor(enabled=enabled, window_size=window_size)
    return _MONITOR


def with_eplb_forward_monitor(fn):
    @wraps(fn)
    def wrapper(self, batch, *args, **kwargs):
        cfg = get_current_atom_config()
        if not getattr(cfg, "eplb_enable", False):
            return fn(self, batch, *args, **kwargs)
        monitor = get_expert_load_monitor(
            enabled=True, window_size=cfg.eplb_load_window_size
        )
        monitor.on_forward_start()
        try:
            return fn(self, batch, *args, **kwargs)
        finally:
            monitor.on_forward_end(getattr(batch, "is_dummy_run", False))

    return wrapper

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import torch

from atom.utils import envs

logger = logging.getLogger("atom")


@dataclass
class _LayerExpertLoadState:
    expert_load_pass: torch.Tensor
    expert_load_window: torch.Tensor
    window_step: int = 0


class ExpertLoadMonitor:
    """Lightweight EPLB-style expert-load monitor for ATOM EP path.

    This monitor tracks per-layer, per-expert token counts in two tensors:
    - ``expert_load_pass``: load accumulated in current pass window.
    - ``expert_load_window``: ring buffer of recent pass windows.

    The implementation intentionally keeps scope narrow:
    - per-rank statistics only (no cross-rank all-reduce yet),
    - collection wired to MORI dispatch output (dispatch_recv_token_num).
    """

    def __init__(self) -> None:
        self._enabled = envs.ATOM_ENABLE_EPLB_LOAD_STATS
        self._window_size = max(1, envs.ATOM_EPLB_LOAD_WINDOW_SIZE)
        self._log_interval = max(1, envs.ATOM_EPLB_LOG_INTERVAL)
        self._offline_rebalance_after_steps = max(
            0, envs.ATOM_EPLB_OFFLINE_REBALANCE_AFTER_STEPS
        )
        self._states: dict[str, _LayerExpertLoadState] = {}
        self._steps = 0
        self._real_steps = 0
        self._offline_rebalance_done = False
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def record_expert_load_pass(
        self, layer_name: str | None, dispatch_recv_token_num: torch.Tensor
    ) -> None:
        if not self._enabled or layer_name is None:
            return
        if dispatch_recv_token_num.numel() == 0:
            return

        # Keep counting on the source device to avoid per-step D2H copies.
        # We only normalize dtype/shape here.
        counts = (
            dispatch_recv_token_num.detach()
            .to(dtype=torch.int64)
            .reshape(-1)
            .contiguous()
        )

        with self._lock:
            state = self._states.get(layer_name)
            if state is None:
                pass_tensor = torch.zeros_like(counts, dtype=torch.int64)
                window = torch.zeros(
                    (self._window_size, counts.numel()),
                    dtype=torch.int64,
                    device=counts.device,
                )
                state = _LayerExpertLoadState(
                    expert_load_pass=pass_tensor,
                    expert_load_window=window,
                )
                self._states[layer_name] = state
            elif (
                state.expert_load_pass.numel() != counts.numel()
                or state.expert_load_pass.device != counts.device
            ):
                # Defensive reset for shape changes caused by model swap/reinit.
                logger.warning(
                    "EPLB monitor reset for layer=%s due to expert-size/device change "
                    "(size %d -> %d, device %s -> %s).",
                    layer_name,
                    state.expert_load_pass.numel(),
                    counts.numel(),
                    state.expert_load_pass.device,
                    counts.device,
                )
                state.expert_load_pass = torch.zeros_like(counts, dtype=torch.int64)
                state.expert_load_window = torch.zeros(
                    (self._window_size, counts.numel()),
                    dtype=torch.int64,
                    device=counts.device,
                )
                state.window_step = 0

            state.expert_load_pass.add_(counts)

    def step(self, *, is_dummy_run: bool = False) -> None:
        if not self._enabled:
            return
        with self._lock:
            if not self._states:
                return
            self._steps += 1
            if is_dummy_run:
                # Drop any load collected during dummy/warmup forwards so it
                # never leaks into the next real step's statistics.
                for state in self._states.values():
                    state.expert_load_pass.zero_()
                return

            self._real_steps += 1
            should_log = (self._real_steps % self._log_interval) == 0

            for layer_name, state in self._states.items():
                state.expert_load_window[state.window_step].copy_(state.expert_load_pass)

                if should_log:
                    load = state.expert_load_pass
                    avg_tokens = load.float().mean().item()
                    max_tokens = load.max().item()
                    balancedness = (avg_tokens / max_tokens) if max_tokens > 0 else 0.0
                    logger.info(
                        "EPLB load stats layer=%s step=%d avg_tokens=%.2f "
                        "max_tokens=%d balancedness=%.4f",
                        layer_name,
                        self._real_steps,
                        avg_tokens,
                        int(max_tokens),
                        balancedness,
                    )

                state.expert_load_pass.zero_()
                state.window_step += 1
                if state.window_step >= self._window_size:
                    state.window_step = 0

            if (
                not self._offline_rebalance_done
                and self._offline_rebalance_after_steps > 0
                and self._real_steps >= self._offline_rebalance_after_steps
            ):
                self._offline_rebalance_done = True
                self._log_offline_rebalance_plan()

    def trigger_offline_rebalance(self, reason: str = "manual") -> None:
        if not self._enabled:
            logger.info(
                "EPLB offline rebalance trigger skipped: load stats monitor is disabled."
            )
            return
        with self._lock:
            self._log_offline_rebalance_plan(reason=reason)

    def _log_offline_rebalance_plan(self, reason: str = "auto") -> None:
        logger.info(
            "EPLB offline rebalance planning trigger (%s) at step=%d real_step=%d.",
            reason,
            self._steps,
            self._real_steps,
        )
        for layer_name, state in self._states.items():
            window_sum = state.expert_load_window.sum(dim=0)
            if window_sum.numel() == 0:
                continue
            top_k = min(8, window_sum.numel())
            top_vals, top_idx = torch.topk(window_sum, k=top_k, largest=True, sorted=True)
            bottom_vals, bottom_idx = torch.topk(
                window_sum, k=top_k, largest=False, sorted=True
            )
            avg_tokens = window_sum.float().mean().item()
            max_tokens = window_sum.max().item()
            balancedness = (avg_tokens / max_tokens) if max_tokens > 0 else 0.0
            logger.info(
                "EPLB offline plan layer=%s avg_tokens=%.2f max_tokens=%d "
                "balancedness=%.4f hot_experts=%s cold_experts=%s",
                layer_name,
                avg_tokens,
                int(max_tokens),
                balancedness,
                list(zip(top_idx.tolist(), top_vals.tolist())),
                list(zip(bottom_idx.tolist(), bottom_vals.tolist())),
            )
        logger.info(
            "EPLB offline plan generated. Note: automatic expert weight remap/transfer "
            "is not applied in this pass."
        )


_EXPERT_LOAD_MONITOR = ExpertLoadMonitor()


def get_expert_load_monitor() -> ExpertLoadMonitor:
    return _EXPERT_LOAD_MONITOR

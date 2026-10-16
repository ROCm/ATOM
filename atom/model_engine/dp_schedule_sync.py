# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
DPScheduleSync — the scheduler's one cross-DP collective per tick.

Under DP attention each rank schedules on its own, but the forwards run in
lockstep, so two scheduler features read the other ranks before a rank builds
its batch:

- the PrefillDelayer (``ATOM_ENABLE_PREFILL_DELAYER``) decides FIRE/HOLD from
  its decision counts summed over the ranks;
- DP prefill cost balancing (``ATOM_DP_PREFILL_BALANCE``) aims every prefill
  step at one cost target, from every rank's estimated chunk cost.

The two are independent: either runs without the other. When both are on they
share this one ``all_reduce(SUM)`` over a cpu int64 buffer (gloo-safe), so the
second costs no extra collective:

  slot i < decision_slots       the delayer's counts (``PrefillDelayer.local_counts``)
  slot decision_slots + r       rank r's chunk cost, one-hot: the sum hands every
                                rank every cost

A cost goes in rounded up to a whole number. Truncating it put a rank's target
half a unit below its own cost whenever that rank set the target (alone in the
step, or costliest within the slack), so the rank cut a token off its chunk,
which the chunk grid then floored to a smaller chunk and an extra prefill step.

Every rank must call ``exchange`` exactly once per scheduler tick, in lockstep.
"""

from __future__ import annotations

import logging
import math

import torch

logger = logging.getLogger(__name__)


class DPScheduleSync:
    __slots__ = ("_rank", "cpu_group", "decision_slots", "dp_size", "prefill_costs")

    def __init__(
        self,
        cpu_group,
        dp_size: int,
        *,
        decision_slots: int,
        prefill_costs: bool,
    ):
        if not decision_slots and not prefill_costs:
            raise ValueError(
                "DPScheduleSync carries the delayer's counts, prefill costs, or "
                "both; with neither on no rank needs the collective"
            )
        self.cpu_group = cpu_group
        self.dp_size = dp_size
        self.decision_slots = decision_slots
        self.prefill_costs = prefill_costs
        # The DP group is stateless (no default process group): ask it directly.
        self._rank = cpu_group.rank()
        if not 0 <= self._rank < dp_size:
            raise ValueError(f"DP rank {self._rank} outside [0, {dp_size})")
        logger.info(
            f"DPScheduleSync initialized: dp_size={dp_size} rank={self._rank} "
            f"decision_slots={decision_slots} prefill_costs={prefill_costs}"
        )

    def exchange(
        self, counts: list[int] | None, prefill_cost: float = 0.0
    ) -> tuple[list[int] | None, list[float]]:
        """Sum this tick's per-rank fields over the DP ranks.

        Args:
            counts: this rank's delayer decision counts, ``decision_slots`` of
                them; None when the delayer is off.
            prefill_cost: this rank's estimated chunk cost, 0 when it has no
                prefill work. Ignored unless the sync carries costs.

        Returns the summed counts (None when the delayer is off) and every
        rank's cost, indexed by rank (empty unless the sync carries costs).
        A failed collective (a peer is down) returns ``(None, [])``: the delayer
        then admits, nothing balances, and the guarded DP-state barrier drives
        shutdown.
        """
        n = self.decision_slots
        assert (counts is None) == (n == 0) and (counts is None or len(counts) == n)
        values = list(counts) if n else []
        if self.prefill_costs:
            costs = [0] * self.dp_size
            costs[self._rank] = math.ceil(prefill_cost)
            values += costs
        buf = torch.tensor(values, dtype=torch.int64)
        try:
            self._sum_over_ranks(buf)
        except RuntimeError:
            logger.warning(
                "DP schedule sync all_reduce failed (peer down?); admitting "
                "prefill unbalanced and deferring to the DP-state shutdown barrier."
            )
            return None, []
        # One readback for all slots (a single .tolist() beats per-slot .item()
        # boundary crossings on this per-tick hot path).
        summed = buf.tolist()
        return (summed[:n] if n else None), [float(c) for c in summed[n:]]

    def _sum_over_ranks(self, buf: torch.Tensor) -> None:
        torch.distributed.all_reduce(
            buf, op=torch.distributed.ReduceOp.SUM, group=self.cpu_group
        )

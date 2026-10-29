# SPDX-License-Identifier: MIT
"""The TP and DP CPU groups DSpark's planner and online estimators talk over.

Every DSpark collective goes through one `DSparkComm`: the joint DP plan, the
online SPS and STS publications and the startup SPS reduction. It wraps the
existing CPU (gloo) groups; nothing here creates a communicator or touches a
device. Outside an initialized process group (unit tests) every group is a
group of one.
"""

from __future__ import annotations

import hashlib
import json

import torch
import torch.distributed as dist


def _size_rank(group) -> tuple[int, int]:
    if group is None or not dist.is_initialized():
        return 1, 0
    return dist.get_world_size(group), dist.get_rank(group)


class DSparkComm:
    def __init__(self, tp_group=None, dp_group=None):
        if tp_group is None and dp_group is None and dist.is_initialized():
            from aiter.dist.parallel_state import get_dp_group, get_tp_group

            tp_group, dp_group = get_tp_group().cpu_group, get_dp_group().cpu_group
        self.tp_group, self.dp_group = tp_group, dp_group
        self.tp_size, self.tp_rank = _size_rank(tp_group)
        self.dp_size, self.dp_rank = _size_rank(dp_group)

    @property
    def tp_leader(self) -> bool:
        return self.tp_rank == 0

    @staticmethod
    def _gather(value: torch.Tensor, group, size: int) -> torch.Tensor:
        if size == 1:
            return value.unsqueeze(0)
        result = [torch.empty_like(value) for _ in range(size)]
        dist.all_gather(result, value, group=group)
        return torch.stack(result)

    def gather_tp(self, value: torch.Tensor) -> torch.Tensor:
        """[tp_size, *value.shape]."""
        return self._gather(value, self.tp_group, self.tp_size)

    def gather_dp(self, value: torch.Tensor) -> torch.Tensor:
        """[dp_size, *value.shape]."""
        return self._gather(value, self.dp_group, self.dp_size)

    def broadcast_tp(self, value: torch.Tensor) -> None:
        """In place, from the TP leader."""
        if self.tp_size > 1:
            dist.broadcast(
                value, src=dist.get_global_rank(self.tp_group, 0), group=self.tp_group
            )

    def _all_reduce(self, value: torch.Tensor, op) -> torch.Tensor:
        for group, size in (
            (self.tp_group, self.tp_size),
            (self.dp_group, self.dp_size),
        ):
            if size > 1:
                dist.all_reduce(value, op=op, group=group)
        return value

    def all_max(self, values) -> list[float]:
        """Element-wise MAX over every TP and DP worker."""
        t = torch.tensor(list(values), dtype=torch.float64)
        return self._all_reduce(t, dist.ReduceOp.MAX).tolist()

    def all_true(self, flag: bool) -> bool:
        """Whether `flag` holds on every TP and DP worker."""
        t = torch.tensor([int(flag)], dtype=torch.int64)
        return bool(self._all_reduce(t, dist.ReduceOp.MIN).item())

    def agree(self, value) -> None:
        """Fail on every worker unless all of them hold the same `value` (JSON).

        Called before a variable-size exchange, so a divergent configuration
        stops at a clear error instead of a mismatched collective.
        """
        digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).digest()
        local = torch.tensor(list(digest), dtype=torch.uint8)
        every = self.gather_dp(self.gather_tp(local))
        if not torch.all(every == local):
            raise ValueError(
                "DSpark workers disagree on profile or graph configuration"
            )

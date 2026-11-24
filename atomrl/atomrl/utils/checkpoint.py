# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Checkpoint save/load utilities."""

from __future__ import annotations

import logging
import re
import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def create_checkpoint_control_group(world_size: int) -> Any | None:
    """Create a CPU control group isolated from model RCCL collectives."""
    if not torch.distributed.is_initialized():
        return None
    return torch.distributed.new_group(
        ranks=list(range(int(world_size))),
        backend="gloo",
    )


def checkpoint_rank_phases(
    rank: int,
    world_size: int,
    *,
    group: Any | None = None,
) -> list[list[int]]:
    """Group checkpoint ranks into one concurrent writer per physical node."""
    if not torch.distributed.is_initialized():
        return [[int(rank)]]

    hostnames: list[str | None] = [None] * int(world_size)
    torch.distributed.all_gather_object(
        hostnames,
        socket.gethostname(),
        group=group,
    )
    ranks_by_host: dict[str, list[int]] = {}
    for global_rank, hostname in enumerate(hostnames):
        ranks_by_host.setdefault(str(hostname), []).append(global_rank)

    phase_count = max(len(ranks) for ranks in ranks_by_host.values())
    return [
        [ranks[phase] for ranks in ranks_by_host.values() if phase < len(ranks)]
        for phase in range(phase_count)
    ]


def run_checkpoint_phase(
    rank: int,
    world_size: int,
    action: Callable[[], None] | None,
    *,
    group: Any | None = None,
) -> None:
    """Run one rank-local checkpoint action and propagate failures to every rank."""
    local_error: Exception | None = None
    if action is not None:
        try:
            action()
        except (OSError, RuntimeError) as exc:
            local_error = exc

    if not torch.distributed.is_initialized():
        if local_error is not None:
            raise local_error
        return

    local_failure = (
        {
            "rank": int(rank),
            "type": type(local_error).__name__,
            "message": str(local_error),
        }
        if local_error is not None
        else None
    )
    failures: list[dict[str, Any] | None] = [None] * int(world_size)
    torch.distributed.all_gather_object(
        failures,
        local_failure,
        group=group,
    )
    failed = [failure for failure in failures if failure is not None]
    if not failed:
        return

    details = "; ".join(
        f"rank {failure['rank']} {failure['type']}: {failure['message']}"
        for failure in failed
    )
    error = RuntimeError(f"checkpoint phase failed: {details}")
    if local_error is not None:
        raise error from local_error
    raise error


class CheckpointManager:
    """Filesystem checkpoint I/O with step-aware filenames."""

    @staticmethod
    def save(state_dict: dict[str, Any], path: str | Path, step: int) -> None:
        """Persist ``state_dict`` together with ``step`` metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"step": int(step), "state_dict": state_dict}
        torch.save(payload, path)
        logger.info("Saved checkpoint to %s (step=%d)", path, step)

    @staticmethod
    def load(path: str | Path) -> dict[str, Any]:
        """Load a checkpoint payload produced by :meth:`save`."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict) or "state_dict" not in payload:
            raise ValueError(f"Invalid checkpoint format at {path}")
        logger.info("Loaded checkpoint from %s (step=%s)", path, payload.get("step"))
        return payload

    @staticmethod
    def get_latest(checkpoint_dir: str | Path) -> str | None:
        """Return the path to the highest-step ``checkpoint_*.pt`` file, if any."""
        root = Path(checkpoint_dir)
        if not root.is_dir():
            return None
        best: tuple[int, Path] | None = None
        pattern = re.compile(r"checkpoint_(\d+)\.pt$")
        for p in root.iterdir():
            m = pattern.match(p.name)
            if not m:
                continue
            step = int(m.group(1))
            if best is None or step > best[0]:
                best = (step, p)
        return str(best[1]) if best else None

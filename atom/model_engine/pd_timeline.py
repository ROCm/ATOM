# SPDX-License-Identifier: MIT
"""Per-request P/D timeline, one log line per request and role.

Marks are wall-clock so a prefill's and a decode's lines join on
``transfer_id`` across nodes. Offsets in the line are ms after
``seq.arrive_time``, which each role's API process stamps on arrival.
"""

from __future__ import annotations

import logging
import time

from atom.utils import envs

logger = logging.getLogger("atom")

_ENABLED = envs.ATOM_PD_TIMELINE


def mark(seq, key: str, *, last: bool = False) -> None:
    """Stamp `key` on `seq`: its first occurrence, or its latest with `last`."""
    if not _ENABLED or seq is None:
        return
    marks = seq.__dict__.setdefault("_pd_timeline", {})
    if last or key not in marks:
        marks[key] = time.time()


def emit(seq, role: str) -> None:
    if not _ENABLED or seq is None:
        return
    marks = seq.__dict__.pop("_pd_timeline", None)
    if not marks:
        return
    t0 = seq.arrive_time or min(marks.values())
    params = seq.kv_transfer_params or {}
    logger.info(
        "[PD-TL] role=%s rid=%s tid=%s isl=%d cached=%d t0=%.3f %s",
        role,
        seq.id,
        params.get("transfer_id"),
        seq.num_prompt_tokens,
        seq.num_cached_tokens,
        t0,
        " ".join(f"{k}={1e3 * (t - t0):.1f}" for k, t in marks.items()),
    )

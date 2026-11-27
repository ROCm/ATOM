# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1 on the ATOM plugin, SGLang 0.5.22: copy an EXTEND's PAGE
rows to the host right after SGLang allocated them.

``ScheduleBatch.prepare_for_extend`` calls ``alloc_for_extend(self)``, which
allocates the batch's KV pages and writes them into ``req_to_token`` on the
scheduler's current stream (under overlap, the schedule stream, which does
not wait for the forward stream). The wrapper calls the original unchanged
and then queues `deepseek_v41_bridge.read_v41_extend_rows_at_alloc`: a D2H of
those rows on the same stream, after that write (`ExtendRowRead`). The
staging of the batch takes it instead of reading the rows back behind the
step in flight. It changes no SGLang value: the tuple it returns is the
original's.

Switch: ``ATOM_SGLANG_V41_EXTEND_ROWS_AT_ALLOC=0`` (`deepseek_v41_envs`) keeps
the read-back, which the staging takes anyway for a batch the copy does not
match (a MIXED batch, one that never ran), so no code stays alive for the
switch alone. Shape guard: ``schedule_batch.alloc_for_extend(batch)``
returning ``(out_cache_loc, req_pool_indices_device, req_pool_indices_cpu)``;
anything else keeps the read-back and logs why (the plugin runs on SGLang
0.5.22 only: `sglang_version_guard` refuses any other at launch, before this
installs). With radix cache on the copy is
taken too: it holds a hit's prefix pages, which the allocation writes into the
row with the extend's own, and a radix checkpoint of one of its rows after the
copy refuses it (`ExtendRowRead.take`, the slot's `HostPageRows` generation).
"""

from __future__ import annotations

import inspect
import logging
from functools import wraps

from atom.plugin.sglang import deepseek_v41_envs
from atom.plugin.sglang.patches.sglang_version_guard import (
    missing_source_needles,
    sglang_version,
)

logger = logging.getLogger("atom")

# The returned tuple, as `alloc_for_extend` spells it.
_RETURN_NEEDLE = "return out_cache_loc, req_pool_indices_device, req_pool_indices_cpu"


def _upstream_shape_error(schedule_batch) -> str | None:
    function = getattr(schedule_batch, "alloc_for_extend", None)
    if function is None:
        return "schedule_batch.alloc_for_extend is missing"
    function = inspect.unwrap(function)
    if list(inspect.signature(function).parameters) != ["batch"]:
        return "alloc_for_extend no longer takes (batch)"
    missing = missing_source_needles(function, (_RETURN_NEEDLE,))
    if missing is None:
        return "alloc_for_extend source is unavailable"
    if missing:
        return "alloc_for_extend no longer returns (loc, indices, indices_cpu)"
    return None


def install_v41_extend_rows_at_alloc_patch() -> bool:
    """Install the row copy after ``alloc_for_extend`` (idempotent). Returns
    whether it is active."""
    from atom.plugin.sglang import deepseek_v41_bridge as bridge

    if not deepseek_v41_envs.ATOM_SGLANG_V41_EXTEND_ROWS_AT_ALLOC:
        logger.info(
            "ATOM DeepSeek-V4.1: EXTEND PAGE rows read back at staging "
            "(ATOM_SGLANG_V41_EXTEND_ROWS_AT_ALLOC=0)"
        )
        return False
    from sglang.srt.managers import schedule_batch

    original = schedule_batch.alloc_for_extend
    if getattr(original, "_atom_v41_extend_rows", False):
        return True
    reason = _upstream_shape_error(schedule_batch)
    if reason is not None:
        logger.warning(
            "ATOM DeepSeek-V4.1: EXTEND PAGE rows at allocation not installed "
            "(%s); an EXTEND reads its rows back at staging",
            reason,
        )
        return False

    @wraps(original)
    def alloc_for_extend(batch):
        out = original(batch)
        _, req_pool_indices, req_pool_indices_cpu = out
        bridge.read_v41_extend_rows_at_alloc(
            batch, req_pool_indices, req_pool_indices_cpu
        )
        return out

    alloc_for_extend._atom_v41_extend_rows = True
    schedule_batch.alloc_for_extend = alloc_for_extend
    logger.info(
        "ATOM DeepSeek-V4.1: EXTEND PAGE rows copied at allocation (after "
        "schedule_batch.alloc_for_extend; sglang %s)",
        sglang_version(),
    )
    return True

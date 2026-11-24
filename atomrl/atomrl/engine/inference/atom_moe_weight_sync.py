# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Coverage guard for AtomRL-to-ATOM weight synchronization.

The trainer sends original fused-MoE parameter names and ATOM's public weight
updater resolves them and applies the device layout.

``assert_bucket_fully_applied`` validates the per-rank updated counts so a
partial or stale weight sync fails loudly.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


# The trainer's fused expert leaf -> the ATOM FusedMoE parameter it belongs in.
# Both are (experts, out, in) with matching dims, and w13's first half along the
# intermediate dim is the gate projection, the second half the up projection --
# the same convention transformers uses, so no chunk/reorder is needed, only the
# rename. ATOM has no nested container below ``experts`` (vLLM >= 0.22 grew a
# ``routed_experts`` segment; this fork did not), so the prefix is untouched.
def assert_bucket_fully_applied(
    responses: Any,
    bucket_meta: dict[str, dict[str, Any]],
    *,
    context: str = "ipc",
    exact: bool = True,
) -> None:
    """Fail when an engine did not apply every tensor in the bucket.

    ATOM reports ``updated`` per bucket and counts anything it could not place
    in ``skipped``, which it only logs at debug level -- so a naming drift shows
    up as a slowly growing train/rollout divergence rather than a crash. A BF16
    trainer sends no quantization scales, so every tensor in the bucket must be
    accounted for and the counts have to match exactly.

    ``exact=False`` turns a shortfall into a warning, for the one case where
    ``updated`` is legitimately lower than the bucket: an FP8 rollout. There
    ATOM's ``_apply_packed_weight`` accumulates the shards of a fused parameter
    in a float32 buffer and requantizes once the last one lands, so q/k/v_proj
    count as a single update and the two that arrived first count as nothing at
    all -- ``updated`` under-reports by the number of shards still pending when
    the bucket closes. ATOM returns only that one number, so the receiver cannot
    tell those apart from genuinely skipped tensors. Measured on example 4 (8B
    ATOM FP8): 28 of 39, the 11 being q/k/v and gate/up shards of the three
    layers the bucket straddles.

    ``ATOMRL_WEIGHT_SYNC_CHECK`` selects ``error`` (default), ``warn`` or
    ``off``, matching the vLLM path's knob.
    """
    mode = os.environ.get("ATOMRL_WEIGHT_SYNC_CHECK", "error").lower()
    if mode == "off":
        return

    expected = len(bucket_meta)
    applied = _updated_counts(responses)
    if not applied:
        logger.warning(
            "ATOM weight sync (%s): no engine reported an updated count; "
            "coverage unchecked",
            context,
        )
        return

    short = [count for count in applied if count != expected]
    if not short:
        return

    message = (
        f"ATOM weight sync ({context}) applied {min(short)}/{expected} tensors of a "
        f"bucket on {len(short)} of {len(applied)} engine(s); the rest were skipped "
        "as unrecognised names, so the rollout engine is now serving a mix of "
        "current and stale weights. Fused MoE experts are the usual cause -- see "
        "atomrl/engine/inference/atom_moe_weight_sync.py. Set "
        "ATOMRL_WEIGHT_SYNC_CHECK=warn to downgrade this to a log line."
    )
    if mode == "warn" or not exact:
        logger.warning(message)
        return
    raise RuntimeError(message)


def _updated_counts(responses: Any) -> list[int]:
    """Pull the ``updated`` counts out of ``broadcast_utility_command_sync``.

    Each engine answers with ``{"cmd": ..., "result": r}``, where ``r`` is what
    ATOM's runner manager returned -- an int for a single runner, a list of them
    when the engine fans out over several.
    """
    counts: list[int] = []
    for response in responses or []:
        result = response.get("result") if isinstance(response, dict) else response
        for value in result if isinstance(result, (list, tuple)) else [result]:
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            counts.append(int(value))
    return counts

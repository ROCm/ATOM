# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1 DSpark on the ATOM plugin, SGLang 0.5.22: prefetch the PAGE
rows a decode round's verify will need, before the round's draft is launched.

``DSparkWorkerV2._forward_decode`` calls ``alloc_verify_window`` right before
``DraftBlockProposer.propose``: the last point of the round that has the
ScheduleBatch (host slots, request ids, SGLang's KV reservation
``spec_info.nxt_kv_lens_cpu``) and nothing of the draft enqueued. The wrapper
calls the original unchanged and then starts
`deepseek_v41_bridge.prefetch_v41_dspark_verify_rows` (an async side-stream
copy of the rows `HostPageRows` does not cover; see `VerifyRowPrefetch`) on
the runner ``_forward_decode`` passes, the target's (the only caller; the
patch is installed by the V4.1 DSpark target surface alone). It changes no
SGLang value: the window it returns is the original's.

Switch: ``ATOM_SGLANG_V41_DSPARK_PREFETCH_ROWS=0`` (the plugin registry,
`deepseek_v41_envs`) keeps the read-back, which the verify needs anyway for
the rows a prefetch does not cover, so no code stays alive for the switch
alone. Shape guard: ``alloc_verify_window(*, batch, ..., model_runner,
...)``; anything else keeps the read-back and logs why. The plugin runs on
SGLang 0.5.22 only (`sglang_version_guard` refuses any other at launch,
before this installs).
"""

from __future__ import annotations

import inspect
import logging
from functools import wraps

from atom.plugin.sglang import deepseek_v41_envs
from atom.plugin.sglang.patches.sglang_version_guard import sglang_version

logger = logging.getLogger("atom")


def _upstream_shape_error(module) -> str | None:
    function = getattr(module, "alloc_verify_window", None)
    if function is None:
        return "dspark_worker_v2.alloc_verify_window is missing"
    params = inspect.signature(function).parameters
    for name in ("batch", "model_runner"):
        if name not in params:
            return f"alloc_verify_window no longer takes {name!r}"
    return None


def install_dspark_verify_rows_prefetch_patch() -> bool:
    """Install the prefetch after ``alloc_verify_window`` (idempotent).
    Returns whether it is active."""
    from atom.plugin.sglang import deepseek_v41_bridge as bridge

    if not deepseek_v41_envs.ATOM_SGLANG_V41_DSPARK_PREFETCH_ROWS:
        logger.info(
            "ATOM DeepSeek-V4.1 DSpark: verify PAGE rows not prefetched "
            "(ATOM_SGLANG_V41_DSPARK_PREFETCH_ROWS=0)"
        )
        return False
    from sglang.srt.speculative.dspark_components import dspark_worker_v2

    original = dspark_worker_v2.alloc_verify_window
    if getattr(original, "_atom_v41_prefetch_rows", False):
        return True
    reason = _upstream_shape_error(dspark_worker_v2)
    if reason is not None:
        logger.warning(
            "ATOM DeepSeek-V4.1 DSpark: verify PAGE row prefetch not installed "
            "(%s); a verify entering a new page reads its rows back",
            reason,
        )
        return False

    @wraps(original)
    def alloc_verify_window(*args, **kwargs):
        window = original(*args, **kwargs)
        bridge.prefetch_v41_dspark_verify_rows(
            kwargs["batch"],
            kwargs["model_runner"].req_to_token_pool.req_to_token,
            bridge.get_deepseek_v41_builder().block_size,
        )
        return window

    alloc_verify_window._atom_v41_prefetch_rows = True
    dspark_worker_v2.alloc_verify_window = alloc_verify_window
    logger.info(
        "ATOM DeepSeek-V4.1 DSpark: verify PAGE rows prefetched before the "
        "draft launch (after dspark_worker_v2.alloc_verify_window; sglang %s)",
        sglang_version(),
    )
    return True

# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1 DSpark on the ATOM plugin, SGLang 0.5.22: prefetch the PAGE
rows a decode round's verify will need, before the round's draft is launched.

``DSparkWorkerV2._forward_decode`` calls ``alloc_verify_window`` right before
``DraftBlockProposer.propose``: the last point of the round that has the
ScheduleBatch (host slots, request ids, SGLang's KV reservation
``spec_info.nxt_kv_lens_cpu``) and nothing of the draft enqueued. The wrapper
calls the original unchanged and then starts
`deepseek_v41_bridge.prefetch_v41_dspark_verify_rows` (an async side-stream
copy of the rows `HostPageRows` does not cover; see `VerifyRowPrefetch`) for
the V4.1 target runner only. It changes no SGLang value: the window it
returns is the original's.

Kill switch: ``ATOM_SGLANG_V41_DSPARK_PREFETCH_ROWS=0``. Version guard: SGLang
0.5.22 with ``alloc_verify_window(*, batch, ..., model_runner, ...)``; any
other version keeps the read-back and logs why.
"""

from __future__ import annotations

import inspect
import logging
from functools import wraps

logger = logging.getLogger("atom")

SUPPORTED_SGLANG_VERSIONS = ("0.5.22",)


def _sglang_version() -> str:
    try:
        import sglang

        return str(getattr(sglang, "__version__", "?"))
    except Exception:  # noqa: BLE001 - a version string
        return "?"


def _upstream_shape_error(module) -> str | None:
    version = _sglang_version()
    if version not in SUPPORTED_SGLANG_VERSIONS:
        return f"sglang {version} not in {SUPPORTED_SGLANG_VERSIONS}"
    function = getattr(module, "alloc_verify_window", None)
    if function is None:
        return "dspark_worker_v2.alloc_verify_window is missing"
    params = inspect.signature(function).parameters
    for name in ("batch", "model_runner"):
        if name not in params:
            return f"alloc_verify_window no longer takes {name!r}"
    return None


def _is_v41_target_runner(model_runner) -> bool:
    pool = getattr(model_runner, "token_to_kv_pool", None)
    return bool(getattr(pool, "is_atom_v41_proxy_pool", False)) and not bool(
        getattr(pool, "is_atom_v41_draft_proxy", False)
    )


def install_dspark_verify_rows_prefetch_patch() -> bool:
    """Install the prefetch after ``alloc_verify_window`` (idempotent).
    Returns whether it is active."""
    from atom.plugin.sglang import deepseek_v41_bridge as bridge

    if not bridge.v41_dspark_prefetch_rows_enabled():
        logger.info(
            "ATOM DeepSeek-V4.1 DSpark: verify PAGE rows not prefetched (%s=0)",
            bridge.V41_DSPARK_PREFETCH_ROWS_ENV,
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
        model_runner = kwargs.get("model_runner")
        if _is_v41_target_runner(model_runner):
            bridge.prefetch_v41_dspark_verify_rows(
                kwargs["batch"],
                model_runner.req_to_token_pool.req_to_token,
                bridge.get_deepseek_v41_builder().block_size,
            )
        return window

    alloc_verify_window._atom_v41_prefetch_rows = True
    dspark_worker_v2.alloc_verify_window = alloc_verify_window
    logger.info(
        "ATOM DeepSeek-V4.1 DSpark: verify PAGE rows prefetched before the "
        "draft launch (after dspark_worker_v2.alloc_verify_window; sglang %s; "
        "%s=0 to disable)",
        _sglang_version(),
        bridge.V41_DSPARK_PREFETCH_ROWS_ENV,
    )
    return True

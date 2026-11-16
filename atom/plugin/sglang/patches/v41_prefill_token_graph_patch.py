# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1 on the ATOM plugin, SGLang 0.5.22: record the target's
prefill token graphs once SGLang has captured every graph of its own.

``Scheduler.init_all_cuda_graphs`` captures the target's graphs, then the
draft worker's (which checks the memory left first). The wrapper calls the
original unchanged and then, on a V4.1 target runner with graphs on, records
the runtime model's token-graph unit at the prefill buckets
(`deepseek_v41_bridge.record_deepseek_v41_prefill_token_graphs`, native's
`ModelRunner._record_tr_graphs`), which every EXTEND that fits a bucket then
replays. With graphs off (``--disable-cuda-graph``) prefills stay eager.

Kill switch: ``ATOM_SGLANG_V41_PREFILL_TOKEN_GRAPHS=0``. Version guard: SGLang
0.5.22 with ``Scheduler.init_all_cuda_graphs(self)`` and a ``tp_worker``
holding the target ``model_runner``; any other version keeps prefills eager
and logs why.
"""

from __future__ import annotations

import inspect
import logging
from functools import wraps

import torch

logger = logging.getLogger("atom")

SUPPORTED_SGLANG_VERSIONS = ("0.5.22",)


def _sglang_version() -> str:
    try:
        import sglang

        return str(getattr(sglang, "__version__", "?"))
    except Exception:  # noqa: BLE001 - a version string
        return "?"


def _upstream_shape_error(scheduler_cls) -> str | None:
    version = _sglang_version()
    if version not in SUPPORTED_SGLANG_VERSIONS:
        return f"sglang {version} not in {SUPPORTED_SGLANG_VERSIONS}"
    function = getattr(scheduler_cls, "init_all_cuda_graphs", None)
    if function is None:
        return "Scheduler.init_all_cuda_graphs is missing"
    if list(inspect.signature(inspect.unwrap(function)).parameters) != ["self"]:
        return "Scheduler.init_all_cuda_graphs no longer takes (self)"
    return None


def v41_target_runner_with_graphs(scheduler):
    """The scheduler's V4.1 target model runner when it captured graphs (the
    one whose proxy pool backs the builder's cache), else None."""
    from atom.plugin.sglang import deepseek_v41_bridge as bridge

    runner = getattr(getattr(scheduler, "tp_worker", None), "model_runner", None)
    if runner is None or getattr(runner, "is_draft_worker", False):
        return None
    pool = getattr(runner, "token_to_kv_pool", None)
    if not getattr(pool, "is_atom_v41_proxy_pool", False) or getattr(
        pool, "is_atom_v41_draft_proxy", False
    ):
        return None
    if bridge._V41_BUILDER is None or bridge._V41_BUILDER.cache is None:
        return None
    if getattr(runner, "decode_cuda_graph_runner", None) is None:
        return None
    return runner


def install_v41_prefill_token_graph_patch() -> bool:
    """Install the recording after ``init_all_cuda_graphs`` (idempotent).
    Returns whether it is active."""
    from atom.plugin.sglang import deepseek_v41_bridge as bridge

    if not bridge.v41_prefill_token_graphs_enabled():
        logger.info(
            "ATOM DeepSeek-V4.1: prefill token graphs not recorded (%s=0)",
            bridge.V41_PREFILL_TOKEN_GRAPHS_ENV,
        )
        return False
    from sglang.srt.managers import scheduler as scheduler_module

    cls = scheduler_module.Scheduler
    original = cls.init_all_cuda_graphs
    if getattr(original, "_atom_v41_prefill_token_graphs", False):
        return True
    reason = _upstream_shape_error(cls)
    if reason is not None:
        logger.warning(
            "ATOM DeepSeek-V4.1: prefill token graphs not installed (%s); "
            "prefills run eagerly",
            reason,
        )
        return False

    @wraps(original)
    def init_all_cuda_graphs(self):
        original(self)
        runner = v41_target_runner_with_graphs(self)
        if runner is not None:
            with torch.no_grad():
                bridge.record_deepseek_v41_prefill_token_graphs(runner)

    init_all_cuda_graphs._atom_v41_prefill_token_graphs = True
    cls.init_all_cuda_graphs = init_all_cuda_graphs
    return True

# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1 on the ATOM plugin, SGLang 0.5.22: record the target's
prefill token graphs once SGLang has captured every graph of its own, and
warm the eager prefill path up before the first request.

``Scheduler.init_all_cuda_graphs`` captures the target's graphs, then the
draft worker's (which checks the memory left first); SGLang calls it with
graphs off too (``--disable-cuda-graph``), and then captures nothing. The
wrapper calls the original unchanged and then, on a V4.1 target runner that
captured decode graphs, records the runtime model's token-graph unit at the
prefill buckets (`deepseek_v41_bridge.record_deepseek_v41_prefill_token_graphs`,
native's `ModelRunner._record_tr_graphs`), which every EXTEND that fits a
bucket then replays. With graphs off prefills stay eager. Then, graphs or
not, it runs one eager prefill at the chunk budget
(`deepseek_v41_bridge.warm_deepseek_v41_eager_prefill`, native's
`ModelRunner.warmup_model`).

Switches (`deepseek_v41_envs`), each on by default and each read on its own
when SGLang's graphs are done: ``ATOM_SGLANG_V41_PREFILL_TOKEN_GRAPHS`` and
``ATOM_SGLANG_V41_PREFILL_WARMUP``. Shape guard:
``Scheduler.init_all_cuda_graphs(self)``; anything else keeps prefills eager
and logs why. The plugin runs on SGLang 0.5.22 only (`sglang_version_guard`
refuses any other at launch, before this installs).
"""

from __future__ import annotations

import inspect
import logging
from functools import wraps

import torch

from atom.plugin.sglang import deepseek_v41_envs

logger = logging.getLogger("atom")


def _upstream_shape_error(scheduler_cls) -> str | None:
    function = getattr(scheduler_cls, "init_all_cuda_graphs", None)
    if function is None:
        return "Scheduler.init_all_cuda_graphs is missing"
    if list(inspect.signature(inspect.unwrap(function)).parameters) != ["self"]:
        return "Scheduler.init_all_cuda_graphs no longer takes (self)"
    return None


def install_v41_prefill_token_graph_patch() -> bool:
    """Install the hook after ``init_all_cuda_graphs`` (idempotent). Returns
    whether it is active."""
    from sglang.srt.managers import scheduler as scheduler_module

    from atom.plugin.sglang import deepseek_v41_bridge as bridge

    cls = scheduler_module.Scheduler
    original = cls.init_all_cuda_graphs
    if getattr(original, "_atom_v41_prefill_token_graphs", False):
        return True
    reason = _upstream_shape_error(cls)
    if reason is not None:
        logger.warning(
            "ATOM DeepSeek-V4.1: prefill token graphs and warm-up not installed "
            "(%s); prefills run eagerly",
            reason,
        )
        return False

    @wraps(original)
    def init_all_cuda_graphs(self):
        original(self)
        # The V4.1 target's runner: the patch is installed only where the
        # V4.1 model is prepared (`install_deepseek_v41_proxy_pool_patch`),
        # and its proxy pool and builder exist once SGLang's graphs are
        # captured.
        runner = self.tp_worker.model_runner
        with torch.no_grad():
            if not deepseek_v41_envs.ATOM_SGLANG_V41_PREFILL_TOKEN_GRAPHS:
                logger.info(
                    "ATOM DeepSeek-V4.1: prefill token graphs not recorded "
                    "(ATOM_SGLANG_V41_PREFILL_TOKEN_GRAPHS=0)"
                )
            elif runner.decode_cuda_graph_runner is None:
                logger.info(
                    "ATOM DeepSeek-V4.1: prefill token graphs not recorded (no "
                    "decode CUDA graphs were captured); prefills run eagerly"
                )
            else:
                bridge.record_deepseek_v41_prefill_token_graphs(runner)
            # One eager prefill at the chunk budget, as native's
            # `warmup_model`: the first served chunk-sized EXTEND would
            # otherwise pay the eager path's first-use costs.
            if deepseek_v41_envs.ATOM_SGLANG_V41_PREFILL_WARMUP:
                bridge.warm_deepseek_v41_eager_prefill(runner)

    init_all_cuda_graphs._atom_v41_prefill_token_graphs = True
    cls.init_all_cuda_graphs = init_all_cuda_graphs
    return True

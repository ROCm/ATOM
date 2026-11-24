# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Reconcile megatron-core's ``te_general_gemm`` with newer TransformerEngine.

``megatron.core.extensions.transformer_engine.te_general_gemm`` calls TE's
``general_gemm`` with ``workspace=get_workspace()``. TE has since dropped that
parameter and allocates the cuBLAS workspace itself inside ``general_gemm``
(``get_cublas_workspace``), so on an image that pairs an older megatron-core with
a newer TE -- vllm/vime-rocm ships megatron-core 0.16.0rc0 with TE 2.12.0.dev0 --
every call raises::

    TypeError: general_gemm() got an unexpected keyword argument 'workspace'

Dense models never reach this. It is specific to the MoE router: ``route`` ->
``router.gating`` -> ``moe_utils.router_gating_linear``, which always goes
through ``RouterGatingLinearFunction`` and only falls back to ``torch.mm`` when
``te_general_gemm`` is unavailable or ``router_dtype`` is float64. So a
Qwen3-8B-Base run is fine and a Qwen3-30B-A3B run dies in the first log-prob
pass.

The fix drops the stale keyword and forwards everything else untouched. Patching
the module-global ``general_gemm`` rather than ``te_general_gemm`` is deliberate:
``moe_utils`` binds ``te_general_gemm`` at import time, but ``te_general_gemm``
resolves ``general_gemm`` from its own module globals on every call, so this one
patch covers the forward and both backward GEMMs.

Self-disabling: if TE still accepts ``workspace``, nothing is touched.
"""

from __future__ import annotations

import inspect
import logging

logger = logging.getLogger(__name__)

_PATCHED_FLAG = "_atomrl_workspace_kwarg_dropped"


def install() -> bool:
    """Patch TE's ``general_gemm`` binding if it no longer takes ``workspace``.

    Returns True when a patch was applied (or was already in place).
    """
    try:
        from megatron.core.extensions import transformer_engine as te_ext
    except ImportError:
        return False

    real = getattr(te_ext, "general_gemm", None)
    if real is None:
        return False
    if getattr(real, _PATCHED_FLAG, False):
        return True

    try:
        params = inspect.signature(real).parameters
    except (TypeError, ValueError):
        return False
    if "workspace" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    ):
        return False

    def general_gemm(*args, **kwargs):
        kwargs.pop("workspace", None)
        return real(*args, **kwargs)

    setattr(general_gemm, _PATCHED_FLAG, True)
    te_ext.general_gemm = general_gemm
    logger.info(
        "[atomrl] megatron/TE compat: dropping the 'workspace' kwarg from "
        "general_gemm (TE allocates it internally)"
    )
    return True

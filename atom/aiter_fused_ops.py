# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Per-process lifecycle of opt-in aiter ops served through aiter providers.

ATOM's part is the policy and its timing (``aiter.plugin``; see
:mod:`atom.aiter_provider`):

* :func:`configure_aiter_fused_ops` -- called by ``ModelRunner.__init__``
  after ``_setup_device_and_distributed()`` and before
  ``_build_and_load_model()``, i.e. before any model layer is built, any
  warm-up forward, KV allocation or CUDA-graph capture: enables the configured
  providers (``aiter.plugin.configure``), then sets up each opt-in call site
  that an enabled provider serves. Default off: nothing imported.
* :func:`prewarm_aiter_fused_ops` -- called after the KV cache is allocated
  and bound, before CUDA-graph capture: the providers warm up (compile / load)
  what the bound caches can select.

Both never raise: an aiter without the interface / routed op, or a missing
provider, keeps ATOM's own implementation, logged once. Not ``atom/plugin/``
(ATOM's vLLM/SGLang/rtp-llm frontend adapter).
"""

from __future__ import annotations

import logging

logger = logging.getLogger("atom")


def _qsa_index_scorer_active(config) -> bool:
    cfg = getattr(config, "qsa_index_scorer", None)
    observed = cfg is not None and bool(getattr(cfg, "active", False))
    return observed or bool(getattr(config, "aiter_provider_enable", ()))


def configure_aiter_fused_ops(config, rank_name: str = "") -> dict:
    """Configure every enabled opt-in op in this process.

    Returns ``{op name: state}`` for the enabled ones (empty while all are
    off: nothing imported). Never raises.
    """
    out: dict = {}
    if getattr(config, "aiter_provider_enable", ()):
        from atom.aiter_provider import configure_aiter_providers

        out["aiter_providers"] = configure_aiter_providers(config, rank_name)
    if _qsa_index_scorer_active(config):
        try:
            from atom.model_ops.qwen4_exp.qsa_index_scorer import configure

            out["qsa_index_scorer"] = configure(config, rank_name)
        except Exception as exc:  # noqa: BLE001 -- fail-open: ATOM's scorer
            logger.warning(
                "QSA index scorer [%s]: configure failed (%r); ATOM scorer",
                rank_name or "-",
                exc,
            )
            out["qsa_index_scorer"] = None
    return out


def prewarm_aiter_fused_ops(config, models, rank_name: str = "") -> dict:
    """Prewarm the enabled ops over the caches bound to ``models``' layers.

    A layer that calls an aiter op exposes the caches it passes to it
    (``qsa_index_scorer_caches()``). Never raises.
    """
    out: dict = {}
    if not _qsa_index_scorer_active(config):
        return out
    try:
        from atom.model_ops.qwen4_exp.qsa_index_scorer import prewarm

        caches = []
        for model in models:
            for module in model.modules():
                fn = getattr(module, "qsa_index_scorer_caches", None)
                bound = fn() if callable(fn) else None
                if bound is not None:
                    caches.append(bound)
        out["qsa_index_scorer"] = prewarm(caches, rank_name)
    except Exception as exc:  # noqa: BLE001 -- variants then build on first use
        logger.warning(
            "QSA index scorer [%s]: prewarm skipped (%r)", rank_name or "-", exc
        )
    return out

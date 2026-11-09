# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Per-process lifecycle of opt-in aiter in-tree ops that own their dispatch.

Such an op is a neutral aiter op (one public entry, aiter's own fused /
native dispatch inside); ATOM's part is the policy and its timing:

* :func:`configure_aiter_fused_ops` -- called by ``ModelRunner.__init__``
  after ``_setup_device_and_distributed()`` and before
  ``_build_and_load_model()``, i.e. before any model layer is built, any
  warm-up forward, KV allocation or CUDA-graph capture. It hands each enabled
  op's ``Config`` policy to aiter (``aiter.ops.<op>.configure``) and logs one
  startup line. Default off: nothing imported.
* :func:`prewarm_aiter_fused_ops` -- called after the KV cache is allocated
  and bound, before CUDA-graph capture: aiter builds the kernel variants the
  bound caches can select.

Both never raise: an aiter without the op's lifecycle API keeps ATOM's own
implementation, logged once. Not ``atom/plugin/`` (ATOM's vLLM/SGLang/rtp-llm
frontend adapter) and not ``aiter.plugin`` (aiter's out-of-tree provider
interface, unused here).
"""

from __future__ import annotations

import logging

logger = logging.getLogger("atom")


def _qsa_index_scorer_active(config) -> bool:
    cfg = getattr(config, "qsa_index_scorer", None)
    return cfg is not None and bool(getattr(cfg, "active", False))


def configure_aiter_fused_ops(config, rank_name: str = "") -> dict:
    """Configure every enabled opt-in op in this process.

    Returns ``{op name: state}`` for the enabled ones (empty while all are
    off: nothing imported). Never raises.
    """
    out: dict = {}
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

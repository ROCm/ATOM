# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Per-process capability checks of opt-in, in-tree aiter fused ops.

Each opt-in op ATOM routes to an aiter op behind a default-off ``Config``
switch gets ONE capability check per worker process, run by
``ModelRunner.__init__`` after ``_setup_device_and_distributed()`` and
before ``_build_and_load_model()`` (so before any model layer is built, any
warm-up forward, KV allocation or CUDA-graph capture). A check imports the
op's aiter module only when its switch is on, logs one startup line, and
never raises: a missing/incompatible op leaves ATOM's stock path in place.

Not ``atom/plugin/`` (that is ATOM's vLLM/SGLang/rtp-llm frontend adapter)
and not ``aiter.plugin`` (aiter's out-of-tree provider interface, unused
here: these ops are aiter in-tree ops).
"""

from __future__ import annotations

import logging

logger = logging.getLogger("atom")


def check_aiter_fused_ops(config, rank_name: str = "") -> dict:
    """Run the capability check of every enabled opt-in fused op.

    Returns ``{op name: capability}`` for the enabled ones (empty while all
    are off: nothing imported). Never raises.
    """
    out: dict = {}
    scorer = getattr(config, "qsa_index_scorer", None)
    if scorer is not None and getattr(scorer, "enabled", False):
        try:
            from atom.model_ops.qwen4_exp.qsa_index_scorer.capability import (
                init_qsa_index_scorer,
            )

            out["qsa_index_scorer"] = init_qsa_index_scorer(config, rank_name)
        except Exception as exc:  # noqa: BLE001 -- fail-open: stock path
            logger.warning(
                "QSA index scorer [%s]: capability check failed (%r); stock scorer",
                rank_name or "-",
                exc,
            )
            out["qsa_index_scorer"] = None
    return out

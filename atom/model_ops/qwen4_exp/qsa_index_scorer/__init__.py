# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Opt-in routing of the QSA index scorer to aiter's fused kernel.

Imported only when ``Config.qsa_index_scorer.active`` (see
``Qwen4ExpAttention.__init__``); with the switch off nothing here -- and
nothing of ``aiter.ops.qsa_index_scorer`` -- is imported.

* :mod:`.route` -- the call-site object (`QSAScorerRoute`): prepare-time
  layout checks, one cached decision per call class, native fallback with a
  reason, one log line per (site, reason).
* :mod:`.select` -- `qsa_select_paged_tokens_routed`, the stock
  `qsa_select_paged_tokens` loop with the scorer injected (AST drift-tested).
* :mod:`.capture` -- the call-signature capture hook (coverage matrix /
  ``gen_profile --plan-file``).
* :mod:`.golden_adapter` -- replays the Phase-1 golden cases through this call
  site on the GPU.

This ``__init__`` imports nothing heavy (no Triton, no aiter).
"""

from .route import (  # noqa: F401
    ATOM_REASONS,
    CONTRACT_REASONS,
    NATIVE_SCORER_PARAMS,
    QSAScorerRoute,
    build_route,
)

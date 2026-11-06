# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Opt-in call-site integration of aiter's in-tree QSA index scorer op.

``aiter.ops.qsa_index_scorer`` (ROCm/aiter#7377) is a new coarse op aiter
defines in-tree. This package is imported only when
``Config.qsa_index_scorer`` is active (``ModelRunner.__init__`` via
:mod:`atom.aiter_fused_ops`, and ``Qwen4ExpAttention.__init__``); with the
switch off nothing here -- and nothing of aiter's module -- is imported.

* :mod:`.capability` -- the once-per-process capability check (aiter module
  present, ``API_VERSION`` compatible, signature, profile loads) and the
  startup line; fail-open (stock scorer), never raises.
* :mod:`.route` -- the call-site object (`QSAScorerRoute`): prepare-time
  layout checks + prewarm at ``bind_caches``, one cached decision per call
  class, ``launch`` per call, per-(path, reason) counters and the exit
  summary.
* :mod:`.select` -- `qsa_select_paged_tokens_routed`, the stock
  `qsa_select_paged_tokens` loop with the scorer injected (AST drift-tested).
* :mod:`.capture` -- the call-signature capture hook (coverage matrix /
  ``gen_profile --plan-file``).
* :mod:`.golden_adapter` -- replays the Phase-1 golden cases through this call
  site on the GPU.

This ``__init__`` imports nothing heavy (no Triton, no aiter).
"""

from .route import (  # noqa: F401
    AITER_NATIVE,
    ATOM_REASONS,
    CONTRACT_REASONS,
    FUSED,
    NATIVE_SCORER_PARAMS,
    PATHS,
    SCORER_PARAMS,
    STOCK,
    QSAScorerRoute,
    build_route,
    process_summary,
)

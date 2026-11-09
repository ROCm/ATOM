# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Opt-in use of aiter's neutral QSA scorer op by the Qwen QSA indexer.

aiter defines and dispatches the op: ``aiter.ops.qsa.qsa_paged_mqa_logits``
(contract ``qsa_paged_mqa_logits`` v1; the 12 leading parameters of ATOM's own
scorer). It serves every valid call -- fused FlyDSL where the configured
profile covers the call class, aiter's native paths otherwise -- so ATOM keeps
no route, no capability pinning and no fallback of its own. This package is
imported only while ``Config.qsa_index_scorer`` is active; with the switch off
nothing here, and nothing of ``aiter.ops.qsa``, is imported.

* :mod:`.enablement` -- per-process ``configure`` / startup ``status`` line,
  :func:`~.enablement.index_scorer` (the scorer the call site injects),
  ``prewarm`` after KV allocation, and the opt-in ``explain`` class observer.
* :mod:`.select` -- ``qsa_select_paged_tokens_with_scorer``: the stock
  ``qsa_select_paged_tokens`` loop with the scorer injected (AST drift-tested
  against the frozen ``qsa.py``).
* :mod:`.capture` -- the call-class capture JSON and its ``merge`` CLI
  (coverage matrix / aiter ``gen_profile --plan-file``).

This ``__init__`` imports nothing heavy (no Triton, no aiter).
"""

from .enablement import (  # noqa: F401
    AITER_QSA_MODULE,
    CONFIGURE_SOURCE,
    ScorerState,
    configure,
    decisions,
    index_scorer,
    prewarm,
)

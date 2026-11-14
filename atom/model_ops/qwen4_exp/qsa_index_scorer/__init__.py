# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Opt-in use of aiter's routed QSA scorer op by the Qwen QSA indexer.

aiter defines and routes the op: ``aiter.ops.qsa.qsa_paged_mqa_logits``
(contract ``qsa_paged_mqa_logits`` v1, ``aiter.plugin``; the 12 leading
parameters of ATOM's own scorer). It serves every valid call -- an enabled
provider (e.g. the fused FlyDSL gfx1250 ``qsa-index-scorer-provider``) on the
call classes it covers, aiter's native paths otherwise -- so ATOM keeps no
route, no capability pinning and no fallback of its own. This package is
imported only while a provider is enabled (``Config.aiter_provider_enable``) or
the observer is requested; otherwise nothing here, and nothing of
``aiter.ops.qsa``, is imported.

* :mod:`.enablement` -- per-process ``configure`` / startup line,
  :func:`~.enablement.index_scorer` (the scorer the call site injects),
  provider warm-up after KV allocation, and the opt-in ``explain`` observer.
* :mod:`.select` -- ``qsa_select_paged_groups_with_scorer`` /
  ``qsa_select_paged_tokens_with_scorer``: the stock row-sharded and
  whole-batch selection entries with the scorer injected (AST drift-tested
  against ``qsa.py``).
* :mod:`.capture` -- the call-class capture JSON and its ``merge`` CLI
  (coverage matrix / the provider's ``gen_profile --plan-file``).

This ``__init__`` imports nothing heavy (no Triton, no aiter).
"""

from .enablement import (  # noqa: F401
    AITER_QSA_MODULE,
    OP_NAME,
    ScorerState,
    configure,
    decisions,
    index_scorer,
    prewarm,
)

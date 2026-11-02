# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Qwen GDN offload over the Kimi-K3 PAGE-image tier.

Qwen has no second codec. ``KimiK3OffloadConnector`` already packs
``page_unit_views`` and unpacks ``state_entry_views``, and it does not assume
KDA shapes. Qwen publishes those views from its MHA pool, Qwen3.8 from its QSA arena
(``pool_layout/mha_page_unit_geometry``) under the ``gdn-paged-state`` layout id
spelled in ``GDNAttentionMetadataBuilder.state_transfer``. These
subclasses exist so the selected layout is ``qwen`` rather than ``kimi_k3``.
"""

from __future__ import annotations

from atom.kv_transfer.offload.hybrid.kimi_k3.connector import (
    KimiK3OffloadConnector,
    KimiK3OffloadScheduler,
)


class QwenOffloadConnector(KimiK3OffloadConnector):
    """Worker side: the paged KV (MHA or QSA) plus the GDN page-image state tier."""


class QwenOffloadScheduler(KimiK3OffloadScheduler):
    """Scheduler side of :class:`QwenOffloadConnector`.

    Inherits ``StateOffloadFace`` through K3, so ``isinstance`` routing still
    sends state stores and loads here.
    """

    def _withholds_last_prompt_token(self, seq) -> bool:
        """Not when the joint boundary lies in the withheld chunk.

        A hybrid claims the joint boundary, which already leaves a token to
        compute. Withholding would floor a chunk-aligned prompt a whole chunk
        below the rungs it kept near its end, so no boundary forms. A hit no
        boundary backs is still withheld: `_decide_load_after_alloc` refuses it.
        """
        if not getattr(seq, "has_per_req_cache", False):
            return super()._withholds_last_prompt_token(seq)
        num_prompt = int(seq.num_prompt_tokens)
        withheld = self._chunk_floor(num_prompt - 1)
        if self._block_manager is None or withheld == self._chunk_floor(num_prompt):
            return True
        return self._block_manager.joint_boundary_tokens(seq, num_prompt) <= withheld

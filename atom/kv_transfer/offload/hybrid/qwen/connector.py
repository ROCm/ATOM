# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Qwen GDN offload over the Kimi-K3 PAGE-image tier.

Qwen has no second codec. ``KimiK3OffloadConnector`` already packs
``page_unit_views`` and unpacks ``state_entry_views``, and it does not assume
KDA shapes. Qwen3-Next and Qwen3.5 publish those views from their MHA pool
(``pool_layout/mha_page_unit_geometry``) under the ``gdn-paged-state`` layout id
spelled in ``GDNAttentionMetadataBuilder.state_transfer``. Qwen3.8
(``qwen4_exp``) publishes them from its QSA arena
(``pool_layout/qsa_page_unit_geometry``) through the same builder contract,
and appends the PLE conv / n-gram windows to the image and to the layout id
(``Qwen4ExpMetadataBuilder.state_transfer``). These subclasses exist so the
selected layout is ``qwen`` rather than ``kimi_k3``.
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

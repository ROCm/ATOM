# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Qwen GDN offload over the Kimi-K3 PAGE-image tier.

Qwen has no second codec. ``KimiK3OffloadConnector`` already packs
``page_unit_views`` and unpacks ``state_entry_views``, and it does not assume
KDA shapes. Qwen publishes those views from its MHA pool
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
    """Worker side: dense MHA KV plus the GDN page-image state tier."""


class QwenOffloadScheduler(KimiK3OffloadScheduler):
    """Scheduler side of :class:`QwenOffloadConnector`.

    Inherits ``StateOffloadFace`` through K3, so ``isinstance`` routing still
    sends state stores and loads here.
    """

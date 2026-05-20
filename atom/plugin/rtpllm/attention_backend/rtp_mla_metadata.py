"""Metadata and static contracts for GLM5 MLA in rtp-llm plugin mode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


GLM5_RTP_BRIDGE_MODE_M0_DENSE = "m0_dense"
GLM5_RTP_BRIDGE_MODE = GLM5_RTP_BRIDGE_MODE_M0_DENSE


GLM5_RTP_OWNERSHIP = {
    "main_q_norm": "DeepseekV2MLAAttention",
    "main_kv_norm": "DeepseekV2MLAAttention",
    "main_rope": "RTPMLAAttention",
    "main_kv_cache": "RTPMLAAttention",
    "indexer_k_norm": "Indexer",
    "indexer_rope": "Indexer",
    "indexer_cache": "Indexer",
    "topk_selector": "Indexer",
}


@dataclass(frozen=True)
class RTPMlaPluginMetadata:
    """Minimal M0 placeholder for RTP MLA metadata.

    M0 intentionally does not model indexer/top-k metadata. M1/M2 should extend
    this structure instead of overloading MHA plugin metadata.
    """

    is_prefill: bool
    slot_mapping: Optional[torch.Tensor] = None
    block_table: Optional[torch.Tensor] = None
    seq_lens: Optional[torch.Tensor] = None


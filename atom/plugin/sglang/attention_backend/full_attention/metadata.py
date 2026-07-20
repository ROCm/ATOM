from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ForwardMetadata:
    """Per-batch metadata consumed by SGLang full-attention backend kernels."""

    # kv_indptr and kv_indices are only used in MLA mode, optional for non-MLA mode
    kv_indptr: Optional[torch.Tensor]
    kv_indices: Optional[torch.Tensor]
    qo_indptr: Optional[torch.Tensor]
    kv_last_page_len: Optional[torch.Tensor]
    max_q_len: Optional[int]
    max_kv_len: Optional[int]
    page_table: Optional[torch.Tensor]
    kv_lens: Optional[torch.Tensor]
    # MLA metadata
    work_metadata: Optional[torch.Tensor] = None
    work_info_set: Optional[torch.Tensor] = None
    work_indptr: Optional[torch.Tensor] = None
    reduce_indptr: Optional[torch.Tensor] = None
    reduce_final_map: Optional[torch.Tensor] = None
    reduce_partial_map: Optional[torch.Tensor] = None
    fp8_prefill_kv_indices: Optional[torch.Tensor] = None
    num_kv_splits: Optional[int] = None
    run_graph: Optional[bool] = True
    # PA metadata for pa_persistent_fwd (only used in decode mode, non-MLA)
    pa_metadata_qo_indptr: Optional[torch.Tensor] = None
    pa_metadata_pages_kv_indptr: Optional[torch.Tensor] = None
    pa_metadata_kv_indices: Optional[torch.Tensor] = None
    pa_metadata_context_lens: Optional[torch.Tensor] = None
    pa_metadata_max_qlen: Optional[int] = None

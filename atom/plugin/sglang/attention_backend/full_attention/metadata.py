from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ForwardMetadata:
    """Per-batch metadata consumed by SGLang full-attention backend kernels."""

    # kv_indptr and kv_indices are only used in MLA mode, optional for non-MLA mode
    kv_indptr: torch.Tensor | None
    kv_indices: torch.Tensor | None
    qo_indptr: torch.Tensor | None
    kv_last_page_len: torch.Tensor | None
    max_q_len: int | None
    max_kv_len: int | None
    page_table: torch.Tensor | None
    kv_lens: torch.Tensor | None
    # MLA metadata
    work_metadata: torch.Tensor | None = None
    work_info_set: torch.Tensor | None = None
    work_indptr: torch.Tensor | None = None
    reduce_indptr: torch.Tensor | None = None
    reduce_final_map: torch.Tensor | None = None
    reduce_partial_map: torch.Tensor | None = None
    fp8_prefill_kv_indices: torch.Tensor | None = None
    num_kv_splits: int | None = None
    run_graph: bool | None = True
    # PA metadata for pa_persistent_fwd (only used in decode mode, non-MLA)
    pa_metadata_qo_indptr: torch.Tensor | None = None
    pa_metadata_pages_kv_indptr: torch.Tensor | None = None
    pa_metadata_kv_indices: torch.Tensor | None = None
    pa_metadata_context_lens: torch.Tensor | None = None
    pa_metadata_max_qlen: int | None = None

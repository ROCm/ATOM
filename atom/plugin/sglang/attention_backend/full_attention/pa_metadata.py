from __future__ import annotations

from typing import Optional

import torch
from aiter import get_pa_metadata_info_v1, get_pa_metadata_v1
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton


def _ensure_buffer(backend, name, size, dtype, zero=True):
    """Allocate or reuse a pa_metadata buffer, growing if needed."""
    if backend.pa_metadata_buffers is None:
        backend.pa_metadata_buffers = {}
    size_val = size[0] if isinstance(size, (tuple, list)) else size
    buf = backend.pa_metadata_buffers.get(name)
    needs_alloc = (
        buf is None
        or buf.shape[0] < size_val
        or (isinstance(size, (tuple, list)) and len(buf.shape) < len(size))
    )
    if needs_alloc:
        factory = torch.zeros if zero else torch.empty
        backend.pa_metadata_buffers[name] = factory(
            size, dtype=dtype, device=backend.device
        )
    elif zero:
        backend.pa_metadata_buffers[name].zero_()


def allocate_pa_metadata_buffers(backend, buffer_specs):
    """Allocate or reuse pa_metadata buffers for the backend."""
    names = [
        "work_metadata_ptrs",
        "work_indptr",
        "work_info",
        "reduce_indptr",
        "reduce_final_map",
        "reduce_partial_map",
    ]
    zero_flags = [False, True, True, True, True, True]
    for name, (size, dtype), zero in zip(names, buffer_specs, zero_flags):
        _ensure_buffer(backend, name, size, dtype, zero=zero)


def build_pa_metadata_for_decode(
    backend,
    batch_size: int,
    tp_q_head_num: Optional[int] = None,
):
    """Build pa_metadata buffers for pa_persistent_fwd in decode mode."""
    max_qlen = 1

    if tp_q_head_num is None:
        tp_q_head_num = backend.num_head

    buffer_specs = get_pa_metadata_info_v1(batch_size, backend.num_kv_head)
    allocate_pa_metadata_buffers(backend, buffer_specs)
    qo_indptr = backend.pa_decode_qo_indptr[: batch_size + 1]

    context_lens = backend.forward_metadata.kv_lens

    kernel_block_size = backend.page_size
    num_blocks_per_seq = (context_lens + kernel_block_size - 1) // kernel_block_size
    pages_kv_indptr = backend.pa_kv_indptr[: batch_size + 1]
    pages_kv_indptr[1 : batch_size + 1] = torch.cumsum(num_blocks_per_seq, dim=0)

    page_table = backend.forward_metadata.page_table

    create_flashinfer_kv_indices_triton[(batch_size,)](
        page_table,
        backend.pa_batch_indices[:batch_size],
        num_blocks_per_seq,
        pages_kv_indptr,
        None,
        backend.pa_kv_indices,
        page_table.stride(0),
    )
    kv_indices = backend.pa_kv_indices

    get_pa_metadata_v1(
        seqlens_qo_indptr=qo_indptr,
        pages_kv_indptr=pages_kv_indptr,
        context_lens=context_lens.int(),
        num_heads_per_head_k=tp_q_head_num // backend.num_kv_head,
        num_heads_k=backend.num_kv_head,
        is_causal=True,
        work_metadata_ptrs=backend.pa_metadata_buffers["work_metadata_ptrs"],
        work_indptr=backend.pa_metadata_buffers["work_indptr"],
        work_info=backend.pa_metadata_buffers["work_info"],
        reduce_indptr=backend.pa_metadata_buffers["reduce_indptr"],
        reduce_final_map=backend.pa_metadata_buffers["reduce_final_map"],
        reduce_partial_map=backend.pa_metadata_buffers["reduce_partial_map"],
        kv_granularity=max(kernel_block_size, 16),
        block_size=kernel_block_size,
        max_seqlen_qo=max_qlen,
        uni_seqlen_qo=max_qlen,
        fast_mode=True,
        topk=-1,
        max_split_per_batch=-1,
    )
    backend.forward_metadata.pa_metadata_qo_indptr = qo_indptr
    backend.forward_metadata.pa_metadata_pages_kv_indptr = pages_kv_indptr
    backend.forward_metadata.pa_metadata_kv_indices = kv_indices
    backend.forward_metadata.pa_metadata_context_lens = context_lens
    backend.forward_metadata.pa_metadata_max_qlen = max_qlen


def build_pa_metadata_for_prefill(backend, batch_size: int):
    """Build page-level metadata for non-MLA prefill mode."""
    block_size = backend.page_size
    context_lens = backend.forward_metadata.kv_lens
    num_blocks_per_seq = (context_lens + block_size - 1) // block_size

    pages_kv_indptr = backend.pa_kv_indptr[: batch_size + 1]
    pages_kv_indptr[1 : batch_size + 1] = torch.cumsum(num_blocks_per_seq, dim=0)

    page_table = backend.forward_metadata.page_table
    create_flashinfer_kv_indices_triton[(batch_size,)](
        page_table,
        backend.pa_batch_indices[:batch_size],
        num_blocks_per_seq,
        pages_kv_indptr,
        None,
        backend.pa_kv_indices,
        page_table.stride(0),
    )

"""MiMo MTP prefill for SGLang releases predating unified AITER prefill.

SGLang 0.5.17 sends 192-wide FP8 draft attention to a CK batch-prefill
kernel that does not support that width. Keep this fallback in the MiMo
plugin; newer SGLang releases provide the unified path themselves.
"""

import torch
import triton
from aiter.ops.triton.attention.unified_attention import unified_attention
from sglang.kernels.ops.attention.utils import (
    launch_reshape_and_cache_flash,
)
from sglang.kernels.ops.kvcache.aiter_unified_attention import (
    scatter_req_to_token_to_page_table_kernel,
)


def init_mimo_mtp_prefill_metadata(backend, batch, *, for_cuda_graph=False):
    if not getattr(backend, "_mimo_legacy_unified_prefill", False):
        return
    mode = batch.forward_mode
    if mode.is_decode_or_idle() or mode.is_target_verify():
        return

    bs = batch.batch_size
    md = backend.forward_metadata
    if mode.is_draft_extend_v2():
        md.max_q_len = backend._resolve_v2_num_draft_tokens()
        md.qo_indptr = backend._set_uniform_qo_indptr(bs, md.max_q_len, backend.device)
    else:
        md.qo_indptr = backend.qo_indptr[: bs + 1]

    if for_cuda_graph:
        page_table = backend.cuda_graph_page_table[:bs]
    else:
        max_kv_len = int(batch.seq_lens_cpu.max().item())
        max_blocks = max(1, triton.cdiv(max_kv_len, backend.page_size))
        page_table = torch.empty(
            (bs, max_blocks), dtype=torch.int32, device=backend.device
        )

    swa_mapping, swa_page_table = None, None
    if backend.use_sliding_window_kv_pool:
        swa_mapping = backend.token_to_kv_pool.full_to_swa_index_mapping
        swa_page_table = (
            backend.cuda_graph_swa_page_table[:bs]
            if for_cuda_graph
            else torch.empty_like(page_table)
        )

    block = 1024
    scatter_req_to_token_to_page_table_kernel[
        (bs, triton.cdiv(page_table.shape[1], block))
    ](
        backend.req_to_token,
        batch.req_pool_indices,
        batch.seq_lens,
        page_table,
        backend.req_to_token.stride(0),
        page_table.stride(0),
        swa_page_table,
        swa_mapping,
        DRAFT_NUM=0,
        PAGE_SIZE=backend.page_size,
        BLOCK_SIZE=block,
        HAS_SWA=swa_mapping is not None,
    )
    backend._mimo_prefill_page_table = page_table
    backend._mimo_prefill_swa_page_table = swa_page_table


def forward_mimo_mtp_prefill(backend, q, k, v, layer, batch, *, save_kv_cache, sinks):
    md = backend.forward_metadata
    k_descale, v_descale = backend._kv_descales(layer)
    k_cache, v_cache = backend.token_to_kv_pool.get_kv_buffer(layer.layer_id)
    k_cache = k_cache.view(
        -1, backend.page_size, layer.tp_k_head_num, layer.qk_head_dim
    )
    v_cache = v_cache.view(-1, backend.page_size, layer.tp_v_head_num, layer.v_head_dim)

    is_swa_layer = (
        layer.sliding_window_size is not None and layer.sliding_window_size > 0
    )
    if k is not None and save_kv_cache:
        assert v is not None
        swa_mapping = None
        if backend.use_sliding_window_kv_pool and is_swa_layer:
            swa_mapping = backend.token_to_kv_pool.full_to_swa_index_mapping.long()
        launch_reshape_and_cache_flash(
            k.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
            v.view(-1, layer.tp_v_head_num, layer.v_head_dim),
            k_cache,
            v_cache,
            batch.out_cache_loc,
            swa_mapping,
            k_scale=k_descale,
            v_scale=v_descale,
        )

    page_table = backend._mimo_prefill_page_table
    window = (-1, -1)
    if is_swa_layer:
        window = (layer.sliding_window_size - 1, 0)
        if backend._mimo_prefill_swa_page_table is not None:
            page_table = backend._mimo_prefill_swa_page_table
    q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
    out = torch.empty_like(q)
    unified_attention(
        q=q,
        k=k_cache,
        v=v_cache,
        out=out,
        cu_seqlens_q=md.qo_indptr,
        seqused_k=batch.seq_lens,
        max_seqlen_q=md.max_q_len,
        max_seqlen_k=page_table.shape[1] * backend.page_size,
        softmax_scale=layer.scaling,
        causal=True,
        window_size=window,
        block_table=page_table,
        softcap=layer.logit_cap,
        q_descale=None,
        k_descale=k_descale,
        v_descale=v_descale,
        sinks=sinks,
    )
    return out.view(-1, layer.tp_q_head_num * layer.v_head_dim)

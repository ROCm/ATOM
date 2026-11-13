"""CUDA-graph metadata shim for Kimi-K3 native ATOM attention."""

from __future__ import annotations

from typing import Any

import torch

from atom.plugin.sglang.attention_backend.full_attention.full_attention_backend import (
    ATOMAttnBackendForSgl,
)


class ATOMKimiK3BackendForSgl(ATOMAttnBackendForSgl):
    """Keep K3's native ATOM MHA metadata at stable graph addresses."""

    _last_atom_kimi_k3_graph_metadata = None

    def __init__(self, model_runner, *args, **kwargs):
        # The outer K3 config is hybrid KDA/full-attention, but every
        # full-attention layer is true MLA.  Initialize the parent as MLA so
        # it owns the complete, graph-stable MLA metadata lifecycle (CSR
        # indices and persistent-worker buffers). K3 TP8 has 12 local heads;
        # present the kernel ABI's padded 16 heads only for initialization.
        from sglang.srt.configs.model_config import AttentionArch
        from sglang.srt.runtime_context import get_parallel

        model_config = model_runner.model_config
        original_attention_arch = model_config.attention_arch
        original_num_attention_heads = model_config.num_attention_heads
        attn_tp_size = get_parallel().attn_tp_size
        local_num_heads = original_num_attention_heads // attn_tp_size
        padded_local_num_heads = max(16, ((local_num_heads + 15) // 16) * 16)
        model_config.attention_arch = AttentionArch.MLA
        model_config.num_attention_heads = padded_local_num_heads * attn_tp_size
        try:
            super().__init__(model_runner, *args, **kwargs)
        finally:
            model_config.attention_arch = original_attention_arch
            model_config.num_attention_heads = original_num_attention_heads
        self.atom_kimi_k3_graph_metadata = None
        self._kimi_graph_context_lens = None
        self._kimi_graph_block_tables = None
        self._kimi_graph_slot_mapping = None
        self._kimi_graph_cu_seqlens_q = None

    def _kimi_pools(self):
        token_pool = getattr(self, "token_to_kv_pool", None)
        if token_pool is None:
            token_pool = getattr(self, "_atom_token_to_kv_pool", None)
        req_pool = getattr(self, "req_to_token_pool", None)
        if req_pool is None:
            req_pool = getattr(self, "_atom_req_to_token_pool", None)
        if token_pool is None or req_pool is None:
            raise RuntimeError("Kimi-K3 attention pools are not bound")
        return token_pool, req_pool

    def _init_kimi_graph_buffers(self, max_bs: int) -> None:
        token_pool, req_pool = self._kimi_pools()
        page_size = int(token_pool.page_size)
        max_context_len = int(req_pool.req_to_token.shape[1])
        max_blocks = max(1, (max_context_len + page_size - 1) // page_size)
        self._kimi_graph_context_lens = torch.zeros(
            max_bs, dtype=torch.int32, device=self.device
        )
        self._kimi_graph_block_tables = torch.zeros(
            (max_bs, max_blocks), dtype=torch.int32, device=self.device
        )
        self._kimi_graph_slot_mapping = torch.zeros(
            max_bs, dtype=torch.int64, device=self.device
        )
        self._kimi_graph_cu_seqlens_q = torch.arange(
            max_bs + 1, dtype=torch.int32, device=self.device
        )

    def _build_kimi_decode_graph_metadata(self, forward_batch: Any) -> None:
        if not forward_batch.forward_mode.is_decode_or_idle():
            self.atom_kimi_k3_graph_metadata = None
            return

        bs = int(forward_batch.batch_size)
        if (
            self._kimi_graph_context_lens is None
            or bs > self._kimi_graph_context_lens.numel()
        ):
            self._init_kimi_graph_buffers(bs)

        from atom.plugin.sglang.kimi_k3_bridge import (
            _attach_sglang_mla_metadata,
            kimi_k3_query_dtype,
        )
        from atom.utils.forward_context import AttentionMetaData, AttnState

        token_pool, req_pool = self._kimi_pools()
        page_size = int(token_pool.page_size)
        max_blocks = self._kimi_graph_block_tables.shape[1]
        num_padding = int(getattr(forward_batch, "num_padding", 0) or 0)
        real_bs = max(bs - num_padding, 0)
        self._kimi_graph_context_lens[:bs].copy_(forward_batch.seq_lens[:bs])
        if num_padding > 0:
            # Padded graph lanes must not keep the fill seq_len=1 on request 0.
            self._kimi_graph_context_lens[real_bs:bs].zero_()
        req_indices = forward_batch.req_pool_indices[:bs]
        token_table = req_pool.req_to_token[req_indices, : max_blocks * page_size]
        self._kimi_graph_block_tables[:bs].copy_(
            (token_table[:, ::page_size] // page_size).to(dtype=torch.int32)
        )
        if num_padding > 0:
            self._kimi_graph_block_tables[real_bs:bs].zero_()
        out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
        # The replay buffer is the captured bucket, padded with 0. Copying
        # all of it writes those zeros back over the -1 sentinel and the
        # padded rows store into request 0's KV slot.
        self._kimi_graph_slot_mapping[:bs].fill_(-1)
        if torch.is_tensor(out_cache_loc):
            n = min(int(out_cache_loc.numel()), real_bs)
            if n:
                self._kimi_graph_slot_mapping[:n].copy_(out_cache_loc[:n])

        metadata = AttentionMetaData(
            cu_seqlens_q=self._kimi_graph_cu_seqlens_q[: bs + 1],
            max_seqlen_q=1,
            max_seqlen_k=int(req_pool.req_to_token.shape[1]),
            min_seqlen_q=1,
            # The graph ABI requires a Python integer here. Avoid a device
            # synchronization during capture; MLA decode consumes the CSR
            # indptr/indices above, not this bookkeeping upper bound.
            total_kv=bs * int(req_pool.req_to_token.shape[1]),
            has_cached=False,
            dropout_p=0.0,
            slot_mapping=self._kimi_graph_slot_mapping[:bs],
            context_lens=self._kimi_graph_context_lens[:bs],
            block_tables=self._kimi_graph_block_tables[:bs],
            state=AttnState.DECODE,
        )
        metadata.dtype_q = kimi_k3_query_dtype()
        self.atom_kimi_k3_graph_metadata = _attach_sglang_mla_metadata(metadata)
        forward_batch.atom_kimi_k3_graph_metadata = self.atom_kimi_k3_graph_metadata
        type(self)._last_atom_kimi_k3_graph_metadata = self.atom_kimi_k3_graph_metadata

    def _retarget_kimi_decode_mla_plan(self, forward_batch: Any) -> None:
        if not forward_batch.forward_mode.is_decode_or_idle():
            return
        import sglang.srt.layers.attention.aiter_backend as sglang_aiter

        from atom.plugin.sglang.kimi_k3_bridge import retarget_kimi_k3_decode_mla_plan

        retarget_kimi_k3_decode_mla_plan(
            getattr(self, "forward_metadata", None),
            batch_size=int(forward_batch.batch_size),
            num_heads=int(getattr(self, "num_head_padded", 16)),
            dtype_q=getattr(self, "kv_cache_dtype", torch.bfloat16),
            dtype_kv=getattr(self, "kv_cache_dtype", torch.bfloat16),
            fast_mode=bool(sglang_aiter.fast_mode),
            intra_batch_mode=bool(sglang_aiter.intra_batch_mode),
            max_split_per_batch=getattr(self, "max_split_per_batch", None),
        )

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int, *args, **kwargs):
        result = super().init_cuda_graph_state(max_bs, max_num_tokens, *args, **kwargs)
        self._init_kimi_graph_buffers(int(max_bs))
        return result

    def init_forward_metadata_out_graph(self, forward_batch, in_capture: bool = False):
        super().init_forward_metadata_out_graph(forward_batch, in_capture=in_capture)
        # super() just wrote a page-128 plan into the persistent work buffers.
        # Rewrite it before the graph records or replays those addresses.
        self._retarget_kimi_decode_mla_plan(forward_batch)
        self._build_kimi_decode_graph_metadata(forward_batch)

    def init_forward_metadata_in_graph(self, forward_batch):
        # SGLang's parent contract performs graph metadata allocation and
        # refreshes outside the capture region. Keep capture free of host
        # synchronization and dynamic allocation.
        return None

    def init_forward_metadata(self, forward_batch):
        super().init_forward_metadata(forward_batch)
        self._retarget_kimi_decode_mla_plan(forward_batch)

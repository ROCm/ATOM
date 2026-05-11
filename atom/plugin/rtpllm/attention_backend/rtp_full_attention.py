from __future__ import annotations

import math
from typing import Any, Optional

import torch

from atom.model_ops.base_attention import BaseAttention
from atom.plugin.prepare import is_plugin_mode, is_rtpllm
from atom.utils.forward_context import get_forward_context


def _align_kv_heads_for_cache(
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    target_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    current_kv_heads = int(key.shape[1])
    if current_kv_heads == int(target_kv_heads):
        return key, value, 1
    if current_kv_heads <= 0 or int(target_kv_heads) <= 0:
        raise ValueError(
            f"invalid kv head count: current={current_kv_heads}, target={target_kv_heads}"
        )
    if int(target_kv_heads) % current_kv_heads != 0:
        raise ValueError(
            f"cannot align kv heads from {current_kv_heads} to {target_kv_heads}"
        )
    dup_factor = int(target_kv_heads) // current_kv_heads
    key_aligned = key.repeat_interleave(dup_factor, dim=1)
    value_aligned = value.repeat_interleave(dup_factor, dim=1)
    return key_aligned, value_aligned, dup_factor


def _resolve_block_tables_for_layer(attn_inputs: object, layer_num: int) -> torch.Tensor | None:
    # Mirror RTP select_block_map_for_layer semantics:
    # 1) compute gid from kv_cache_layer_to_group[layer]
    # 2) if by-group block map exists, select by gid
    # 3) otherwise fallback to current kv_cache_kernel_block_id_device
    current = getattr(attn_inputs, "kv_cache_kernel_block_id_device", None)
    by_group = getattr(attn_inputs, "kv_cache_kernel_block_id_device_by_group", None)

    gid = 0
    layer_to_group = getattr(attn_inputs, "kv_cache_layer_to_group", None)
    if layer_to_group is not None:
        try:
            gid = int(layer_to_group[layer_num].item())
        except Exception:
            gid = 0

    if isinstance(by_group, (list, tuple)) and len(by_group) > gid:
        t = by_group[gid]
        if t is not None and t.numel() > 0:
            return t
    return current


def _run_nonasm_paged_attention(
    *,
    query: torch.Tensor,
    paged_kv_cache: torch.Tensor,
    kv_scale_base: torch.Tensor | None,
    seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    layer_num: int,
) -> torch.Tensor:
    import aiter

    key_cache = paged_kv_cache.select(1, 0)
    value_cache = paged_kv_cache.select(1, 1)
    num_kv_heads = key_cache.shape[1]
    head_size = query.shape[2]
    num_seqs, num_heads, _ = query.shape
    block_size = value_cache.shape[2]
    max_seq_len = int(seq_lens.max().item()) if int(seq_lens.numel()) > 0 else 1
    scale = 1.0 / math.sqrt(head_size)

    partition_size = 256
    max_num_partitions = (max_seq_len + partition_size - 1) // partition_size

    output = torch.empty_like(query).view((num_seqs, num_heads, head_size))
    tmp_output = torch.empty(
        size=(num_seqs, num_heads, max_num_partitions, head_size),
        dtype=output.dtype,
        device=output.device,
    )
    exp_sums = torch.empty(
        size=(num_seqs, num_heads, max_num_partitions),
        dtype=torch.float32,
        device=output.device,
    )
    max_logits = torch.ones_like(exp_sums)

    k_scale = None
    v_scale = None
    if (
        key_cache.dtype in (torch.float8_e4m3fnuz, torch.float8_e4m3fn)
        and value_cache.dtype in (torch.float8_e4m3fnuz, torch.float8_e4m3fn)
        and kv_scale_base is not None
    ):
        k_scale = kv_scale_base.select(1, 0)
        v_scale = kv_scale_base.select(1, 1)
    else:
        # Keep fallback semantics aligned with RTP non-ASM decode path.
        unit_scale = torch.ones(1, dtype=torch.float32, device=query.device)
        k_scale = unit_scale
        v_scale = unit_scale

    aiter.paged_attention_rocm(
        output,
        exp_sums,
        max_logits,
        tmp_output,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        float(scale),
        block_tables,
        seq_lens,
        block_size,
        max_seq_len,
        None,  # alibi_slopes
        "auto",  # kv_cache_dtype
        k_scale,
        v_scale,
        None,  # fp8_out_scale
        partition_size,
    )
    return output


class RTPFullAttention(BaseAttention):
    """RTP-style full attention adapter for rtpllm plugin mode."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
        kv_cache_dtype: str = "bf16",
        layer_num: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            kv_cache_dtype=kv_cache_dtype,
            layer_num=layer_num,
            **kwargs,
        )
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.num_kv_heads = int(num_kv_heads)
        self.scale = float(scale)
        self.layer_num = int(layer_num)
        # Cached reshape view — invalidated when kv_cache_base is reallocated.
        self._paged_kv_cache: torch.Tensor | None = None
        self._raw_kv_data_ptr: int = 0
        self._raw_kv_numel: int = 0
        self._kernel_seq_size_per_block: int = 0
        # Cached fused KV write ops — built once on first forward.
        self._fused_prefill_op: Any | None = None
        self._fused_decode_op: Any | None = None

    def _ensure_fused_ops(
        self,
        layer_cache: object,
        kv_head_num: int,
        kernel_seq_size_per_block: int,
        dtype: torch.dtype,
    ) -> None:
        if self._fused_prefill_op is not None:
            return
        from rtp_llm.ops import AttentionConfigs, KvCacheDataType
        from rtp_llm.ops.compute_ops import (
            FusedRopeKVCacheDecodeOpNonAsm,
            FusedRopeKVCachePrefillOpNonAsm,
        )
        attn_configs = AttentionConfigs()
        attn_configs.head_num = self.num_heads
        # Use post-alignment kv head count so the fused-write op interprets the
        # qkv strides consistently with how we lay out the (already-aligned) k/v.
        attn_configs.kv_head_num = int(kv_head_num)
        attn_configs.size_per_head = self.head_dim
        attn_configs.tokens_per_block = kernel_seq_size_per_block
        attn_configs.kernel_tokens_per_block = kernel_seq_size_per_block
        attn_configs.is_causal = True
        attn_configs.use_mla = False
        attn_configs.q_scaling = 1.0
        attn_configs.dtype = dtype
        kv_dtype = getattr(getattr(layer_cache, "kv_cache_base", None), "dtype", None)
        attn_configs.kv_cache_dtype = (
            KvCacheDataType.FP8
            if kv_dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
            else KvCacheDataType.BASE
        )
        attn_configs.need_rope_kv_cache = False
        self._fused_prefill_op = FusedRopeKVCachePrefillOpNonAsm(attn_configs)
        self._fused_decode_op = FusedRopeKVCacheDecodeOpNonAsm(attn_configs)

    def _forward_impl_plugin_mode(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        import aiter

        from rtp_llm.models_py.modules.factory.attention.common import reshape_paged_kv_cache

        del positions, kwargs
        fwd_ctx = get_forward_context()
        if fwd_ctx is None:
            raise ValueError("RTPAttention requires forward context in plugin mode.")

        attn_metadata = getattr(fwd_ctx, "attn_metadata", None)
        if attn_metadata is None:
            raise ValueError("RTPAttention requires attn_metadata in forward context.")

        attn_inputs = getattr(attn_metadata, "rtp_attn_inputs", None)
        if attn_inputs is None:
            raise ValueError("RTPAttention requires rtp_attn_inputs in attn_metadata.")

        kv_cache_data = getattr(fwd_ctx, "kv_cache_data", None)
        if kv_cache_data is None:
            raise ValueError("RTPAttention requires kv_cache_data in forward context.")
        layer_cache_entry = kv_cache_data.get(f"layer_{self.layer_num}")
        if layer_cache_entry is None or layer_cache_entry.k_cache is None:
            raise ValueError(
                f"RTPAttention requires layer cache for layer_{self.layer_num}."
            )
        layer_cache = layer_cache_entry.k_cache

        q = query.view(-1, self.num_heads, self.head_dim)
        k = key.view(-1, self.num_kv_heads, self.head_dim)
        v = value.view(-1, self.num_kv_heads, self.head_dim)
        is_prefill = bool(getattr(attn_inputs, "is_prefill", False))
        raw = getattr(layer_cache, "kv_cache_base", None)
        if raw is None:
            raise ValueError(
                f"RTPAttention layer_{self.layer_num} missing kv_cache_base."
            )
        kernel_seq_size_per_block = int(
            getattr(attn_metadata, "rtp_kernel_seq_size_per_block", 0)
            or getattr(layer_cache, "kernel_seq_size_per_block", 0)
            or getattr(layer_cache, "seq_size_per_block", 0)
            or 16
        )
        # Invalidate cached view when the underlying buffer is reallocated (more blocks added).
        if (
            self._paged_kv_cache is None
            or raw.data_ptr() != self._raw_kv_data_ptr
            or raw.numel() != self._raw_kv_numel
        ):
            paged_kv = reshape_paged_kv_cache(
                raw,
                num_kv_heads=self.num_kv_heads,
                tokens_per_block=kernel_seq_size_per_block,
                head_dim=self.head_dim,
            )
            if paged_kv.dim() != 5 or int(paged_kv.shape[1]) != 2:
                raise ValueError(
                    f"RTPAttention expects paged kv cache [num_blocks,2,H,T,D], "
                    f"got {tuple(paged_kv.shape)}"
                )
            self._paged_kv_cache = paged_kv
            self._raw_kv_data_ptr = raw.data_ptr()
            self._raw_kv_numel = raw.numel()
            self._kernel_seq_size_per_block = kernel_seq_size_per_block

        paged_kv = self._paged_kv_cache
        key_cache = paged_kv.select(1, 0)
        value_cache = paged_kv.select(1, 1)
        target_kv_heads = int(key_cache.shape[1])
        k, v, _ = _align_kv_heads_for_cache(
            key=k,
            value=v,
            target_kv_heads=target_kv_heads,
        )
        seq_lens = getattr(attn_metadata.plugin_metadata, "seq_lens", None)
        if seq_lens is None:
            raise ValueError("RTPAttention requires seq_lens in plugin_metadata.")
        block_tables = _resolve_block_tables_for_layer(attn_inputs, int(self.layer_num))
        if block_tables is None or block_tables.numel() == 0:
            raise ValueError(
                f"RTPAttention requires block table for layer_{self.layer_num}."
            )
        block_tables = block_tables.to(device=q.device, dtype=torch.int32, non_blocking=True)
        seq_lens = seq_lens.to(device=q.device, dtype=torch.int32, non_blocking=True)

        self._ensure_fused_ops(
            layer_cache,
            int(k.shape[1]),
            self._kernel_seq_size_per_block,
            q.dtype,
        )
        op = self._fused_prefill_op if is_prefill else self._fused_decode_op
        qkv = torch.cat(
            [
                q.reshape(q.shape[0], -1),
                k.reshape(k.shape[0], -1),
                v.reshape(v.shape[0], -1),
            ],
            dim=-1,
        ).contiguous()
        params = op.prepare(attn_inputs)
        op.forward(qkv, layer_cache, params)
        cu_seqlens_q = getattr(attn_inputs, "cu_seqlens", None)
        if is_prefill and cu_seqlens_q is not None and cu_seqlens_q.numel() > 1:
            cu_seqlens_q = cu_seqlens_q.to(device=q.device, dtype=torch.int32, non_blocking=True)
            q_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(torch.int32)
            num_seqs = int(q_lens.numel())
            max_q_len = int(q_lens.max().item()) if num_seqs > 0 else 0
            prefix_lengths = getattr(attn_inputs, "prefix_lengths", None)
            has_prefix = bool(
                prefix_lengths is not None
                and prefix_lengths.numel() > 0
                and int(prefix_lengths.max().item()) > 0
            )
            if has_prefix:
                key_cache_aiter = key_cache
                value_cache_aiter = value_cache
                x = 16 // key_cache_aiter.element_size()
                kv_sizes = key_cache_aiter.shape
                key_cache_aiter = key_cache_aiter.view(
                    kv_sizes[0], kv_sizes[1], kv_sizes[3] // x, kv_sizes[2], x
                )
                value_cache_aiter = value_cache_aiter.view(
                    kv_sizes[0], kv_sizes[1], kv_sizes[2] // x, kv_sizes[3], x
                )
                kv_indptr = torch.zeros(num_seqs + 1, dtype=torch.int32, device=q.device)
                kv_page_indices = torch.zeros(1, dtype=torch.int32, device=q.device)
                q_descale = None
                k_descale = None
                v_descale = None
                if key_cache_aiter.dtype in (torch.float8_e4m3fnuz, torch.float8_e4m3fn):
                    q_descale = torch.ones(1, dtype=torch.float32, device=q.device)
                    k_descale = torch.ones(1, dtype=torch.float32, device=q.device)
                    v_descale = torch.ones(1, dtype=torch.float32, device=q.device)
                output = aiter.mha_batch_prefill_func(
                    q,
                    key_cache_aiter,
                    value_cache_aiter,
                    cu_seqlens_q,
                    kv_indptr,
                    kv_page_indices,
                    max_q_len,
                    int(seq_lens[:num_seqs].max().item()) if num_seqs > 0 else 1,
                    causal=True,
                    block_table=block_tables[:num_seqs],
                    seqlen_k=seq_lens[:num_seqs],
                    q_descale=q_descale,
                    k_descale=k_descale,
                    v_descale=v_descale,
                )
            else:
                output = aiter.flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    cu_seqlens_q,
                    cu_seqlens_q,
                    max_q_len,
                    max_q_len,
                    dropout_p=0.0,
                    causal=True,
                )
            output = output.reshape(int(q.shape[0]), self.num_heads * self.head_dim)
            output = output.view(-1, self.num_heads * self.head_dim)
            return output

        num_seqs = int(q.shape[0])
        output = _run_nonasm_paged_attention(
            query=q,
            paged_kv_cache=paged_kv,
            kv_scale_base=getattr(layer_cache, "kv_scale_base", None),
            seq_lens=seq_lens[:num_seqs],
            block_tables=block_tables[:num_seqs],
            layer_num=int(self.layer_num),
        )
        output = output.view(num_seqs, -1)
        return output

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        q_scale: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        del q_scale
        if not is_plugin_mode() or not is_rtpllm():
            raise NotImplementedError("RTPFullAttention is only supported in rtpllm plugin mode.")
        return self._forward_impl_plugin_mode(
            query=query,
            key=key,
            value=value,
            positions=positions,
            **kwargs,
        )


RTPAttention = RTPFullAttention

from __future__ import annotations

import math
from typing import Optional

import torch
try:
    import aiter
except (ImportError, ModuleNotFoundError):  # pragma: no cover - runtime fallback
    aiter = None

try:
    from rtp_llm.models_py.modules.factory.attention.common import reshape_paged_kv_cache
except (ImportError, ModuleNotFoundError):  # pragma: no cover - runtime fallback
    reshape_paged_kv_cache = None

try:
    from rtp_llm.ops import AttentionConfigs, KvCacheDataType
    from rtp_llm.ops.compute_ops import (
        FusedRopeKVCacheDecodeOpNonAsm,
        FusedRopeKVCachePrefillOpNonAsm,
    )
except (ImportError, ModuleNotFoundError):  # pragma: no cover - runtime fallback
    AttentionConfigs = None
    KvCacheDataType = None
    FusedRopeKVCacheDecodeOpNonAsm = None
    FusedRopeKVCachePrefillOpNonAsm = None

from atom.model_ops.base_attention import BaseAttention
from atom.plugin.prepare import is_plugin_mode, is_rtpllm
from atom.utils.forward_context import get_forward_context


def _align_kv_heads_for_cache(
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    target_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    current_kv_heads = int(key.shape[1])
    if current_kv_heads == int(target_kv_heads):
        return key, value
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
    return key_aligned, value_aligned


def _write_kv_cache_with_rtp_fused_kernel(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_cache: object,
    attn_inputs: object,
    tokens_per_block: int,
    qkv_buffer: torch.Tensor | None = None,
    fused_op: object | None = None,
) -> bool:
    if fused_op is None:
        return False

    q_flat = query.reshape(query.shape[0], -1)
    k_flat = key.reshape(key.shape[0], -1)
    v_flat = value.reshape(value.shape[0], -1)
    total_dim = int(q_flat.shape[1] + k_flat.shape[1] + v_flat.shape[1])
    if (
        qkv_buffer is None
        or qkv_buffer.device != query.device
        or qkv_buffer.dtype != query.dtype
        or int(qkv_buffer.shape[0]) != int(query.shape[0])
        or int(qkv_buffer.shape[1]) != total_dim
    ):
        qkv = torch.empty(
            (int(query.shape[0]), total_dim),
            dtype=query.dtype,
            device=query.device,
        )
    else:
        qkv = qkv_buffer
    q_end = int(q_flat.shape[1])
    k_end = q_end + int(k_flat.shape[1])
    qkv[:, :q_end].copy_(q_flat)
    qkv[:, q_end:k_end].copy_(k_flat)
    qkv[:, k_end:].copy_(v_flat)
    op = fused_op
    params = op.prepare(attn_inputs)
    _ = op.forward(qkv, layer_cache, params)
    return True


def _resolve_block_tables_for_layer(
    attn_inputs: object,
    layer_num: int,
    *,
    layer_group_map: dict[int, int] | None = None,
) -> torch.Tensor | None:
    # Mirror RTP select_block_map_for_layer semantics:
    # 1) compute gid from kv_cache_layer_to_group[layer]
    # 2) if by-group block map exists, select by gid
    # 3) otherwise fallback to current kv_cache_kernel_block_id_device
    current = getattr(attn_inputs, "kv_cache_kernel_block_id_device", None)
    by_group = getattr(attn_inputs, "kv_cache_kernel_block_id_device_by_group", None)

    gid = (
        int(layer_group_map[layer_num])
        if (layer_group_map is not None and layer_num in layer_group_map)
        else 0
    )

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
    max_seq_len: int,
) -> torch.Tensor:
    if aiter is None:
        raise ValueError("RTPAttention requires aiter for nonasm paged attention.")

    key_cache = paged_kv_cache.select(1, 0)
    value_cache = paged_kv_cache.select(1, 1)
    num_kv_heads = key_cache.shape[1]
    head_size = query.shape[2]
    num_seqs, num_heads, _ = query.shape
    block_size = value_cache.shape[2]
    max_seq_len = int(max_seq_len)
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
        self._fused_qkv_buf: torch.Tensor | None = None
        self._paged_kv_cache: torch.Tensor | None = None
        self._paged_kv_cache_sig: tuple[int, int, int, int, int] | None = None
        self._fused_kv_op_cache: dict[
            tuple[torch.dtype, str, int, int, int, int, bool], object
        ] = {}
        self._backend_ready = aiter is not None and reshape_paged_kv_cache is not None

    def _get_fused_qkv_buffer(
        self,
        *,
        num_tokens: int,
        total_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        buf = self._fused_qkv_buf
        if (
            buf is None
            or buf.device != device
            or buf.dtype != dtype
            or int(buf.shape[0]) != int(num_tokens)
            or int(buf.shape[1]) != int(total_dim)
        ):
            buf = torch.empty((num_tokens, total_dim), dtype=dtype, device=device)
            self._fused_qkv_buf = buf
        return buf

    def _get_paged_kv_cache(
        self,
        *,
        raw: torch.Tensor,
        tokens_per_block: int,
    ) -> torch.Tensor:
        signature = (
            int(raw.data_ptr()),
            int(raw.numel()),
            int(self.num_kv_heads),
            int(self.head_dim),
            int(tokens_per_block),
        )
        cached = self._paged_kv_cache
        if cached is None or self._paged_kv_cache_sig != signature:
            cached = reshape_paged_kv_cache(
                raw,
                num_kv_heads=self.num_kv_heads,
                tokens_per_block=tokens_per_block,
                head_dim=self.head_dim,
            )
            self._paged_kv_cache = cached
            self._paged_kv_cache_sig = signature
        return cached

    def _get_fused_kv_op(
        self,
        *,
        query_dtype: torch.dtype,
        kv_cache_dtype: torch.dtype | None,
        tokens_per_block: int,
        num_kv_heads: int,
        is_prefill: bool,
    ) -> object | None:
        if (
            AttentionConfigs is None
            or KvCacheDataType is None
            or FusedRopeKVCacheDecodeOpNonAsm is None
            or FusedRopeKVCachePrefillOpNonAsm is None
        ):
            return None
        kv_dtype_key = (
            "fp8"
            if kv_cache_dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
            else "base"
        )
        cache_key = (
            query_dtype,
            kv_dtype_key,
            int(tokens_per_block),
            int(num_kv_heads),
            bool(is_prefill),
        )
        op = self._fused_kv_op_cache.get(cache_key)
        if op is not None:
            return op
        attn_configs = AttentionConfigs()
        attn_configs.head_num = int(self.num_heads)
        attn_configs.kv_head_num = int(num_kv_heads)
        attn_configs.size_per_head = int(self.head_dim)
        attn_configs.tokens_per_block = int(tokens_per_block)
        attn_configs.kernel_tokens_per_block = int(tokens_per_block)
        attn_configs.is_causal = True
        attn_configs.use_mla = False
        attn_configs.q_scaling = 1.0
        attn_configs.dtype = query_dtype
        if kv_dtype_key == "fp8":
            attn_configs.kv_cache_dtype = KvCacheDataType.FP8
        else:
            attn_configs.kv_cache_dtype = KvCacheDataType.BASE
        # Keep RoPE mocked on ATOM side for this experiment;
        # we only reuse RTP fused address/write semantics here.
        attn_configs.need_rope_kv_cache = False
        if is_prefill:
            op = FusedRopeKVCachePrefillOpNonAsm(attn_configs)
        else:
            op = FusedRopeKVCacheDecodeOpNonAsm(attn_configs)
        self._fused_kv_op_cache[cache_key] = op
        return op

    def _forward_impl_plugin_mode(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del positions, kwargs
        if not self._backend_ready:
            raise ValueError(
                "RTPAttention requires aiter and reshape_paged_kv_cache in plugin mode."
            )
        fwd_ctx = get_forward_context()
        if fwd_ctx is None:
            raise ValueError("RTPAttention requires forward context in plugin mode.")

        attn_metadata = fwd_ctx.attn_metadata
        if attn_metadata is None:
            raise ValueError("RTPAttention requires attn_metadata in forward context.")

        attn_inputs = attn_metadata.rtp_attn_inputs
        if attn_inputs is None:
            raise ValueError("RTPAttention requires rtp_attn_inputs in attn_metadata.")

        kv_cache_data = fwd_ctx.kv_cache_data
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
            getattr(attn_metadata, "rtp_kernel_seq_size_per_block", 0) or 16
        )
        paged_kv = self._get_paged_kv_cache(
            raw=raw,
            tokens_per_block=kernel_seq_size_per_block,
        )
        if paged_kv.dim() != 5 or int(paged_kv.shape[1]) != 2:
            raise ValueError(
                f"RTPAttention expects paged kv cache [num_blocks,2,H,T,D], got {tuple(paged_kv.shape)}"
            )

        key_cache = paged_kv.select(1, 0)
        value_cache = paged_kv.select(1, 1)
        target_kv_heads = int(key_cache.shape[1])
        if target_kv_heads != self.num_kv_heads:
            k, v = _align_kv_heads_for_cache(
                key=k,
                value=v,
                target_kv_heads=target_kv_heads,
            )
        layer_group_map = getattr(attn_metadata, "rtp_layer_group_map", None)
        block_tables = _resolve_block_tables_for_layer(
            attn_inputs,
            int(self.layer_num),
            layer_group_map=layer_group_map,
        )
        if block_tables is None or block_tables.numel() == 0:
            raise ValueError(
                f"RTPAttention requires block table for layer_{self.layer_num}."
            )
        plugin_md = attn_metadata.plugin_metadata
        seq_lens = plugin_md.seq_lens
        if seq_lens is None:
            raise ValueError("RTPAttention requires block tables and sequence lengths.")
        fused_qkv_dim = int(
            self.num_heads * self.head_dim + 2 * int(k.shape[1]) * self.head_dim
        )
        fused_qkv_buf = self._get_fused_qkv_buffer(
            num_tokens=int(q.shape[0]),
            total_dim=fused_qkv_dim,
            device=q.device,
            dtype=q.dtype,
        )
        used_fused_write = _write_kv_cache_with_rtp_fused_kernel(
            query=q,
            key=k,
            value=v,
            layer_cache=layer_cache,
            attn_inputs=attn_inputs,
            tokens_per_block=kernel_seq_size_per_block,
            qkv_buffer=fused_qkv_buf,
            fused_op=self._get_fused_kv_op(
                query_dtype=q.dtype,
                kv_cache_dtype=getattr(raw, "dtype", None),
                tokens_per_block=kernel_seq_size_per_block,
                num_kv_heads=int(k.shape[1]),
                is_prefill=is_prefill,
            ),
        )
        if not used_fused_write:
            raise RuntimeError(
                "RTP fused KV write is required but unavailable; "
                "fallback slot_mapping path is removed."
            )
        cu_seqlens_q = getattr(plugin_md, "rtp_cu_seqlens_q", None)
        if is_prefill and cu_seqlens_q is not None and cu_seqlens_q.numel() > 1:
            q_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(torch.int32)
            num_seqs = int(q_lens.numel())
            max_q_len = int(plugin_md.max_query_len)
            max_seq_len = int(plugin_md.max_seq_len)
            has_prefix = bool(getattr(plugin_md, "rtp_has_prefix", False))
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
                    max_seq_len,
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
            return output.reshape(int(q.shape[0]), self.num_heads * self.head_dim)

        num_seqs = int(q.shape[0])
        output = _run_nonasm_paged_attention(
            query=q,
            paged_kv_cache=paged_kv,
            kv_scale_base=getattr(layer_cache, "kv_scale_base", None),
            seq_lens=seq_lens[:num_seqs],
            block_tables=block_tables[:num_seqs],
            max_seq_len=int(plugin_md.max_seq_len),
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

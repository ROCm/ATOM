from __future__ import annotations

import math
from typing import Optional

import torch
try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - runtime fallback
    triton = None
    tl = None

from atom.model_ops.base_attention import BaseAttention
from atom.plugin.prepare import is_plugin_mode, is_rtpllm
from atom.utils.forward_context import get_forward_context

if triton is not None:

    @triton.jit
    def _reshape_and_cache_shuffle_kernel(
        key_ptr,  # [num_tokens, num_kv_heads, head_size]
        value_ptr,  # [num_tokens, num_kv_heads, head_size]
        key_cache_ptr,  # [num_blocks, num_kv_heads, head_size // x, block_size, x]
        value_cache_ptr,  # [num_blocks, num_kv_heads, block_size // x, head_size, x]
        slot_mapping_ptr,  # [num_tokens]
        k_scale_ptr,
        v_scale_ptr,
        x,
        k_stride0,
        k_stride1,
        v_stride0,
        v_stride1,
        key_cache_stride0,
        key_cache_stride1,
        key_cache_stride2,
        key_cache_stride3,
        key_cache_stride4,
        value_cache_stride0,
        value_cache_stride1,
        value_cache_stride2,
        value_cache_stride3,
        value_cache_stride4,
        block_size,
        head_size,
        BLOCK_SIZE: tl.constexpr,
        QUANT: tl.constexpr,
    ):
        tid = tl.program_id(0)
        head_id = tl.program_id(1)
        offset = tl.arange(0, BLOCK_SIZE)
        mask = offset < head_size
        slot_id = tl.load(slot_mapping_ptr + tid)
        if slot_id < 0:
            return
        block_id = slot_id // block_size
        block_offset = slot_id % block_size

        src_offset_k = tid * k_stride0 + head_id * k_stride1 + offset
        src_offset_v = tid * v_stride0 + head_id * v_stride1 + offset
        k_val = tl.load(key_ptr + src_offset_k, mask=mask, other=0)
        v_val = tl.load(value_ptr + src_offset_v, mask=mask, other=0)

        if QUANT:
            k_scale = tl.load(k_scale_ptr)
            v_scale = tl.load(v_scale_ptr)
            k_dtype = key_cache_ptr.type.element_ty
            v_dtype = value_cache_ptr.type.element_ty
            k_val = (k_val.to(tl.float32) / k_scale).to(k_dtype)
            v_val = (v_val.to(tl.float32) / v_scale).to(v_dtype)

        k_hdx = offset // x
        k_x = offset % x
        dst_k = (
            block_id * key_cache_stride0
            + head_id * key_cache_stride1
            + k_hdx * key_cache_stride2
            + block_offset * key_cache_stride3
            + k_x * key_cache_stride4
        )
        tl.store(key_cache_ptr + dst_k, k_val, mask=mask)

        v_bdx = block_offset // x
        v_x = block_offset % x
        dst_v = (
            block_id * value_cache_stride0
            + head_id * value_cache_stride1
            + v_bdx * value_cache_stride2
            + offset * value_cache_stride3
            + v_x * value_cache_stride4
        )
        tl.store(value_cache_ptr + dst_v, v_val, mask=mask)


def _reshape_and_cache_shuffle_triton(
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> bool:
    if triton is None:
        return False
    if key.numel() == 0 or slot_mapping.numel() == 0:
        return True
    if not (
        key.is_cuda
        and value.is_cuda
        and key_cache.is_cuda
        and value_cache.is_cuda
        and slot_mapping.is_cuda
    ):
        return False

    num_tokens, num_kv_heads, head_size = key.shape
    num_blocks, _, block_size, _ = key_cache.shape
    x = 16 // int(key_cache.element_size())
    if x <= 0 or head_size % x != 0 or block_size % x != 0:
        return False

    try:
        new_key_cache = key_cache.view(
            num_blocks, num_kv_heads, head_size // x, block_size, x
        )
        new_value_cache = value_cache.view(
            num_blocks, num_kv_heads, block_size // x, head_size, x
        )
    except Exception:
        return False

    kv_cache_dtype = "fp8" if key_cache.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz) else "auto"
    quant = kv_cache_dtype.startswith("fp8")
    k_scale = torch.ones(1, dtype=torch.float32, device=key.device)
    v_scale = torch.ones(1, dtype=torch.float32, device=key.device)

    grid = (num_tokens, num_kv_heads)
    _reshape_and_cache_shuffle_kernel[grid](
        key,
        value,
        new_key_cache,
        new_value_cache,
        slot_mapping.to(dtype=torch.int32, device=key.device),
        k_scale,
        v_scale,
        x,
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        new_key_cache.stride(0),
        new_key_cache.stride(1),
        new_key_cache.stride(2),
        new_key_cache.stride(3),
        new_key_cache.stride(4),
        new_value_cache.stride(0),
        new_value_cache.stride(1),
        new_value_cache.stride(2),
        new_value_cache.stride(3),
        new_value_cache.stride(4),
        block_size,
        head_size,
        BLOCK_SIZE=head_size,
        QUANT=quant,
    )
    return True


def _write_kv_cache_from_slot_mapping(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor | None,
) -> None:
    if slot_mapping is None or slot_mapping.numel() == 0:
        return
    # Align with RTP paged KV cache layout directly and avoid aiter asm_layout path.
    slots = slot_mapping.to(device=key.device, dtype=torch.int64, non_blocking=True)
    valid = slots >= 0
    slots = slots[valid].contiguous()
    if slots.numel() == 0:
        return
    key = key[valid].contiguous()
    value = value[valid].contiguous()
    if _reshape_and_cache_shuffle_triton(
        key=key,
        value=value,
        key_cache=k_cache,
        value_cache=v_cache,
        slot_mapping=slots,
    ):
        return
    block_size = int(k_cache.shape[2])
    block_ids = torch.div(slots, block_size, rounding_mode="floor")
    offsets = torch.remainder(slots, block_size)
    k_cache[block_ids, :, offsets, :] = key
    v_cache[block_ids, :, offsets, :] = value


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


def _write_kv_cache_with_rtp_fused_kernel(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_cache: object,
    attn_inputs: object,
    tokens_per_block: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    layer: int | None = None,
) -> bool:
    try:
        from rtp_llm.ops import AttentionConfigs, KvCacheDataType
        from rtp_llm.ops.compute_ops import (
            FusedRopeKVCacheDecodeOpNonAsm,
            FusedRopeKVCachePrefillOpNonAsm,
        )
    except Exception:
        return False

    try:
        attn_configs = AttentionConfigs()
        attn_configs.head_num = int(num_heads)
        attn_configs.kv_head_num = int(num_kv_heads)
        attn_configs.size_per_head = int(head_dim)
        attn_configs.tokens_per_block = int(tokens_per_block)
        attn_configs.kernel_tokens_per_block = int(tokens_per_block)
        attn_configs.is_causal = True
        attn_configs.use_mla = False
        attn_configs.q_scaling = 1.0
        attn_configs.dtype = query.dtype
        kv_dtype = getattr(getattr(layer_cache, "kv_cache_base", None), "dtype", None)
        if kv_dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            attn_configs.kv_cache_dtype = KvCacheDataType.FP8
        else:
            attn_configs.kv_cache_dtype = KvCacheDataType.BASE

        # Keep RoPE mocked on ATOM side for this experiment;
        # we only reuse RTP fused address/write semantics here.
        attn_configs.need_rope_kv_cache = False

        qkv = torch.cat(
            [
                query.reshape(query.shape[0], -1),
                key.reshape(key.shape[0], -1),
                value.reshape(value.shape[0], -1),
            ],
            dim=-1,
        ).contiguous()
        is_prefill = bool(getattr(attn_inputs, "is_prefill", False))
        if is_prefill:
            op = FusedRopeKVCachePrefillOpNonAsm(attn_configs)
        else:
            op = FusedRopeKVCacheDecodeOpNonAsm(attn_configs)
        params = op.prepare(attn_inputs)
        _ = op.forward(qkv, layer_cache, params)
        return True
    except Exception:
        return False


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


def _build_slot_mapping_for_layer(
    *,
    positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    block_tables: torch.Tensor,
    seq_size_per_block: int,
) -> torch.Tensor:
    if block_tables.dim() == 1:
        block_tables = block_tables.unsqueeze(0)
    lengths = query_start_loc[1:] - query_start_loc[:-1]
    seq_id = torch.repeat_interleave(
        torch.arange(int(lengths.numel()), device=positions.device, dtype=torch.int64),
        lengths.to(dtype=torch.int64),
    )
    block_col = torch.div(
        positions.to(dtype=torch.int32), int(seq_size_per_block), rounding_mode="floor"
    )
    slot_base = block_tables.to(dtype=torch.int64)[seq_id, block_col.to(dtype=torch.int64)]
    token_offset = torch.remainder(positions.to(dtype=torch.int32), int(seq_size_per_block))
    return (slot_base * int(seq_size_per_block) + token_offset.to(dtype=torch.int64)).contiguous()


def _build_decode_slot_mapping_for_layer(
    *,
    seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    seq_size_per_block: int,
) -> torch.Tensor:
    if block_tables.dim() == 1:
        block_tables = block_tables.unsqueeze(0)
    if seq_lens.dim() != 1:
        seq_lens = seq_lens.view(-1)
    if int(seq_lens.numel()) != int(block_tables.shape[0]):
        raise ValueError(
            "RTPAttention decode slot mapping requires seq_lens/block_tables batch match "
            f"(seq_lens={int(seq_lens.numel())}, block_tables={int(block_tables.shape[0])})."
        )
    # Decode writes one token per sequence at logical index (seq_len - 1),
    # matching RTP decode path semantics.
    token_idx = torch.clamp(seq_lens.to(torch.int64) - 1, min=0)
    block_col = torch.div(
        token_idx, int(seq_size_per_block), rounding_mode="floor"
    ).to(torch.int64)
    row_idx = torch.arange(
        int(seq_lens.numel()), device=seq_lens.device, dtype=torch.int64
    )
    slot_base = block_tables.to(torch.int64)[row_idx, block_col]
    token_offset = torch.remainder(token_idx, int(seq_size_per_block))
    return (slot_base * int(seq_size_per_block) + token_offset).contiguous()


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
        paged_kv = reshape_paged_kv_cache(
            raw,
            num_kv_heads=self.num_kv_heads,
            tokens_per_block=kernel_seq_size_per_block,
            head_dim=self.head_dim,
        )
        if paged_kv.dim() != 5 or int(paged_kv.shape[1]) != 2:
            raise ValueError(
                f"RTPAttention expects paged kv cache [num_blocks,2,H,T,D], got {tuple(paged_kv.shape)}"
            )

        key_cache = paged_kv.select(1, 0)
        value_cache = paged_kv.select(1, 1)
        target_kv_heads = int(key_cache.shape[1])
        k, v, _ = _align_kv_heads_for_cache(
            key=k,
            value=v,
            target_kv_heads=target_kv_heads,
        )
        query_start_loc = getattr(attn_metadata.plugin_metadata, "query_start_loc", None)
        block_tables = _resolve_block_tables_for_layer(attn_inputs, int(self.layer_num))
        if block_tables is None or block_tables.numel() == 0:
            raise ValueError(
                f"RTPAttention requires block table for layer_{self.layer_num}."
            )
        seq_size_per_block = kernel_seq_size_per_block
        positions = getattr(getattr(fwd_ctx, "context", None), "positions", None)
        seq_lens = getattr(attn_metadata.plugin_metadata, "seq_lens", None)
        if block_tables is None or seq_lens is None:
            raise ValueError("RTPAttention requires block tables and sequence lengths.")
        block_tables = block_tables.to(device=q.device, dtype=torch.int32, non_blocking=True)
        seq_lens = seq_lens.to(device=q.device, dtype=torch.int32, non_blocking=True)
        if is_prefill:
            if positions is None or positions.numel() == 0:
                raise ValueError("RTPAttention prefill requires non-empty positions.")
            if query_start_loc is None or query_start_loc.numel() < 2:
                raise ValueError("RTPAttention prefill requires valid query_start_loc.")
            slot_mapping = _build_slot_mapping_for_layer(
                positions=positions.to(
                    device=q.device, dtype=torch.int32, non_blocking=True
                ),
                query_start_loc=query_start_loc.to(
                    device=q.device, dtype=torch.int32, non_blocking=True
                ),
                block_tables=block_tables,
                seq_size_per_block=seq_size_per_block,
            )
        else:
            slot_mapping = _build_decode_slot_mapping_for_layer(
                seq_lens=seq_lens[: int(q.shape[0])],
                block_tables=block_tables[: int(q.shape[0])],
                seq_size_per_block=seq_size_per_block,
            )
        used_fused_write = _write_kv_cache_with_rtp_fused_kernel(
            query=q,
            key=k,
            value=v,
            layer_cache=layer_cache,
            attn_inputs=attn_inputs,
            tokens_per_block=seq_size_per_block,
            num_heads=self.num_heads,
            num_kv_heads=int(k.shape[1]),
            head_dim=self.head_dim,
            layer=int(self.layer_num),
        )
        if not used_fused_write:
            _write_kv_cache_from_slot_mapping(
                key_cache,
                value_cache,
                k,
                v,
                slot_mapping,
            )
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

"""Contract-executable sparse MLA backend for GLM5 rtp-llm plugin mode."""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from typing import Any, Optional

import torch


class _SparseUnavailable(RuntimeError):
    pass


@dataclass
class _AbsorbedWeights:
    w_kc: torch.Tensor
    w_vc: torch.Tensor


@dataclass
class _AtomSparseMetadata:
    qo_indptr: torch.Tensor
    paged_kv_indptr: torch.Tensor
    paged_kv_indices: torch.Tensor
    paged_kv_last_page_len: torch.Tensor
    work_meta_data: torch.Tensor
    work_indptr: torch.Tensor
    work_info_set: torch.Tensor
    reduce_indptr: torch.Tensor
    reduce_final_map: torch.Tensor
    reduce_partial_map: torch.Tensor
    padded_num_heads: int
    head_repeat_factor: int


class _ContractSparseMlaImpl:
    """CPU/mock sparse implementation used before the real RTP kernel is wired."""

    def __init__(self, v_head_dim: int) -> None:
        self.v_head_dim = int(v_head_dim)
        self.calls = []

    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: object,
        layer_id: int,
        *,
        topk_indices: torch.Tensor,
        attn_metadata: object,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.calls.append(
            {
                "q": q,
                "compressed_kv": compressed_kv,
                "k_pe": k_pe,
                "kv_cache": kv_cache,
                "layer_id": layer_id,
                "topk_indices": topk_indices,
                "attn_metadata": attn_metadata,
                "positions": positions,
            }
        )
        return q.new_zeros((q.shape[0], q.shape[1], self.v_head_dim))


class _RealSparseMlaImpl:
    """Runtime sparse MLA adapter for ATOM-owned GLM5 weights and RTP KV cache."""

    def __init__(
        self,
        *,
        mla_modules: Any,
        v_head_dim: int,
        scale: Optional[float] = None,
    ) -> None:
        self.mla_modules = mla_modules
        self.v_head_dim = int(v_head_dim)
        self.kv_lora_rank = int(getattr(mla_modules, "kv_lora_rank"))
        self.qk_nope_head_dim = int(getattr(mla_modules, "qk_nope_head_dim"))
        self.qk_rope_head_dim = int(getattr(mla_modules, "qk_rope_head_dim"))
        self.num_heads = int(getattr(mla_modules, "num_heads", 0) or 0)
        self.rotary_emb = getattr(mla_modules, "rotary_emb", None)
        self.kv_b_proj = getattr(mla_modules, "kv_b_proj", None)
        self.scale = (
            float(scale)
            if scale is not None
            else float((self.qk_nope_head_dim + self.qk_rope_head_dim) ** -0.5)
        )
        self._absorbed_weights: _AbsorbedWeights | None = None
        self._cache_write_scale: dict[torch.device, torch.Tensor] = {}

    @staticmethod
    def _unwrap_linear_output(value: Any) -> torch.Tensor:
        if isinstance(value, tuple):
            value = value[0]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected kv_b_proj to return Tensor, got {type(value)!r}.")
        return value

    def _infer_num_heads(self, q: torch.Tensor) -> int:
        if self.num_heads > 0:
            return self.num_heads
        self.num_heads = int(q.shape[1])
        return self.num_heads

    def _read_kv_b_proj_weight(self) -> torch.Tensor:
        if self.kv_b_proj is None:
            raise _SparseUnavailable("GLM5 RTP sparse MLA requires kv_b_proj.")
        try:
            from atom.model_ops.utils import get_and_maybe_dequant_weights

            weight = get_and_maybe_dequant_weights(self.kv_b_proj)
        except Exception:
            weight = getattr(self.kv_b_proj, "weight", None)
        if not isinstance(weight, torch.Tensor):
            raise _SparseUnavailable("GLM5 RTP sparse MLA cannot read kv_b_proj.weight.")
        if weight.dtype in (
            getattr(torch, "float8_e4m3fn", None),
            getattr(torch, "float8_e4m3fnuz", None),
            getattr(torch, "float8_e5m2", None),
            getattr(torch, "float8_e5m2fnuz", None),
        ):
            raise _SparseUnavailable(
                "GLM5 RTP sparse MLA needs dequantized kv_b_proj weights for "
                "the current adapter."
            )
        return weight

    def _get_absorbed_weights(self, q: torch.Tensor) -> _AbsorbedWeights:
        cached = self._absorbed_weights
        if cached is not None and cached.w_kc.device == q.device:
            return cached

        weight = self._read_kv_b_proj_weight().to(device=q.device)
        num_heads = self._infer_num_heads(q)
        expected_out = num_heads * (self.qk_nope_head_dim + self.v_head_dim)
        if weight.ndim != 2:
            raise _SparseUnavailable(
                f"GLM5 RTP sparse MLA got invalid kv_b_proj weight shape {tuple(weight.shape)}."
            )
        if int(weight.shape[0]) == expected_out and int(weight.shape[1]) == self.kv_lora_rank:
            kv_b_weight = weight.T.contiguous()
        elif int(weight.shape[1]) == expected_out and int(weight.shape[0]) == self.kv_lora_rank:
            kv_b_weight = weight.contiguous()
        else:
            raise _SparseUnavailable(
                "GLM5 RTP sparse MLA kv_b_proj weight shape mismatch "
                f"(got={tuple(weight.shape)}, expected_out={expected_out}, "
                f"kv_lora_rank={self.kv_lora_rank})."
            )

        kv_b_weight = kv_b_weight.view(
            self.kv_lora_rank,
            num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        w_uk, w_uv = kv_b_weight.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        absorbed = _AbsorbedWeights(
            w_kc=w_uk.permute(1, 2, 0).contiguous(),
            w_vc=w_uv.permute(1, 0, 2).contiguous(),
        )
        self._absorbed_weights = absorbed
        return absorbed

    def _apply_rope(
        self,
        q: torch.Tensor,
        k_pe: torch.Tensor,
        positions: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rope_dim = int(self.qk_rope_head_dim)
        if rope_dim == 0:
            return q, k_pe
        if self.rotary_emb is None:
            raise _SparseUnavailable("GLM5 RTP sparse MLA requires rotary_emb.")
        if positions is None or int(positions.numel()) != int(q.shape[0]):
            raise _SparseUnavailable(
                "GLM5 RTP sparse MLA requires per-token positions for RoPE "
                f"(positions={None if positions is None else int(positions.numel())}, "
                f"tokens={int(q.shape[0])})."
            )
        q_rope = q.clone()
        k_pe_rope = k_pe.clone()
        rotated_q_pe, rotated_k_pe = self.rotary_emb(
            positions.to(device=q.device, dtype=torch.long),
            q_rope[..., -rope_dim:],
            k_pe_rope,
        )
        q_rope[..., -rope_dim:] = rotated_q_pe
        return q_rope, rotated_k_pe

    def _cache_dtype_name(self, kv_cache_base: torch.Tensor) -> str:
        fp8_dtypes = {
            dtype
            for dtype in (
                getattr(torch, "float8_e4m3fn", None),
                getattr(torch, "float8_e4m3fnuz", None),
                getattr(torch, "float8_e5m2", None),
                getattr(torch, "float8_e5m2fnuz", None),
                torch.uint8,
            )
            if dtype is not None
        }
        if kv_cache_base.dtype not in fp8_dtypes:
            return "auto"
        explicit = os.getenv("ATOM_RTP_MLA_FP8_CACHE_DTYPE", "").strip()
        if explicit:
            return explicit
        return "fp8_model1_mla" if self.kv_lora_rank == 448 else "fp8_ds_mla"

    def _write_current_to_cache(
        self,
        *,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Any,
        attn_metadata: Any,
    ) -> torch.Tensor:
        kv_cache_base = getattr(kv_cache, "kv_cache_base", None)
        if not isinstance(kv_cache_base, torch.Tensor) or kv_cache_base.numel() == 0:
            raise _SparseUnavailable("GLM5 RTP sparse MLA requires kv_cache_base.")
        slot_mapping = getattr(attn_metadata, "slot_mapping", None)
        if slot_mapping is None:
            plugin_metadata = getattr(attn_metadata, "plugin_metadata", None)
            slot_mapping = getattr(plugin_metadata, "slot_mapping", None)
        if not isinstance(slot_mapping, torch.Tensor):
            raise _SparseUnavailable("GLM5 RTP sparse MLA requires slot_mapping.")
        try:
            from aiter import concat_and_cache_mla
        except Exception as exc:
            raise _SparseUnavailable(f"aiter.concat_and_cache_mla unavailable: {exc}") from exc

        scale = self._cache_write_scale.get(compressed_kv.device)
        if scale is None:
            scale = torch.tensor(1.0, dtype=torch.float32, device=compressed_kv.device)
            self._cache_write_scale[compressed_kv.device] = scale
        try:
            concat_and_cache_mla(
                compressed_kv,
                k_pe,
                kv_cache_base,
                slot_mapping.to(device=compressed_kv.device, dtype=torch.int64),
                kv_cache_dtype=self._cache_dtype_name(kv_cache_base),
                scale=scale,
            )
        except Exception as exc:
            raise _SparseUnavailable(f"concat_and_cache_mla failed: {exc}") from exc
        return kv_cache_base

    @staticmethod
    def _build_req_id_per_token(
        attn_metadata: Any,
        num_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        plugin_metadata = getattr(attn_metadata, "plugin_metadata", None)
        req_id = getattr(plugin_metadata, "req_id_per_token", None)
        if isinstance(req_id, torch.Tensor) and int(req_id.numel()) >= num_tokens:
            return req_id[:num_tokens].to(device=device, dtype=torch.int32)
        query_start_loc = getattr(plugin_metadata, "query_start_loc", None)
        if query_start_loc is None:
            query_start_loc = getattr(plugin_metadata, "rtp_cu_seqlens_q", None)
        if query_start_loc is None:
            query_start_loc = getattr(attn_metadata, "cu_seqlens_q", None)
        if isinstance(query_start_loc, torch.Tensor) and int(query_start_loc.numel()) >= 2:
            qsl = query_start_loc.to(device=device, dtype=torch.int64)
            lengths = qsl[1:] - qsl[:-1]
            return torch.repeat_interleave(
                torch.arange(int(lengths.numel()), device=device, dtype=torch.int32),
                lengths,
            )[:num_tokens].contiguous()
        return torch.arange(num_tokens, device=device, dtype=torch.int32)

    @staticmethod
    def _block_table(attn_metadata: Any, device: torch.device) -> torch.Tensor:
        plugin_metadata = getattr(attn_metadata, "plugin_metadata", None)
        block_table = getattr(plugin_metadata, "block_table", None)
        if block_table is None:
            block_table = getattr(attn_metadata, "block_tables", None)
        if not isinstance(block_table, torch.Tensor):
            raise _SparseUnavailable("GLM5 RTP sparse MLA requires block_table.")
        if block_table.ndim == 1:
            block_table = block_table.unsqueeze(0)
        return block_table.to(device=device, dtype=torch.int32)

    @staticmethod
    def _convert_topk_to_global(
        *,
        topk_indices: torch.Tensor,
        attn_metadata: Any,
        block_size: int,
    ) -> torch.Tensor:
        num_tokens, topk = topk_indices.shape
        device = topk_indices.device
        block_table = _RealSparseMlaImpl._block_table(attn_metadata, device)
        req_id = _RealSparseMlaImpl._build_req_id_per_token(
            attn_metadata, num_tokens, device
        ).to(dtype=torch.long)
        token_indices = topk_indices.to(device=device, dtype=torch.long)
        valid = token_indices >= 0
        block_cols = torch.div(
            torch.clamp(token_indices, min=0),
            int(block_size),
            rounding_mode="floor",
        )
        offsets = torch.remainder(torch.clamp(token_indices, min=0), int(block_size))
        valid = valid & (req_id[:, None] >= 0) & (req_id[:, None] < block_table.shape[0])
        valid = valid & (block_cols >= 0) & (block_cols < block_table.shape[1])
        safe_req = torch.clamp(req_id, min=0, max=max(int(block_table.shape[0]) - 1, 0))
        safe_cols = torch.clamp(block_cols, min=0, max=max(int(block_table.shape[1]) - 1, 0))
        block_ids = block_table.to(dtype=torch.long)[safe_req[:, None], safe_cols]
        valid = valid & (block_ids >= 0)
        global_indices = block_ids * int(block_size) + offsets
        return torch.where(valid, global_indices, torch.zeros_like(global_indices)).to(
            dtype=torch.int32
        )

    @staticmethod
    def _decode_indptr(
        *,
        num_tokens: int,
        topk: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qo_indptr = torch.arange(num_tokens + 1, device=device, dtype=torch.int32)
        paged_kv_indptr = (
            torch.arange(num_tokens + 1, device=device, dtype=torch.int32) * int(topk)
        )
        paged_kv_last_page_len = torch.ones(
            (num_tokens,), device=device, dtype=torch.int32
        )
        return qo_indptr, paged_kv_indptr, paged_kv_last_page_len

    @staticmethod
    def _generate_sparse_seqlen_torch(
        *,
        query_lens: torch.Tensor,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
        topk: int,
        num_tokens: int,
    ) -> torch.Tensor:
        out = torch.zeros((num_tokens,), dtype=torch.int32, device=query_lens.device)
        for req_id in range(int(query_lens.numel())):
            q_len = int(query_lens[req_id].item())
            seq_len = int(seq_lens[req_id].item())
            start = int(query_start_loc[req_id].item())
            if q_len <= 0 or seq_len <= 0:
                continue
            context_start = seq_len - q_len
            offsets = torch.arange(q_len, device=query_lens.device, dtype=torch.int32)
            out[start : start + q_len] = torch.clamp(
                context_start + offsets + 1,
                min=0,
                max=int(topk),
            )
        return out

    @staticmethod
    def _aiter_dtype_for_tensor(tensor: torch.Tensor) -> Any:
        try:
            from aiter import dtypes
        except Exception as exc:
            raise _SparseUnavailable(f"aiter dtypes unavailable: {exc}") from exc

        fp8_dtypes = {
            dtype
            for dtype in (
                getattr(torch, "float8_e4m3fn", None),
                getattr(torch, "float8_e4m3fnuz", None),
                getattr(torch, "float8_e5m2", None),
                getattr(torch, "float8_e5m2fnuz", None),
                torch.uint8,
                getattr(dtypes, "fp8", None),
            )
            if dtype is not None
        }
        if tensor.dtype in fp8_dtypes:
            return dtypes.fp8
        if tensor.dtype == torch.float16:
            return dtypes.d_dtypes["fp16"]
        return dtypes.d_dtypes["bf16"]

    def _build_atom_sparse_metadata(
        self,
        *,
        q_latent: torch.Tensor,
        kv_cache_base: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: Any,
        block_size: int,
    ) -> _AtomSparseMetadata:
        if torch.cuda.is_current_stream_capturing():
            raise _SparseUnavailable("ATOM sparse MLA metadata is not graph-capture safe yet.")
        try:
            from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
            from atom.plugin.attention_mla_sparse import (
                generate_sparse_seqlen_triton,
                triton_convert_req_index_to_global_index,
            )
        except Exception as exc:
            raise _SparseUnavailable(f"ATOM sparse MLA metadata helpers unavailable: {exc}") from exc

        plugin_metadata = getattr(attn_metadata, "plugin_metadata", None)
        if plugin_metadata is None:
            raise _SparseUnavailable("GLM5 RTP sparse MLA requires plugin metadata.")

        num_tokens = int(q_latent.shape[0])
        num_heads = int(q_latent.shape[1])
        topk = int(topk_indices.shape[1])
        device = q_latent.device

        query_start_loc = getattr(plugin_metadata, "query_start_loc", None)
        if query_start_loc is None:
            query_start_loc = getattr(plugin_metadata, "rtp_cu_seqlens_q", None)
        if not isinstance(query_start_loc, torch.Tensor) or int(query_start_loc.numel()) < 2:
            raise _SparseUnavailable("GLM5 RTP sparse MLA requires query_start_loc.")
        query_start_loc = query_start_loc.to(device=device, dtype=torch.int32).contiguous()

        seq_lens = getattr(plugin_metadata, "seq_lens", None)
        if seq_lens is None:
            seq_lens = getattr(attn_metadata, "context_lens", None)
        if not isinstance(seq_lens, torch.Tensor) or int(seq_lens.numel()) + 1 != int(
            query_start_loc.numel()
        ):
            raise _SparseUnavailable("GLM5 RTP sparse MLA requires seq_lens per request.")
        seq_lens = seq_lens.to(device=device, dtype=torch.int32).contiguous()

        req_id = self._build_req_id_per_token(attn_metadata, num_tokens, device).to(
            dtype=torch.int32
        )
        block_table = self._block_table(attn_metadata, device).to(dtype=torch.int32)
        topk_indices_i32 = topk_indices.to(device=device, dtype=torch.int32).contiguous()
        query_lens = (query_start_loc[1:] - query_start_loc[:-1]).contiguous()

        if device.type == "cpu":
            sparse_seqlen = self._generate_sparse_seqlen_torch(
                query_lens=query_lens,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                topk=topk,
                num_tokens=num_tokens,
            )
        else:
            sparse_seqlen = generate_sparse_seqlen_triton(
                query_lens,
                seq_lens,
                query_start_loc,
                topk,
                num_tokens,
                int(torch.max(query_lens).detach().cpu().item()) if num_tokens else 1,
            )

        qo_indptr = torch.arange(num_tokens + 1, device=device, dtype=torch.int32)
        paged_kv_indptr = torch.zeros((num_tokens + 1,), device=device, dtype=torch.int32)
        torch.cumsum(sparse_seqlen, dim=0, out=paged_kv_indptr[1:])
        paged_kv_last_page_len = torch.ones((num_tokens,), device=device, dtype=torch.int32)
        paged_kv_indices = torch.zeros((num_tokens * topk,), device=device, dtype=torch.int32)

        triton_convert_req_index_to_global_index(
            req_id,
            block_table,
            topk_indices_i32,
            paged_kv_indptr,
            paged_kv_indices,
            BLOCK_SIZE=int(block_size),
            NUM_TOPK_TOKENS=topk,
        )

        padded_num_heads = max(num_heads, 16)
        if padded_num_heads % num_heads != 0:
            padded_num_heads = ((padded_num_heads + num_heads - 1) // num_heads) * num_heads
        head_repeat_factor = padded_num_heads // num_heads
        q_dtype = self._aiter_dtype_for_tensor(q_latent)
        kv_dtype = self._aiter_dtype_for_tensor(kv_cache_base)
        (
            (work_meta_data_size, work_meta_data_type),
            (work_indptr_size, work_indptr_type),
            (work_info_set_size, work_info_set_type),
            (reduce_indptr_size, reduce_indptr_type),
            (reduce_final_map_size, reduce_final_map_type),
            (reduce_partial_map_size, reduce_partial_map_type),
        ) = get_mla_metadata_info_v1(
            max(num_tokens, 1),
            1,
            padded_num_heads,
            q_dtype,
            kv_dtype,
            is_sparse=True,
            fast_mode=True,
        )
        work_meta_data = torch.empty(work_meta_data_size, dtype=work_meta_data_type, device=device)
        work_indptr = torch.empty(work_indptr_size, dtype=work_indptr_type, device=device)
        work_info_set = torch.empty(work_info_set_size, dtype=work_info_set_type, device=device)
        reduce_indptr = torch.empty(reduce_indptr_size, dtype=reduce_indptr_type, device=device)
        reduce_final_map = torch.empty(
            reduce_final_map_size, dtype=reduce_final_map_type, device=device
        )
        reduce_partial_map = torch.empty(
            reduce_partial_map_size, dtype=reduce_partial_map_type, device=device
        )
        get_mla_metadata_v1(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_last_page_len,
            padded_num_heads,
            1,
            True,
            work_meta_data,
            work_info_set,
            work_indptr,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            page_size=1,
            kv_granularity=16,
            max_seqlen_qo=1,
            uni_seqlen_qo=1,
            fast_mode=True,
            dtype_q=q_dtype,
            dtype_kv=kv_dtype,
        )
        return _AtomSparseMetadata(
            qo_indptr=qo_indptr,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            work_meta_data=work_meta_data,
            work_indptr=work_indptr,
            work_info_set=work_info_set,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
            padded_num_heads=padded_num_heads,
            head_repeat_factor=head_repeat_factor,
        )

    def _run_sparse_decode(
        self,
        *,
        q_latent: torch.Tensor,
        kv_cache_base: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: Any,
        block_size: int,
    ) -> torch.Tensor:
        try:
            from flash_mla import flash_mla_sparse_fwd
        except Exception as exc:
            return self._run_aiter_sparse_decode(
                q_latent=q_latent,
                kv_cache_base=kv_cache_base,
                topk_indices=topk_indices,
                attn_metadata=attn_metadata,
                block_size=block_size,
            )

        latent_dim = int(q_latent.shape[-1])
        global_topk = self._convert_topk_to_global(
            topk_indices=topk_indices,
            attn_metadata=attn_metadata,
            block_size=block_size,
        )
        try:
            kv_buffer = kv_cache_base.reshape(-1, latent_dim)
            output, _, _ = flash_mla_sparse_fwd(
                q_latent,
                kv_buffer,
                global_topk.contiguous().unsqueeze(1),
                self.scale,
                d_v=self.kv_lora_rank,
            )
        except Exception as exc:
            raise _SparseUnavailable(f"flash_mla_sparse_fwd failed: {exc}") from exc
        return output

    def _run_aiter_sparse_decode(
        self,
        *,
        q_latent: torch.Tensor,
        kv_cache_base: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: Any,
        block_size: int,
    ) -> torch.Tensor:
        try:
            from aiter.mla import mla_decode_fwd
        except Exception as exc:
            raise _SparseUnavailable(f"aiter.mla_decode_fwd unavailable: {exc}") from exc

        num_tokens, num_heads, latent_dim = q_latent.shape
        sparse_meta = self._build_atom_sparse_metadata(
            q_latent=q_latent,
            kv_cache_base=kv_cache_base,
            topk_indices=topk_indices,
            attn_metadata=attn_metadata,
            block_size=block_size,
        )
        if sparse_meta.head_repeat_factor > 1:
            q_for_kernel = q_latent.repeat_interleave(
                sparse_meta.head_repeat_factor, dim=1
            )
        else:
            q_for_kernel = q_latent
        output = torch.empty(
            (num_tokens, sparse_meta.padded_num_heads, self.kv_lora_rank),
            dtype=q_for_kernel.dtype,
            device=q_latent.device,
        )
        try:
            kv_buffer = kv_cache_base.reshape(-1, 1, 1, latent_dim)
            mla_decode_fwd(
                q_for_kernel,
                kv_buffer,
                output,
                sparse_meta.qo_indptr,
                sparse_meta.paged_kv_indptr,
                sparse_meta.paged_kv_indices,
                sparse_meta.paged_kv_last_page_len,
                1,
                sm_scale=self.scale,
                page_size=1,
                work_meta_data=sparse_meta.work_meta_data,
                work_indptr=sparse_meta.work_indptr,
                work_info_set=sparse_meta.work_info_set,
                reduce_indptr=sparse_meta.reduce_indptr,
                reduce_final_map=sparse_meta.reduce_final_map,
                reduce_partial_map=sparse_meta.reduce_partial_map,
            )
        except Exception as exc:
            raise _SparseUnavailable(f"mla_decode_fwd failed: {exc}") from exc
        if sparse_meta.head_repeat_factor > 1:
            output = output[:, :: sparse_meta.head_repeat_factor, :].contiguous()
        return output

    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: object,
        layer_id: int,
        *,
        topk_indices: torch.Tensor,
        attn_metadata: object,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del layer_id
        if attn_metadata is None:
            raise _SparseUnavailable("GLM5 RTP sparse MLA requires attn_metadata.")
        if getattr(getattr(attn_metadata, "plugin_metadata", None), "is_dummy_warmup", False):
            raise _SparseUnavailable("GLM5 RTP sparse MLA skips dummy warmup.")
        q_rope, k_pe_rope = self._apply_rope(q, k_pe, positions)
        kv_cache_base = self._write_current_to_cache(
            compressed_kv=compressed_kv,
            k_pe=k_pe_rope,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        absorbed = self._get_absorbed_weights(q_rope)
        q_nope = q_rope[..., : self.qk_nope_head_dim]
        q_latent_nope = torch.bmm(
            q_nope.transpose(0, 1).to(dtype=absorbed.w_kc.dtype),
            absorbed.w_kc,
        ).transpose(0, 1)
        q_latent = torch.empty(
            q.shape[0],
            q.shape[1],
            self.kv_lora_rank + self.qk_rope_head_dim,
            dtype=q_latent_nope.dtype,
            device=q.device,
        )
        q_latent[..., : self.kv_lora_rank] = q_latent_nope
        if self.qk_rope_head_dim > 0:
            q_latent[..., self.kv_lora_rank :] = q_rope[
                ..., -self.qk_rope_head_dim :
            ].to(dtype=q_latent.dtype)

        block_size = int(getattr(attn_metadata, "rtp_seq_size_per_block", 0) or 0)
        if block_size <= 0:
            plugin_metadata = getattr(attn_metadata, "plugin_metadata", None)
            block_size = int(getattr(plugin_metadata, "sparse_block_size", 0) or 0)
        if block_size <= 0:
            raise _SparseUnavailable("GLM5 RTP sparse MLA requires physical block size.")
        latent_output = self._run_sparse_decode(
            q_latent=q_latent,
            kv_cache_base=kv_cache_base,
            topk_indices=topk_indices,
            attn_metadata=attn_metadata,
            block_size=block_size,
        )
        output = torch.bmm(
            latent_output.transpose(0, 1).to(dtype=absorbed.w_vc.dtype),
            absorbed.w_vc,
        ).transpose(0, 1)
        return output.to(dtype=q.dtype)


class RTPSparseMlaBackend:
    """M2 sparse top-k consumption contract.

    This backend intentionally avoids importing RTP CUDA sparse kernels. It only
    validates and threads the sparse contract so M2.5 can replace the mock impl.
    """

    def __init__(
        self,
        *,
        dense_backend: object,
        sparse_impl: Optional[object] = None,
        v_head_dim: Optional[int] = None,
        mla_modules: Optional[object] = None,
        scale: Optional[float] = None,
    ) -> None:
        self.dense_backend = dense_backend
        self.v_head_dim = int(
            v_head_dim
            if v_head_dim is not None
            else getattr(dense_backend, "v_head_dim")
        )
        if sparse_impl is not None:
            self.sparse_impl = sparse_impl
            self._default_mock = False
        elif mla_modules is not None and all(
            hasattr(mla_modules, attr)
            for attr in (
                "kv_lora_rank",
                "qk_nope_head_dim",
                "qk_rope_head_dim",
                "kv_b_proj",
                "rotary_emb",
            )
        ):
            self.sparse_impl = _RealSparseMlaImpl(
                mla_modules=mla_modules,
                v_head_dim=self.v_head_dim,
                scale=scale,
            )
            self._default_mock = False
        else:
            self.sparse_impl = _ContractSparseMlaImpl(self.v_head_dim)
            self._default_mock = True

    @staticmethod
    def _get_attn_metadata() -> object:
        try:
            from atom.utils.forward_context import get_forward_context

            return getattr(get_forward_context(), "attn_metadata", None)
        except Exception:
            return None

    @staticmethod
    def _validate_topk_indices(q: torch.Tensor, topk_indices: torch.Tensor) -> None:
        if topk_indices.ndim != 2:
            raise ValueError(
                "Expected topk_indices to be rank-2 [T,K], "
                f"got shape {tuple(topk_indices.shape)}"
            )
        if topk_indices.dtype != torch.int32:
            raise ValueError(
                f"Expected topk_indices dtype torch.int32, got {topk_indices.dtype}"
            )
        if topk_indices.shape[0] != q.shape[0]:
            raise ValueError(
                "Expected topk_indices first dimension to match q tokens, "
                f"got {topk_indices.shape[0]} and {q.shape[0]}"
            )

    @staticmethod
    def _enable_sparse_mock() -> bool:
        return os.getenv("ATOM_RTP_ENABLE_SPARSE_MLA_MOCK", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    @staticmethod
    def _strict_sparse() -> bool:
        return os.getenv("ATOM_RTP_SPARSE_MLA_STRICT", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    @staticmethod
    def _impl_accepts_positions(impl: object) -> bool:
        try:
            signature = inspect.signature(impl.forward)
        except (AttributeError, TypeError, ValueError):
            return False
        return "positions" in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

    @staticmethod
    def _call_accepts_positions(callable_obj: object) -> bool:
        try:
            signature = inspect.signature(callable_obj)
        except (TypeError, ValueError):
            return False
        return "positions" in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

    def _dense_forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: object,
        layer_id: int,
        topk_indices: Optional[torch.Tensor],
        positions: Optional[torch.Tensor],
    ) -> torch.Tensor:
        kwargs = {"topk_indices": topk_indices}
        if self._call_accepts_positions(self.dense_backend.forward):
            kwargs["positions"] = positions
        return self.dense_backend.forward(
            q,
            compressed_kv,
            k_pe,
            kv_cache,
            layer_id,
            **kwargs,
        )

    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: object,
        layer_id: int,
        topk_indices: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if topk_indices is None:
            return self._dense_forward(
                q, compressed_kv, k_pe, kv_cache, layer_id, None, positions
            )

        self._validate_topk_indices(q, topk_indices)
        if (
            (self._default_mock and not self._enable_sparse_mock())
            or not callable(getattr(self.sparse_impl, "forward", None))
        ):
            return self._dense_forward(
                q, compressed_kv, k_pe, kv_cache, layer_id, topk_indices, positions
            )

        kwargs = {
            "topk_indices": topk_indices,
            "attn_metadata": self._get_attn_metadata(),
        }
        if self._impl_accepts_positions(self.sparse_impl):
            kwargs["positions"] = positions
        try:
            return self.sparse_impl.forward(
                q,
                compressed_kv,
                k_pe,
                kv_cache,
                layer_id,
                **kwargs,
            )
        except _SparseUnavailable:
            if self._strict_sparse():
                raise
            return self._dense_forward(
                q, compressed_kv, k_pe, kv_cache, layer_id, topk_indices, positions
            )

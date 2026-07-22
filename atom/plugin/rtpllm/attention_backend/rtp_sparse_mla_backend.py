"""Sparse MLA backend for GLM5 rtp-llm plugin mode."""

from __future__ import annotations

import importlib
import inspect
import os
from dataclasses import dataclass
from typing import Any, Optional

import torch

from atom.utils.custom_register import direct_register_custom_op


class _SparseUnavailable(RuntimeError):
    pass


def _resolve_plugin_sparse_index_converter():
    """Resolve the plugin-style request-local topk to global KV index converter."""
    errors: list[str] = []
    for module_name in (
        # Compatibility import path used by earlier plugin layouts.
        "atom.plugin.attention_mla_sparse",
        # Current plugin helper location with the same call signature.
        "atom.plugin.vllm.attention.layer_sparse_mla",
    ):
        try:
            module = importlib.import_module(module_name)
            return getattr(module, "triton_convert_req_index_to_global_index")
        except Exception as exc:
            errors.append(f"{module_name}: {exc}")
    raise _SparseUnavailable(
        "plugin sparse MLA index converter unavailable; " + "; ".join(errors)
    )


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
    page_size: int


class _LightweightSparseMlaImpl:
    """Lightweight implementation for unit tests and explicit dependency injection."""

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
        self._cg_sparse_bufs: dict[str, torch.Tensor] | None = None
        self._cg_workspace_signature: tuple[Any, ...] | None = None
        self._enable_sparse_validate = (
            os.getenv("ATOM_RTP_GLM5_SPARSE_VALIDATE", "0") == "1"
        )

    @staticmethod
    def _validate_sparse_index_contract(
        *,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        num_tokens: int,
        page_size: int,
        max_slots: int,
    ) -> None:
        if int(paged_kv_indptr.numel()) != num_tokens + 1:
            raise _SparseUnavailable(
                "GLM5 RTP sparse MLA invalid paged_kv_indptr length "
                f"(got={int(paged_kv_indptr.numel())}, expected={num_tokens + 1})."
            )
        if int(paged_kv_indptr[0].item()) != 0:
            raise _SparseUnavailable(
                "GLM5 RTP sparse MLA paged_kv_indptr[0] must be 0, "
                f"got {int(paged_kv_indptr[0].item())}."
            )
        if num_tokens > 0:
            deltas = paged_kv_indptr[1:] - paged_kv_indptr[:-1]
            if bool((deltas < 0).any().item()):
                raise _SparseUnavailable(
                    "GLM5 RTP sparse MLA paged_kv_indptr must be non-decreasing."
                )
        used = int(paged_kv_indptr[-1].item())
        if used < 0 or used > int(paged_kv_indices.numel()):
            raise _SparseUnavailable(
                "GLM5 RTP sparse MLA paged_kv_indptr[-1] out of range "
                f"(used={used}, capacity={int(paged_kv_indices.numel())})."
            )
        if used == 0:
            return
        used_indices = paged_kv_indices[:used]
        min_index = int(used_indices.min().item())
        max_index = int(used_indices.max().item())
        if min_index < 0 or max_index >= max_slots:
            raise _SparseUnavailable(
                "GLM5 RTP sparse MLA produced out-of-range paged_kv_indices "
                f"(min={min_index}, max={max_index}, slots={max_slots}, "
                f"page_size={page_size})."
            )

    @staticmethod
    def _validate_sparse_last_page_contract(
        *,
        paged_kv_indptr: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        num_tokens: int,
        page_size: int,
    ) -> None:
        if int(paged_kv_last_page_len.numel()) != int(num_tokens):
            raise _SparseUnavailable(
                "GLM5 RTP sparse MLA invalid paged_kv_last_page_len length "
                f"(got={int(paged_kv_last_page_len.numel())}, expected={int(num_tokens)})."
            )
        if num_tokens <= 0:
            return
        deltas = paged_kv_indptr[1:] - paged_kv_indptr[:-1]
        active_mask = deltas > 0
        if not bool(active_mask.any().item()):
            return
        active_last_page_len = paged_kv_last_page_len[active_mask]
        min_last_page_len = int(active_last_page_len.min().item())
        max_last_page_len = int(active_last_page_len.max().item())
        if min_last_page_len < 1 or max_last_page_len > int(page_size):
            raise _SparseUnavailable(
                "GLM5 RTP sparse MLA invalid paged_kv_last_page_len range "
                f"(min={min_last_page_len}, max={max_last_page_len}, "
                f"page_size={int(page_size)})."
            )
        if int(page_size) == 1 and bool((active_last_page_len != 1).any().item()):
            raise _SparseUnavailable(
                "GLM5 RTP sparse MLA expects paged_kv_last_page_len==1 when page_size=1."
            )

    @staticmethod
    def _kv_token_slot_capacity(kv_cache_base: torch.Tensor) -> int:
        if kv_cache_base.ndim <= 0:
            return 0
        latent_dim = int(kv_cache_base.shape[-1]) if kv_cache_base.ndim >= 1 else 0
        if latent_dim <= 0:
            return 0
        return int(kv_cache_base.numel() // latent_dim)

    def _infer_num_heads(self, q: torch.Tensor) -> int:
        num_heads = int(q.shape[1])
        if self.num_heads != num_heads:
            self.num_heads = num_heads
        return num_heads

    def _infer_num_heads_from_weight(self, fallback: int) -> int:
        try:
            weight = self._read_kv_b_proj_weight()
        except Exception:
            return int(fallback)
        per_head_dim = int(self.qk_nope_head_dim + self.v_head_dim)
        if per_head_dim <= 0 or weight.ndim != 2:
            return int(fallback)
        for dim in weight.shape:
            dim_i = int(dim)
            if dim_i > 0 and dim_i % per_head_dim == 0:
                candidate = dim_i // per_head_dim
                if candidate > 0:
                    return max(int(fallback), int(candidate))
        return int(fallback)

    def _read_kv_b_proj_weight(self) -> torch.Tensor:
        if self.kv_b_proj is None:
            raise _SparseUnavailable("GLM5 RTP sparse MLA requires kv_b_proj.")
        try:
            from atom.model_ops.utils import get_and_maybe_dequant_weights

            weight = get_and_maybe_dequant_weights(self.kv_b_proj)
        except Exception:
            weight = getattr(self.kv_b_proj, "weight", None)
        if not isinstance(weight, torch.Tensor):
            raise _SparseUnavailable(
                "GLM5 RTP sparse MLA cannot read kv_b_proj.weight."
            )
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
        if (
            int(weight.shape[0]) == expected_out
            and int(weight.shape[1]) == self.kv_lora_rank
        ):
            kv_b_weight = weight.T.contiguous()
        elif (
            int(weight.shape[1]) == expected_out
            and int(weight.shape[0]) == self.kv_lora_rank
        ):
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
        w_uk, w_uv = kv_b_weight.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
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
        in_capture = torch.cuda.is_current_stream_capturing()
        if in_capture:
            if self._cg_sparse_bufs is None:
                raise _SparseUnavailable(
                    "GLM5 RTP sparse MLA capture requires RoPE buffers."
                )
            if positions.device != q.device or positions.dtype != torch.long:
                raise _SparseUnavailable(
                    "GLM5 RTP sparse MLA capture requires int64 positions on device."
                )
            if not positions.is_contiguous():
                raise _SparseUnavailable(
                    "GLM5 RTP sparse MLA capture requires contiguous positions."
                )
            q_rope = self._cg_sparse_bufs["q_rope"][
                : q.shape[0], : q.shape[1], : q.shape[2]
            ]
            q_rope.copy_(q)
            if k_pe.dim() == 2:
                k_pe_rope = self._cg_sparse_bufs["k_pe_rope_2d"][
                    : k_pe.shape[0], : k_pe.shape[1]
                ]
            elif k_pe.dim() == 3 and int(k_pe.shape[1]) == 1:
                k_pe_rope = self._cg_sparse_bufs["k_pe_rope_3d"][
                    : k_pe.shape[0], : k_pe.shape[1], : k_pe.shape[2]
                ]
            elif k_pe.dim() == 3:
                k_pe_rope = self._cg_sparse_bufs["k_pe_rope_heads"][
                    : k_pe.shape[0], : k_pe.shape[1], : k_pe.shape[2]
                ]
            else:
                raise _SparseUnavailable(
                    f"GLM5 RTP sparse MLA capture got invalid k_pe ndim={k_pe.dim()}."
                )
            k_pe_rope.copy_(k_pe)
            rope_positions = positions.view(-1)
        else:
            q_rope = q.clone()
            k_pe_rope = k_pe.clone()
            rope_positions = positions.reshape(-1).to(device=q.device, dtype=torch.long)
        rotated_q_pe, rotated_k_pe = self.rotary_emb(
            rope_positions,
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
        # RTP allocates GLM5 FP8 MLA KV cache in the aiter 576-byte/token layout.
        return "fp8"

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
            raise _SparseUnavailable(
                f"aiter.concat_and_cache_mla unavailable: {exc}"
            ) from exc

        scale = self._cache_write_scale.get(compressed_kv.device)
        if scale is None:
            scale = torch.tensor(1.0, dtype=torch.float32, device=compressed_kv.device)
            self._cache_write_scale[compressed_kv.device] = scale
        in_capture = torch.cuda.is_current_stream_capturing()
        if in_capture:
            if (
                slot_mapping.device != compressed_kv.device
                or slot_mapping.dtype != torch.int64
            ):
                raise _SparseUnavailable(
                    "GLM5 RTP sparse MLA capture requires int64 slot_mapping on device."
                )
            slot_mapping_for_cache = slot_mapping
        else:
            slot_mapping_for_cache = slot_mapping.to(
                device=compressed_kv.device, dtype=torch.int64
            )
        try:
            concat_and_cache_mla(
                compressed_kv,
                k_pe,
                kv_cache_base,
                slot_mapping_for_cache,
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
        if (
            isinstance(query_start_loc, torch.Tensor)
            and int(query_start_loc.numel()) >= 2
        ):
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
        if int(block_size) <= 0:
            raise _SparseUnavailable(
                f"GLM5 RTP sparse MLA requires positive block_size, got {block_size}."
            )
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
        valid = (
            valid & (req_id[:, None] >= 0) & (req_id[:, None] < block_table.shape[0])
        )
        valid = valid & (block_cols >= 0) & (block_cols < block_table.shape[1])
        safe_req = torch.clamp(req_id, min=0, max=max(int(block_table.shape[0]) - 1, 0))
        safe_cols = torch.clamp(
            block_cols, min=0, max=max(int(block_table.shape[1]) - 1, 0)
        )
        block_ids = block_table.to(dtype=torch.long)[safe_req[:, None], safe_cols]
        valid = valid & (block_ids >= 0)
        global_indices = block_ids * int(block_size) + offsets
        return torch.where(valid, global_indices, torch.zeros_like(global_indices)).to(
            dtype=torch.int32
        )

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

    @staticmethod
    def _aiter_dtype_for_torch_dtype(
        dtype: torch.dtype, *, assume_fp8: bool = False
    ) -> Any:
        try:
            from aiter import dtypes
        except Exception as exc:
            raise _SparseUnavailable(f"aiter dtypes unavailable: {exc}") from exc
        if assume_fp8:
            return dtypes.fp8
        if dtype == torch.float16:
            return dtypes.d_dtypes["fp16"]
        return dtypes.d_dtypes["bf16"]

    def _resolve_topk_for_prewarm(self) -> int:
        for obj, attr in (
            (getattr(self.mla_modules, "indexer", None), "index_topk"),
            (getattr(self.mla_modules, "indexer", None), "topk_tokens"),
            (self.mla_modules, "index_topk"),
            (getattr(self.mla_modules, "config", None), "index_topk"),
        ):
            value = getattr(obj, attr, None) if obj is not None else None
            if value is not None:
                return int(value)
        return 2048

    @staticmethod
    def _metadata_token_budget(*, num_tokens: int, topk: int) -> int:
        # Sparse decode can materialize up to num_tokens * topk ragged entries.
        # Use this upper bound to avoid undersized work/reduce metadata buffers.
        return max(int(num_tokens) * max(int(topk), 1), 1)

    @staticmethod
    def _validate_capture_sparse_buffer_capacity(
        *,
        sparse_bufs: dict[str, torch.Tensor],
        num_tokens: int,
        topk: int,
    ) -> None:
        needed_indices = int(num_tokens) * int(topk)
        if int(sparse_bufs["paged_kv_indices"].numel()) < needed_indices:
            raise _SparseUnavailable(
                "GLM5 RTP sparse MLA capture paged_kv_indices buffer is too small "
                f"(buffer={int(sparse_bufs['paged_kv_indices'].numel())}, "
                f"required={needed_indices})."
            )
        if int(sparse_bufs["qo_indptr"].numel()) < int(num_tokens) + 1:
            raise _SparseUnavailable(
                "GLM5 RTP sparse MLA capture qo_indptr buffer is too small."
            )
        if int(sparse_bufs["paged_kv_indptr"].numel()) < int(num_tokens) + 1:
            raise _SparseUnavailable(
                "GLM5 RTP sparse MLA capture paged_kv_indptr buffer is too small."
            )
        if int(sparse_bufs["paged_kv_last_page_len"].numel()) < int(num_tokens):
            raise _SparseUnavailable(
                "GLM5 RTP sparse MLA capture paged_kv_last_page_len buffer is too small."
            )

    def prewarm_for_cuda_graph(
        self,
        *,
        max_num_tokens: int,
        max_seq_len: int,
        query_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        del max_seq_len
        try:
            from aiter import dtypes, get_mla_metadata_info_v1
        except Exception as exc:
            raise _SparseUnavailable(
                f"aiter metadata prewarm unavailable: {exc}"
            ) from exc

        max_tokens = int(max_num_tokens)
        if max_tokens <= 0:
            return
        num_heads = int(
            self.num_heads or getattr(self.mla_modules, "num_local_heads", 0) or 0
        )
        if num_heads <= 0:
            # Lazily inferred in eager path; graph capture needs a stable budget.
            num_heads = int(getattr(self.mla_modules, "num_heads", 0) or 1)
        num_heads = self._infer_num_heads_from_weight(num_heads)
        self.num_heads = num_heads
        padded_num_heads = max(num_heads, 16)
        if padded_num_heads % num_heads != 0:
            padded_num_heads = (
                (padded_num_heads + num_heads - 1) // num_heads
            ) * num_heads
        topk = self._resolve_topk_for_prewarm()
        latent_dim = self.kv_lora_rank + self.qk_rope_head_dim
        q_dtype = self._aiter_dtype_for_torch_dtype(query_dtype)
        kv_dtype = self._aiter_dtype_for_torch_dtype(query_dtype, assume_fp8=True)
        metadata_budget_tokens = self._metadata_token_budget(
            num_tokens=max_tokens, topk=topk
        )
        (
            (work_meta_data_size, work_meta_data_type),
            (work_indptr_size, work_indptr_type),
            (work_info_set_size, work_info_set_type),
            (reduce_indptr_size, reduce_indptr_type),
            (reduce_final_map_size, reduce_final_map_type),
            (reduce_partial_map_size, reduce_partial_map_type),
        ) = get_mla_metadata_info_v1(
            metadata_budget_tokens,
            1,
            padded_num_heads,
            q_dtype,
            kv_dtype,
            is_sparse=True,
            fast_mode=True,
        )
        self._cg_sparse_bufs = {
            "qo_indptr": torch.arange(max_tokens + 1, device=device, dtype=torch.int32),
            "sparse_seqlen": torch.empty(max_tokens, device=device, dtype=torch.int32),
            "paged_kv_indptr": torch.empty(
                max_tokens + 1, device=device, dtype=torch.int32
            ),
            "paged_kv_last_page_len": torch.ones(
                max_tokens, device=device, dtype=torch.int32
            ),
            "paged_kv_indices": torch.empty(
                max_tokens * topk, device=device, dtype=torch.int32
            ),
            "q_rope": torch.empty(
                max_tokens,
                num_heads,
                self.qk_nope_head_dim + self.qk_rope_head_dim,
                device=device,
                dtype=query_dtype,
            ),
            "k_pe_rope_2d": torch.empty(
                max_tokens, self.qk_rope_head_dim, device=device, dtype=query_dtype
            ),
            "k_pe_rope_3d": torch.empty(
                max_tokens, 1, self.qk_rope_head_dim, device=device, dtype=query_dtype
            ),
            "k_pe_rope_heads": torch.empty(
                max_tokens,
                num_heads,
                self.qk_rope_head_dim,
                device=device,
                dtype=query_dtype,
            ),
            "q_latent_nope_t": torch.empty(
                num_heads,
                max_tokens,
                self.kv_lora_rank,
                device=device,
                dtype=query_dtype,
            ),
            "q_latent": torch.empty(
                max_tokens, num_heads, latent_dim, device=device, dtype=query_dtype
            ),
            "q_for_kernel": torch.empty(
                max_tokens,
                padded_num_heads,
                latent_dim,
                device=device,
                dtype=query_dtype,
            ),
            "q_for_kernel_fp8": torch.empty(
                max_tokens,
                padded_num_heads,
                latent_dim,
                device=device,
                dtype=dtypes.fp8,
            ),
            "latent_output": torch.empty(
                max_tokens,
                padded_num_heads,
                self.kv_lora_rank,
                device=device,
                dtype=query_dtype,
            ),
            "final_output_t": torch.empty(
                num_heads, max_tokens, self.v_head_dim, device=device, dtype=query_dtype
            ),
            "work_meta_data": torch.empty(
                work_meta_data_size, dtype=work_meta_data_type, device=device
            ),
            "work_indptr": torch.empty(
                work_indptr_size, dtype=work_indptr_type, device=device
            ),
            "work_info_set": torch.empty(
                work_info_set_size, dtype=work_info_set_type, device=device
            ),
            "reduce_indptr": torch.empty(
                reduce_indptr_size, dtype=reduce_indptr_type, device=device
            ),
            "reduce_final_map": torch.empty(
                reduce_final_map_size, dtype=reduce_final_map_type, device=device
            ),
            "reduce_partial_map": torch.empty(
                reduce_partial_map_size, dtype=reduce_partial_map_type, device=device
            ),
        }
        self._cg_sparse_bufs["paged_kv_indptr"].zero_()
        self._cache_write_scale[device] = torch.tensor(
            1.0, dtype=torch.float32, device=device
        )
        self._cg_workspace_signature = (
            max_tokens,
            padded_num_heads,
            topk,
            query_dtype,
            device,
        )

    def _build_atom_sparse_metadata(
        self,
        *,
        q_latent: torch.Tensor,
        kv_cache_base: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: Any,
        block_size: int,
    ) -> _AtomSparseMetadata:
        try:
            from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

            triton_convert_req_index_to_global_index = (
                _resolve_plugin_sparse_index_converter()
            )
        except Exception as exc:
            raise _SparseUnavailable(
                f"ATOM sparse MLA metadata helpers unavailable: {exc}"
            ) from exc

        plugin_metadata = getattr(attn_metadata, "plugin_metadata", None)
        if plugin_metadata is None:
            raise _SparseUnavailable("GLM5 RTP sparse MLA requires plugin metadata.")

        num_tokens = int(q_latent.shape[0])
        num_heads = int(q_latent.shape[1])
        topk = int(topk_indices.shape[1])
        device = q_latent.device
        in_capture = torch.cuda.is_current_stream_capturing()
        cg_bufs = getattr(plugin_metadata, "cg_bufs", None)
        sparse_bufs = self._cg_sparse_bufs

        query_start_loc = getattr(plugin_metadata, "query_start_loc", None)
        if query_start_loc is None:
            query_start_loc = getattr(plugin_metadata, "rtp_cu_seqlens_q", None)
        if (
            not isinstance(query_start_loc, torch.Tensor)
            or int(query_start_loc.numel()) < 2
        ):
            raise _SparseUnavailable("GLM5 RTP sparse MLA requires query_start_loc.")
        if in_capture:
            if query_start_loc.device != device or query_start_loc.dtype != torch.int32:
                raise _SparseUnavailable(
                    "GLM5 RTP sparse MLA capture requires int32 query_start_loc on device."
                )
        else:
            query_start_loc = query_start_loc.to(
                device=device, dtype=torch.int32
            ).contiguous()

        seq_lens = getattr(plugin_metadata, "seq_lens", None)
        if seq_lens is None:
            seq_lens = getattr(attn_metadata, "context_lens", None)
        if not isinstance(seq_lens, torch.Tensor) or int(seq_lens.numel()) + 1 != int(
            query_start_loc.numel()
        ):
            raise _SparseUnavailable(
                "GLM5 RTP sparse MLA requires seq_lens per request."
            )
        if in_capture:
            if seq_lens.device != device or seq_lens.dtype != torch.int32:
                raise _SparseUnavailable(
                    "GLM5 RTP sparse MLA capture requires int32 seq_lens on device."
                )
        else:
            seq_lens = seq_lens.to(device=device, dtype=torch.int32).contiguous()

        if in_capture:
            if not isinstance(cg_bufs, dict) or sparse_bufs is None:
                raise _SparseUnavailable(
                    "GLM5 RTP sparse MLA capture requires prewarmed buffers."
                )
            req_id = cg_bufs.get("seq_id_i32", None)
            if not isinstance(req_id, torch.Tensor):
                raise _SparseUnavailable(
                    "GLM5 RTP sparse MLA capture requires prewarmed seq_id_i32."
                )
            req_id = req_id[:num_tokens]
            block_table = getattr(plugin_metadata, "block_table", None)
            if not isinstance(block_table, torch.Tensor):
                raise _SparseUnavailable(
                    "GLM5 RTP sparse MLA capture requires block_table."
                )
            if block_table.device != device or block_table.dtype != torch.int32:
                raise _SparseUnavailable(
                    "GLM5 RTP sparse MLA capture requires int32 block_table on device."
                )
            topk_indices_i32 = topk_indices
            if (
                topk_indices_i32.device != device
                or topk_indices_i32.dtype != torch.int32
            ):
                raise _SparseUnavailable(
                    "GLM5 RTP sparse MLA capture requires int32 topk_indices on device."
                )
            if not topk_indices_i32.is_contiguous():
                raise _SparseUnavailable(
                    "GLM5 RTP sparse MLA capture requires contiguous topk_indices."
                )
            self._validate_capture_sparse_buffer_capacity(
                sparse_bufs=sparse_bufs,
                num_tokens=num_tokens,
                topk=topk,
            )
            sparse_seqlen = sparse_bufs["sparse_seqlen"][:num_tokens]
            torch.clamp(seq_lens[:num_tokens], min=0, max=topk, out=sparse_seqlen)
            max_query_len_for_sparse = 1
        else:
            req_id = self._build_req_id_per_token(attn_metadata, num_tokens, device).to(
                dtype=torch.int32
            )
            block_table = self._block_table(attn_metadata, device).to(dtype=torch.int32)
            topk_indices_i32 = topk_indices.to(
                device=device, dtype=torch.int32
            ).contiguous()
            # Keep prefill aligned with ATOM sparse metadata contract: token-ragged
            # representation always uses max_q_len=1.
            max_query_len_for_sparse = 1
            # Derive sparse lengths directly from indexer output validity. This is
            # robust for chunked prefill where seq_lens may be chunk-local.
            sparse_seqlen = torch.sum(topk_indices_i32 >= 0, dim=1, dtype=torch.int32)

        if in_capture:
            qo_indptr = sparse_bufs["qo_indptr"][: num_tokens + 1]
            paged_kv_indptr = sparse_bufs["paged_kv_indptr"][: num_tokens + 1]
            paged_kv_indptr[0].zero_()
            paged_kv_last_page_len = sparse_bufs["paged_kv_last_page_len"][:num_tokens]
            paged_kv_indices = sparse_bufs["paged_kv_indices"][: num_tokens * topk]
        else:
            eager_sig = (
                int(num_tokens),
                int(topk),
                str(device),
            )
            cached_eager = getattr(plugin_metadata, "_rtp_sparse_eager_workspace", None)
            if (
                isinstance(cached_eager, dict)
                and cached_eager.get("signature") == eager_sig
            ):
                qo_indptr = cached_eager["qo_indptr"]
                paged_kv_indptr = cached_eager["paged_kv_indptr"]
                paged_kv_last_page_len = cached_eager["paged_kv_last_page_len"]
                paged_kv_indices = cached_eager["paged_kv_indices"]
            else:
                qo_indptr = torch.empty(
                    num_tokens + 1, device=device, dtype=torch.int32
                )
                paged_kv_indptr = torch.empty(
                    num_tokens + 1, device=device, dtype=torch.int32
                )
                paged_kv_last_page_len = torch.empty(
                    num_tokens, device=device, dtype=torch.int32
                )
                paged_kv_indices = torch.empty(
                    num_tokens * topk, device=device, dtype=torch.int32
                )
                try:
                    plugin_metadata._rtp_sparse_eager_workspace = {
                        "signature": eager_sig,
                        "qo_indptr": qo_indptr,
                        "paged_kv_indptr": paged_kv_indptr,
                        "paged_kv_last_page_len": paged_kv_last_page_len,
                        "paged_kv_indices": paged_kv_indices,
                    }
                except Exception:
                    pass
            qo_indptr.copy_(
                torch.arange(num_tokens + 1, device=device, dtype=torch.int32)
            )
            paged_kv_indptr.zero_()
            paged_kv_last_page_len.fill_(1)
        torch.cumsum(sparse_seqlen, dim=0, out=paged_kv_indptr[1:])

        if not in_capture and int(block_size) <= 0:
            raise _SparseUnavailable(
                f"GLM5 RTP sparse MLA requires positive block_size, got {block_size}."
            )

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
            padded_num_heads = (
                (padded_num_heads + num_heads - 1) // num_heads
            ) * num_heads
        head_repeat_factor = padded_num_heads // num_heads
        q_dtype = self._aiter_dtype_for_tensor(q_latent)
        kv_dtype = self._aiter_dtype_for_tensor(kv_cache_base)
        reuse_eager_metadata = False
        if in_capture:
            work_meta_data = sparse_bufs["work_meta_data"]
            work_indptr = sparse_bufs["work_indptr"]
            work_info_set = sparse_bufs["work_info_set"]
            reduce_indptr = sparse_bufs["reduce_indptr"]
            reduce_final_map = sparse_bufs["reduce_final_map"]
            reduce_partial_map = sparse_bufs["reduce_partial_map"]
        else:
            eager_meta_sig = (
                int(num_tokens),
                int(topk),
                int(padded_num_heads),
                str(q_dtype),
                str(kv_dtype),
                str(device),
            )
            cached_eager_meta = getattr(
                plugin_metadata, "_rtp_sparse_eager_meta_workspace", None
            )
            if (
                isinstance(cached_eager_meta, dict)
                and cached_eager_meta.get("signature") == eager_meta_sig
            ):
                work_meta_data = cached_eager_meta["work_meta_data"]
                work_indptr = cached_eager_meta["work_indptr"]
                work_info_set = cached_eager_meta["work_info_set"]
                reduce_indptr = cached_eager_meta["reduce_indptr"]
                reduce_final_map = cached_eager_meta["reduce_final_map"]
                reduce_partial_map = cached_eager_meta["reduce_partial_map"]
                reuse_eager_metadata = bool(
                    cached_eager_meta.get("metadata_ready", False)
                )
            else:
                metadata_budget_tokens = self._metadata_token_budget(
                    num_tokens=num_tokens, topk=topk
                )
                (
                    (work_meta_data_size, work_meta_data_type),
                    (work_indptr_size, work_indptr_type),
                    (work_info_set_size, work_info_set_type),
                    (reduce_indptr_size, reduce_indptr_type),
                    (reduce_final_map_size, reduce_final_map_type),
                    (reduce_partial_map_size, reduce_partial_map_type),
                ) = get_mla_metadata_info_v1(
                    metadata_budget_tokens,
                    1,
                    padded_num_heads,
                    q_dtype,
                    kv_dtype,
                    is_sparse=True,
                    fast_mode=True,
                )
                work_meta_data = torch.empty(
                    work_meta_data_size, dtype=work_meta_data_type, device=device
                )
                work_indptr = torch.empty(
                    work_indptr_size, dtype=work_indptr_type, device=device
                )
                work_info_set = torch.empty(
                    work_info_set_size, dtype=work_info_set_type, device=device
                )
                reduce_indptr = torch.empty(
                    reduce_indptr_size, dtype=reduce_indptr_type, device=device
                )
                reduce_final_map = torch.empty(
                    reduce_final_map_size, dtype=reduce_final_map_type, device=device
                )
                reduce_partial_map = torch.empty(
                    reduce_partial_map_size,
                    dtype=reduce_partial_map_type,
                    device=device,
                )
                try:
                    plugin_metadata._rtp_sparse_eager_meta_workspace = {
                        "signature": eager_meta_sig,
                        "work_meta_data": work_meta_data,
                        "work_indptr": work_indptr,
                        "work_info_set": work_info_set,
                        "reduce_indptr": reduce_indptr,
                        "reduce_final_map": reduce_final_map,
                        "reduce_partial_map": reduce_partial_map,
                        "metadata_ready": False,
                    }
                except Exception:
                    pass
        capture_meta_sig = (
            int(num_tokens),
            int(topk),
            int(padded_num_heads),
            str(q_dtype),
            str(kv_dtype),
            str(device),
        )
        reuse_capture_metadata = False
        if in_capture:
            cached_capture_meta = getattr(
                plugin_metadata, "_rtp_sparse_capture_meta_workspace", None
            )
            if (
                isinstance(cached_capture_meta, dict)
                and cached_capture_meta.get("signature") == capture_meta_sig
            ):
                work_meta_data = cached_capture_meta["work_meta_data"]
                work_indptr = cached_capture_meta["work_indptr"]
                work_info_set = cached_capture_meta["work_info_set"]
                reduce_indptr = cached_capture_meta["reduce_indptr"]
                reduce_final_map = cached_capture_meta["reduce_final_map"]
                reduce_partial_map = cached_capture_meta["reduce_partial_map"]
                reuse_capture_metadata = True
        kv_token_slots = self._kv_token_slot_capacity(kv_cache_base)
        page_size = 1
        max_page_slots = int(kv_token_slots)

        if in_capture and int(paged_kv_indices.numel()) > 0:
            # Capture path cannot run host-synced range checks; clamp indices into
            # the current kv slot range to avoid kernel-side OOB accesses.
            paged_kv_indices.clamp_(min=0, max=max(int(max_page_slots) - 1, 0))

        if not in_capture and self._enable_sparse_validate:
            self._validate_sparse_index_contract(
                paged_kv_indptr=paged_kv_indptr,
                paged_kv_indices=paged_kv_indices,
                num_tokens=num_tokens,
                page_size=page_size,
                max_slots=max_page_slots,
            )

        if not reuse_capture_metadata and not reuse_eager_metadata:
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
                page_size=page_size,
                kv_granularity=16,
                max_seqlen_qo=max_query_len_for_sparse,
                uni_seqlen_qo=max_query_len_for_sparse,
                fast_mode=True,
                dtype_q=q_dtype,
                dtype_kv=kv_dtype,
            )
            if not in_capture:
                cached_eager_meta = getattr(
                    plugin_metadata, "_rtp_sparse_eager_meta_workspace", None
                )
                if isinstance(cached_eager_meta, dict):
                    cached_eager_meta["metadata_ready"] = True
            if in_capture:
                plugin_metadata._rtp_sparse_capture_meta_workspace = {
                    "signature": capture_meta_sig,
                    "work_meta_data": work_meta_data,
                    "work_indptr": work_indptr,
                    "work_info_set": work_info_set,
                    "reduce_indptr": reduce_indptr,
                    "reduce_final_map": reduce_final_map,
                    "reduce_partial_map": reduce_partial_map,
                }
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
            page_size=page_size,
        )

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
            raise _SparseUnavailable(
                f"aiter.mla_decode_fwd unavailable: {exc}"
            ) from exc

        num_tokens, num_heads, latent_dim = q_latent.shape
        sparse_meta = self._build_atom_sparse_metadata(
            q_latent=q_latent,
            kv_cache_base=kv_cache_base,
            topk_indices=topk_indices,
            attn_metadata=attn_metadata,
            block_size=block_size,
        )
        in_capture = torch.cuda.is_current_stream_capturing()
        page_size = 1
        if sparse_meta.head_repeat_factor > 1:
            if in_capture and self._cg_sparse_bufs is not None:
                q_for_kernel = self._cg_sparse_bufs["q_for_kernel"][
                    :num_tokens, : sparse_meta.padded_num_heads, :
                ]
                # Capture path: use one broadcasted copy to fill repeated heads,
                # avoiding per-repeat slice copies in the decode hot path.
                q_for_kernel.view(
                    num_tokens,
                    num_heads,
                    sparse_meta.head_repeat_factor,
                    latent_dim,
                ).copy_(q_latent.unsqueeze(2))
            else:
                q_for_kernel = (
                    q_latent.unsqueeze(2)
                    .expand(-1, -1, sparse_meta.head_repeat_factor, -1)
                    .reshape(num_tokens, sparse_meta.padded_num_heads, latent_dim)
                )
        else:
            q_for_kernel = q_latent
        output_dtype = q_for_kernel.dtype
        if in_capture and self._cg_sparse_bufs is not None:
            output = self._cg_sparse_bufs["latent_output"][
                :num_tokens, : sparse_meta.padded_num_heads, :
            ]
        else:
            output = torch.empty(
                (num_tokens, sparse_meta.padded_num_heads, self.kv_lora_rank),
                dtype=output_dtype,
                device=q_latent.device,
            )
        fp8_scale_kwargs = {}
        if self._cache_dtype_name(kv_cache_base) == "fp8":
            kv_scale = self._cache_write_scale.get(kv_cache_base.device)
            if kv_scale is None:
                kv_scale = torch.tensor(
                    1.0, dtype=torch.float32, device=kv_cache_base.device
                )
                self._cache_write_scale[kv_cache_base.device] = kv_scale
            fp8_scale_kwargs = {"q_scale": kv_scale, "kv_scale": kv_scale}
            try:
                from aiter import dtypes
            except Exception as exc:
                raise _SparseUnavailable(f"aiter dtypes unavailable: {exc}") from exc
            if in_capture and self._cg_sparse_bufs is not None:
                q_for_kernel_fp8 = self._cg_sparse_bufs["q_for_kernel_fp8"][
                    :num_tokens, : sparse_meta.padded_num_heads, :
                ]
                q_for_kernel_fp8.copy_(q_for_kernel)
                q_for_kernel = q_for_kernel_fp8
            else:
                q_for_kernel = q_for_kernel.to(dtype=dtypes.fp8)
        try:
            kv_buffer = kv_cache_base.reshape(-1, 1, 1, latent_dim)
            if (
                not in_capture
                and self._enable_sparse_validate
                and int(sparse_meta.paged_kv_indices.numel()) > 0
            ):
                self._validate_sparse_index_contract(
                    paged_kv_indptr=sparse_meta.paged_kv_indptr,
                    paged_kv_indices=sparse_meta.paged_kv_indices,
                    num_tokens=num_tokens,
                    page_size=page_size,
                    max_slots=int(kv_buffer.shape[0]),
                )
                self._validate_sparse_last_page_contract(
                    paged_kv_indptr=sparse_meta.paged_kv_indptr,
                    paged_kv_last_page_len=sparse_meta.paged_kv_last_page_len,
                    num_tokens=num_tokens,
                    page_size=page_size,
                )
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
                page_size=page_size,
                work_meta_data=sparse_meta.work_meta_data,
                work_indptr=sparse_meta.work_indptr,
                work_info_set=sparse_meta.work_info_set,
                reduce_indptr=sparse_meta.reduce_indptr,
                reduce_final_map=sparse_meta.reduce_final_map,
                reduce_partial_map=sparse_meta.reduce_partial_map,
                **fp8_scale_kwargs,
            )
        except Exception as exc:
            raise _SparseUnavailable(f"mla_decode_fwd failed: {exc}") from exc
        if sparse_meta.head_repeat_factor > 1:
            output = output[:, :: sparse_meta.head_repeat_factor, :]
            if not in_capture:
                output = output.contiguous()
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
        if getattr(
            getattr(attn_metadata, "plugin_metadata", None), "is_dummy_warmup", False
        ):
            return q.new_zeros((q.shape[0], q.shape[1], self.v_head_dim))
        q_rope, k_pe_rope = self._apply_rope(q, k_pe, positions)
        kv_cache_base = self._write_current_to_cache(
            compressed_kv=compressed_kv,
            k_pe=k_pe_rope,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        absorbed = self._get_absorbed_weights(q_rope)
        q_nope = q_rope[..., : self.qk_nope_head_dim]
        in_capture = torch.cuda.is_current_stream_capturing()
        if in_capture:
            if self._cg_sparse_bufs is None:
                raise _SparseUnavailable(
                    "GLM5 RTP sparse MLA capture requires q buffers."
                )
            if q_nope.dtype != absorbed.w_kc.dtype:
                raise _SparseUnavailable(
                    "GLM5 RTP sparse MLA capture requires q_nope dtype to match absorbed weights."
                )
            q_latent_nope_t = self._cg_sparse_bufs["q_latent_nope_t"][
                : q.shape[1], : q.shape[0], :
            ]
            torch.bmm(q_nope.transpose(0, 1), absorbed.w_kc, out=q_latent_nope_t)
            q_latent_nope = q_latent_nope_t.transpose(0, 1)
            q_latent = self._cg_sparse_bufs["q_latent"][
                : q.shape[0],
                : q.shape[1],
                : self.kv_lora_rank + self.qk_rope_head_dim,
            ]
        else:
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
            raise _SparseUnavailable(
                "GLM5 RTP sparse MLA requires physical block size."
            )
        latent_output = self._run_aiter_sparse_decode(
            q_latent=q_latent,
            kv_cache_base=kv_cache_base,
            topk_indices=topk_indices,
            attn_metadata=attn_metadata,
            block_size=block_size,
        )
        if in_capture:
            if latent_output.dtype != absorbed.w_vc.dtype:
                raise _SparseUnavailable(
                    "GLM5 RTP sparse MLA capture requires latent output dtype to match absorbed weights."
                )
            output_t = self._cg_sparse_bufs["final_output_t"][
                : q.shape[1], : q.shape[0], :
            ]
            torch.bmm(latent_output.transpose(0, 1), absorbed.w_vc, out=output_t)
            output = output_t.transpose(0, 1)
            if output.dtype != q.dtype:
                raise _SparseUnavailable(
                    "GLM5 RTP sparse MLA capture requires final output dtype to match q."
                )
            return output
        output = torch.bmm(
            latent_output.transpose(0, 1).to(dtype=absorbed.w_vc.dtype),
            absorbed.w_vc,
        ).transpose(0, 1)
        return output.to(dtype=q.dtype)


class RTPSparseMlaBackend:
    """Sparse MLA backend used by GLM5 RTP plugin mode.

    Real GLM5 layers use ATOM-owned MLA modules and the AITER sparse decode
    kernel. The lightweight implementation is kept for unit tests and explicit
    injection only; production paths refuse dense fallback when sparse execution
    is unavailable.
    """

    def __init__(
        self,
        *,
        sparse_impl: Optional[object] = None,
        v_head_dim: Optional[int] = None,
        mla_modules: Optional[object] = None,
        scale: Optional[float] = None,
    ) -> None:
        if v_head_dim is None:
            if mla_modules is None or not hasattr(mla_modules, "v_head_dim"):
                raise ValueError(
                    "RTPSparseMlaBackend requires v_head_dim or mla_modules.v_head_dim."
                )
            v_head_dim = getattr(mla_modules, "v_head_dim")
        self.v_head_dim = int(v_head_dim)
        if sparse_impl is not None:
            self.sparse_impl = sparse_impl
            self._uses_lightweight_impl = False
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
            self._uses_lightweight_impl = False
        else:
            self.sparse_impl = _LightweightSparseMlaImpl(self.v_head_dim)
            self._uses_lightweight_impl = True
        self._sparse_impl_accepts_positions = self._impl_accepts_positions(
            self.sparse_impl
        )

    def prepare_cuda_graph(self, attn_inputs) -> None:  # noqa: ANN001
        del attn_inputs

    def prewarm_for_cuda_graph(
        self,
        *,
        max_num_tokens: int,
        max_seq_len: int,
        query_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        sparse_prewarm = getattr(self.sparse_impl, "prewarm_for_cuda_graph", None)
        if callable(sparse_prewarm):
            sparse_prewarm(
                max_num_tokens=max_num_tokens,
                max_seq_len=max_seq_len,
                query_dtype=query_dtype,
                device=device,
            )

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
    def _impl_accepts_positions(impl: object) -> bool:
        try:
            signature = inspect.signature(impl.forward)
        except (AttributeError, TypeError, ValueError):
            return False
        return "positions" in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
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
        attn_metadata = self._get_attn_metadata()
        if getattr(
            getattr(attn_metadata, "plugin_metadata", None), "is_dummy_warmup", False
        ):
            return q.new_zeros((q.shape[0], q.shape[1], self.v_head_dim))

        if topk_indices is None:
            if self._uses_lightweight_impl:
                return q.new_zeros((q.shape[0], q.shape[1], self.v_head_dim))
            raise _SparseUnavailable(
                "GLM5 RTP sparse MLA requires topk_indices; refusing dense fallback."
            )
        self._validate_topk_indices(q, topk_indices)
        if self._uses_lightweight_impl or not callable(
            getattr(self.sparse_impl, "forward", None)
        ):
            raise _SparseUnavailable(
                "GLM5 RTP sparse MLA is unavailable; refusing dense fallback."
            )

        kwargs = {
            "topk_indices": topk_indices,
            "attn_metadata": attn_metadata,
        }
        if self._sparse_impl_accepts_positions:
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
        except _SparseUnavailable as exc:
            raise _SparseUnavailable(
                "GLM5 RTP sparse MLA unavailable; dense fallback is disabled. "
                f"root_cause={exc}"
            ) from exc


def _run_rtp_sparse_attn_indexer_topk_only(
    hidden_states: torch.Tensor,
    kv_cache: torch.Tensor,
    q_input: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: Optional[str],
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    k_norm_eps: float,
    positions: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    weights_scale: float,
    is_neox_style: bool,
    use_qk_rope_cache_fusion: bool,
    context: Any,
    attn_metadata: Any,
) -> torch.Tensor:
    from aiter import (
        cp_gather_indexer_k_quant_cache,
        dtypes,
        indexer_k_quant_and_cache,
        indexer_qk_rope_quant_and_cache,
        top_k_per_row_decode,
        top_k_per_row_prefill,
    )
    from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits
    from aiter.ops.triton.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits
    from atom.config import get_current_atom_config

    slot_mapping = getattr(attn_metadata, "slot_mapping", None)
    if slot_mapping is None:
        raise _SparseUnavailable("RTP sparse indexer requires slot_mapping metadata.")
    if topk_indices_buffer is None:
        raise _SparseUnavailable("RTP sparse indexer requires topk_indices_buffer.")
    if topk_indices_buffer.dim() != 2:
        raise _SparseUnavailable(
            "RTP sparse indexer requires a 2D topk_indices_buffer; "
            f"got shape={tuple(topk_indices_buffer.shape)}."
        )

    if bool(getattr(context, "is_dummy_run", False)):
        return torch.zeros_like(weights, dtype=torch.float32)

    num_tokens = int(hidden_states.shape[0])
    if num_tokens <= 0:
        return weights
    topk_indices = topk_indices_buffer[:num_tokens, :topk_tokens]
    if topk_indices.dtype != torch.int32:
        raise _SparseUnavailable(
            f"RTP sparse indexer topk buffer must be int32, got {topk_indices.dtype}."
        )

    runner_block_size = int(get_current_atom_config().kv_cache_block_size)
    kv_cache = kv_cache.view(-1, runner_block_size, kv_cache.shape[-1])

    if use_qk_rope_cache_fusion:
        q_bf16 = q_input
        q_fp8 = torch.empty_like(q_bf16, dtype=dtypes.fp8)
        weights_out = torch.empty(
            weights.shape, device=weights.device, dtype=torch.float32
        )
        indexer_qk_rope_quant_and_cache(
            q_bf16,
            q_fp8,
            weights,
            weights_out,
            k,
            kv_cache,
            slot_mapping,
            k_norm_weight,
            k_norm_bias,
            positions,
            cos_cache,
            sin_cache,
            k_norm_eps,
            quant_block_size,
            scale_fmt,
            weights_scale,
            preshuffle=True,
            is_neox=is_neox_style,
        )
        weights = weights_out
    else:
        q_fp8 = q_input
        indexer_k_quant_and_cache(
            k,
            kv_cache,
            slot_mapping,
            quant_block_size,
            scale_fmt,
            preshuffle=True,
        )

    is_prefill = bool(getattr(context, "is_prefill", False))
    max_seqlen_k = int(getattr(attn_metadata, "max_seqlen_k", 0) or 0)
    if is_prefill and max_seqlen_k <= int(topk_tokens):
        return weights

    if is_prefill:
        total_seq_lens = int(hidden_states.shape[0])
        has_cached = bool(getattr(attn_metadata, "has_cached", False))
        total_kv = (
            int(getattr(attn_metadata, "total_kv", total_seq_lens))
            if has_cached
            else total_seq_lens
        )
        k_fp8 = torch.empty([total_kv, head_dim], device=k.device, dtype=dtypes.fp8)
        k_scale = torch.empty([total_kv, 1], device=k.device, dtype=torch.float32)
        block_tables = getattr(attn_metadata, "block_tables", None)
        cu_seqlens_q = getattr(attn_metadata, "cu_seqlens_q", None)
        if block_tables is None or cu_seqlens_q is None:
            raise _SparseUnavailable(
                "RTP sparse prefill indexer requires block_tables and cu_seqlens_q."
            )
        cu_seqlens_k = (
            getattr(attn_metadata, "cu_seqlens_k", None) if has_cached else cu_seqlens_q
        )
        if cu_seqlens_k is None:
            raise _SparseUnavailable(
                "RTP sparse prefill indexer requires cu_seqlens_k."
            )
        cp_gather_indexer_k_quant_cache(
            kv_cache,
            k_fp8,
            k_scale.view(dtypes.fp8),
            block_tables,
            cu_seqlens_k,
            preshuffle=True,
        )
        cu_seqlen_ks = getattr(attn_metadata, "cu_seqlen_ks", None)
        cu_seqlen_ke = getattr(attn_metadata, "cu_seqlen_ke", None)
        if cu_seqlen_ks is None or cu_seqlen_ke is None:
            raise _SparseUnavailable(
                "RTP sparse prefill indexer requires cu_seqlen_ks/cu_seqlen_ke."
            )
        num_decode_tokens = 0
        logits = fp8_mqa_logits(
            Q=q_fp8[num_decode_tokens:num_tokens],
            KV=k_fp8,
            kv_scales=k_scale,
            weights=weights[num_decode_tokens:num_tokens],
            cu_starts=cu_seqlen_ks,
            cu_ends=cu_seqlen_ke,
        )
        top_k_per_row_prefill(
            logits=logits,
            rowStarts=cu_seqlen_ks,
            rowEnds=cu_seqlen_ke,
            indices=topk_indices[num_decode_tokens:num_tokens, :topk_tokens],
            values=None,
            numRows=logits.shape[0],
            stride0=logits.stride(0),
            stride1=logits.stride(1),
        )
        return weights

    max_seqlen_q = int(getattr(attn_metadata, "max_seqlen_q", 1) or 1)
    num_decode_tokens = int(context.batch_size) * max_seqlen_q
    kv_cache_for_logits = kv_cache.unsqueeze(-2)
    padded_q_fp8_decode_tokens = q_fp8[:num_decode_tokens].reshape(
        int(context.batch_size), -1, *q_fp8.shape[1:]
    )
    batch_size, next_n, _heads, _dim = padded_q_fp8_decode_tokens.shape
    logits = torch.empty(
        [batch_size * next_n, int(max_model_len)],
        dtype=torch.float32,
        device=hidden_states.device,
    )
    context_lens = getattr(attn_metadata, "context_lens", None)
    block_tables = getattr(attn_metadata, "block_tables", None)
    if context_lens is None or block_tables is None:
        raise _SparseUnavailable(
            "RTP sparse decode indexer requires context_lens and block_tables."
        )
    deepgemm_fp8_paged_mqa_logits(
        padded_q_fp8_decode_tokens,
        kv_cache_for_logits,
        weights[:num_decode_tokens],
        logits,
        context_lens,
        block_tables,
        int(max_model_len),
        KVBlockSize=runner_block_size,
        Preshuffle=True,
    )
    top_k_per_row_decode(
        logits,
        next_n,
        context_lens,
        topk_indices[:num_decode_tokens, :topk_tokens],
        logits.shape[0],
        logits.stride(0),
        logits.stride(1),
    )
    return weights


def rtp_sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_input: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: Optional[str],
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    k_norm_eps: float,
    positions: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    weights_scale: float,
    is_neox_style: bool,
    use_qk_rope_cache_fusion: bool,
) -> torch.Tensor:
    try:
        from atom.utils.forward_context import get_forward_context

        forward_context = get_forward_context()
    except Exception:
        forward_context = None
    context = getattr(forward_context, "context", None)
    attn_metadata = getattr(forward_context, "attn_metadata", None)
    # For short prefill (ctx <= topk buffer width), DeepSeek indexer returns early and
    # doesn't write topk buffer. Emit causal full-history indices to keep sparse path valid.
    if (
        context is not None
        and bool(getattr(context, "is_prefill", False))
        and attn_metadata is not None
        and topk_indices_buffer is not None
        and topk_indices_buffer.dim() == 2
        and positions is not None
    ):
        max_seqlen_k = int(getattr(attn_metadata, "max_seqlen_k", 0) or 0)
        topk_capacity = int(topk_indices_buffer.shape[1])
        if max_seqlen_k > 0 and max_seqlen_k <= topk_capacity:
            num_tokens = int(hidden_states.shape[0])
            if num_tokens > 0:
                positions_i32 = positions.to(
                    device=topk_indices_buffer.device, dtype=torch.int32
                ).view(-1)
                row_limits = (
                    (positions_i32 + 1).clamp(min=0, max=topk_tokens).view(-1, 1)
                )
                col_ids = torch.arange(
                    topk_tokens,
                    device=topk_indices_buffer.device,
                    dtype=torch.int32,
                ).view(1, -1)
                causal_topk = torch.where(
                    col_ids < row_limits,
                    col_ids.expand(num_tokens, topk_tokens),
                    torch.full(
                        (num_tokens, topk_tokens),
                        -1,
                        device=topk_indices_buffer.device,
                        dtype=torch.int32,
                    ),
                )
                topk_indices_buffer[:num_tokens, :topk_tokens].copy_(causal_topk)
            return weights

    if context is not None and attn_metadata is not None:
        return _run_rtp_sparse_attn_indexer_topk_only(
            hidden_states,
            kv_cache,
            q_input,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            k_norm_weight,
            k_norm_bias,
            k_norm_eps,
            positions,
            cos_cache,
            sin_cache,
            weights_scale,
            is_neox_style,
            use_qk_rope_cache_fusion,
            context,
            attn_metadata,
        )

    from atom.models.deepseek_v2 import sparse_attn_indexer

    return sparse_attn_indexer(
        hidden_states,
        k_cache_prefix,
        kv_cache,
        q_input,
        k,
        weights,
        quant_block_size,
        scale_fmt,
        topk_tokens,
        head_dim,
        max_model_len,
        total_seq_lens,
        topk_indices_buffer,
        k_norm_weight,
        k_norm_bias,
        k_norm_eps,
        positions,
        cos_cache,
        sin_cache,
        weights_scale,
        is_neox_style,
        use_qk_rope_cache_fusion,
    )


def rtp_sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_input: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: Optional[str],
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    k_norm_eps: float,
    positions: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    weights_scale: float,
    is_neox_style: bool,
    use_qk_rope_cache_fusion: bool,
) -> torch.Tensor:
    from atom.models.deepseek_v2 import sparse_attn_indexer_fake

    return sparse_attn_indexer_fake(
        hidden_states,
        k_cache_prefix,
        kv_cache,
        q_input,
        k,
        weights,
        quant_block_size,
        scale_fmt,
        topk_tokens,
        head_dim,
        max_model_len,
        total_seq_lens,
        topk_indices_buffer,
        k_norm_weight,
        k_norm_bias,
        k_norm_eps,
        positions,
        cos_cache,
        sin_cache,
        weights_scale,
        is_neox_style,
        use_qk_rope_cache_fusion,
    )


direct_register_custom_op(
    op_name="rtp_sparse_attn_indexer",
    op_func=rtp_sparse_attn_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=rtp_sparse_attn_indexer_fake,
)

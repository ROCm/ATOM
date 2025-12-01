import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from aiter import (
    concat_and_cache_mla,
    dtypes,
    flash_attn_varlen_func,
)
from aiter.mla import mla_decode_fwd
from aiter.rotary_embedding import RotaryEmbedding
from torch import nn
from tqdm import tqdm

from atom.model_ops.utils import get_and_maybe_dequant_weights
from atom.utils.forward_context import (
    AttentionMetaData,
    ForwardContext,
    get_forward_context,
)

from aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (  # noqa: E501 # isort: skip
    batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant as _aiter_triton_fp8_bmm,
)

# from aiter.ops.triton.fused_kv_cache import fused_qk_rope_cat_and_cache_mla
from aiter import fused_qk_rope_concat_and_cache_mla
from aiter.dist.parallel_state import get_dp_group

torch.set_printoptions(threshold=10_000)

logger = logging.getLogger("atom")


# MLA Specific Arguments
@dataclass
class MLAModules:
    """Modules used in MLA."""

    q_lora_rank: Optional[int]
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    qk_head_dim: int
    v_head_dim: int
    rotary_emb: torch.nn.Module
    q_proj: Optional[torch.nn.Module]
    kv_b_proj: torch.nn.Module
    o_proj: torch.nn.Module
    indexer: Optional[torch.nn.Module]


def dynamic_per_batched_tensor_quant(
    x: torch.Tensor, dtype: torch.dtype = torch.float8_e4m3fn
):
    DTYPE_MAX = torch.finfo(dtype).max
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-10)
    scale = DTYPE_MAX / amax
    x_scl_sat = (x * scale).clamp(min=-DTYPE_MAX, max=DTYPE_MAX)
    return x_scl_sat.to(dtype).contiguous(), scale.float().reciprocal()


class MLAAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        kv_cache_dtype: str,
        layer_num: int = 0,
        mla_modules: MLAModules = None,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype if kv_cache_dtype == "fp8" else "auto"
        self.dtype = dtype

        self.q_lora_rank = mla_modules.q_lora_rank
        self.kv_lora_rank = mla_modules.kv_lora_rank
        self.qk_nope_head_dim = mla_modules.qk_nope_head_dim
        self.qk_rope_head_dim = mla_modules.qk_rope_head_dim
        self.qk_head_dim = mla_modules.qk_head_dim
        self.v_head_dim = mla_modules.v_head_dim
        self.rotary_emb = mla_modules.rotary_emb
        self.q_proj = mla_modules.q_proj
        self.o_proj = mla_modules.o_proj
        self.kv_b_proj = mla_modules.kv_b_proj
        self.kv_cache = torch.tensor([])
        self.one_scale = torch.tensor(1.0, dtype=torch.float32)
        self._k_scale = self.one_scale
        self._q_scale = self.one_scale
        self.topk_indices_buffer = (
            mla_modules.indexer.topk_indices_buffer
            if mla_modules.indexer is not None
            else None
        )
        self.layer_num = layer_num

    def process_weights_after_loading(self):
        kv_b_proj_weight = get_and_maybe_dequant_weights(self.kv_b_proj).T
        assert kv_b_proj_weight.shape == (
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
        ), (
            f"{kv_b_proj_weight.shape=}, "
            f"{self.kv_lora_rank=}, "
            f"{self.num_heads=}, "
            f"{self.qk_nope_head_dim=}, "
            f"{self.v_head_dim=}"
        )
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        W_UK, W_UV = kv_b_proj_weight.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        if True:  # is_rocm_aiter_fp8bmm_enabled():
            W_K = W_UK.transpose(0, 1)  # 16 512 128
            W_V = W_UV.permute(1, 2, 0)  # 16 128 512
            self.W_K, self.W_K_scale = dynamic_per_batched_tensor_quant(
                W_K, dtype=dtypes.fp8
            )
            self.W_V, self.W_V_scale = dynamic_per_batched_tensor_quant(
                W_V, dtype=dtypes.fp8
            )

            # The kernel operates on non-padded inputs. Hence, pre-compiling
            # triton kernel to avoid runtime compilation for unseen batch sizes
            # Pre-compile for batch sizes 1 to 1024 to cover most use-cases.
            # On DS-R1, this step adds roughly 50s to the model loading time.
            # max_batch_size = 1024  # [ToDo] Find the optimal upper limit
            # pre_compilation_list = list(range(1, max_batch_size + 1))
            # if get_tp_group().is_first_rank:
            #     pre_compilation_list = tqdm(
            #         pre_compilation_list,
            #         desc="[Aiter Triton] Pre-compiling fp8 BMM kernel",
            #         total=max_batch_size,
            #     )

            # for m in pre_compilation_list:
            #     x = torch.empty(
            #         (self.W_K.shape[0], m, self.W_K.shape[2]),
            #         dtype=torch.bfloat16,
            #         device=self.W_K.device,
            #     )
            #     _aiter_triton_fp8_bmm(
            #         x, self.W_K, self.W_K_scale, group_size=128, transpose_bm=True
            #     )

            #     x = torch.empty(
            #         (self.W_V.shape[0], m, self.W_V.shape[2]),
            #         dtype=torch.bfloat16,
            #         device=self.W_V.device,
            #     )
            #     _aiter_triton_fp8_bmm(
            #         x, self.W_V, self.W_V_scale, group_size=128, transpose_bm=True
            #     )
        else:
            # Convert from (L, N, V) to (N, L, V)
            self.W_UV = W_UV.transpose(0, 1)
            # Convert from (L, N, P) to (N, P, L)
            self.W_UK_T = W_UK.permute(1, 2, 0)

    def _v_up_proj_and_o_proj(self, x):
        # Convert from (B, N, L) to (N, B, L)
        x = x.view(-1, self.num_heads, self.kv_lora_rank).transpose(0, 1)
        # Multiply (N, B, L) x (N, L, V) -> (N, B, V), Convert from (N, B, V) to (B, N, V)
        # x = torch.bmm(x, self.W_UV).transpose(0, 1)
        x = _aiter_triton_fp8_bmm(
            x, self.W_V, self.W_V_scale, group_size=128, transpose_bm=True
        )
        # Convert from (B, N, V) to (B, N * V)
        x = x.reshape(-1, self.num_heads * self.v_head_dim)
        return self.o_proj(x)

    def _q_proj_and_k_up_proj(self, x):
        q_nope, q_pe = (
            self.q_proj(x)
            .view(-1, self.num_heads, self.qk_head_dim)
            .split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        )

        # Convert from (B, N, P) to (N, B, P)
        q_nope = q_nope.transpose(0, 1)
        # Multiply (N, B, P) x (N, P, L) -> (N, B, L), Convert from (N, B, L) to (B, N, L)
        # ql_nope = torch.bmm(q_nope, self.W_UK_T).transpose(0, 1)
        ql_nope = _aiter_triton_fp8_bmm(
            q_nope, self.W_K, self.W_K_scale, group_size=128, transpose_bm=True
        )
        return ql_nope, q_pe

    def _forward_prefill(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_rope: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: AttentionMetaData,
    ) -> torch.Tensor:
        assert attn_metadata is not None

        kv_nope = self.kv_b_proj(kv_c_normed).view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        if k_rope.dim() == 2:
            k_rope = k_rope.unsqueeze(1)
        k = torch.cat((k_nope, k_rope.expand((*k_nope.shape[:-1], -1))), dim=-1)

        output = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=attn_metadata.cu_seqlens_q,
            cu_seqlens_k=attn_metadata.cu_seqlens_k,
            max_seqlen_q=attn_metadata.max_seqlen_q,
            max_seqlen_k=attn_metadata.max_seqlen_k,
            min_seqlen_q=attn_metadata.min_seqlen_q,
            dropout_p=attn_metadata.dropout_p,
            softmax_scale=self.scale,
            causal=True,
        )

        return self.o_proj(output.flatten(start_dim=-2))

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: AttentionMetaData,
    ) -> torch.Tensor:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata is not None
        B = q.shape[0]

        o = torch.empty(
            B, self.num_heads, self.kv_lora_rank, dtype=self.dtype, device=q.device
        )

        kv_buffer = kv_c_and_k_pe_cache.unsqueeze(2)
        paged_kv_indptr = attn_metadata.kv_indptr
        paged_kv_indices = attn_metadata.kv_indices
        if self.topk_indices_buffer is not None:
            paged_kv_indptr = attn_metadata.sparse_kv_indptr
            paged_kv_indices = triton_convert_req_index_to_global_index(
                attn_metadata.cu_seqlens_q,
                attn_metadata.kv_indptr,
                paged_kv_indptr,
                attn_metadata.kv_indices,
                self.topk_indices_buffer[:B],
            )

        # q_scale = kv_scale = None
        # if self.kv_cache_dtype.startswith("fp8"):
        #     q = q.to(dtypes.fp8)
        #     q_scale = kv_scale = self.one_scale

        dp_size = get_dp_group().world_size
        is_fp8 = self.kv_cache_dtype.startswith("fp8")
        use_persistent_mode = not (dp_size == 8 and not is_fp8)

        if not use_persistent_mode:
            # DP + bf16: disable persistent mode to avoid overflow
            work_meta_data = None
            work_indptr = None
            work_info_set = None
            reduce_indptr = None
            reduce_final_map = None
            reduce_partial_map = None
        else:
            work_meta_data = attn_metadata.work_meta_data
            work_indptr = attn_metadata.work_indptr
            work_info_set = attn_metadata.work_info_set
            reduce_indptr = attn_metadata.reduce_indptr
            reduce_final_map = attn_metadata.reduce_final_map
            reduce_partial_map = attn_metadata.reduce_partial_map

        mla_decode_fwd(
            q,
            kv_buffer.view(-1, 1, 1, q.shape[-1]),
            o,
            attn_metadata.cu_seqlens_q,
            paged_kv_indptr,
            paged_kv_indices,
            attn_metadata.kv_last_page_lens,
            attn_metadata.max_q_len,
            self.scale,
            0.0,
            None,
            None,
            work_meta_data,
            work_indptr,
            work_info_set,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            self._q_scale,
            self._k_scale,
        )

        return self._v_up_proj_and_o_proj(o)

    def forward(
        self,
        q: torch.Tensor,  # query in unified attn
        k_nope: torch.Tensor,
        k_rope: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # kv_cache = self.kv_cache
        forward_context: ForwardContext = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        context = forward_context.context
        if attn_metadata.slot_mapping.numel():
            # not dummy run
            kv_cache_data = forward_context.kv_cache_data
            if f"layer_{self.layer_num}" in kv_cache_data:
                kv_cache = kv_cache_data[f"layer_{self.layer_num}"].k_cache
            else:
                kv_cache = torch.tensor([])
        else:
            # dummy run before allocate kv_cache, thus we create manually
            kv_cache = torch.tensor([])

        if context.is_prefill:
            prefill_q = self.q_proj(q).view(-1, self.num_heads, self.qk_head_dim)
            prefill_q_pe = prefill_q[..., self.qk_nope_head_dim :]
            self.rotary_emb(positions, prefill_q_pe, k_rope)

            if kv_cache.numel() > 0:
                concat_and_cache_mla(
                    k_nope,
                    k_rope.squeeze(1),
                    kv_cache,
                    attn_metadata.slot_mapping.flatten(),
                    kv_cache_dtype=self.kv_cache_dtype,
                    scale=self._k_scale,
                )

            output = self._forward_prefill(
                prefill_q, k_nope, k_rope, kv_cache, attn_metadata
            )
        else:
            q_nope, q_rope = self._q_proj_and_k_up_proj(q)

            if kv_cache.numel() > 0:
                # decode_q = torch.empty(
                #     (
                #         q_nope.shape[0],
                #         self.num_heads,
                #         self.kv_lora_rank + self.qk_rope_head_dim,
                #     ),
                #     dtype=kv_cache.dtype,
                #     device=q_nope.device,
                # )
                # fused_qk_rope_concat_and_cache_mla(
                #     q_nope,
                #     q_rope,
                #     k_nope,
                #     k_rope,
                #     kv_cache.view(
                #         kv_cache.shape[0], -1, self.kv_lora_rank + self.qk_rope_head_dim
                #     ),
                #     decode_q,
                #     attn_metadata.slot_mapping,
                #     self._k_scale,
                #     self._q_scale,
                #     positions,
                #     self.rotary_emb.cos_cache,
                #     self.rotary_emb.sin_cache,
                #     is_neox=self.rotary_emb.is_neox_style,
                #     is_nope_first=True,
                # )
                from aiter.ops.triton.fused_kv_cache import fused_qk_rope_cat_and_cache_mla
                decode_q, _, _, _ = fused_qk_rope_cat_and_cache_mla(
                    q_nope,
                    q_rope,
                    k_nope.view(-1, self.num_kv_heads, self.kv_lora_rank),
                    k_rope.view(-1, self.num_kv_heads, self.qk_rope_head_dim),
                    kv_cache,
                    attn_metadata.slot_mapping,
                    positions,
                    self.rotary_emb.cos_cache,
                    self.rotary_emb.sin_cache,
                    k_scale=self._k_scale,
                    is_neox=self.rotary_emb.is_neox_style,
                    q_out_dtype=kv_cache.dtype,
                )

            output = self._forward_decode(decode_q, kv_cache, attn_metadata)

        return output


import triton
import triton.language as tl


@triton.jit
def _convert_req_index_to_global_index_kernel(
    qo_indptr,  # int32 [num_requests]
    kv_indptr,  # int32 [num_requests+1]
    page_kv_indptr,  # int32 [num_requests+1]
    kv_indices,  # int32 [num_requests * max_num_blocks_per_req]
    token_indices_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    out_kv_indices,  # int32
    # shapes (compile-time where possible)
    NUM_TOPK_TOKENS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,  # tile width along columns
    # strides (in elements)
    ti_stride0,
    ti_stride1,
):
    # program_id(0) -> batch_id (row)
    # program_id(1) -> tile index along columns
    batch_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    # Each program covers BLOCK_N consecutive columns
    indice_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load request id for this token (no mask: grid is exact)
    kv_start = tl.load(kv_indptr + batch_id)
    kv_end = tl.load(kv_indptr + batch_id + 1)
    out_kv_start = tl.load(page_kv_indptr + batch_id)
    kv_len = kv_end - kv_start
    qo_start = tl.load(qo_indptr + batch_id)
    qo_end = tl.load(qo_indptr + batch_id + 1)

    for token_id in range(qo_start, qo_end):
        # Load token indices for this tile
        ti_ptr = token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
        tok = tl.load(ti_ptr)  # int32

        # Guard block_table access
        valid_mask = (indice_id < kv_len) & (indice_id < NUM_TOPK_TOKENS)
        out_val = tl.load(
            kv_indices + kv_start + tok,
            mask=valid_mask,
            other=0,
        )

        # Store results
        out_ptr_ij = out_kv_indices + out_kv_start + indice_id
        tl.store(
            out_ptr_ij,
            out_val,
            mask=valid_mask,
        )


def triton_convert_req_index_to_global_index(
    qo_indptr: torch.Tensor,  # int32 [num_tokens + 1]
    kv_indptr: torch.Tensor,  # int32 [num_tokens + 1]
    page_kv_indptr: torch.Tensor,  # int32 [num_tokens + 1]
    kv_indices: torch.Tensor,  # int32 [total_kv_seqlen]
    token_indices: torch.Tensor,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    BLOCK_SIZE: int = 1,  # page_block_size = 1 for now
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 128,  # tile width along columns
):
    """
    out[token_id, indice_id] =
        block_table[req_id[token_id],
            token_indices[token_id, indice_id] // BLOCK_SIZE] * BLOCK_SIZE
        + token_indices[token_id, indice_id] % BLOCK_SIZE

    Only when token_indices[token_id, indice_id] == -1 do we output -1.
    For safety, we also output -1 if the derived block_id would be
        out-of-bounds.
    """
    assert kv_indices.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert token_indices.shape[1] == NUM_TOPK_TOKENS
    assert NUM_TOPK_TOKENS % BLOCK_N == 0, (
        f"NUM_TOPK_TOKENS ({NUM_TOPK_TOKENS}) must be divisible by"
        f"BLOCK_N ({BLOCK_N})"
    )

    num_batch = kv_indptr.shape[0] - 1
    tiles_per_row = NUM_TOPK_TOKENS // BLOCK_N

    # Ensure contiguous tensors on the same device
    qo_indptr_c = qo_indptr.contiguous()
    kv_indptr_c = kv_indptr.contiguous()
    kv_indices_c = kv_indices.contiguous()
    token_indices_c = token_indices.contiguous()
    page_kv_indptr_c = page_kv_indptr.contiguous()
    # TODO: not support mtp
    new_kv_indices = torch.empty_like(kv_indices)

    # Strides in elements
    ti_stride0, ti_stride1 = token_indices_c.stride()

    # Exact 2D grid: tokens × column tiles
    grid = (num_batch, tiles_per_row)

    _convert_req_index_to_global_index_kernel[grid](
        qo_indptr_c,
        kv_indptr_c,
        page_kv_indptr_c,
        kv_indices_c,
        token_indices_c,
        new_kv_indices,
        # shapes / constexprs
        NUM_TOPK_TOKENS,
        BLOCK_SIZE,
        BLOCK_N,
        # strides
        ti_stride0,
        ti_stride1,
    )
    return new_kv_indices

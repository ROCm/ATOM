import torch
from torch import nn
from atom.utils.context import get_context, Context
from atom.model_ops.linear import ColumnParallelLinear, RowParallelLinear
from atom.model_ops.utils import get_and_maybe_dequant_weights
from aiter.rotary_embedding import RotaryEmbedding
from aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (  # noqa: E501 # isort: skip
    batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant as aiter_triton_fp8_bmm,
)
from vllm import _custom_ops as ops

from aiter import flash_attn_varlen_func, dtypes
from aiter.mla import mla_decode_fwd

from typing import Optional
from tqdm import tqdm
import logging

from aiter.dist.parallel_state import (
    get_pp_group,
    get_tp_group,
    get_tensor_model_parallel_world_size,
)

logger = logging.getLogger("atom")


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
        # MLA Specific Arguments
        q_lora_rank: Optional[int],
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        rotary_emb: RotaryEmbedding,
        q_proj: ColumnParallelLinear,
        kv_b_proj: ColumnParallelLinear,
        o_proj: RowParallelLinear,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype if kv_cache_dtype == "fp8" else "auto"

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.rotary_emb = rotary_emb
        self.q_proj = q_proj
        self.o_proj = o_proj
        self.kv_b_proj = kv_b_proj
        self.kv_cache = torch.tensor([])
        self._k_scale = torch.tensor(1.0, dtype=torch.float32)

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
            max_batch_size = 1024  # [ToDo] Find the optimal upper limit
            pre_compilation_list = list(range(1, max_batch_size + 1))
            if get_tp_group().is_first_rank:
                pre_compilation_list = tqdm(
                    pre_compilation_list,
                    desc="[Aiter Triton] Pre-compiling fp8 BMM kernel",
                    total=max_batch_size,
                )

            for m in pre_compilation_list:
                x = torch.empty(
                    (self.W_K.shape[0], m, self.W_K.shape[2]),
                    dtype=torch.bfloat16,
                    device=self.W_K.device,
                )
                aiter_triton_fp8_bmm(
                    x, self.W_K, self.W_K_scale, group_size=128, transpose_bm=True
                )

                x = torch.empty(
                    (self.W_V.shape[0], m, self.W_V.shape[2]),
                    dtype=torch.bfloat16,
                    device=self.W_V.device,
                )
                aiter_triton_fp8_bmm(
                    x, self.W_V, self.W_V_scale, group_size=128, transpose_bm=True
                )
        else:
            # Convert from (L, N, V) to (N, L, V)
            self.W_UV = W_UV.transpose(0, 1)
            # Convert from (L, N, P) to (N, P, L)
            self.W_UK_T = W_UK.permute(1, 2, 0)

    def _aiter_mla_decode_fwd(
        self,
        q: torch.Tensor,
        kv_buffer: torch.Tensor,
        o: torch.Tensor,
        qo_indptr: torch.Tensor,
        max_seqlen_qo: int,
        kv_indptr: Optional[torch.Tensor] = None,
        kv_indices: Optional[torch.Tensor] = None,
        kv_last_page_lens: Optional[torch.Tensor] = None,
        sm_scale: float = 1.0,
        logit_cap: float = 0.0,
    ) -> None:
        mla_decode_fwd(
            q,
            kv_buffer.view(-1, 1, 1, q.shape[-1]),
            o,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            max_seqlen_qo,
            sm_scale=sm_scale,
            logit_cap=logit_cap,
        )

    def _v_up_proj_and_o_proj(self, x):
        # Convert from (B, N, L) to (N, B, L)
        x = x.view(-1, self.num_heads, self.kv_lora_rank).transpose(0, 1)
        # Multiply (N, B, L) x (N, L, V) -> (N, B, V), Convert from (N, B, V) to (B, N, V)
        # x = torch.bmm(x, self.W_UV).transpose(0, 1)
        x = aiter_triton_fp8_bmm(
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
        ql_nope = aiter_triton_fp8_bmm(
            q_nope, self.W_K, self.W_K_scale, group_size=128, transpose_bm=True
        )
        return ql_nope, q_pe

    def _forward_prefill(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        context: Context,
    ) -> torch.Tensor:
        assert context is not None

        kv_nope = self.kv_b_proj(kv_c_normed).view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        if k_pe.dim() == 2:
            k_pe = k_pe.unsqueeze(1)
        k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)

        output = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=context.cu_seqlens_q,
            cu_seqlens_k=context.cu_seqlens_k,
            max_seqlen_q=context.max_seqlen_q,
            max_seqlen_k=context.max_seqlen_k,
            min_seqlen_q=context.min_seqlen_q,
            dropout_p=context.dropout_p,
            softmax_scale=self.scale,
            causal=True,
        )

        return self.o_proj(output.flatten(start_dim=-2))

    def _forward_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        context: Context,
    ) -> torch.Tensor:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert context is not None
        B = q_nope.shape[0]

        q = torch.cat([q_nope, q_pe], dim=-1)
        o = torch.empty(
            B, self.num_heads, self.kv_lora_rank, dtype=q.dtype, device=q.device
        )

        kv_buffer = kv_c_and_k_pe_cache.unsqueeze(2)

        self._aiter_mla_decode_fwd(
            q,
            kv_buffer,
            o,
            context.cu_seqlens_q,
            context.max_q_len,
            context.kv_indptr,
            context.kv_indices,
            context.kv_last_page_lens,
            self.scale,
        )

        return self._v_up_proj_and_o_proj(o)

    def forward(
        self,
        q: torch.Tensor,  # query in unified attn
        k_c_normed: torch.Tensor,  # key in unified attn
        k_pe: torch.Tensor,  # value in unified attn
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # TODO: kv cache and context
        kv_cache = self.kv_cache
        context = get_context()

        if context.is_prefill:
            prefill_q = self.q_proj(q).view(-1, self.num_heads, self.qk_head_dim)
            prefill_q_pe = prefill_q[..., self.qk_nope_head_dim :]
            self.rotary_emb(positions, prefill_q_pe, k_pe)

            if kv_cache.numel() > 0:
                ops.concat_and_cache_mla(
                    k_c_normed,
                    k_pe.squeeze(1),
                    kv_cache,
                    context.slot_mapping.flatten(),
                    kv_cache_dtype=self.kv_cache_dtype,
                    scale=self._k_scale,
                )

            output = self._forward_prefill(
                prefill_q, k_c_normed, k_pe, kv_cache, context
            )
        else:
            decode_ql_nope, decode_q_pe = self._q_proj_and_k_up_proj(q)
            self.rotary_emb(positions, decode_q_pe, k_pe)

            if kv_cache.numel() > 0:
                ops.concat_and_cache_mla(
                    k_c_normed,
                    k_pe.squeeze(1),
                    kv_cache,
                    context.slot_mapping.flatten(),
                    kv_cache_dtype=self.kv_cache_dtype,
                    scale=self._k_scale,
                )

            output = self._forward_decode(
                decode_ql_nope, decode_q_pe, kv_cache, context
            )

        return output

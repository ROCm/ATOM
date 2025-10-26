import torch
from torch import nn
from atom.utils.context import get_context, Context
from atom.utils.custom_register import direct_register_custom_op
from atom.utils.forward_context import ForwardContext, get_forward_context
from atom.model_ops.linear import ColumnParallelLinear, RowParallelLinear
from atom.model_ops.utils import get_and_maybe_dequant_weights
from aiter.rotary_embedding import RotaryEmbedding
from aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (  # noqa: E501 # isort: skip
    batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant as _aiter_triton_fp8_bmm,
)
from aiter import (
    flash_attn_varlen_func,
    concat_and_cache_mla,
    dtypes,
    QuantType,
    gemm_a8w8_blockscale,
    get_hip_quant,
)

from aiter.mla import mla_decode_fwd
from aiter.jit.utils.torch_guard import torch_compile_guard

from typing import Optional, Tuple
from tqdm import tqdm
import logging

from aiter.dist.parallel_state import get_tp_group


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


def aiter_triton_fp8_bmm(
    X: torch.Tensor,
    WQ: torch.Tensor,
    w_scale: torch.Tensor,
    group_size: int = 128,
    transpose_bm: Optional[bool] = False,
) -> torch.Tensor:
    return _aiter_triton_fp8_bmm(X, WQ, w_scale, group_size=group_size, transpose_bm=transpose_bm)


def fake_bmm(
    X: torch.Tensor,
    WQ: torch.Tensor,
    w_scale: torch.Tensor,
    group_size: int = 128,
    transpose_bm: Optional[bool] = False,
) -> torch.Tensor:
    return torch.zeros([X.shape[0], X.shape[1], WQ.shape[1]], dtype=X.dtype, device=X.device)


direct_register_custom_op(
    op_name="aiter_triton_fp8_bmm",
    op_func=aiter_triton_fp8_bmm,
    mutates_args=[],
    fake_impl=fake_bmm,
)


def aiter_mla_decode_fwd(
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

def _forward_rope_fake(
    positions: torch.Tensor,
    # if     is_nope_first
    # [[batch_size, seq_len, num_heads, nope_size+rope_size]
    # if NOT is_nope_first
    # [[batch_size, seq_len, num_heads, rope_size+nope_size],
    query: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
    head_size: int,
    rotary_dim: int,
    key: Optional[torch.Tensor] = None,
    offsets: Optional[torch.Tensor] = None,
    is_nope_first: bool=False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    query_shape = query.shape
    if key is not None:
        key_shape = key.shape
        return query.view(query_shape), key.view(key_shape)
    else:
        return query.view(query_shape)

@torch_compile_guard(gen_fake=_forward_rope_fake)
def _forward_rope(
    positions: torch.Tensor,
    # if     is_nope_first
    # [[batch_size, seq_len, num_heads, nope_size+rope_size]
    # if NOT is_nope_first
    # [[batch_size, seq_len, num_heads, rope_size+nope_size],
    query: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
    head_size: int,
    rotary_dim: int,
    key: Optional[torch.Tensor] = None,
    offsets: Optional[torch.Tensor] = None,
    is_nope_first: bool=False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    import aiter

    assert (
        cos.dtype == query.dtype
    ), f"cos_cache dtype ({cos.dtype}) does not match query dtype ({query.dtype})"

    rotate_style = 0 if is_neox_style else 1

    num_tokens = positions.numel()

    query_shape = query.shape
    query = query.view(1, num_tokens, -1, head_size)
    if key is not None:
        key_shape = key.shape
        key = key.view(1, num_tokens, -1, head_size)

    positions = positions.view(*query.shape[:2])
    if offsets is not None:
        offsets = offsets.view(*query.shape[:2])

    if not is_nope_first:
        query_ = query[..., : rotary_dim]
        key_ = key[..., : rotary_dim] if key is not None else None
    else:
        query_ = query[..., -rotary_dim:]
        key_ = key[..., -rotary_dim:] if key is not None else None

    if key_ is not None:
        if offsets is None:
            aiter.rope_cached_positions_2c_fwd_inplace(
                query_,
                key_,
                cos,
                sin,
                positions,
                rotate_style,
                reuse_freqs_front_part=True,
                nope_first=is_nope_first,
            )
        else:
            aiter.rope_cached_positions_offsets_2c_fwd_inplace(
                query_,
                key_,
                cos,
                sin,
                positions,
                offsets,
                rotate_style,
                reuse_freqs_front_part=True,
                nope_first=is_nope_first,
            )
        return query.view(query_shape), key.view(key_shape)
    else:
        if offsets is None:
            aiter.rope_cached_positions_fwd_inplace(
                query_,
                cos,
                sin,
                positions,
                rotate_style,
                reuse_freqs_front_part=True,
                nope_first=is_nope_first,
            )
        else:
            aiter.rope_cached_positions_offsets_fwd_inplace(
                query_,
                cos,
                sin,
                positions,
                offsets,
                rotate_style,
                reuse_freqs_front_part=True,
                nope_first=is_nope_first,
            )
        return query.view(query_shape)


def _q_proj_and_k_up_proj(x, q_proj_weight_data, q_proj_weight_scale_data, q_proj_tp_dim, q_proj_tp_size, q_proj_reduce_results, W_K, W_K_scale, num_heads, qk_head_dim, qk_nope_head_dim, qk_rope_head_dim):
    q_nope, q_pe = (
        torch.split(_forward_linear_fp8_no_bias(x, q_proj_weight_data, q_proj_weight_scale_data, q_proj_tp_dim, q_proj_tp_size, q_proj_reduce_results)
                    .view(-1, num_heads, qk_head_dim),
                    [qk_nope_head_dim, qk_rope_head_dim], dim=-1)
    )

    # Convert from (B, N, P) to (N, B, P)
    q_nope = q_nope.transpose(0, 1)
    # Multiply (N, B, P) x (N, P, L) -> (N, B, L), Convert from (N, B, L) to (B, N, L)
    # ql_nope = torch.bmm(q_nope, self.W_UK_T).transpose(0, 1)
    ql_nope = torch.ops.aiter.aiter_triton_fp8_bmm(
        q_nope, W_K, W_K_scale, group_size=128, transpose_bm=True
    )
    return ql_nope, q_pe


def _forward_prefill(
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    kv_c_and_k_pe_cache: torch.Tensor,
    kv_b_proj_weight_data,
    kv_b_proj_weight_scale_data,
    kv_b_proj_tp_dim,
    kv_b_proj_tp_size,
    kv_b_proj_reduce_results,
    o_proj_weight_data,
    o_proj_weight_scale_data,
    o_proj_tp_dim,
    o_proj_tp_size,
    o_proj_reduce_results,
    num_heads,
    qk_nope_head_dim,
    v_head_dim,
    scale,
    context: Context,
) -> torch.Tensor:
    assert context is not None

    kv_nope = _forward_linear_fp8_no_bias(kv_c_normed, kv_b_proj_weight_data, kv_b_proj_weight_scale_data, kv_b_proj_tp_dim, kv_b_proj_tp_size, kv_b_proj_reduce_results).view(
        -1, num_heads, qk_nope_head_dim + v_head_dim
    )
    k_nope, v = torch.split(kv_nope, [qk_nope_head_dim, v_head_dim], dim=-1)

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
        softmax_scale=scale,
        causal=True,
    )

    # return o_proj(output.flatten(start_dim=-2))
    return _forward_linear_fp8_no_bias(output.flatten(start_dim=-2), o_proj_weight_data, o_proj_weight_scale_data, o_proj_tp_dim, o_proj_tp_size, o_proj_reduce_results)


def _v_up_proj_and_o_proj(x, o_proj_weight_data, o_proj_weight_scale_data, o_proj_tp_dim, o_proj_tp_size, o_proj_reduce_results, W_V, W_V_scale, num_heads, kv_lora_rank, v_head_dim):
    # Convert from (B, N, L) to (N, B, L)
    x = x.view(-1, num_heads, kv_lora_rank).transpose(0, 1)
    # Multiply (N, B, L) x (N, L, V) -> (N, B, V), Convert from (N, B, V) to (B, N, V)
    # x = torch.bmm(x, self.W_UV).transpose(0, 1)
    x = torch.ops.aiter.aiter_triton_fp8_bmm(
        x, W_V, W_V_scale, group_size=128, transpose_bm=True
    )
    # Convert from (B, N, V) to (B, N * V)
    x = x.reshape(-1, num_heads * v_head_dim)
    # return o_proj(x)
    return _forward_linear_fp8_no_bias(x, o_proj_weight_data, o_proj_weight_scale_data, o_proj_tp_dim, o_proj_tp_size, o_proj_reduce_results)


def _forward_decode(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_c_and_k_pe_cache: torch.Tensor,
    o_proj_weight_data,
    o_proj_weight_scale_data,
    o_proj_tp_dim,
    o_proj_tp_size,
    o_proj_reduce_results,
    W_V,
    W_V_scale,
    num_heads,
    kv_lora_rank,
    v_head_dim,
    scale,
    context: Context,
) -> torch.Tensor:
    assert kv_c_and_k_pe_cache.numel() > 0
    assert context is not None
    B = q_nope.shape[0]

    q = torch.cat([q_nope, q_pe], dim=-1)
    o = torch.empty(
        B, num_heads, kv_lora_rank, dtype=q.dtype, device=q.device
    )

    kv_buffer = kv_c_and_k_pe_cache.unsqueeze(2)

    aiter_mla_decode_fwd(
        q,
        kv_buffer,
        o,
        context.cu_seqlens_q,
        context.max_q_len,
        context.kv_indptr,
        context.kv_indices,
        context.kv_last_page_lens,
        scale,
    )

    return _v_up_proj_and_o_proj(o, o_proj_weight_data, o_proj_weight_scale_data, o_proj_tp_dim, o_proj_tp_size, o_proj_reduce_results, W_V, W_V_scale, num_heads, kv_lora_rank, v_head_dim)


def _forward_linear_fp8_no_bias(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
                                tp_dim: int, tp_size: int, reduce_results: bool
                                ) -> torch.Tensor:
    quant_type = QuantType.per_1x128
    quant_func = get_hip_quant(quant_type)
    x, x_scale = quant_func(
        x,
        quant_dtype=dtypes.fp8,
        scale=None,
    )
    y = gemm_a8w8_blockscale(
        x, weight, x_scale, weight_scale, dtype=dtypes.bf16
    )
    if tp_dim == 1 and tp_size > 1 and reduce_results:
        y = get_tp_group().all_reduce(y, ca_fp8_quant=False)
    return y


def mla_attention(
    q: torch.Tensor,  # query in unified attn
    k_c_normed: torch.Tensor,  # key in unified attn
    k_pe: torch.Tensor,  # value in unified attn
    positions: torch.Tensor,
    W_K: torch.Tensor,
    W_V: torch.Tensor,
    W_K_scale: torch.Tensor,
    W_V_scale: torch.Tensor,
    rotary_emb_cos_cache: torch.Tensor,
    rotary_emb_sin_cache: torch.Tensor,
    rotary_emb_is_neox_style: bool,
    rotary_emb_head_size: int,
    rotary_emb_rotary_dim: int,
    q_proj_weight_data: torch.Tensor,
    q_proj_weight_scale_data: torch.Tensor,
    q_proj_tp_dim: int,
    q_proj_tp_size: int,
    q_proj_reduce_results: bool,
    kv_b_proj_weight_data: torch.Tensor,
    kv_b_proj_weight_scale_data: torch.Tensor,
    kv_b_proj_tp_dim: int,
    kv_b_proj_tp_size: int,
    kv_b_proj_reduce_results: bool,
    o_proj_weight_data: torch.Tensor,
    o_proj_weight_scale_data: torch.Tensor,
    o_proj_tp_dim: int,
    o_proj_tp_size: int,
    o_proj_reduce_results: bool,
    kv_lora_rank: int,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    qk_head_dim: int,
    v_head_dim: int,
    kv_cache_dtype: str,
    scale: float,
    layer_num: int
) -> torch.Tensor:
    context = get_context()
    if context.slot_mapping.numel():
        forward_context: ForwardContext = get_forward_context()
        attn_metadata_ = forward_context.no_compile_layers[layer_num]
        kv_cache = attn_metadata_.k_cache
    else:
        # dummy run before allocate kv_cache, thus we create manually
        kv_cache = torch.tensor([])

    _k_scale = torch.tensor(1.0, dtype=torch.float32)

    if context.is_prefill:
        prefill_q = _forward_linear_fp8_no_bias(q, q_proj_weight_data, q_proj_weight_scale_data,
                                                q_proj_tp_dim, q_proj_tp_size, q_proj_reduce_results).view(-1, num_heads, qk_head_dim)
        prefill_q_pe = prefill_q[..., qk_nope_head_dim:]
        _forward_rope(positions, prefill_q_pe, rotary_emb_cos_cache, rotary_emb_sin_cache,
                      rotary_emb_is_neox_style, rotary_emb_head_size, rotary_emb_rotary_dim, k_pe)

        if kv_cache.numel() > 0:
            concat_and_cache_mla(
                k_c_normed,
                k_pe.squeeze(1),
                kv_cache,
                context.slot_mapping.flatten(),
                kv_cache_dtype=kv_cache_dtype,
                scale=_k_scale,
            )

        output = _forward_prefill(
            prefill_q, k_c_normed, k_pe, kv_cache, kv_b_proj_weight_data, kv_b_proj_weight_scale_data, kv_b_proj_tp_dim, kv_b_proj_tp_size, kv_b_proj_reduce_results, o_proj_weight_data, o_proj_weight_scale_data, o_proj_tp_dim, o_proj_tp_size, o_proj_reduce_results, num_heads, qk_nope_head_dim, v_head_dim, scale, context
        )
    else:
        decode_ql_nope, decode_q_pe = _q_proj_and_k_up_proj(q, q_proj_weight_data, q_proj_weight_scale_data, q_proj_tp_dim,
                                                            q_proj_tp_size, q_proj_reduce_results, W_K, W_K_scale, num_heads, qk_head_dim, qk_nope_head_dim, qk_rope_head_dim)
        _forward_rope(positions, decode_q_pe, rotary_emb_cos_cache, rotary_emb_sin_cache,
                      rotary_emb_is_neox_style, rotary_emb_head_size, rotary_emb_rotary_dim, k_pe)

        if kv_cache.numel() > 0:
            concat_and_cache_mla(
                k_c_normed,
                k_pe.squeeze(1),
                kv_cache,
                context.slot_mapping.flatten(),
                kv_cache_dtype=kv_cache_dtype,
                scale=_k_scale,
            )

        output = _forward_decode(
            decode_ql_nope, decode_q_pe, kv_cache, o_proj_weight_data, o_proj_weight_scale_data, o_proj_tp_dim, o_proj_tp_size, o_proj_reduce_results, W_V, W_V_scale, num_heads, kv_lora_rank, v_head_dim, scale, context
        )

    return output


def fake_mla_attention(
    q: torch.Tensor,  # query in unified attn
    k_c_normed: torch.Tensor,  # key in unified attn
    k_pe: torch.Tensor,  # value in unified attn
    positions: torch.Tensor,
    W_K: torch.Tensor,
    W_V: torch.Tensor,
    W_K_scale: torch.Tensor,
    W_V_scale: torch.Tensor,
    rotary_emb_cos_cache: torch.Tensor,
    rotary_emb_sin_cache: torch.Tensor,
    rotary_emb_is_neox_style: bool,
    rotary_emb_head_size: int,
    rotary_emb_rotary_dim: int,
    q_proj_weight_data: torch.Tensor,
    q_proj_weight_scale_data: torch.Tensor,
    q_proj_tp_dim: int,
    q_proj_tp_size: int,
    q_proj_reduce_results: bool,
    kv_b_proj_weight_data: torch.Tensor,
    kv_b_proj_weight_scale_data: torch.Tensor,
    kv_b_proj_tp_dim: int,
    kv_b_proj_tp_size: int,
    kv_b_proj_reduce_results: bool,
    o_proj_weight_data: torch.Tensor,
    o_proj_weight_scale_data: torch.Tensor,
    o_proj_tp_dim: int,
    o_proj_tp_size: int,
    o_proj_reduce_results: bool,
    kv_lora_rank: int,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    qk_head_dim: int,
    v_head_dim: int,
    kv_cache_dtype: str,
    scale: float,
    layer_num: int
) -> torch.Tensor:
    output_shape = list(q.shape)
    output_shape[-1] = 7168  # TODO: read from hidden_size
    return torch.zeros(output_shape, dtype=q.dtype, device=q.device)


direct_register_custom_op(
    op_name="mla_attention",
    op_func=mla_attention,
    mutates_args=[],
    fake_impl=fake_mla_attention,
)


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
        layer_num: int = 0,
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
        W_UK, W_UV = torch.split(kv_b_proj_weight,
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
                torch.ops.aiter.aiter_triton_fp8_bmm(
                    x, self.W_K, self.W_K_scale, group_size=128, transpose_bm=True
                )

                x = torch.empty(
                    (self.W_V.shape[0], m, self.W_V.shape[2]),
                    dtype=torch.bfloat16,
                    device=self.W_V.device,
                )
                torch.ops.aiter.aiter_triton_fp8_bmm(
                    x, self.W_V, self.W_V_scale, group_size=128, transpose_bm=True
                )
        else:
            # Convert from (L, N, V) to (N, L, V)
            self.W_UV = W_UV.transpose(0, 1)
            # Convert from (L, N, P) to (N, P, L)
            self.W_UK_T = W_UK.permute(1, 2, 0)

    def forward(
        self,
        q: torch.Tensor,  # query in unified attn
        k_c_normed: torch.Tensor,  # key in unified attn
        k_pe: torch.Tensor,  # value in unified attn
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.aiter.mla_attention(q, k_c_normed, k_pe, positions, self.W_K, self.W_V, self.W_K_scale, self.W_V_scale,
                                             self.rotary_emb.cos_cache, self.rotary_emb.sin_cache, self.rotary_emb.is_neox_style, self.rotary_emb.head_size, self.rotary_emb.rotary_dim,
                                             self.q_proj.weight.data, self.q_proj.weight_scale.data, self.q_proj.tp_dim, self.q_proj.tp_size, self.q_proj.reduce_results,
                                             self.kv_b_proj.weight.data, self.kv_b_proj.weight_scale.data, self.kv_b_proj.tp_dim, self.kv_b_proj.tp_size, self.kv_b_proj.reduce_results,
                                             self.o_proj.weight.data, self.o_proj.weight_scale.data, self.o_proj.tp_dim, self.o_proj.tp_size, self.o_proj.reduce_results,
                                             self.kv_lora_rank, self.num_heads,
                                             self.qk_nope_head_dim, self.qk_rope_head_dim, self.qk_head_dim, self.v_head_dim, self.kv_cache_dtype,
                                             self.scale, self.layer_num)

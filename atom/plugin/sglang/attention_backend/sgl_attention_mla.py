"""Sglang-specific MLA forward and weight processing for DeepseekV2/V3.

DeepSeek MLA (Multi-Latent Attention) forward logic for sglang plugin mode:
absorbed BMM computation, MHA/MLA path dispatch (prefill -> MHA, decode -> MLA),
kv_b_proj weight splitting (w_kc/w_vc), and monkey-patch setup via
setup_deepseek_for_sglang().

This module is lazily imported from base_model_wrapper.py only when running in
sglang plugin mode (``is_sglang() == True``).  Keeping all sglang-dependent
imports here avoids crashing when sglang is not installed.

TODO: rewrite this file once sglang's attention flow is unified into ATOM's
attention layer — the MLA absorbed path and MHA dispatch will then be handled
natively by ATOM's attention ops, making this sglang-specific module unnecessary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple, Optional

import torch
from aiter import dtypes
from aiter.dist.parallel_state import get_tensor_model_parallel_world_size, get_tp_group
from aiter.ops.triton.fused_fp8_quant import fused_rms_fp8_group_quant
from atom.model_ops.base_attention import Attention
from atom.model_ops.attention_mla import fused_qk_rope_concat_and_cache_mla
from atom.models.utils import maybe_prefix

# sglang imports
from sglang.srt.layers.communicator import AttentionInputs, get_attn_tp_context
from sglang.srt.layers.attention.nsa.utils import nsa_use_prefill_cp
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.models.deepseek_common.utils import (
    _use_aiter_gfx95,
    _is_hip,
    _is_cpu,
    _is_cpu_amx_available,
    _is_cuda,
    _is_fp8_fnuz,
    _is_npu,
    awq_dequantize_func,
)
from sglang.srt.layers.quantization.rocm_mxfp4_utils import (
    batched_gemm_afp4wfp4_pre_quant,
)
from aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
    batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant,
)
from sglang.srt.layers.quantization.fp8_kernel import (
    per_tensor_quant_mla_fp8,
    per_token_group_quant_mla_deep_gemm_masked_fp8,
)
from sglang.srt.utils import bind_or_assign, get_bool_env_var

if TYPE_CHECKING:
    from atom.models.deepseek_v2 import DeepseekV2MLAAttention


# bmm_fp8 custom-op wrapper (adapted from sglang forward_mla.py)
if _is_cuda:
    from sgl_kernel import bmm_fp8 as _raw_bmm_fp8
    from sglang.srt.utils.custom_op import register_custom_op

    @register_custom_op(mutates_args=["out"])
    def _bmm_fp8_op(
        A: torch.Tensor,
        B: torch.Tensor,
        out: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
    ) -> None:
        _raw_bmm_fp8(A, B, A_scale, B_scale, out.dtype, out)

    def bmm_fp8(A, B, A_scale, B_scale, dtype, out=None):
        if out is None:
            out = torch.empty(
                (A.shape[0], A.shape[1], B.shape[2]),
                device=A.device,
                dtype=dtype,
            )
        _bmm_fp8_op(A, B, out, A_scale, B_scale)
        return out

else:

    def bmm_fp8(A, B, A_scale, B_scale, dtype, out=None):
        raise RuntimeError("bmm_fp8 requires CUDA (sgl_kernel)")


# NamedTuple for prepare → core data flow
class SglPrepareResult(NamedTuple):
    q_pe: torch.Tensor
    k_pe: torch.Tensor
    q_nope_out: torch.Tensor
    k_nope: torch.Tensor
    forward_batch: Any
    zero_allocator: Any
    positions: torch.Tensor
    topk_indices: Optional[torch.Tensor]
    llama_4_scaling: Optional[Any]


class SglMhaPrepareResult(NamedTuple):
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    forward_batch: Any


def _unwrap_linear_output(output: Any) -> torch.Tensor:
    """Normalize ATOM/public-SGLang linear outputs to a tensor."""
    if isinstance(output, tuple):
        return output[0]
    return output


# Init helpers
def init_sgl_attrs(
    attn: DeepseekV2MLAAttention,
    config,
    kv_cache_dtype: str = "bf16",
) -> None:
    """Initialise sglang-only attributes on DeepseekV2MLAAttention."""
    from sglang.srt.configs.model_config import is_deepseek_nsa

    attn.use_nsa = is_deepseek_nsa(config)
    attn.use_deep_gemm_bmm = False
    attn.alt_stream = None
    attn.kv_cache_dtype = kv_cache_dtype
    attn.use_fused_qk_rope_concat_and_cache_mla = _use_aiter_gfx95
    attn.current_sgl_plugin_attn_path = None
    attn.w_kc, attn.w_vc = None, None
    attn.w_scale = None
    attn.w_scale_k = None
    attn.w_scale_v = None
    attn.attn_mha = Attention(
        num_heads=attn.num_local_heads,
        head_dim=attn.qk_head_dim,
        scale=attn.scaling,
        num_kv_heads=attn.num_local_heads,
        kv_cache_dtype=kv_cache_dtype,
        layer_num=attn.layer_num,
        use_mla=False,
        v_head_dim=attn.v_head_dim,
        prefix=maybe_prefix(attn.prefix, "attn_mha"),
    )
    if hasattr(attn.attn_mha, "attn"):
        attn.attn_mha.attn.kv_b_proj = None


# Absorbed batched-matmul (shared by prepare and core)
def mla_absorbed_bmm(
    attn: DeepseekV2MLAAttention,
    inp: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: Optional[torch.Tensor],
    weight_scale_k: Optional[torch.Tensor],
    out_dim: int,
) -> torch.Tensor:
    """Batched matmul for MLA absorbed weights (w_kc / w_vc).

    Handles deep_gemm, mxfp4, fp8-triton, fp8-cublas, and bf16 fallback paths.
    inp: (num_tokens, num_heads, in_dim) — token-major
    Returns: (num_tokens, num_heads, out_dim) — token-major
    """
    if attn.use_deep_gemm_bmm:
        from sglang.srt.layers import deep_gemm_wrapper

        val, scale, masked_m, expected_m, aligned_m = (
            per_token_group_quant_mla_deep_gemm_masked_fp8(inp.transpose(0, 1))
        )
        out = inp.new_empty((attn.num_local_heads, aligned_m, out_dim))
        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
            (val, scale),
            (weight, weight_scale_k),
            out,
            masked_m,
            expected_m,
        )
        return out[:, :expected_m, :].transpose(0, 1)

    if _is_hip:
        if _use_aiter_gfx95 and weight.dtype == torch.uint8:
            x = inp.transpose(0, 1)
            out = torch.empty(
                x.shape[0],
                x.shape[1],
                weight.shape[2],
                device=x.device,
                dtype=torch.bfloat16,
            )
            batched_gemm_afp4wfp4_pre_quant(
                x,
                weight.transpose(-2, -1),
                weight_scale_k.transpose(-2, -1),
                torch.bfloat16,
                out,
            )
            return out.transpose(0, 1)

        if (_use_aiter_gfx95 and weight.dtype == torch.float8_e4m3fn) or (
            get_is_capture_mode() and weight.dtype == torch.float8_e4m3fnuz
        ):
            out = (
                batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
                    X=inp,
                    WQ=weight.transpose(-1, -2),
                    w_scale=weight_scale,
                    group_size=128,
                    YQ=None,
                    transpose_bm=False,
                    transpose_bm_in=True,
                    dtype=torch.bfloat16,
                )
            )
            return out.transpose(0, 1)

        w_bf16 = weight.to(torch.bfloat16)
        if weight_scale is not None:
            w_bf16 = w_bf16 * weight_scale
        out = torch.bmm(
            inp.to(torch.bfloat16).transpose(0, 1),
            w_bf16,
        )
        return out.transpose(0, 1)

    # CUDA fp8 path
    if weight.dtype == torch.float8_e4m3fn:
        val, scale = per_tensor_quant_mla_fp8(
            inp.transpose(0, 1),
            torch.zeros((1,), dtype=torch.float32, device=inp.device),
        )
        out = bmm_fp8(val, weight, scale, weight_scale, torch.bfloat16)
        return out.transpose(0, 1)

    # bf16 fallback
    return torch.bmm(inp.transpose(0, 1), weight).transpose(0, 1)


# Forward: prepare → core
def forward_sgl_prepare(
    attn: DeepseekV2MLAAttention,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    **model_kwargs,
) -> SglPrepareResult:
    """Prepare QKV for sglang MLA attention (adapted from sglang forward_absorb_prepare)."""
    hidden_states_scale = None
    if isinstance(hidden_states, tuple):
        hidden_states, hidden_states_scale = hidden_states

    forward_batch = model_kwargs.get("forward_batch", None)
    zero_allocator = model_kwargs.get("zero_allocator", None)
    llama_4_scaling = model_kwargs.get("llama_4_scaling", None)
    q_lora = None
    topk_indices = None

    if attn.q_lora_rank is not None:
        q, latent_cache = (
            get_attn_tp_context()
            .fetch_qkv_latent()
            .split(
                [attn.q_lora_rank, attn.kv_lora_rank + attn.qk_rope_head_dim],
                dim=-1,
            )
        )

        if (
            q.shape[0] != positions.shape[0]
            and get_tensor_model_parallel_world_size() > 1
        ):
            qkv_lora = torch.cat([q, latent_cache], dim=-1)
            qkv_lora = get_tp_group().all_gather(qkv_lora, dim=0)
            if qkv_lora.shape[0] < positions.shape[0]:
                raise RuntimeError(
                    f"qkv_lora gather mismatch: got {qkv_lora.shape[0]}, "
                    f"expected {positions.shape[0]}"
                )
            qkv_lora = qkv_lora[: positions.shape[0]]
            q, latent_cache = torch.split(
                qkv_lora,
                [attn.q_lora_rank, attn.kv_lora_rank + attn.qk_rope_head_dim],
                dim=-1,
            )

        k_nope = latent_cache[..., : attn.kv_lora_rank]

        # overlap qk norm
        if attn.alt_stream is not None and get_is_capture_mode():
            current_stream = torch.cuda.current_stream()
            attn.alt_stream.wait_stream(current_stream)
            q = attn.q_a_layernorm(q)
            with torch.cuda.stream(attn.alt_stream):
                k_nope = attn.kv_a_layernorm(k_nope)
            current_stream.wait_stream(attn.alt_stream)
        else:
            q = attn.q_a_layernorm(q)
            k_nope = attn.kv_a_layernorm(k_nope)

        if attn.use_nsa:
            if q_lora is None:
                q_lora = q

        # overlap q_b_proj and indexer during decode
        if (
            attn.alt_stream is not None
            and get_is_capture_mode()
            and forward_batch.forward_mode.is_decode_or_idle()
            and q_lora is not None
        ):
            current_stream = torch.cuda.current_stream()
            attn.alt_stream.wait_stream(current_stream)
            with torch.cuda.stream(attn.alt_stream):
                k_nope = k_nope.unsqueeze(1)
                q = attn.q_b_proj(q).view(-1, attn.num_local_heads, attn.qk_head_dim)
            topk_indices = attn.indexer(
                x=hidden_states,
                q_lora=q_lora,
                positions=positions,
                forward_batch=forward_batch,
                layer_id=attn.layer_num,
            )
            current_stream.wait_stream(attn.alt_stream)
        else:
            k_nope = k_nope.unsqueeze(1)
            q = attn.q_b_proj(q).view(-1, attn.num_local_heads, attn.qk_head_dim)
            if q_lora is not None:
                topk_indices = attn.indexer(
                    x=hidden_states,
                    q_lora=q_lora,
                    positions=positions,
                    forward_batch=forward_batch,
                    layer_id=attn.layer_num,
                )
    else:
        q = _unwrap_linear_output(attn.q_proj(hidden_states)).view(
            -1, attn.num_local_heads, attn.qk_head_dim
        )
        latent_cache = _unwrap_linear_output(attn.kv_a_proj_with_mqa(hidden_states))
        k_nope = latent_cache[..., : attn.kv_lora_rank]
        k_nope = attn.kv_a_layernorm(k_nope).unsqueeze(1)

    q_nope, q_pe = q.split([attn.qk_nope_head_dim, attn.qk_rope_head_dim], dim=-1)
    k_pe = latent_cache[..., attn.kv_lora_rank :].unsqueeze(1)

    q_nope_out = mla_absorbed_bmm(
        attn, q_nope, attn.w_kc, attn.w_scale, attn.w_scale_k, attn.kv_lora_rank
    )

    if attn.rotary_emb is not None and not attn.use_fused_qk_rope_concat_and_cache_mla:
        q_pe, k_pe = attn.rotary_emb(positions, q_pe, k_pe)

    if nsa_use_prefill_cp(forward_batch):
        k_nope, k_pe = attn.rebuild_cp_kv_cache(
            latent_cache, forward_batch, k_nope, k_pe
        )

    return SglPrepareResult(
        q_pe=q_pe,
        k_pe=k_pe,
        q_nope_out=q_nope_out,
        k_nope=k_nope,
        forward_batch=forward_batch,
        zero_allocator=zero_allocator,
        positions=positions,
        topk_indices=topk_indices,
        llama_4_scaling=llama_4_scaling,
    )


def forward_sgl_core(
    attn: DeepseekV2MLAAttention,
    prepared: SglPrepareResult,
) -> torch.Tensor:
    """Core MLA attention computation for sglang (adapted from sglang forward_absorb_core)."""
    save_kv_cache = True

    if attn.use_fused_qk_rope_concat_and_cache_mla:
        mla_attn = _get_sglang_radix_attn(attn.mla_attn)
        kv_cache = prepared.forward_batch.token_to_kv_pool.get_key_buffer(
            mla_attn.layer_id
        )
        q_out_dtype = (
            dtypes.fp8
            if attn.kv_cache_dtype == "fp8_e4m3"
            else prepared.q_nope_out.dtype
        )
        q = torch.empty(
            (
                prepared.q_nope_out.shape[0],
                attn.num_local_heads,
                attn.kv_lora_rank + attn.qk_rope_head_dim,
            ),
            dtype=q_out_dtype,
            device=prepared.q_nope_out.device,
        )

        fused_qk_rope_concat_and_cache_mla(
            prepared.q_nope_out,
            prepared.q_pe,
            prepared.k_nope,
            prepared.k_pe,
            kv_cache,
            q,
            prepared.forward_batch.out_cache_loc,
            mla_attn.k_scale,
            mla_attn.k_scale,
            prepared.positions,
            attn.rotary_emb.cos_cache,
            attn.rotary_emb.sin_cache,
            is_neox=attn.rotary_emb.is_neox_style,
            is_nope_first=True,
        )
        # Decode/speculative MLA consumes q plus packed MLA cache directly.
        k = None
        v = None
        save_kv_cache = False
    else:
        q = torch.cat([prepared.q_nope_out, prepared.q_pe], dim=-1)
        k = torch.cat([prepared.k_nope, prepared.k_pe], dim=-1)
        v = prepared.k_nope

    if prepared.llama_4_scaling is not None:
        q = q * prepared.llama_4_scaling

    extra_kwargs = {}
    if prepared.topk_indices is not None:
        extra_kwargs["topk_indices"] = prepared.topk_indices

    attn_output = attn.mla_attn(
        q,
        k,
        v,
        forward_batch=prepared.forward_batch,
        save_kv_cache=save_kv_cache,
        **extra_kwargs,
    )
    attn_output = attn_output.view(-1, attn.num_local_heads, attn.kv_lora_rank)

    # up-proj by w_vc
    attn_bmm_output = mla_absorbed_bmm(
        attn, attn_output, attn.w_vc, attn.w_scale, attn.w_scale_v, attn.v_head_dim
    ).flatten(1, 2)

    return attn.o_proj(attn_bmm_output)


def _dispatch_sgl_plugin_attn_path(forward_batch) -> str:
    """Decide the attention algorithm for this batch based on forward_mode.

    Returns "mha" for extend/prefill (uses standard Q×K×V with flash_attn)
    or "mla" for decode (uses absorbed weights + mla_decode_fwd).

    This is the per-batch *routing* decision, distinct from
    ``_can_run_sgl_mha_now`` which is a *capability* gate checking whether
    the model configuration supports the MHA path at all.
    """
    if forward_batch.forward_mode.is_extend_without_speculative():
        return "mha"
    return "mla"


def forward_sgl_plugin_mode_mla(
    attn: DeepseekV2MLAAttention,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    **model_kwargs,
) -> torch.Tensor:
    prepared = forward_sgl_prepare(attn, positions, hidden_states, **model_kwargs)
    return forward_sgl_core(attn, prepared)


def _get_sglang_radix_attn(attn_module):
    return attn_module.attn if hasattr(attn_module, "attn") else attn_module


def _set_mla_kv_buffer_for_mha(
    attn: DeepseekV2MLAAttention,
    kv_a: torch.Tensor,
    k_pe: torch.Tensor,
    forward_batch,
) -> None:
    attn_mha = _get_sglang_radix_attn(attn.attn_mha)
    cache_k = torch.cat([kv_a.unsqueeze(1), k_pe], dim=-1)
    forward_batch.token_to_kv_pool.set_kv_buffer(
        attn_mha,
        forward_batch.out_cache_loc,
        cache_k,
        cache_k,
    )


def _can_run_sgl_mha_now(attn: DeepseekV2MLAAttention, forward_batch) -> bool:
    """Check if the model configuration supports the MHA attention path.

    This is a *capability* gate — NSA models and MXFP4-quantised weights
    (uint8) cannot use the MHA path. Distinct from
    ``_dispatch_sgl_plugin_attn_path`` which routes each batch.
    """
    del forward_batch
    if attn.use_nsa:
        return False
    if attn.kv_b_proj.weight.dtype == torch.uint8:
        return False
    return True


def forward_sgl_mha_prepare(
    attn: DeepseekV2MLAAttention,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    **model_kwargs,
) -> SglMhaPrepareResult:
    forward_batch = model_kwargs.get("forward_batch", None)
    if forward_batch is None:
        raise RuntimeError("forward_batch is required in forward_sgl_mha_prepare")

    hidden_states_scale = None
    if isinstance(hidden_states, tuple):
        hidden_states, hidden_states_scale = hidden_states

    attn_mha = _get_sglang_radix_attn(attn.attn_mha)
    if getattr(attn_mha, "kv_b_proj", None) is None:
        attn_mha.kv_b_proj = attn.kv_b_proj

    if attn.q_lora_rank is not None:
        q, latent_cache = (
            get_attn_tp_context()
            .fetch_qkv_latent()
            .split(
                [attn.q_lora_rank, attn.kv_lora_rank + attn.qk_rope_head_dim],
                dim=-1,
            )
        )

        if (
            q.shape[0] != positions.shape[0]
            and get_tensor_model_parallel_world_size() > 1
        ):
            qkv_lora = torch.cat([q, latent_cache], dim=-1)
            qkv_lora = get_tp_group().all_gather(qkv_lora, dim=0)
            if qkv_lora.shape[0] < positions.shape[0]:
                raise RuntimeError(
                    f"qkv_lora gather mismatch: got {qkv_lora.shape[0]}, "
                    f"expected {positions.shape[0]}"
                )
            qkv_lora = qkv_lora[: positions.shape[0]]
            q, latent_cache = torch.split(
                qkv_lora,
                [attn.q_lora_rank, attn.kv_lora_rank + attn.qk_rope_head_dim],
                dim=-1,
            )

        if _use_aiter_gfx95 and attn.q_b_proj.weight.dtype == torch.float8_e4m3fn:
            (q, q_scale), _, _, _ = fused_rms_fp8_group_quant(
                q,
                attn.q_a_layernorm.weight,
                attn.q_a_layernorm.eps,
                None,
                None,
                None,
                group_size=128,
                dtype_quant=torch.float8_e4m3fn,
                res1=None,
                output_unquantized_inp1=False,
                transpose_scale=True,
            )
            q = _unwrap_linear_output(attn.q_b_proj(q, q_scale)).view(
                -1, attn.num_local_heads, attn.qk_head_dim
            )
        else:
            q = attn.q_a_layernorm(q)
            q = _unwrap_linear_output(attn.q_b_proj(q)).view(
                -1, attn.num_local_heads, attn.qk_head_dim
            )
    else:
        q = _unwrap_linear_output(attn.q_proj(hidden_states, hidden_states_scale)).view(
            -1, attn.num_local_heads, attn.qk_head_dim
        )
        latent_cache = _unwrap_linear_output(
            attn.kv_a_proj_with_mqa(hidden_states, hidden_states_scale)
        )

    _, q_pe = q.split([attn.qk_nope_head_dim, attn.qk_rope_head_dim], dim=-1)
    kv_a, _ = latent_cache.split([attn.kv_lora_rank, attn.qk_rope_head_dim], dim=-1)
    latent_cache = latent_cache.unsqueeze(1)

    if _use_aiter_gfx95 and attn.kv_b_proj.weight.dtype == torch.float8_e4m3fn:
        (kv_a_quanted, kv_a_quanted_scale), kv_a, _, _ = fused_rms_fp8_group_quant(
            kv_a,
            attn.kv_a_layernorm.weight,
            attn.kv_a_layernorm.eps,
            None,
            None,
            None,
            group_size=128,
            dtype_quant=torch.float8_e4m3fn,
            res1=None,
            output_unquantized_inp1=True,
            transpose_scale=True,
        )
    else:
        kv_a_quanted = None
        kv_a = attn.kv_a_layernorm(kv_a)

    k_pe = latent_cache[:, :, attn.kv_lora_rank :]
    if attn.rotary_emb is not None:
        q_pe, k_pe = attn.rotary_emb(positions, q_pe, k_pe)
    q[..., attn.qk_nope_head_dim :] = q_pe

    _set_mla_kv_buffer_for_mha(attn, kv_a, k_pe, forward_batch)

    if kv_a_quanted is not None:
        kv = _unwrap_linear_output(attn.kv_b_proj(kv_a_quanted, kv_a_quanted_scale))
    else:
        kv = _unwrap_linear_output(attn.kv_b_proj(kv_a))
    kv = kv.view(-1, attn.num_local_heads, attn.qk_nope_head_dim + attn.v_head_dim)
    k_nope = kv[..., : attn.qk_nope_head_dim]
    v = kv[..., attn.qk_nope_head_dim :]
    k = torch.cat(
        [k_nope, k_pe.expand(-1, attn.num_local_heads, -1)],
        dim=-1,
    )
    return SglMhaPrepareResult(q=q, k=k, v=v, forward_batch=forward_batch)


def forward_sgl_mha_core(
    attn: DeepseekV2MLAAttention,
    prepared: SglMhaPrepareResult,
) -> torch.Tensor:
    attn_output = attn.attn_mha(
        prepared.q,
        prepared.k,
        prepared.v,
        forward_batch=prepared.forward_batch,
        save_kv_cache=False,
    )
    attn_output = attn_output.reshape(-1, attn.num_local_heads * attn.v_head_dim)
    return attn.o_proj(attn_output)


def forward_sgl_plugin_mode_mha(
    attn: DeepseekV2MLAAttention,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    **model_kwargs,
) -> torch.Tensor:
    forward_batch = model_kwargs.get("forward_batch", None)
    if forward_batch is None:
        raise RuntimeError("forward_batch is required in forward_sgl_plugin_mode_mha")
    if not _can_run_sgl_mha_now(attn, forward_batch):
        attn.current_sgl_plugin_attn_path = "mla_fallback"
        return forward_sgl_plugin_mode_mla(
            attn,
            positions,
            hidden_states,
            **model_kwargs,
        )
    prepared = forward_sgl_mha_prepare(attn, positions, hidden_states, **model_kwargs)
    return forward_sgl_mha_core(attn, prepared)


def prepare_qkv_latent(
    attn: DeepseekV2MLAAttention,
    hidden_states: torch.Tensor,
    forward_batch,
) -> torch.Tensor:
    """Prepare QKV latent tensor for the sglang communicator."""
    assert attn.q_lora_rank is not None
    hidden_states_scale = None
    if isinstance(hidden_states, tuple):
        hidden_states, hidden_states_scale = hidden_states
    qkv_lora = attn.fused_qkv_a_proj(hidden_states, hidden_states_scale)

    # Fallback: when communicator does not enable input_scattered gather,
    # force qkv latent token dimension to align with positions.
    expected_tokens = 0
    if hasattr(forward_batch, "positions") and forward_batch.positions is not None:
        expected_tokens = int(forward_batch.positions.shape[0])
    if expected_tokens <= 0:
        expected_tokens = int(getattr(forward_batch, "seq_lens_sum", 0) or 0)

    if (
        expected_tokens > 0
        and qkv_lora.shape[0] != expected_tokens
        and get_tensor_model_parallel_world_size() > 1
    ):
        qkv_lora = get_tp_group().all_gather(qkv_lora, dim=0)
        if qkv_lora.shape[0] > expected_tokens:
            qkv_lora = qkv_lora[:expected_tokens]
        elif qkv_lora.shape[0] < expected_tokens:
            raise RuntimeError(
                f"prepare_qkv_latent gather mismatch: got {qkv_lora.shape[0]}, "
                f"expected {expected_tokens}"
            )
    return qkv_lora


# Top-level forward entry point
def forward_sgl_plugin_mode(
    attn: DeepseekV2MLAAttention,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    **model_kwargs,
) -> torch.Tensor:
    """Full MLA forward in sglang plugin mode."""
    forward_batch = model_kwargs.get("forward_batch", None)
    if forward_batch is None:
        raise RuntimeError("forward_batch is required in forward_sgl_plugin_mode")

    attn_tp_context = get_attn_tp_context()
    with attn_tp_context.maybe_input_scattered(forward_batch):
        if attn.q_lora_rank is not None:
            attn_tp_context.set_attn_inputs(
                AttentionInputs(
                    hidden_states,
                    forward_batch,
                    lambda hs, fb: prepare_qkv_latent(attn, hs, fb),
                )
            )
        attn_path = _dispatch_sgl_plugin_attn_path(forward_batch)
        attn.current_sgl_plugin_attn_path = attn_path
        if attn_path == "mha":
            return forward_sgl_plugin_mode_mha(
                attn,
                positions,
                hidden_states,
                **model_kwargs,
            )
        if attn_path == "mla":
            return forward_sgl_plugin_mode_mla(
                attn,
                positions,
                hidden_states,
                **model_kwargs,
            )
        raise ValueError(f"Unsupported plugin attention path: {attn_path}")


# Weight post-processing: decomposed into sub-functions
def _read_kv_b_proj_weight(attn: DeepseekV2MLAAttention) -> torch.Tensor:
    """Read kv_b_proj weight, handling AWQ and fnuz dtypes."""
    if hasattr(attn.kv_b_proj, "qweight"):
        awq_dequant = awq_dequantize_func()
        if awq_dequant is None:
            raise ValueError(
                "AWQ dequantize function is not supported for current device"
            )
        w = awq_dequant(
            attn.kv_b_proj.qweight,
            attn.kv_b_proj.scales,
            attn.kv_b_proj.qzeros,
        ).T
    else:
        w = attn.kv_b_proj.weight

    # On ROCm, ATOM creates parameters with fnuz dtype but loads fn bytes.
    # View-cast back to fn so the normalize path works correctly.
    if _is_fp8_fnuz and w.dtype == torch.float8_e4m3fnuz:
        w = w.view(torch.float8_e4m3fn)

    return w


def _get_weight_block_size(attn: DeepseekV2MLAAttention) -> Optional[list[int]]:
    """Derive weight_block_size from ATOM's quant_type system."""
    from aiter import QuantType as _AiterQuantType

    qt = getattr(attn.kv_b_proj, "quant_type", None)
    if qt == _AiterQuantType.per_1x128:
        return [128, 128]
    elif qt == _AiterQuantType.per_1x32:
        return [1, 32]
    return None


def _process_fp8_weight(
    attn: DeepseekV2MLAAttention,
    w: torch.Tensor,
    weight_block_size: Optional[list[int]],
) -> tuple[torch.Tensor, bool, Optional[torch.Tensor]]:
    """Process FP8 weights for kv_b_proj.

    Returns (w, use_deep_gemm_bmm, block_scale).
    """
    from atom.model_ops.utils import normalize_e4m3fn_to_e4m3fnuz
    from sglang.srt.layers.quantization.fp8_utils import (
        block_quant_dequant,
        block_quant_to_tensor_quant,
        channel_quant_to_tensor_quant,
        inverse_transform_scale_ue8m0,
    )
    from sglang.srt.layers.deep_gemm_wrapper import (
        ENABLE_JIT_DEEPGEMM,
        DEEPGEMM_BLACKWELL,
    )
    from sglang.srt.model_loader.utils import should_deepgemm_weight_requant_ue8m0

    use_deep_gemm_bmm = False
    block_scale = None

    if weight_block_size is not None:
        assert hasattr(attn.kv_b_proj, "weight_scale_inv") or hasattr(
            attn.kv_b_proj, "weight_scale"
        )
        weight_scale = (
            attn.kv_b_proj.weight_scale
            if hasattr(attn.kv_b_proj, "weight_scale")
            else attn.kv_b_proj.weight_scale_inv
        )

        if _is_fp8_fnuz and w.dtype == torch.float8_e4m3fn:
            weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                weight=w, weight_scale=weight_scale, input_scale=None
            )
        else:
            weight = w

        if should_deepgemm_weight_requant_ue8m0(
            weight_block_size=weight_block_size
        ) and getattr(weight_scale, "format_ue8m0", False):
            weight_scale = inverse_transform_scale_ue8m0(
                weight_scale, mn=weight.shape[-2]
            )

        if _is_cuda and weight_block_size[0] == 128 and weight_block_size[1] == 128:
            if (
                ENABLE_JIT_DEEPGEMM
                and not DEEPGEMM_BLACKWELL
                and get_bool_env_var("SGL_USE_DEEPGEMM_BMM", "false")
            ):
                block_scale = weight_scale
                use_deep_gemm_bmm = True
            else:
                w = block_quant_dequant(
                    weight, weight_scale, weight_block_size, torch.bfloat16
                )
        else:
            w, scale = block_quant_to_tensor_quant(
                weight, weight_scale, weight_block_size
            )
            attn.w_scale = scale
    else:
        if w.dtype == torch.float8_e4m3fn and _is_fp8_fnuz:
            weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                weight=w, weight_scale=attn.kv_b_proj.weight_scale, input_scale=None
            )
        else:
            weight = w
            weight_scale = attn.kv_b_proj.weight_scale

        w, scale = channel_quant_to_tensor_quant(weight, weight_scale)
        attn.w_scale = scale

    return w, use_deep_gemm_bmm, block_scale


def _process_int8_weight(
    attn: DeepseekV2MLAAttention,
    w: torch.Tensor,
    weight_block_size: Optional[list[int]],
) -> torch.Tensor:
    """Process INT8 weights for kv_b_proj."""
    from sglang.srt.layers.quantization.int8_utils import (
        block_dequant as int8_block_dequant,
    )

    if weight_block_size is not None:
        assert hasattr(attn.kv_b_proj, "weight_scale_inv")
        return int8_block_dequant(
            w, attn.kv_b_proj.weight_scale_inv, weight_block_size
        ).to(torch.bfloat16)
    else:
        return w.to(torch.bfloat16) * attn.kv_b_proj.weight_scale.to(torch.bfloat16)


def _split_and_assign_kc_vc(
    attn: DeepseekV2MLAAttention,
    w: torch.Tensor,
    use_deep_gemm_bmm: bool,
    block_scale: Optional[torch.Tensor],
    weight_block_size: Optional[list[int]],
) -> None:
    """Split weight into kc/vc and assign to attn."""
    from atom.model_ops.utils import quark_post_load_weights

    w_kc, w_vc = w.unflatten(0, (-1, attn.qk_nope_head_dim + attn.v_head_dim)).split(
        [attn.qk_nope_head_dim, attn.v_head_dim], dim=1
    )

    # quark fp4 special path
    quant_method = getattr(attn.kv_b_proj, "quant_method", None)
    quant_config = getattr(quant_method, "quant_config", None)
    if (
        _use_aiter_gfx95
        and quant_config is not None
        and quant_config.get_name() == "quark"
    ):
        w_kc, attn.w_scale_k, w_vc, attn.w_scale_v = quark_post_load_weights(
            attn, w, "mxfp4"
        )

    if not use_deep_gemm_bmm:
        attn.w_kc = bind_or_assign(
            attn.w_kc, w_kc.transpose(1, 2).contiguous().transpose(1, 2)
        )
        w_vc = w_vc.contiguous().transpose(1, 2)
        if _is_npu:
            w_vc = w_vc.contiguous()
        attn.w_vc = bind_or_assign(attn.w_vc, w_vc)

        if hasattr(attn.kv_b_proj, "weight_scale") and attn.w_scale is None:
            attn.w_scale = bind_or_assign(attn.w_scale, attn.kv_b_proj.weight_scale)
            if _is_hip:
                attn.w_scale *= 2.0

        if _is_cpu and _is_cpu_amx_available and w.dtype == torch.float8_e4m3fn:
            attn.w_kc = attn.w_kc.to(torch.bfloat16) * attn.w_scale
            attn.w_vc = attn.w_vc.to(torch.bfloat16) * attn.w_scale
    else:
        num_tiles_k = attn.qk_nope_head_dim // weight_block_size[1]
        num_tiles_n = attn.v_head_dim // weight_block_size[0]
        ws_kc, ws_vc = block_scale.unflatten(
            0, (-1, (num_tiles_k + num_tiles_n))
        ).split([num_tiles_k, num_tiles_n], dim=1)

        attn.w_scale_k = bind_or_assign(
            attn.w_scale_k, ws_kc.transpose(1, 2).contiguous()
        )
        attn.w_scale_v = bind_or_assign(attn.w_scale_v, ws_vc.contiguous())
        attn.w_kc = bind_or_assign(attn.w_kc, w_kc.transpose(1, 2).contiguous())
        attn.w_vc = bind_or_assign(attn.w_vc, w_vc.contiguous())
        attn.use_deep_gemm_bmm = True


def process_mla_kv_b_proj_after_loading(attn: DeepseekV2MLAAttention) -> None:
    """Process kv_b_proj weights after loading for sglang MLA mode.

    Orchestrates reading, quantization handling, and splitting of
    kv_b_proj into absorbed w_kc / w_vc weights.
    """
    w = _read_kv_b_proj_weight(attn)
    weight_block_size = _get_weight_block_size(attn)

    use_deep_gemm_bmm = False
    block_scale = None

    # fp8 path
    if w.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        w, use_deep_gemm_bmm, block_scale = _process_fp8_weight(
            attn, w, weight_block_size
        )

    # int8 path
    if w.dtype == torch.int8:
        w = _process_int8_weight(attn, w, weight_block_size)

    # split and assign kc/vc
    _split_and_assign_kc_vc(attn, w, use_deep_gemm_bmm, block_scale, weight_block_size)


# One-time model setup (called from base_model_wrapper.py)
def setup_deepseek_for_sglang(model) -> None:
    """Patch a DeepseekV2/V3 model for sglang plugin mode.

    - Initialises sglang TP context
    - Patches each MLAAttention.forward to dispatch to the sglang MLA path
    - Registers process_weights_after_loading hooks
    - Stores atom_config on the model
    """
    config = model.config

    # Store atom_config (needed by load_weights in the OOT wrapper)
    if not hasattr(model, "atom_config"):
        from atom.config import get_current_atom_config

        model.atom_config = get_current_atom_config()

    kv_cache_dtype = model.atom_config.kv_cache_dtype

    # Initialise sglang TP context for MLA gather/scatter
    from sglang.srt.configs.model_config import is_deepseek_nsa
    from sglang.srt.layers.communicator import get_attn_tp_context

    get_attn_tp_context().init_context(config.q_lora_rank, is_deepseek_nsa(config))

    # Patch each MLAAttention instance
    from atom.models.deepseek_v2 import DeepseekV2MLAAttention

    for module in model.modules():
        if isinstance(module, DeepseekV2MLAAttention):
            _patch_mla_attention_for_sglang(module, config, kv_cache_dtype)


def _patch_mla_attention_for_sglang(attn, config, kv_cache_dtype: str = "bf16") -> None:
    """Patch a single DeepseekV2MLAAttention for sglang plugin mode.

    We patch attn.forward (rather than relying solely on ops.Attention =
    RadixAttention) because MLA's absorbed-weight forward path replaces the
    *entire* forward method — including RoPE, and absorbed
    BMM — not just the attention backend.  ops.Attention = RadixAttention
    handles the backend layer (flash_attn / paged_attn dispatch) and is
    already set via set_attn_cls(); this patch sits above that layer.
    """
    init_sgl_attrs(attn, config, kv_cache_dtype)

    def patched_forward(
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        from atom.plugin.sglang.models.base_model_wrapper import (
            get_current_forward_batch,
        )

        kwargs["forward_batch"] = get_current_forward_batch()
        return forward_sgl_plugin_mode(attn, positions, hidden_states, **kwargs)

    attn.forward = patched_forward
    attn.process_weights_after_loading = lambda: process_mla_kv_b_proj_after_loading(
        attn
    )

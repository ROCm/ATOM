# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Helper functions for DeepSeek MLA in SGLang plugin mode.

This module now contains only the low-level helpers that are still shared by
the SGLang DeepSeek MLA wrapper and the install-time weight hooks:
absorbed BMM math, small utility helpers, non-absorbed cache staging, and
kv_b_proj post-load processing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

import torch
from aiter import QuantType, dtypes, get_hip_quant
from atom.model_ops.base_attention import Attention
from atom.model_ops.attention_mla import (
    dynamic_per_batched_tensor_quant,
)
from atom.models.deepseek_v2 import (
    _fuse_rmsnorm_quant,
    _mxfp4_activation_quant_layout,
)
from atom.models.utils import maybe_prefix

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
from aiter.utility.fp4_utils import e8m0_to_f32, mxfp4_to_f32
from aiter.ops.triton.batched_gemm_afp4wfp4_pre_quant import (
    batched_gemm_a16wfp4,
)

try:
    from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
except ImportError:
    gemm_afp4wfp4 = None

try:
    from aiter.ops.triton.fused_gemm_afp4wfp4_split_cat import (
        fused_gemm_afp4wfp4_preshuffle_split_cat,
    )
except ImportError:
    fused_gemm_afp4wfp4_preshuffle_split_cat = None

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


logger = logging.getLogger(__name__)


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


def _unwrap_linear_output(output: Any) -> torch.Tensor:
    """Normalize ATOM/public-SGLang linear outputs to a tensor."""
    if isinstance(output, tuple):
        return output[0]
    return output


def use_sglang_fp8_prefill_attn() -> bool:
    return (
        get_bool_env_var("SGLANG_AITER_FP8_PREFILL_ATTN", "True") and _use_aiter_gfx95
    )


def try_fused_mxfp4_kv_b_proj_fp8(
    kv_b_proj: Any,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    *,
    num_heads: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Return FP8 k/v from MXFP4 kv_b_proj when the fused split-cat path fits."""
    if fused_gemm_afp4wfp4_preshuffle_split_cat is None:
        return None
    weight = getattr(kv_b_proj, "weight", None)
    weight_scale = getattr(kv_b_proj, "weight_scale", None)
    if weight is None or weight_scale is None:
        return None
    if weight.dtype not in (torch.uint8, getattr(dtypes, "fp4x2", None)):
        return None
    if weight.dim() != 2 or weight.shape[0] % 16 != 0:
        return None
    if weight_scale.shape[0] % 32 != 0:
        return None

    kvc_for_gemm = kv_c_normed.reshape(-1, kv_c_normed.shape[-1]).contiguous()
    if k_pe.dim() == 2:
        k_pe = k_pe.unsqueeze(1)

    m = kvc_for_gemm.shape[0]
    q_input, x_scale = get_hip_quant(QuantType.per_1x32)(
        kvc_for_gemm,
        quant_dtype=dtypes.fp4x2,
        shuffle=(m >= 32),
    )
    if m >= 32:
        x_scale = x_scale.view(torch.uint8).view(x_scale.shape[0] // 32, -1)
    else:
        x_scale = x_scale[:m, ...].view(torch.uint8)

    return fused_gemm_afp4wfp4_preshuffle_split_cat(
        q_input.view(torch.uint8),
        weight.view(torch.uint8).view(weight.shape[0] // 16, -1),
        k_pe.expand((-1, num_heads, -1)),
        x_scale,
        weight_scale.view(torch.uint8).view(weight_scale.shape[0] // 32, -1),
        qk_nope_head_dim,
        v_head_dim,
        dtypes.fp8,
    )


def _linear_quant_type_value(linear: Any) -> Optional[int]:
    quant_type = getattr(linear, "quant_type", None)
    return None if quant_type is None else getattr(quant_type, "value", quant_type)


def _fuse_qk_rmsnorm_and_q_quant(
    attn: DeepseekV2MLAAttention,
    q: torch.Tensor,
    k_nope: torch.Tensor,
    *,
    output_unquantized_q: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Fuse q/k RMSNorm and q quant using ATOM's DeepSeek-V2 path."""

    if getattr(attn, "quant_dtype", None) == dtypes.fp4x2:
        q_shuffle, q_scale_shuffle_padding = _mxfp4_activation_quant_layout(q.shape[0])
    else:
        q_shuffle = False
        q_scale_shuffle_padding = False

    (q_quantized, q_scale), q_normed, k_nope_normed, _ = _fuse_rmsnorm_quant(
        q,
        attn.q_a_layernorm.weight,
        attn.q_a_layernorm.eps,
        k_nope,
        attn.kv_a_layernorm.weight,
        attn.kv_a_layernorm.eps,
        None,
        dtype_quant=attn.quant_dtype,
        shuffle=q_shuffle,
        scale_shuffle_padding=q_scale_shuffle_padding,
        group_size=128,
        quant_type=_linear_quant_type_value(attn.q_b_proj),
        output_unquantized_inp1=output_unquantized_q,
        transpose_scale=True,
    )
    return q_quantized, q_scale, q_normed, k_nope_normed


def _q_b_proj_with_optional_scale(
    attn: DeepseekV2MLAAttention,
    q: torch.Tensor,
    q_scale: Optional[torch.Tensor],
) -> torch.Tensor:
    if q_scale is None:
        return _unwrap_linear_output(attn.q_b_proj(q))
    if getattr(attn, "quant_dtype", None) == dtypes.fp4x2:
        return _unwrap_linear_output(attn.q_b_proj(q, q_scale))
    return _unwrap_linear_output(attn.q_b_proj(q, q_scale))


def _fuse_qk_rmsnorm(
    attn: DeepseekV2MLAAttention,
    q: torch.Tensor,
    k_nope: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse q/k RMSNorm without quantizing q."""
    (q_normed, _), _, k_nope_normed, _ = _fuse_rmsnorm_quant(
        q,
        attn.q_a_layernorm.weight,
        attn.q_a_layernorm.eps,
        k_nope,
        attn.kv_a_layernorm.weight,
        attn.kv_a_layernorm.eps,
        None,
        dtype_quant=torch.bfloat16,
        shuffle=False,
        scale_shuffle_padding=False,
        group_size=128,
        quant_type=None,
        output_unquantized_inp1=False,
        transpose_scale=False,
    )
    return q_normed, k_nope_normed


def _prepare_weight_for_bmm(
    weight: torch.Tensor, in_dim: int, out_dim: int
) -> torch.Tensor:
    """Normalize absorbed weight layout for torch.bmm fallback."""
    if weight.shape[1] == in_dim and weight.shape[2] == out_dim:
        return weight
    if weight.shape[1] == out_dim and weight.shape[2] == in_dim:
        return weight.transpose(-2, -1)
    raise RuntimeError(
        "Unexpected absorbed weight shape for bmm fallback: "
        f"{tuple(weight.shape)} with in_dim={in_dim}, out_dim={out_dim}"
    )


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
    attn.use_fused_qk_rope_concat_and_cache_mla = (
        kv_cache_dtype == "fp8_e4m3" or _use_aiter_gfx95
    )
    attn.current_sgl_plugin_attn_path = None
    attn.w_kc, attn.w_vc = None, None
    attn.w_scale = None
    attn.w_scale_k = None
    attn.w_scale_v = None
    attn.attn_non_absorbed = Attention(
        num_heads=attn.num_local_heads,
        head_dim=attn.qk_head_dim,
        scale=attn.scaling,
        num_kv_heads=attn.num_local_heads,
        kv_cache_dtype=kv_cache_dtype,
        layer_num=attn.layer_num,
        use_mla=False,
        v_head_dim=attn.v_head_dim,
        prefix=maybe_prefix(attn.prefix, "attn_non_absorbed"),
    )
    _bind_non_absorbed_kv_b_proj(attn)


def mla_absorbed_bmm(
    attn: DeepseekV2MLAAttention,
    inp: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: Optional[torch.Tensor],
    weight_scale_k: Optional[torch.Tensor],
    out_dim: int,
) -> torch.Tensor:
    """Batched matmul for MLA absorbed weights (w_kc / w_vc)."""
    effective_weight_scale = (
        weight_scale_k if weight_scale_k is not None else weight_scale
    )

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
            x = inp.transpose(0, 1)
            out = (
                batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
                    X=x,
                    WQ=weight,
                    w_scale=effective_weight_scale,
                    group_size=128,
                    YQ=None,
                    transpose_bm=True,
                    transpose_bm_in=False,
                    dtype=torch.bfloat16,
                )
            )
            return out

        w_bf16 = _prepare_weight_for_bmm(weight, inp.shape[-1], out_dim).to(
            torch.bfloat16
        )
        if effective_weight_scale is not None:
            w_bf16 = w_bf16 * effective_weight_scale
        out = torch.bmm(
            inp.to(torch.bfloat16).transpose(0, 1),
            w_bf16,
        )
        return out.transpose(0, 1)

    if weight.dtype == torch.float8_e4m3fn:
        val, scale = per_tensor_quant_mla_fp8(
            inp.transpose(0, 1),
            torch.zeros((1,), dtype=torch.float32, device=inp.device),
        )
        out = bmm_fp8(val, weight, scale, effective_weight_scale, torch.bfloat16)
        return out.transpose(0, 1)

    return torch.bmm(inp.transpose(0, 1), weight).transpose(0, 1)


def mla_v_up_proj(
    attn: DeepseekV2MLAAttention,
    inp: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: Optional[torch.Tensor],
    weight_scale_k: Optional[torch.Tensor],
    out_dim: int,
) -> torch.Tensor:
    """Project MLA decode output to a flat o_proj input."""
    effective_weight_scale = (
        weight_scale_k if weight_scale_k is not None else weight_scale
    )
    if _is_hip and _use_aiter_gfx95 and weight.dtype == torch.uint8:
        x = inp.transpose(0, 1)
        out = torch.empty(
            (inp.shape[0], attn.num_local_heads * out_dim),
            device=inp.device,
            dtype=torch.bfloat16,
        )
        out_3d = out.view(inp.shape[0], attn.num_local_heads, out_dim)
        batched_gemm_a16wfp4(
            x,
            weight.transpose(-2, -1),
            weight_scale_k.transpose(-2, -1),
            dtype=torch.bfloat16,
            y=out_3d,
            transpose_bm=True,
            prequant=True,
            y_scale=None,
        )
        return out

    if _is_hip and (
        (_use_aiter_gfx95 and weight.dtype == torch.float8_e4m3fn)
        or (get_is_capture_mode() and weight.dtype == torch.float8_e4m3fnuz)
    ):
        x = inp.transpose(0, 1)
        out = torch.empty(
            (inp.shape[0], attn.num_local_heads * out_dim),
            device=inp.device,
            dtype=torch.bfloat16,
        )
        out_3d = out.view(inp.shape[0], attn.num_local_heads, out_dim)
        batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
            X=x,
            WQ=weight,
            w_scale=effective_weight_scale,
            group_size=128,
            YQ=out_3d,
            transpose_bm=True,
            transpose_bm_in=False,
            dtype=torch.bfloat16,
        )
        return out

    return mla_absorbed_bmm(
        attn, inp, weight, weight_scale, weight_scale_k, out_dim
    ).flatten(1, 2)


def _get_sglang_radix_attn(attn_module):
    return attn_module.attn if hasattr(attn_module, "attn") else attn_module


def _bind_non_absorbed_kv_b_proj(attn: DeepseekV2MLAAttention) -> None:
    """Expose DeepSeek's latent-KV projection on the non-absorbed SGLang layer."""

    if not hasattr(attn, "attn_non_absorbed"):
        return
    attn_non_absorbed = _get_sglang_radix_attn(attn.attn_non_absorbed)
    attn_non_absorbed.kv_b_proj = attn.kv_b_proj


def _concat_mha_k_for_non_absorbed(
    attn: DeepseekV2MLAAttention,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
) -> torch.Tensor:
    k = k_nope.new_empty(
        k_nope.shape[0],
        attn.num_local_heads,
        attn.qk_nope_head_dim + attn.qk_rope_head_dim,
    )

    try:
        from sglang.srt.layers.attention.utils import concat_and_cast_mha_k_triton
    except ImportError as exc:
        logger.warning(
            "Unable to import concat_and_cast_mha_k_triton; "
            "falling back to torch native non-absorbed K concat: %s",
            exc,
        )
    else:
        concat_and_cast_mha_k_triton(k, k_nope, k_pe)
        return k

    k[..., : attn.qk_nope_head_dim] = k_nope
    k[..., attn.qk_nope_head_dim :] = k_pe
    return k


def _set_mla_kv_buffer_for_non_absorbed(
    attn: DeepseekV2MLAAttention,
    kv_a: torch.Tensor,
    k_pe: torch.Tensor,
    forward_batch,
) -> None:
    attn_non_absorbed = _get_sglang_radix_attn(attn.attn_non_absorbed)
    cache_k = torch.cat([kv_a.unsqueeze(1), k_pe], dim=-1)
    forward_batch.token_to_kv_pool.set_kv_buffer(
        attn_non_absorbed,
        forward_batch.out_cache_loc,
        cache_k,
        cache_k,
    )


def _is_mxfp4_kv_b_proj(attn: DeepseekV2MLAAttention) -> bool:
    kv_b_proj = attn.kv_b_proj
    params_dtype = getattr(kv_b_proj, "params_dtype", None)
    if params_dtype == dtypes.fp4x2 or params_dtype == getattr(
        torch, "float4_e2m1fn_x2", None
    ):
        return True

    quant_type = getattr(kv_b_proj, "quant_type", None)
    if getattr(quant_type, "name", "") == "per_1x32" or str(quant_type).endswith(
        "per_1x32"
    ):
        return True

    quant_method = getattr(kv_b_proj, "quant_method", None)
    quant_config = getattr(quant_method, "quant_config", None)
    return bool(
        quant_config is not None
        and quant_config.get_name() == "quark"
        and kv_b_proj.weight.dtype == torch.uint8
    )


def _can_run_non_absorbed_mla_now(
    attn: DeepseekV2MLAAttention,
    forward_batch,
) -> bool:
    """Check if the model configuration supports the non-absorbed MLA path.

    This is a capability gate. NSA models cannot use the non-absorbed path.
    MXFP4 ``kv_b_proj`` weights are supported because that path expands K/V
    through ``attn.kv_b_proj`` itself, which owns the per_1x32 GEMM
    implementation.
    """
    del forward_batch
    if attn.use_nsa:
        return False
    if attn.kv_b_proj.weight.dtype == torch.uint8 and not _is_mxfp4_kv_b_proj(attn):
        return False
    return True


def _is_static_quark_mxfp4_linear(linear: Any) -> bool:
    layer_quant_config = getattr(linear, "layer_quant_config", None)
    return (
        getattr(linear, "weight", None) is not None
        and linear.weight.dim() == 2
        and layer_quant_config is not None
        and layer_quant_config.quant_method == "quark"
        and layer_quant_config.quant_dtype == dtypes.fp4x2
        and getattr(layer_quant_config.quant_type, "name", None) == "per_1x32"
        and getattr(linear, "source_quant_dtype", None) is None
    )


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
        if _is_static_quark_mxfp4_linear(attn.kv_b_proj):
            w = getattr(
                attn.kv_b_proj,
                "_mxfp4_unshuffled_weight",
                attn.kv_b_proj.weight,
            )
        else:
            w = attn.kv_b_proj.weight

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
    """Process FP8 weights for kv_b_proj."""
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


def _split_kc_vc_like_vllm(
    attn: DeepseekV2MLAAttention, w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split kv_b_proj weight using vLLM's transpose-first layout."""
    kv_b_proj_weight = w.T
    assert kv_b_proj_weight.shape == (
        attn.kv_lora_rank,
        attn.num_local_heads * (attn.qk_nope_head_dim + attn.v_head_dim),
    ), (
        f"{kv_b_proj_weight.shape=}, "
        f"{attn.kv_lora_rank=}, "
        f"{attn.num_local_heads=}, "
        f"{attn.qk_nope_head_dim=}, "
        f"{attn.v_head_dim=}"
    )
    kv_b_proj_weight = kv_b_proj_weight.view(
        attn.kv_lora_rank,
        attn.num_local_heads,
        attn.qk_nope_head_dim + attn.v_head_dim,
    )
    w_uk, w_uv = kv_b_proj_weight.split(
        [attn.qk_nope_head_dim, attn.v_head_dim], dim=-1
    )
    return w_uk.transpose(0, 1).contiguous(), w_uv.permute(1, 2, 0).contiguous()


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

    # Quark MXFP4 LinearBase modules store quantization on layer_quant_config;
    # there is no kv_b_proj.quant_method object to inspect in this path.
    layer_quant_config = getattr(attn.kv_b_proj, "layer_quant_config", None)
    quant_method = getattr(attn.kv_b_proj, "quant_method", None)
    quant_config = getattr(quant_method, "quant_config", None)
    is_quark_quant_method = (
        quant_config is not None and quant_config.get_name() == "quark"
    )
    is_quark_mxfp4_linear = (
        layer_quant_config is not None
        and layer_quant_config.quant_method == "quark"
        and layer_quant_config.quant_dtype == dtypes.fp4x2
        and getattr(layer_quant_config.quant_type, "name", None) == "per_1x32"
    )
    if _use_aiter_gfx95 and (is_quark_quant_method or is_quark_mxfp4_linear):
        quark_weight = w
        weight_scale = getattr(attn.kv_b_proj, "_mxfp4_unshuffled_weight_scale", None)
        if is_quark_mxfp4_linear and weight_scale is not None:
            quark_weight = mxfp4_to_f32(w.view(torch.uint8)).to(torch.bfloat16)
            quark_weight = quark_weight * e8m0_to_f32(
                weight_scale.repeat_interleave(32, dim=-1)
            ).to(torch.bfloat16)
        w_kc, attn.w_scale_k, w_vc, attn.w_scale_v = quark_post_load_weights(
            attn, quark_weight, "mxfp4"
        )

    if not use_deep_gemm_bmm:
        use_vllm_weight_layout = _is_hip and not (
            is_quark_quant_method or is_quark_mxfp4_linear
        )

        if use_vllm_weight_layout:
            w_kc, w_vc = _split_kc_vc_like_vllm(attn, w)
        else:
            w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
            w_vc = w_vc.contiguous().transpose(1, 2)

        if w.dtype == torch.bfloat16 and (_is_hip or _is_cuda):
            w_kc, w_scale_k = dynamic_per_batched_tensor_quant(w_kc, dtype=dtypes.fp8)
            w_vc, w_scale_v = dynamic_per_batched_tensor_quant(w_vc, dtype=dtypes.fp8)
            attn.w_scale_k = bind_or_assign(attn.w_scale_k, w_scale_k)
            attn.w_scale_v = bind_or_assign(attn.w_scale_v, w_scale_v)

        attn.w_kc = bind_or_assign(attn.w_kc, w_kc)
        if _is_npu:
            w_vc = w_vc.contiguous()
        attn.w_vc = bind_or_assign(attn.w_vc, w_vc)

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
    _bind_non_absorbed_kv_b_proj(attn)
    if _is_static_quark_mxfp4_linear(attn.kv_b_proj) and not getattr(
        attn.kv_b_proj, "_sgl_mxfp4_process_done", False
    ):
        attn.kv_b_proj.process_weights_after_loading()

    w = _read_kv_b_proj_weight(attn)
    weight_block_size = _get_weight_block_size(attn)

    use_deep_gemm_bmm = False
    block_scale = None

    if w.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        w, use_deep_gemm_bmm, block_scale = _process_fp8_weight(
            attn, w, weight_block_size
        )

    if w.dtype == torch.int8:
        w = _process_int8_weight(attn, w, weight_block_size)

    _split_and_assign_kc_vc(attn, w, use_deep_gemm_bmm, block_scale, weight_block_size)


def _patch_linear_for_sglang_mxfp4_preserve(linear: Any) -> None:
    """Preserve a DeepSeek MLA projection's original MXFP4 layout before shuffle."""
    if getattr(linear, "_sgl_mxfp4_preserve_patched", False):
        return

    orig_process_weights_after_loading = linear.process_weights_after_loading

    def process_weights_after_loading_with_mxfp4_preserve():
        if getattr(linear, "_sgl_mxfp4_process_done", False):
            return

        if _is_static_quark_mxfp4_linear(linear):
            linear._mxfp4_unshuffled_weight = linear.weight.detach().clone()
            linear._mxfp4_unshuffled_weight_scale = linear.weight_scale.detach().clone()
        orig_process_weights_after_loading()
        linear._sgl_mxfp4_process_done = True

    linear.process_weights_after_loading = (
        process_weights_after_loading_with_mxfp4_preserve
    )
    linear._sgl_mxfp4_preserve_patched = True


def _patch_attention_projs_for_sglang_mxfp4(
    attn: DeepseekV2MLAAttention,
) -> None:
    """Preserve DeepSeek MLA q/kv projections' original MXFP4 layouts."""
    if hasattr(attn, "q_b_proj"):
        _patch_linear_for_sglang_mxfp4_preserve(attn.q_b_proj)
    _patch_linear_for_sglang_mxfp4_preserve(attn.kv_b_proj)

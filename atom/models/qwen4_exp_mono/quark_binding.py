# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Zero-copy contracts for the fixed Quark weight and QSA cache layouts.

These checks run during the all-rank bind, before any layer writes state or
publishes kernel pointers. Native owns storage; immutable records retain it.
"""

from dataclasses import dataclass, fields

import torch

from atom.models.qwen4_exp_mono.dims import Dims
from atom.models.qwen4_exp_mono.plan import LayerTask
from atom.models.qwen4_exp_mono.shapes import (
    HC,
    HIDDEN,
    QSA_COMPRESS_RATIO,
    QSA_HEAD_DIM,
    QSA_INDEX_HEAD_DIM,
    QSA_MROPE_INTERLEAVED,
    QSA_MROPE_SECTION,
    QSA_ROTARY_DIM,
    ROUTER_OUTPUTS,
)
from atom.models.qwen4_exp_mono.weights import (
    GdnWeights,
    HcWeights,
    QsaWeights,
    _bind_gdn,
    _bind_hc,
    _bind_qsa,
    _bind_rope,
    _tensor,
)
from atom.mono.runtime.consensus import MonoUnsupported


@dataclass(frozen=True)
class QuarkQsaCache:
    k: torch.Tensor
    v: torch.Tensor
    k_scale: torch.Tensor | None
    v_scale: torch.Tensor | None
    raw: torch.Tensor
    compressed: torch.Tensor
    positions: torch.Tensor


@dataclass(frozen=True)
class QuarkQsaEpilogue:
    """Norm/RoPE owners for Native PTPC projections and Mono KV publication."""

    q_norm: torch.Tensor
    k_norm: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    index_q_norm: torch.Tensor
    index_k_norm: torch.Tensor
    index_cos: torch.Tensor
    index_sin: torch.Tensor


@dataclass(frozen=True)
class QuarkLayerWeights:
    attn_hc: HcWeights
    mlp_hc: HcWeights
    router: torch.Tensor
    gdn: GdnWeights | None
    qsa: QsaWeights | QuarkQsaEpilogue | None
    cache: QuarkQsaCache | None
    w13: torch.Tensor
    w2: torch.Tensor
    w13_scale: torch.Tensor
    w2_scale: torch.Tensor


def _check_device(record, device):
    for field in fields(record):
        value = getattr(record, field.name)
        if isinstance(value, torch.Tensor):
            if value.device != device:
                raise MonoUnsupported(
                    f"Quark {field.name} is on {value.device}, expected {device}"
                )
        elif value is not None:
            _check_device(value, device)


def bind_hc(module, *, has_inject=True):
    if (
        module.hc_count != HC
        or module.hc_norm.variance_epsilon != 1e-6
        or (module.block_inject_weight is not None) != has_inject
    ):
        raise MonoUnsupported(
            "Quark HC needs four streams, epsilon 1e-6 and the expected injection"
        )
    return _bind_hc(module, has_inject=has_inject)


def _check_qsa_math(attn, max_model_len):
    from atom.model_ops.layernorm import GemmaRMSNorm

    for name, rope in (("QSA", attn.rotary_emb), ("index", attn.indexer.rotary_emb)):
        if (
            getattr(rope, "rotary_dim", None) != QSA_ROTARY_DIM
            or tuple(getattr(rope, "mrope_section", ()) or ()) != QSA_MROPE_SECTION
            or getattr(rope, "mrope_interleaved", False) != QSA_MROPE_INTERLEAVED
            or not getattr(rope, "is_neox_style", False)
            or rope.cos_cache.shape[0] < max_model_len
            or rope.cos_cache.shape != rope.sin_cache.shape
        ):
            raise MonoUnsupported(
                f"Quark {name} RoPE requires rotary64, interleaved (11,11,10) and full capacity"
            )
    norms = (
        attn.q_norm,
        attn.k_norm,
        attn.indexer.q_layernorm,
        attn.indexer.k_layernorm,
    )
    if any(
        not isinstance(norm, GemmaRMSNorm) or norm.variance_epsilon != 1e-6
        for norm in norms
    ):
        raise MonoUnsupported("Quark QSA requires Gemma norm and epsilon 1e-6")
    if not attn.qk_norm.add_unit_offset:
        raise MonoUnsupported("Quark QSA requires Gemma norm weights")
    if any(
        p.bias is not None
        for p in (attn.qkv_proj, attn.indexer.index_qk_proj, attn.o_proj)
    ):
        raise MonoUnsupported("Quark QSA projection fusion requires no bias")


def bind_qsa_cache(attn, dims: Dims):
    from aiter import dtypes

    k = getattr(attn, "k_cache", None)
    if not isinstance(k, torch.Tensor) or k.ndim != 4:
        raise MonoUnsupported("Quark QSA requires a paged NHD K/V cache")
    pages, block = k.shape[:2]
    if pages <= 0 or block <= 0 or block % QSA_COMPRESS_RATIO:
        raise MonoUnsupported(
            "Quark QSA cache pages must contain complete index groups"
        )
    if k.dtype not in (torch.bfloat16, dtypes.fp8):
        raise MonoUnsupported(f"Quark QSA unsupported KV dtype {k.dtype}")
    shape = (pages, block, dims.qsa_kv_heads, QSA_HEAD_DIM)
    k = _tensor(attn, "k_cache", shape, k.dtype)
    v = _tensor(attn, "v_cache", shape, k.dtype)
    if k.dtype == dtypes.fp8:
        ks = _tensor(attn, "k_scale_cache", shape[:3], torch.float32)
        vs = _tensor(attn, "v_scale_cache", shape[:3], torch.float32)
    else:
        ks, vs = attn.k_scale_cache, attn.v_scale_cache
        if ks is not None or vs is not None:
            raise MonoUnsupported("Quark BF16 KV must not have FP8 scales")
    if attn.compressed_key_scale is not None:
        raise MonoUnsupported("Quark QSA index fusion requires BF16 index")
    return QuarkQsaCache(
        k,
        v,
        ks,
        vs,
        _tensor(
            attn, "raw_key_cache", (pages, block, 1, QSA_INDEX_HEAD_DIM), torch.bfloat16
        ),
        _tensor(
            attn,
            "compressed_key_cache",
            (pages, block // QSA_COMPRESS_RATIO, 1, QSA_INDEX_HEAD_DIM),
            torch.bfloat16,
        ),
        _tensor(attn, "rope_position_cache", (pages, block, 1, 3), torch.int64),
    )


def bind_layer(
    layer, task: LayerTask, dims: Dims, device, max_model_len, *, native_attention=False
):
    from aiter import QuantType, dtypes

    # MTP retains Native projections and ReplaySSM. Only QSA norm/RoPE and
    # cache tensors are consumed by Mono, so do not bind PTPC as BF16 weights.
    qsa = None if native_attention else _bind_qsa(layer, task, dims)
    cache = None
    if not task.is_gdn:
        _check_qsa_math(layer.self_attn, max_model_len)
        cache = bind_qsa_cache(layer.self_attn, dims)
        if native_attention:
            attn = layer.self_attn
            cos, sin = _bind_rope(attn.rotary_emb, "QSA RoPE")
            qsa = QuarkQsaEpilogue(
                _tensor(attn, "q_norm.weight", (QSA_HEAD_DIM,), torch.bfloat16),
                _tensor(attn, "k_norm.weight", (QSA_HEAD_DIM,), torch.bfloat16),
                cos,
                sin,
                _tensor(
                    attn.indexer,
                    "q_layernorm.weight",
                    (QSA_INDEX_HEAD_DIM,),
                    torch.bfloat16,
                ),
                _tensor(
                    attn.indexer,
                    "k_layernorm.weight",
                    (QSA_INDEX_HEAD_DIM,),
                    torch.bfloat16,
                ),
                *_bind_rope(attn.indexer.rotary_emb, "QSA index RoPE"),
            )
    elif native_attention:
        _check_mtp_gdn(layer.linear_attn, dims)
    experts = layer.mlp.experts
    if (
        experts.quant_method.quant_dtype != dtypes.fp4x2
        or experts.quant_method.quant_type != QuantType.per_1x32
    ):
        raise MonoUnsupported("Quark experts require MXFP4 with 1x32 E8M0 scales")
    intermediate = experts.w13_weight.shape[1] // 2
    count = dims.local_experts + 1
    # gfx950 stock AITER's GGUU shuffle keeps a 2D E8M0 plane, padded to
    # 256 rows and eight scale columns (one scale covers 32 weight values).
    scale_rows = lambda n: (count * n + 255) // 256 * 256
    scale_cols = lambda k: (k // 32 + 7) // 8 * 8
    bound = QuarkLayerWeights(
        bind_hc(layer.attn_hyper_connection),
        bind_hc(layer.mlp_hyper_connection),
        _tensor(layer.mlp, "gate.weight", (ROUTER_OUTPUTS, HIDDEN), torch.bfloat16),
        None if native_attention else _bind_gdn(layer, task, dims),
        qsa,
        cache,
        _tensor(
            experts,
            "w13_weight",
            (count, 2 * intermediate, HIDDEN // 2),
            dtypes.fp4x2,
            shuffled=True,
        ),
        _tensor(
            experts,
            "w2_weight",
            (count, HIDDEN, intermediate // 2),
            dtypes.fp4x2,
            shuffled=True,
        ),
        _bind_scale(
            experts,
            "w13_weight_scale",
            (scale_rows(2 * intermediate), scale_cols(HIDDEN)),
        ),
        _bind_scale(
            experts,
            "w2_weight_scale",
            (scale_rows(HIDDEN), scale_cols(intermediate)),
        ),
    )
    _check_device(bound, device)
    return bound


def _check_mtp_gdn(attn, dims):
    """Admit the q4 prepare ABI before the first HC/cache/state mutation."""
    from atom.utils.forward_context import get_forward_context

    impl = attn.attn.impl
    if (
        dims.tp != 1
        or (
            impl.key_dim,
            impl.value_dim,
            impl.num_v_heads,
            impl.head_k_dim,
            impl.head_v_dim,
        )
        != (dims.gdn_key_dim, dims.gdn_value_dim, dims.gdn_value_heads, 128, 128)
        or impl.activation not in (True, "silu", "swish")
        or impl.conv1d.bias is not None
    ):
        raise MonoUnsupported(
            "MTP GDN preparation needs TP1, sigmoid gates and width4 SiLU convolution"
        )
    _tensor(impl, "conv1d.weight", (dims.gdn_conv_dim, 1, 4), torch.bfloat16)
    _tensor(impl, "A_log", (dims.gdn_value_heads,), torch.bfloat16)
    _tensor(impl, "dt_bias", (dims.gdn_value_heads,), torch.bfloat16)
    cache = get_forward_context().kv_cache_data[f"layer_{impl.layer_num}"]
    if cache.k_cache.dtype != torch.bfloat16 or cache.k_cache.ndim != 3:
        raise MonoUnsupported("MTP GDN preparation requires BF16 convolution history")
    history = cache.k_cache
    if history.shape[1] != dims.gdn_conv_dim:
        history = history.transpose(-1, -2)
    if history.shape[1] != dims.gdn_conv_dim or history.shape[2] < 6:
        raise MonoUnsupported("MTP q4 convolution history must hold six tokens")


def _bind_scale(experts, name, shape):
    from aiter import dtypes

    # Native Mxfp4MoEMethod allocates E8M0 bytes as uint8 and its scale shuffle
    # preserves that dtype. Typed E8M0 has the same encoding; other byte-sized
    # formats do not. The caller first validates the MXFP4 / per_1x32 contract.
    dtype = getattr(getattr(experts, name, None), "dtype", None)
    if dtype not in (torch.uint8, dtypes.fp8_e8m0):
        raise MonoUnsupported(f"{name} requires E8M0 storage, got {dtype}")
    return _tensor(experts, name, shape, dtype)

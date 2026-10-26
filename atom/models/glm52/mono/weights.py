# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The layer weight mapping: a loaded GLM-5.2 layer's tensors, as the kernel's
``LayerWeights`` ABI.

The ``tensors`` dict keys and the ``LayerWeights`` constructor argument order
are read positionally by the kernel — move or rename either and the failure
is silent wrong numerics on a GPU, not a compile error.
"""

from __future__ import annotations

import torch
from aiter import dtypes

from atom.models.glm52.mono.kernels.geometry_config import (
    EPS,
    SOFTMAX_SCALE,
    Mxfp4ScaleLayout,
    Mxfp4WeightLayout,
    glm5_tp_config,
)
from atom.models.glm52.mono.kernels.layout import (
    INDEX_DIM,
    INDEX_HEADS,
    INDEX_Q_ROWS,
)
from atom.mono.runtime.consensus import MonoUnsupported
from atom.mono.weights.layer import (
    LayerWeights,
    atom_mxfp4_storage_view,
    linear_bf16,
    linear_ptpc_fp8,
)

# ``indexer.wk_weights_proj`` packs two row blocks into one BF16 linear: the
# first ``INDEX_DIM`` rows are the index-K projection (`w_index_k`), the next
# ``INDEX_HEADS`` rows are the per-head index weight (`w_index_w`). Neither
# the kernel nor the checkpoint names this split; it is read out positionally
# here, matching ATOM's own (non-fused) ``Indexer`` module.
_WK_WEIGHTS_PROJ_ROWS = INDEX_DIM + INDEX_HEADS


def need(ok: bool, what: str) -> None:
    if not ok:
        raise MonoUnsupported(what)


def bf16_vector(tensor: torch.Tensor, name: str, size: int) -> torch.Tensor:
    need(tensor.dtype is torch.bfloat16, f"{name} must be BF16, got {tensor.dtype}")
    need(tensor.shape == (size,) and tensor.is_contiguous(), f"{name} layout")
    return tensor


def fp32_vector(tensor: torch.Tensor, name: str, size: int) -> torch.Tensor:
    need(tensor.dtype is torch.float32, f"{name} must be FP32, got {tensor.dtype}")
    need(tensor.shape == (size,) and tensor.is_contiguous(), f"{name} layout")
    return tensor


def _block128_fp8(
    matrix: torch.Tensor, *, name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a row-major BF16/float matrix into FP8 with one absmax-based
    scale per (128 row, 128 col) block.

    This is the format the fused TP4 indexer's ``unit_fp8x2`` GEMM unit reads
    for ``w_index_k``/``w_index_q`` (``atom/models/glm52/mono/kernels/
    glm/kernel.py``): ``scale_rows=SCALE_BM=128`` (``atom/model_ops/
    monokernel/config.py``), one fp32 scale per 128x128 block, flattened
    row-major as ``scale[row_block, col_block]`` -- NOT a single tensor-wide
    scalar (``dynamic_per_batched_tensor_quant``) and NOT the checkpoint's
    native per-row scale, both of which are the wrong granularity here. The
    returned FP8 tensor is plain row-major (the caller, ``tp4_op.py``'s
    ``Glm5TP4MonoKernel.__init__``, applies ``pack_fp8`` itself); the scale is
    returned exactly as the kernel reads it, with no further repacking.

    Direction matches ``dynamic_per_batched_tensor_quant``: the reader
    multiplies by the stored scale, so ``scale = amax / fp8_max`` and
    dequant is ``fp8.float() * scale``.
    """

    need(
        matrix.ndim == 2 and matrix.is_contiguous(),
        f"{name} must be a contiguous 2-D matrix, got {tuple(matrix.shape)}",
    )
    rows, cols = matrix.shape
    need(
        rows % 128 == 0 and cols % 128 == 0,
        f"{name} must align to 128x128 scale blocks, got {(rows, cols)}",
    )
    fp8_max = torch.finfo(dtypes.fp8).max
    blocked = matrix.to(torch.float32).view(rows // 128, 128, cols // 128, 128)
    amax = blocked.abs().amax(dim=(1, 3)).clamp_min(1e-4)
    scale = (amax / fp8_max).contiguous()
    quant = (blocked / scale.view(rows // 128, 1, cols // 128, 1)).clamp(
        -fp8_max, fp8_max
    )
    fp8 = quant.to(dtypes.fp8).reshape(rows, cols).contiguous()
    return fp8, scale


def indexer_weights(layer, cfg) -> dict[str, torch.Tensor]:
    """The seven fused-indexer tensors for one "full" layer.

    Only valid for layers where ``layer.self_attn.indexer is not None`` --
    "shared" layers carry no indexer module at all and must never reach here
    (see ``runner.py``'s full/shared split, which reuses exactly the
    predicate ``_refresh_indexer`` already used).
    """

    attn = layer.self_attn
    indexer = attn.indexer
    need(indexer is not None, "fused indexer requires layer.self_attn.indexer")
    need(
        indexer.n_head == INDEX_HEADS,
        f"indexer.n_head {indexer.n_head} != build constant INDEX_HEADS {INDEX_HEADS}",
    )
    need(
        indexer.head_dim == INDEX_DIM,
        f"indexer.head_dim {indexer.head_dim} != build constant INDEX_DIM {INDEX_DIM}",
    )

    wk_proj = indexer.wk_weights_proj
    wk_bf16 = linear_bf16(
        wk_proj,
        name="indexer.wk_weights_proj[:INDEX_DIM]",
        logical_rows=_WK_WEIGHTS_PROJ_ROWS,
        logical_cols=cfg.hidden,
        row_start=0,
        row_count=INDEX_DIM,
    )
    w_index_w = linear_bf16(
        wk_proj,
        name="indexer.wk_weights_proj[INDEX_DIM:]",
        logical_rows=_WK_WEIGHTS_PROJ_ROWS,
        logical_cols=cfg.hidden,
        row_start=INDEX_DIM,
        row_count=INDEX_HEADS,
    )
    wq_bf16 = linear_bf16(
        indexer.wq_b,
        name="indexer.wq_b",
        logical_rows=INDEX_Q_ROWS,
        logical_cols=cfg.q_lora,
    )
    w_index_k, s_index_k = _block128_fp8(wk_bf16, name="w_index_k")
    w_index_q, s_index_q = _block128_fp8(wq_bf16, name="w_index_q")

    g_index_k = fp32_vector(indexer.k_norm.weight, "indexer.k_norm.weight", INDEX_DIM)
    b_index_k = fp32_vector(indexer.k_norm.bias, "indexer.k_norm.bias", INDEX_DIM)

    return {
        "w_index_k": w_index_k,
        "s_index_k": s_index_k,
        "w_index_w": w_index_w,
        "w_index_q": w_index_q,
        "s_index_q": s_index_q,
        "g_index_k": g_index_k,
        "b_index_k": b_index_k,
    }


def _batched_fp8(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_dtypes = {
        dtype
        for dtype in (
            getattr(torch, "float8_e4m3fn", None),
            getattr(torch, "float8_e4m3fnuz", None),
        )
        if dtype is not None
    }
    need(
        weight.dtype in fp8_dtypes
        and tuple(weight.shape) == shape
        and weight.is_contiguous(),
        f"{name} weight {weight.dtype} {tuple(weight.shape)}",
    )
    need(
        scale.dtype == torch.float32 and scale.numel() == 1 and scale.is_contiguous(),
        f"{name} tensor scale",
    )
    return weight.view(shape[0] * shape[1], shape[2]), scale.view(1)


def _atom_byte_view(tensor: torch.Tensor) -> torch.Tensor:
    view = tensor.view(torch.uint8)
    view.is_shuffled = bool(getattr(tensor, "is_shuffled", False))
    return view


def layer_weights(
    layer, rank: int, npes: int, *, with_indexer: bool = False
) -> LayerWeights:
    cfg = glm5_tp_config(npes)
    attn = layer.self_attn
    moe = layer.mlp
    experts = moe.experts
    physical_experts = cfg.n_experts + cfg.num_shared_experts

    # The kernel bakes `SOFTMAX_SCALE` in; it takes no scale from the caller.
    # Production derives the same value, but then squares a YaRN mscale into it
    # (`deepseek_v2.py`: `self.scaling = self.scaling * mscale * mscale`), so a
    # config with `factor=32, mscale_all_dim=1` serves 0.11333 while the kernel
    # would use 0.0625 -- different attention probabilities, silently. Refuse
    # rather than serve it: propagating the scale into the build is a real
    # feature and would need GPU numerics to confirm, while this costs nothing
    # on a config that does not adjust the scale.
    need(
        attn.scaling == SOFTMAX_SCALE,
        f"attention scale {attn.scaling} != the kernel's {SOFTMAX_SCALE} "
        "(YaRN mscale?)",
    )
    need(experts.global_num_experts == cfg.n_experts, "routed expert count")
    need(
        experts.local_num_experts == physical_experts,
        "shared expert is not fused into ATOM expert storage",
    )
    need(
        experts.num_fused_shared_experts == cfg.num_shared_experts,
        "fused shared-expert count",
    )
    need(not experts.use_ep, "expert parallelism")
    # ``is_guinterleave`` exists only on ``Mxfp4MoEMethod``. A GLM-5/GLM-5.2
    # FP8 checkpoint has the same ``glm_moe_dsa`` model type and reaches here
    # with ``Fp8MoEMethod``, where a bare attribute read raises AttributeError
    # -- which ``_build`` does not catch, so the server dies instead of
    # falling back. Everything below reads MXFP4 storage, so refuse first.
    gu_interleave = getattr(experts.quant_method, "is_guinterleave", None)
    need(gu_interleave is not None, "experts are not MXFP4")
    need(not gu_interleave, "ATOM_MOE_GU_ITLV must be 0")
    need(experts.intermediate_size_per_partition == cfg.inter, "expert width")
    for name, norm, size in (
        ("input_layernorm", layer.input_layernorm, cfg.hidden),
        ("q_a_layernorm", attn.q_a_layernorm, cfg.q_lora),
        ("kv_a_layernorm", attn.kv_a_layernorm, cfg.kv_lora),
        ("post_attention_layernorm", layer.post_attention_layernorm, cfg.hidden),
    ):
        need(norm.eps == EPS, f"{name} epsilon {norm.eps} != {EPS}")
        bf16_vector(norm.weight, f"{name}.weight", size)

    w_ug = _atom_byte_view(experts.w13_weight)
    s_ug = experts.w13_weight_scale.view(torch.uint8)
    w_dn = _atom_byte_view(experts.w2_weight)
    s_dn = experts.w2_weight_scale.view(torch.uint8)
    atom_mxfp4_storage_view(
        w_ug,
        name="w_ug",
        logical_rows=physical_experts * 2 * cfg.inter,
        logical_k=cfg.hidden,
        scale=False,
    )
    atom_mxfp4_storage_view(
        s_ug,
        name="s_ug",
        logical_rows=physical_experts * 2 * cfg.inter,
        logical_k=cfg.hidden,
        scale=True,
    )
    atom_mxfp4_storage_view(
        w_dn,
        name="w_dn",
        logical_rows=physical_experts * cfg.hidden,
        logical_k=cfg.inter,
        scale=False,
    )
    atom_mxfp4_storage_view(
        s_dn,
        name="s_dn",
        logical_rows=physical_experts * cfg.hidden,
        logical_k=cfg.inter,
        scale=True,
    )

    impl = attention_impl(layer)
    w_uk, s_uk = _batched_fp8(
        impl.W_K,
        impl.W_K_scale,
        name="W_K",
        shape=(cfg.local_heads, cfg.kv_lora, cfg.nope_dim),
    )
    w_uv, s_uv = _batched_fp8(
        impl.W_V,
        impl.W_V_scale,
        name="W_V",
        shape=(cfg.local_heads, cfg.v_dim, cfg.kv_lora),
    )
    w_qkv_a, s_qkv_a = linear_ptpc_fp8(
        attn.fused_qkv_a_proj,
        name="fused_qkv_a_proj",
        logical_rows=cfg.qkv_a_rows,
        logical_cols=cfg.hidden,
    )
    w_q_b, s_q_b = linear_ptpc_fp8(
        attn.q_b_proj,
        name="q_b_proj",
        logical_rows=cfg.local_heads * (cfg.nope_dim + cfg.pe_dim),
        logical_cols=cfg.q_lora,
    )
    w_o, s_o = linear_ptpc_fp8(
        attn.o_proj,
        name="o_proj",
        logical_rows=cfg.hidden,
        logical_cols=cfg.local_heads * cfg.v_dim,
    )
    bias = moe.gate.e_score_correction_bias
    need(
        bias is not None and bias.dtype is torch.float32,
        "router correction bias must be FP32",
    )
    need(
        bias.shape == (cfg.n_experts,) and bias.is_contiguous(),
        "router correction bias layout",
    )

    tensors = {
        "g_in": bf16_vector(
            layer.input_layernorm.weight, "input_layernorm.weight", cfg.hidden
        ),
        "g_q": bf16_vector(
            attn.q_a_layernorm.weight, "q_a_layernorm.weight", cfg.q_lora
        ),
        "g_kv": bf16_vector(
            attn.kv_a_layernorm.weight, "kv_a_layernorm.weight", cfg.kv_lora
        ),
        "g_post": bf16_vector(
            layer.post_attention_layernorm.weight,
            "post_attention_layernorm.weight",
            cfg.hidden,
        ),
        "w_qkv_a": w_qkv_a,
        "s_qkv_a": s_qkv_a,
        "w_q_b": w_q_b,
        "s_q_b": s_q_b,
        "w_uk": w_uk,
        "s_uk": s_uk,
        "w_uv": w_uv,
        "s_uv": s_uv,
        "w_o": w_o,
        "s_o": s_o,
        "w_r": linear_bf16(
            moe.gate,
            name="gate",
            logical_rows=cfg.n_experts,
            logical_cols=cfg.hidden,
        ),
        "bias": bias,
        "w_ug": w_ug,
        "s_ug": s_ug,
        "w_dn": w_dn,
        "s_dn": s_dn,
    }
    if with_indexer:
        tensors.update(indexer_weights(layer, cfg))
    return LayerWeights(
        cfg.local_heads,
        tensors,
        cfg,
        rank,
        npes,
        mxfp4_weight_layout=Mxfp4WeightLayout.ATOM,
        mxfp4_scale_layout=Mxfp4ScaleLayout.ATOM,
        physical_experts=physical_experts,
    )


def attention_impl(layer):
    wrapped = layer.self_attn.mla_attn
    return getattr(wrapped, "impl", wrapped)

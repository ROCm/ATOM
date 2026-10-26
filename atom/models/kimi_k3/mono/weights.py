# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The loaded K3 layer tensors the mono kernels read.

Almost nothing is copied: a kernel reads a parameter where the original model
keeps it, so a layout the kernels were not written for is refused here rather
than read wrong. Every tensor is checked against the shape and dtype the kernels
assume for the rank's shard.

Three groups are the exception, and each says why:

* four small vectors the kernels read as fp32 that the model holds in bf16 (the
  causal-conv weights, ``A_log``, ``dt_bias``, the router's correction bias) --
  together under 100 KB a layer;
* the latent MoE's down- and up-projections. The model quantizes both to MXFP4
  at load while the kernels' latent GEMVs are bf16, so they are dequantized once
  here (``layouts`` undoes AITER's two shuffles first). This is the port's whole
  weight budget: ~58 MB a mono layer, ~4 GB a rank over K3's KDA layers. A
  deployment that cannot spare it has to turn mono off, not run it degraded.
* the fused KDA in-projection, which must be unquantized. ATOM's online quant
  turns it into preshuffled per-token FP8 padded to 6400 rows; the kernels read
  bf16. It is refused rather than dequantized, because at 90 MB a layer that
  copy is larger than everything else here put together.

What is *not* copied is everything large: the expert planes (1.2 GB and 616 MB a
layer) and their E8M0 scales are read exactly where ``FusedMoE`` left them.
``moe_shuffle_scale`` on gfx950 is ``aiter.ops.shuffle.shuffle_scale``, whose
permutation is the one ``atom.mono.device.mx.scale_index`` indexes a byte of --
the same layout the MoE kernel's scale readers were written for. That holds only
for the non-interleaved gate/up layout, so ``is_guinterleave`` is checked.
"""

from dataclasses import dataclass

import torch

from atom.models.kimi_k3.mono.config import (
    ATTN_RES_BLOCK,
    EXPERTS,
    HEAD_DIM,
    HIDDEN,
    LATENT,
    TOPK,
    Dims,
    layer_block,
)
from atom.models.kimi_k3.mono.layouts import (
    SCALE_GROUP_ALIGN,
    SCALE_ROW_ALIGN,
    unshuffle_scale,
    unshuffle_weight,
)
from atom.mono.runtime.consensus import MonoUnsupported

BF16 = torch.bfloat16
F32 = torch.float32
# an E8M0 scale covers this many weight elements along K
MX_GROUP = 32


def _need(ok: bool, what: str) -> None:
    if not ok:
        raise MonoUnsupported(what)


def _eps(norm, name: str) -> float:
    """An RMSNorm's epsilon. ATOM's ``RMSNorm`` calls it ``eps`` while K3's own
    ``KimiRMSNormGated`` calls it ``variance_epsilon``, and the kernels bake it
    into a build key -- so read it one way everywhere rather than per site."""
    for attr in ("variance_epsilon", "eps"):
        value = getattr(norm, attr, None)
        if value is not None:
            return float(value)
    raise MonoUnsupported(f"{name}: no RMSNorm epsilon")


def _param(module, name: str):
    """A module's weight Parameter -- not ``.data``: ``is_shuffled`` is a Python
    attribute on the Parameter, and ``.data`` hands back a fresh tensor object
    without it, so a preshuffled weight reads as plain and the GEMM it feeds
    produces noise that correlates with nothing."""
    weight = getattr(module, "weight", None)
    _need(weight is not None, f"{name}: no weight")
    return weight


def _plain_bf16(module, name: str, shape: tuple[int, ...]) -> torch.Tensor:
    """A linear's weight as the kernels read it: bf16, unshuffled, exactly
    ``shape``."""
    w = _param(module, name)
    _need(
        w.dtype == BF16,
        f"{name} is {w.dtype}: the kernels read bf16, so this projection must "
        "stay out of the online quant (it is unquantized in the checkpoint)",
    )
    _need(not getattr(w, "is_shuffled", False), f"{name} is preshuffled")
    _need(not getattr(module, "is_output_padded", False), f"{name} output padded")
    _need(tuple(w.shape) == shape, f"{name} is {tuple(w.shape)}, expected {shape}")
    _need(w.is_contiguous(), f"{name} not contiguous")
    return w.data


def _vec(param, n: int, name: str, dtype: torch.dtype = BF16) -> torch.Tensor:
    t = getattr(param, "data", param)
    _need(t.numel() == n, f"{name} has {t.numel()} elements, expected {n}")
    _need(t.is_contiguous(), f"{name} not contiguous")
    _need(t.dtype == dtype, f"{name} is {t.dtype}, expected {dtype}")
    return t


def _as_f32(param, n: int, name: str) -> torch.Tensor:
    """A small vector the kernels read as fp32, converted once (a few KB)."""
    t = getattr(param, "data", param)
    _need(t.numel() == n, f"{name} has {t.numel()} elements, expected {n}")
    return t.reshape(-1).float().contiguous()


def _mxfp4_to_bf16(module, name: str, shape: tuple[int, ...]) -> torch.Tensor:
    """A preshuffled MXFP4 linear as bf16 ``shape``: both AITER shuffles undone
    (``layouts``), then the E2M1 values scaled by their E8M0 group scale."""
    from atom.model_ops.quant_v4 import dequant_fp4_e2m1

    w = _param(module, name)
    scale = getattr(module, "weight_scale", None)
    rows, cols = shape
    _need(scale is not None, f"{name}: no weight_scale")
    _need(w.element_size() == 1, f"{name} is {w.dtype}, expected packed FP4")
    _need(
        tuple(w.shape) == (rows, cols // 2),
        f"{name} is {tuple(w.shape)}, expected {(rows, cols // 2)} packed",
    )
    groups = cols // MX_GROUP
    packed = w.data.view(torch.uint8)
    # Both AITER layouts are applied under one condition and must be undone
    # under the same one. `LinearBase.process_weights_after_loading` shuffles
    # the weight when `weight_is_stored_preshuffled` holds -- for per_1x32
    # fp4x2 that is `not use_fp4_non_shuffle_triton_gemm()` -- and e8m0-shuffles
    # the scale on that same predicate (`linear.py`). So the weight's
    # `is_shuffled` is an exact proxy for the scale's state, and under
    # ATOM_USE_FP4_NON_SHUFFLE_TRITON_GEMM=1 neither is permuted: inverse-
    # permuting the row-major scale plane there is silent, because the routed
    # expert planes are tile-aligned and no shape check fires.
    if getattr(w, "is_shuffled", False):
        packed = unshuffle_weight(packed)
        plane = unshuffle_scale(scale.data, rows, groups)
    else:
        _need(
            scale.data.numel() == rows * groups,
            f"{name}: unshuffled scale has {scale.data.numel()} elements, "
            f"expected {rows * groups}",
        )
        plane = scale.data.reshape(rows, groups)
    return dequant_fp4_e2m1(
        packed.view(torch.int8), plane.view(torch.float8_e8m0fnu), MX_GROUP, BF16
    )


@dataclass(frozen=True)
class AttnRes:
    """One attention-residual mixing site, as K1 and K2 fold it: the mix's norm,
    its per-channel projection, and the out_norm the mix is normed by."""

    nw: torch.Tensor
    qk: torch.Tensor
    out_nw: torch.Tensor
    eps: float
    out_eps: float


@dataclass(frozen=True)
class KdaLayer:
    """What K1 reads for one KDA layer."""

    layer_id: int
    blocks: int  # AttnRes block sources before this layer
    write_idx: int  # the block this layer writes, or -1
    seam: AttnRes
    w_in: torch.Tensor
    w_fb: torch.Tensor
    conv_w: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    on_w: torch.Tensor
    onorm_eps: float
    lower_bound: float


@dataclass(frozen=True)
class MoeLayer:
    """What K2 reads for one layer: o_proj, then the latent MoE."""

    layer_id: int
    blocks: int
    write_idx: int
    seam: AttnRes
    w_o: torch.Tensor
    w_gate: torch.Tensor
    bias: torch.Tensor
    w_ld: torch.Tensor
    ln_w: torch.Tensor
    w_up: torch.Tensor
    w_sgu: torch.Tensor
    w_sd: torch.Tensor
    w13: torch.Tensor
    w13s: torch.Tensor
    w2: torch.Tensor
    w2s: torch.Tensor
    ln_eps: float


def _seam(site, name: str) -> AttnRes:
    """An ``AttnRes`` module's three tensors. The kernels multiply the norm and
    the projection per element in fp32 rather than reading the module's folded
    ``score_weight``, so the two are read apart."""
    _need(site.proj is not None and site.norm is not None, f"{name}: residuals off")
    out_norm = site.out_norm
    return AttnRes(
        nw=_vec(site.norm.weight, HIDDEN, f"{name}.norm"),
        qk=_vec(site.proj.weight, HIDDEN, f"{name}.proj").reshape(HIDDEN),
        out_nw=_vec(out_norm.weight, HIDDEN, f"{name}.out_norm"),
        eps=_eps(site.norm, f"{name}.norm"),
        out_eps=_eps(out_norm, f"{name}.out_norm"),
    )


def bind_kda(layer, layer_id: int, dims: Dims) -> KdaLayer:
    """One KDA layer's K1 tensors."""
    attn = layer.self_attn
    name = f"layer {layer_id}"
    _need(layer.is_linear_attn, f"{name}: not a KDA layer")
    _need(
        getattr(attn, "_in_proj_fused", False),
        f"{name}: in_proj is not fused yet (process_weights_after_loading)",
    )
    _need(
        (attn.num_local_heads, attn.head_dim) == (dims.heads, HEAD_DIM),
        f"{name}: {attn.num_local_heads} heads of {attn.head_dim}",
    )
    _need(attn.conv_kernel_size == 4, f"{name}: conv width {attn.conv_kernel_size}")
    lower = attn._kda_gate_lower_bound
    _need(lower is not None, f"{name}: no KDA gate lower bound")
    blocks, write_idx = layer_block(layer_id)
    return KdaLayer(
        layer_id=layer_id,
        blocks=blocks,
        write_idx=write_idx,
        seam=_seam(layer.self_attention_attn_res, f"{name} attn seam"),
        # [q | k | v | g | beta | f_a]; the kernels' offsets follow this order
        w_in=_plain_bf16(attn.in_proj, f"{name} in_proj", (dims.fused, HIDDEN)),
        w_fb=_plain_bf16(attn.f_b_proj, f"{name} f_b_proj", (dims.proj, HEAD_DIM)),
        conv_w=_as_f32(attn.conv_weight, dims.qkv * 4, f"{name} conv_weight").view(
            dims.qkv, 4
        ),
        a_log=_as_f32(attn.A_log, dims.heads, f"{name} A_log"),
        dt_bias=_as_f32(attn.dt_bias, dims.proj, f"{name} dt_bias"),
        on_w=_vec(attn.o_norm.weight, HEAD_DIM, f"{name} o_norm"),
        onorm_eps=_eps(attn.o_norm, f"{name} o_norm"),
        lower_bound=float(lower),
    )


def _experts(moe, layer_id: int, dims: Dims):
    """``FusedMoE``'s four MXFP4 planes, read where it left them."""
    experts = moe.experts
    method = experts.quant_method
    name = f"layer {layer_id} experts"
    _need(experts.top_k == TOPK, f"{name}: top-{experts.top_k}")
    _need(
        not getattr(method, "is_guinterleave", True),
        f"{name}: gate/up interleaved (ATOM_MOE_GU_ITLV); the scale readers "
        "follow shuffle_scale's plain layout",
    )
    _need(
        not getattr(method, "fp8_activations", False),
        f"{name}: FP8 activations; the kernels feed the experts bf16",
    )
    w13, w2 = experts.w13_weight, experts.w2_weight
    _need(
        getattr(w13, "is_shuffled", False) and getattr(w2, "is_shuffled", False),
        f"{name}: expert planes are not preshuffled",
    )
    # Which weight prep ran, by the discriminator ATOM itself uses (moe.py's
    # `w13_swizzle_layout is not None` test): `create_weights` sets the
    # attribute to None on every layer, so its mere presence says nothing --
    # only the Triton branch gives it a value, and only MegaMoE builds
    # `_mega_w1` (and then releases the raw planes this reads).
    _need(
        getattr(experts, "w13_swizzle_layout", None) is None,
        f"{name}: the Triton MXFP4 weight prep ran, not the lane-shuffled one",
    )
    _need(
        getattr(experts, "_mega_w1", None) is None,
        f"{name}: MegaMoE owns the experts and has released the raw planes",
    )
    inter = dims.inter
    want = {
        "w13": (EXPERTS * 2 * inter * LATENT // 2, w13),
        "w2": (EXPERTS * LATENT * inter // 2, w2),
    }
    for plane, (nbytes, t) in want.items():
        got = t.numel() * t.element_size()
        _need(got == nbytes, f"{name}: {plane} is {got} B, expected {nbytes}")
    s13, s2 = experts.w13_weight_scale, experts.w2_weight_scale
    for plane, t, rows, groups in (
        ("w13_scale", s13, EXPERTS * 2 * inter, LATENT // MX_GROUP),
        ("w2_scale", s2, EXPERTS * LATENT, inter // MX_GROUP),
    ):
        pad_rows = -(-rows // SCALE_ROW_ALIGN) * SCALE_ROW_ALIGN
        pad_groups = -(-groups // SCALE_GROUP_ALIGN) * SCALE_GROUP_ALIGN
        got = t.numel() * t.element_size()
        _need(
            got == pad_rows * pad_groups,
            f"{name}: {plane} is {got} B, expected {pad_rows} x {pad_groups}",
        )
    return w13.data, s13.data, w2.data, s2.data


def bind_moe(layer, layer_id: int, dims: Dims, rank: int) -> MoeLayer:
    """One layer's K2 tensors: o_proj and the latent MoE."""
    name = f"layer {layer_id}"
    moe = getattr(layer, "block_sparse_moe", None)
    _need(moe is not None, f"{name}: not a MoE layer")
    _need(moe.use_latent_moe, f"{name}: not a latent MoE")
    _need(moe.shared_experts is not None, f"{name}: no shared experts")
    _need(moe.routed_expert_norm is not None, f"{name}: no routed norm")
    # No dual-stream refusal. `_use_dual_stream` is a static config flag that
    # only selects a branch inside `KimiSparseMoeBlock.forward`, and a mono
    # layer never calls that module at all -- K2 computes the latent MoE from
    # the tensors bound here. The alt stream cannot contend with the kernel
    # either: `dual_stream_moe_forward` joins before it returns
    # (`current_stream.wait_stream(alt_stream)`, deepseek_v2.py), so a
    # production layer's shared-expert work is complete before the next layer
    # launches anything. Refusing on it disarmed mono in the DEFAULT
    # environment -- K3 has shared experts and an alt stream, and
    # ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD defaults to 1024 -- which cost three
    # GSM8K runs that scored the production path while reporting "mono".
    gate = _param(moe.gate, f"{name} gate")
    _need(
        gate.dtype == BF16 and tuple(gate.shape) == (EXPERTS, HIDDEN),
        f"{name}: gate is {gate.dtype} {tuple(gate.shape)}",
    )
    w13, s13, w2, s2 = _experts(moe, layer_id, dims)
    # the up-projection is replicated in the model; the kernel's stage computes
    # this rank's row band of it and the ranks' bands are summed into the
    # layer's output, so only the band is bound.
    #
    # `.clone()`, not `.contiguous()`: a row `narrow` of a row-major tensor is
    # already contiguous, so `contiguous()` hands back the SAME view and the
    # whole dequantized 7168x3584 plane stays referenced by it -- 49 MiB held
    # per layer per rank to keep the 6.1 MiB band, which is memory the lazy
    # bind does not have and so turns mono off. `_mxfp4_to_bf16` returns a
    # fresh contiguous tensor, so cloning the band copies once and frees it.
    w_up = _mxfp4_to_bf16(
        moe.routed_expert_up_proj, f"{name} routed_expert_up_proj", (HIDDEN, LATENT)
    )
    return MoeLayer(
        layer_id=layer_id,
        blocks=layer_block(layer_id)[0],
        write_idx=layer_block(layer_id)[1],
        seam=_seam(layer.mlp_attn_res, f"{name} mlp seam"),
        w_o=_plain_bf16(layer.self_attn.o_proj, f"{name} o_proj", (HIDDEN, dims.proj)),
        w_gate=gate.data,
        bias=_as_f32(moe.gate.e_score_correction_bias, EXPERTS, f"{name} gate bias"),
        w_ld=_mxfp4_to_bf16(
            moe.routed_expert_down_proj,
            f"{name} routed_expert_down_proj",
            (LATENT, HIDDEN),
        ),
        ln_w=_vec(moe.routed_expert_norm.weight, LATENT, f"{name} routed_expert_norm"),
        w_up=w_up.narrow(0, rank * dims.up_n, dims.up_n).clone(),
        w_sgu=_plain_bf16(
            moe.shared_experts.gate_up_proj,
            f"{name} shared gate_up_proj",
            (2 * dims.sh_inter, HIDDEN),
        ),
        w_sd=_plain_bf16(
            moe.shared_experts.down_proj,
            f"{name} shared down_proj",
            (HIDDEN, dims.sh_inter),
        ),
        w13=w13,
        w13s=s13,
        w2=w2,
        w2s=s2,
        ln_eps=_eps(moe.routed_expert_norm, f"{name} routed_expert_norm"),
    )


@dataclass(frozen=True)
class LayerPlan:
    """What a decoder layer contributes to a mono step: its K1 and K2, or
    neither (the layer falls back to the original modules)."""

    layer_id: int
    kda: KdaLayer | None
    moe: MoeLayer | None

    @property
    def mono(self) -> bool:
        return self.kda is not None and self.moe is not None


def bind_model(model, dims: Dims, rank: int) -> list[LayerPlan]:
    """Every decoder layer, in order.

    A layer takes the kernels only when it is a KDA layer with a MoE FFN and at
    least one AttnRes block behind it: layer 0 has a dense MLP and no block, and
    the full-attention layers keep the original MLA path (and their paged KV
    cache) because the kernels have no attention of their own.
    """
    layers = model.layers
    plans = []
    for layer_id, layer in enumerate(layers):
        blocks, _ = layer_block(layer_id)
        takes = (
            bool(getattr(layer, "is_linear_attn", False))
            and hasattr(layer, "block_sparse_moe")
            and blocks >= 1
        )
        if not takes:
            plans.append(LayerPlan(layer_id, None, None))
            continue
        plans.append(
            LayerPlan(
                layer_id,
                bind_kda(layer, layer_id, dims),
                bind_moe(layer, layer_id, dims, rank),
            )
        )
    _need(
        any(p.mono for p in plans),
        f"no layer takes the kernels ({len(layers)} layers, block size "
        f"{ATTN_RES_BLOCK})",
    )
    return plans

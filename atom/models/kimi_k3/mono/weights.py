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
* the latent MoE's down- and up-projections. An MXFP4 checkpoint quantizes both
  at load while the kernels' latent GEMVs are bf16, so they are dequantized once
  here (``layouts`` undoes AITER's two shuffles first). This is the port's whole
  weight budget: ~58 MB a mono layer, ~4 GB a rank over K3's KDA layers. A
  deployment that cannot spare it has to turn mono off, not run it degraded. A
  bf16 checkpoint has no scale: its down-projection is bound as is, and only the
  up-projection's rank band is copied (that is the whole cost there).
* the fused KDA in-projection, which must be unquantized. ATOM's online quant
  turns it into preshuffled per-token FP8 padded to 6400 rows; the kernels read
  bf16. It is refused rather than dequantized, because at 90 MB a layer that
  copy is larger than everything else here put together.

What is *not* copied is everything large: the expert planes (1.2 GB and 616 MB a
layer) and their E8M0 scales are read exactly where ``FusedMoE`` left them.
``moe_shuffle_scale`` on gfx950 is ``aiter.ops.shuffle.shuffle_scale``, whose
permutation is the one ``atom.mono.device.mx.scale_index`` indexes a byte of.

The gfx1250 K2 path requires AITER's native n32k4 scales and binds both
separated GGUU and production interleaved GUGU gate/up layouts zero-copy. The
FP4 payload is likewise never repacked here.
"""

import logging
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
from atom.models.kimi_k3.mono.expert_scales import (
    KimiExpertScaleHandles,
    bind_expert_scale,
)
from atom.models.kimi_k3.mono.kernels.stages.moe_expert_layout import (
    ExpertScaleLayout,
    ExpertScaleSpec,
)
from atom.models.kimi_k3.mono.layouts import (
    SCALE_GROUP_ALIGN,
    SCALE_ROW_ALIGN,
    unshuffle_scale,
    unshuffle_weight,
)
from atom.mono.plan.architecture import GFX1250, Architecture
from atom.mono.runtime.consensus import MonoUnsupported

logger = logging.getLogger("atom")

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


def _block_fp8(module, name: str, shape: tuple[int, int]):
    """ATOM's online per_1x128 FP8 linear as gfx1250's TDM GEMV reads it, in
    place: ``(weight, scale)``, the weight in AITER's 16x16 preshuffle with its
    rows zero-padded to 128, the scale its ``[rows / 128, cols / 128]`` E8M0
    block scales."""
    rows, cols = shape
    padded = -(-rows // 128) * 128
    w = _param(module, name)
    scale = getattr(module, "weight_scale", None)
    _need(w.dtype == torch.float8_e4m3fn, f"{name} is {w.dtype}, expected FP8")
    _need(
        scale is not None and scale.dtype == torch.float8_e8m0fnu,
        f"{name}: the kernels read E8M0 block scales "
        "(ATOM_FP8_BLOCKSCALE_USE_E8M0_SCALE=1)",
    )
    _need(
        getattr(w, "is_shuffled", False),
        f"{name}: the kernels read the preshuffled weight "
        "(ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE=1)",
    )
    _need(
        tuple(w.shape) == (padded, cols),
        f"{name} is {tuple(w.shape)}, expected {(padded, cols)}",
    )
    _need(
        tuple(scale.shape) == (padded // 128, cols // 128),
        f"{name} scale is {tuple(scale.shape)}, expected a 128x128 block scale",
    )
    _need(w.is_contiguous() and scale.is_contiguous(), f"{name} not contiguous")
    return w.data, scale.data


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


def _mxfp4_planes(module, name: str, shape: tuple[int, int]):
    """A preshuffled MXFP4 linear as row-major planes: ``(packed, scale)``,
    ``[rows, cols / 2]`` E2M1 pairs (low nibble first) and ``[rows, cols / 32]``
    E8M0 bytes, both AITER shuffles undone (``layouts``). None for a bf16
    checkpoint, which has no scale."""
    w = _param(module, name)
    scale = getattr(module, "weight_scale", None)
    if scale is None and w.dtype == BF16:
        return None
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
    return packed.contiguous(), plane.view(torch.uint8).contiguous()


def _mxfp4_to_bf16(module, name: str, shape: tuple[int, ...]) -> torch.Tensor:
    """A preshuffled MXFP4 linear as bf16 ``shape``: ``_mxfp4_planes``, then the
    E2M1 values scaled by their E8M0 group scale. A bf16 checkpoint is already
    what the kernels read."""
    from atom.model_ops.quant_v4 import dequant_fp4_e2m1

    planes = _mxfp4_planes(module, name, shape)
    if planes is None:
        return _plain_bf16(module, name, shape)
    packed, plane = planes
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
    w_in_scale: torch.Tensor | None  # E8M0 block scales of an FP8 ``w_in``
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
    gu_interleave: bool
    ln_eps: float


@dataclass(frozen=True)
class RoutedMoeLayer:
    """K2 tensors needed by the routed path; shared experts stay quantized."""

    layer_id: int
    blocks: int
    write_idx: int
    seam: AttnRes
    w_gate: torch.Tensor
    bias: torch.Tensor
    w_ld: torch.Tensor
    ln_w: torch.Tensor
    w_up: torch.Tensor
    w13: torch.Tensor
    w13s: torch.Tensor
    w2: torch.Tensor
    w2s: torch.Tensor
    gu_interleave: bool
    ln_eps: float


@dataclass(frozen=True)
class ExternalMoeLayer(RoutedMoeLayer):
    """gfx1250 K2 around external collectives; o_proj remains a model op."""

    w_sgu: torch.Tensor
    w_sd: torch.Tensor
    # The model's whole routed up-projection when it is plain BF16 (read in
    # place); None when only the dequantized rank band ``w_up`` exists.
    w_up_full: torch.Tensor | None = None
    # E8M0 block scales of an online per_1x128 FP8 ``w_sgu`` / ``w_sd``, or
    # None (BF16).
    w_sgu_scale: torch.Tensor | None = None
    w_sd_scale: torch.Tensor | None = None
    # The latent down-projection's MXFP4 planes (``_mxfp4_planes``) where the
    # split-K front reads them (ATOM_KIMI_MONO_SPLITK=front), else None.
    w_ld_fp4: torch.Tensor | None = None
    w_ld_scale: torch.Tensor | None = None


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


def bind_kda(layer, layer_id: int, dims: Dims, fp8_in_proj: bool = False) -> KdaLayer:
    """One KDA layer's K1 tensors. ``fp8_in_proj``: the K1 reads an online
    per_1x128 FP8 in_proj as it is; otherwise in_proj must stay BF16."""
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
    in_proj = f"{name} in_proj"
    w_in, w_in_scale = (
        _block_fp8(attn.in_proj, in_proj, (dims.fused, HIDDEN))
        if fp8_in_proj and _param(attn.in_proj, in_proj).dtype != BF16
        else (_plain_bf16(attn.in_proj, in_proj, (dims.fused, HIDDEN)), None)
    )
    return KdaLayer(
        layer_id=layer_id,
        blocks=blocks,
        write_idx=write_idx,
        seam=_seam(layer.self_attention_attn_res, f"{name} attn seam"),
        # [q | k | v | g | beta | f_a]; the kernels' offsets follow this order
        w_in=w_in,
        w_in_scale=w_in_scale,
        w_fb=_plain_bf16(attn.f_b_proj, f"{name} f_b_proj", (dims.proj, HEAD_DIM)),
        conv_w=_as_f32(attn.conv_weight, dims.qkv * 4, f"{name} conv_weight").view(
            dims.qkv, 4
        ),
        a_log=_as_f32(attn.A_log, dims.heads, f"{name} A_log"),
        dt_bias=_as_f32(attn.dt_bias, dims.proj, f"{name} dt_bias"),
        # the model holds o_norm in fp32 (`control_dtype`); bound as is, no copy
        on_w=_vec(attn.o_norm.weight, HEAD_DIM, f"{name} o_norm", F32),
        onorm_eps=_eps(attn.o_norm, f"{name} o_norm"),
        lower_bound=float(lower),
    )


def _experts(moe, layer_id: int, dims: Dims):
    """``FusedMoE``'s four MXFP4 planes, read where it left them."""
    experts = moe.experts
    method = experts.quant_method
    name = f"layer {layer_id} experts"
    _need(experts.top_k == TOPK, f"{name}: top-{experts.top_k}")
    gu_interleave = bool(getattr(method, "is_guinterleave", True))
    _need(
        not gu_interleave or getattr(method, "is_gfx1250", False),
        f"{name}: gate/up interleaving requires the gfx1250 K2 reader",
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
        if getattr(method, "is_gfx1250", False):
            expected = rows * groups
        else:
            pad_rows = -(-rows // SCALE_ROW_ALIGN) * SCALE_ROW_ALIGN
            pad_groups = -(-groups // SCALE_GROUP_ALIGN) * SCALE_GROUP_ALIGN
            expected = pad_rows * pad_groups
        got = t.numel() * t.element_size()
        _need(
            got == expected,
            f"{name}: {plane} is {got} B, expected {expected}",
        )
    return w13.data, s13.data, w2.data, s2.data


def bind_k2_expert_scales(
    moe,
    layer_id: int,
    dims: Dims,
    rank: int,
    architecture: Architecture,
) -> KimiExpertScaleHandles:
    """Bind the explicit gfx1250 K2 scale arguments.

    This is deliberately separate from ``_experts``: the production gfx950
    binder continues to return AITER's existing tensors byte-for-byte.
    The gfx1250 runner calls this function and passes the returned handles to
    its K2 declaration.
    """

    if architecture is not GFX1250:
        raise ValueError(
            "bind_k2_expert_scales is the architecture-explicit gfx1250 path"
        )
    experts = moe.experts
    method = experts.quant_method
    name = f"layer {layer_id} experts"
    _need(hasattr(method, "is_gfx1250"), f"{name}: no source scale-layout metadata")
    _need(
        method.is_gfx1250,
        f"{name}: gfx1250 mono requires native n32k4 expert scales",
    )
    source_layout = ExpertScaleLayout.GFX1250_N32K4

    w13_spec = ExpertScaleSpec(
        "w13_scale",
        EXPERTS,
        2 * dims.inter,
        LATENT,
        rank,
        dims.tp,
    )
    w2_spec = ExpertScaleSpec(
        "w2_scale",
        EXPERTS,
        LATENT,
        dims.inter,
        rank,
        dims.tp,
    )
    handles = KimiExpertScaleHandles(
        architecture,
        bind_expert_scale(
            experts.w13_weight_scale,
            spec=w13_spec,
            source_layout=source_layout,
            architecture=architecture,
        ),
        bind_expert_scale(
            experts.w2_weight_scale,
            spec=w2_spec,
            source_layout=source_layout,
            architecture=architecture,
        ),
    )
    handles.validate()
    return handles


def bind_routed_moe(
    layer,
    layer_id: int,
    dims: Dims,
    rank: int,
    architecture: Architecture | None = None,
) -> RoutedMoeLayer:
    """Bind the routed K2 path without touching quantized shared projections."""

    name = f"layer {layer_id}"
    moe = getattr(layer, "block_sparse_moe", None)
    _need(moe is not None, f"{name}: not a MoE layer")
    _need(moe.use_latent_moe, f"{name}: not a latent MoE")
    _need(moe.routed_expert_norm is not None, f"{name}: no routed norm")
    gate = _param(moe.gate, f"{name} gate")
    _need(
        gate.dtype == BF16 and tuple(gate.shape) == (EXPERTS, HIDDEN),
        f"{name}: gate is {gate.dtype} {tuple(gate.shape)}",
    )
    w13, s13, w2, s2 = _experts(moe, layer_id, dims)
    if architecture is GFX1250:
        scales = bind_k2_expert_scales(
            moe, layer_id, dims, rank, architecture
        ).kernel_arguments()
        s13, s2 = scales["w13s"], scales["w2s"]
    gu_interleave = bool(
        getattr(
            getattr(getattr(moe, "experts", None), "quant_method", None),
            "is_guinterleave",
            False,
        )
    )
    w_up = _mxfp4_to_bf16(
        moe.routed_expert_up_proj,
        f"{name} routed_expert_up_proj",
        (HIDDEN, LATENT),
    )
    return RoutedMoeLayer(
        layer_id=layer_id,
        blocks=layer_block(layer_id)[0],
        write_idx=layer_block(layer_id)[1],
        seam=_seam(layer.mlp_attn_res, f"{name} mlp seam"),
        w_gate=gate.data,
        bias=_as_f32(
            moe.gate.e_score_correction_bias,
            EXPERTS,
            f"{name} gate bias",
        ),
        w_ld=_mxfp4_to_bf16(
            moe.routed_expert_down_proj,
            f"{name} routed_expert_down_proj",
            (LATENT, HIDDEN),
        ),
        ln_w=_vec(
            moe.routed_expert_norm.weight,
            LATENT,
            f"{name} routed_expert_norm",
        ),
        w_up=w_up.narrow(0, rank * dims.up_n, dims.up_n).clone(),
        w13=w13,
        w13s=s13,
        w2=w2,
        w2s=s2,
        gu_interleave=gu_interleave,
        ln_eps=_eps(moe.routed_expert_norm, f"{name} routed_expert_norm"),
    )


def bind_external_moe(
    layer,
    layer_id: int,
    dims: Dims,
    rank: int,
    architecture: Architecture,
    fp8_shared: bool = False,
    fp4_latent: bool = False,
) -> ExternalMoeLayer:
    """Bind native K2 local stages while leaving o_proj on its quantized path.
    ``fp8_shared``: K2 reads an online per_1x128 FP8 shared gate/up and down
    as they are; otherwise they must stay BF16. ``fp4_latent``: an MXFP4
    latent down-projection's planes too (``ExternalMoeLayer.w_ld_fp4``)."""

    routed = bind_routed_moe(
        layer,
        layer_id,
        dims,
        rank,
        architecture,
    )
    name = f"layer {layer_id}"
    shared = layer.block_sparse_moe.shared_experts
    _need(shared is not None, f"{name}: no shared experts")
    up = _param(layer.block_sparse_moe.routed_expert_up_proj, f"{name} routed up")
    whole_up = (
        up.dtype == BF16
        and not getattr(up, "is_shuffled", False)
        and tuple(up.shape) == (HIDDEN, LATENT)
        and up.is_contiguous()
    )

    def linear(module, what, shape):
        if fp8_shared and _param(module, what).dtype != BF16:
            return _block_fp8(module, what, shape)
        return _plain_bf16(module, what, shape), None

    w_sgu, w_sgu_scale = linear(
        shared.gate_up_proj,
        f"{name} shared gate_up_proj",
        (2 * dims.sh_inter, HIDDEN),
    )
    w_sd, w_sd_scale = linear(
        shared.down_proj, f"{name} shared down_proj", (HIDDEN, dims.sh_inter)
    )
    planes = (
        _mxfp4_planes(
            layer.block_sparse_moe.routed_expert_down_proj,
            f"{name} routed_expert_down_proj",
            (LATENT, HIDDEN),
        )
        if fp4_latent
        else None
    )
    w_ld_fp4, w_ld_scale = planes if planes is not None else (None, None)
    return ExternalMoeLayer(
        **routed.__dict__,
        w_up_full=up.data if whole_up else None,
        w_sgu=w_sgu,
        w_sgu_scale=w_sgu_scale,
        w_sd=w_sd,
        w_sd_scale=w_sd_scale,
        w_ld_fp4=w_ld_fp4,
        w_ld_scale=w_ld_scale,
    )


@dataclass(frozen=True)
class MlaFrontLayer:
    """What the gfx1250 MLA front (``mla_front_gfx1250``) reads for one
    full-attention layer: fused_qkv_a_proj and this rank's g_proj rows in
    ``_block_fp8``'s layout, the q_a / kv_a norm weights. ``q_dtype``: the
    FP8 q_b_proj's activation quant writes (the front quantizes q for it), or
    None (it hands q_b_proj BF16 q, which quantizes it itself)."""

    layer_id: int
    w_a: torch.Tensor
    s_a: torch.Tensor
    w_g: torch.Tensor
    s_g: torch.Tensor
    q_norm: torch.Tensor
    kv_norm: torch.Tensor
    q_eps: float
    kv_eps: float
    q_dtype: torch.dtype | None

    @property
    def gate_rows(self) -> int:
        return self.w_g.size(0)

    @property
    def key(self) -> tuple[int, float, float]:
        """What its build bakes in."""
        return self.gate_rows, self.q_eps, self.kv_eps


def _q_b_activation_quant(q_b_proj) -> torch.dtype | None:
    """The FP8 dtype ``LinearBase.forward`` quantizes q_b_proj's input to with
    ``per_1x128_e8m0_quant`` in its transposed (preshuffle GEMM) layout, the
    one the front writes; None for any other activation path."""
    from aiter import QuantType, dtypes

    from atom.utils import envs

    takes = (
        getattr(q_b_proj, "native_a8_group_rows", None) is None
        and q_b_proj.quant_type.value == QuantType.per_1x128.value
        and q_b_proj.params_dtype == dtypes.fp8
        and bool(getattr(q_b_proj, "blockscale_e8m0_scale", False))
        and envs.ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE
    )
    return q_b_proj.params_dtype if takes else None


def bind_mla_front(layer, layer_id: int, dims: Dims) -> MlaFrontLayer:
    """One full-attention layer's MLA front tensors, read in place. Refused
    where the production path would compute anything else from the same
    input: a DSA layer (its indexer reads the projections), an input or q
    norm that folds a quant, projection widths or layouts other than the
    front's."""
    from atom.models.kimi_k3.mono.kernels.stages import mla_front_layout as lay

    attn = layer.self_attn
    name = f"layer {layer_id}"
    _need(not getattr(layer, "is_linear_attn", True), f"{name}: not full attention")
    _need(getattr(attn, "indexer", None) is None, f"{name}: a DSA indexer layer")
    _need(
        list(getattr(attn, "projection_split_sizes", ()))
        == [lay.Q_LORA, lay.KV_LORA, lay.ROPE],
        f"{name}: projection widths {getattr(attn, 'projection_split_sizes', None)}",
    )
    _need(
        not getattr(attn, "fuse_qknorm_quant", True),
        f"{name}: the q norm folds q_b_proj's quant",
    )
    _need(
        layer.self_attention_attn_res.out_quant_dtype is None,
        f"{name}: the input norm folds a quant; the front reads the BF16 row",
    )
    gate_rows = attn.num_heads * attn.v_head_dim // dims.tp
    for module, what in (
        (attn.fused_qkv_a_proj, "fused_qkv_a_proj"),
        (attn.g_proj, "g_proj"),
    ):
        _need(getattr(module, "bias", None) is None, f"{name} {what} has a bias")
    w_a, s_a = _block_fp8(
        attn.fused_qkv_a_proj, f"{name} fused_qkv_a_proj", (lay.QKV_A_ROWS, HIDDEN)
    )
    w_g, s_g = _block_fp8(attn.g_proj, f"{name} g_proj", (gate_rows, HIDDEN))
    _need(gate_rows % lay.GROUP_ROWS == 0, f"{name}: {gate_rows} gate rows")
    return MlaFrontLayer(
        layer_id=layer_id,
        w_a=w_a,
        s_a=s_a,
        w_g=w_g,
        s_g=s_g,
        q_norm=_vec(attn.q_a_layernorm.weight, lay.Q_LORA, f"{name} q_a_layernorm"),
        kv_norm=_vec(attn.kv_a_layernorm.weight, lay.KV_LORA, f"{name} kv_a_layernorm"),
        q_eps=_eps(attn.q_a_layernorm, f"{name} q_a_layernorm"),
        kv_eps=_eps(attn.kv_a_layernorm, f"{name} kv_a_layernorm"),
        q_dtype=_q_b_activation_quant(attn.q_b_proj),
    )


@dataclass(frozen=True)
class MlaAbsorbLayer:
    """What the gfx1250 MLA absorb (``mla_absorb_gfx1250``) reads for one
    full-attention layer, behind its front: q_b_proj in ``_block_fp8``'s
    layout, the absorb's FP8 ``W_UK`` and scale, and the decode branch's
    q_out shape -- ``heads`` projected (the DCP group's under QREP, else this
    rank's), ``out_heads`` (more when the zero head pad is folded into
    q_out), ``compute_all`` (DCP: every token's q_out; else a token without a
    KV slot keeps zeros). ``attention``: the layer's ``MLAAttention``, which
    runs the decode from the absorb's q_out."""

    layer_id: int
    w_qb: torch.Tensor
    s_qb: torch.Tensor
    w_uk: torch.Tensor
    s_uk: torch.Tensor
    heads: int
    out_heads: int
    compute_all: bool
    attention: object

    @property
    def key(self) -> tuple[int, int, bool]:
        """What its build bakes in."""
        return self.heads, self.out_heads, self.compute_all


def bind_mla_absorb(layer, layer_id: int, front: MlaFrontLayer) -> MlaAbsorbLayer:
    """One full-attention layer's MLA absorb tensors, read in place. Refused
    where the decode branch of ``MLAAttention.forward_impl`` would compute
    anything else: a front handing q_b_proj BF16 q, a sparse layer, the
    segmented or shuffled KV cache, a KV cache or q not FP8, static scales
    other than 1, a rotary embedding other than the NoPE identity, the BF16
    or FP4 absorb, other widths."""
    from atom.model_ops import attention_mla
    from atom.model_ops.rotary_embedding import NoPositionalRotaryEmbedding
    from atom.models.kimi_k3.mono.kernels.stages import mla_absorb_layout as lay

    name = f"layer {layer_id}"
    attn = layer.self_attn
    impl = getattr(getattr(attn, "attn", None), "impl", None)
    _need(front.q_dtype is not None, f"{name}: the front hands q_b_proj BF16 q")
    _need(
        callable(getattr(impl, "forward_decode_q_out", None)),
        f"{name}: not an MLAAttention",
    )
    _need(not impl.is_sparse_mla, f"{name}: sparse MLA")
    _need(not impl.use_seg_mla, f"{name}: the segmented KV cache")
    _need(not impl.kv_shuffled, f"{name}: the shuffled KV cache")
    _need(impl.kv_cache_dtype == "fp8", f"{name}: the KV cache is not FP8")
    _need(
        float(impl._k_scale) == 1.0 and float(impl._q_scale) == 1.0,
        f"{name}: static q / k scales other than 1",
    )
    rope = impl.rotary_emb
    _need(
        isinstance(rope, NoPositionalRotaryEmbedding)
        and rope.rotary_dim == lay.ROPE
        and rope.is_neox_style,
        f"{name}: a rotary embedding other than the NoPE identity",
    )
    _need(
        (impl.q_lora_rank, impl.kv_lora_rank, impl.qk_nope_head_dim)
        == (lay.Q_LORA, lay.KV_LORA, lay.NOPE)
        and impl.qk_rope_head_dim == lay.ROPE,
        f"{name}: MLA widths",
    )
    _need(not attention_mla.is_rocm_aiter_fp4bmm_enabled(), f"{name}: the FP4 absorb")
    if impl.qrep_enabled:
        w_uk, s_uk, heads = impl.W_K_qrep, impl.W_K_qrep_scale, impl.qrep_num_heads
    else:
        w_uk, s_uk, heads = impl.W_K, impl.W_K_scale, impl.num_heads
    _need(s_uk is not None, f"{name}: the BF16 absorb (ATOM_MLA_ABSORB_BF16)")
    _need(heads % lay.HEADS_PER_CLUSTER == 0, f"{name}: {heads} query heads")
    _need(
        w_uk.dtype == torch.float8_e4m3fn
        and tuple(w_uk.shape) == (heads, lay.KV_LORA, lay.NOPE)
        and w_uk.is_contiguous(),
        f"{name}: W_UK is {w_uk.dtype} {tuple(w_uk.shape)}",
    )
    _need(
        s_uk.dtype == torch.float32
        and s_uk.numel() == 1
        and s_uk.device == w_uk.device,
        f"{name}: W_UK's scale is not one f32 beside it",
    )
    w_qb, s_qb = _block_fp8(
        attn.q_b_proj, f"{name} q_b_proj", (heads * lay.HEAD, lay.Q_LORA)
    )
    if impl.dcp_world_size > 1:
        out_heads, compute_all = heads, True
    else:
        out_heads = impl.padded_num_heads if impl._fused_q_head_pad else heads
        compute_all = False
    return MlaAbsorbLayer(
        layer_id=layer_id,
        w_qb=w_qb,
        s_qb=s_qb,
        w_uk=w_uk,
        s_uk=s_uk,
        heads=heads,
        out_heads=out_heads,
        compute_all=compute_all,
        attention=impl,
    )


@dataclass(frozen=True)
class MlaBackLayer:
    """What the gfx1250 MLA back half (``mla_back_gfx1250``) reads for one
    full-attention layer, behind its absorb: the DCP group's FP8 ``W_UV`` and
    scale (project before merge) and this rank's o_proj in ``_block_fp8``'s
    layout. ``dcp`` ranks merge ``local_heads`` heads each, this rank
    ``dcp_rank`` of them."""

    layer_id: int
    w_uv: torch.Tensor
    s_uv: torch.Tensor
    w_o: torch.Tensor
    s_o: torch.Tensor
    local_heads: int
    dcp: int
    dcp_rank: int

    @property
    def heads(self) -> int:
        """The DCP group's query heads: the decode's partials carry them."""
        return self.dcp * self.local_heads

    @property
    def key(self) -> tuple[int, int]:
        """What its build bakes in."""
        return self.dcp, self.local_heads


def _merge_is_aiters(dcp_ops) -> bool:
    """Whether this tree's ``dcp_ops.cp_lse_a2a`` hands a decode merge to the
    DCP group's fused kernel (``cp_group.dcp_lse_merge``: aiter's
    ``dcp_merge_ll_gfx1250``, the merge the back half reproduces) instead of
    always running the Triton pack, all-to-all and combine, whose rows are
    not the kernel's bit for bit. Read off the function: the name it looks up
    on the group."""
    import inspect

    merge = getattr(dcp_ops, "cp_lse_a2a", None)
    code = getattr(inspect.unwrap(merge) if callable(merge) else None, "__code__", None)
    return code is not None and (
        "dcp_lse_merge" in code.co_names or "dcp_lse_merge" in code.co_consts
    )


def bind_mla_back(
    layer, layer_id: int, dims: Dims, absorb: MlaAbsorbLayer, rank: int
) -> MlaBackLayer:
    """One full-attention layer's MLA back half tensors, read in place.
    Refused where ``MLAAttention._dcp_project_merge_out``, the gate and
    o_proj would compute anything else: no DCP or a DCP group other than the
    TP group (the kernel merges and all-reduces among the same ranks), the
    merge on the latent (project before merge off), a sparse layer, a merge
    other than aiter's ``dcp_merge_ll_gfx1250`` behind the all-to-all backend
    (the mailbox combine; a communicator without the merge; a tree whose
    ``cp_lse_a2a`` does not take it, ``_merge_is_aiters``: the Triton
    combine's bits are not reproduced), the BF16 or FP4 V up-projection,
    other widths, an o_proj the in-kernel one cannot read."""
    from atom.model_ops import attention_mla, dcp_ops
    from atom.models.kimi_k3.mono.kernels.stages import mla_back_layout as lay

    name = f"layer {layer_id}"
    impl = absorb.attention
    dcp = int(getattr(impl, "dcp_world_size", 1))
    _need(dcp > 1, f"{name}: no DCP, nothing to merge")
    _need(
        dcp == dims.tp and int(impl.dcp_rank) == rank,
        f"{name}: the DCP group ({dcp} ranks) is not the TP group",
    )
    _need(
        bool(getattr(impl, "pbm_enabled", False)),
        f"{name}: the merge carries the latent (project before merge is off)",
    )
    _need(not impl.is_sparse_mla, f"{name}: sparse MLA")
    _need(
        impl.dcp_comm_backend == "a2a",
        f"{name}: the DCP merge runs the {impl.dcp_comm_backend!r} backend",
    )
    _need(
        callable(getattr(impl.dcp_group, "dcp_lse_merge", None)),
        f"{name}: the DCP group has no aiter merge kernel to reproduce",
    )
    _need(
        _merge_is_aiters(dcp_ops),
        f"{name}: this tree's DCP merge is the Triton combine, not aiter's "
        "merge kernel (dcp_ops.cp_lse_a2a never takes the group's)",
    )
    _need(
        not attention_mla.is_rocm_aiter_fp4bmm_enabled(),
        f"{name}: the FP4 V up-projection",
    )
    _need(
        (impl.kv_lora_rank, impl.v_head_dim) == (lay.KV_LORA, lay.V_DIM),
        f"{name}: MLA widths",
    )
    local_heads = int(impl.num_heads)
    _need(
        local_heads * lay.V_DIM == dims.proj,
        f"{name}: {local_heads} heads a rank, o_proj reads {dims.proj}",
    )
    heads = dcp * local_heads
    w_uv = getattr(impl, "W_V_dcp", None)
    s_uv = getattr(impl, "W_V_dcp_scale", None)
    _need(
        w_uv is not None and s_uv is not None,
        f"{name}: the BF16 V up-projection (ATOM_MLA_ABSORB_BF16)",
    )
    _need(
        w_uv.dtype == torch.float8_e4m3fn
        and tuple(w_uv.shape) == (heads, lay.V_DIM, lay.KV_LORA)
        and w_uv.is_contiguous(),
        f"{name}: W_UV is {w_uv.dtype} {tuple(w_uv.shape)}",
    )
    _need(
        s_uv.dtype == torch.float32
        and s_uv.numel() == 1
        and s_uv.device == w_uv.device,
        f"{name}: W_UV's scale is not one f32 beside it",
    )
    try:
        lay.BackPlan(1, dims.tp, dcp, local_heads)
    except ValueError as why:
        raise MonoUnsupported(f"{name}: {why}") from why
    w_o, s_o = bind_fp8_o_proj(layer, layer_id, dims)
    return MlaBackLayer(
        layer_id=layer_id,
        w_uv=w_uv,
        s_uv=s_uv,
        w_o=w_o,
        s_o=s_o,
        local_heads=local_heads,
        dcp=dcp,
        dcp_rank=rank,
    )


def bind_fp8_o_proj(layer, layer_id: int, dims: Dims):
    """A KDA layer's o_proj as gfx1250's in-kernel o_proj reads it:
    ``(weight, scale)``, ``_block_fp8``'s per_1x128 FP8 layout."""
    return _block_fp8(
        layer.self_attn.o_proj, f"layer {layer_id} o_proj", (HIDDEN, dims.proj)
    )


def bind_moe(layer, layer_id: int, dims: Dims, rank: int) -> MoeLayer:
    """One layer's K2 tensors: o_proj and the latent MoE."""
    name = f"layer {layer_id}"
    moe = getattr(layer, "block_sparse_moe", None)
    _need(moe is not None, f"{name}: not a MoE layer")
    _need(moe.use_latent_moe, f"{name}: not a latent MoE")
    _need(moe.shared_experts is not None, f"{name}: no shared experts")
    _need(moe.routed_expert_norm is not None, f"{name}: no routed norm")
    routed = bind_routed_moe(layer, layer_id, dims, rank)
    return MoeLayer(
        layer_id=routed.layer_id,
        blocks=routed.blocks,
        write_idx=routed.write_idx,
        seam=routed.seam,
        w_o=_plain_bf16(layer.self_attn.o_proj, f"{name} o_proj", (HIDDEN, dims.proj)),
        w_gate=routed.w_gate,
        bias=routed.bias,
        w_ld=routed.w_ld,
        ln_w=routed.ln_w,
        w_up=routed.w_up,
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
        w13=routed.w13,
        w13s=routed.w13s,
        w2=routed.w2,
        w2s=routed.w2s,
        gu_interleave=routed.gu_interleave,
        ln_eps=routed.ln_eps,
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
        """K1 and K2 both."""
        return self.kda is not None and self.moe is not None

    @property
    def ffn_only(self) -> bool:
        """K2 alone: a full-attention layer whose attention stays on the
        production path (``bind_model``)."""
        return self.kda is None and self.moe is not None

    @property
    def k1_only(self) -> bool:
        """K1 alone; projection, MoE and their collectives stay production."""
        return self.kda is not None and self.moe is None


def bind_model(
    model, dims: Dims, rank: int, *, bind_k2: bool = True, fp8_in_proj: bool = False
) -> list[LayerPlan]:
    """Every decoder layer, in order.

    A layer takes K1 and K2 only when it is a KDA layer with a MoE FFN and at
    least one AttnRes block behind it: layer 0 has a dense MLP and no block, and
    the full-attention layers keep the original MLA attention (and their paged KV
    cache) because the kernels have no attention of their own.

    Those full-attention layers get K2 alone -- ``o_proj`` and the latent MoE
    are not KDA's, only the core they read is. K2 restarts the prefix only on a
    layer that closes a block, which this path does not bank (a block closes at
    ``layer_id % ATTN_RES_BLOCK == 0``; K3's full-attention layers are 3 mod 4),
    so a layer that would is refused rather than read wrong.
    """
    layers = model.layers
    plans = []
    for layer_id, layer in enumerate(layers):
        blocks, _ = layer_block(layer_id)
        has_ffn = hasattr(layer, "block_sparse_moe") and blocks >= 1
        takes = bool(getattr(layer, "is_linear_attn", False)) and has_ffn
        if not takes and has_ffn:
            if not bind_k2:
                plans.append(LayerPlan(layer_id, None, None))
                continue
            # One layer's refusal leaves that layer on the production path; it
            # must not take the KDA layers' mono down with it.
            try:
                _need(
                    layer_block(layer_id)[1] < 0,
                    "a full-attention layer that closes a block",
                )
                moe = bind_moe(layer, layer_id, dims, rank)
                plans.append(LayerPlan(layer_id, None, moe))
            except MonoUnsupported as e:
                logger.warning(
                    "Kimi-K3 mono: layer %d keeps its production K2 (%s)", layer_id, e
                )
                plans.append(LayerPlan(layer_id, None, None))
            continue
        if not takes:
            plans.append(LayerPlan(layer_id, None, None))
            continue
        plans.append(
            LayerPlan(
                layer_id,
                bind_kda(layer, layer_id, dims, fp8_in_proj),
                bind_moe(layer, layer_id, dims, rank) if bind_k2 else None,
            )
        )
    _need(
        any(p.mono if bind_k2 else p.k1_only for p in plans),
        f"no layer takes the kernels ({len(layers)} layers, block size "
        f"{ATTN_RES_BLOCK})",
    )
    return plans

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications: namespace/dependency extraction for ATOM, 2026-10-13.

"""Host-side weight packing into the DeepSeek-V4 kernel's 16-row MFMA tiles."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from atom.models.deepseek_v4.mono.config import (
    ExpertWeight,
    MoeMode,
    V4Config,
    moe_format,
)

__all__ = [
    "BoundLayer",
    "LayerWeights",
    "bind_layer",
    "pack_bf16",
    "pack_fp8",
    "pack_layer_weights",
    "pack_mxfp4",
    "pack_mxfp4_scales",
]


@dataclass
class LayerWeights:
    """One rank's logical weights; packing happens before capture, never per width."""

    cfg: V4Config
    t: dict[str, torch.Tensor]


def pack_mxfp4(q: torch.Tensor, gate_up: bool) -> torch.Tensor:
    """An MXFP4 bank [E, N, K / 2] in ATOM's gfx950 layout (aiter ``shuffle_weight``,
    ``is_guinterleave=True``), so the kernel can read ATOM's own bank."""
    q = q.view(torch.uint8)
    e, n, kp = q.shape
    if n % (32 if gate_up else 16) or kp % 64:
        raise ValueError(
            f"MXFP4 bank dimensions must be divisible by (16 per half, 128), got {(n, kp * 2)}"
        )
    if gate_up:
        w = q.view(e, 2, n // 32, 16, kp // 64, 4, 16).permute(0, 2, 1, 4, 5, 3, 6)
    else:
        w = q.view(e, n // 16, 16, kp // 64, 4, 16).permute(0, 1, 3, 4, 2, 5)
    return w.contiguous().view(e, n, kp)


def pack_mxfp4_scales(s: torch.Tensor, gate_up: bool) -> torch.Tensor:
    """E8M0 scales [E, N, K / 32] in ATOM's gfx950 layout (aiter ``shuffle_scale``,
    ``is_guinterleave=True``); K / 32 is padded to a multiple of 8 with 1.0 (0x7F)."""
    s = s.view(torch.uint8)
    e, n, kb = s.shape
    kb8 = -(-kb // 8) * 8
    if kb8 != kb:
        s = torch.cat(
            [s, torch.full((e, n, kb8 - kb), 0x7F, dtype=torch.uint8, device=s.device)],
            dim=-1,
        )
    if gate_up:
        t = s.view(e, 2, n // 32, 16, kb8 // 8, 2, 4).permute(0, 2, 4, 6, 3, 5, 1)
    else:
        t = s.view(e, n // 32, 2, 16, kb8 // 8, 2, 4).permute(0, 1, 4, 6, 3, 5, 2)
    return t.contiguous().view(e, n, kb8)


ATTENTION_NAMES = ("w_qkv_a", "w_q_b", "w_o_a", "w_o_b")
# CSA layers only
OPTIONAL_ATTENTION_NAMES = ("w_i_q_b",)
EXPERT_NAMES = ("w_ug", "w_dn")
# the shared expert of an MXFP4 bank stays FP8
SHARED_EXPERT_NAMES = ("w_sug", "w_sdn")
HC_NAMES = ("hc_attn_fn", "hc_ffn_fn")


EXPERT_BANK_NAMES = ("w_ug", "s_ug", "w_dn", "s_dn")


def pack_layer_weights(
    tensors: dict[str, torch.Tensor],
    moe_mode: MoeMode | str = MoeMode.A8W4,
    experts: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Pack every matrix the V4 kernel reads; ``experts``, if given, is an already
    packed MXFP4 bank (e.g. ATOM's) used as is."""
    need = (
        (*ATTENTION_NAMES, "w_r")
        if experts is not None
        else (*ATTENTION_NAMES, *EXPERT_NAMES, "w_r")
    )
    missing = [name for name in need if name not in tensors]
    if missing:
        raise ValueError(f"missing layer weights: {', '.join(missing)}")
    packed = {name: pack_fp8(tensors[name]) for name in ATTENTION_NAMES}
    packed.update(
        {
            name: pack_fp8(tensors[name])
            for name in OPTIONAL_ATTENTION_NAMES
            if name in tensors
        }
    )
    if experts is not None:
        if moe_format(moe_mode).weight is not ExpertWeight.MXFP4_BLOCK32:
            raise ValueError("an external expert bank is MXFP4 (ATOM's layout)")
        packed.update({name: experts[name] for name in EXPERT_BANK_NAMES})
    elif moe_format(moe_mode).weight is ExpertWeight.MXFP4_BLOCK32:
        for name, scale, gate_up in (("w_ug", "s_ug", True), ("w_dn", "s_dn", False)):
            packed[name] = pack_mxfp4(tensors[name], gate_up)
            packed[scale] = pack_mxfp4_scales(tensors[scale], gate_up)
    else:
        packed.update({name: pack_fp8(tensors[name]) for name in EXPERT_NAMES})
    packed.update(
        {
            name: pack_fp8(tensors[name])
            for name in SHARED_EXPERT_NAMES
            if name in tensors
        }
    )
    packed["w_r"] = pack_bf16(tensors["w_r"])
    if "w_qkv_c" in tensors:
        packed["w_qkv_c"] = pack_bf16(tensors["w_qkv_c"])
    for name in HC_NAMES:
        if name in tensors:
            packed[name] = pack_hc_fn(tensors[name])
    return packed


def pack_hc_fn(fn: torch.Tensor) -> torch.Tensor:
    """An fp32 mixer [rows, K] as bf16 [hi; lo] (hi + lo ~ 16 mantissa bits), as ATOM's mHC."""
    fn = fn.float()
    hi = fn.to(torch.bfloat16)
    lo = (fn - hi.float()).to(torch.bfloat16)
    return pack_bf16(torch.cat([hi, lo]))


def pack_fp8(q: torch.Tensor) -> torch.Tensor:
    """Pack FP8 ``[..., N, K]`` for the kernel's 16-row, 64-K MFMA tiles."""

    *lead, rows, k = q.shape
    if rows % 16 or k % 64:
        raise ValueError(
            f"FP8 matrix dimensions must be divisible by (16, 64), got {(rows, k)}"
        )
    w8 = q.view(torch.uint8).reshape(*lead, rows // 16, 16, k // 64, 2, 4, 8)
    nlead = len(lead)
    order = list(range(nlead)) + [nlead + position for position in (0, 2, 4, 1, 3, 5)]
    return w8.permute(*order).contiguous().view(-1)


def pack_bf16(w: torch.Tensor) -> torch.Tensor:
    """Pack BF16 ``[N, K]`` for the kernel's 16-row, 64-K MFMA tiles."""

    if w.ndim != 2:
        raise ValueError(f"BF16 packing expects a matrix, got shape {tuple(w.shape)}")
    rows, k = w.shape
    if rows % 16 or k % 64:
        raise ValueError(
            f"BF16 matrix dimensions must be divisible by (16, 64), got {(rows, k)}"
        )
    w16 = w.view(torch.int16).reshape(rows // 16, 16, k // 64, 2, 4, 8)
    return w16.permute(0, 2, 3, 4, 1, 5).contiguous().view(-1)


@dataclass(frozen=True)
class BoundLayer:
    """Post-load V4 binding: large expert banks are views, small matrices packed once.

    The PR's attention layout differs from ordinary ATOM's 16x16 shuffle.
    Lossless byte permutations/scales are permitted; silent requantization is
    not. This does not admit real cache geometry or install a serving path.
    Any weight reload requires rebinding and graph invalidation.
    """

    layer_id: int
    weights: LayerWeights
    packed: dict[str, torch.Tensor]
    owner: object
    sources: tuple[tuple[object, str, torch.Tensor, tuple], ...]

    def assert_current(self):
        for owner, name, tensor, signature in self.sources:
            if (
                getattr(owner, name, None) is not tensor
                or _signature(tensor) != signature
            ):
                from atom.mono.runtime.consensus import MonoUnsupported

                raise MonoUnsupported(
                    f"V4 layer {self.layer_id}: weight reload invalidated the binding"
                )


def _signature(tensor):
    return (
        tensor.data_ptr(),
        tensor._version,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
        bool(getattr(tensor, "is_shuffled", False)),
    )


def _logical_fp8(weight: torch.Tensor) -> torch.Tensor:
    """Losslessly invert ATOM's gfx950 16x16 weight shuffle, never dequantize."""
    if not getattr(weight, "is_shuffled", False):
        return weight.detach()
    n, k = weight.shape
    raw = weight.view(torch.uint8).view(n // 16, k // 32, 2, 16, 16)
    return raw.permute(0, 3, 1, 2, 4).contiguous().view(n, k).view(weight.dtype)


def bind_layer(block, cfg: V4Config) -> BoundLayer:
    """Bind an actual ATOM V4 TP8 block *after* ordinary loader processing.

    Strict first profile: gfx950 V4-Pro, FP8 block128 attention/shared, native
    interleaved MXFP4 expert bank, selecting CSA FP4 index, hash or bias router.
    BF16-fallback wo_a, fused-shared and EP are refused. CSA reuse layers bind
    only the main compressor and consume the prior refresh layer's ``i_sel``.
    Ordinary/PR arithmetic equality still needs the separate teacher/cache gate.
    """
    from aiter import QuantType
    from aiter.fused_moe import resolve_activation_dtype
    from aiter.ops.flydsl.moe_common import GateMode

    from atom.model_ops.attentions.pool_layout.v4_pool_fields import (
        FP4_GFX950_OPUS,
        FP4_GFX950_PRESHUFFLE,
    )
    from atom.mono.runtime.consensus import MonoUnsupported

    attn, ffn = block.attn, block.ffn
    expected = (
        7168,
        16,
        1536,
        512,
        64,
        2,
        1024,
        384,
        6,
        384,
        128,
        4,
        20,
        1e-6,
        2.5,
        10.0,
    )
    actual = (
        cfg.hidden,
        cfg.heads,
        cfg.q_lora,
        cfg.head_dim,
        cfg.rope_dim,
        cfg.o_groups,
        cfg.o_lora,
        cfg.n_experts,
        cfg.top_k,
        cfg.inter,
        cfg.window,
        cfg.hc_mult,
        cfg.hc_sinkhorn_iters,
        cfg.hc_eps,
        cfg.route_scale,
        cfg.swiglu_limit,
    )
    if actual != expected or cfg.indexer_hadamard or cfg.eps != 1e-6:
        raise MonoUnsupported(f"unsupported PR V4-Pro binding geometry/math: {actual}")
    if not getattr(attn, "_is_gfx950", False) or attn.tp_size != 8 or ffn.tp_size != 8:
        raise MonoUnsupported("V4 post-load binder requires gfx950 pure TP8")
    expected_skip_topk = cfg.indexed and not cfg.index_topk_refresh
    invalid_index_policy = (
        bool(attn.skip_topk) != expected_skip_topk
        or (
            cfg.indexed
            and cfg.index_topk_refresh
            and cfg.index_topk_source_layer != int(block.layer_id)
        )
        or (
            expected_skip_topk
            and not 0 <= cfg.index_topk_source_layer < int(block.layer_id)
        )
    )
    if (
        attn.compress_ratio != cfg.compress_ratio
        or attn.kv_fp8 != cfg.kv_fp8
        or invalid_index_policy
        or attn.n_local_heads != cfg.heads
        or attn.n_local_groups != cfg.o_groups
        or block.hc_mult != cfg.hc_mult
        or block.hc_sinkhorn_iters != cfg.hc_sinkhorn_iters
        or block.hc_eps != cfg.hc_eps
        or block.norm_eps != cfg.eps
        or block.HC_POST_MULT != 2.0
        or ffn.routed_scaling_factor != cfg.route_scale
        or ffn.swiglu_limit != cfg.swiglu_limit
    ):
        raise MonoUnsupported(
            "V4 block policy does not match the compiled geometry/math"
        )
    if ffn.shared_experts is None or getattr(ffn, "_fuse_shared_into_routed", False):
        raise MonoUnsupported(
            "the PR first profile requires a separate FP8 shared expert"
        )
    if cfg.attention_fp8_bf16:
        projections = [attn.wqkv_a, attn.wq_b, attn.wo_b]
        if cfg.indexed and cfg.index_topk_refresh:
            projections.append(attn.indexer.wq_b)
        if (
            any(not m.blockscale_e8m0_scale for m in projections)
            or not attn.q_norm.use_fused_quant
            or not attn.q_norm._aiter_e8m0_scale
        ):
            raise MonoUnsupported(
                "real attention profile requires E8M0 block128 and fused q_norm FP8"
            )
    if cfg.shared_bf16_block128:
        shared = ffn.shared_experts
        if (
            shared.use_fused_clamp_act_mul
            or shared.swiglu_limit != cfg.swiglu_limit
            or any(
                not m.blockscale_e8m0_scale for m in (shared.gate_up_proj, shared.w2)
            )
        ):
            raise MonoUnsupported(
                "real shared profile requires non-fused BF16 activation and E8M0 block128"
            )
    experts, method = ffn.experts, ffn.experts.quant_method
    if (
        not getattr(method, "is_guinterleave", False)
        or getattr(method, "use_triton", False)
        or getattr(method, "use_triton_ep", False)
    ):
        raise MonoUnsupported(
            "routed expert bank is not the native gfx950 interleaved layout"
        )
    if (
        resolve_activation_dtype(
            method.quant_type,
            experts.w13_weight.dtype,
            activation=experts.activation,
            gate_mode=GateMode.INTERLEAVE,
            M=1,
            hidden_dtype=torch.bfloat16,
            gfx="gfx950",
        )
        != torch.float8_e4m3fn
    ):
        raise MonoUnsupported(
            "ordinary S1 routed dispatch is not the controlled A8W4 profile"
        )
    sources = []
    dev = block.attn_norm.weight.device

    def read(owner, name, shape, dtypes):
        value = getattr(owner, name, None)
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or value.dtype not in dtypes
            or not value.is_contiguous()
            or value.device != dev
            or not value.is_cuda
        ):
            raise MonoUnsupported(
                f"V4 layer {block.layer_id}: unsupported {name} {getattr(value, 'shape', None)}"
            )
        sources.append((owner, name, value, _signature(value)))
        return value

    fp8, bf16, f32 = (torch.float8_e4m3fn,), (torch.bfloat16,), (torch.float32,)
    scales = (torch.uint8, torch.float8_e8m0fnu, torch.float32)
    t = {}

    def matrix(name, module, n, k):
        if module.quant_type != QuantType.per_1x128:
            raise MonoUnsupported(f"{name}: only block128 FP8 is supported")
        weight = read(module, "weight", (n, k), fp8)
        scale = read(module, "weight_scale", (n // 128, k // 128), scales)
        t[f"w_{name}"] = _logical_fp8(weight)
        t[f"s_{name}"] = (
            scale.float().contiguous()
            if scale.dtype != torch.uint8
            else torch.exp2(scale.float() - 127).contiguous()
        )
        if not torch.isfinite(t[f"s_{name}"]).all():
            raise MonoUnsupported(f"{name}: nonfinite block scale")

    for name, module, n, k in (
        ("qkv_a", attn.wqkv_a, cfg.q_lora + cfg.head_dim, cfg.hidden),
        ("q_b", attn.wq_b, cfg.heads * cfg.head_dim, cfg.q_lora),
        ("o_b", attn.wo_b, cfg.hidden, cfg.o_groups * cfg.o_lora),
        ("sug", ffn.shared_experts.gate_up_proj, 2 * cfg.inter, cfg.hidden),
        ("sdn", ffn.shared_experts.w2, cfg.hidden, cfg.inter),
    ):
        matrix(name, module, n, k)
    # wo_a's parent hook marks quant_type No; cached plain view/scales prove its layout.
    if not attn._wo_a_mxscale or attn._is_preshuffle:
        raise MonoUnsupported("wo_a must retain the plain FP8 grouped-LoRA path")
    t["w_o_a"] = read(
        attn, "_wo_a_w_fp8", (cfg.o_groups, cfg.o_lora, cfg.group_dim), fp8
    ).view(cfg.o_groups * cfg.o_lora, cfg.group_dim)
    woa = read(attn.wo_a, "weight", (cfg.o_groups * cfg.o_lora, cfg.group_dim), fp8)
    if woa.data_ptr() != t["w_o_a"].data_ptr() or getattr(woa, "is_shuffled", False):
        raise MonoUnsupported("wo_a cached view no longer names plain parameter bytes")
    scale = read(
        attn,
        "_wo_a_w_scale",
        (cfg.o_groups, cfg.o_lora // 128, cfg.group_dim // 128),
        (torch.uint8, torch.float8_e8m0fnu),
    )
    t["s_o_a"] = (
        torch.exp2(scale.view(torch.uint8).float() - 127)
        .view(cfg.o_groups * cfg.o_lora // 128, cfg.group_dim // 128)
        .contiguous()
    )
    for name, owner, shape in (
        ("g_in", block.attn_norm, (cfg.hidden,)),
        ("g_post", block.ffn_norm, (cfg.hidden,)),
        ("g_q", attn.q_norm, (cfg.q_lora,)),
        ("g_kv", attn.kv_norm, (cfg.head_dim,)),
    ):
        t[name] = read(owner, "weight", shape, bf16)
    t["attn_sink"] = read(attn, "attn_sink", (cfg.heads,), f32)
    t["w_r"] = read(ffn.gate, "weight", (cfg.n_experts, cfg.hidden), bf16)
    if ffn.is_hash_layer:
        lookup = ffn.gate.tid2eid
        t["tid2eid"] = read(
            ffn.gate, "tid2eid", (lookup.shape[0], cfg.top_k), (torch.int32,)
        )
        t["bias"] = torch.zeros(cfg.n_experts, dtype=torch.float32, device=dev)
    else:
        t["bias"] = read(ffn.gate, "e_score_correction_bias", (cfg.n_experts,), f32)
    tails = []
    for prefix, compressor, width in (
        ("", attn.compressor, cfg.head_dim),
        (
            "i_",
            attn.indexer.compressor if cfg.indexed and cfg.index_topk_refresh else None,
            cfg.index_head_dim,
        ),
    ):
        if compressor is None:
            continue
        cw = cfg.c_coff * width
        tails.append(read(compressor.wkv_gate, "weight", (2 * cw, cfg.hidden), bf16))
        t[prefix + "ape"] = read(compressor, "ape", (cfg.compress_ratio, cw), f32)
        t["g_ickv" if prefix else "g_ckv"] = read(
            compressor.norm, "weight", (width,), bf16
        )
    if tails:
        t["w_qkv_c"] = torch.cat(tails).contiguous()
    if cfg.indexed and cfg.index_topk_refresh:
        idx = attn.indexer
        if (
            not idx._indexer_fp4
            or idx.indexer_layout not in (FP4_GFX950_PRESHUFFLE, FP4_GFX950_OPUS)
            or idx.indexer_layout != cfg.indexer_layout
            or idx.index_topk != cfg.index_topk
        ):
            raise MonoUnsupported(
                "CSA requires the selecting gfx950 FP4 indexer profile"
            )
        matrix("i_q_b", idx.wq_b, cfg.index_heads * cfg.index_head_dim, cfg.q_lora)
        t["i_w"] = read(idx.weights_proj, "weight", (cfg.index_heads, cfg.hidden), bf16)
    hc_packed = {}
    for side in ("attn", "ffn"):
        name = f"hc_{side}_fn"
        fn = read(
            block,
            name,
            (cfg.hc_mix, cfg.hc_mult * cfg.hidden),
            (torch.int32, torch.float32),
        )
        if fn.dtype == torch.int32:
            # AITER gfx950 stores [8 hi,8 lo] blocks. Preserve the original
            # split bytes in the PR tile; don't round/reconstruct and split again.
            parts = fn.view(torch.bfloat16).view(cfg.hc_mix, -1, 2, 8)
            hi = parts[:, :, 0].reshape(cfg.hc_mix, -1)
            lo = parts[:, :, 1].reshape(cfg.hc_mix, -1)
            pad = cfg.hc_rows - cfg.hc_mix
            hi, lo = (torch.nn.functional.pad(x, (0, 0, 0, pad)) for x in (hi, lo))
            t[name] = hi.float() + lo.float()
            hc_packed[name] = pack_bf16(torch.cat([hi, lo]))
        else:
            t[name] = torch.nn.functional.pad(fn, (0, 0, 0, cfg.hc_rows - cfg.hc_mix))
        t[f"hc_{side}_base"] = read(block, f"hc_{side}_base", (cfg.hc_mix,), f32)
        t[f"hc_{side}_scale"] = read(block, f"hc_{side}_scale", (3,), f32)
    bank = {
        "w_ug": read(
            experts,
            "w13_weight",
            (cfg.n_experts, 2 * cfg.inter, cfg.hidden // 2),
            (torch.float4_e2m1fn_x2,),
        ),
        "w_dn": read(
            experts,
            "w2_weight",
            (cfg.n_experts, cfg.hidden, cfg.inter // 2),
            (torch.float4_e2m1fn_x2,),
        ),
        "s_ug": read(
            experts,
            "w13_weight_scale",
            (cfg.n_experts * 2 * cfg.inter, cfg.hidden // 32),
            (torch.uint8, torch.float8_e8m0fnu),
        ),
        "s_dn": read(
            experts,
            "w2_weight_scale",
            (cfg.n_experts * cfg.hidden, -(-(cfg.inter // 32) // 8) * 8),
            (torch.uint8, torch.float8_e8m0fnu),
        ),
    }
    if not all(getattr(bank[name], "is_shuffled", False) for name in ("w_ug", "w_dn")):
        raise MonoUnsupported(
            "MXFP4 bank has not completed the ordinary loader shuffle"
        )
    # Detached tensor objects keep the exact storage alive across a Parameter
    # .data replacement; owner/signature checks catch a required graph rebuild.
    t = {name: value.detach() for name, value in t.items()}
    bank = {name: value.detach() for name, value in bank.items()}
    packed = pack_layer_weights(t, MoeMode.A8W4, bank)
    packed.update(hc_packed)
    # Production launches merge ``weights.t`` with ``packed`` and the packed
    # entry wins for every duplicate key.  Keeping the logical FP8 matrices,
    # compressor concat and reconstructed FP32 mHC functions after packing
    # therefore retains several GiB of dead storage across 61 layers.  The
    # source Parameter objects remain owned and generation-checked through
    # ``sources``; retain here only tensors the launch reads without a packed
    # replacement (scales, norms, compressor metadata, bias and lookup tables).
    runtime_t = {name: value for name, value in t.items() if name not in packed}
    return BoundLayer(
        int(block.layer_id),
        LayerWeights(cfg, runtime_t),
        packed,
        block,
        tuple(sources),
    )

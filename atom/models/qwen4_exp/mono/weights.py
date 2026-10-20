# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The kernel arguments of one decoder layer, from ATOM's loaded modules.

ATOM's MXFP4 checkpoint holds the values the kernels compute with, in ATOM's
layouts:

- **Read in place.** The routed and shared experts (``w13`` / ``w2`` and their
  scales: aiter's gate/up-interleaved shuffle, which keeps the kernels' 16 x 128
  tiles), the router, the HC inject rows, the conv and ``ba`` weights.
  The MTP draft layer's shared expert is not fused into its experts: it
  becomes a mono-owned copy in a fused slot's layout (:func:`expert_slot`).
- **Mono-owned copies.** The dense MXFP4 linears are row-major in ATOM and
  become the kernels' tiles (QSA's rows also put in the mono's order). The HC
  low-rank pair is bf16 in ATOM and is requantized to MXFP4: the same values,
  and the original scales but for a few blocks (:func:`mxfp4_exact`). The
  norms fold ``1 +``, ``A_log`` and ``dt_bias`` widen to f32 and the RoPE
  tables repeat their 32 columns.

Every tensor is checked against the shape and dtype the kernels assume; a
model loaded differently is refused (``MonoUnsupported``) instead of misread.
The arguments are named as the kernels' ``LAYER_ARGS`` and the ``qsa.STATE``
/ ``ple.STATE`` tables name them.
"""

from dataclasses import dataclass

import torch
from aiter.ops.shuffle import moe_shuffle_scale, moe_shuffle_weight

from atom.mono.runtime.consensus import MonoUnsupported

from .kernels.config import DEFAULT as C
from .kernels.p2 import SCALE_ONE, dequantize_mxfp4, k_iters, scales_chain, scales_k4

BF16, F32, U8 = torch.bfloat16, torch.float32, torch.uint8
FP4X2, E8M0 = torch.float4_e2m1fn_x2, torch.float8_e8m0fnu
# the HC low-rank down projection runs split-K over this many chains
HC_DOWN_PARTS = 4
# aiter pads a routed expert's intermediate width to a 256-K step
MOE_INTER_PADDED = -(-C.moe_inter // 256) * 256


@dataclass
class LayerWeights:
    """One decoder layer's kernel arguments: ``args`` by ``LAYER_ARGS`` name,
    the QSA norms by ``qsa.STATE`` name (QSA layers), the PLE weights by
    ``ple.STATE`` name (the PLE layer)."""

    layer_id: int
    kind: str
    args: dict[str, torch.Tensor]
    qsa_norms: dict[str, torch.Tensor] | None = None
    ple: dict[str, torch.Tensor] | None = None


def _need(ok: bool, what: str) -> None:
    if not ok:
        raise MonoUnsupported(what)


def parameters(layer: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Every parameter, buffer and tensor attribute of ``layer`` by dotted name,
    detached; ATOM's ``is_shuffled`` flag (a Python attribute of the
    parameter, which ``detach`` drops) is carried over."""
    out = {}
    for module_name, module in layer.named_modules():
        tensors = list(module._parameters.items()) + list(module._buffers.items())
        tensors += [
            (attr, value)
            for attr, value in vars(module).items()
            if isinstance(value, torch.Tensor) and not attr.startswith("_")
        ]
        for attr, value in tensors:
            if value is not None:
                out.setdefault(f"{module_name}.{attr}" if module_name else attr, value)
    detached = {}
    for k, v in out.items():
        detached[k] = v.detach()
        if getattr(v, "is_shuffled", False):
            detached[k].is_shuffled = True
    return detached


def _get(prm, name, shape, dtype):
    t = prm.get(name)
    _need(t is not None, f"no {name}")
    _need(
        tuple(t.shape) == tuple(shape) and t.dtype == dtype,
        f"{name} is {tuple(t.shape)} {t.dtype}, expected {tuple(shape)} {dtype}",
    )
    _need(t.is_contiguous(), f"{name} not contiguous")
    return t


def _mxfp4(prm, name, N, K):
    """The row-major ``[N, K/2]`` weight and ``[N, K/32]`` scales of a native
    MXFP4 linear (never shuffled)."""
    _need(
        not getattr(prm.get(name + ".weight"), "is_shuffled", False), f"{name} shuffled"
    )
    return (
        _get(prm, name + ".weight", (N, K // 2), FP4X2).view(U8),
        _get(prm, name + ".weight_scale", (N, K // 32), E8M0).view(U8),
    )


def p2_tiles(weight, scale):
    """Row-major MXFP4 ``weight [N, K/2]`` + ``scale [N, K/32]`` -> the kernels'
    tiles ``[N/16][k_iters][4][16 rows][16 bytes]`` and small scales
    ``[N/16][k_iters][4][16 rows]`` (lane ``g * 16 + n`` holds row ``n``'s
    32-K block ``g`` of the 128-K step). K pads to 256 with zero codes and
    identity scales."""
    N, Kb = weight.shape
    nb, ki = (N + 15) // 16, k_iters(2 * Kb)
    w = torch.zeros(nb * 16, ki * 64, dtype=U8, device=weight.device)
    w[:N, :Kb] = weight
    s = torch.full((nb * 16, ki * 4), SCALE_ONE, dtype=U8, device=weight.device)
    s[:N, : scale.shape[1]] = scale
    tiles = w.view(nb, 16, ki, 4, 16).permute(0, 2, 3, 1, 4).contiguous().view(-1)
    small = s.view(nb, 16, ki, 4).permute(0, 2, 3, 1).contiguous().view(-1)
    return tiles, small


FP4_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def mxfp4_exact(weight):
    """``weight`` (bf16 ``[N, K]``, every 32-block an MXFP4 block's exact
    values) -> packed codes ``[N, K/2]`` + E8M0 ``[N, K/32]`` that dequantize to
    it bitwise, with the original quantizer's scale wherever the values
    determine it.

    The quantizer picks ``ceil(log2(amax / 6))``, clamped to [1, 254], on the
    original values, so a block can keep largest code 3 (an original amax in
    (3, 3.5) times its scale). On the dequantized values the same rule picks
    one exponent lower with every code doubled: the same values, but not the
    same scaled-MFMA result. So a block whose amax is 6 times that rule's
    scale steps one exponent up when its codes are exact there. A block the
    quantizer stored with largest code 6 and no code 0.5 or 1.5 has the same
    values; it comes out one exponent too high (15 of 2,048,000 blocks in the
    full model)."""
    N, K = weight.shape
    values = weight.float().view(N, K // 32, 32)
    amax = values.abs().amax(dim=2)
    exponent = torch.ceil(torch.log2(amax.clamp(min=2**-126) / 6.0)) + 127.0
    exponent = torch.where(amax == 0, 1.0, exponent.clamp(1, 254))
    grid = torch.tensor(FP4_GRID, device=weight.device)
    up = values.abs() / torch.exp2(exponent - 126.0).unsqueeze(-1)
    up = (amax == 6.0 * torch.exp2(exponent - 127.0)) & torch.isin(up, grid).all(-1)
    exponent = torch.where(up & (exponent < 254), exponent + 1, exponent)
    normalized = values / torch.exp2(exponent - 127.0).unsqueeze(-1)
    m = normalized.abs()
    codes = torch.zeros_like(m, dtype=U8)
    # exact values: only the grid points themselves occur
    for code, value in enumerate(FP4_GRID[1:], start=1):
        codes[m == value] = code
    codes[torch.signbit(normalized)] |= 8
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).view(N, K // 2)
    scales = exponent.to(U8)
    exact = torch.equal(dequantize_mxfp4(packed, scales), weight.float())
    return packed, scales, exact


# ------------------------------------------------------------------ layer parts


def _norm(prm, name, n):
    """Gemma-style ``1 + w`` norm weight, rounded to bf16."""
    return (1.0 + _get(prm, name, (n,), BF16).float()).to(BF16)


def hc(prm, h):
    """Hyper-connection ``h`` (e.g. ``attn_hyper_connection.``): the norm, the
    low-rank down / up pair requantized to MXFP4, the inject rows."""
    width, rank = C.hc_width, C.hc_lowrank
    down = _get(prm, h + "input_mix_weight_down.weight", (rank, width), BF16)
    up = _get(prm, h + "input_mix_weight_up.weight", (width, rank), BF16)
    out = {"hc_norm_w": _norm(prm, h + "hc_norm.weight", width)}
    for arg, w, N, K in (("down", down, rank, width), ("up", up, width, rank)):
        packed, scales, exact = mxfp4_exact(w)
        _need(exact, f"{h}input_mix_weight_{arg} is not exact MXFP4 values")
        tiles, small = p2_tiles(packed, scales)
        out["w_" + arg] = tiles
        out["s_" + arg] = (
            scales_chain(small, N, K, parts=HC_DOWN_PARTS)
            if arg == "down"
            else scales_k4(small, N, K)
        )
    out["w_inj"] = _get(
        prm, h + "block_inject_weight.weight", (C.hc_count, width), BF16
    )
    return out


def mlp_arg(k):
    """The MLP HC's kernel argument for attention HC argument ``k``."""
    return "m_" + k.removeprefix("hc_")


def hcs(prm):
    return {
        **hc(prm, "attn_hyper_connection."),
        **{mlp_arg(k): v for k, v in hc(prm, "mlp_hyper_connection.").items()},
    }


def moe(prm, experts, method=None):
    """The router and the experts, read in place (``method``: the experts'
    quant method, when bound from a live module). ``w13`` holds every expert's
    gate / up rows interleaved a 16-row block at a time (block ``2 j`` gate,
    ``2 j + 1`` up), the intermediate padded to 768; the shared expert is the
    last one. The scales are aiter's ``shuffle_scale`` order.
    A layer whose shared expert is its own MLP (ATOM's MTP draft layer) holds
    ``experts`` slots; its shared expert is then copied as a fused slot
    (:func:`expert_slot`)."""
    m = "mlp."
    separate = m + "shared_expert.gate_up_proj.weight" in prm
    E, H, inter = experts + (not separate), C.hidden, MOE_INTER_PADDED
    w13 = _get(prm, m + "experts.w13_weight", (E, 2 * inter, H // 2), FP4X2).view(U8)
    w2 = _get(prm, m + "experts.w2_weight", (E, H, inter // 2), FP4X2).view(U8)
    _need(
        getattr(prm[m + "experts.w13_weight"], "is_shuffled", False), "w13 not shuffled"
    )
    _need(
        getattr(prm[m + "experts.w2_weight"], "is_shuffled", False), "w2 not shuffled"
    )
    if method is not None:
        _need(
            getattr(method, "is_guinterleave", False)
            and getattr(method, "fp8_activations", False),
            "routed experts not A8W4 with interleaved gate / up rows",
        )
    s13 = _get(prm, m + "experts.w13_weight_scale", (E * 2 * inter, H // 32), U8)
    s2 = _get(prm, m + "experts.w2_weight_scale", (E * H, inter // 32), U8)
    s13, s2 = s13.view(E, -1), s2.view(E, -1)
    if separate:
        sh = m + "shared_expert."
        shared = expert_slot(
            *_mxfp4(prm, sh + "gate_up_proj", 2 * C.shared_inter, H),
            *_mxfp4(prm, sh + "down_proj", H, C.shared_inter),
        )
    else:
        shared = (w13[experts], s13[experts], w2[experts], s2[experts])
    return {
        # router rows, then the shared expert's gate row (ATOM merges them)
        "r_w": _get(prm, m + "gate.weight", (experts + 1, H), BF16),
        "e_w_gu": w13,
        "e_s_gu": s13,
        "sh_w_gu": shared[0],
        "sh_s_gu": shared[1],
        "e_w_dn": w2,
        "e_s_dn": s2,
        "sh_w_dn": shared[2],
        "sh_s_dn": shared[3],
    }


def expert_slot(w_gu, s_gu, w_dn, s_dn):
    """A dense MXFP4 MLP (row-major ``gate_up [2 I, H/2]``, its gate rows then
    its up rows, and ``down [H, I/2]``, with their ``[., K/32]`` scales) as one
    expert slot of ATOM's MXFP4 MoE: placed as its loader places an expert
    (``I`` padded to 768 with zero codes and scales) and shuffled as
    ``Mxfp4MoEMethod`` shuffles them (aiter's gate/up interleave). Returns the
    slot's ``(w13, w13 scales, w2, w2 scales)``, flat as :func:`moe` indexes
    the fused slots."""
    H, P = C.hidden, MOE_INTER_PADDED
    inter = w_gu.shape[0] // 2
    _need(inter <= P, f"shared expert intermediate {inter} > {P}")
    dev = w_gu.device
    w13 = torch.zeros(1, 2 * P, H // 2, dtype=U8, device=dev)
    s13 = torch.zeros(1, 2 * P, H // 32, dtype=U8, device=dev)
    for half in range(2):
        rows = slice(half * inter, (half + 1) * inter)
        w13[0, half * P : half * P + inter] = w_gu[rows]
        s13[0, half * P : half * P + inter] = s_gu[rows]
    w2 = torch.zeros(1, H, P // 2, dtype=U8, device=dev)
    s2 = torch.zeros(1, H, P // 32, dtype=U8, device=dev)
    w2[0, :, : inter // 2] = w_dn
    s2[0, :, : inter // 32] = s_dn
    w13 = moe_shuffle_weight(w13, 1, is_guinterleave=True, gate_up=True)
    w2 = moe_shuffle_weight(w2, 1, is_guinterleave=True, gate_up=False)
    s13 = moe_shuffle_scale(s13.view(2 * P, -1), 1, is_guinterleave=True, gate_up=True)
    s2 = moe_shuffle_scale(s2.view(H, -1), 1, is_guinterleave=True, gate_up=False)
    return w13[0], s13.view(-1), w2[0], s2.view(-1)


def delta(prm):
    """A DeltaNet layer's attention weights (its state is bound separately)."""
    a = "linear_attn."
    H, qkv, value = C.hidden, C.delta_qkv, C.delta_value
    heads = C.delta_v_heads
    w, s = _mxfp4(prm, a + "in_proj_qkvz", qkv + value, H)
    tiles, small = p2_tiles(w, s)
    # the q, k, v and z scales repack as two matrices, as the mono's
    split = qkv // 16 * k_iters(H) * 64
    s_qkvz = torch.cat(
        [scales_k4(small[:split], qkv, H), scales_k4(small[split:], value, H)]
    )
    w_out, s_out = p2_tiles(*_mxfp4(prm, a + "out_proj", H, value))
    return {
        "w_qkvz": tiles,
        "s_qkvz": s_qkvz,
        "conv_w": _get(prm, a + "conv1d.weight", (qkv, 1, C.delta_conv), BF16),
        # [b; a] rows, as the mono's
        "ba_w": _get(prm, a + "in_proj_ba.weight", (2 * heads, H), BF16),
        "a_log": _get(prm, a + "A_log", (heads,), prm[a + "A_log"].dtype).float(),
        "dt_bias": _get(prm, a + "dt_bias", (heads,), prm[a + "dt_bias"].dtype).float(),
        "norm_w": _get(prm, a + "norm.weight", (C.delta_v_dim,), BF16),
        "w_out": w_out,
        "s_out": scales_k4(s_out, H, value),
    }


def qsa_rows():
    """Row order of the mono's fused QSA projection, as rows of ATOM's
    ``[qkv_proj; index_qk_proj]``: index q and k, then per head its query
    and gate, then k, v. ``qkv_proj`` is ``[gate | query | k | v]``."""
    q, d = C.q_heads * C.head_dim, C.head_dim
    kv = 2 * C.kv_heads * d
    index = C.index_heads * C.index_dim + C.index_dim
    rows = [torch.arange(2 * q + kv, 2 * q + kv + index)]
    for h in range(C.q_heads):
        rows += [
            torch.arange(q + h * d, q + (h + 1) * d),
            torch.arange(h * d, (h + 1) * d),
        ]
    rows.append(torch.arange(2 * q, 2 * q + kv))
    return torch.cat(rows)


def qsa(prm):
    """A QSA layer's projection weights, and its norm weights by ``qsa.STATE``
    name (the rest of that table is state)."""
    a = "self_attn."
    H, d = C.hidden, C.head_dim
    q = C.q_heads * d
    index = C.index_heads * C.index_dim + C.index_dim
    w, s = _mxfp4(prm, a + "qkv_proj", 2 * q + 2 * C.kv_heads * d, H)
    iw, i_s = _mxfp4(prm, a + "indexer.index_qk_proj", index, H)
    order = qsa_rows().to(w.device)
    tiles, small = p2_tiles(torch.cat([w, iw])[order], torch.cat([s, i_s])[order])
    w_out, s_out = p2_tiles(*_mxfp4(prm, a + "o_proj", H, q))
    weights = {
        "w_qsa": tiles,
        "s_qsa": scales_k4(small, C.qsa_proj, H),
        "w_out": w_out,
        "s_out": scales_k4(s_out, H, q),
    }
    norms = {
        "q_norm": _norm(prm, a + "q_norm.weight", d),
        "k_norm": _norm(prm, a + "k_norm.weight", d),
        # The indexer's norms keep the raw ``w``: the kernel adds the 1 in f32
        # (``qsa.norm(..., zero_centered=True)``), as ATOM's and HF's Gemma norm.
        "iq_norm": _get(prm, a + "indexer.q_layernorm.weight", (C.index_dim,), BF16),
        "ik_norm": _get(prm, a + "indexer.k_layernorm.weight", (C.index_dim,), BF16),
    }
    return weights, norms


def ple(prm):
    """The PLE prelude's weights by ``ple.STATE`` name (the rest is state)."""
    p = "ple."
    H, width = C.hidden, C.hc_width
    key_w, key_s = p2_tiles(*_mxfp4(prm, p + "key_proj", width, H))
    val_w, val_s = p2_tiles(*_mxfp4(prm, p + "value_proj", H, H))
    return {
        "key_w": key_w,
        "key_s": scales_k4(key_s, width, H),
        "val_w": val_w,
        "val_s": scales_k4(val_s, H, H),
        "nq": _norm(prm, p + "norm_query.weight", width),
        "nk": _norm(prm, p + "norm_key.weight", width),
        "nc": _norm(prm, p + "norm_conv.weight", width),
        "conv_w": _get(prm, p + "conv1d.weight", (width, 1, C.ple_conv), BF16),
    }


def rope(prm, positions):
    """The mono's RoPE tables ``[positions, rotary_dim]`` bf16 from a QSA
    layer's ``rotary_emb`` caches, which hold the first half of each row."""
    half = C.rotary_dim // 2
    out = []
    for name in ("cos_cache", "sin_cache"):
        t = prm["self_attn.rotary_emb." + name]
        _need(
            t.dtype == BF16 and t.shape[-1] == half, f"rotary {name} {tuple(t.shape)}"
        )
        _need(t.shape[0] >= positions, f"rotary {name} has {t.shape[0]} positions")
        t = t.reshape(t.shape[0], half)[:positions]
        out.append(torch.cat([t, t], dim=-1).contiguous())
    return tuple(out)


def bind_layer(layer, layer_id: int, kind: str, experts: int) -> LayerWeights:
    """Layer ``layer_id``'s kernel arguments (``kind`` as
    ``kernels.config.Qwen4Config.kind``) from its loaded module."""
    return bind_parameters(
        parameters(layer), layer_id, kind, experts, layer.mlp.experts.quant_method
    )


def bind_parameters(
    prm, layer_id: int, kind: str, experts: int, method=None
) -> LayerWeights:
    """:func:`bind_layer` from the layer's tensors by dotted name (as
    :func:`parameters` returns them)."""
    try:
        args = hcs(prm) | moe(prm, experts, method)
        norms = table = None
        if kind == "qsa":
            weights, norms = qsa(prm)
            args |= weights
        else:
            args |= delta(prm)
        if kind == "delta_ple":
            table = ple(prm)
    except MonoUnsupported as e:
        raise MonoUnsupported(f"layer {layer_id}: {e}") from None
    return LayerWeights(layer_id, kind, args, norms, table)

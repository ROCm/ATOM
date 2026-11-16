# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The hi programs' weights from the tensors ATOM loaded (no second read of the checkpoint).

``block_tensors``: a target layer's or a DSpark draft stage's tensors on this TP rank, in the checkpoint's form and
names (what the hi runtime's ``layer_impl.pack(local=True)`` reads); ``expert_packs``: its routed experts' packs;
``ctx_tensors`` / ``engram_tensors``: the draft's context write's and an Engram layer's projection, likewise;
``rope_args``: ATOM's RoPE caches, read in place. ``check_block`` / ``check_experts`` / ``check_linears``: what these
refuse, raised before anything is packed (the install's check, ``hi_runner.check``).

ATOM holds every tensor these packers read, rank-local exactly as they slice it (TP without EP: ``wq_b`` / ``wo_a``
rows, ``wo_b`` columns, ``attn_sink``, the shared expert's and every routed expert's width slice), and changes them
at load by byte permutations only, undone or followed here:

- the FP8 linears' 16 x 16 preshuffle (``is_shuffled``), undone (``_unshuffle_linear_weight``); their E8M0 32 x 32
  block scales are the checkpoint's;
- the fused projections, split again: ``attn.wqkv_a`` = [wq_a; wkv], ``attn.compressor.wkv_gate`` = [wkv; wgate],
  ``ffn.shared_experts.gate_up_proj`` = [w1; w3] (their scales' rows likewise);
- the routed experts (``FusedMoE``, the gfx950 FlyDSL path): the weights in aiter's lane-level gate / up interleaved
  order, which is the hi runtime's unit order (``gemv.pack_weight``), so the packs are ATOM's bytes, read in place;
  the E8M0 codes in aiter's ``shuffle_scale`` order, read in place too (``expert_packs``).

The CPU check holds these packs to the checkpoint's, byte for byte (``tests/test_deepseek_v41_mono_hi_weights.py``).
"""

from __future__ import annotations

import logging

import torch

from atom.mono.runtime.consensus import MonoUnsupported
from atom.mono.weights.layer import _unshuffle_linear_weight

logger = logging.getLogger("atom")

_GROUP = 32  # the checkpoint's E8M0 block: 32 x 32 (dense FP8) / 1 x 32 (routed experts' FP4)


def check_linears(*lins) -> None:
    """``_dense``'s refusal, without its copy: each of ``lins`` (ATOM's FP8 linears) holds FP8."""
    for lin in lins:
        if lin.weight.dtype != torch.float8_e4m3fn:
            raise MonoUnsupported(
                f"V4.1 mono hi weights: an FP8 linear holds {lin.weight.dtype}"
            )


@torch.no_grad()
def _dense(
    lin, rows: tuple[int, int] | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """``lin`` (an ATOM FP8 linear) -> its weight in the checkpoint's row-major order (rows ``rows`` of a fused one)
    and its E8M0 block scale (the rows' blocks)."""
    check_linears(lin)
    w, s = _unshuffle_linear_weight(lin.weight).detach(), lin.weight_scale.detach()
    if rows is not None:
        a, b = rows
        w, s = w[a:b], s[a // _GROUP : b // _GROUP]
    return w, s


@torch.no_grad()
def block_tensors(block, prefix: str) -> tuple[dict, dict]:
    """``block`` (ATOM's V4.1 ``Block``: a target layer or a draft stage) -> (``w``, ``raw``): its tensors on this rank
    keyed as the checkpoint names them relative to the block (``attn.wq_a.weight``, ..), and the FP8 ``wo_a`` keyed
    with ``prefix`` (``layers.<i>.`` / ``mtp.<s>.``): ``layer_impl.pack(local=True)``'s inputs, the routed experts
    aside (``expert_packs``)."""
    a, f = block.attn, block.ffn
    q_lora = a.wqkv_a.output_sizes[0]
    kv_rows = sum(a.wqkv_a.output_sizes)
    w = {}
    w["attn.wq_a.weight"], w["attn.wq_a.scale"] = _dense(a.wqkv_a, (0, q_lora))
    w["attn.wkv.weight"], w["attn.wkv.scale"] = _dense(a.wqkv_a, (q_lora, kv_rows))
    w["attn.wq_b.weight"], w["attn.wq_b.scale"] = _dense(a.wq_b)
    w["attn.wo_b.weight"], w["attn.wo_b.scale"] = _dense(a.wo_b)
    gu = f.shared_experts.gate_up_proj
    half = gu.weight.shape[0] // 2
    w["ffn.shared_experts.w1.weight"], w["ffn.shared_experts.w1.scale"] = _dense(
        gu, (0, half)
    )
    w["ffn.shared_experts.w3.weight"], w["ffn.shared_experts.w3.scale"] = _dense(
        gu, (half, 2 * half)
    )
    w["ffn.shared_experts.w2.weight"], w["ffn.shared_experts.w2.scale"] = _dense(
        f.shared_experts.w2
    )
    plain = {
        "attn.q_norm.weight": a.q_norm.weight, "attn.kv_norm.weight": a.kv_norm.weight, "attn.attn_sink": a.attn_sink,
        "attn_norm.weight": block.attn_norm.weight, "ffn_norm.weight": block.ffn_norm.weight,
        "ffn.gate.weight": f.gate.weight, "ffn.gate.bias": f.gate.e_score_correction_bias,
    }  # fmt: skip
    for name in (
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_attn_base",
        "hc_ffn_fn",
        "hc_ffn_scale",
        "hc_ffn_base",
    ):
        plain[name] = getattr(block, name)
    comp, idx = getattr(a, "compressor", None), getattr(a, "indexer", None)
    if comp is not None:  # [wkv; wgate] (ratio 1: wkv alone)
        kv = comp.wkv_gate.weight
        n = kv.shape[0] // (1 if comp.ratio == 1 else 2)
        plain["attn.compressor.wkv.weight"] = kv[:n]
        if comp.ratio != 1:
            plain["attn.compressor.wgate.weight"] = kv[n:]
        plain["attn.compressor.norm.weight"] = comp.norm.weight
    if idx is not None:
        w["attn.indexer.wq_b.weight"], w["attn.indexer.wq_b.scale"] = _dense(idx.wq_b)
        plain["attn.indexer.weights_proj.weight"] = idx.weights_proj.weight
        if getattr(idx, "wk", None) is not None:
            plain["attn.indexer.wk.weight"] = idx.wk.weight
            plain["attn.indexer.k_norm.weight"] = idx.k_norm.weight
    eng = getattr(block, "engram", None)
    if eng is not None:
        plain["engram.q_weight"], plain["engram.k_weight"] = eng.q_weight, eng.k_weight
    w.update({k: t.detach() for k, t in plain.items()})
    wo_a, wo_a_s = _dense(a.wo_a)
    return w, {f"{prefix}attn.wo_a.weight": wo_a, f"{prefix}attn.wo_a.scale": wo_a_s}


@torch.no_grad()
def check_block(block, inter: int, tp: int) -> None:
    """What ``block_tensors`` and ``expert_packs`` refuse in ``block`` (ATOM's V4.1 ``Block``: a target layer or a
    draft stage, routed experts of ``inter`` features at TP ``tp``), raised as they would (``MonoUnsupported``) but
    before anything is packed: every FP8 linear they read, the routed experts' layout and padding codes
    (``check_experts``). The install's check (``hi_runner.check``)."""
    a, f = block.attn, block.ffn
    lins = [a.wqkv_a, a.wq_b, a.wo_b, a.wo_a]
    lins += [f.shared_experts.gate_up_proj, f.shared_experts.w2]
    idx = getattr(a, "indexer", None)
    if idx is not None:
        lins.append(idx.wq_b)
    check_linears(*lins)
    check_experts(f.experts, inter, tp)


@torch.no_grad()
def engram_tensors(engram) -> dict:
    """ATOM's ``EngramOp`` -> its ``wkv`` as ``engram_io.pack_kv`` reads it (``engram.wkv.{weight,scale}``)."""
    wt, sc = _dense(engram.wkv)
    return {"engram.wkv.weight": wt, "engram.wkv.scale": sc}


@torch.no_grad()
def ctx_tensors(draft, stages: int) -> dict:
    """ATOM's DSpark draft model -> the context write's tensors as ``draft.pack`` reads them (main_proj whole,
    replicated as ATOM holds it, the stages' wkv rows of ``wqkv_a``, the norms).
    """
    out = {}
    out["mtp.0.main_proj.weight"], out["mtp.0.main_proj.scale"] = _dense(
        draft.mtp[0].main_proj
    )
    out["mtp.0.main_norm.weight"] = draft.mtp[0].main_norm.weight.detach()
    for s in range(stages):
        a = draft.mtp[s].attn
        q_lora = a.wqkv_a.output_sizes[0]
        rows = (q_lora, sum(a.wqkv_a.output_sizes))
        out[f"mtp.{s}.attn.wkv.weight"], out[f"mtp.{s}.attn.wkv.scale"] = _dense(
            a.wqkv_a, rows
        )
        out[f"mtp.{s}.attn.kv_norm.weight"] = a.kv_norm.weight.detach()
    return out


@torch.no_grad()
def _expert_layout(experts, inter: int, tp: int) -> tuple:
    """``experts``' (ATOM's ``FusedMoE``, the gfx950 FlyDSL layout) shapes, checked: (E, 2 pad, K / 2, hidden, pad,
    live, gate / up's K groups, w13 codes, w2 codes, down's padded K groups)."""
    w13, w2 = experts.w13_weight, experts.w2_weight
    if (
        w13.dim() != 3
        or w2.dim() != 3
        or not (
            getattr(w13, "is_shuffled", False) and getattr(w2, "is_shuffled", False)
        )
    ):
        raise MonoUnsupported(
            f"V4.1 mono hi weights: the routed experts are not in the gfx950 FlyDSL layout "
            f"(w13 {tuple(w13.shape)}, w2 {tuple(w2.shape)}, shuffled {getattr(w13, 'is_shuffled', False)})"
        )
    e, n2, kb = w13.shape  # [E, 2 pad, K / 2]
    hidden, pad = w2.shape[1], n2 // 2
    live = inter // tp
    groups = kb * 2 // _GROUP  # gate / up's K groups (K = hidden)
    s13 = experts.w13_weight_scale.detach().view(torch.uint8)
    s2 = experts.w2_weight_scale.detach().view(torch.uint8)
    g2 = s2.numel() // (e * hidden)  # down's K groups, padded to a multiple of 8
    if (w2.shape[2] * 2 != pad or hidden != kb * 2 or live > pad or s13.numel() != e * n2 * groups
            or s2.numel() != e * hidden * g2 or g2 % 8 or g2 < pad // _GROUP or groups % 8):  # fmt: skip
        raise MonoUnsupported(
            f"V4.1 mono hi weights: routed experts w13 {tuple(w13.shape)} / {tuple(s13.shape)}, w2 {tuple(w2.shape)} / "
            f"{tuple(s2.shape)} are not inter {inter} / tp {tp}"
        )
    return e, n2, kb, hidden, pad, live, groups, s13, s2, g2


def _unshuffled_codes(
    experts, inter: int, tp: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The codes in row-major order: gate / up [E, 2, pad, groups] (aiter's ``shuffle_scale``: ``[E, n1, k1, kl,
    nl, kp, g]`` of the rows ``[E, g, n1, nl]`` and the groups ``[k1, kp, kl]``) and down [E, H, padded groups]
    (``[E, n1, k1, kl, nl, kp, np]`` of the rows ``[E, n1, np, nl]``)."""
    e, _n2, _kb, hidden, pad, _live, groups, s13, s2, g2 = _expert_layout(
        experts, inter, tp
    )
    c13 = (
        s13.reshape(e, pad // 16, groups // 8, 4, 16, 2, 2)
        .permute(0, 6, 1, 4, 2, 5, 3)
        .reshape(e, 2, pad, groups)
    )
    c2 = (
        s2.reshape(e, hidden // 32, g2 // 8, 4, 16, 2, 2)
        .permute(0, 1, 6, 4, 2, 5, 3)
        .reshape(e, hidden, g2)
    )
    return c13, c2


@torch.no_grad()
def check_experts(experts, inter: int, tp: int) -> None:
    """``experts``' layout (``_expert_layout``) and padding codes, refused (``MonoUnsupported``) before anything is
    packed. The kernels skip the padding's weights (rows past ``inter / tp`` of each half, down's K past it) but
    apply their codes, to zero weights: any finite code gives 0, so a padding code 255 (E8M0 NaN) is refused. This
    rank's shard's codes: the install agrees on the verdict over the ranks (``hi_dispatch.install_hi``).
    """
    _, _, _, _, pad, live, _, _, _, _ = _expert_layout(experts, inter, tp)
    c13, c2 = _unshuffled_codes(experts, inter, tp)
    if bool((c13[:, :, live:] == 255).any()) or bool(
        (c2[:, :, live // _GROUP : pad // _GROUP] == 255).any()
    ):
        raise MonoUnsupported(
            "V4.1 mono hi weights: a routed expert's padding scale code is 255 (E8M0 NaN)"
        )


@torch.no_grad()
def expert_packs(
    experts, inter: int, tp: int, name: str = "moe"
) -> dict[str, torch.Tensor]:
    """``experts`` (ATOM's ``FusedMoE`` at TP ``tp``, without EP: every expert's slice of this rank's ``inter / tp``
    features, zero-padded to the 128 step) -> the routed experts' args, all ATOM's own bytes (flat views):

    - ``<name>.ug.w`` / ``<name>.down.w``: ``w13_weight`` / ``w2_weight`` (aiter's gfx950 lane-level order, ``[E, n0,
      gate | up, k0, j, r, b]`` / ``[E, n0, k0, j, r, b]``, is the kernels' unit order);
    - ``<name>.ug.ws`` / ``<name>.down.ws``: ``w13_weight_scale`` / ``w2_weight_scale`` in aiter's ``shuffle_scale``
      order, which the kernels read in place (``MoEExperts(atom_scales=True)``).

    Their padding codes are checked before (``check_experts``).
    """
    _, _, _, _, _, _, _, s13, s2, _ = _expert_layout(experts, inter, tp)
    out = {
        f"{name}.ug.w": experts.w13_weight.detach().view(torch.uint8).reshape(-1),
        f"{name}.ug.ws": s13.reshape(-1),
        f"{name}.down.w": experts.w2_weight.detach().view(torch.uint8).reshape(-1),
        f"{name}.down.ws": s2.reshape(-1),
    }
    for t in out.values():
        t._atom_alias = True
    return out


# packed key -> (the ATOM block's parameter or buffer, its rows): the plain bf16 / f32 tensors the hi runtime packs
# as ATOM keeps them (``layer_impl.pack``: the checkpoint's tensors, converted to the dtype ATOM holds them in)
PLAIN = {
    "router.w": ("ffn.gate.weight", None),
    "router.bias": ("ffn.gate.e_score_correction_bias", None),
    "hca.fn": ("hc_attn_fn", None),
    "hca.scale": ("hc_attn_scale", None),
    "hca.base": ("hc_attn_base", None),
    "hcf.fn": ("hc_ffn_fn", None),
    "hcf.scale": ("hc_ffn_scale", None),
    "hcf.base": ("hc_ffn_base", None),
    "attn_norm.g": ("attn_norm.weight", None),
    "ffn_norm.g": ("ffn_norm.weight", None),
    "q_norm.g": ("attn.q_norm.weight", None),
    "kv_norm.g": ("attn.kv_norm.weight", None),
    "attn.sink": ("attn.attn_sink", None),
    "eng.gw": ("engram.gate_weight", None),
    "idx.wproj": ("attn.indexer.weights_proj.weight", None),
    # the compressor's wkv and (ratio 2) wgate: ATOM's fused wkv_gate, wkv rows first
    "comp.wkv": ("attn.compressor.wkv_gate.weight", (0, 512)),
    "comp.wgate": ("attn.compressor.wkv_gate.weight", (512, 1024)),
}


def alias_plain(block, packed: dict) -> int:
    """``packed``'s plain tensors (``PLAIN``) replaced by ``block``'s (a target layer or a draft stage of ATOM's model)
    where those hold the same bytes (contiguous, the same dtype and size); the others keep their packs. -> the number
    aliased."""
    tensors = {**dict(block.named_parameters()), **dict(block.named_buffers())}
    n = 0
    for key, (name, rows) in PLAIN.items():
        ours, theirs = packed.get(key), tensors.get(name)
        if ours is None or theirs is None:
            continue
        theirs = theirs.data if rows is None else theirs.data[rows[0] : rows[1]]
        if (
            theirs.dtype != ours.dtype
            or theirs.numel() != ours.numel()
            or not theirs.is_contiguous()
            or theirs.device != ours.device
            or not torch.equal(theirs.reshape(ours.shape), ours)
        ):
            continue
        view = theirs.reshape(ours.shape)
        view._atom_alias = True
        packed[key] = view
        n += 1
    return n


def rope_args(rope) -> dict:
    """The pointer-table entries of ATOM's ``RotaryEmbedding`` ``rope``: its bf16 cos / sin caches [positions, 32]
    (``rope`` / ``rope.sin``: the hi device code's ``rope_cs``)."""
    cos, sin = rope.cos_cache, rope.sin_cache
    if (
        cos.dtype != torch.bfloat16
        or cos.shape != sin.shape
        or cos.shape[-1] != 32
        or not (cos.is_contiguous() and sin.is_contiguous())
    ):
        raise MonoUnsupported(
            f"V4.1 mono hi: RoPE caches {cos.dtype} {tuple(cos.shape)} / {tuple(sin.shape)}, not bf16 [n, 32]"
        )
    return {"rope": cos, "rope.sin": sin}

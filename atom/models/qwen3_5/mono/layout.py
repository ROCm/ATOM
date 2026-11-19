# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host-side shapes of the Qwen3.5 MonoKernel: execution constants, per-stage task counts and the
mailbox layout."""

from atom.models.qwen3_5.mono.config import experts_mxfp4, fp4_mids, proj_kind
from atom.mono.plan.check import RegionDecl
from atom.mono.plan.execution import BLOCKS
from atom.mono.plan.layout import DIAG_WORDS
from atom.mono.plan.trace import Space

PROJ_TILE = 16  # rows of one MFMA row group
PROJ_GROUPS = 2  # row groups per proj task (four waves each split K)
ROW_TILE = 32  # rows per out_proj / attention peer-reduce tile
ROUTER_TILE = 8  # router rows per task (a part of a 16-row MFMA group)
UG8 = 8  # intermediates one up/gate task owns (8 gate + 8 up rows)
MOE_MISC = 16  # misc words before the MoE input's FP8 block scales
BS = 16  # tokens per KV-cache block (ATOM's kv_cache_block_size)


def shape_dims(cfg) -> dict:
    """The shape arguments layout(), stage_tasks() and the builder must share identically."""
    return {
        "hidden": cfg.hidden,
        "hk": cfg.hk,
        "hv": cfg.hv,
        "k_dim": cfg.linear_k_dim,
        "v_dim": cfg.linear_v_dim,
        "heads": cfg.local_heads,
        "head_dim": cfg.head_dim,
        "n_experts": cfg.n_experts,
        "top_k": cfg.top_k,
        "inter": cfg.local_inter,
        "weights": cfg.weights,
        "qk_fusion": cfg.qk_fusion,
        "pa_flydsl": cfg.pa_flydsl,
    }


def router_rows(n_experts: int) -> int:
    """Router GEMV rows: the experts, the shared-expert gate, padded to a 16-row group."""
    return -(-(n_experts + 1) // 16) * 16


def router_spt(S: int, n_experts: int) -> int:
    """Samples per router task: the fewest (at most 8) that fit the grid in one round."""
    n_router = router_rows(n_experts) // ROUTER_TILE
    spt = 1
    while S * n_router > BLOCKS * spt and spt < ROUTER_TILE:
        spt *= 2
    return spt


def dn_tile(S: int, hidden: int) -> int:
    """Hidden rows per expert-down / FFN peer-reduce task: 32 at S = 1, else about one task per CTA;
    a multiple of 16 so a tile never straddles ``reduce_rows``'s row groups."""
    return 32 if S == 1 else max(16, -(-(hidden // BLOCKS) // 16) * 16)


# ATOM's FlyDSL plan shape at head_dim 256 (``flydsl_plan_shape``; checked by the runner)
FLYDSL_MAX_PARTS = 64
FLYDSL_BUDGET = 256


def flydsl_capacity(seqs: int) -> int:
    """``plan_pa_decode``'s slot capacity for ``seqs`` sequences."""
    return min(FLYDSL_MAX_PARTS * seqs, max(seqs, FLYDSL_BUDGET))


def pa_splits(seqs: int) -> int:
    """KV splits of the full-attention decode for ``seqs`` sequences, as ATOM's ``dense_decode_splits``:
    each split rounds its partial output to bf16, so the kernel must take the same ones.
    """
    rec = min(-(-2 * BLOCKS // seqs), 8)
    boost = 1 << (min(32, -(-128 // seqs)).bit_length() - 1)
    return min(32, max(rec, boost))


# How baseline ATOM's BF16 GEMMs (aiter tuned_gemm picks per (shape, M) on gfx950) round partial sums;
# ``tests/test_qwen3_5_mono_layout.py`` checks the table against the installed aiter.
# ``("halves",)``: bf16(bf16(sum K half 0) + bf16(sum K half 1)) per 256-wide K tile (two K-waves).
# ``("splitk", n)``: n K ranges, each rounded to bf16 and added (the kernel takes range order).
# ``("single",)``: one fp32 sum, one rounding.
_SPLITK_512x4096 = {
    1: ("single",),
    2: ("splitk", 8),
    3: ("splitk", 4),
    4: ("splitk", 4),
}
_SPLITK_512x4096.update({m: ("splitk", 8) for m in range(5, 9)})
_SPLITK_512x4096.update({m: ("splitk", 4) for m in range(9, 17)})
BF16_GEMM_RULES = {
    # o_proj / out_proj of the MXFP4 checkpoint (N 4096, K 2048 a rank)
    "out": {m: ("halves",) if m <= 8 or m == 12 else ("single",) for m in range(1, 17)},
    # the MXFP4 checkpoint's shared gate_up (N 512, K 4096)
    "shared_ug": _SPLITK_512x4096,
    # the router and shared-expert gate merged (N 513 padded to 528, K 4096)
    "router": {m: ("single",) for m in range(1, 17)},
}


def bf16_gemms(cfg) -> list[tuple[str, int, int, str | None]]:
    """Baseline ATOM's BF16 decode GEMMs for a ``Q35Config`` rank, as (name, N, K, ``BF16_GEMM_RULES``
    key or None for one fp32 pass)."""
    gemms = [("router", router_rows(cfg.n_experts), cfg.hidden, "router")]
    if cfg.proj_kind != "ptpc":
        gemms.append(("in_proj_ba", 2 * cfg.hv, cfg.hidden, None))
    if cfg.weights == "mxfp4":
        sh = cfg.shared_inter // cfg.tp
        gemms += [
            ("in_proj_qkvz", cfg.conv_dim + cfg.value_dim, cfg.hidden, None),
            (
                "qkv_proj",
                cfg.local_heads * 2 * cfg.head_dim + 2 * cfg.head_dim,
                cfg.hidden,
                None,
            ),
            ("out_proj", cfg.hidden, cfg.value_dim, "out"),
            ("o_proj", cfg.hidden, cfg.local_heads * cfg.head_dim, "out"),
            ("shared gate_up", 2 * sh, cfg.hidden, "shared_ug"),
            ("shared down", cfg.hidden, sh, None),
        ]
    return gemms


def ptpc_gemms(cfg) -> list[tuple[str, int, int, str]]:
    """Baseline ATOM's per-token FP8 GEMMs for a ``Q35Config`` rank, as (name, N padded to 128, K,
    epilogue order: "w_x" ``(acc * w_scale) * x_scale``, "x_w" ``(acc * x_scale) * w_scale``).
    """
    if cfg.proj_kind != "ptpc":
        return []

    def r128(n):
        return -(-n // 128) * 128

    return [
        (
            "in_proj_qkvzba",
            r128(cfg.conv_dim + cfg.value_dim + 2 * cfg.hv),
            cfg.hidden,
            "w_x",
        ),
        (
            "qkv_proj",
            r128(cfg.local_heads * 2 * cfg.head_dim + 2 * cfg.head_dim),
            cfg.hidden,
            "w_x",
        ),
        ("out_proj", cfg.hidden, cfg.value_dim, "x_w"),
        ("o_proj", cfg.hidden, cfg.local_heads * cfg.head_dim, "x_w"),
    ]


def aiter_ptpc_order(pick: dict | None) -> str | tuple:
    """The epilogue order of an aiter a8w8 bpreshuffle config ``pick`` (None: untuned, its default CK
    kernel): "w_x" (CK), "x_w" (FlyDSL's 16x32x1024 without split-K), else ``("unknown", pick)``.
    """
    if not pick:
        return "w_x"
    name = str(pick.get("kernelName") or "")
    if pick.get("libtype") == "ck" and int(pick.get("splitK") or 0) == 0:
        return "w_x"
    if (
        pick.get("libtype") == "flydsl"
        and name.startswith("flydsl_bpreshuflle_16x32x1024_")
        and int(pick.get("splitK") or 0) == 0
    ):
        return "x_w"
    return ("unknown", pick.get("libtype"), name, pick.get("splitK"))


def aiter_bf16_rule(pick: dict | None) -> tuple:
    """The partial-sum rounding (a ``BF16_GEMM_RULES`` value) of an aiter tuned_gemm config ``pick``
    (None: untuned, torch); ``("unknown", name)`` for an unrecognized FlyDSL pick."""
    pick = pick or {}
    name = pick.get("kernelName") or ""
    if pick.get("libtype") == "flydsl":
        params = name.split("_")
        try:
            tile = next(p for p in params if p.startswith("t") and "x" in p)
            ks = next(p for p in params if p.startswith("ks"))
            waves = next(p for p in params if p.startswith("w") and "x" in p)
            k_waves = int(waves[1:].split("x")[2])
            split_k = 1 if ks == "ks1" else int(pick["splitK"])
        except (StopIteration, IndexError, KeyError, ValueError):
            return ("unknown", name)
        if split_k > 1 and k_waves == 1:
            return ("splitk", split_k)
        if split_k == 1 and k_waves == 2 and tile.split("x")[2] == "256":
            return ("halves",)
        if split_k == 1 and k_waves == 1:
            return ("single",)
        return ("unknown", name)
    return ("single",)


# from XIN_MIN rows the FP8 checkpoint's input norm + quant runs once (stage ``xin``), not per projection task
XIN_MIN = 4

# ATOM_MONO_TIMELINE stamps per CTA: kernel entry, each stage's entry, kernel end
TL_POINTS = 12


def xin_stage(S: int, weights: str) -> bool:
    """Whether the layer quantizes its input once (stage ``xin``) for the FP8 projections."""
    return weights == "fp8" and S >= XIN_MIN


def bf16_gemm_rule(name: str, m: int) -> tuple:
    """The partial-sum rounding of baseline ATOM's BF16 GEMM ``name`` (``BF16_GEMM_RULES``) at M ``m``."""
    return BF16_GEMM_RULES[name][m]


def gdn_dims(dims: dict) -> dict:
    """Derived GDN shard sizes from the build dims."""
    key_dim = dims["hk"] * dims["k_dim"]
    value_dim = dims["hv"] * dims["v_dim"]
    return {
        "key_dim": key_dim,
        "value_dim": value_dim,
        "conv_dim": 2 * key_dim + value_dim,
    }


def _align(n, a=256):
    return (n + a - 1) // a * a


def out_k(full: bool, dims: dict) -> int:
    """K of out_proj: the mixer's output width (GDN value dim / heads x head dim)."""
    return dims["heads"] * dims["head_dim"] if full else gdn_dims(dims)["value_dim"]


def layout(
    S: int,
    npes: int,
    full: bool,
    *,
    hidden,
    hk,
    hv,
    k_dim,
    v_dim,
    heads,
    head_dim,
    n_experts,
    top_k,
    inter,
    weights,
    qk_fusion=False,
    pa_flydsl=False,
    tokens_per_seq=1,
):
    """Byte offsets of the per-rank scratch and of the symmetric buffer. Every mailbox holds
    ``(value, tag)`` int32 pairs (8 bytes); bf16 mailboxes pack two elements per pair.
    """
    parts = (
        flydsl_capacity(S // tokens_per_seq) * tokens_per_seq
        if pa_flydsl
        else S * pa_splits(S // tokens_per_seq)
    )
    dims = {
        "hk": hk,
        "hv": hv,
        "k_dim": k_dim,
        "v_dim": v_dim,
        "heads": heads,
        "head_dim": head_dim,
    }
    pr = 8
    slots = top_k + 1
    if full:
        mixer = [
            ("qg", S * heads * 2 * head_dim * pr),  # q | gate per head (bf16 values)
            ("kvr", S * 2 * head_dim * pr),  # raw k then v of this rank's kv head
            ("q", S * heads * head_dim * pr),  # normed, rotated query
            (
                "kvnew",
                S * (2 * head_dim + 2) * pr,
            ),  # this token's K / V FP8 codes + their scales
            # per split: bf16 output, running max and exp sum
            ("sp_out", parts * heads * head_dim // 2 * pr),
            ("sp_ml", parts * heads * 2 * pr),
        ]
    else:
        g = gdn_dims(dims)
        mixer = [
            ("conv", S * g["conv_dim"] * pr),  # conv(qkv) output, one f32 per pair
            ("z", S * g["value_dim"] * pr),
            ("ba", S * 2 * hv * pr),  # b then a, per v-head
        ]
    if full and qk_fusion and weights == "fp8":  # gated, f32
        out_in = [("gated", S * out_k(full, dims) * pr)]
    elif full or proj_kind(weights) != "block":  # the mixer's output, bf16 pairs
        out_in = [("gated", S * out_k(full, dims) // 2 * pr)]
    else:  # FP8 checkpoint's GDN output: FP8 codes and per-128 scales
        out_in = [
            ("gq", S * out_k(full, dims) // 4 * pr),
            ("gqs", S * out_k(full, dims) // 128 * pr),
        ]
    # stage xin's output: FP8 codes and per-128 scales
    xin = (
        [("xinq", S * hidden // 4 * pr), ("xins", S * hidden // 128 * pr)]
        if xin_stage(S, weights)
        else []
    )
    items = (
        xin
        + mixer
        + out_in
        + [
            # the f32 post-mixer stream
            ("a", S * hidden * pr),
            (
                "scores",
                S * router_rows(n_experts) * pr,
            ),  # router logits (bf16-rounded) + shared gate
            ("xq", S * hidden // 4 * pr),  # MoE input, four FP8 bytes per pair
            ("xqs", S * hidden // 128 * pr),
            ("sel", S * slots * pr),
            ("prob", S * slots * pr),
            # silu(gate) * up as f32, or (fp4_mids) MXFP4 codes then E8M0 scales
            (
                "mid",
                (
                    (S * slots * inter // 8 + S * slots * inter // 32)
                    if fp4_mids(weights, S)
                    else S * slots * inter
                )
                * pr,
            ),
            # MXFP4 experts: MoE input as MXFP4 (codes then E8M0 scales); "mxfp4" also as bf16 for the
            # BF16 shared expert
            *([("x2b", S * hidden // 2 * pr)] if weights == "mxfp4" else []),
            *(
                [("xb", (S * hidden // 8 + S * hidden // 32) * pr)]
                if experts_mxfp4(weights)
                else []
            ),
        ]
    )
    # a debug build's wait records, one per CTA
    items = items + [("diag", BLOCKS * DIAG_WORDS * 4)]
    off, scratch = 0, {}
    for name, size in items:
        scratch[name] = off
        off += _align(size)
    scratch["_bytes"] = off
    part = npes * S * hidden * pr
    sym = {"attn": 0, "ffn": part, "_bytes": 2 * part}
    return scratch, sym


def mailbox_regions(S: int, full: bool, weights: str, npes: int) -> list[RegionDecl]:
    """Every mailbox region, its space and the stage that puts it (checked by ``contract.py``).
    ``attn`` / ``ffn`` are the TP peer-reduce exchanges, absent at one rank."""
    r = [
        RegionDecl("a", Space.SCRATCH, "out"),
        RegionDecl("scores", Space.SCRATCH, "router"),
    ]
    if xin_stage(S, weights):
        r += [RegionDecl(n, Space.SCRATCH, "xin") for n in ("xinq", "xins")]
    if full:
        r += [RegionDecl(n, Space.SCRATCH, "proj") for n in ("qg", "kvr")]
        r += [RegionDecl(n, Space.SCRATCH, "qk") for n in ("q", "kvnew")]
        r += [RegionDecl(n, Space.SCRATCH, "split") for n in ("sp_out", "sp_ml")]
        r += [RegionDecl("gated", Space.SCRATCH, "merge")]
    else:
        r += [RegionDecl(n, Space.SCRATCH, "proj") for n in ("conv", "z")]
        r += [RegionDecl("ba", Space.SCRATCH, "proj_ba")]
        gdn_out = ("gated",) if proj_kind(weights) != "block" else ("gq", "gqs")
        r += [RegionDecl(n, Space.SCRATCH, "gdn") for n in gdn_out]
    moe_in = {"fp8": ("xq", "xqs"), "mxfp4": ("x2b", "xb"), "mxfp4_attnfp8": ("xb",)}[
        weights
    ]
    r += [RegionDecl(n, Space.SCRATCH, "router") for n in moe_in]
    r += [RegionDecl(n, Space.SCRATCH, "ug") for n in ("mid", "sel", "prob")]
    if npes > 1:
        r += [
            RegionDecl("attn", Space.PEER, "out", exchange=True),
            RegionDecl("ffn", Space.PEER, "down", exchange=True),
        ]
    return r


def stage_tasks(
    S: int,
    full: bool,
    *,
    hidden,
    hk,
    hv,
    k_dim,
    v_dim,
    heads,
    head_dim,
    n_experts,
    top_k,
    inter,
    weights="fp8",
    qk_fusion=False,
    pa_flydsl=False,
    tokens_per_seq=1,
):
    """[(stage name, task count)] in execution order."""
    xin = [("xin", S)] if xin_stage(S, weights) else []
    if full:
        mixer = [
            ("proj", (heads * 2 * head_dim + 2 * head_dim) // PROJ_TILE // PROJ_GROUPS),
            ("qk", S * (heads + 1)),
            (
                "split",
                (
                    flydsl_capacity(S // tokens_per_seq)
                    if pa_flydsl
                    else S * pa_splits(S // tokens_per_seq)
                ),
            ),
            ("merge", S * heads),
        ]
    else:
        g = gdn_dims({"hk": hk, "hv": hv, "k_dim": k_dim, "v_dim": v_dim})
        mixer = [
            ("proj", (g["conv_dim"] + g["value_dim"]) // PROJ_TILE // PROJ_GROUPS),
            ("proj_ba", -(-(2 * hv // PROJ_TILE) // PROJ_GROUPS)),
            ("gdn", S // tokens_per_seq * hv),
        ]
    return (
        xin
        + mixer
        + [
            ("out", hidden // ROW_TILE),
            # SPT samples per task; an odd S leaves the last task a sample short (masked)
            (
                "router",
                -(-S // router_spt(S, n_experts))
                * (router_rows(n_experts) // ROUTER_TILE),
            ),
            # a tile per (routed slot, UG8 intermediates) plus the shared expert; fp4_mids: 32-mid blocks
            (
                "ug",
                (
                    inter // 32 + S * top_k * (inter // 32)
                    if fp4_mids(weights, S)
                    else S * top_k * (inter // UG8)
                ),
            ),
            ("down", hidden // dn_tile(S, hidden)),
        ]
    )

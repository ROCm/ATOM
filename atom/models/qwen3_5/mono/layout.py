# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host-side shapes of the Qwen3.5 MonoKernel: execution constants, per-stage task counts and the
mailbox layout."""

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
    """Hidden rows per expert-down / FFN peer-reduce task: 32 at S = 1, else about one task per CTA,
    a multiple of 16 so a tile never straddles the row groups ``reduce_rows`` emits."""
    return 32 if S == 1 else max(16, -(-(hidden // BLOCKS) // 16) * 16)


def pa_splits(seqs: int) -> int:
    """KV splits of the full-attention decode for ``seqs`` sequences: baseline ATOM's
    ``dense_decode_splits`` (aiter's ``get_recommended_splits`` on BLOCKS CUs, occupancy 2, one KV head
    a rank, raised to a power-of-two share of 128 programs, at most 32). Its splits are rounding points
    (each rounds its partial output to bf16), so the kernel takes the same ones."""
    rec = min(-(-2 * BLOCKS // seqs), 8)
    boost = 1 << (min(32, -(-128 // seqs)).bit_length() - 1)
    return min(32, max(rec, boost))


# Baseline ATOM's BF16 GEMMs are aiter tuned_gemm (tgemm.mm) picks per (shape, M), and some picks round
# partial sums: aiter 45f966883's tuned configs on gfx950 (``tests/test_qwen3_5_mono_layout.py`` checks the
# table against the installed aiter). ``("halves",)``: the FlyDSL hgemm with two K-waves (t16x16x256,
# w1x1x2), each 256-wide K tile's first and second 128 summed by one wave each, both rounded to bf16 and
# added: bf16(bf16(sum half 0) + bf16(sum half 1)). ``("splitk", n)``: the FlyDSL split-K hgemm, K in n
# equal ranges, each range's sum rounded to bf16 and added into the bf16 output by atomics (the order
# is not deterministic; the kernel takes range order). ``("single",)``: one fp32 sum, one rounding
# (torch / hipBLASLt, skinny, Triton: Triton's split-K reduces fp32 partials).
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
    # the router and the shared-expert gate: one merged GEMM (N 513, K 4096; both checkpoints), torch
    "router": {m: ("single",) for m in range(1, 17)},
}


# Past XIN_MIN rows the FP8 checkpoint's input norm + FP8 quant runs once a layer (stage ``xin``, a task per row)
# instead of in every projection task (each restaged every row: the cost grew with the rows)
XIN_MIN = 4

# ATOM_MONO_TIMELINE: a CTA's stamps (atom.mono.device.stamps): kernel entry, each stage's entry, kernel end
TL_POINTS = 12


def xin_stage(S: int, weights: str) -> bool:
    """Whether the layer quantizes its input once (stage ``xin``) for the FP8 projections."""
    return weights == "fp8" and S >= XIN_MIN


def baseline_m(S: int, tokens_per_seq: int = 1) -> int:
    """The M of baseline ATOM's decode GEMMs for an S-row step: CUDA graphs pad a plain decode batch to
    its capture size (1 .. 8, then multiples of 16); an MTP verify's batch (at most 4 requests) is
    captured as is."""
    if S <= 8 or tokens_per_seq > 1:
        return S
    return -(-S // 16) * 16


def bf16_gemm_rule(name: str, S: int, tokens_per_seq: int = 1) -> tuple:
    """The partial-sum rounding of baseline ATOM's BF16 GEMM ``name`` (``BF16_GEMM_RULES``) at an S-row step."""
    return BF16_GEMM_RULES[name][baseline_m(S, tokens_per_seq)]


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
    tokens_per_seq=1,
):
    """Byte offsets of the per-rank scratch and of the symmetric buffer. Every mailbox holds
    ``(value, tag)`` int32 pairs (8 bytes); bf16 mailboxes pack two elements per pair.
    """
    ns = pa_splits(S // tokens_per_seq)
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
            ("sp_out", S * ns * heads * head_dim // 2 * pr),  # each split's bf16 output
            ("sp_ml", S * ns * heads * 2 * pr),  # and its running max, exp sum
        ]
    else:
        g = gdn_dims(dims)
        mixer = [
            ("conv", S * g["conv_dim"] * pr),  # conv(qkv) output, one f32 per pair
            ("z", S * g["value_dim"] * pr),
            ("ba", S * 2 * hv * pr),  # b then a, per v-head
        ]
    if (
        full and qk_fusion and weights == "fp8"
    ):  # gated, f32 (one a pair): quantized from f32
        out_in = [("gated", S * out_k(full, dims) * pr)]
    elif full or weights == "mxfp4":  # the mixer's output, bf16 pairs
        out_in = [("gated", S * out_k(full, dims) // 2 * pr)]
    else:  # the FP8 checkpoint's GDN output: FP8 (four bytes a pair) and its per-128 scales
        out_in = [
            ("gq", S * out_k(full, dims) // 4 * pr),
            ("gqs", S * out_k(full, dims) // 128 * pr),
        ]
    # stage xin's output: the normed input as FP8 (four bytes a pair) and its per-128 scales
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
            # the f32 post-mixer stream (one value a pair): the MoE input norm's operand
            ("a", S * hidden * pr),
            (
                "scores",
                S * router_rows(n_experts) * pr,
            ),  # router logits (bf16-rounded) + shared gate
            ("xq", S * hidden // 4 * pr),  # MoE input, four FP8 bytes per pair
            ("xqs", S * hidden // 128 * pr),
            ("sel", S * slots * pr),
            ("prob", S * slots * pr),
            ("mid", S * slots * inter * pr),
            # MXFP4 checkpoint: the MoE input as bf16 pairs, normed (the shared expert's) and per-32 MXFP4
            # quantize-dequantized (the routed experts')
            *(
                [("x2b", S * hidden // 2 * pr), ("xb", S * hidden // 2 * pr)]
                if weights == "mxfp4"
                else []
            ),
        ]
    )
    # a debug build's wait records (one a CTA, atom.mono.device.sync), zeroed with the mailboxes at a
    # step's start
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
    """The layer kernel's hand-offs: every mailbox region, its space and the stage that puts it (checked on a
    traced build, ``contract.py``). ``attn`` / ``ffn`` are the TP peer reduces (each rank's partials, polled
    by the putting stage: an exchange), absent at one rank."""
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
        gdn_out = ("gated",) if weights == "mxfp4" else ("gq", "gqs")
        r += [RegionDecl(n, Space.SCRATCH, "gdn") for n in gdn_out]
    moe_in = ("x2b", "xb") if weights == "mxfp4" else ("xq", "xqs")
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
    tokens_per_seq=1,
):
    """[(stage name, task count)] in execution order (``tokens_per_seq``: MTP verify, a GDN task per sequence)."""
    xin = [("xin", S)] if xin_stage(S, weights) else []
    if full:
        mixer = [
            ("proj", (heads * 2 * head_dim + 2 * head_dim) // PROJ_TILE // PROJ_GROUPS),
            ("qk", S * (heads + 1)),
            ("split", S * pa_splits(S // tokens_per_seq)),
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
            # one tile per (routed slot, 8 intermediates); the first inter / UG8 also carry the shared expert
            ("ug", S * top_k * (inter // UG8)),
            ("down", hidden // dn_tile(S, hidden)),
        ]
    )

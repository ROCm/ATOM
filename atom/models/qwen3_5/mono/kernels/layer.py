# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen3.5-MoE decoder layer (Gated DeltaNet or gated full attention) in ONE persistent launch per rank.

One launch of ``grid = 256 CTAs x 512 threads`` runs the whole layer for this rank's shard, up to 16
samples (rows). GDN::

    input RMSNorm (Gemma 1 + w) -> qkv | z GEMV (+ causal conv1d, SiLU, conv-state shift)
      -> b | a BF16 GEMV -> per (sample, v-head): L2 norms, gates, delta rule on the bf16 state
         (or, verifying drafted tokens, on ReplaySSM's checkpoint and records),
         gated RMSNorm x silu(z) -> out_proj -> attention TP peer reduce + residual   (sym attn)
      -> post-attention RMSNorm -> router (512 experts + the shared-expert gate) + the MoE input
      -> softmax top-10 -> shared + 10 routed expert up/gate + SiLU
      -> expert down + route weighting -> MoE TP peer reduce + residual -> x_out     (sym ffn)

Full attention replaces the mixer: q | gate, k, v GEMV -> Gemma q / k norm + partial RoPE, K / V into
ATOM's paged FP8 cache -> split attention (one wave per query head) -> merge x sigmoid(gate) -> out_proj.

``weights`` is the checkpoint's format: "fp8" (FP8 128 x 128 matrices, W8A8 experts, the shared expert
a slot of the bank) or "mxfp4" (BF16 projections, A4W4 MXFP4 experts, a BF16 shared expert).

Task ``t`` of a stage runs on CTA ``(stage_base + t) % 256``; every CTA walks the stages in order;
hand-offs are tagged ``(value, tag)`` pairs (atom.mono.device.sync), polled until the tag is this launch's
(``layer + 1``; every mailbox zeroed at a step's start).
"""

import enum

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr
from flydsl.expr.typing import Int32, Int64, T

from atom.models.qwen3_5.mono.config import GLUON_VERIFY_TOKENS, MAX_VERIFY_TOKENS
from atom.models.qwen3_5.mono.kernels.attn import attn_stages
from atom.models.qwen3_5.mono.kernels.attn_flydsl import flydsl_lds_words
from atom.models.qwen3_5.mono.kernels.common import common_defs
from atom.models.qwen3_5.mono.kernels.ffn import ffn_stages
from atom.models.qwen3_5.mono.kernels.gdn import gdn_stages
from atom.models.qwen3_5.mono.kernels.helpers import SHARED_SOURCE_KEY
from atom.models.qwen3_5.mono.kernels.mixer import mixer_defs, out_stages
from atom.models.qwen3_5.mono.layout import (
    BF16_GEMM_RULES,
    BS,
    MOE_MISC,
    PROJ_GROUPS,
    PROJ_TILE,
    ROUTER_TILE,
    ROW_TILE,
    TL_POINTS,
    UG8,
    dn_tile,
    gdn_dims,
    layout,
    mailbox_regions,
    out_k,
    pa_splits,
    router_rows,
    stage_tasks,
    xin_stage,
)
from atom.mono.device.ops import rsrc, uniform
from atom.mono.device.stamps import stamp, stamp_begin, stamp_flush
from atom.mono.device.sync import Mailbox
from atom.mono.plan.execution import BLOCKS, THREADS, WAVES
from atom.mono.runtime.debug import region_ids

_CURRENT_STREAM = fx.Stream(None)


def build_qwen35_kernel(
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
    eps,
    rope_dim,
    weights="fp8",
    qk_fusion=False,
    pa_flydsl=False,
    tokens_per_seq=1,
    replay_cap=0,
    debug=False,
    timeline=False,
):
    """Return the ``@flyc.jit`` launcher for one rank's whole layer: full attention when ``full``, else
    Gated DeltaNet. ``heads`` / ``inter`` are this rank's (one K / V head per rank). ``tokens_per_seq`` > 1:
    an MTP verify step, rows ``q * tokens_per_seq + j`` = token j of sequence q (consecutive positions).
    ``replay_cap`` > 0: the verify runs on ReplaySSM GDN state (a checkpoint per request and ``replay_cap``
    records of (k, u, g) a v-head), as ATOM keeps it by default under speculative decoding.
    """
    assert 1 <= S <= 16, "the samples sit in the MFMA's 16 B columns"
    assert (
        k_dim == v_dim == 128
    ), "the recurrence maps 8 K lanes x 4 values and 64 V rows per pass"
    assert hv % hk == 0
    assert n_experts % 64 == 0 and top_k < 16
    assert inter % 128 == 0 and hidden % 128 == 0
    HIDDEN = hidden
    T_SEQ = tokens_per_seq
    # the M baseline ATOM's GEMMs run this step at: the step's rows, which ATOM pads to its capture size
    # (graph replay, and an eager decode alike) before the model sees them. The BF16 GEMMs' rounding
    # and the MXFP4 combine's order follow it
    GEMM_M = S
    assert S % T_SEQ == 0
    REPLAY = replay_cap > 0
    CAP = replay_cap
    assert not REPLAY or (
        not full and CAP >= 2 * T_SEQ
    ), "two verify windows fit the record buffer"
    HK, HV, K_DIM, V_DIM = hk, hv, k_dim, v_dim
    VPK = HV // HK
    g = gdn_dims({"hk": hk, "hv": hv, "k_dim": k_dim, "v_dim": v_dim})
    KEY_DIM, VALUE_DIM, CONV_DIM = g["key_dim"], g["value_dim"], g["conv_dim"]
    assert (CONV_DIM + VALUE_DIM) % (PROJ_TILE * PROJ_GROUPS) == 0 and CONV_DIM % (
        PROJ_TILE * PROJ_GROUPS
    ) == 0
    assert (2 * HV) % (
        PROJ_TILE * PROJ_GROUPS
    ) == 0, "b | a rows must fill whole proj tasks"
    assert VALUE_DIM % 256 == 0 and HIDDEN % ROW_TILE == 0
    N_EXPERTS, TOP_K, INTER = n_experts, top_k, inter
    # "mxfp4": BF16 projections, MXFP4 routed experts on MXFP4 activations (A4W4), a BF16 shared expert
    WFMT = weights
    MX = WFMT == "mxfp4"
    assert WFMT in ("fp8", "mxfp4")
    assert not MX or GEMM_M in BF16_GEMM_RULES["out"], "no BF16 GEMM rule at this M"
    EPS = eps
    MOE_SLOTS = TOP_K + 1
    SHARED_EXPERT = N_EXPERTS
    ROUTER_ROWS = router_rows(N_EXPERTS)
    N_ROUTER = ROUTER_ROWS // ROUTER_TILE
    ID_BITS = max(8, (N_EXPERTS - 1).bit_length())
    ID_MASK = (1 << ID_BITS) - 1
    # the top-k key keeps a logit's high bits and the expert id in its low ID_BITS: exact, since the
    # router logits are bf16 (their f32 low 16 bits zero); ties go to the lower id, as aiter's top-k
    assert ID_BITS <= 16, "the router key would drop logit bits"
    XQ_BLOCKS = HIDDEN // 128
    XIN = xin_stage(S, WFMT)  # the input norm + FP8 quant once a layer (stage xin)
    PUBLISH_BLOCKS = XQ_BLOCKS
    XQ_WAVES = (PUBLISH_BLOCKS + N_ROUTER - 1) // N_ROUTER
    assert XQ_WAVES <= WAVES
    UG_PER_SLOT = INTER // UG8
    N_UG_TASKS = TOP_K * UG_PER_SLOT
    DN_TILE = dn_tile(S, HIDDEN)
    N_DN_TILES = HIDDEN // DN_TILE
    # full attention: this rank's query heads share its one K / V head
    FULL = full
    QKF = bool(qk_fusion) and full
    QKF_O = (
        QKF and weights == "fp8"
    )  # o_proj's input quantized from f32 (its W8A8 checkpoint)
    H, HD, ROPE = heads, head_dim, rope_dim
    KVD = HD
    QG = H * 2 * HD
    SCALE = HD**-0.5
    NSPLIT = pa_splits(S // T_SEQ)
    # ATOM_PA_FLYDSL: decode attention as aiter's FlyDSL paged decode (kernels/attn_flydsl.py)
    PA_FLYDSL = bool(pa_flydsl) and full
    assert (
        not full or PA_FLYDSL or T_SEQ <= GLUON_VERIFY_TOKENS
    ), "past 4 tokens a request the original path decodes with FlyDSL only"
    CPS = 256  # keys of a decode-attention partition (aiter's CONTEXT_PARTITION_SIZE)
    if FULL:
        assert H == WAVES, "split attention runs one query head per wave"
        assert HD % 64 == 0 and HD <= 512 and ROPE % 2 == 0 and ROPE <= HD
        assert BS == 16 and HD == 256 and CPS == 256, "a partition is 16 cache blocks"
        assert NSPLIT <= 64 and T_SEQ <= MAX_VERIFY_TOKENS
    OUT_K = out_k(
        full,
        {
            "hk": hk,
            "hv": hv,
            "k_dim": k_dim,
            "v_dim": v_dim,
            "heads": heads,
            "head_dim": head_dim,
        },
    )
    assert OUT_K % 256 == 0
    W = npes
    G = BLOCKS
    dims = {
        "hidden": hidden,
        "hk": hk,
        "hv": hv,
        "k_dim": k_dim,
        "v_dim": v_dim,
        "heads": heads,
        "head_dim": head_dim,
        "n_experts": n_experts,
        "top_k": top_k,
        "inter": inter,
        "weights": weights,
        "qk_fusion": qk_fusion,
        "pa_flydsl": pa_flydsl,
    }
    SC, SY = layout(S, W, full, tokens_per_seq=T_SEQ, **dims)
    stages = stage_tasks(S, full, tokens_per_seq=T_SEQ, **dims)
    base, acc = {}, 0
    for name, n in stages:
        base[name] = acc % G
        acc += n
    assert (
        len(base) + 2 <= TL_POINTS
    ), "a timeline point a stage, the kernel's entry and end"
    XN = max(
        S * HIDDEN // 2,
        1024,
        (
            (
                16 * (CPS + 1 + CPS // 4 + 2 + HD // 4 + 2)
                + 64
                + T_SEQ * HD // 2
                + 2 * T_SEQ
            )
            if FULL
            else 0
        ),
        flydsl_lds_words(T_SEQ, HD) if PA_FLYDSL else 0,
    )  # split: scores | P' bytes | Q bytes | per-head factors | this launch's K / V bytes, scales
    ON = max(
        # MXFP4: the expert down's routed tile and, after it, the shared expert's
        S * max(ROW_TILE, PROJ_TILE * PROJ_GROUPS, DN_TILE * (2 if MX else 1)),
        V_DIM,
        HD,
        H * HD if FULL else 0,  # a split's output, every head
    )
    MISC = (
        max(
            MOE_MISC + S * XQ_BLOCKS,
            S * MOE_SLOTS * INTER // 128,
            2 * NSPLIT,
            64 if PA_FLYDSL else 0,  # the merge's partition weights
        )
        + 8
    )

    @fx.struct
    class Smem:
        x: fx.Array[
            fx.Float32, XN, 16
        ]  # bf16 / FP8 activations, the recurrence's inputs
        out: fx.Array[fx.Float32, ON, 16]
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        misc: fx.Array[fx.Float32, MISC, 16]
        keys: fx.Array[fx.Int32, S * MOE_SLOTS, 16]
        dnw: fx.Array[fx.Float32, S * MOE_SLOTS, 16]
        tls: fx.Array[fx.Int64, TL_POINTS if timeline else 1, 16]  # ATOM_MONO_TIMELINE

    # a debug build's wait records name their region by id
    REGION_IDS = (
        region_ids(d.name for d in mailbox_regions(S, full, WFMT, npes))
        if debug
        else None
    )
    # the stages read this build's constants from bc; _cache_tag puts them in the JIT cache key,
    # with the digest of the shared helper sources (outside this directory, so outside the key)
    bc = dict(locals())
    _cache_tag = (SHARED_SOURCE_KEY,) + tuple(
        (n, v)
        for n, v in sorted(bc.items())
        if isinstance(v, (bool, int, float, str, tuple, type(None), enum.Enum))
    )

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def qwen35_kernel(
        h_in: Int64,
        x_out: Int64,
        slots_in: Int64,
        slots_out: Int64,
        n_accepted: Int64,
        write_pos: Int64,
        rec_k: Int64,
        rec_u: Int64,
        rec_g: Int64,
        conv_state: Int64,
        ssm_state: Int64,
        positions: Int64,
        context_lens: Int64,
        block_tables: Int64,
        slot_mapping: Int64,
        k_cache: Int64,
        v_cache: Int64,
        k_scale: Int64,
        v_scale: Int64,
        rope_cos: Int64,
        rope_sin: Int64,
        g_qn1: Int64,
        g_kn1: Int64,
        g_in1: Int64,
        w_qkvz: Int64,
        s_qkvz: Int64,
        w_ba: Int64,
        w_conv: Int64,
        a_log: Int64,
        dt_bias: Int64,
        g_norm: Int64,
        w_out: Int64,
        s_out: Int64,
        g_post1: Int64,
        w_r: Int64,
        w_ug: Int64,
        s_ug: Int64,
        w_dn: Int64,
        s_dn: Int64,
        w_sh_ug: Int64,
        w_sh_dn: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        layer: Int32,
        bt_stride: Int32,
        tl: Int64,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % 64
        wave = tid // 64
        lds = fx.SharedAllocator().allocate(Smem).peek()
        xs = lds.x.ptr
        outs = lds.out.ptr
        red = lds.red.ptr
        misc = lds.misc.ptr
        keys = lds.keys.ptr
        dnw = lds.dnw.ptr
        tls = lds.tls.ptr
        # the full-attention projection's matrix comes in the GDN qkvz slot
        w_qkv, s_qkv = w_qkvz, s_qkvz
        # this launch's mailbox: tag layer + 1, every mailbox zeroed at the step's start (the runner's
        # StepMailboxes); ATOM_MONO_DEBUG: every wait bounded, a give-up recorded in the diag records
        mbox = Mailbox(
            layer,
            scratch + fx.Int64(SC["diag"]) if const_expr(debug) else None,
            REGION_IDS,
        )
        stamp_begin(timeline, tls, tid, TL_POINTS)
        r_peers = rsrc(peers)
        # each wave sends to one peer
        pv = fx.Vector(
            bo.buffer_load(r_peers, fx.min(wave, W - 1) * 2, vec_width=2, dtype=T.i32)
        )
        peer_dst = (fx.Int64(uniform(pv[1])) << 32) | fx.Int64(
            fx.Uint32(uniform(pv[0]))
        )

        _ = _cache_tag
        ctx = dict(bc)
        ctx.update(locals())
        ctx.update(common_defs(ctx))
        ctx.update(mixer_defs(ctx))
        if const_expr(FULL):
            attn_stages(ctx)
        else:
            gdn_stages(ctx)
        out_stages(ctx)
        ffn_stages(ctx)
        stamp(timeline, tls, tid, TL_POINTS - 1)
        stamp_flush(timeline, tls, tl, tid, bid, TL_POINTS)

    @flyc.jit
    def launch_qwen35(
        h_in: Int64,
        x_out: Int64,
        slots_in: Int64,
        slots_out: Int64,
        n_accepted: Int64,
        write_pos: Int64,
        rec_k: Int64,
        rec_u: Int64,
        rec_g: Int64,
        conv_state: Int64,
        ssm_state: Int64,
        positions: Int64,
        context_lens: Int64,
        block_tables: Int64,
        slot_mapping: Int64,
        k_cache: Int64,
        v_cache: Int64,
        k_scale: Int64,
        v_scale: Int64,
        rope_cos: Int64,
        rope_sin: Int64,
        g_qn1: Int64,
        g_kn1: Int64,
        g_in1: Int64,
        w_qkvz: Int64,
        s_qkvz: Int64,
        w_ba: Int64,
        w_conv: Int64,
        a_log: Int64,
        dt_bias: Int64,
        g_norm: Int64,
        w_out: Int64,
        s_out: Int64,
        g_post1: Int64,
        w_r: Int64,
        w_ug: Int64,
        s_ug: Int64,
        w_dn: Int64,
        s_dn: Int64,
        w_sh_ug: Int64,
        w_sh_dn: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        layer: Int32,
        bt_stride: Int32,
        tl: Int64,
        stream: fx.Stream = _CURRENT_STREAM,
    ):
        qwen35_kernel(
            h_in,
            x_out,
            slots_in,
            slots_out,
            n_accepted,
            write_pos,
            rec_k,
            rec_u,
            rec_g,
            conv_state,
            ssm_state,
            positions,
            context_lens,
            block_tables,
            slot_mapping,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            rope_cos,
            rope_sin,
            g_qn1,
            g_kn1,
            g_in1,
            w_qkvz,
            s_qkvz,
            w_ba,
            w_conv,
            a_log,
            dt_bias,
            g_norm,
            w_out,
            s_out,
            g_post1,
            w_r,
            w_ug,
            s_ug,
            w_dn,
            s_dn,
            w_sh_ug,
            w_sh_dn,
            scratch,
            sym,
            peers,
            rank,
            layer,
            bt_stride,
            tl,
        ).launch(grid=(G,), block=(THREADS,), stream=stream)

    return flyc.compile[{"llvm_options": {"disable-vector-combine": True}}](
        launch_qwen35
    )

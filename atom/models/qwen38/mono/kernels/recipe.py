# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen3.8-Flash-Next layer recipes (the Quark checkpoint): the blocks of each layer kind, in dataflow order.

Args (plain memory): ``h_in`` [rows, 4 x 2560] bf16 (the layer input: four residual streams), ``h_out`` [rows, 4 x
2560] bf16 (the layer output), the blocks' weights and the mixer's caches and states (ATOM's pools, read and written
in place). Every hand-off inside the layer is a channel:

- **delta**: ``h_in`` -> ``HCIn("attn_hc")`` -> ``xa`` (MXFP8) / ``xab`` (BF16) / ``inj_a`` -> ``in_proj`` Linear
  (qkv | z, FP8 rows) -> ``proj`` (BF16) -> ``GDNCore`` (b, a in the core from ``xab``) -> ``core`` (MXFP8) ->
  ``out_proj`` Linear + ``HCWrite`` (residual ``h_in``) -> ``h1`` (BF16, the four streams) -> ``HCIn("mlp_hc")`` ->
  ``xm`` / ``xmb`` / ``inj_m`` -> Router (from ``xmb``) -> the line MoE (gathered gate_up with the fused shared
  expert, down + ``Combine(hc=HCWrite)``, residual ``h1``) -> ``h_out``. The PLE layer runs Native PLE first, then
  this program.
- **qsa**: ``h_in`` -> ``HCIn("attn_hc")`` -> ``xa`` -> ``qkv`` (q_proj per head [q, gate] | k | v, FP8 rows) ->
  ``proj``; ``xab`` -> ``idx_qk`` (the index projection, bf16 as ATOM keeps it) -> ``iqk``; ``QKNormRope`` (+ K / V
  append) -> ``qn`` / ``kv``; ``Indexer`` (raw index key and block key appends) -> ``sel`` (I32); ``SparseAttnCore``
  -> ``attn`` (MXFP8) -> ``o_proj`` Linear + ``HCWrite`` -> ``h1`` -> the MLP half as in delta.
- **mtp** (the MTP drafter's layer, ``mtp_bf16_blocks``): the qsa dataflow on BF16 weights and caches, bound to the
  drafter, with RoPE tables one row on (``rope_tables``).

State args: GDN ``gdn.<conv_pool | ssm_pool | conv_slot | conv_off | ssm_read | ssm_write>``
(``blocks.gdn``); QSA ``k_cache`` / ``v_cache`` (+ ``k_scale`` / ``v_scale``) /
``index_raw`` / ``index_cmp`` (paged on ``block_table``), ``seq_lens``. The routed experts are Mono's A8W4 tensors
``moe.w13`` / ``moe.w13_scale`` / ``moe.w2`` / ``moe.w2_scale`` (slot E = the shared expert), bound in place.

Knobs (``schedules.KNOBS``): per block constructor knobs by block name (``in_proj`` / ``out_proj`` / ``qkv`` /
``idx_qk`` / ``qkv_idx`` / ``o_proj``: ``Linear``; ``gdn``; ``attn_hc`` / ``mlp_hc``: ``HCIn``; ``moe``: ``router`` /
``line``), ``place`` ({stage name: first CTA}), ``hold`` ({stage name: s_sleep units}) and ``dual``.
"""

import copy

from atom.models.qwen38.mono.kernels.blocks.base import Channel, Fmt
from atom.models.qwen38.mono.kernels.blocks.gdn import GDNCore
from atom.models.qwen38.mono.kernels.blocks.hc import HCIn, HCNames, HCWrite
from atom.models.qwen38.mono.kernels.blocks.linear import BF16Out, Linear
from atom.models.qwen38.mono.kernels.blocks.moe import Combine, moe_blocks
from atom.models.qwen38.mono.kernels.geometry import (
    QSA_KINDS,
    ROPE_OFFSET,
    G,
    Geometry,
)
from atom.models.qwen38.mono.kernels.layer import Schedule
from atom.models.qwen38.mono.kernels.program import PlanError
from atom.models.qwen38.mono.kernels.target import MI355X

HC_PREFIX = {"attn": "attn_hyper_connection", "mlp": "mlp_hyper_connection"}
GDN_LINEARS = ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z")
QSA_LINEARS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.indexer.index_qk_proj",
)
# the quark_attnfp8 checkpoint family: every FP8 projection is ``Linear(weight_format="mxfp8", row_scale=True)`` (E4M3
# values as stored, the fp32 row scale on the sum); the HC down / up weights are real bf16
QUARK = "quark_attnfp8"
FP8_ROWS = {"weight_format": "mxfp8", "row_scale": True}
# ATOM Qwen's QSA caches (``--kv_cache_dtype fp8 --block-size 64``): fp8 K / V rows with fp32 (token, head) scales,
# bf16 index caches, pages of 64 tokens; the drafter's caches are bf16 on pages of ``MTP_KV_BLOCK`` tokens
QSA_KV = {"fp8": True, "fp8_row": True, "kv_block": 64}
MTP_KV_BLOCK = 64
# the next layer's attention-HC down weight (``attn_hc.dw``, ~6.5 MB) touched into the Infinity Cache from this
# layer's program (``Prefetch``: args hcpf.w / hcpf.n): after this stage, on this many CTAs
HCPF_AFTER, HCPF_TASKS = "moe.router", 32


def _fp8(knobs: dict) -> dict:
    """A projection's constructor knobs: FP8 rows under the schedule's knobs."""
    return {**FP8_ROWS, **knobs}


def hc_names(which: str) -> HCNames:
    p = HC_PREFIX[which]
    return HCNames(
        f"{p}.hc_norm",
        f"{p}.input_mix_weight_down",
        f"{p}.input_mix_weight_up",
        f"{p}.block_inject_weight",
    )


class Channels:
    """The layer's channels for a geometry (names are global within a layer program)."""

    def __init__(self, geo: Geometry = G):
        h, w = geo.hidden, geo.hc_width
        self.xa = Channel("xa", h, Fmt.MXFP8)  # the attention HC's mixed row
        self.xab = Channel("xab", h, Fmt.BF16)
        self.inj_a = Channel("inj_a", geo.hc_count, Fmt.F32)
        self.h1 = Channel("h1", w, Fmt.BF16)  # the streams after the attention combine
        self.xm = Channel("xm", h, Fmt.MXFP8)  # the MLP HC's mixed row
        self.xmb = Channel("xmb", h, Fmt.BF16)
        self.inj_m = Channel("inj_m", geo.hc_count, Fmt.F32)
        # qkv | z
        self.gdn_proj = Channel("proj", geo.conv_dim + geo.gdn_value_width, Fmt.BF16)
        self.core = Channel("core", geo.gdn_value_width, Fmt.MXFP8)
        # q_proj (per head [q, gate]) | k | v | index q | index k
        self.qsa_proj = Channel(
            "proj",
            2 * geo.q_width + 2 * geo.kv_width + (geo.index_heads + 1) * geo.index_dim,
            Fmt.BF16,
        )
        self.attn = Channel("attn", geo.q_width, Fmt.MXFP8)


def _hc_in(which: str, src, ch: Channels, geo: Geometry, knobs: dict) -> HCIn:
    if which == "attn":
        out, bf, inj = ch.xa, ch.xab, ch.inj_a
    else:
        out, bf, inj = ch.xm, ch.xmb, ch.inj_m
    name = f"{which}_hc"
    kw = {"weights_fmt": "bf16", **knobs.get(name, {})}
    return HCIn(name, src, geo.hc, out, inj, hc_names(which), bf16_out=bf, **kw)


def mlp_blocks(ch: Channels, geo: Geometry, knobs: dict) -> list:
    """The MLP half: h1 -> the MLP HC -> MoE (router, gathered gate_up with the fused shared
    expert, gathered down + weighted combine into the streams) -> ``h_out``. The router's select
    stage is left out (the gathered stages select from its partials)."""
    moe_knobs = dict(knobs.get("moe", {}))
    moe_knobs["router"] = {"select": False, **moe_knobs.get("router", {})}
    combine = Combine(hc=HCWrite(geo.hc_count, ch.inj_m, out="h_out", channel=ch.h1))
    moe = moe_blocks(
        "moe",
        geo.moe,
        ch.xmb,
        ch.xm,
        combine,
        knobs=moe_knobs,
        base=knobs.get("moe_base", 0),
    )
    return [_hc_in("mlp", ch.h1, ch, geo, knobs)] + moe


def delta_blocks(geo: Geometry = G, knobs: dict | None = None, replay: int = 0) -> list:
    """GDN and MoE (after Native PLE on the PLE layer). ``replay``: the ReplaySSM record capacity (0: classic
    verify)."""
    knobs = knobs or {}
    ch = Channels(geo)
    return [
        _hc_in("attn", "h_in", ch, geo, knobs),
        Linear("in_proj", GDN_LINEARS, ch.xa, ch.gdn_proj.width, BF16Out(ch.gdn_proj),
               **_fp8(knobs.get("in_proj", {}))),
        GDNCore("gdn", proj=ch.gdn_proj.name, core=ch.core, geo=geo.gdn, ba=ch.xab, replay=replay,
                **knobs.get("gdn", {})),
        Linear("out_proj", ("linear_attn.out_proj",), ch.core, geo.hidden,
               HCWrite(geo.hc_count, ch.inj_a, publish=ch.h1, arg="h_in"), **_fp8(knobs.get("out_proj", {}))),
    ] + mlp_blocks(ch, geo, knobs)  # fmt: skip


def qsa_blocks(
    geo: Geometry = G, knobs: dict | None = None, table_width: int = 2048
) -> list:
    """A QSA (sparse attention + MoE) layer; ``table_width``: block-table entries per request (the context bucket).
    q / k / v are the FP8 rows Linear (``qkv``); ATOM keeps ``indexer.index_qk_proj`` bf16 (its online PTPC excludes
    ``*indexer*``), so the index projection is its own bf16 x bf16 Linear on the mixed row's BF16 channel.
    """
    from atom.models.qwen38.mono.kernels.blocks.indexer import Indexer
    from atom.models.qwen38.mono.kernels.blocks.qknorm import QKNormRope
    from atom.models.qwen38.mono.kernels.blocks.sparse import SparseAttnCore

    knobs = knobs or {}
    ch = Channels(geo)
    tw = {"table_width": table_width}
    kv = QSA_KV
    knobs = {**knobs, "qk": {**kv, **knobs.get("qk", {})}, "sa": {**kv, **knobs.get("sa", {})},
             "idx": {"kv_block": kv["kv_block"], **knobs.get("idx", {})}}  # fmt: skip
    pw = 2 * geo.q_width + 2 * geo.kv_width
    proj = Channel("proj", pw, Fmt.BF16)
    iqk = Channel("iqk", (geo.index_heads + 1) * geo.index_dim, Fmt.BF16)
    return [
        _hc_in("attn", "h_in", ch, geo, knobs),
        Linear("qkv", QSA_LINEARS[:3], ch.xa, pw, BF16Out(proj), **_fp8(knobs.get("qkv", {}))),
        Linear("idx_qk", QSA_LINEARS[3:], ch.xab, iqk.width, BF16Out(iqk),
               **{"weight_format": "bf16", **knobs.get("idx_qk", {})}),
        QKNormRope(geo.attn, proj=proj.name, proj_width=pw, **tw, **knobs.get("qk", {})),
        Indexer(geo.indexer, iqk=iqk.name, iqk_width=iqk.width, col=0, rope="idx_rope", **tw, **knobs.get("idx", {})),
        SparseAttnCore(geo.attn, geo.indexer, proj=proj.name, out=ch.attn, proj_width=pw, **tw, **knobs.get("sa", {})),
        Linear("o_proj", ("self_attn.o_proj",), ch.attn, geo.hidden,
               HCWrite(geo.hc_count, ch.inj_a, arg="h_in", publish=ch.h1), **_fp8(knobs.get("o_proj", {}))),
    ] + mlp_blocks(ch, geo, knobs)  # fmt: skip


def mtp_bf16_blocks(geo: Geometry = G, knobs: dict | None = None, table_width: int = 2048, batch: int = 1,
                    spec: int = 1) -> list:  # fmt: skip
    """The MTP drafter's layer as ATOM keeps it (``mtp.*`` outside the online PTPC quantization): every projection
    bf16 x bf16 (``Linear(weight_format="bf16")`` on the mixed rows' BF16 channels), the attention's output a BF16
    channel, bf16 K / V and index caches on pages of ``MTP_KV_BLOCK`` tokens, the router's select stage, then the bf16
    experts (``moe_bf16``: one row, the shared expert as slot ``experts``) and the combine into the streams.

    ``spec`` > 1 (the drafter's first step, ATOM's ``selected_rows``): the attention on every row of the step (their
    K / V and index keys cached), then each request's selected row (the int32 arg ``sel`` [batch], rows of the step:
    ``SelectRow``) through the MLP half, planned for ``batch`` rows (``AtRows``): ``h_out`` holds those rows.
    """
    from atom.models.qwen38.mono.kernels.blocks.indexer import Indexer
    from atom.models.qwen38.mono.kernels.blocks.moe import Router
    from atom.models.qwen38.mono.kernels.blocks.moe_bf16 import DownBF16, GateUpBF16
    from atom.models.qwen38.mono.kernels.blocks.qknorm import QKNormRope
    from atom.models.qwen38.mono.kernels.blocks.select_row import AtRows, SelectRow
    from atom.models.qwen38.mono.kernels.blocks.sparse import SparseAttnCore
    from atom.models.qwen38.mono.kernels.program import Scenario

    knobs = knobs or {}
    ch = Channels(geo)
    pw = ch.qsa_proj.width
    icol = 2 * geo.q_width + 2 * geo.kv_width
    tw = {"table_width": table_width}
    kvb = MTP_KV_BLOCK
    attn = Channel("attn", geo.q_width, Fmt.BF16)
    ids = Channel("moe.ids", geo.top_k, Fmt.I32)
    wts = Channel("moe.wts", geo.top_k + 1, Fmt.F32)
    act = Channel("moe.act", (geo.top_k + 1) * geo.moe_inter, Fmt.BF16)
    bf = {"weight_format": "bf16"}
    h1 = ch.h1
    tail = []
    if spec > 1:
        h1 = Channel("h1s", ch.h1.width, ch.h1.fmt)
        tail = [SelectRow("sel_row", ch.h1, h1, out_rows=batch)]
    mlp = [
        _hc_in("mlp", h1, ch, geo, knobs),
        Router("moe.router", ch.xmb, "moe.router_w", geo.experts, geo.top_k, ids, wts, shared=1, select=True),
        GateUpBF16("moe.gu", ch.xmb, ids, wts, act, geo.experts, geo.moe_inter, geo.top_k),
        DownBF16("moe.dn", act, ids, HCWrite(geo.hc_count, ch.inj_m, out="h_out", channel=h1), geo.experts,
                 geo.moe_inter, geo.hidden, geo.top_k),
    ]  # fmt: skip
    if spec > 1:
        mlp = [AtRows(b, Scenario(batch, 1)) for b in mlp]
    return [
        _hc_in("attn", "h_in", ch, geo, knobs),
        Linear("qkv_idx", QSA_LINEARS, ch.xab, pw, BF16Out(ch.qsa_proj), **{**bf, **knobs.get("qkv_idx", {})}),
        QKNormRope(geo.attn, proj=ch.qsa_proj.name, proj_width=pw, kv_block=kvb, **tw),
        Indexer(geo.indexer, iqk=ch.qsa_proj.name, iqk_width=pw, col=icol, rope="idx_rope", kv_block=kvb, **tw),
        SparseAttnCore(geo.attn, geo.indexer, proj=ch.qsa_proj.name, out=attn, proj_width=pw, kv_block=kvb, **tw),
        Linear("o_proj", ("self_attn.o_proj",), attn, geo.hidden,
               HCWrite(geo.hc_count, ch.inj_a, arg="h_in", publish=ch.h1), **{**bf, **knobs.get("o_proj", {})}),
        *tail,
        *mlp,
    ]  # fmt: skip


def rope_tables(blocks, packed: dict, kind: str) -> dict:
    """``packed`` (the blocks' bound weights) with the QSA blocks' RoPE tables for ``kind``: as
    packed for qsa; for the MTP drafter (``config.ROPE_OFFSET``) row ``i`` holds position ``i +
    offset`` (the same builders, ``offset`` more rows, the first ``offset`` dropped: the same
    shape, so the qsa binary and its bind checks apply unchanged)."""
    from atom.models.qwen38.mono.kernels.blocks import indexer, qknorm

    off = ROPE_OFFSET.get(kind, 0)
    if not off:
        return packed
    out = dict(packed)
    for blk in blocks:
        name = getattr(blk, "names", {}).get("rope")
        if isinstance(blk, qknorm.QKNormRope):
            g, dev = blk.geo, packed[name].device
            table = qknorm.rope_table(
                blk.max_positions + off, g.rotary_dim, g.rope_theta, dev
            )
        elif isinstance(blk, indexer.Indexer):
            table = indexer.rope_table(
                blk.max_positions + off, blk.geo, packed[name].device
            )
        else:
            continue
        out[name] = table[off:].contiguous()
    return out


def overlay_chunk(program, knobs: dict | None = None) -> None:
    """The indexer's chunk stage (builds whose block table is wider than one chunk) goes on the
    CTAs right after the final stage's (the attention's first CTAs), instead of pushing the
    final, the attention and the rest of the layer ``rows x chunks`` CTAs on: the layer's
    placement is the same at every width, so a build at ATOM's width (``max_model_len / 16``)
    serves short contexts at the narrow build's speed. Short rows: a chunk task is one
    ``seq_lens`` load in front of an attention task that waits for ``sel`` anyway; long rows:
    those attention tasks start when their CTA's chunk task ends, before ``sel`` exists. A CTA
    runs its chunk tasks before its final task (stage order), so the final's polls of the
    candidates always progress. Unless ``knobs["place"]`` places the chunk or the final stage.
    """
    names = [st.name for st in program.stages]
    place = {
        k: v for k, v in (knobs or {}).get("place", {}).items() if v is not None
    }  # (None: the derived base)
    if "idx.chunk" not in names or "idx.chunk" in place or "idx.final" in place:
        return
    nxt = 0
    for st in program.stages:  # plan()'s rule for unplaced stages
        if st.name == "idx.chunk":
            base = nxt % MI355X.cus if st.base is None else st.base
            st.base = (base + 1) % MI355X.cus
            program.stages[names.index("idx.final")].base = base
            return
        nxt = (nxt % MI355X.cus if st.base is None else st.base) + st.tasks


def schedule(knobs: dict | None = None) -> Schedule:
    """The layer's placements from ``knobs["place"]`` ({stage: first CTA}; None: the derived base)."""
    return Schedule(
        place={k: v for k, v in (knobs or {}).get("place", {}).items() if v is not None}
    )


def compose_layer(
    kind: str,
    sc,
    geo: Geometry = G,
    table_width: int = 2048,
    family: str = QUARK,
    replay: int = 0,
):
    """The one way to build a Flash layer program: the recipe of ``kind`` ("delta" | "qsa" | "mtp": the drafter's,
    named as the qsa program) with the scenario's schedule (``schedules.KNOBS``) -> (blocks, Program).
    ``table_width``: QSA's block-table bucket; ``family``: the checkpoint family (``QUARK`` only); ``replay``: the
    GDN verify's ReplaySSM record capacity (0: classic).
    """
    from atom.models.qwen38.mono.kernels import schedules
    from atom.models.qwen38.mono.kernels.blocks.prefetch import Prefetch
    from atom.models.qwen38.mono.kernels.layer import compose

    if family != QUARK:
        raise ValueError(
            f"Qwen Mono builds the {QUARK} checkpoint family only, got {family!r}"
        )
    mtp = kind == "mtp"
    if (kind, sc.batch, sc.spec) not in schedules.KNOBS:
        raise PlanError(f"no Qwen Mono schedule for {kind} {sc.name}")
    knobs = copy.deepcopy(schedules.KNOBS[(kind, sc.batch, sc.spec)])
    if kind in QSA_KINDS:
        kind = "qsa"  # one program name: the MTP layer differs in its bound weights, caches and RoPE tables
    if mtp:
        blocks = mtp_bf16_blocks(geo, knobs, table_width, batch=sc.batch, spec=sc.spec)
    elif kind == "qsa":
        blocks = qsa_blocks(geo, knobs, table_width)
    elif kind == "delta":
        blocks = delta_blocks(geo, knobs, replay)
    else:
        raise ValueError(f"unknown Flash layer kind {kind!r}")
    sched = schedule(knobs)
    if not mtp:
        sched.insert.append((HCPF_AFTER, Prefetch("hcpf", 8 << 20, HCPF_TASKS, None)))
    program = compose(f"flash_{kind}", sc, blocks, schedule=sched)
    overlay_chunk(program, knobs)
    # (the composed BF16 drafter, and the GDN layer under ReplaySSM, sit past the SGPR budget: spills into VGPR
    # lanes, no memory)
    if mtp or (kind == "delta" and replay):
        program.sgpr_spill_ok = True
    for st in program.stages:  # entry holds by stage name ({stage: s_sleep units})
        st.hold = int(knobs.get("hold", {}).get(st.name, st.hold))
    _dual(program, knobs.get("dual"))
    return blocks, program


def _dual(program, knob: dict | None) -> None:
    """Dual-copy mailboxes (``Program.dual``) from the schedule knob ``"dual"`` ({"max": bytes}), built only where
    MI355X's home rule holds (``dual.available``)."""
    if not knob:
        return
    from atom.models.qwen38.mono.kernels.dual import available

    if not available():
        return
    program.dual = True
    program.dual_max = int(knob.get("max", 0))

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek-V4.1 layer recipes: the blocks of each layer kind, in dataflow order.

The contract is a hardware-arithmetic reference of the checkpoint's layer (its operations with the kernels'
roundings): a layer's ``h_out`` within D10 of it, routing ids and the index sources' selections exact (the near-tie
rule). Every kind is one program at TP4 (64 / 4 heads a rank; the KV path, q_a, the router and HC replicated; wo_b
row-parallel; the routed and the shared experts sharded), a single-layer 16-row build two (``split``: the attention
kernel, then the FFN kernel). Stages, in order (channels between them):

    seam A        HCSeam: fold of the previous layer's MoE output into the 4 streams (layer 0: the embedding streams,
                  no fold) -> lin_a (BF16), gate_a (F32 24); Engram layers: R -> ra, Engram gate half -> re, then
                  the mixes on re
    attn_norm     NormQuant -> xa (MXFP8); sources also xab (BF16) for the Compressor / Indexer
    comp          sources: Compressor (xab) -> this step's compressed entries (ckvn) into the cache, index keys
    wq_a, wkv     Linear MXFP8 -> qr (BF16 1280), kv (BF16 512)
    q_norm        NormQuant -> qn (MXFP8 1280)
    kv_norm       NormQuant -> kvq (MXFP8 512), kvb (BF16 512)
    idx           index sources: Indexer (qn, xab) -> sel (the step rows' top-512), stored to ``sel_out``
    wq_b          Linear MXFP8 -> q (BF16 heads x 512)
    kvprep        KVPrep -> kvn (BF16 512)
    attn          window layers: MLAWindowAttn; compressed layers: MLASparseAttn over the window and the top-512
                  compressed keys -> o (MXFP8 heads x 512); ringw: RingWrite (the window ring)
    wo_a          Linear groups (MXFP8 from the checkpoint's FP8, bf16 arithmetic) -> oa (MXFP8)
    wo_b          Linear MXFP8, TP partials -> seam F
    seam F        HCSeam (fold of the attention output) -> R_F (plain ``r_f``), lin_f (BF16), gate_f (F32 24)
    ffn_norm      NormQuant -> xf (MXFP8), xfb (BF16)
    router        MoERouter -> idx, wts (F32 6)
    moe           MoEExperts (the routed experts and the folded shared expert) -> moe_out (BF16) -> ``ffn_out``,
                  or in a chain its partials pushed for the next layer's seam A

The layer's last fold (hc_post of the MoE output into the streams) is the next layer's seam-A fold, so a layer hands
over (R_F, the MoE output, the FFN seam's gate) as plain memory.

Args a layer reads: ``h_in`` [rows, 4, 5120] bf16 (the streams R before seam A), ``gin`` [rows, 24] f32 (the
previous FFN seam's gate; layer 0: the one-hot pre), ``pos`` [batch] int32, ``ring`` [batch, ring_slots, 512] bf16,
``rope`` / ``rope.sin`` (ATOM's bf16 RoPE caches), the weights (``layer_impl.pack``); compressed layers ``ckv`` (the
compressed cache: the owner's rows in ATOM's pool, in place, with the request words and the page geometry,
``paged.REQ``; bf16 rows, or MXFP8 ones with ``KV_FMT`` "fp8") and ``sel`` (consumers: the step rows' selection), sources their caches and states
(``window_layer_blocks``). Written: ``r_f`` [rows, 4, 5120] bf16, ``gate_f`` [rows, 24] f32, ``ffn_out`` [rows, 5120]
bf16, the ring, ATOM's caches (a live row's entries).

**Knobs** (``schedules``: ``DEFAULT``, ``SCHEDULES[(kind, rows)]``, a build's own). The rules
that depend on the build rather than on (kind, rows), resolved here: ``MOE_SUM`` / ``MOE_OWN_QUADS`` / ``HC_ENTRY``
(the chain's exchange: ``_moe_mode``), ``SHARED`` (fold at TP > 1), ``FFN_NORM`` "auto" (router on window layers),
``NORM_SPLIT`` "auto" (ffn_norm in ``FFN_NORM_SPLIT`` tasks), ``SPARSE_RING8`` "auto" (from 6 rows on consumers), and the placement passes (``moe_route_early``: ``IDX_AFTER_POOL``,
``SPARSE_SMERGE``'s placement, ``HCF_GATE_AT``, ``WO_A_HOLD``, ``NUMA_DUAL``), which read the program's stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace

import flydsl.expr as fx
from flydsl.expr.typing import T

from ..blocks.base import Block, Channel, Fmt, Proto
from ..blocks.gfx950 import norm
from ..blocks.gfx950.attention.dsa_indexer import Indexer
from ..blocks.gfx950.attention.mla_sparse import MLASparseAttn
from ..blocks.gfx950.attention.mla_window import KVPrep, MLAWindowAttn, RingWrite
from ..blocks.gfx950.compressor import Compressor
from ..blocks.gfx950.gathered import DotNorm
from ..blocks.gfx950.linear import BF16Out, Linear, MXFP8Out, NormIn, SwiGLU
from ..blocks.gfx950.mhc import GATHER_TILE, MIX, HCSeam, PeerPartial
from ..blocks.gfx950.moe_experts import MoEExperts
from ..blocks.gfx950.router import MoERouter
from ..device.gfx950 import buffer_ops as bo
from ..device.gfx950 import channels as chan
from ..device.gfx950 import mailbox as mb
from ..device.gfx950.ops import rsrc, traced
from ..plan.program import Scenario, Stage, numa_stages
from . import schedules
from .config import G

HIDDEN, QR, KVD, HEADS, HD, OGROUPS, ORANK = (
    G.hidden,
    G.q_lora,
    G.head_dim,
    G.heads,
    G.head_dim,
    G.o_groups,
    G.o_lora,
)
EXPERTS, TOPK, INTER, LIMIT, ROUTE_SCALE = (
    G.experts,
    G.topk,
    G.moe_inter,
    G.swiglu_limit,
    G.route_scale,
)
EPS = G.eps


def _fold_shared(tp: int, shared: str) -> bool:
    return shared == "fold" or (shared == "auto" and tp > 1)


def _ch(name, width, fmt):
    return Channel(name, width, fmt)


def channels(
    tp: int = 1, xa_proto: str | None = None, topk: int | None = None
) -> SimpleNamespace:
    """The layer's channels for a TP-``tp`` rank (q / o / oa carry this rank's heads and wo_a groups); ``topk``: the
    routing channels' width (default ``TOPK``: the target's; the DSpark draft stages route top-3).
    """
    heads, groups = HEADS // tp, OGROUPS // tp
    topk = TOPK if topk is None else topk
    xa_proto = xa_proto or schedules.current()["XA_PROTO"]
    return SimpleNamespace(
        lin_a=_ch("lin_a", HIDDEN, Fmt.BF16), gate_a=_ch("gate_a", MIX, Fmt.F32),
        xa=Channel("xa", HIDDEN, Fmt.MXFP8, Proto(xa_proto)),
        qr=_ch("qr", QR, Fmt.BF16), kv=_ch("kv", KVD, Fmt.BF16), qn=_ch("qn", QR, Fmt.MXFP8),
        kvq=_ch("kvq", KVD, Fmt.MXFP8), kvb=_ch("kvb", KVD, Fmt.BF16), q=_ch("q", heads * HD, Fmt.BF16),
        kvn=_ch("kvn", KVD, Fmt.BF16), o=_ch("o", heads * HD, Fmt.MXFP8), oa=_ch("oa", groups * ORANK, Fmt.MXFP8),
        ao=_ch("ao", HIDDEN, Fmt.BF16), lin_f=_ch("lin_f", HIDDEN, Fmt.BF16), gate_f=_ch("gate_f", MIX, Fmt.F32),
        xf=_ch("xf", HIDDEN, Fmt.MXFP8), xfb=_ch("xfb", HIDDEN, Fmt.BF16), idx=_ch("idx", topk, Fmt.F32),
        wts=_ch("wts", topk, Fmt.F32), idx_r=_ch("idx_r", topk, Fmt.F32), wts_r=_ch("wts_r", topk, Fmt.F32),
        mid=_ch("mid", INTER, Fmt.MXFP8), shared=_ch("shared", HIDDEN, Fmt.BF16),
        moe_out=_ch("moe_out", HIDDEN, Fmt.BF16), lin_z=_ch("lin_z", HIDDEN, Fmt.BF16),
        ra=_ch("ra", 4 * HIDDEN, Fmt.BF16), xab=_ch("xab", HIDDEN, Fmt.BF16), ckvn=_ch("ckvn", HD, Fmt.BF16),
        ikey=_ch("ikey", 128, Fmt.BF16), sel=_ch("sel", 512, Fmt.F32), lfss=_ch("lfss", HIDDEN // 16, Fmt.F32),
        re=_ch("re", 4 * HIDDEN, Fmt.BF16), lass=_ch("lass", HIDDEN // 16, Fmt.F32),
        rf=_ch("rf", 4 * HIDDEN, Fmt.BF16),
    )  # fmt: skip


C = channels(1, schedules.DEFAULT["XA_PROTO"])


@dataclass(frozen=True)
class Source:
    """A compressed layer's own sources: ``kv`` (a KV source: the Compressor, this step's entries into the cache
    and, ``index`` too, its index keys), ``index`` (an index source: the Indexer, this layer's top-512 selection,
    stored for the consumers), ``cand`` ("source": layer 20's candidate bitmap; "reader": layers 24 .. 36 read it).
    """

    ratio: int
    cache_len: int  # the compressed cache's entries (>= n_max)
    kv: bool = False
    index: bool = False
    n_max: int = (
        0  # the Indexer's positions (a context bucket: max context / ratio, a multiple of 32)
    )
    cand: str = ""
    state_slots: int = (
        0  # ratio 2: the Compressor's state slots (0: ratio, the reference's; >= spec + 1:
    )
    #                       rejection-safe, a rejected draft row overwrites no slot an accepted row still needs)


def _tokens(source: Source | None) -> int:
    """An index source's context bucket in tokens (its Indexer's n_max x ratio: ``schedules.knobs(tokens=)``)."""
    return source.n_max * source.ratio if source is not None and source.index else 0


def _nspec(width, threads, wps):
    return norm.NormSpec(
        width=width, eps=EPS, threads=threads, wps=wps, plain=True, plain_once=True
    )


def _norm_splits(k) -> dict[str, int]:
    """``NORM_SPLIT`` resolved: {norm stage: tasks a row} ("auto": ffn_norm in ``FFN_NORM_SPLIT`` tasks)."""
    if k.NORM_SPLIT == "auto":
        return {"ffn_norm": k.FFN_NORM_SPLIT}
    return {e.split(":")[0]: int(e.split(":")[1]) for e in k.NORM_SPLIT.split(";") if e}


def _nq(name, src, out, gamma, width, threads, wps, bf16_out=None, splits=None, nsplit=0, gathered=False,
        prev=False):  # fmt: skip
    spec = _nspec(width, threads, wps)
    s = nsplit or (splits or {}).get(name, 1)
    if (
        s > 1
    ):  # each of a row's s tasks quantizes width / s (4 words a thread), its sum in the one-task order
        spec = norm.NormSpec(width=width, eps=EPS, threads=width // 2 // s // 4, wps=4, plain=True, plain_once=True,
                             split=s, sum_threads=threads, sum_wps=wps)  # fmt: skip
    extra = (
        {"src_gathered": GATHER_TILE, "src_prev": prev} if gathered else {}
    )  # a sharded seam's LIN (``HCSeam``)
    return norm.NormQuant(name, src, out, gamma, bf16_out=bf16_out, spec=spec, **extra)


@traced
def _store(k, t, src, dst, width):
    """Row t of channel ``src`` (``width`` pairs a row: BF16 two values a word, F32 one) -> plain arg ``dst`` (a pair
    a thread a round)."""
    for i in range(k.tid, width, 512):
        w = mb.poll([(k.mb[src], t * width + i, 1)], k.tag)[0][0]
        bo.buffer_store(w, rsrc(k.args[dst]), t * width + i)


class Store(Block):
    """A BF16 / F32 channel -> plain memory [rows, width] (bf16 / 32-bit words: what the next launch reads)."""

    def __init__(self, name: str, src: Channel, dst: str):
        self.name, self.src, self.dst = name, src, dst

    def stages(self, sc: Scenario) -> list[Stage]:
        pr = self.src.pairs_per_row()
        return [Stage(self.name, sc.rows, partial(_store, src=self.src.name, dst=self.dst, width=pr),
                      args=(self.dst,), key=("store", self.src.name, self.dst, pr), row_tasks=1)]  # fmt: skip


@traced
def _replay_row(k, t, gate, chans):
    """Row t of the replayed routing (plain args ``<ch>`` [rows, TOPK], f32 / int32 bits) into channels ``chans``,
    published once the router has published row t (its last ``gate`` value): the router stays on the MoE's chain.
    """
    mb.poll([(k.mb[gate], t * TOPK + TOPK - 1, 1)], k.tag)
    it = t * (TOPK // 2) + fx.min(k.tid, fx.Int32(TOPK // 2 - 1))
    for ch in chans:
        w = fx.Vector(
            bo.buffer_load(rsrc(k.args[ch]), it * 2, vec_width=2, dtype=T.i32)
        )
        if k.tid < TOPK // 2:
            chan.put_words(k.mb[ch], it * 2, [w[0], w[1]], k.tag)


class Replay(Block):
    """Channels ``chans`` ([rows, TOPK] F32) published from plain args of the same names: replayed routing, a row as
    soon as the router (writing channel ``after``, unread) has published that row."""

    def __init__(self, name: str, chans, after: Channel):
        self.name, self.chans, self.after = name, tuple(chans), after

    def stages(self, sc: Scenario) -> list[Stage]:
        names = tuple(ch.name for ch in self.chans)
        writes = tuple(
            r for ch in self.chans for r in ch.regions(sc.rows, (self.name,))
        )
        return [Stage(self.name, sc.rows, partial(_replay_row, gate=self.after.name, chans=names), writes=writes,
                      args=names, key=("replay", self.after.name, names))]  # fmt: skip


CHAIN_PEER = (
    "moe.part",
    "moe.part.dag",
    "lin_n",
)  # a chain's cross-launch peer regions (the MoE partial; with
#                                                    MOE_SUM "own" the owners' gathered sums; a sharded residual's next
#                                                    LIN, SHARD_SEAM_A "end"), pinned first in every program


def chain_peer_first(
    tp: int, knobs: dict | None = None, rows: int | None = None
) -> tuple[str, ...]:
    """A chain's programs' ``Program.peer_first``: ``CHAIN_PEER``; with a sharded residual only the MoE partial (the
    chain's one cross-launch region: no launch gathers owner sums for the next, and the tail's own lie last).
    """
    return (
        ("moe.part", "lin_n")
        if residual_layout(tp, knobs=knobs, rows=rows) == "sharded"
        else CHAIN_PEER
    )


def residual_layout(
    tp: int, split: bool = False, knobs: dict | None = None, rows: int | None = None
) -> str:
    """``RESIDUAL`` resolved for a layer: "sharded" (TP, one kernel: R column-owned, ``HCSeam(residual="sharded")``)
    or "replicated"; "auto": sharded from ``R_SHARD_ROWS`` to ``R_SHARD_MAX_ROWS`` rows (``rows`` None:
    replicated). A sharded layer's R args (``h_in``, ``r_a``, ``r_f``) hold the rank's columns:
    [rows, 4, hidden / tp]."""
    kn = knobs or schedules.current()
    v = kn["RESIDUAL"]
    if (
        v == "auto"
    ):  # (measured at one request of 6 / 8 / 16 rows and B = 2-4 requests of 6)
        v = (
            "sharded"
            if rows is not None and kn["R_SHARD_ROWS"] <= rows <= kn["R_SHARD_MAX_ROWS"]
            else "replicated"
        )
    return "sharded" if v == "sharded" and tp > 1 and not split else "replicated"


def rs_pack(tp: int, knobs: dict | None = None, rows: int | None = None) -> str:
    """``RS_PACK`` resolved for a step: the reduce-scatter partials' format of every two-shot exchange (wo_b ->
    seam F, the MoE combine -> its owner stage or the next seam A): "f32" ((fp32, tag) pairs, 8 B a value), "f32p" (a
    lane's 4 values as a tagged quad of 3 + a (value, tag) pair, 6 B) or "f32q" (3 values + tags a 16 B quad across
    lanes, 5.5 B; slower: its lane exchange), the same sums either way; "auto": f32p from
    ``RS_PACK_ROWS`` rows (``rows`` None: pairs) where an owner's columns are whole waves (HIDDEN / tp a multiple of
    256: measured at TP4; at 8 ranks the MoE combine's wave crosses an owner boundary, ``push_rs(two_owners=)``, where
    f32p is not measured yet: pairs). By the step's rows and tp only (a chain's launches read each other's partials:
    ``schedules.CHAIN_KNOBS``)."""
    kn = knobs or schedules.current()
    v = kn["RS_PACK"]
    if v not in ("pair", "f32q", "f32p", "auto"):
        raise ValueError(f"RS_PACK 'pair', 'f32q', 'f32p' or 'auto', got {v!r}")
    if v == "auto":
        whole = (
            tp > 1 and (HIDDEN // tp) % (4 * 64) == 0
        )  # (an MoE combine wave's 256 columns: one owner)
        v = (
            "f32p"
            if whole and rows is not None and rows >= kn["RS_PACK_ROWS"]
            else "pair"
        )
    return v if v in ("f32q", "f32p") and tp > 1 else "f32"


def _order(k) -> str:
    return "fold" if k["SHARED_ORDER"] == "auto" else k["SHARED_ORDER"]


def chain_pend(
    tp: int, rows: int | None, order: str | None = None, knobs: dict | None = None
):
    """The ``HCSeam`` pend of a program that reads the previous launch's chained MoE partials (a layer's seam A with
    ``prev_moe``, or a reader after the chain such as the verify head): the owners' gathered bf16 sums
    (``moe.part.dag``) when that launch ran "own", else the pushed partials (``moe.part``).
    """
    kn = knobs or schedules.current()
    order = order or _order(kn)
    rq = rs_pack(tp, kn, rows)
    if (
        residual_layout(tp, knobs=kn, rows=rows) == "sharded"
    ):  # the reduce-scatter partials of the rank's columns
        return PeerPartial("moe.part", "rs", prev_launch=True, rs_fmt=rq)
    m = _moe_mode(tp, order, rows=rows, chain=True, knobs=kn)
    if m == "own":
        return PeerPartial("moe.part.dag", "ag", prev_launch=True)
    return PeerPartial(
        "moe.part", "ref3" if order == "ref" else "f32", prev_launch=True
    )


def _moe_mode(tp: int, order: str, split: bool = False, rows: int | None = None, chain: bool = False,
              knobs: dict | None = None) -> str:  # fmt: skip
    """``MOE_SUM`` resolved for a layer's MoE partial: "push", "2shot" or "own" (only in a chain: ``chain`` = the
    partial is pushed for the next launch's seam A); two-shot needs TP and the "fold" order. "auto": 2shot in a
    two-kernel layer's FFN kernel; in a chain "own" from ``MOE_OWN_ROWS`` rows on, push otherwise.
    """
    kn = knobs or schedules.current()
    m = kn["MOE_SUM"]
    if m == "auto":
        m = (
            "2shot"
            if split
            else (
                "own"
                if chain and rows is not None and rows >= kn["MOE_OWN_ROWS"]
                else "push"
            )
        )
    return "2shot" if m == "own" and not chain else m


_NO_TP = object()


def sgpr_spill_ok(sparse: int, source: Source | None, rows: int | None = None, engram: bool = False,
                  sparse_ratio: int = 0, tp: int | None = _NO_TP, split: bool = False) -> bool:  # fmt: skip
    """Whether a kind's one-kernel program needs ``Program(sgpr_spill_ok=True)`` (SGPR spills into VGPR lanes, never
    to memory): the kind's ``SPILL_OK`` at ``rows`` (the scenario's; None: the rows-less entry), "auto" there: the
    sources (over the SGPR budget), the consumers with ``RING_FIRST``, and a sharded residual's programs where the
    kind's ``SHARD_SPILL_OK`` says so (measured: the window kinds sit at the SGPR budget). ``tp`` / ``split``: the
    program's, for ``RESIDUAL`` "auto" (``residual_layout``; tp None: a TP build)."""
    if tp is _NO_TP:
        raise TypeError(
            'sgpr_spill_ok: pass the program\'s tp (``RESIDUAL`` "auto" depends on it)'
        )
    kind = schedules.kind_of(sparse, source, engram, sparse_ratio)
    kn = schedules.knobs(kind, rows)
    ok = kn["SPILL_OK"]
    if ok != "auto":
        return bool(ok)
    if kn["RESIDUAL"] == "auto":  # (residual_layout's rule; tp unknown: a TP build)
        shard = residual_layout(2 if tp is None else tp, split, kn, rows) == "sharded"
    shard = shard and bool(kn["SHARD_SPILL_OK"])
    return (
        source is not None
        or (bool(sparse) and schedules.current()["RING_FIRST"] in (True, "auto"))
        or shard
    )


def window_layer_blocks(first: bool, split: bool = False, tp: int = 1, replay: bool = False, push_moe: bool = False,
                        prev_moe: bool = False, sparse: int = 0, source: Source | None = None,
                        ring_slots: int | None = None, engram: bool = False,
                        rows: int | None = None,
                        head: dict | None = None,
                        sparse_ratio: int = 0,
                        knobs: dict | None = None,
                        experts: int | None = None,
                        topk: int | None = None,
                        full_r_a: bool = False,
                        target=None) -> list[Block] | tuple[list[Block], list[Block]]:  # fmt: skip
    """The blocks of a layer's program, in order (module docstring); every layer kind (the name predates the
    compressed ones). ``first``: layer 0 (seam A without a fold: ``h_in`` is the embedding streams, ``gin`` carries
    the one-hot pre; a single-layer build takes every kind so, from its fixture's streams). ``tp``: a rank of a TP group,
    the official model's split: 64 / tp heads (wq_b, the attention, wo_a's 8 / tp groups) and a row-parallel wo_b
    (``Linear(tp_partial=WO_B_PARTIAL)``, summed in fp32), experts sharded (``MoEExperts``); the KV path, q_a, the
    router, the shared expert and HC replicated. ``replay`` (synthetic timing): the router runs (on synthetic weights,
    into unread channels) and the MoE reads plain args ``idx`` / ``wts`` (routing replayed from real taps), each row
    published once the router has published it. ``rows``: the scenario's (the schedule's row-count entries); None:
    the entries of an unknown row count. ``knobs``: values over the schedule's (``schedules.knobs``).

    **A chain of layer launches (TP, one kernel a layer):** ``push_moe``: the MoE only pushes its partials
    (``MoEExperts(dsum=False)``: its combine is the program's last stage; no ``ffn_out`` store); ``prev_moe``: seam
    A's fold sums the previous launch's pushed partials (``PeerPartial(prev_launch=True)``, the same step's previous
    layer) instead of reading ``pend``. ``prev_moe`` without ``push_moe``: the chain's tail (its own MoE summed and
    stored to plain ``ffn_out``: the hand-off out of a chain). Inside a chain a ``prev_moe`` layer also pushes
    (``Stage.prev_reads``: the region's writer is the program's last stage, its own combine), so every layer of a
    chain pushes: the step's first layer ``push_moe`` only, the rest both. The launches share one peer group.

    ``sparse`` (> 0: a compressed consumer layer, the compressed cache's length): ``MLASparseAttn`` over the window
    and the top-512 compressed keys, from plain args ``sel`` (the step rows' positions) and ``ckv`` (the cache).
    ``sparse_ratio``: its compression ratio (the cache's page geometry).
    ``source`` (a source layer, ``Source``): its Compressor / Indexer; the attention reads this launch's selection
    and entries from their channels, the selection is stored to ``sel_out`` for the consumers. Host args (ATOM's
    storage, in place): the KV cache ``ckv``, ``comp.kv_state`` / ``comp.score_state`` (ratio 2: the compress rings),
    the FP4 index plane ``index_k`` / ``index_k.s``; ``idx.bound`` (and the candidate bitmap ``cand`` /
    ``cand.bound``).

    ``head`` (the last layer, one kernel, not ``push_moe``: ``head.head_blocks``' knobs): the verify head in this
    launch after the MoE (``deepseek_v41.head``, its fold reading this launch's channels: seam F's R also into channel
    ``rf``, its gate and the MoE output).

    ``engram`` (layers 1 and 14): the Engram's critical-path half on the streams before this layer's mixes. Seam A
    splits in three: ``hca0`` (``HCSeam(r_only=True)``: the fold, or ``first`` R as given, into channel ``ra``),
    ``Engram("eng", mode="gate")`` (``ra`` -> ``re``; reads the plain ``eng.kv`` [rows, 25600] bf16 an earlier launch's
    ``engram_kv_blocks`` wrote, and ``eng.gw``), then ``hca`` without a fold on ``re`` (LIN, the mixes). Seam F's
    residual is the Engram's output (``re``; plain ``r_e`` across the split).

    ``experts`` / ``topk`` (default ``EXPERTS`` / ``TOPK``, the target's 384 / 6): the routed experts and the router's
    k, for the DSpark draft stages (128 / 3, ``dspark_n_routed_experts`` / ``dspark_num_experts_per_tok``).

    **A sharded residual** (``RESIDUAL`` "sharded", TP, one kernel): R is column-owned, so the R args a launch
    reads and writes (``h_in``, ``r_a``, ``r_f``) are the rank's columns [rows, 4, hidden / tp],
    except what leaves a chain whole: a chain's tail writes ``r_f`` whole (gathered), and ``full_r_a`` (the DSpark aux
    taps' layers) ``r_a``, and what enters it: a chain's first layer reads the whole ``h_in`` (the engine's). The Engram
    layers gather R for the Engram's gate (it needs every column).

    ``target``: None or gfx950's (no other target's block set here)."""
    if target is not None and target.arch != "gfx950":
        raise NotImplementedError(f"deepseek_v41 on {target.arch}: gfx950 only")
    experts = EXPERTS if experts is None else experts
    topk = TOPK if topk is None else topk
    kind = schedules.kind_of(sparse, source, engram, sparse_ratio, split)
    kn = schedules.knobs(kind, rows, over=knobs, tokens=_tokens(source))
    k = SimpleNamespace(**kn)
    fold_shared = _fold_shared(tp, k.SHARED)
    if head is not None and (split or push_moe):
        raise ValueError(
            "head: the chain's last layer in one kernel (its MoE summed in this launch)"
        )
    if (push_moe or prev_moe) and (tp < 2 or split or not fold_shared):
        raise ValueError(
            "push_moe / prev_moe: TP, one kernel, the folded shared expert"
        )
    if prev_moe and first:
        raise ValueError(
            "prev_moe: the step's first layer has no previous launch (plain input)"
        )
    order = _order(kn)
    C = channels(tp, k.XA_PROTO, topk)
    heads, groups = HEADS // tp, OGROUPS // tp
    wo_b_consumer = (
        tp > 1 and not split and k.WO_B_SUM in ("consumer", "2shot")
    )  # seam F sums wo_b's partials
    # the router normalizes its K slices (window layers: no compressed cache, no sources)
    rnorm = k.FFN_NORM == "router" or (
        k.FFN_NORM == "auto" and not sparse and source is None
    )
    shard = residual_layout(tp, split, kn, rows) == "sharded"
    if (
        shard
    ):  # R column-owned: the seams own their columns, LIN gathered, no owner stages
        if (
            k.SEAM_NORM
            or rnorm
            or k.WO_B_SUM != "2shot"
            or order != "fold"
            or not k.R_CHANNEL
        ):
            raise ValueError("RESIDUAL 'sharded': not with SEAM_NORM, FFN_NORM 'router', WO_B_SUM other than '2shot', "
                             "SHARED_ORDER 'ref' or R_CHANNEL off (yet)")  # fmt: skip
        C.ra = _ch("ra", 4 * HIDDEN // tp, Fmt.BF16)  # R_A of the rank's columns
        C.rf = _ch("rf", 4 * HIDDEN // tp, Fmt.BF16)  # R_F of the rank's columns
        C.lin_n = _ch("lin_n", HIDDEN, Fmt.BF16)  # the next layer's LIN ("end")
    rsd = {"residual": "sharded"} if shard else {}
    rq = rs_pack(
        tp, kn, rows
    )  # the two-shot exchanges' partials ("f32" pairs or "f32q")
    rs_f = (
        wo_b_consumer and k.WO_B_SUM == "2shot"
    )  # seam F sums wo_b's reduce-scatter partials (in ``rq``)
    # an Engram layer's seam A after the Engram (its whole output on every rank): LIN and mixes replicated, LIN local
    eng_rep = shard and engram and k.SHARD_ENGRAM_A == "replicated"
    # seam A's fold and LIN at the end of the launch that produced the MoE output (``hcz``, LIN-only), the next
    # launch's seam A the mixes of R as given
    end = shard and k.SHARD_SEAM_A == "end" and (push_moe or prev_moe)
    ga = {"gate_out_arg": "gate_a"} if split else {}
    early = bool(k.HC_EARLY)
    own_q = k.OWN_QUADS
    m_in = (
        _moe_mode(tp, order, rows=rows, chain=True, knobs=kn) if prev_moe else ""
    )  # the previous launch's exchange
    m_out = _moe_mode(tp, order, split, rows, chain=push_moe, knobs=kn)  # this layer's
    if (
        shard
    ):  # the MoE's reduce-scatter partials are the next seam A's (pushed only) or its own owner's (dsum)
        m_in, m_out = ("rs" if prev_moe and not end else ""), "2shot"
    if (
        m_in == "own" and m_out == "push"
    ):  # a chain's tail after "own": its MoE two-shot (owner stage + dsum), so the
        m_out = "2shot"  # program declares moe.part.dag (its seam A reads the previous launch's) and the pinned
        #                  regions keep the chain's layout (moe.part in the rs layout)
    pend_a = chain_pend(tp, rows, order, kn) if prev_moe else "pend"
    if (
        k.HC_DIRECT
    ):  # seam A's fold + LIN a thread a (row, 2 columns), LIN put straight from registers
        ga = {**ga, "direct": True}
    pad = k.HC_LDS_PAD
    if "a" in pad:
        ga = {**ga, "lds_pad": True}
    entry = k.HC_ENTRY is True or (
        k.HC_ENTRY == "auto" and rows is not None and m_in not in ("", "own")
    )
    # what enters a chain from outside it (a first layer's, an unchained layer's h_in) and what leaves it (an unchained
    # layer's R_F, the aux taps' R_A) are whole; only the chain's own hand-offs are the rank's columns
    whole_in = {"res_in_full": True} if shard and not prev_moe else {}
    full_r_a = full_r_a or bool(
        k.FULL_R_A
    )  # (``FULL_R_A``: every layer, the aux taps' form; measurement)
    late = (
        {"res_gather_late": True} if k.R_GATHER == "late" else {}
    )  # (the R gathers off the slices)
    if (
        shard and full_r_a
    ):  # R_A whole too (gathered): the plain r_a, seam F reads the rank's columns' channel
        rsd_a = {**rsd, "res_gather": "rag", "res_full": "r_a", **late}
        res_a = {"res_out": None, "res_ch": C.ra}
    else:
        rsd_a, res_a = rsd, {
            "res_out": "r_a",
            "res_ch": C.ra if k.R_CHANNEL and not split else None,
        }
    seam_a = (HCSeam("hca", "h_in", C.lin_a, "gin", "hca.fn", "hca.scale", "hca.base", gate_out=C.gate_a,
                     early=early, **ga, **rsd, **whole_in)  # (the engine's h_in)
              if first else
              HCSeam("hca", "h_in", C.lin_a, "gin", "hca.fn", "hca.scale", "hca.base", gate_out=C.gate_a,
                     early=early, collapse=False, **{n: v for n, v in ga.items() if n != "direct"}, **rsd)
              if end else
              HCSeam("hca", "h_in", C.lin_a, "gin", "hca.fn", "hca.scale", "hca.base", pend=pend_a, **res_a,
                     gate_out=C.gate_a, early=early, entry=entry, **ga, **rsd_a, **whole_in))  # fmt: skip
    if (
        engram
    ):  # seam A around the Engram (docstring): R -> ra, the gate half -> re, the mixes / LIN on re
        from ..blocks.gfx950.engram import Engram

        rag = _ch(
            "rag", 4 * HIDDEN, Fmt.BF16
        )  # sharded: R gathered for the Engram (its gate needs every column)
        r0 = (
            {"res_gather": rag.name, "hidden": HIDDEN, **rsd}
            if shard
            else {"res_ch": C.ra}
        )
        seam_a = [
            HCSeam("hca0", "h_in", None, "gin", "hca.fn", "hca.scale", "hca.base", r_only=True, **r0, **whole_in,
                   **({} if first else {"pend": pend_a, "res_out": "r_a"})),
            Engram("eng", rag if shard else C.ra, "ids", "pos", "hist", "tmap", 0, out=C.re,
                   out_arg="r_e" if split or not k.R_CHANNEL else "", mode="gate", kv_arg="eng.kv",
                   **({"x_gathered": True} if shard else {})),
            HCSeam("hca", C.re, C.lin_a, "gin", "hca.fn", "hca.scale", "hca.base", gate_out=C.gate_a,
                   early=early, **{n: v for n, v in ga.items() if n != "direct"},
                   **({} if eng_rep else rsd)),  # (R: the Engram's; ``SHARD_ENGRAM_A``)
        ]  # fmt: skip
    else:
        seam_a = [seam_a]
    src = source
    sparse = src.cache_len if src is not None else sparse
    rs = k.RING_SLOTS if ring_slots is None else ring_slots
    smerge = k.SPARSE_SMERGE in (True, 1)
    ckv4 = (
        k.SPARSE_CKV4 if k.SPARSE_CKV4 != "auto" else False
    )  # (no FP4 copy: the cache is ATOM's pool)
    ring8 = (
        (6 if src is None else False) if k.SPARSE_RING8 == "auto" else k.SPARSE_RING8
    )
    # ATOM's KV pool's rows (``KV_FMT``): the compressed layers read and the KV sources write MXFP8 rows with "fp8"
    if k.KV_FMT not in ("bf16", "fp8"):
        raise ValueError(f"KV_FMT 'bf16' or 'fp8', got {k.KV_FMT!r}")
    kv8 = k.KV_FMT == "fp8"
    attn = (MLASparseAttn("attn", C.q, C.kvn, C.o, "ring", "pos", "rope", "attn.sink",
                          C.sel if src is not None and src.index else "sel", "ckv", sparse, heads=heads,
                          new=C.ckvn if src is not None and src.kv else None,
                          ratio=src.ratio if src is not None else 1, ring_slots=rs,
                          ring_first=bool(k.RING_FIRST), new_early=bool(k.NEW_EARLY),
                          dsplit=1 if smerge else k.SPARSE_DSPLIT, smerge=smerge,
                          mheads=(k.SPARSE_MHEADS_SH if shard and k.SPARSE_MHEADS_SH else k.SPARSE_MHEADS) if smerge
                          else 16,
                          wsplit=smerge and k.SPARSE_WSPLIT in (True, 1),
                          ckv4=ckv4, cache_ratio=src.ratio if src is not None else sparse_ratio, ring8=ring8,
                          lean=bool(k.SPARSE_LEAN), vtr=bool(k.SPARSE_VTR), kfirst=bool(k.SPARSE_KFIRST),
                          unrope_f32=bool(k.ATTN_OUT_F32_ROPE), kv8=kv8)
            if sparse else
            MLAWindowAttn("attn", C.q, C.kvn, C.o, "ring", "pos", "rope", "attn.sink", heads=heads,
                          ring_slots=rs, ring_first=bool(k.RING_FIRST), dsplit=k.ATTN_DSPLIT,
                          unrope_f32=bool(k.ATTN_OUT_F32_ROPE)))  # fmt: skip
    comp, idx = [], []
    ctiles = k.SHARD_COMP_TILES if shard and k.SHARD_COMP_TILES else k.COMP_TILES
    if (
        k.IDX_SCORE_F32
        and src is not None
        and src.ratio == 2
        and rows is not None
        and rows >= 18
    ):
        # COMP_TILES 4 sits at 256 VGPRs on the ratio-2 KV + index sources at 18 rows; the f32 scores'
        # program spills one there (133K / 1M buckets, replicated): 2 tiles, as at 24 rows
        ctiles = min(ctiles, 2)
    if (
        src is not None and src.kv
    ):  # beside wq_a / wkv: its dots depend only on attn_norm
        comp = [
            Compressor(
                "comp",
                C.xab,
                src.ratio,
                C.ckvn,
                "pos",
                "rope",
                "ckv",
                src.cache_len,
                kv_state="comp.kv_state" if src.ratio == 2 else "",
                score_state="comp.score_state" if src.ratio == 2 else "",
                slots=src.state_slots,
                index_out=C.ikey if src.index else None,
                kslices=k.COMP_KSLICES,
                tiles=ctiles,
                qat=bool(k.COMPRESS_QAT),
                kv8=kv8,
            )
        ]
    if src is not None and src.index:
        cand = (
            {"cand_out": "cand", "cand_bound": "cand.bound"}
            if src.cand == "source"
            else (
                {"candidates": "cand", "compact": int(k.IDX_COMPACT)}
                if src.cand == "reader"
                else {}
            )
        )
        f32 = bool(
            k.IDX_SCORE_F32
        )  # (f32 scores; the bf16 scores' shortcuts off, schedule entries included)
        idx = [Indexer("idx", C.qn, C.xab, src.ratio, src.n_max, C.sel, "pos", "rope", "index_k", src.cache_len,
                       "idx.bound", new_keys=C.ikey if src.kv else None, kfmt=k.INDEX_KFMT, wq_b_rows=k.IDX_WQB_ROWS,
                       kpack_lanes=bool(k.KPACK_LANES), subtiles=int(k.IDX_SUBTILES), short_tiles=bool(k.IDX_SHORT),
                       live=bool(k.IDX_LIVE), persist=int(k.IDX_PERSIST), rows_per_task=int(k.IDX_ROWS),
                       seg=int(k.IDX_SEG), cseg=int(k.IDX_CSEG), split=tp if k.IDX_SPLIT else 0,
                       punroll=int(k.IDX_PUNROLL), pcap=int(k.IDX_PCAP) // src.ratio,
                       lean_epi=bool(k.IDX_LEAN_EPI) and not f32,
                       pnarrow=bool(k.IDX_PNARROW), clist_sparse=bool(k.IDX_CLIST_SPARSE),
                       topk_packed=bool(k.IDX_TOPK_PACKED) and not f32, cmask_lds=bool(k.IDX_CMASK_LDS),
                       bmax_bf16=bool(k.IDX_BMAX_BF16) and not f32, topk_xhoist=bool(k.IDX_XHOIST),
                       scores_f32=f32, rows_long=int(k.IDX_ROWS_LONG), rcap=int(k.IDX_RCAP) // src.ratio,
                       items=bool(k.IDX_ITEMS), kmfma=k.INDEX_KLAYOUT == "mfma", **cand),
               Store("sel_store", C.sel, "sel_out")]  # fmt: skip
    comp_first = k.COMP_FIRST is True
    q_stage = k.Q_NORM == "stage" or (
        src is not None and src.index
    )  # the Indexer reads q_norm's MXFP8 output
    splits = _norm_splits(k)
    attention = [
        *seam_a,
        *(
            ()
            if k.SEAM_NORM
            else (
                _nq(
                    "attn_norm",
                    C.lin_n if end and prev_moe else C.lin_a,
                    C.xa,
                    "attn_norm.g",
                    HIDDEN,
                    320,
                    8,
                    bf16_out=C.xab if src is not None else None,
                    splits=splits,
                    nsplit=(
                        int(k.SRC_NORM_SPLIT) if src is not None and not split else 0
                    ),
                    gathered=shard and not eng_rep,
                    prev=end and prev_moe,
                ),
            )
        ),
        *(comp if comp_first else ()),
        Linear("wq_a", ("wq_a",), C.xa, QR, BF16Out(C.qr), weight_format="mxfp8"),
        Linear(
            "wkv",
            ("wkv",),
            C.xa,
            KVD,
            BF16Out(C.kv),
            weight_format="mxfp8",
            k_split=k.WKV_KSPLIT,
        ),
        *(() if comp_first else comp),
        *(
            (_nq("q_norm", C.qr, C.qn, "q_norm.g", QR, 160, 4, splits=splits),)
            if q_stage
            else ()
        ),
        _nq(
            "kv_norm",
            C.kv,
            C.kvq,
            "kv_norm.g",
            KVD,
            64,
            4,
            bf16_out=C.kvb,
            splits=splits,
        ),
        *idx,
        Linear(
            "wq_b",
            ("wq_b",),
            (
                C.qn
                if k.Q_NORM == "stage"
                else NormIn(C.qr, "q_norm.g", "q_norm", _nspec(QR, 160, 4))
            ),
            heads * HD,
            BF16Out(C.q),
            weight_format="mxfp8",
            group_rows=k.WQB_ROWS,
        ),
        KVPrep("kvprep", C.kvb, C.kvn, "pos", "rope", kv8=kv8),
        attn,
        RingWrite("ringw", attn),
        # bf16 compute: the reference's wo_a is a plain bf16 einsum (the checkpoint's FP8 times its block scales, the
        # QAT'd o), whose arithmetic the scaled MX MFMA does not reproduce
        Linear(
            "wo_a",
            ("wo_a",),
            C.o,
            groups * ORANK,
            MXFP8Out(C.oa),
            weight_format="mxfp8",
            groups=groups,
            compute=k.WO_A_COMPUTE,
            prewiden=k.WO_A_PREWIDEN if k.WO_A_COMPUTE == "bf16" else 0,
        ),
        Linear(
            "wo_b",
            ("wo_b",),
            C.oa,
            HIDDEN,
            BF16Out(C.ao),
            weight_format="mxfp8",
            tp_partial=(
                (rq if rs_f and k.WO_B_PARTIAL == "f32" else k.WO_B_PARTIAL)
                if tp > 1
                else None
            ),
            tp_sum=(
                ("rs" if k.WO_B_SUM == "2shot" else "consumer")
                if wo_b_consumer
                else "producer"
            ),
        ),
    ]
    pend_f = (PeerPartial("wo_b.tp", "rs" if shard else "2shot", rs_fmt=rq) if rs_f
              else PeerPartial("wo_b.tp", k.WO_B_PARTIAL) if wo_b_consumer
              else "attn_out" if split else C.ao)  # fmt: skip
    rx, rn = (C.lin_f, DotNorm("lfss", "ffn_norm.g", EPS)) if rnorm else (C.xfb, None)
    ffn = [
        # R_A from seam A of this launch: its tagged channel (a plain arg read on another XCD can be stale); the
        # plain arg across launches (split) or the layer input (first)
        HCSeam("hcf", ("r_e" if split or not k.R_CHANNEL else C.re) if engram else
               "h_in" if first or end else "r_a" if split or not k.R_CHANNEL else C.ra, C.lin_f,
               "gate_a" if split else C.gate_a, "hcf.fn", "hcf.scale", "hcf.base", pend=pend_f,
               **({"res_out": None, "res_gather": "rfg", "res_full": "r_f", **late} if shard and not push_moe
                  and not end else {"res_out": None if end else "r_f"}),  # (leaving the chain: R_F whole)
               gate_out=C.gate_f, gate_out_arg="gate_f",
               ssq_out=C.lfss if rnorm else None, own_quads=own_q, early=early,
               res_ch=C.rf if head is not None or end or (shard and not push_moe) else None, lds_pad="f" in pad, **rsd,
               **({"res_in_full": True} if shard and first else {})),
        _nq("ffn_norm", C.lin_f, C.xf, "ffn_norm.g", HIDDEN, 320, 8, bf16_out=None if rnorm else C.xfb,
            splits=splits, gathered=shard),
        *((MoERouter("router", rx, "router.w", "router.bias", C.idx_r, C.wts_r, experts, topk,
                     route_scale=ROUTE_SCALE, kslices=int(k.ROUTER_KSLICES), dot_tiles=k.ROUTER_TILES, norm=rn,
                     scores=k.ROUTER_SCORES),
           Replay("replay", (C.idx, C.wts), after=C.wts_r)) if replay else
          (MoERouter("router", rx, "router.w", "router.bias", C.idx, C.wts, experts, topk,
                     route_scale=ROUTE_SCALE, kslices=int(k.ROUTER_KSLICES), dot_tiles=k.ROUTER_TILES, norm=rn,
                     scores=k.ROUTER_SCORES),)),
        *(() if fold_shared else (
            Linear("se_gu", ("se_w1", "se_w3"), C.xf, 2 * INTER, SwiGLU(C.mid, rounding="fp32", limit=LIMIT),
                   weight_format="mxfp8"),
            Linear("se_d", ("se_w2",), C.mid, HIDDEN, BF16Out(C.shared), weight_format="mxfp8"),
        )),
        MoEExperts("moe", C.xf, C.idx, C.wts, C.moe_out, experts, INTER, limit=LIMIT, dsum=not push_moe,
                   layout=k.MOE_LAYOUT, reduce="push" if m_out == "push" else "2shot", chain_own=m_out == "own",
                   own_quads=(128 if m_out == "own" else 512) if k.MOE_OWN_QUADS == "auto" else k.MOE_OWN_QUADS,
                   ug_g=k.MOE_UG_G, down_g=k.MOE_DOWN_G, rt_poll=k.MOE_RT_POLL,
                   rt_lean=k.MOE_RT_LEAN, ug_ring=int(k.MOE_UG_RING), down_ring=int(k.MOE_DOWN_RING),
                   rs_fmt=rq if m_out != "push" else "f32",
                   # (TP intra: the routed experts' scale codes are ATOM's, read in place: ``hi_weights.expert_packs``)
                   atom_scales=tp > 1 and k.MOE_LAYOUT == "intra",
                   **({"shared_inter": INTER, "shared_order": order} if fold_shared
                      else {"shared": C.shared})),
        *(() if push_moe else (Store("store", C.moe_out, "ffn_out"),)),
        # "end": the next layer's seam A fold + LIN (LIN-only, this launch's MoE rs partials): R -> r_f (the layer's
        # output R, the next launch's h_in), LIN gathered for the next launch's attn_norm
        *((HCSeam("hcz", C.rf, C.lin_n, C.gate_f, "hcf.fn", "hcf.scale", "hcf.base",
                  pend=PeerPartial("moe.part", "rs", rs_fmt=rq), res_out="r_f", mixes=False, early=early,
                  residual="sharded"),) if end else ()),
    ]  # fmt: skip
    return (attention, ffn) if split else attention + ffn


def engram_kv_spill_ok(tp: int) -> bool:
    """``engram_kv_stages``' program: SGPR spills into VGPR lanes allowed past ``schedules.KEY_TP`` (TP8 B3S6:
    1)."""
    return tp > schedules.KEY_TP


def engram_kv_stages(sc: Scenario, names: dict[str, int] | None = None, hist_len: int = 0,
                     wkv_knobs: dict | None = None, knobs: dict | None = None) -> list[Stage]:  # fmt: skip
    """``engram_kv_blocks``' stages, the blocks interleaved by step (every block's look, then every gath, wkv, kvx,
    store): one layer's lookups don't wait behind the other's projection."""
    per = [b.stages(sc) for b in engram_kv_blocks(names, hist_len, wkv_knobs, knobs)]
    steps = ("look", "enc", "gath", "wkv", "kvx", "store")
    return sorted((st for sts in per for st in sts),
                  key=lambda st: next(i for i, n in enumerate(steps) if st.name.split(".")[1] == n))  # fmt: skip


def moe_route_early(stages: list[Stage], knobs: dict | None = None) -> list[Stage]:
    """``stages`` with MoEExperts' route moved up to right after the routing (router / replay stages). It depends only
    on the routing; placed after the shared expert's stages, its one CTA first runs a shared-expert task. Also every
    placement pass every V4.1 program takes (``idx_after_pool``, ``smerge_place``, ``cand_apart``, ``hcf_gate_place``,
    ``wo_a_hold``)
    and the dual-copy mailboxes (``NUMA_DUAL``). ``knobs``: the build's (default: the live knobs).
    """
    kn = knobs or schedules.current()
    sharded = any(
        st.name.endswith(".blk") for st in stages
    )  # a sharded residual's seams: block sums, gates aside
    aside = [st for st in stages if sharded and st.name in SHARD_ASIDE]
    stages = [st for st in stages if st not in aside]
    stages = smerge_place(idx_after_pool(stages, knobs=kn), knobs=kn)
    stages = wo_a_hold(hcf_gate_place(cand_apart(stages, knobs=kn), knobs=kn), kn)
    names = [st.name for st in stages]
    dual = kn["NUMA_DUAL"]
    if (
        dual
    ):  # dual-copy (numa2) mailboxes: every tagged scratch region up to NUMA_DUAL KiB (window layers: 64,
        #      their program is at the SGPR budget: 256 spilled one SGPR on layer 0 at 6 rows)
        names = {st.name for st in stages}
        window = "attn.m" not in names and "attn.c" not in names
        numa_stages(
            stages + aside, (f"auto:{min(int(dual), 64) if window else int(dual)}",)
        )
    return shard_place(stages, aside) if aside else stages


SHARD_ASIDE = (
    "hca.blk",
    "hca.gate",
    "hca.rpush",
    "hca.rfull",
    "hcf.blk",
    "hcf.gate",
    "hcf.rpush",
    "hcf.rfull",
)
# shard_place: (stage, the stage it goes before (program order), the stages it must not share a CTA with)
_EARLY_ATTN = (
    "attn_norm",
    "wq_a",
    "wkv",
    "q_norm",
    "kv_norm",
    "kvprep",
    "attn",
    "attn.c",
    "comp.",
    "idx.",
    "eng.",
)
_SHARD_SLOTS = (
    ("hca.blk", "wq_a", _EARLY_ATTN),
    ("hca.gate", "wo_a", ("wo_a", "wo_b")),
    (
        "hca.rpush",
        "moe.rt",
        ("ffn_norm", "router.", "moe.sh_ug", "moe.route"),
    ),  # (the links idle there)
    ("hca.rfull", "moe.rt", ("ffn_norm", "router.", "moe.sh_ug", "moe.route")),
    ("hcf.blk", "ffn_norm", ("ffn_norm", "router.", "moe.sh_ug", "moe.route")),
    ("hcf.gate", "ffn_norm", ("ffn_norm", "router.", "moe.sh_ug", "moe.route")),
    ("hcf.rpush", "moe.rt", ("ffn_norm", "router.", "moe.sh_ug", "moe.route")),
    ("hcf.rfull", "moe.rt", ("ffn_norm", "router.", "moe.sh_ug", "moe.route")),
)


def shard_place(stages: list[Stage], aside: list[Stage], cus: int = 256) -> list[Stage]:
    """A sharded residual's seams (``RESIDUAL`` "sharded"): every other stage's base where ``plan`` puts it without
    them (the replicated layout's), then each seam's block-sum and gate stage back in program order (``_SHARD_SLOTS``)
    on the first window of CTAs that runs none of the listed stages: their tasks wait for the other ranks' block sums
    (~3 µs after the slices), so no stage on the path may queue behind them on a CTA. Seam A's gate goes right before
    wo_a: its consumer is seam F."""
    nxt = 0
    for st in stages:
        if st.base is None:
            st.base = nxt % cus
        nxt = st.base + st.tasks
    by = {st.name: st for st in aside}
    taken: set[int] = set()
    for name, before, avoid in _SHARD_SLOTS:
        st = by.get(name)
        if st is None:
            continue
        busy = set(taken)
        for o in stages:
            if any(
                o.name == n or (n.endswith(".") and o.name.startswith(n)) for n in avoid
            ):
                busy |= {(o.base + t) % cus for t in range(min(o.tasks, cus))}
        # the first window of CTAs none of the listed stages runs on, else the one with the fewest
        st.base = min(
            range(cus),
            key=lambda c: (len(busy & {(c + t) % cus for t in range(st.tasks)}), c),
        )
        taken |= {(st.base + t) % cus for t in range(st.tasks)}
        at = next((i for i, o in enumerate(stages) if o.name == before), None)
        stages.insert(at, st)
    return stages


def wo_a_hold(stages: list[Stage], knobs: dict | None = None) -> list[Stage]:
    """``WO_A_HOLD``: wo_a's entry hold (``Stage.hold``) on window layers ("auto": 64 when the program has no sparse
    merge and wo_a has at most 8 rows)."""
    hold = (knobs or schedules.current())["WO_A_HOLD"]
    names = {st.name for st in stages}
    for st in stages:
        if st.name != "wo_a":
            continue
        rows = (
            st.emit.keywords["p"].rows
            if hasattr(st.emit, "keywords") and "p" in st.emit.keywords
            else 0
        )
        window = "attn.m" not in names and "attn.c" not in names
        units = (64 if window and 0 < rows <= 8 else 0) if hold == "auto" else int(hold)
        if units:
            st.hold = units
    return stages


def idx_after_pool(
    stages: list[Stage], cus: int = 256, knobs: dict | None = None
) -> list[Stage]:
    """``IDX_AFTER_POOL``: in a one-kernel KV + index source (stages ``comp.pool`` and ``idx.wq_b``, no ``store_ao``)
    the Indexer's first stage starts on the CTA after ``comp.pool``'s last (its tasks no longer wait behind the pool's),
    the later stages following it as ``plan`` places them."""
    v = (knobs or schedules.current())["IDX_AFTER_POOL"]
    names = [st.name for st in stages]
    on = v in (True, 1) or (v == "auto" and "store_ao" not in names)
    if not on or "comp.pool" not in names or "idx.wq_b" not in names:
        return stages
    i, j = names.index("comp.pool"), names.index("idx.wq_b")
    nxt = 0
    for st in stages[: i + 1]:
        nxt = (st.base if st.base is not None else nxt % cus) + st.tasks
    stages[j].base = nxt % cus
    return stages


def smerge_place(
    stages: list[Stage], cus: int = 256, knobs: dict | None = None
) -> list[Stage]:
    """``SPARSE_SMERGE``'s placement (stages ``attn.m``, ``ringw``, ``wo_a`` in a row, as ``plan`` would place them):
    RingWrite on the merge's first CTAs (after their merge tasks: nothing in this launch waits for it) and wo_a on the
    CTAs right after the merge's, so no wo_a task waits behind a merge task to start its weight prefetch; wo_b
    follows as ``plan`` places it."""
    kn = knobs or schedules.current()
    names = [st.name for st in stages]
    if not (kn["SPARSE_SMERGE"] in (True, 1, "auto") and kn["SMERGE_PLACE"]) or not {
        "attn.m",
        "ringw",
        "wo_a",
    } <= set(names):
        return stages
    i, j, w = names.index("attn.m"), names.index("ringw"), names.index("wo_a")
    nxt = 0
    for st in stages[:i]:
        nxt = (st.base if st.base is not None else nxt % cus) + st.tasks
    m = stages[i]
    m.base = nxt % cus
    stages[j].base = m.base
    stages[w].base = (m.base + m.tasks) % cus
    return stages


_APART = ("wq_b", "attn.m", "ringw", "wo_a", "wo_b", "hcf.slice")


def cand_apart(
    stages: list[Stage], cus: int = 256, knobs: dict | None = None
) -> list[Stage]:
    """``IDX_CAND_APART``: in the candidate source (stage ``idx.cmask``), every stage's base as ``plan`` would place it,
    then the attention path's stages (``_APART``: wq_b, the merge with its RingWrite, wo_a, wo_b, seam F's slices)
    from the CTA after the candidate chain's last (``idx.cmask``'s): none waits behind ctopk / cmask, one task a row
    each that end late at long contexts (layer 20's own attention does not read the bitmap). wo_a then shares the
    merge's first CTAs (``SMERGE_PLACE``'s placement given up there). 2: also the selection's TopK above level 0 (its levels,
    the split merge) and ``sel_store`` on the first CTAs from their own that run none of the candidate chain's late
    stages nor ``_APART``'s. "auto" (a build without the kind's knobs: the live ones): the ratio-1 KV source's
    schedule at the cmask stage's rows and bucket (``schedules.LONG_ROWS``)."""
    on = (knobs or schedules.current())["IDX_CAND_APART"]
    names = [st.name for st in stages]
    if "idx.cmask" not in names or not set(_APART) <= set(names):
        return stages
    cm = stages[names.index("idx.cmask")]
    if isinstance(on, str) and on == "auto":
        p = cm.emit.keywords["p"]
        on = schedules.knobs(schedules.KV1, p.rows, tokens=p.n_max * p.ratio)[
            "IDX_CAND_APART"
        ]
    if on not in (True, 1, 2):
        return stages
    nxt = 0
    for st in stages:
        if st.base is None:
            st.base = nxt % cus
        nxt = st.base + st.tasks
    for st in stages:
        if st.name in _APART:
            st.base = (cm.base + cm.tasks) % cus
    if on == 2:
        late = [
            st
            for st in stages
            if st.name.startswith("idx.ctopk.")
            and st.name != "idx.ctopk.l0"
            or st is cm
        ]
        busy = {(st.base + t) % cus for st in late + [st for st in stages if st.name in _APART]
                for t in range(min(st.tasks, cus))}  # fmt: skip
        for st in stages:
            if not (
                st.name.startswith("idx.topk.")
                and st.name != "idx.topk.l0"
                or st.name == "sel_store"
            ):
                continue
            for c in range(st.base, st.base + cus):
                if not busy & {(c + t) % cus for t in range(st.tasks)}:
                    st.base = c % cus
                    break
    return stages


_GATE_AVOID = ("ffn_norm", "router.", "moe.sh_ug", "moe.route")


def hcf_gate_place(
    stages: list[Stage], cus: int = 256, knobs: dict | None = None
) -> list[Stage]:
    """``HCF_GATE_AT``: every stage's base as ``plan`` would place it, then seam F's gate on the CTAs from that one
    ("auto": only when its plan CTAs run one of ``_GATE_AVOID``'s stages, the first window of CTAs from hcf.slice's
    first that runs none); nothing else moves."""
    at = (knobs or schedules.current())["HCF_GATE_AT"]
    names = [st.name for st in stages]
    if at is None or "hcf.gate" not in names or "hcf.slice" not in names:
        return stages
    nxt = 0
    for st in stages:
        if st.base is None:
            st.base = nxt % cus
        nxt = st.base + st.tasks
    gate = stages[names.index("hcf.gate")]
    if not 6 <= gate.tasks < 16:
        return stages
    busy = set()
    for st in stages:
        if st.name.startswith(_GATE_AVOID):
            busy |= {(st.base + t) % cus for t in range(min(st.tasks, cus))}
    if not busy & {(gate.base + t) % cus for t in range(gate.tasks)}:
        return stages
    start = stages[names.index("hcf.slice")].base
    for c in range(start, start + cus):
        if not busy & {(c + t) % cus for t in range(gate.tasks)}:
            gate.base = c % cus
            break
    return stages


ENGRAM_KV = {
    "eng1": 1,
    "eng14": 14,
}  # the token-only halves' block names -> their layers (``engram_kv_blocks``)


def engram_kv_blocks(names: dict[str, int] | None = None, hist_len: int = 0,
                     wkv_knobs: dict | None = None, knobs: dict | None = None) -> list[Block]:  # fmt: skip
    """The Engram's token-only half (``Engram(mode="kv")``: hash, lookup, wkv, at TP the kv all-gathered) for each
    of ``names`` (block name -> layer; default ``ENGRAM_KV``), for a launch before the Engram layer's own (layer 0's,
    or a small launch of its own). Block ``n`` writes plain ``n.kv`` [rows, 25600] bf16: the layer's ``eng.kv``.
    Args: ``ids`` int32 [rows] (the step's tokens), ``pos``, ``hist`` int32 [batch, ``hist_len``] (compressed ids by
    position; this launch writes its rows'), ``tmap``, per block ``n.hash`` / ``n.tq`` / ``n.ts`` and wkv
    (``engram_io.pack_kv``); ``wkv_knobs``: wkv's ``Linear`` knobs (default the plan's); ``knobs``: the build's
    (``ENGRAM_LOOK_DST``, ``ENGRAM_STORE_CHUNKS``, ``ENGRAM_STAGED``; default the live knobs).
    """
    from ..blocks.gfx950.engram import Engram

    kn = knobs or schedules.current()
    return [Engram(n, C.ra, "ids", "pos", "hist", "tmap", hist_len, mode="kv", kv_arg=f"{n}.kv", wkv_knobs=wkv_knobs,
                   look_dst=kn["ENGRAM_LOOK_DST"], store_chunks=kn["ENGRAM_STORE_CHUNKS"],
                   staged=f"{n}.rows" if kn["ENGRAM_STAGED"] else "")
            for n in (ENGRAM_KV if names is None else names)]  # fmt: skip


def cache_len(ctx: int, ratio: int) -> int:
    """The compressed positions a step can see at context ``ctx`` (+ 16: a verify's rows), a multiple of 32: a
    compressed layer's cache length and the Indexer's ``n_max`` (as the engine hook sizes them).
    """
    return -(-((ctx + 16) // ratio) // 32) * 32

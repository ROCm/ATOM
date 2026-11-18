# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``HCSeam``: DeepSeek-V4.1's manifold-constrained hyper-connections (mHC) at a sub-layer seam.

The residual is ``HC`` (4) streams of the hidden width. Each sub-layer (attention, then the MoE) reads one
row collapsed from the streams and adds its output back into all of them, mixed (checkpoint
``inference/model.py`` ``Block``):

- ``hc_mixes(x)`` -> (pre, post, comb): rstd over the flattened 4 x hidden row, ``MIX`` = 24 fp32 dots with
  ``hc_fn`` [24, 4 hidden], then ``hc_split_sinkhorn`` with ``hc_scale`` [3] / ``hc_base`` [24]: pre =
  sigmoid(.) + eps, post = 2 sigmoid(.), comb = a 4 x 4 softmax then ``SINKHORN`` row / column normalizations;
- ``hc_pre(x, pre)`` = bf16(sum_j pre[j] x[j]): the sub-layer's input;
- ``hc_post(y, x, post, comb)`` = bf16(post[o] y + sum_i comb[i, o] x[i]) for each stream o.

The mixes computed from a sub-layer's residual are used one seam later: a seam's input row uses the *previous*
seam's pre, and its post / comb fold the previous sub-layer's output in. So one seam does, per row, from the
previous seam's gate (24 values: pre | post | comb):

1. ``fold``: R = hc_post(pend, R_prev, post, comb), the residual after the previous sub-layer (or, without
   ``fold``, R = the given streams as they are: a layer-0 input);
2. LIN = hc_pre(R, pre): the sub-layer's input row, a BF16 channel;
3. this seam's gate (24 values, an F32 channel and / or a plain arg) = hc_split_sinkhorn(hc_mixes(R)).

Two stages (the task shape of ATOM's V4.1 K1 ``slice`` / ``gate``, ``atom/models/deepseek_v41/mono/kernels/
attn_pre.py``, MIT, adapted): ``slice`` tasks own ``COLS`` hidden columns of all 4 streams for every row
(R, LIN, and the row's partial mix projection and sum of squares, an exchange region); a ``gate`` task a
row sums the partials in slice order and runs the Sinkhorn. Arithmetic: the reference's fp32 (fold and
collapse in stream order, one rounding each; true divisions in the Sinkhorn), the 24 dots on the fp32 MFMA;
the split-K sums differ from torch's only in order.

**The norm's partial sums** (``ssq_out``, an F32 channel of width / 16): the slice also publishes, per row and 16
LIN columns, the sum of squares of the bf16 LIN words in a ``NormQuant`` thread's order (``v_dot2_f32_bf16`` word by
word from 0): the per-thread partials of ``NormSpec(threads=width / 16, wps=8)``. A consumer can then finish the RMSNorm
of its own K slice (``GatheredDot(norm=DotNorm(...))``) with the norm stage's bits, without waiting for that stage.

**R only** (``r_only=True``): the slice stage does step 1 alone, R into ``res_ch`` (and ``res_out`` with a fold),
no LIN, mixes or gate stage; without a fold R is ``res_in`` copied into ``res_ch``. A seam split around a step on R
(V4.1's Engram before layers 1 and 14: the fold, the Engram on R, then a seam without a fold reading its output).

**LIN only** (``mixes=False``): the slice stage does steps 1 and 2 (the fold and LIN, and ``ssq_out``), no mix
projection and no gate stage (no ``fn`` / ``scale`` / ``base`` args are read); ``res_out`` is optional. The collapse
before a head: the last sub-layer's output folded in and the streams collapsed with the last FFN seam's pre (the
reference's ``hc_pre(h, pre_mix)`` after the last layer), nothing mixed for a next sub-layer.

**pend from TP partials** (``pend=PeerPartial(region, fmt)``): the previous sub-layer is a row-parallel
``Linear(tp_sum="consumer")`` that only pushed its partials; the fold sums the ranks' partials of its columns
itself (``peer.sum_partials``, rank order, fp32) and rounds to bf16 (the producer's ``BF16Out`` bits), its R
loads already in flight. One hop and the producer's wait for the slowest rank fewer.

**Sharded residual** (``residual="sharded"``, TP): R is column-owned. Rank r holds columns r * width / tp .. of every
stream (its R args and channels are [rows, HC, width / tp]) and its seam runs only the slices of those columns
(``slices / tp`` of them), each as ``tp`` tasks, one a destination rank: the fold of the previous sub-layer's output,
which arrives as two-shot reduce-scatter partials of the rank's own columns (``PeerPartial(fmt="rs")``: summed here in
rank order, then bf16: the owner stage's bits, with no owner stage), and the collapse, LIN put into that rank's slot of
the gathered PEER region ``lin`` (``peer.tile_local``; readers: ``NormQuant(src_gathered=)``). The destination = this
rank's task also writes R (``res_out`` / ``res_ch``, this rank's columns) and the mixes' per-slice partials (local);
a ``<name>.blk`` stage sums them in blocks of ``gate_block`` slices (the gate's own first sums) and puts the block sums
into every rank's slot of ``<name>.pmixg``, from which the gate takes the blocks in slice order: every output the same
bits as the replicated seam's. No R, no owner stage and no mix partials of other ranks' columns cross a link.
A sharded seam can also run as its two halves in two launches: ``mixes=False`` (the fold and LIN, at the end of the
launch that produced the sub-layer output) and ``collapse=False`` (R as given, the mixes only, in the next launch:
its sub-layer input is the gathered LIN the previous launch put, ``NormQuant(src_prev=True)``).
Where something needs the whole R (an Engram gate, a hand-off out of a chain): ``res_gather`` also puts every task's
R into its destination's slot of a gathered PEER region (tiled as LIN, rows x HC rows), and ``res_full`` copies that
region into a plain [rows, HC, width] arg (stage ``<name>.rfull``); a sharded seam may read a whole-width R
(``res_in`` of width HC x width: e.g. the Engram's output) at its own columns.
"""

import math
from dataclasses import dataclass, replace
from functools import partial

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from ...device.gfx950 import buffer_ops as bo
from ...device.gfx950 import channels as chan
from ...device.gfx950 import mailbox as mb
from ...device.gfx950 import peer
from ...device.gfx950.ops import (
    CM_DEV,
    CM_SYS,
    bf2_f32,
    bf16_pair,
    bf16_round,
    butterfly,
    rsrc,
    traced,
)
from ...plan.program import Region, Scenario, Stage
from ...plan.trace import Space, note
from ...target import MI355X
from ..base import Block, Channel, Fmt
from .norm import dot2_bf16

THREADS, WAVES = MI355X.threads, MI355X.waves
HC = 4
MIX = HC * (HC + 2)  # 24: pre 4 | post 4 | comb 16
PART = MIX + 1  # a slice's partials a row: the 24 dots, then the sum of squares
COLS = 32  # hidden columns a slice task
KT = HC * COLS  # projection columns a slice task (128)
GATE_BLOCK = 8  # slices a gate thread sums (``gate_block``: fewer where a rank's share of the slices needs it)
TILE = 16  # rows an MFMA pass (its N)
COMB0 = 16  # comb[i][o] on gate lane COMB0 + 4 i + o: a row sum is lanes ^ 1, 2, a column sum ^ 4, 8
GATHER_TILE = (
    COLS // 2
)  # pairs: a sharded seam's gathered LIN tile (a slice task's columns)
LOG2E = math.log2(math.e)


@dataclass(frozen=True)
class PeerPartial:
    """A fold's ``pend`` as TP partials: the PEER region a ``Linear(tp_sum="consumer")`` pushes into
    (``<name>.tp``) and their format ("bf16" | "f32" | "f32q"; "ref3": ``MoEExperts``' two fields, routed + shared,
    ``peer.sum_partials3``: bf16(shared) added after the routed sum). ``prev_launch``: pushed by the *previous
    launch* (the previous layer's ``MoEExperts(dsum=False)``, ``<name>.part``): its tag (this one's - 1, the same
    step's previous layer) and parity slot, declared as the slice stage's ``Stage.prev_reads``. "ag": the owners'
    bf16 sums of a two-shot exchange, already gathered (``peer.read_ag``: the previous launch's
    ``MoEExperts(chain_own=True)``, region ``<name>.part.dag``); no owner stage here. "rs" (a sharded residual's seam,
    ``HCSeam(residual="sharded")``): the ranks' fp32 reduce-scatter partials of this rank's own columns (the
    ``push_rs`` layout: ``Linear(tp_sum="rs")``, ``MoEExperts(reduce="2shot")``'s combine), summed in the slice.
    ``rs_fmt`` ("rs" / "2shot"): the reduce-scatter partials' format, "f32" (value, tag) pairs or "f32q" / "f32p"
    packed (the producer's ``tp_partial`` / ``rs_fmt``), summed in rank order either way: the
    same bits."""

    region: str
    fmt: str
    prev_launch: bool = False
    rs_fmt: str = "f32"


@dataclass(frozen=True)
class SeamNorm:
    """The sub-layer's input norm fused into the seam's slices (``HCSeam(norm=)``): out = MXFP8(RMSNorm(LIN) * gamma)
    as ``NormQuant(spec)`` would (and its BF16 copy ``bf16_out``). Every slice puts its rows' sum-of-squares partials
    (``ssq_out``: NormQuant's thread partials), polls the 320 partials of every row, sums them in NormQuant's order
    (``norm.row_rstd``) and quantizes its own 32 columns (one MXFP8 group a row): the same bits as the norm stage,
    without its hop (the stage polled the whole LIN row from every slice). ``spec``: one task a row, threads =
    width / 16, 8 words a thread (the ``ssq_out`` partials' map)."""

    out: Channel
    gamma: str
    spec: object
    bf16_out: Channel | None = None


def gate_block(slices: int, tp: int) -> int:
    """Slices a gate thread sums first (the gate's block sums): ``GATE_BLOCK``, or the largest of 4, 2, 1 that splits
    each of ``tp`` ranks' share of the slices into whole blocks (V4.1's 160 slices at TP8: 4). A function of (slices,
    tp) whatever the residual layout, so a sharded seam keeps the replicated seam's bits at every tp; TP1-4 keep 8.
    """
    return next(b for b in (GATE_BLOCK, 4, 2, 1) if slices % (tp * b) == 0)


@dataclass(frozen=True)
class _Plan:
    name: str
    rows: int
    width: int  # hidden
    fold: bool
    res_in: str  # the previous seam's residual (or, without fold, R itself), bf16 [rows, HC, width]: a plain arg,
    #              or (``res_in_ch``) a tagged BF16 channel of width HC x width written earlier in this launch
    pend: str  # fold: the previous sub-layer's output row (a BF16 channel, or a plain bf16 [rows, width] arg)
    pend_arg: bool
    gate_in: str  # the previous seam's gate (an F32 channel of width MIX, or a plain f32 [rows, MIX] arg)
    gate_in_arg: bool
    res_out: str  # fold: plain bf16 [rows, HC, width] for R ("" if none)
    lin: str  # BF16 channel of width ``width``: the sub-layer's input row
    gate_out: str  # F32 channel of width MIX ("" if none)
    gate_out_arg: str  # plain f32 [rows, MIX] ("" if none)
    fn: str  # f32 [MIX, HC width]
    scale: str  # f32 [3]
    base: str  # f32 [MIX]
    eps: float  # the rstd's (norm_eps)
    hc_eps: float
    sinkhorn: int
    pend_fmt: str = (
        ""  # pend from TP partials (``PeerPartial``): their format; "" otherwise
    )
    pend_tp: int = 1
    pend_prev: bool = False  # ... pushed by the previous launch
    res_in_ch: bool = (
        False  # res_in is a tagged BF16 channel (R written earlier in this launch), not a plain arg
    )
    res_ch: str = (
        ""  # fold: R also into this tagged BF16 channel (a later seam of this launch reads it)
    )
    ssq: str = (
        ""  # the norm's partial sums of LIN (an F32 channel of width / 16, module docstring); "" if none
    )
    own_quads: int = (
        THREADS  # two-shot owner: element quads a task (a thread each): fewer, more tasks pull less each
    )
    r_only: bool = (
        False  # the slice writes R only (``res_ch`` / ``res_out``): no LIN, mixes or gate stage
    )
    early: bool = (
        False  # the previous gate's values loaded at task start, before the pend poll (``HCSeam(early=)``)
    )
    # ``norm`` (``SeamNorm``): the sub-layer's RMSNorm + MXFP8 in the slices (module docstring); "" / None: none
    direct: bool = (
        False  # ``HCSeam(direct=)``: a thread folds all 4 streams of (row, 2 columns) and puts LIN at once
    )
    norm_q: str = ""  # the MXFP8 output channel
    norm_dense: bool = False  # ... a bulk + flag (FLAG) channel
    norm_bf: str = ""  # a BF16 copy of the normalized row ("" if none)
    gamma: str = ""  # the norm's weight (bf16 [width])
    nspec: object = (
        None  # its ``NormSpec`` (the sum order: one task a row, threads = width / 16, 8 words)
    )
    entry: bool = (
        False  # the first task's loads + pend attempts issued at the stage's entry (``HCSeam(entry=)``)
    )
    mixes: bool = (
        True  # False: the slice's fold and LIN only (no mix projection, no gate stage; module docstring)
    )
    kp: int = (
        KT  # LDS row pitch (words) of the task's R rows and hc_fn rows (``HCSeam(lds_pad=)``: KT + 4)
    )
    gtp: int = (
        1  # ``residual="sharded"``: the TP ranks R is column-owned over (module docstring); 1: replicated
    )
    collapse: bool = (
        True  # False: no LIN (``HCSeam(collapse=False)``: the mixes of R as given only)
    )
    rin_full: bool = (
        False  # sharded: res_in holds the whole width (read at this rank's columns)
    )
    rg: str = ""  # sharded: R gathered into this PEER region too (``res_gather``)
    rfull: str = ""  # sharded: ... and copied whole into this plain arg (``res_full``)
    rg_late: bool = (
        False  # ... pushed by a stage of its own (``<name>.rpush``) from ``res_ch``, not by the slices
    )
    gb: int = GATE_BLOCK  # slices a gate block (``gate_block``)
    pend_rs: str = (
        "f32"  # pend_fmt "rs" / "2shot": the reduce-scatter partials' format ("f32" | "f32p" | "f32q")
    )

    @property
    def slices(self):
        return self.width // COLS

    @property
    def owned(self):  # a sharded residual's seam: this rank's columns only
        return self.gtp > 1

    @property
    def fan(
        self,
    ):  # sharded with LIN or a gathered R: a task a (slice, destination rank)
        return self.owned and (
            (self.collapse and not self.r_only) or (self.rg != "" and not self.rg_late)
        )

    @property
    def ns(self):  # the slices this rank runs (sharded: its own columns')
        return self.slices // self.gtp

    @property
    def rw(self):  # R's columns a stream on this rank
        return self.width // self.gtp

    @property
    def nb(self):  # sharded: the gate blocks of this rank's slices
        return self.ns // self.gb

    @property
    def pmix(self):
        return f"{self.name}.pmix"

    @property
    def pmixg(self):  # sharded: the gathered block sums of the mix partials (PEER)
        return f"{self.name}.pmixg"

    @property
    def rl_words(
        self,
    ):  # LDS: R of the task's columns, [row][stream][COLS] f32 (row pitch kp)
        return self.rows * self.kp

    @property
    def fl(self):  # LDS word: the task's hc_fn columns [MIX][KT] f32 (row pitch kp)
        return self.rl_words

    @property
    def red(self):  # LDS word: MFMA partials [wave][2][64 lanes][4]
        return self.fl + MIX * self.kp

    @property
    def sq(
        self,
    ):  # LDS word: the task's LIN words [row][COLS / 2] (``ssq``), after the MFMA partials (LIN only: R)
        return self.red + WAVES * 2 * 64 * 4 if self.mixes else self.rl_words

    @property
    def lds_words(self):
        if self.r_only:
            return 0
        return self.nred + (self.rows * WAVES if self.norm_q else 0)

    @property
    def nred(
        self,
    ):  # LDS word: the norm's per-row wave partials (``norm.row_rstd``), after the LIN words
        return self.sq + (self.rows * COLS // 2 if self.ssq else 0)


class HCSeam(Block):
    """One seam's slice and gate stages (module docstring)."""

    def __init__(
        self,
        name: str,
        res_in: str | Channel,
        lin: Channel,
        gate_in: Channel | str,
        fn: str,
        scale: str,
        base: str,
        pend: Channel | str | None = None,
        res_out: str | None = None,
        gate_out: Channel | None = None,
        gate_out_arg: str | None = None,
        res_ch: Channel | None = None,
        ssq_out: Channel | None = None,
        own_quads: int = THREADS,
        eps: float = 1e-20,
        hc_eps: float = 1e-6,
        sinkhorn: int = 20,
        r_only: bool = False,
        early: bool = False,
        norm: SeamNorm | None = None,
        direct: bool = False,
        entry: bool = False,
        mixes: bool = True,
        lds_pad: bool = False,
        residual: str = "replicated",
        collapse: bool = True,
        res_gather: str | None = None,
        res_full: str | None = None,
        hidden: int | None = None,
        res_in_full: bool = False,
        res_gather_late: bool = False,
    ):
        if residual not in ("replicated", "sharded"):
            raise ValueError(
                f"HCSeam {name}: residual 'replicated' or 'sharded', got {residual!r}"
            )
        sharded = residual == "sharded"
        if (
            r_only
        ):  # R alone: into its channel (sharded: or a gathered region); no LIN (``lin`` None), mixes, gate
            if (res_ch is None and not res_gather and not (residual == "sharded" and res_out)) or lin is not None \
                    or gate_out is not None or gate_out_arg or ssq_out is not None:  # fmt: skip
                raise ValueError(
                    f"HCSeam {name}: r_only writes R to res_ch only (lin / gate / ssq outputs None)"
                )
            width = (
                res_ch.width // HC
                if res_ch is not None and residual != "sharded"
                else hidden or -1
            )
        else:
            width = lin.width if lin is not None else -1
        if (
            not r_only and (lin is None or lin.fmt is not Fmt.BF16 or lin.flagged)
        ) or width % (COLS * GATE_BLOCK):
            raise ValueError(
                f"HCSeam {name}: lin must be a tagged BF16 channel, width a multiple of {COLS * GATE_BLOCK}"
            )
        for ch, w in ((gate_in, MIX), (gate_out, MIX)):
            if isinstance(ch, Channel) and (
                ch.fmt is not Fmt.F32 or ch.width != w or ch.flagged
            ):
                raise ValueError(
                    f"HCSeam {name}: gate channels are tagged F32 of width {MIX}"
                )
        if isinstance(pend, PeerPartial) and pend.fmt not in (
            "bf16",
            "f32",
            "f32q",
            "ref3",
            "2shot",
            "ag",
            "rs",
        ):
            raise ValueError(
                f"HCSeam {name}: PeerPartial fmt 'bf16', 'f32', 'f32q', 'ref3', '2shot', 'ag' or 'rs', got {pend.fmt!r}"
            )
        if isinstance(pend, PeerPartial) and (pend.fmt == "rs") != sharded:
            raise ValueError(
                f"HCSeam {name}: a sharded residual's fold takes 'rs' partials (and only it does)"
            )
        if isinstance(pend, PeerPartial) and (pend.rs_fmt not in ("f32", "f32q", "f32p") or
                                              (pend.rs_fmt != "f32" and pend.fmt not in ("rs", "2shot"))):  # fmt: skip
            raise ValueError(f"HCSeam {name}: PeerPartial rs_fmt 'f32' or ('rs' / '2shot') 'f32q' / 'f32p', got "
                             f"{pend.rs_fmt!r}")  # fmt: skip
        if sharded and (norm is not None or ssq_out is not None):
            raise ValueError(f"HCSeam {name}: residual 'sharded': no norm, ssq_out")
        if sharded and r_only and not hidden:
            raise ValueError(
                f"HCSeam {name}: a sharded r_only seam takes the width (hidden=)"
            )
        if (res_gather or res_full) and not sharded or res_full and not res_gather:
            raise ValueError(
                f"HCSeam {name}: res_gather (and res_full from it) are a sharded seam's"
            )
        if not collapse and (
            not sharded or pend is not None or not mixes or direct or res_ch is not None
        ):
            raise ValueError(
                f"HCSeam {name}: collapse=False: a sharded seam's mixes of R as given (no fold, R out)"
            )
        if isinstance(pend, Channel) and (
            pend.fmt is not Fmt.BF16 or pend.width != width or pend.flagged
        ):
            raise ValueError(
                f"HCSeam {name}: pend must be a tagged BF16 channel of width {width}"
            )
        for ch in (res_in, res_ch):  # (sharded: HC x width / tp, checked by ``plan``)
            if isinstance(ch, Channel) and (ch.fmt is not Fmt.BF16 or (ch.width != HC * width and not sharded)
                                            or ch.flagged):  # fmt: skip
                raise ValueError(
                    f"HCSeam {name}: R channels are tagged BF16 of width {HC} x {width}"
                )
        if ssq_out is not None and (
            ssq_out.fmt is not Fmt.F32 or ssq_out.width * 16 != width or ssq_out.flagged
        ):
            raise ValueError(
                f"HCSeam {name}: ssq_out is a tagged F32 channel of width {width // 16}"
            )
        if res_ch is not None and pend is None and not r_only:
            raise ValueError(f"HCSeam {name}: res_ch is R of a fold (or of r_only)")
        if not mixes and (r_only or gate_out is not None or gate_out_arg):
            raise ValueError(
                f"HCSeam {name}: mixes=False writes LIN (and R / ssq) only: no gate outputs, not r_only"
            )
        if (pend is None) != (res_out is None) and not (
            pend is not None and (not mixes or res_ch is not None or res_gather)
        ):
            raise ValueError(
                f"HCSeam {name}: a fold takes the previous output (pend) and writes R (res_out)"
            )
        self.name, self.res_in, self.lin, self.gate_in, self.pend = (
            name,
            res_in,
            lin,
            gate_in,
            pend,
        )
        self.res_out, self.gate_out, self.gate_out_arg = (
            res_out or "",
            gate_out,
            gate_out_arg or "",
        )
        if not 0 < own_quads <= THREADS or own_quads % 64:
            raise ValueError(
                f"HCSeam {name}: own_quads a multiple of 64 up to {THREADS}"
            )
        self.res_ch, self.ssq_out, self.own_quads = res_ch, ssq_out, own_quads
        self.r_only, self.width, self.mixes = r_only, width, mixes
        # the previous gate's values (post / comb for the fold, pre for the collapse) issued at the task's start, with
        # the R loads, ahead of the pend poll, and checked where they are used: one round trip instead of up to three
        # (a gate channel's poll, then the pend's, then the collapse's pre poll after the barrier); the same bits
        self.early = early
        self.norm = norm
        # a thread a (row, 2 columns): the fold of all 4 streams in registers and LIN put straight from them (no LDS
        # round and CTA barrier before the LIN put: the put is the seam's critical output); the same operations
        if direct and (
            r_only or isinstance(res_in, Channel) or not isinstance(gate_in, str)
        ):
            raise ValueError(
                f"HCSeam {name}: direct needs R and the gate from plain args (seam A), not r_only"
            )
        self.direct = direct
        # the slice's R and hc_fn rows in LDS at a pitch of KT + 4 words: the mixes' MFMA operand reads (16 hc_fn rows,
        # or the step's token rows, KT = 128 words apart: one bank) were 16-way / rows-way bank conflicts; only LDS
        # addresses change (the same bits).
        self.lds_pad = lds_pad
        # a chained seam's first poll (the previous launch's gathered sums, ``PeerPartial("ag", prev_launch=True)``) and
        # the task's R / previous-gate loads issued at the stage's entry (``Stage.pre``), before the task loop's
        # address set-up: they are in flight from the launch's first instructions instead of ~0.45 µs later; the same
        # bits. Only where it applies (a fold of plain R / gate args); elsewhere off.
        self.entry = entry
        self.residual = residual  # module docstring, "Sharded residual"
        self.collapse = collapse
        self.res_gather, self.res_full = res_gather or "", res_full or ""
        # sharded: ``res_in`` (a plain arg) holds the whole width (a chain's first layer: the engine's whole streams)
        self.res_in_full = res_in_full
        # the R gather off the slices: a stage of its own pushes R from ``res_ch`` (placed by the composition off the
        # path: the slices' LIN puts don't share the links with it)
        if res_gather_late and (not res_gather or res_ch is None):
            raise ValueError(
                f"HCSeam {name}: res_gather_late pushes the gathered R from res_ch"
            )
        self.res_gather_late = res_gather_late
        self.fn, self.scale, self.base = fn, scale, base
        self.eps, self.hc_eps, self.sinkhorn = eps, hc_eps, sinkhorn

    def plan(self, sc: Scenario) -> _Plan:
        def name_of(x):
            return (
                x
                if isinstance(x, str)
                else (x.region if isinstance(x, PeerPartial) else x.name)
            )

        peer_pend = isinstance(self.pend, PeerPartial)
        if peer_pend and sc.tp < 2:
            raise ValueError(
                f"HCSeam {self.name}: a PeerPartial pend needs a TP scenario"
            )
        gtp = sc.tp if self.residual == "sharded" else 1
        rin_full = gtp > 1 and (
            self.res_in_full
            or (
                isinstance(self.res_in, Channel)
                and self.res_in.width == HC * self.width
            )
        )
        if gtp > 1:
            for ch in (self.res_in, self.res_ch):
                if (
                    isinstance(ch, Channel)
                    and ch.width != HC * self.width // gtp
                    and not (ch is self.res_in and rin_full)
                ):
                    raise ValueError(
                        f"HCSeam {self.name}: a sharded residual's R channels are {HC} x width / tp"
                    )
            if self.width // COLS % gtp:
                raise ValueError(f"HCSeam {self.name}: sharded: whole slices a rank")
        return _Plan(
            self.name, sc.rows, self.width, self.pend is not None, name_of(self.res_in),
            name_of(self.pend) if self.pend is not None else "", isinstance(self.pend, str),
            name_of(self.gate_in), isinstance(self.gate_in, str), self.res_out,
            self.lin.name if self.lin is not None else "",
            self.gate_out.name if self.gate_out is not None else "", self.gate_out_arg,
            self.fn, self.scale, self.base, self.eps, self.hc_eps, self.sinkhorn,
            self.pend.fmt if peer_pend else "", sc.tp if peer_pend else 1, peer_pend and self.pend.prev_launch,
            isinstance(self.res_in, Channel), self.res_ch.name if self.res_ch is not None else "",
            self.ssq_out.name if self.ssq_out is not None else "", self.own_quads, self.r_only, self.early,
            direct=self.direct, norm_q=self.norm.out.name if self.norm else "",
            norm_dense=bool(self.norm and self.norm.out.dense),
            norm_bf=self.norm.bf16_out.name if self.norm and self.norm.bf16_out is not None else "",
            gamma=self.norm.gamma if self.norm else "", nspec=self.norm.spec if self.norm else None,
            entry=self.entry and peer_pend and self.pend.fmt in ENTRY_FMTS and self.pend.prev_launch
            and isinstance(self.res_in, str) and isinstance(self.gate_in, str) and not self.r_only
            and self.width // COLS <= MI355X.cus,
            mixes=self.mixes,
            kp=KT + 4 if self.lds_pad else KT,
            gtp=gtp,
            collapse=self.collapse,
            rin_full=rin_full,
            rg=self.res_gather,
            rfull=self.res_full,
            rg_late=self.res_gather_late,
            gb=gate_block(self.width // COLS, sc.tp),
            pend_rs=self.pend.rs_fmt if peer_pend else "f32",
        )  # fmt: skip

    def stages(self, sc: Scenario) -> list[Stage]:
        p = self.plan(sc)
        sname, gname = f"{self.name}.slice", f"{self.name}.gate"
        s_args = ([] if p.res_in_ch else [p.res_in]) + (
            [] if p.r_only or not p.mixes else [p.fn]
        )
        if p.fold and p.pend_arg:
            s_args.append(p.pend)
        if p.gate_in_arg and (p.fold or not p.r_only):
            s_args.append(p.gate_in)
        if p.res_out:
            s_args.append(p.res_out)
        if (
            p.owned
        ):  # LIN gathered: a slot a rank (its columns, tiled); R of this rank's columns
            s_writes = (
                [Region(p.lin, p.rows * p.rw // 2, (sname,), space=Space.PEER)]
                if p.lin and p.collapse
                else []
            )
        else:
            s_writes = (
                list(self.lin.regions(p.rows, (sname,))) if self.lin is not None else []
            )
        if self.res_ch is not None:
            s_writes += list(self.res_ch.regions(p.rows, (sname,)))
        if (
            p.rg and not p.rg_late
        ):  # R gathered: a slot a rank (its columns, rows x HC rows, tiled)
            s_writes.append(
                Region(p.rg, p.rows * HC * p.rw // 2, (sname,), space=Space.PEER)
            )
        if (
            self.ssq_out is not None
        ):  # (with norm: polled by the slices too, an exchange)
            s_writes += [
                replace(r, exchange=self.norm is not None)
                for r in self.ssq_out.regions(p.rows, (sname,))
            ]
        if not p.r_only and p.mixes:
            s_writes.append(
                Region(p.pmix, p.ns * p.rows * PART, (sname,), exchange=not p.owned)
            )
        g_args = [p.scale, p.base] + ([p.gate_out_arg] if p.gate_out_arg else [])
        g_writes = (
            list(self.gate_out.regions(p.rows, (gname,)))
            if self.gate_out is not None
            else []
        )
        own = []
        if (
            p.pend_fmt == "2shot"
        ):  # the owner's sums of its column quarter, gathered as bf16 (``_own``)
            oname = f"{self.name}.own"
            ag = Region(f"{p.pend}.ag", peer.shot2_pairs(p.rows, p.width, p.pend_tp, "bf16"), (oname,),
                        space=Space.PEER)  # fmt: skip
            tasks = p.pend_tp * -(
                -p.rows * p.width // p.pend_tp // 4 // p.own_quads
            )  # a destination rank a task
            own = [Stage(oname, tasks, partial(_own, p=p), writes=(ag,), key=(p,) + ("own",),
                         prev_reads=(p.pend,) if p.pend_prev else ())]  # fmt: skip
        sl = [Stage(sname, p.ns * (p.gtp if p.fan else 1), partial(_slice, p=p), lds_words=p.lds_words,
                    writes=tuple(s_writes), args=tuple(s_args), key=(p,) + ("slice",),
                    pre=partial(_entry, p=p) if p.entry else None,
                    prev_reads=(p.pend,) if p.pend_prev and p.pend_fmt != "2shot" else ())]  # fmt: skip
        if (
            p.rg_late
        ):  # R pushed from res_ch by a stage of its own: a task a RPUSH_SLICES slices, every destination
            pname = f"{self.name}.rpush"
            sl.append(Stage(pname, p.ns // RPUSH_SLICES, partial(_rpush, p=p), key=(p,) + ("rpush",),
                            writes=(Region(p.rg, p.rows * HC * p.rw // 2, (pname,), space=Space.PEER),)))  # fmt: skip
        if p.rfull:  # the gathered R whole into the plain arg: a task a (row, stream)
            sl.append(Stage(f"{self.name}.rfull", p.rows * HC, partial(_rfull, p=p), args=(p.rfull,),
                            key=(p,) + ("rfull",)))  # fmt: skip
        if p.r_only or not p.mixes:
            return own + sl
        if (
            p.owned
        ):  # the block sums of this rank's mix partials, to every rank (module docstring)
            bname = f"{self.name}.blk"
            sl.append(Stage(bname, p.nb * p.gtp, partial(_blk, p=p), key=(p,) + ("blk",),
                            writes=(Region(p.pmixg, p.nb * p.rows * PART, (bname,), space=Space.PEER),)))  # fmt: skip
        return own + sl + [
            Stage(gname, p.rows, partial(_gate, p=p), lds_words=-(-(p.slices // p.gb * PART) // 4) * 4,
                  writes=tuple(g_writes), args=tuple(g_args), key=(p,) + ("gate",), row_tasks=1),
        ]  # fmt: skip


# ------------------------------------------------------------------ device: slice


def _gate_val(k, p, t, j):
    """The previous seam's gate value j of row t (a plain f32 arg)."""
    return fx.Float32(
        bo.buffer_load(rsrc(k.args[p.gate_in]), t * MIX + j, vec_width=1, dtype=T.f32)
    )


def _pend_y(k, p, t, col, c0=None, tries=None):
    """(plain Python, traced inline) the previous sub-layer's output at (row t, columns col, col + 1): two f32.
    ``tries``: the stage entry's attempts at its pend loads (``_entry``; ``c0`` the task's first column), re-polled
    if stale."""
    if tries is not None:
        e = t * p.width + col
        s4 = _pend_sum(k, p, _pend_words(k, p, _specs_at(k, p, c0, t, col - c0), tries))
        hi = (e % 4) == 2
        y = [hi.select(s4[2], s4[0]), hi.select(s4[3], s4[1])]
        return y if p.pend_fmt == "ag" else [bf16_round(y[0]), bf16_round(y[1])]
    if p.pend_arg:
        y = bf2_f32(
            fx.Int32(
                bo.buffer_load(
                    rsrc(k.args[p.pend]),
                    (t * p.width + col) // 2,
                    vec_width=1,
                    dtype=T.i32,
                )
            )
        )
    elif (
        p.pend_fmt == "2shot"
    ):  # the owners' bf16 sums (``_own``), from the owner's slot
        e = t * p.width + col
        s4 = peer.read_ag(k, f"{p.pend}.ag", e - e % 4, p.width, "bf16")
        hi = (e % 4) == 2
        y = [hi.select(s4[2], s4[0]), hi.select(s4[3], s4[1])]
    elif p.pend_fmt == "ag":  # the owners' bf16 sums, gathered by the previous launch
        e = t * p.width + col
        s4 = peer.read_ag(
            k, p.pend, e - e % 4, p.width, "bf16", k.tag - 1 if p.pend_prev else None
        )
        hi = (e % 4) == 2
        y = [hi.select(s4[2], s4[0]), hi.select(s4[3], s4[1])]
    elif (
        p.pend_fmt == "rs"
    ):  # a sharded residual: the ranks' reduce-scatter partials of this rank's columns
        el = (
            t * p.rw + col - k.rank * p.rw
        )  # the element in this rank's owner slice (``peer.owned``'s)
        s4 = peer.sum_partials(
            k,
            p.pend,
            p.pend_rs,
            el - el % 4,
            p.pend_tp,
            k.tag - 1 if p.pend_prev else None,
        )
        hi = (el % 4) == 2
        y = [bf16_round(hi.select(s4[2], s4[0])), bf16_round(hi.select(s4[3], s4[1]))]
    elif p.pend_fmt != "":  # TP partials: this pair's 4-element item summed, then bf16
        e = t * p.width + col
        ptag = k.tag - 1 if p.pend_prev else None
        s4 = peer.sum_partials(k, p.pend, p.pend_fmt, e - e % 4, p.pend_tp, ptag)
        hi = (e % 4) == 2
        y = [bf16_round(hi.select(s4[2], s4[0])), bf16_round(hi.select(s4[3], s4[1]))]
    return y


@traced
def _fold_pair(k, p, c0, q, valid, pre=None, me=None, dst=None):
    """R of pair q = (row, stream, two columns) of this task's: folded (or read), to LDS (and res_out). ``p.early``:
    a gate channel's post / comb as an attempt issued before the pend poll, checked after it. ``pre``: this pair's
    loads issued at the stage's entry (``_entry``). ``me`` (sharded): R is written by the destination = this rank's
    task."""
    per_row = HC * COLS // 2
    t, o, cc = q // per_row, (q % per_row) // (COLS // 2), 2 * (q % (COLS // 2))
    col = c0 + cc
    rpairs = [_rpair(k, p, t, i, col, True) for i in range(HC)]
    if valid:
        if const_expr(
            p.res_in_ch
        ):  # tagged R of this launch: an attempt now, its tags checked where it is used
            tries = _res_try(k, p, rpairs)
        else:
            words = [
                bo.buffer_load(rsrc(k.args[p.res_in]), w, vec_width=1, dtype=T.i32)
                for w in rpairs
            ]
        if const_expr(p.fold):
            gidx = [t * MIX + HC + o] + [
                t * MIX + 2 * HC + HC * i + o for i in range(HC)
            ]
            if const_expr(p.gate_in_arg):
                post = _gate_val(k, p, t, HC + o)
                comb = [_gate_val(k, p, t, 2 * HC + HC * i + o) for i in range(HC)]
            elif const_expr(p.early):
                gtries = _gate_try(k, p, gidx)
            else:
                g = mb.poll(
                    [(k.mb[p.gate_in], t * MIX + HC + o, 1)]
                    + [
                        (k.mb[p.gate_in], t * MIX + 2 * HC + HC * i + o, 1)
                        for i in range(HC)
                    ],
                    k.tag,
                )
                post = g[0][0].bitcast(fx.Float32)
                comb = [g[1 + i][0].bitcast(fx.Float32) for i in range(HC)]
            y = _pend_y(k, p, t, col, c0, pre["pend"] if pre is not None else None)
            if const_expr(
                p.early and not p.gate_in_arg
            ):  # the gate attempt, checked now (re-polled if stale)
                g5 = _gate_check5(
                    k, p, gidx, [w[0] for w in gtries], [w[1] for w in gtries]
                )
                post = g5[0].bitcast(fx.Float32)
                comb = [g5[1 + i].bitcast(fx.Float32) for i in range(HC)]
            rs = (
                _res_words(k, p, rpairs, tries)
                if p.res_in_ch
                else [bf2_f32(fx.Int32(w)) for w in words]
            )
            out = []
            for d in range_constexpr(2):
                acc = comb[0] * rs[0][d]
                for i in range_constexpr(1, HC):
                    acc = acc + comb[i] * rs[i][d]
                out.append(bf16_round(post * y[d] + acc))
            if const_expr(p.owned):
                _r_store(k, p, me, dst, t, o, col, out[0], out[1])
            else:
                if const_expr(p.res_out != ""):
                    bo.buffer_store(
                        bf16_pair(out[0], out[1]).bitcast(fx.Int32),
                        rsrc(k.args[p.res_out]),
                        ((t * HC + o) * p.width + col) // 2,
                    )
                if const_expr(p.res_ch != ""):
                    mb.put(
                        k.mb[p.res_ch],
                        ((t * HC + o) * p.width + col) // 2,
                        bf16_pair(out[0], out[1]),
                        k.tag,
                    )
            vals = out
        else:
            rs = (
                _res_words(k, p, rpairs, tries)
                if p.res_in_ch
                else [bf2_f32(fx.Int32(w)) for w in words]
            )
            vals = _pick_stream(rs, o)
            if const_expr(
                p.r_only and p.owned
            ):  # R as it is, into its channel / the gathered region
                _r_store(k, p, me, dst, t, o, col, vals[0], vals[1])
        if const_expr(not p.r_only):
            for d in range_constexpr(2):
                fx.ptr_store(vals[d], k.ldsf + (t * p.kp + o * COLS + cc + d))


def _rpair(k, p, t, i, col, read=False):
    """(plain Python, traced inline) R's pair of (row t, stream i, columns col, col + 1) in an R arg or channel:
    [rows, HC, width] (sharded: this rank's [rows, HC, width / tp], ``col`` one of its columns; ``read`` of a
    whole-width ``res_in``: the whole row's)."""
    if p.owned and not (read and p.rin_full):
        return ((t * HC + i) * p.rw + col - k.rank * p.rw) // 2
    return ((t * HC + i) * p.width + col) // 2


def _r_store(k, p, me, dst, t, o, col, v0, v1):
    """Sharded: R of (row t, stream o, columns col, col + 1) to ``res_out`` / ``res_ch`` by the destination = this
    rank's task (``me`` None: every task, one a slice), and (``res_gather``) by every task into its destination's
    slot of the gathered region."""
    w = _rpair(k, p, t, o, col)
    _r_out_if(k, p, me, w, v0, v1)
    if p.rg and not p.rg_late:
        loc = peer.tile_local(
            t * HC + o, (col - k.rank * p.rw) // 2, p.rows * HC, GATHER_TILE
        )
        mb.put(peer.to_rank(k, p.rg, dst), loc, bf16_pair(v0, v1), k.tag)


@traced
def _r_out_if(k, p, me, w, v0, v1):
    if me:
        _r_out(k, p, w, v0, v1)


def _r_out(k, p, w, v0, v1):
    if p.res_out != "":
        bo.buffer_store(bf16_pair(v0, v1).bitcast(fx.Int32), rsrc(k.args[p.res_out]), w)
    if p.res_ch != "":
        mb.put(k.mb[p.res_ch], w, bf16_pair(v0, v1), k.tag)


def _lin_put(k, p, dst, t, col, vals):
    """Sharded: LIN of (row t, columns col, col + 1) into rank ``dst``'s copy of the gathered region, this rank's slot
    (tiled: a task's COLS columns of every row one contiguous run, ``peer.tile_local``).
    """
    loc = peer.tile_local(t, (col - k.rank * p.rw) // 2, p.rows, GATHER_TILE)
    mb.put_bf(peer.to_rank(k, p.lin, dst), loc * 2, vals, k.tag)


ENTRY_FMTS = (
    "ag",
    "f32",
    "bf16",
)  # the previous launch's pend formats ``HCSeam(entry=)`` issues at the stage's entry


def _specs_at(k, p, c0, t, cc):
    """The poll specs [(Addr, pair, 2)] of the previous launch's pend for the pair at row ``t``, column ``c0 + cc``
    (``c0`` the task's first): the owner's gathered sums ("ag", ``peer.read_ag``'s pair) or every rank's partial in rank
    order ("f32" / "bf16", ``peer.sum_partials``')."""
    e = t * p.width + c0 + cc
    return _push_specs(k, p, (e - e % 4) // 2)


def _push_specs(k, p, i):
    """The ranks' partial pairs of item ``i`` (= first element / 2), rank order (``peer.sum_partials``')."""
    tag = k.tag - 1
    half, out = k.peer_pairs[p.pend] // 2, []
    for src in range(p.pend_tp):
        a = peer.from_rank(k, p.pend, src, tag)
        out += [(a, i, 2), (a, half + i, 2)]
    return out


def _entry(k, first, p):
    """``Stage.pre`` of a chained seam's slice (``HCSeam(entry=)``): per pair round of the CTA's first task, one attempt
    at each of the previous launch's pend loads (issued first: the critical ones), the 4 R words and the previous gate's
    post / comb, issued before the task loop's set-up. A CTA without a task addresses the last task's (unused).
    """
    c0 = fx.min(first, fx.Int32(p.slices - 1)) * COLS
    if (
        p.direct
    ):  # a thread a (row, 2 columns): the 4 streams' R words and all 24 gate values (``_direct_pair``)
        lin_pairs, out = p.rows * COLS // 2, []
        for r in range(-(-lin_pairs // THREADS)):
            q = fx.min(k.tid + THREADS * r, lin_pairs - 1)
            t, cc = q // (COLS // 2), 2 * (q % (COLS // 2))
            col = c0 + cc
            pend = _entry_pend(k, p, c0, t, cc)
            words = [bo.buffer_load(rsrc(k.args[p.res_in]), ((t * HC + j) * p.width + col) // 2, vec_width=1,
                                    dtype=T.i32) for j in range(HC)]  # fmt: skip
            out.append(
                {
                    "pend": pend,
                    "r": words,
                    "g": [_gate_val(k, p, t, j) for j in range(MIX)],
                }
            )
        return out
    pairs, per_row = p.rows * HC * COLS // 2, HC * COLS // 2
    out = []
    for r in range(-(-pairs // THREADS)):
        q = fx.min(k.tid + THREADS * r, pairs - 1)
        t, o, cc = q // per_row, (q % per_row) // (COLS // 2), 2 * (q % (COLS // 2))
        col = c0 + cc
        pend = _entry_pend(k, p, c0, t, cc)
        words = [bo.buffer_load(rsrc(k.args[p.res_in]), ((t * HC + j) * p.width + col) // 2, vec_width=1, dtype=T.i32)
                 for j in range(HC)]  # fmt: skip
        g = [_gate_val(k, p, t, HC + o)] + [
            _gate_val(k, p, t, 2 * HC + HC * j + o) for j in range(HC)
        ]
        out.append({"pend": pend, "r": words, "g": g})
    return out


def _entry_pend(k, p, c0, t, cc):
    """One attempt at each of the pend loads of (row t, column c0 + cc), issued now, not waited for."""
    pend = []
    for a, i, n in _specs_at(k, p, c0, t, cc):
        note(a.region, a.space, "poll")
        pend.append(fx.Vector(bo.buffer_load(rsrc(a.value), fx.Int32(i) * 2, vec_width=2 * n, dtype=T.i32,
                                             cache_modifier=CM_SYS)))  # fmt: skip
    return pend


def _pend_words(k, p, specs, tries):
    """The entry's attempts (per spec: 2 pairs of value, tag) -> per spec its 2 value words, all polled again if any
    tag is not the previous launch's."""
    tag = k.tag - 1
    bad = None
    for w in tries:
        b = (w[1] != tag) | (w[3] != tag)
        bad = b if bad is None else bad | b
    v = fx.Vector.from_elements([w[j] for w in tries for j in (0, 2)], fx.Int32)
    v = _repoll(specs, v, bad, tag)
    return [[v[2 * s], v[2 * s + 1]] for s in range(len(tries))]


@traced
def _repoll(specs, v, bad, tag):
    if bad:
        got = mb.poll(specs, tag)
        v = fx.Vector.from_elements([x for g in got for x in g], fx.Int32)
    return v


def _pend_sum(k, p, words):
    """Per spec value words -> the 4 elements: the owner's bf16 sums ("ag"), or the ranks' partials summed in rank
    order (fp32; ``peer.sum_partials``' order)."""
    parts = [
        [x.bitcast(fx.Float32) for x in words[2 * s] + words[2 * s + 1]]
        for s in range(p.pend_tp)
    ]
    acc = parts[0]
    for s in range(1, len(parts)):
        acc = [acc[q] + parts[s][q] for q in range(4)]
    return acc


def _res_try(k, p, rpairs):
    """One attempt at the 4 streams' R pairs (value, tag) from the R channel, issued now, not waited for."""
    a = k.mb[p.res_in]
    note(a.region, a.space, "poll")
    return [fx.Vector(bo.buffer_load(rsrc(a.value), fx.Int32(w) * 2, vec_width=2, dtype=T.i32, cache_modifier=CM_DEV))
            for w in rpairs]  # fmt: skip


def _res_words(k, p, rpairs, tries):
    """The attempt's R pairs as [stream] -> (col, col + 1) f32, polled again if a tag is not current."""
    w = _res_check(k, p, rpairs, [t[0] for t in tries], [t[1] for t in tries])
    return [bf2_f32(fx.Int32(x)) for x in w]


@traced
def _res_check(k, p, rpairs, vals, tags):
    w0, w1, w2, w3 = vals
    if (
        (tags[0] != k.tag)
        | (tags[1] != k.tag)
        | (tags[2] != k.tag)
        | (tags[3] != k.tag)
    ):
        got = mb.poll([(k.mb[p.res_in], w, 1) for w in rpairs], k.tag)
        w0, w1, w2, w3 = got[0][0], got[1][0], got[2][0], got[3][0]
    return [w0, w1, w2, w3]


def _gate_try(k, p, idx):
    """One attempt at gate channel pairs ``idx`` (value, tag), issued now, not waited for."""
    a = k.mb[p.gate_in]
    note(a.region, a.space, "poll")
    return [fx.Vector(bo.buffer_load(rsrc(a.value), fx.Int32(i) * 2, vec_width=2, dtype=T.i32, cache_modifier=CM_DEV))
            for i in idx]  # fmt: skip


@traced
def _gate_check5(k, p, idx, vals, tags):
    w0, w1, w2, w3, w4 = vals
    if (
        (tags[0] != k.tag)
        | (tags[1] != k.tag)
        | (tags[2] != k.tag)
        | (tags[3] != k.tag)
        | (tags[4] != k.tag)
    ):
        got = mb.poll([(k.mb[p.gate_in], i, 1) for i in idx], k.tag)
        w0, w1, w2, w3, w4 = got[0][0], got[1][0], got[2][0], got[3][0], got[4][0]
    return [w0, w1, w2, w3, w4]


@traced
def _gate_check4(k, p, idx, vals, tags):
    w0, w1, w2, w3 = vals
    if (
        (tags[0] != k.tag)
        | (tags[1] != k.tag)
        | (tags[2] != k.tag)
        | (tags[3] != k.tag)
    ):
        got = mb.poll([(k.mb[p.gate_in], i, 1) for i in idx], k.tag)
        w0, w1, w2, w3 = got[0][0], got[1][0], got[2][0], got[3][0]
    return [w0, w1, w2, w3]


def _pre_issue(k, p, q):
    """``p.early``: collapse pair q's row's pre values issued now (plain loads, or a channel attempt)."""
    t = q // (COLS // 2)
    if p.gate_in_arg:
        return [_gate_val(k, p, t, j) for j in range(HC)]
    return _gate_try(k, p, [t * MIX + j for j in range(HC)])


def _pick_stream(rs, o):
    """Stream o's pair of ``rs`` (o traced): selects, no indexing by a traced value."""
    v0, v1 = rs[0][0], rs[0][1]
    for i in range(1, HC):
        v0 = (o == i).select(rs[i][0], v0)
        v1 = (o == i).select(rs[i][1], v1)
    return [v0, v1]


@traced
def _direct_pair(k, p, c0, q, valid, ent=None, dst=None, me=None):
    """``p.direct``: (row t, columns cc, cc + 1) of this task's: R of the 4 streams folded (or read) in registers,
    LIN = sum_j pre[j] R[j] put at once, then R stored (res_out, res_ch) and into LDS for the mixes. The operations
    and orders of ``_fold_pair`` + ``_collapse_pair``: the same bits. ``ent``: this pair's loads issued at the stage's
    entry (``_entry``)."""
    t, cc = q // (COLS // 2), 2 * (q % (COLS // 2))
    col = c0 + cc
    if valid:
        if const_expr(
            ent is not None
        ):  # the entry's loads (``_entry``): the gate's 24 values in order
            words, g = ent["r"], ent["g"]
            pre = g[:HC]
        else:
            words = [bo.buffer_load(rsrc(k.args[p.res_in]), _rpair(k, p, t, i, col, True), vec_width=1, dtype=T.i32)
                     for i in range(HC)]  # fmt: skip
            pre = [_gate_val(k, p, t, j) for j in range(HC)]
        rs = []
        if const_expr(p.fold):
            if const_expr(ent is not None):
                post = g[HC : 2 * HC]
                comb = [[g[2 * HC + HC * i + o] for i in range(HC)] for o in range(HC)]
            else:
                post = [_gate_val(k, p, t, HC + o) for o in range(HC)]
                comb = [
                    [_gate_val(k, p, t, 2 * HC + HC * i + o) for i in range(HC)]
                    for o in range(HC)
                ]
            y = _pend_y(k, p, t, col, c0, ent["pend"] if ent is not None else None)
            xs = [bf2_f32(fx.Int32(w)) for w in words]
            for o in range_constexpr(HC):
                out = []
                for d in range_constexpr(2):
                    acc = comb[o][0] * xs[0][d]
                    for i in range_constexpr(1, HC):
                        acc = acc + comb[o][i] * xs[i][d]
                    out.append(bf16_round(post[o] * y[d] + acc))
                rs.append(out)
        else:
            rs = [bf2_f32(fx.Int32(w)) for w in words]
        lin = []
        for d in range_constexpr(2):
            acc = pre[0] * rs[0][d]
            for j in range_constexpr(1, HC):
                acc = acc + pre[j] * rs[j][d]
            lin.append(acc)
        if const_expr(p.owned):
            _lin_put(k, p, dst, t, col, lin)
        else:
            chan.put_bf16(k.mb[p.lin], t * p.width + col, lin, k.tag)
        for o in range_constexpr(HC):
            if const_expr(p.fold and p.owned):
                _r_store(k, p, me, dst, t, o, col, rs[o][0], rs[o][1])
            elif const_expr(p.fold):
                word = bf16_pair(rs[o][0], rs[o][1]).bitcast(fx.Int32)
                bo.buffer_store(
                    word, rsrc(k.args[p.res_out]), ((t * HC + o) * p.width + col) // 2
                )
                if const_expr(p.res_ch != ""):
                    mb.put(
                        k.mb[p.res_ch], ((t * HC + o) * p.width + col) // 2, word, k.tag
                    )
            for d in range_constexpr(2):
                fx.ptr_store(rs[o][d], k.ldsf + (t * p.kp + o * COLS + cc + d))


@traced
def _collapse_pair(k, p, c0, q, valid, pre0=None, dst=None):
    """LIN of pair q = (row, two columns): sum_j pre[j] R[j] in stream order, one bf16 rounding. ``pre0``: the row's
    pre values issued at the task's start (``p.early``: plain values, or a channel attempt checked here).
    """
    t, cc = q // (COLS // 2), 2 * (q % (COLS // 2))
    if valid:
        if const_expr(pre0 is not None and p.gate_in_arg):
            pre = pre0
        elif const_expr(pre0 is not None):
            w4 = _gate_check4(
                k,
                p,
                [t * MIX + j for j in range(HC)],
                [w[0] for w in pre0],
                [w[1] for w in pre0],
            )
            pre = [w.bitcast(fx.Float32) for w in w4]
        elif const_expr(p.gate_in_arg):
            pre = [_gate_val(k, p, t, j) for j in range(HC)]
        else:
            g = mb.poll([(k.mb[p.gate_in], t * MIX + j, 1) for j in range(HC)], k.tag)
            pre = [g[j][0].bitcast(fx.Float32) for j in range(HC)]
        out = []
        for d in range_constexpr(2):
            acc = pre[0] * fx.ptr_load(k.ldsf + (t * p.kp + cc + d))
            for j in range_constexpr(1, HC):
                acc = acc + pre[j] * fx.ptr_load(
                    k.ldsf + (t * p.kp + j * COLS + cc + d)
                )
            out.append(acc)
        if const_expr(p.owned):
            _lin_put(k, p, dst, t, c0 + cc, out)
        else:
            chan.put_bf16(k.mb[p.lin], t * p.width + c0 + cc, out, k.tag)
        if const_expr(
            p.ssq != ""
        ):  # the word as put (bf16 pair), for the norm's partial sums
            fx.ptr_store(
                bf16_pair(out[0], out[1]).bitcast(fx.Int32),
                k.lds + (p.sq + t * (COLS // 2) + cc // 2),
            )


@traced
def _ssq_put(k, p, task):
    """Thread (row t, half h) < rows x 2: the dot2 sum of squares of LIN words 8 h .. 8 h + 7 of the task's row t
    (word order, from 0: a NormQuant thread's partial over its 16 elements) -> ``ssq`` element t (width / 16) + 2 task
    + h."""
    if k.tid < p.rows * 2:
        t, h = k.tid // 2, k.tid % 2
        acc = fx.Float32(0.0)
        for j in range_constexpr(COLS // 4):
            w = fx.Int32(
                fx.ptr_load(k.lds + (p.sq + t * (COLS // 2) + h * (COLS // 4) + j))
            )
            acc = dot2_bf16(w, w, acc)
        mb.put(k.mb[p.ssq], t * (p.width // 16) + task * 2 + h, acc, k.tag)


@traced
def _sumsq(k, p, task, t):
    if k.wave == t % WAVES:
        x0 = fx.ptr_load(k.ldsf + (t * p.kp + 2 * k.lane))
        x1 = fx.ptr_load(k.ldsf + (t * p.kp + 2 * k.lane + 1))
        sq = butterfly(x0 * x0 + x1 * x1, (32, 16, 8, 4, 2, 1))
        if k.lane == 0:
            mb.put(k.mb[p.pmix], (task * p.rows + t) * PART + MIX, sq, k.tag)


@traced
def _mix_tile(k, p, task, t0, n):
    """The 24 partial dots of rows t0 .. t0 + n: fp32 MFMA 16x16x4, A = hc_fn rows (two 16-row tiles),
    B = R^T (rows as N), K split over the waves, the waves' partials summed in wave order.
    """
    lane, wave = k.lane, k.wave
    fl, red = k.ldsf + p.fl, k.ldsf + p.red
    tok = t0 + fx.min(lane % 16, n - 1)
    kw = KT // WAVES
    cs = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(2)]
    for kk in range_constexpr(kw // 4):
        colk = wave * kw + lane // 16 + 4 * kk
        b = fx.ptr_load(k.ldsf + (tok * p.kp + colk))
        for h in range_constexpr(2):
            row = h * 16 + lane % 16
            av = fx.ptr_load(fl + (fx.min(row, MIX - 1) * p.kp + colk))
            av = (row < MIX).select(av, fx.Float32(0.0))
            ops = rocdl._split_mfma_operands([av, b, cs[h], 0, 0, 0])
            cs[h] = fx.Vector(rocdl.mfma_f32_16x16x4f32(T.vec(4, T.f32), *ops).result)
    for h in range_constexpr(2):
        fx.ptr_store(cs[h], red + ((wave * 2 + h) * 64 + lane) * 4)
    gpu.barrier()
    if k.tid < MIX * n:
        tl = k.tid // MIX
        j = k.tid % MIX
        h = j // 16
        ln = 16 * ((j % 16) // 4) + tl
        tot = fx.Float32(0.0)
        for w in range_constexpr(WAVES):
            tot = tot + fx.ptr_load(red + (((w * 2 + h) * 64 + ln) * 4 + j % 4))
        mb.put(k.mb[p.pmix], (task * p.rows + t0 + tl) * PART + j, tot, k.tag)
    gpu.barrier()


@traced
def _own(k, t, p):
    """Two-shot, the owner's half (``own_quads`` threads a task): elements ``4 i .. + 4`` of this rank's column quarter
    (rows x width / tp, the reduce-scatter's layout): the ranks' fp32 partials summed in rank order
    (``peer.sum_partials``: the one-shot's bits), rounded to bf16 (seam F's y), pushed to one rank a task
    (``<pend>.ag``; the remote puts spread over pend_tp x the CTAs: one task pushing
    to every rank in turn reached the last rank ~2 us after the first)."""
    n = p.rows * p.width // p.pend_tp
    chunks = -(-n // 4 // p.own_quads)
    dst, i = t // chunks, (t % chunks) * p.own_quads + k.tid
    # (dyn_rows: the live rows' elements only; the readers clamp to them, ``peer.read_ag``)
    ok = i * 4 < (n if isinstance(k.n, int) else k.n * (p.width // p.pend_tp))
    if const_expr(p.own_quads < THREADS):
        ok = ok & (k.tid < p.own_quads)
    if ok:
        el = i * 4
        s4 = peer.sum_partials(
            k, p.pend, p.pend_rs, el, p.pend_tp, k.tag - 1 if p.pend_prev else None
        )
        peer.push_partial(
            k, f"{p.pend}.ag", "bf16", el, [bf16_round(v) for v in s4], [dst]
        )


def _fn_word(p, e):
    """LDS word (from ``p.fl``) of hc_fn element ``e`` of the task's [MIX][KT] block: row e / KT at pitch ``p.kp``."""
    return e if p.kp == KT else (e // KT) * p.kp + e % KT


def _slice(k, task, p):
    dst = me = None
    if (
        p.owned
    ):  # with LIN: a task a (slice of this rank's columns, destination rank): LIN to that rank only
        if p.fan:
            task, dst = task % p.ns, task // p.ns
            me = dst == k.rank
        c0 = (task + k.rank * p.ns) * COLS
    else:
        c0 = task * COLS
    if p.r_only:  # R alone (the fold, or R copied), into its channel
        pairs = p.rows * HC * COLS // 2
        for r in range_constexpr(-(-pairs // THREADS)):
            q = k.tid + THREADS * r
            _fold_pair(k, p, c0, fx.min(q, pairs - 1), q < pairs, None, me, dst)
        return
    # the task's hc_fn columns first: no wait decides them
    fns = []
    for i in range_constexpr(MIX * KT // THREADS if p.mixes else 0):
        e = k.tid + THREADS * i
        j, o = e // KT, (e % KT) // COLS
        fns.append(bo.buffer_load(rsrc(k.args[p.fn]), j * HC * p.width + o * p.width + c0 + e % COLS, vec_width=1,
                                  dtype=T.f32))  # fmt: skip
    lin_pairs = p.rows * COLS // 2
    if (
        p.direct
    ):  # the fold and LIN of (row, 2 columns) a thread, LIN put before the mixes' LDS set-up
        for r in range_constexpr(-(-lin_pairs // THREADS)):
            q = k.tid + THREADS * r
            _direct_pair(
                k,
                p,
                c0,
                fx.min(q, lin_pairs - 1),
                q < lin_pairs,
                k.pre[r] if p.entry else None,
                dst,
                me,
            )
        for i in range_constexpr(MIX * KT // THREADS):
            fx.ptr_store(
                fx.Float32(fns[i]), k.ldsf + (p.fl + _fn_word(p, k.tid + THREADS * i))
            )
        gpu.barrier()
        _tail(k, p, task, me)
        return
    pres = [None] * -(-lin_pairs // THREADS)
    if (
        p.early and p.collapse
    ):  # the collapse's pre values with the task's first loads (a clamped row: a valid address)
        pres = [
            _pre_issue(k, p, fx.min(k.tid + THREADS * r, lin_pairs - 1))
            for r in range(len(pres))
        ]
    pairs = p.rows * HC * COLS // 2
    for r in range_constexpr(-(-pairs // THREADS)):
        q = k.tid + THREADS * r
        _fold_pair(
            k,
            p,
            c0,
            fx.min(q, pairs - 1),
            q < pairs,
            k.pre[r] if p.entry else None,
            me,
            dst,
        )
    for i in range_constexpr(MIX * KT // THREADS if p.mixes else 0):
        fx.ptr_store(
            fx.Float32(fns[i]), k.ldsf + (p.fl + _fn_word(p, k.tid + THREADS * i))
        )
    gpu.barrier()
    for r in range_constexpr(-(-lin_pairs // THREADS) if p.collapse else 0):
        q = k.tid + THREADS * r
        _collapse_pair(k, p, c0, fx.min(q, lin_pairs - 1), q < lin_pairs, pres[r], dst)
    if p.ssq:
        gpu.barrier()
    _tail(k, p, task, me)


def _tail(k, p, task, me):
    """``_slice_tail``; sharded: by the destination = this rank's task only (the others put LIN and are done)."""
    if p.fan:
        _owned_tail(k, p, task, me)
    else:
        _slice_tail(k, p, task)


@traced
def _owned_tail(k, p, task, me):
    if me:
        _slice_tail(k, p, task)


def _slice_tail(k, p, task):
    """After LIN (and, ``ssq``, a barrier): the norm's partials, the fused norm, the mixes' partials."""
    if p.ssq:
        _ssq_put(k, p, task)
    if not p.mixes:
        gpu.barrier()  # the next task (or stage) reuses the LDS
        return
    for t in range_constexpr(p.rows):
        _sumsq(k, p, task, t)
    for t0 in range_constexpr(0, p.rows, TILE):
        _mix_tile(k, p, task, t0, min(TILE, p.rows - t0))


RPUSH_SLICES = 2  # ``res_gather_late``: slices a push task (a few CTAs off the path, each to every rank in turn)


@traced
def _rpush(k, t, p):
    """``res_gather_late``: slices ``RPUSH_SLICES t ..`` of this rank's columns, every (row, stream)'s pairs polled from
    ``res_ch`` and put into every rank's slot of the gathered region (the slices' tiles), the ranks rotated by this
    one's (the destinations' first puts spread)."""
    pairs = p.rows * HC * COLS // 2
    for j in range_constexpr(RPUSH_SLICES):
        s = t * RPUSH_SLICES + j
        for r in range_constexpr(-(-pairs // THREADS)):
            q = fx.min(k.tid + THREADS * r, fx.Int32(pairs - 1))
            ro, cc = q // (COLS // 2), q % (
                COLS // 2
            )  # (row x stream, pair of the slice)
            w = mb.poll([(k.mb[p.res_ch], (ro * p.rw + s * COLS) // 2 + cc, 1)], k.tag)[
                0
            ][0]
            loc = peer.tile_local(ro, s * (COLS // 2) + cc, p.rows * HC, GATHER_TILE)
            for d in range_constexpr(p.gtp):
                _rpush_put(
                    k, p, (k.rank + 1 + d) % p.gtp, loc, w, k.tid + THREADS * r < pairs
                )


@traced
def _rpush_put(k, p, dst, loc, w, ok):
    if ok:
        mb.put(peer.to_rank(k, p.rg, dst), loc, w, k.tag)


@traced
def _rfull(k, t, p):
    """Sharded, ``res_full``: (row, stream) ``t`` of the gathered R (every owner's columns) into the plain arg."""
    wp = p.width // 2
    a = peer.from_rank(k, p.rg, 0)
    for r in range_constexpr(-(-wp // THREADS)):
        i = fx.min(k.tid + THREADS * r, fx.Int32(wp - 1))
        w = mb.poll(
            [(a, peer.tile_index(t * wp + i, wp, p.gtp, p.rows * HC, GATHER_TILE), 1)],
            k.tag,
        )[0][0]
        if k.tid + THREADS * r < wp:
            bo.buffer_store(w, rsrc(k.args[p.rfull]), t * wp + i)


@traced
def _blk(k, t, p):
    """Sharded: gate block ``b`` of this rank's slices (its slices ``b gb ..``), every (row, partial) summed
    in slice order (the gate's own first sums: the same bits), put into rank ``dst``'s copy of ``pmixg``, this rank's
    slot (a destination a task)."""
    b, dst = t % p.nb, t // p.nb
    n = p.rows * PART
    a = peer.to_rank(k, p.pmixg, dst)
    for r in range_constexpr(-(-n // THREADS)):
        i = fx.min(k.tid + THREADS * r, fx.Int32(n - 1))
        row, jp = i // PART, i % PART
        vals = mb.poll([(k.mb[p.pmix], ((b * p.gb + s) * p.rows + row) * PART + jp, 1)
                        for s in range(p.gb)], k.tag)  # fmt: skip
        acc = vals[0][0].bitcast(fx.Float32)
        for s in range_constexpr(1, p.gb):
            acc = acc + vals[s][0].bitcast(fx.Float32)
        if k.tid + THREADS * r < n:
            mb.put(a, (b * p.rows + row) * PART + jp, acc, k.tag)


# ------------------------------------------------------------------ device: gate


def _row_sum(v):
    return butterfly(v, (1, 2))


def _col_sum(v):
    return butterfly(v, (4, 8))


def _sigmoid(z):
    return fx.Float32(1.0) / (fx.Float32(1.0) + fx.Float32(fmath.exp(-z)))


@traced
def _gate_block_sum(k, p, t, red, i, groups):
    """Gate item ``i`` = (block g, partial jp) of row t: the block's slices summed in order, into LDS."""
    if i < groups * PART:
        g, jp = i // PART, i % PART
        if const_expr(
            p.owned
        ):  # sharded: block g's sums from its owner's slot (``_blk``: the same sums)
            ix = (
                (g // p.nb) * k.peer_pairs[p.pmixg]
                + ((g % p.nb) * p.rows + t) * PART
                + jp
            )
            acc = mb.poll([(peer.from_rank(k, p.pmixg, 0), ix, 1)], k.tag)[0][
                0
            ].bitcast(fx.Float32)
        else:
            vals = mb.poll([(k.mb[p.pmix], ((g * p.gb + s) * p.rows + t) * PART + jp, 1)
                            for s in range(p.gb)], k.tag)  # fmt: skip
            acc = vals[0][0].bitcast(fx.Float32)
            for s in range_constexpr(1, p.gb):
                acc = acc + vals[s][0].bitcast(fx.Float32)
        fx.ptr_store(acc, red + (g * PART + jp))


@traced
def _gate(k, t, p):
    """Row t: the slices' partials summed in slice order -> rstd, the 24 mixes -> pre / post / comb."""
    lane = k.lane
    red = k.ldsf
    groups = p.slices // p.gb
    # lane -> its mix: pre / post on lanes 0 .. 7, comb from lane COMB0; scale / base issued before the poll
    j = (lane < COMB0).select(lane % (2 * HC), fx.min(lane - COMB0 + 2 * HC, MIX - 1))
    sc = [
        fx.Float32(bo.buffer_load(rsrc(k.args[p.scale]), i, vec_width=1, dtype=T.f32))
        for i in range(3)
    ]
    base = fx.Float32(bo.buffer_load(rsrc(k.args[p.base]), j, vec_width=1, dtype=T.f32))
    for ps in range_constexpr(
        -(-(groups * PART) // THREADS)
    ):  # one pass at GATE_BLOCK's 20 blocks (500 threads)
        _gate_block_sum(k, p, t, red, k.tid + ps * THREADS, groups)
    gpu.barrier()
    if k.wave == 0:
        mix = fx.ptr_load(red + j)
        sq = fx.ptr_load(red + MIX)
        for g in range_constexpr(1, groups):
            mix = mix + fx.ptr_load(red + (g * PART + j))
            sq = sq + fx.ptr_load(red + (g * PART + MIX))
        # the reference: mixes = (x @ fn^T) * rsqrt(mean(x^2) + eps), then mixes * scale + base
        v = mix * fx.Float32(
            fmath.rsqrt(sq / fx.Float32(HC * p.width) + fx.Float32(p.eps))
        )
        out = fx.Float32(0.0)
        if lane < HC:
            out = _sigmoid(v * sc[0] + base) + fx.Float32(p.hc_eps)
        if (lane >= HC) & (lane < 2 * HC):
            out = fx.Float32(2.0) * _sigmoid(v * sc[1] + base)
        cv = v * sc[2] + base
        # comb: softmax over a row of 4, + eps, then column / row normalizations (true divisions)
        m = butterfly(cv, (1, 2), fx.max)
        cv = fx.Float32(fmath.exp(cv - m))
        cv = cv / _row_sum(cv) + fx.Float32(p.hc_eps)
        cv = cv / (_col_sum(cv) + fx.Float32(p.hc_eps))
        for _ in range_constexpr(p.sinkhorn - 1):
            cv = cv / (_row_sum(cv) + fx.Float32(p.hc_eps))
            cv = cv / (_col_sum(cv) + fx.Float32(p.hc_eps))
        is_comb = (lane >= COMB0) & (lane < COMB0 + HC * HC)
        out = is_comb.select(cv, out)
        jo = is_comb.select(lane - COMB0 + 2 * HC, lane)
        if (lane < 2 * HC) | is_comb:
            if const_expr(p.gate_out != ""):
                mb.put(k.mb[p.gate_out], t * MIX + jo, out, k.tag)
            if const_expr(p.gate_out_arg != ""):
                bo.buffer_store(out, rsrc(k.args[p.gate_out_arg]), t * MIX + jo)
    gpu.barrier()

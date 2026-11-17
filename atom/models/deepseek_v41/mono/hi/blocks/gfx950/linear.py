# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``Linear``: y = MXFP8(x) @ W^T, fp32 accumulation, for every row of the step.

The weight format is the block's ``weight_format`` (``gemv.WeightFormat``): ``"mxfp4"``
(default) or ``"mxfp8"`` (E4M3 + E8M0 per row and 32 K, e.g.
FP8 block-scaled checkpoints through ``weights.mx.fp8_block_to_mxfp8``). A weight object
(``pack``) has ``packed`` (MXFP4 [N, K/2] bytes or MXFP8 [N, K]), ``codes`` [N, K/32] and
optionally ``fmt`` (its format name; absent means MXFP4). The rest of this docstring
describes MXFP4; MXFP8 differs only in unit size (2 KiB, 8 VGPRs a lane in flight).

A W4A8 GEMV ("skinny GEMM on
MFMA"). One source serves any number of rows (1..16): the step's rows are the
MFMA N dimension (``device.gfx950.gemv``); rows < 16 leave columns unused.

**Inputs.** The activation is an MXFP8 ``Channel`` of width K (polled), or
``NormIn``: the task normalizes and quantizes the bf16 source rows itself (a BF16
channel such as h1, polled, or a plain bf16 arg such as h_in) with the
prologue norm (``blocks.gfx950.norm``: same device functions, so the same bytes
reach the MFMA as through a ``NormQuant`` stage with the same ``NormSpec``) and
writes the fp8 words and scale codes into its own LDS activation layout (the norm
strategy "recompute in the consumer prologue"). The
weight is the given HF linears stacked along N in order (e.g. q, k, v), packed
by ``pack`` into two pointer-table args ``<name>.w`` / ``<name>.ws`` (layout:
``gemv.pack_weight``, 16-row groups x 128-K units, aiter's 16x16 shuffle). For
``SwiGLU`` the gate and up rows are interleaved per 16-row group (gate group
``i`` is packed group ``2 i``, up group ``i`` is ``2 i + 1``), so a task owns
matching gate / up columns.

**Outputs** (``out=``), each with the reference's rounding:

- ``BF16Out(ch)``: ch := bf16(acc);
- ``F32Out(ch)``: ch := acc (an F32 channel, e.g. logits for ``TopK``; tagged only);
- ``Residual(out, arg= | channel=, publish=)``: out := bf16(bf16(acc) + residual),
  the residual a plain bf16 arg [rows, N] (written before the launch) or a BF16
  channel (written earlier in the launch: polled); written to the plain bf16 arg
  ``out`` [rows, N] and, with ``publish``, to a BF16 channel too;
- ``SwiGLU(ch)``: mid := bf16(silu(min(g, L)) * clamp(u, -L, L)) (fp32 math, g / u the bf16 gate / up, L the
  ``limit``), MXFP8-quantized
  (group 32, ``weights.mx.quantize_mxfp8``) into MXFP8 channel ``ch`` of
  width N / 2;
- ``MXFP8Out(ch)``: MXFP8 of bf16(acc);
- ``HCWrite`` (``blocks.gfx950.hc``): the hyper-connection combine, out_s := bf16(res_s +
  bf16(acc) * inj_s) into each of n residual streams (the residual a plain arg or a BF16
  channel, the injection multipliers an F32 channel), to a plain arg and / or a BF16 channel.

**Row-parallel TP, the sum in the consumer** (``tp_sum="consumer"``): the epilogue only pushes the partial
(``peer.push_partial``); the output spec is not run and the output channel is not written: the consuming stage
sums the ranks' partials itself (``peer.sum_partials``, e.g. ``HCSeam(pend=PeerPartial(...))``), its other
loads in flight, and the producer no longer waits for the slowest rank.

**Row-parallel TP** (``tp_partial``, a ``Scenario.tp`` > 1 program; ``plan.program``): the
weight is this rank's K shard and acc a partial sum. Before the output spec, every epilogue item's
partial (``"bf16"``: rounded to bf16, the reference's ``RowParallelLinear``; ``"f32"``: as is) goes
to this rank's slot of every rank's peer region ``<name>.tp``, and the item's ``tp`` slots are polled
and summed in rank order in fp32: acc := that sum, the same bits on every rank. The all-reduce costs
one peer round trip inside the epilogue, no stage of its own (``PeerAllReduce``). An item waits only
for the same item on the other ranks: with ``k_split`` 1 the same task, which runs in the same CTA
order there; with K slices whichever slice arrives last, so a TP shape with K slices must fit one grid
round (no slice queued behind a waiting one).

**bf16 compute** (``compute="bf16"``, MXFP8 weights): each lane's 32 FP8 weight and activation values are widened
to bf16 with their E8M0 scales (exact) and fed to the bf16 MFMA (exact products, fp32 sums): the arithmetic of a
plain bf16 GEMM, for where the reference runs one on FP8-representable operands (DeepSeek-V4.1's ``wo_a``). The
scaled MX MFMA truncates within its K groups; this does not. Over 16 rows, the same per 16-row chunk (a weight unit
widened once, an MFMA per chunk): a row's bits do not depend on the row count.

**FP8 per-channel weights** (``row_scale``, MXFP8 weights): E4M3 rows with every block code 127 and one fp32 scale
an output row (Quark's per-channel FP8): an item's totals, after the K waves' and K
slices' sums, times its row's scale (arg ``<name>.rs``, prefetched with the weights), then the bias and the output
spec as above.

**Block-diagonal weights** (``groups`` > 1, e.g. DeepSeek-V4.1's grouped ``wo_a``): the input channel holds
``groups`` K-wide pieces per row and output rows ``j N / groups .. (j + 1) N / groups`` (weight rows likewise)
read piece j only; a task's row groups never straddle two groups.

**Task shape** (the shape knobs): a task owns ``group_rows``
output rows (G = group_rows / 16 row groups) over a 1 / ``k_split`` slice of K;
the CTA's 8 waves form ``wave_rows`` x (8 / ``wave_rows``) (row groups x K
waves; ``kwave`` "contiguous": each K wave a K range, as K slices take them). With ``k_split`` > 1 every slice
publishes its fp32 partials (an
exchange region) and counts itself in on an arrival counter (modulo ``k_split``,
no tag); the last slice to arrive sums the slices in slice order and runs the
epilogue. Knobs left unset are
derived per scenario (``Linear.plan``): the smallest ``k_split`` whose input slice
fits the LDS and the registers (prefetch >= ``P_MIN``, compiled spill-free), the
fewest row groups per task within one grid round, the fewest ``wave_rows`` that
leave a wave at most ``GW_MAX`` row groups (and whose partials fit the LDS).

**Schedule of a task** (split phase): the task's weight scale
codes, then the first ``prefetch`` weight units of every wave (nontemporal) are
issued before the input is polled; one speculative poll attempt of the whole
input slice goes out after the first third of them (vmcnt is in order: it lands
before the rest of the prefetch, and if the input was there the MFMAs start
while the rest streams; stale pairs are polled again). The input slice and the
scales land in LDS; each unit's MFMA then issues the load ``prefetch`` units
ahead (K step by K step, LDS operands one step ahead, the order fenced by
``sched_barrier``); the K waves' partials are summed in wave order through LDS
(overlaying the activation), then the epilogue. ``prefetch`` is derived from the
register budget and checked by compiling (``_prefetch``, ``_fit``).
"""

import hashlib
import json
import os
from dataclasses import astuple, dataclass
from functools import partial
from pathlib import Path

import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from ...device.gfx950 import buffer_ops as bo
from ...device.gfx950 import channels as chan
from ...device.gfx950 import gemv, gemv16, peer
from ...device.gfx950 import mailbox as mb
from ...device.gfx950.ops import (
    bf16_round,
    fresh,
    rsrc,
    traced,
)
from ...geometry import NormGeometry
from ...plan.program import PlanError, Region, Scenario, Stage
from ...plan.trace import Space, note
from ...target import MI355X
from ..base import Block, Channel, Fmt
from . import norm as nq
from .hc import HCWrite
from .warm import idle_issue, idle_sink

WAVES = MI355X.waves
WAVE_LANES = MI355X.wave
THREADS = MI355X.threads
VGPR_BUDGET = 256  # 512 threads, one CTA per CU: two waves per SIMD
LEAD_DIV = 3  # the input's first poll attempt follows 1 / LEAD_DIV of the prefetch (settle-once)
GW_MAX = 4  # row groups per wave before the default shape adds wave rows (gate_up: 2 or 4 wave rows -0.5 us)
P_MIN = 8  # prefetched units per wave below which the derived shape splits K instead
# plan_rule "units" (Linear._unit_candidates): a task's fixed cost and a K split's extra hop, as bytes a CU streams in
# that time (~1.3 µs at ~25 GB/s a CU when every CU streams)
TASK_BYTES = 32 << 10
HOP_BYTES = 32 << 10
STREAM_FIT_MARGIN = (
    8  # more for a streamed input (its composed layers' pressure, as FIT_MARGIN's)
)
FIT_MARGIN = (
    8  # VGPRs a fitted stage leaves free: the composed kernel's other stages (and their
)
# pointers, held in SGPRs from kernel start) cost the stage a few registers (1-5 spills seen in a composed MLP)
NORM_ROWS = 4  # NormIn rows normalized per norm_quant call (their words stay live until it returns)
# ``weight_format="bf16"`` (A16W16): plain bf16 weights in ``gemv16.pack_bf16``'s units (16 rows x 128 K, 64 B a lane),
# a BF16 channel input landed as bf16 words, four bf16 MFMAs a unit (``gemv16``): a BF16 GEMM's exact products, fp32
# sums
BF16W = gemv.WeightFormat("bf16", -1, 16)


def wformat(name: str) -> gemv.WeightFormat:
    """The weight format ``name``: ``gemv.FORMATS``' MX formats or ``BF16W`` (a plan's ``"mxfp4/x16"``: MXFP4)."""
    name = name.split("/")[0]
    return BF16W if name == "bf16" else gemv.FORMATS[name]


# **A16W4** (MXFP4 weights and a BF16 channel input): the input lands as bf16 words (as ``weight_format="bf16"``),
# each MXFP4 unit is decoded to its exact bf16 values (``gemv16.decode_fp4``, the lane's E8M0 code) and fed to four
# bf16 MFMAs with the input's bf16 at the same K (``gemv16.b_operands``): bf16 activations x the weights' MXFP4 values,
# exact products, fp32 sums. The plan's ``wfmt`` is ``"mxfp4/x16"``.
X16 = "/x16"


STREAM_ROWS = 32  # rows above which the derived shape streams its input (``Linear(xchunk=)``): resident slices fit
#                  up to 32 rows, at 64 none does


# ------------------------------------------------------------------ output specs


@dataclass(frozen=True)
class BF16Out:
    channel: Channel


@dataclass(frozen=True)
class F32Out:
    channel: Channel  # F32, width N, tagged


@dataclass(frozen=True)
class MXFP8Out:
    channel: Channel


@dataclass(frozen=True)
class ArgOut:
    """bf16(acc) into the plain bf16 arg ``arg`` [rows, N] (an engine's buffer: e.g. the drafter's q / k / v for the
    engine's own RoPE and cache insert)."""

    arg: str


@dataclass(frozen=True)
class SwiGLU:
    channel: Channel  # MXFP8, width N / 2
    # "fp32" (DeepSeek-V4.1's Expert): gate and up rounded to bf16 (the projections' outputs), clamped (``limit`` > 0:
    # up to [-limit, limit], gate to <= limit), then silu(gate) * up in fp32 and one bf16 rounding
    rounding: str = "fp32"
    limit: float = 0.0

    def __post_init__(self):
        if self.rounding != "fp32":
            raise ValueError(f"SwiGLU: rounding 'fp32', got {self}")


def _swi_round(out: SwiGLU) -> str:
    """The plan's SwiGLU rounding."""
    return out.rounding


@dataclass(frozen=True)
class Residual:
    out: str  # plain bf16 arg [rows, N]
    arg: str | None = None  # residual from a plain bf16 arg [rows, N] ...
    channel: Channel | None = None  # ... or from a BF16 channel
    publish: Channel | None = None  # also publish the sum as a BF16 channel
    # also publish, per row and 16 output columns, the dot2 sum of squares of the sum's bf16 words in a ``NormQuant``
    # thread's order (word order from 0): an F32 channel of width N / 16, ``HCSeam(ssq_out=)``'s partials, so a
    # ``GatheredDot(norm=DotNorm(ssq=))`` normalizes its own K slice with the norm stage's bits (no whole-row poll)
    ssq: Channel | None = None

    def __post_init__(self):
        if (self.arg is None) == (self.channel is None):
            raise ValueError("Residual needs exactly one of arg= and channel=")
        if self.ssq is not None and (self.ssq.fmt is not Fmt.F32 or self.ssq.flagged):
            raise ValueError("Residual ssq: a tagged F32 channel (width N / 16)")


@dataclass(frozen=True)
class NormIn:
    """The input as a norm prologue: rows of ``src`` (a BF16 ``Channel`` of width K,
    polled, or the name of a plain bf16 [rows, K] arg) through zero-centred RMSNorm
    with gamma arg ``gamma`` (``layer.retained[weight]``, packed by
    ``Linear.pack``) and MXFP8 quantization, by ``spec`` (default ``norm.prologue(norm)``:
    the same mapping and reduction order as ``NormQuant``'s channel stage). Give the
    norm geometry ``norm`` (width, eps) or a ``spec``. A ``spec`` with a ``pitch``: rows
    of ``pitch`` elements (a plain arg's stride, the GEMM's K), the norm over the first
    ``width``, the tail zero in the task's activation (fp8 0, code 127)."""

    src: Channel | str
    gamma: str
    weight: str = "input_layernorm"
    spec: nq.NormSpec | None = None
    norm: NormGeometry | None = None

    def __post_init__(self):
        if self.spec is None:
            if self.norm is None:
                raise ValueError("NormIn needs the norm geometry (norm=) or a spec")
            object.__setattr__(self, "spec", nq.prologue(self.norm))
        if self.spec.split != 1 or not self.spec.shared or self.spec.qmap.chunks != 1:
            raise ValueError(
                f"NormIn needs a one-chunk, shared-sum, unsplit NormSpec, got {self.spec}"
            )
        if self.spec.pitch and isinstance(self.src, Channel):
            raise ValueError("NormIn with a pitch reads a plain bf16 arg")
        if isinstance(self.src, Channel) and (
            self.src.fmt is not Fmt.BF16 or self.src.width != self.spec.width
        ):
            raise ValueError(
                f"NormIn reads a plain bf16 arg or a BF16 channel of width {self.spec.width}"
            )

    @property
    def width(self) -> int:
        return self.spec.row_width

    @property
    def src_name(self) -> str:
        return self.src.name if isinstance(self.src, Channel) else self.src


# ------------------------------------------------------------------ the plan of one stage


@dataclass(frozen=True)
class _Plan:
    """Every Python constant the emit closes over (also the stage's build key)."""

    name: str
    rows: int
    n: int  # output features
    k: int
    g: int  # row groups per task
    k_split: int  # K slices (tasks) per row block
    wr: int  # waves along row groups
    wk: int  # waves along K
    prefetch: int  # units issued ahead, per wave
    lead: int  # units issued before the input's first poll attempt
    kind: str  # "bf16" | "f32" | "mxfp8" | "swiglu" | "residual"
    x: str  # input channel name ("" with a norm prologue)
    out_ch: str  # output channel name ("" if none)
    res_arg: str
    res_ch: str
    out_arg: str
    part: str  # exchange region of K-slice partials ("" if k_split == 1)
    ctr: str  # per row block arrival counters of the K slices ("" if k_split == 1)
    norm_src: str = ""  # NormIn: source channel / arg name ("": an MXFP8 channel input)
    norm_ch: bool = False  # NormIn: the source is a channel (polled), else a plain arg
    gamma: str = ""  # NormIn: gamma arg
    nspec: nq.NormSpec | None = None  # NormIn: the norm's device mapping
    norm_rows: int = 0  # NormIn: rows per norm_quant call
    norm_pre: int = (
        0  # NormIn: rows whose loads go out with the weight prefetch (the rest when reached)
    )
    x_proto: str = (
        "tagged"  # the input channel's ``Proto`` ("flag": flags, then dense words)
    )
    out_proto: str = (
        "tagged"  # the output channel's ``Proto`` (flag: one flag per row block)
    )
    wfmt: str = "mxfp4"  # the weight format (``gemv.FORMATS``)
    tp: int = 1  # TP ranks (row-parallel with ``tp_partial``)
    kwave: str = (
        "interleave"  # the K waves' K steps: "interleave" (wave w: w, w + wk, ...) | "contiguous" (a range)
    )
    # weight units a wave issues before its input has landed (the rest of the prefetch right after): fewer keep the
    # stage's prefetch from flooding memory while its producer finishes; 0: the whole prefetch, as before
    pre_units: int = 0
    tp_partial: str = (
        ""  # row-parallel TP: the exchanged partial's format ("bf16" | "f32"), "" if none
    )
    tp_sum: str = (
        "producer"  # row-parallel TP: who sums the ranks' partials ("producer" | "consumer" | "rs": the
    )
    #                           consumer, two-shot: each column quarter to its owner only, ``peer.push_rs``; its
    #                           tp_partial "f32" pairs or "f32p" / "f32q" packed)
    groups: int = (
        1  # block-diagonal weight: output rows of group j read input columns j K .. (j + 1) K
    )
    x_split: int = (
        1  # row-parallel TP over a whole input row: rank r reads input columns r K .. (r + 1) K
    )
    # SwiGLU: the rounding ("fp32", see ``SwiGLU``); every other Linear keeps the default below, as the plan
    # digests (``linear_fits``) hash every field
    swi_round: str = "qwen"
    swi_limit: float = 0.0  # SwiGLU "fp32": the clamp (0: none)
    compute: str = (
        "scaled"  # "scaled": the MX MFMA; "bf16": MXFP8 widened to bf16, bf16 MFMA (module docstring)
    )
    hcw: HCWrite | None = None  # kind "hcwrite": the combine's spec
    prewiden: int = (
        0  # bf16 compute: prefetched units a wave widens to bf16 before its input's tags are checked
    )
    bias: str = (
        ""  # the f32 bias arg [N] added to acc before the output's rounding ("": none)
    )
    # the f32 per-output-row weight scale arg [N] (packed row order) multiplying acc first ("": none; ``row_scale``)
    row_scale: str = ""
    # the input streamed through two LDS slots of ``xchunk`` K steps each, polled during the task (``_stream_ring``),
    # instead of the whole slice resident; 0: resident (``Linear(xchunk=)``)
    xchunk: int = 0
    xlead: int = (
        1  # a streamed input: slots whose poll is in flight (registers) while a slot streams
    )
    # always False (a streamed input's slots by loads into registers and ``ds_write``s); kept: the stage keys and the
    # plan digests hash every field
    xcopy: bool = False

    @property
    def a16(
        self,
    ) -> (
        bool
    ):  # ``weight_format="bf16"``: bf16 weights and input (module constants, ``BF16W``)
        return self.wfmt == "bf16"

    @property
    def x16(
        self,
    ) -> bool:  # A16W4 (``X16``): MXFP4 weights decoded to bf16, a bf16 input
        return self.wfmt.endswith(X16)

    @property
    def xk(self):  # the input channel's width
        return self.k * self.groups * self.x_split

    seg: int = 1  # rows per request (the scenario's spec)
    ssq: str = ""  # Residual(ssq=): the sum of squares channel ("" if none)
    # always at their defaults (no conv input or output); kept: the stage keys and the plan digests hash every field
    conv_coef: str = ""
    conv_w: int = 0
    conv_d0: int = 0
    conv_d1: int = 0
    conv_base: str = ""
    cin_src: str = ""
    cin_coef: str = ""
    cin_w: int = 0
    cin_d0: int = 0
    cin_d1: int = 0
    cin_base: str = ""

    @property
    def tp_reg(self):  # the peer region of the TP partials
        return f"{self.name}.tp" if self.tp_partial else ""

    @property
    def fmt(self) -> gemv.WeightFormat:
        return wformat(self.wfmt)

    @property
    def n_rg(self):
        return self.n // 16

    @property
    def steps(self):  # K steps of 128
        return self.k // 128

    @property
    def ks(self):  # K steps per slice
        return self.steps // self.k_split

    @property
    def c(self):  # K steps per wave (the last waves may have fewer)
        return -(-self.ks // self.wk)

    @property
    def rchunks(
        self,
    ):  # 16-row chunks of the step's rows (the MFMA's N): one per 16 rows, each weight unit read once
        return -(-self.rows // 16)

    @property
    def gw(self):  # row groups per wave
        return self.g // self.wr

    @property
    def blocks(self):
        return -(-self.n_rg // self.g)

    @property
    def tasks(self):
        return self.blocks * self.k_split

    @property
    def kse(self):  # K elements per slice
        return self.ks * 128

    @property
    def xkse(
        self,
    ):  # K elements of the input an LDS buffer holds: the slice, or one ring slot (``xchunk``)
        return self.xchunk * 128 if self.xchunk else self.kse

    @property
    def pitch(
        self,
    ):  # LDS words per activation row (+4: bank spread; a bf16 input: two values a word)
        return self.xkse // 2 + 4 if self.a16 or self.x16 else self.xkse // 4 + 4

    @property
    def spitch(self):  # LDS words per row of scale codes (odd: bank spread)
        return self.xkse // 32 + 1

    @property
    def xs_base(
        self,
    ):  # the codes, after the words (in a ring slot: from the slot's start)
        return self.rows * self.pitch

    @property
    def xslot(self):  # LDS words of one input buffer (words and codes)
        return self.rows * (self.pitch + self.spitch)

    @property
    def ws_base(
        self,
    ):  # LDS words: the task's weight scale codes, [row group][K step][64 lanes] bytes
        return self.xslot * (self.xbufs if self.xchunk else 1)

    @property
    def xbufs(
        self,
    ):  # a streamed input's LDS slots: two; with ``xcopy`` one per slot in flight and the streaming one
        return self.xlead + 1 if self.xcopy else 2

    @property
    def ws_chunks(
        self,
    ):  # 16 B chunks of the task's weight scale codes (bf16 weights, row-scaled FP8: none)
        return 0 if self.a16 or self.row_scale else self.g * self.ks * 4

    @property
    def flag_word(self):  # LDS word: "this task is its row block's last K slice"
        return max(self.ws_base + self.ws_chunks * 4, self.wk * self.g * self.rows * 16)

    @property
    def norm_red(self):  # LDS word: NormIn's per-row wave partials (norm_rows x 8 f32)
        return -(-(self.flag_word + 1) // 4) * 4

    @property
    def lds_words(self):
        end = self.norm_red + (
            self.nspec.lds_words(self.norm_rows) if self.norm_src else 0
        )
        return -(-end // 4) * 4  # 16 B multiple

    @property
    def out_quads(self):  # epilogue items: (row, row group (or pair), 4 features)
        per = self.g // 2 if self.kind == "swiglu" else self.g
        return self.rows * per * 4

    @property
    def w_bytes(self):
        return self.fmt.weight_bytes(self.n, self.k)

    @property
    def s_bytes(self):
        return (
            16 if self.a16 else self.n * self.k // 32
        )  # (bf16 weights: a placeholder scales arg)


def default_groups(n_rg: int, k_split: int, align: int) -> int:
    """Row groups per task by default: the fewest (a multiple of ``align``) that
    keep the task count within one round of the grid."""
    g = align
    while -(-n_rg // g) * k_split > MI355X.cus:
        g += align
    return g


def _wave_rows(g: int) -> list[int]:
    """Candidate wave rows for G row groups, preferred first: the fewest that leave
    a wave at most ``GW_MAX`` row groups (accumulators eat the prefetch's
    registers), then the rest."""
    ws = [w for w in (1, 2, 4, 8) if g % w == 0]
    return [w for w in ws if g // w <= GW_MAX] + [w for w in ws if g // w > GW_MAX]


def _prefetch(
    gw: int,
    units: int,
    poll_regs: int,
    scale_regs: int,
    unit_regs: int = 4,
    rc: int = 1,
    b_rc: int | None = None,
) -> int:
    """Units a wave keeps in flight: as many as the VGPR budget holds.
    A unit in flight costs ``unit_regs`` VGPRs (MXFP4: 16 B of weights, MXFP8: 32 B); reserved: the
    accumulators (4 per row group), two K steps of LDS operands (B: 9, weight
    scales: 1 per row group), the input poll's batch (``poll_regs``) and the
    task's scale chunks (``scale_regs``), both live while the prefetched units
    are, and 48 for addresses and the epilogue. ``rc``: 16-row chunks (accumulators and B operands per chunk);
    ``b_rc``: the chunks whose B operands are live at once (default ``rc``; a streamed input: 1).
    """
    free = (
        VGPR_BUDGET
        - 4 * gw * rc
        - 2 * (9 * (rc if b_rc is None else b_rc) + gw)
        - poll_regs
        - scale_regs
        - 48
    )
    return max(1, min(units, free // unit_regs))


def _dense_chunks(rows: int, kse: int) -> tuple[int, int]:
    """16-byte loads per thread of a FLAG input slice: (fp8 words, scale codes)."""
    return -(-rows * kse // 16 // THREADS), -(-rows * kse // 128 // THREADS)


def _poll_regs(rows: int, kse: int) -> int:
    """VGPRs per thread of the input's first poll attempt, live from its issue to its
    landing: value + tag words of the whole slice (2-pair specs of FP8 words,
    1-pair specs of codes) and each spec's LDS destination."""
    nq = -(-rows * kse // 8 // THREADS)
    ns = -(-rows * kse // 32 // THREADS)
    return 5 * nq + 3 * ns


# ------------------------------------------------------------------ the derived prefetch, checked

_FITTED: dict = {}


def fit_cache() -> Path:
    """The on-disk cache of the fits derived by probes (only where the frozen table does not apply): ``dsv41_hi_fits``
    in FlyDSL's runtime cache directory (``FLYDSL_RUNTIME_CACHE_DIR``, FlyDSL's default otherwise).
    """
    from flydsl.utils import env as flydsl_env

    return Path(flydsl_env.runtime.cache_dir) / "dsv41_hi_fits"


# the frozen fits (``linear_fits.FITS``: the served compositions' depths, derived by ``_fit``'s probes with the FlyDSL
# ``linear_fits.FLYDSL``) used in place of the probes with that FlyDSL; ``FROZEN`` False: every fit derived (the
# contract check's, which compares what it derives with the table: ``DERIVED``, plan digest -> depth)
FROZEN = True
DERIVED: dict = {}


def plan_digest(p) -> str:
    """A fit's frozen-table key: its plan's (``_Plan``) digest."""
    return hashlib.sha256(repr(astuple(p)).encode()).hexdigest()[:16]


def _frozen(p):
    """``p``'s frozen depth (``linear_fits``), or "missing" (not frozen, another FlyDSL, or ``FROZEN`` off)."""
    if not FROZEN:
        return "missing"
    import importlib.metadata as md

    from . import linear_fits

    try:
        if md.version("flydsl") != linear_fits.FLYDSL:
            return "missing"
    except md.PackageNotFoundError:
        return "missing"
    return linear_fits.FITS.get(plan_digest(p), "missing")


def _fit_cached(key: str):
    f = fit_cache() / f"{key}.json"
    try:
        return json.loads(f.read_text())["prefetch"]
    except (OSError, ValueError, KeyError):
        return "missing"


def _fit_store(key: str, pre) -> None:
    try:
        cache = fit_cache()
        cache.mkdir(parents=True, exist_ok=True)
        tmp = cache / f"{key}.{os.getpid()}.tmp"
        tmp.write_text(json.dumps({"prefetch": pre}))
        tmp.replace(cache / f"{key}.json")
    except OSError:
        pass  # a cache only


def fit_key(*parts) -> str:
    """A fit's cache key: ``parts`` (the plan and what else the fit depends on), the hi source and the toolchain
    and environment (``emit.compile.toolchain_identity``): a fit cached by another FlyDSL / LLVM / target / knob set
    (another container on a shared ``fit_cache()``) is another key, never a different prefetch under this one.
    """
    from ...emit.compile import toolchain_identity
    from ...emit.kernel import source_digest

    return hashlib.sha256(
        repr((parts, source_digest(), toolchain_identity())).encode()
    ).hexdigest()[:24]


def _fit(lin, sc: Scenario, p: _Plan) -> _Plan | None:
    """``p`` with the deepest prefetch at or below the model's that compiles without
    scratch or spills, or None if none does: the stage is compiled after stand-in
    producers of its input channels (``source_stages``, as in the block evals) and
    the depth lowered until the resource gate passes with ``FIT_MARGIN`` VGPRs to
    spare (or the depth is 1). The frozen table's depth where it has the plan (``_frozen``); else cached per plan,
    source digest and toolchain (``fit_key``), in process and on disk (``fit_cache()``).
    """
    from dataclasses import replace

    from ...emit.compile import ResourceError
    from ...plan.program import Program
    from ...runtime.kernel import MonoKernel

    key = fit_key(astuple(p), ())
    if key not in _FITTED and (frozen := _frozen(p)) != "missing":
        _FITTED[key] = frozen
    if key not in _FITTED and (cached := _fit_cached(key)) != "missing":
        _FITTED[key] = DERIVED[plan_digest(p)] = cached
    if key not in _FITTED:
        stages = source_stages(lin.x, sc.rows)
        q, _FITTED[key] = p, None
        margin = FIT_MARGIN + (STREAM_FIT_MARGIN if p.xchunk else 0)
        if (
            p.xchunk
        ):  # streamed: the pressure at depth 1, then the depth it leaves room for, up to every unit of the
            #           task (the register model is not the cap: it overcounts the ring's poll, 122 VGPRs at its 10)

            def vgprs(
                pre,
            ):  # None: the probe does not build (resources, or its compile / link fails: skipped)
                from flydsl.compiler.diagnostics import DSLCompileError

                q = replace(p, prefetch=pre, lead=max(1, pre // LEAD_DIV))
                try:
                    return MonoKernel(
                        Program(f"fit_{p.name}", sc, stages + lin._stages(q))
                    ).resources["vgpr"]
                except (ResourceError, DSLCompileError):
                    return None

            if (v1 := vgprs(1)) is not None:
                pre = min(
                    p.c * p.gw,
                    max(1, 1 + (VGPR_BUDGET - margin - v1) // p.fmt.lane_words),
                )
                while pre > 1 and (
                    (v := vgprs(pre)) is None or v > VGPR_BUDGET - margin
                ):
                    pre -= 1
                _FITTED[key] = pre
        while not p.xchunk:
            try:
                kern = MonoKernel(Program(f"fit_{p.name}", sc, stages + lin._stages(q)))
                if kern.resources["vgpr"] <= VGPR_BUDGET - margin or q.prefetch == 1:
                    _FITTED[key] = q.prefetch
                    break
            except ResourceError:
                if q.prefetch == 1:
                    break
            pre = max(1, q.prefetch - max(1, q.prefetch // 8))
            q = replace(q, prefetch=pre, lead=max(1, pre // LEAD_DIV))
        _fit_store(key, _FITTED[key])
        DERIVED[plan_digest(p)] = _FITTED[key]
    pre = _FITTED[key]
    return (
        None if pre is None else replace(p, prefetch=pre, lead=max(1, pre // LEAD_DIV))
    )


# ------------------------------------------------------------------ the block


class Linear(Block):
    """One W4A8 projection (module docstring)."""

    def __init__(
        self,
        name: str,
        linears: tuple[str, ...],
        x: Channel,
        n: int,
        out,
        group_rows: int | None = None,
        wave_rows: int | None = None,
        k_split: int | None = None,
        weight_format: str = "mxfp4",
        tp_partial: str | None = None,
        groups: int = 1,
        compute: str = "scaled",
        tp_sum: str = "producer",
        kwave: str = "interleave",
        plan_rule: str = "one_round",
        pre_units: int | None = None,
        prewiden: int = 0,
        bias: tuple[str, ...] | None = None,
        xchunk: int | str | None = None,
        xlead: int | None = None,
        x_split: int = 1,
        row_scale: bool = False,
    ):
        if (
            tp_sum not in ("producer", "consumer", "rs")
            or (tp_sum != "producer" and not tp_partial)
            or (tp_sum == "rs" and tp_partial not in ("f32", "f32q", "f32p"))
        ):
            raise ValueError(f"Linear {name}: tp_sum 'producer' or 'consumer' (consumer: a tp_partial; 'rs': an "
                             f"'f32' or 'f32q' one), got {tp_sum!r}")  # fmt: skip
        self.tp_sum = tp_sum
        if tp_partial not in (
            None,
            "bf16",
            "f32",
            "f32q",
            "f32p",
        ):  # f32q: fp32 packed (peer.py)
            raise ValueError(
                f"Linear {name}: tp_partial {tp_partial!r} not in (None, 'bf16', 'f32', 'f32q', 'f32p')"
            )
        if tp_partial and isinstance(out, SwiGLU):
            raise ValueError(
                f"Linear {name}: SwiGLU is column-parallel, it takes no tp_partial"
            )
        self.tp_partial = tp_partial or ""
        if weight_format not in gemv.FORMATS and weight_format != "bf16":
            raise ValueError(
                f"Linear {name}: weight format {weight_format!r} not in {sorted(gemv.FORMATS)} or 'bf16'"
            )
        self.weight_format = weight_format
        a16 = weight_format == "bf16"
        # A16W4 (``X16``): MXFP4 weights, a BF16 channel input
        self.x16 = x16 = not a16 and isinstance(x, Channel) and x.fmt is Fmt.BF16
        if x16 and (
            weight_format != "mxfp4" or x.flagged or groups != 1 or compute != "scaled" or prewiden or xchunk
            or bias is not None or isinstance(out, (HCWrite, SwiGLU))
        ):  # fmt: skip
            raise ValueError(f"Linear {name}: a BF16 input takes MXFP4 (A16W4) or bf16 weights; A16W4: a "
                             "tagged channel, no groups / compute / prewiden / xchunk / bias / SwiGLU / HCWrite")  # fmt: skip
        if a16 and (
            not isinstance(x, Channel) or x.fmt is not Fmt.BF16 or x.flagged or groups != 1 or compute != "scaled"
            or prewiden or xchunk or bias is not None or isinstance(out, HCWrite)
        ):  # fmt: skip
            raise ValueError(f"Linear {name}: bf16 weights take a tagged BF16 channel input, no groups / compute / "
                             "prewiden / xchunk / bias / HCWrite")  # fmt: skip
        if not a16 and not x16 and not isinstance(x, NormIn) and x.fmt is not Fmt.MXFP8:
            raise ValueError("Linear's input must be an MXFP8 channel or a NormIn")
        if isinstance(x, NormIn) and isinstance(x.src, Channel) and x.src.flagged:
            raise ValueError(
                "NormIn polls a tagged BF16 channel (bulk + flag source not implemented)"
            )
        if (
            isinstance(out, Residual)
            and out.channel is not None
            and out.channel.flagged
        ):
            raise ValueError(
                "Residual polls a tagged BF16 channel (bulk + flag residual not implemented)"
            )
        if (
            isinstance(out, Residual)
            and out.ssq is not None
            and (out.ssq.width * 16 != n or (tp_sum or "producer") != "producer")
        ):
            raise ValueError(
                f"Linear {name}: Residual ssq: width N / 16, the producer's TP sum"
            )
        if groups > 1 and (
            isinstance(x, NormIn) or x.width % groups or n % (16 * groups)
        ):
            raise ValueError(f"Linear {name}: groups {groups} needs an MXFP8 channel input of width groups x K and "
                             f"N a multiple of 16 groups (N {n})")  # fmt: skip
        self.groups = groups
        if compute not in ("scaled", "bf16") or (
            compute == "bf16" and weight_format != "mxfp8"
        ):
            raise ValueError(
                f"Linear {name}: compute 'scaled' or 'bf16' (bf16: MXFP8 weights), got {compute!r}"
            )
        self.compute = compute
        if prewiden and (compute != "bf16" or isinstance(x, NormIn) or x.dense):
            raise ValueError(
                f"Linear {name}: prewiden needs the bf16 compute and a tagged MXFP8 channel input"
            )
        self.prewiden = (
            prewiden  # units a wave widens before its input lands (``_Plan.prewiden``)
        )
        # bias: the layer's retained bf16 biases of ``linears`` (in their order), added in fp32 to acc before the
        # output's rounding (HF's ``nn.Linear(bias=True)``: bf16(x W^T + b)); not with SwiGLU / HCWrite / TP
        if bias is not None and (
            isinstance(out, (SwiGLU, HCWrite))
            or tp_partial
            or len(bias) != len(linears)
        ):
            raise ValueError(
                f"Linear {name}: a bias per linear, not with SwiGLU, HCWrite or tp_partial"
            )
        self.bias = tuple(bias) if bias is not None else None
        # row_scale: FP8 weights with one fp32 scale an output row (FP8 per-channel checkpoints, e.g. Quark's
        # "fp8_e4m3 per_channel"): the E4M3 bytes packed as MXFP8 with every block code 127 (scale 1, exact) and acc x
        # the row's scale, after every sum (K waves, K slices) and before the bias and the output's rounding. Each
        # linear's record carries ``row_scale`` (fp32 [N_i]). Not with a TP partial: the row-parallel sum's consumer
        # would never scale it
        if row_scale and (weight_format != "mxfp8" or tp_partial):
            raise ValueError(
                f"Linear {name}: row_scale takes MXFP8 weights (E4M3 rows, codes 127), no tp_partial"
            )
        self.row_scale = row_scale
        # x_split: the input channel holds every rank's K columns of a row (a whole row, e.g. a replicated input); a
        # row-parallel TP Linear (tp_partial) reads its rank's K slice of it (rank r: columns r K .. (r + 1) K)
        if x_split > 1 and (
            groups > 1 or not tp_partial or isinstance(x, NormIn) or x.width % x_split
        ):
            raise ValueError(f"Linear {name}: x_split needs a row-parallel TP (tp_partial), no groups, a channel input "
                             f"of width x_split x K")  # fmt: skip
        self.x_split = x_split
        self.name, self.linears, self.x, self.n, self.k, self.out = (
            name, tuple(linears), x, n, x.width // groups // x_split, out
        )  # fmt: skip
        if self.k % 128 or n % 16:
            raise ValueError(f"Linear {name}: K % 128 and N % 16 must be 0 (N {n}, K {self.k}): the scaled MFMA's K "
                             "block; pad other widths with zero tails (NormSpec.pitch, BLOCK_CATALOG 'K')")  # fmt: skip
        align = {SwiGLU: 4, MXFP8Out: 2}.get(type(out), 1)
        if isinstance(out, F32Out) and (
            out.channel.fmt is not Fmt.F32
            or out.channel.width != n
            or out.channel.flagged
        ):
            raise ValueError(f"F32Out needs a tagged F32 channel of width N ({n})")
        self.align = align
        self.group_rows, self.wave_rows, self.k_split = group_rows, wave_rows, k_split
        if kwave not in ("interleave", "contiguous"):
            raise ValueError(
                f"Linear {name}: kwave {kwave!r} not in ('interleave', 'contiguous')"
            )
        self.kwave = kwave
        self.pre_units = pre_units  # weight units a wave issues before its input lands (``_Plan.pre_units``)
        if plan_rule not in ("one_round", "units"):
            raise ValueError(
                f"Linear {name}: plan_rule {plan_rule!r} not in ('one_round', 'units')"
            )
        self.plan_rule = plan_rule
        # the input streamed through a 2-slot LDS ring of ``xchunk`` K steps (``_stream_ring``): None derives it
        # above ``STREAM_ROWS`` rows (no resident shape fits there); 0: never; n: always, n K steps a slot; "auto":
        # always, a slot of one K step a K wave
        if xchunk not in (None, 0, "auto") and not (
            isinstance(xchunk, int) and xchunk > 0
        ):
            raise ValueError(
                f"Linear {name}: xchunk None, 0, 'auto' or K steps > 0, got {xchunk!r}"
            )
        if xchunk and (
            not isinstance(x, Channel) or compute != "scaled" or kwave != "interleave"
        ):
            raise ValueError(f"Linear {name}: a streamed input (xchunk) needs an MXFP8 channel input, the scaled MX "
                             "compute and interleaved K waves")  # fmt: skip
        self.xchunk = xchunk
        self.xlead = xlead  # a streamed input's slots in flight (None: the deepest of 4 .. 1 that fits)
        if group_rows is not None and (group_rows % 16 or (group_rows // 16) % align):
            raise ValueError(
                f"Linear {name}: group_rows {group_rows} must be a multiple of {16 * align}"
            )
        if wave_rows is not None and WAVES % wave_rows:
            raise ValueError(
                f"Linear {name}: wave_rows {wave_rows} must divide {WAVES}"
            )
        if k_split is not None and (self.k // 128) % k_split:
            raise ValueError(
                f"Linear {name}: k_split {k_split} does not divide {self.k // 128} K steps"
            )

    # ---- plan side

    @property
    def weight_args(self) -> tuple[str, str]:
        return f"{self.name}.w", f"{self.name}.ws"

    def plan(self, sc: Scenario) -> _Plan:
        """The stage's plan for ``sc``. Knobs left None are derived: ``k_split`` is
        the smallest whose input slice fits the LDS and leaves the prefetch at
        least ``P_MIN`` units and compiles spill-free, ``group_rows`` the fewest row
        groups per task within one grid round, ``wave_rows`` the fewest that leave
        a wave at most ``GW_MAX`` row groups and whose K-wave partials fit the
        LDS. The prefetch depth is the register model's (``_prefetch``), checked
        by compiling the stage and lowered until it has no spills (``_fit``).
        ``plan_rule`` "units" (default "one_round", the rule above): the shape whose busiest CTA streams the fewest
        weight units, over 1-3 grid rounds and every K split, fewer K slices on a tie (``_unit_candidates``): a
        task whose scale block does not fit beside the prefetch then takes fewer rows over the whole K in two rounds
        instead of half the K in one (the linear measurements H-LM)."""
        if self.tp_partial and sc.tp < 2:
            raise PlanError(
                f"Linear {self.name}: tp_partial needs a TP scenario (tp > 1), got {sc.name}"
            )
        if self.x_split > 1 and sc.tp != self.x_split:
            raise PlanError(
                f"Linear {self.name}: x_split {self.x_split} needs TP {self.x_split}, got {sc.name}"
            )
        if self.x16 and sc.rows > 16:
            raise PlanError(
                f"Linear {self.name}: a bf16 input up to 16 rows (one MFMA chunk), got {sc.rows}"
            )
        steps = self.k // 128
        splits = (
            [self.k_split]
            if self.k_split
            else [d for d in range(1, steps + 1) if steps % d == 0]
        )
        if self.xchunk or (
            self.xchunk is None and sc.rows > STREAM_ROWS and self._streamable()
        ):
            return self._plan_streamed(sc, steps, splits)
        try:
            return self._plan_resident(sc, steps, splits)
        except (
            PlanError
        ):  # 17-32 rows with no resident shape (e.g. 24): streamed, where none built before
            if self.xchunk is None and sc.rows > 16 and self._streamable():
                return self._plan_streamed(sc, steps, splits)
            if (
                self.weight_format == "bf16"
                and sc.rows > 16
                and self.plan_rule != "units"
            ):
                # bf16 weights past 16 rows (not streamable: the whole slice's bf16 words are polled): the "units"
                # shapes, where none built before (the one-round group counts starve the prefetch from 32 rows)
                for kt, g in self._unit_candidates(sc, steps, splits):
                    fitted = self._first_fit(sc, g, kt, strict=False)
                    if fitted is not None:
                        return fitted
            raise

    def _streamable(self) -> bool:
        x = self.x
        return (isinstance(x, Channel) and self.compute == "scaled" and self.kwave == "interleave"
                and self.weight_format != "bf16" and not self.x16)  # fmt: skip

    def _plan_streamed(self, sc: Scenario, steps: int, splits: list[int]) -> _Plan:
        """The input through the LDS ring: shapes by the busiest CTA's bytes (``_unit_candidates``: a K split's fp32
        partials count, they grow with the rows), each with the first wave split and slot size that fits.
        """
        for kt, g in self._unit_candidates(sc, steps, splits):
            fitted = self._first_fit(sc, g, kt, strict=False, stream=True)
            if fitted is not None:
                return fitted
        if (
            self.group_rows and sc.rows > STREAM_ROWS
        ):  # a given task shape with none past STREAM_ROWS: the plan's
            for kt, g in self._unit_candidates(
                sc, steps, splits, given=False
            ):  # (the batch bands: a wq_b's 32 / 48 at 48)
                fitted = self._first_fit(sc, g, kt, strict=False, stream=True)
                if fitted is not None:
                    return fitted
        raise PlanError(
            f"Linear {self.name}: no streamed task shape fits at {sc.rows} rows (LDS / registers)"
        )

    def _plan_resident(self, sc: Scenario, steps: int, splits: list[int]) -> _Plan:
        for kt in splits:
            g = (
                self.group_rows // 16
                if self.group_rows
                else default_groups(self.n // 16, kt, self.align)
            )
            g = self._in_group(g)
            if not self.group_rows:
                g = self._owner_groups(sc, g)
            fitted = self._first_fit(sc, g, kt, strict=True)
            if fitted is not None:
                return fitted
        raise PlanError(
            f"Linear {self.name}: no task shape fits at {sc.rows} rows (LDS / registers)"
        )

    def _straddles(self, sc: Scenario, g: int) -> bool:
        """``tp_sum`` "rs": a task of ``g`` row groups would hold columns of two owners (``peer.push_rs`` puts a wave's
        lanes to one owner: the other owner's pairs never arrive, its consumer waits).
        """
        if self.tp_sum != "rs" or not self.tp_partial:
            return False
        q = self.n // 16 // sc.tp  # an owner's row groups
        return any(
            (b * g) // q != (min((b + 1) * g, self.n // 16) - 1) // q
            for b in range(-(-(self.n // 16) // g))
        )

    def _refuse_straddle(self, sc: Scenario, g: int) -> None:
        if self._straddles(sc, g):
            raise PlanError(f"Linear {self.name}: {g} row groups a task cross an owner's columns ({sc.tp} owners of "
                            f"{self.n // sc.tp}; tp_sum 'rs' pushes a wave to one owner)")  # fmt: skip

    def _owner_groups(self, sc: Scenario, g: int) -> int:
        """``g`` or, ``tp_sum`` "rs" with a task across an owner boundary, the next row-group count (a multiple of
        ``align``) whose tasks each stay in one owner's columns."""
        while self._straddles(sc, g) and g < self.n // 16:
            g += self.align
        return g

    def _in_group(self, g: int) -> int:
        per_group = self.n // self.groups // 16
        while (
            self.groups > 1 and per_group % g and g > self.align
        ):  # a task's row groups in one group
            g -= self.align
        return g

    def _first_fit(
        self, sc: Scenario, g: int, kt: int, strict: bool, stream: bool = False
    ) -> _Plan | None:
        """The first wave split / norm choice of ``g`` row groups x ``kt`` K slices that fits (LDS, registers,
        compiled spill-free), or None. ``strict``: a given ``wave_rows`` that does not divide ``g`` is an error.
        ``stream``: the input through the LDS ring, the largest slot (``xchunk``: 4, 2 or 1 K steps a K wave, or the
        given one) that fits."""
        self._refuse_straddle(sc, g)
        ks = self.k // 128 // kt
        for wr in [self.wave_rows] if self.wave_rows else _wave_rows(g):
            wk = WAVES // wr
            if (
                stream
            ):  # slots of one K step a K wave, the deepest lead that fits first; or the given slot / lead
                xcs = (
                    [self.xchunk]
                    if isinstance(self.xchunk, int) and self.xchunk
                    else [wk]
                )
                leads = [self.xlead] if self.xlead else [4, 3, 2, 1]
                choices = [
                    (0, 0, xc, ld)
                    for xc in xcs
                    for ld in leads
                    if xc % wk == 0 and ks % xc == 0 and xc < ks
                ]
            else:
                choices = [(0, 0, 0, 1)]
            for nrows, npre, xc, ld in choices:
                p = self._plan(sc, g, wr, kt, nrows, npre, xc, ld)
                if (
                    p.lds_words * 4 > MI355X.lds_bytes - 1024
                ):  # (1 KiB kept free past the arena)
                    continue
                if not self.k_split and not xc and p.prefetch < min(P_MIN, p.c * p.gw):
                    continue  # the input attempt would starve the prefetch: split K instead (streamed: compiled)
                fitted = _fit(self, sc, p)
                if fitted is None or (
                    not self.k_split and fitted.prefetch < min(P_MIN, p.c * p.gw)
                ):
                    continue
                return fitted
        return None

    def _unit_candidates(
        self, sc: Scenario, steps: int, splits: list[int], given: bool = True
    ) -> list[tuple[int, int]]:
        """``plan_rule`` "units": (k_split, row groups) shapes by the busiest CTA's bytes (a stage ends when its
        busiest CTA's bytes are in): its tasks' weight bytes plus ``TASK_BYTES`` a task (prologue, epilogue), and with
        K slices their fp32 partials (written and read, tagged pairs) plus ``HOP_BYTES`` for the last slice's poll;
        then the fewest K slices, then the fewest tasks. Row groups per task: given, else for 1, 2 and 3 grid rounds
        the fewest that fit them, and those rounded up to a multiple of 4 and of 8 (a split over the waves); ``given``
        False: the latter whatever ``group_rows``."""
        n_rg, cus = self.n // 16, MI355X.cus
        out = set()
        for kt in splits:
            gs = set()
            for rounds in (1, 2, 3):
                g = self.align
                while -(-n_rg // g) * kt > rounds * cus:
                    g += self.align
                gs |= {g, -(-g // 4) * 4, -(-g // 8) * 8}
            for g in gs:
                g = self._in_group(g)
                if g % self.align == 0 and not self._straddles(sc, g):
                    out.add((kt, g))

        unit = wformat(self.weight_format).unit_bytes

        def cost(c):
            kt, g = c
            tasks = -(-n_rg // g) * kt
            rounds = -(-tasks // cus)
            busiest = rounds * (g * (steps // kt) * unit + TASK_BYTES)
            if kt > 1:
                busiest += rounds * 2 * sc.rows * g * 16 * 8 + HOP_BYTES
            return (busiest, kt, tasks)

        return sorted(out, key=cost)

    def _plan(
        self,
        sc: Scenario,
        g: int,
        wr: int,
        kt: int,
        nrows: int = 0,
        npre: int = 0,
        xchunk: int = 0,
        xlead: int = 1,
    ) -> _Plan:
        out = self.out
        kind = {BF16Out: "bf16", F32Out: "f32", MXFP8Out: "mxfp8", SwiGLU: "swiglu", Residual: "residual",
                HCWrite: "hcwrite", ArgOut: "argout"}[type(out)]  # fmt: skip
        out_ch = out.channel.name
        wk = WAVES // wr
        ks = self.k // 128 // kt
        c = -(-ks // wk)
        norm = self.x if isinstance(self.x, NormIn) else None
        if (
            xchunk and self.x.dense
        ):  # the dense loads of the slots in flight (after the flags, at the task's start)
            in_regs = xlead * 4 * sum(_dense_chunks(sc.rows, xchunk * 128)) + 2
        elif (
            xchunk
        ):  # the attempts of the slots in flight (``xlead`` of them, live while a slot streams)
            in_regs = xlead * _poll_regs(sc.rows, xchunk * 128)
        elif (
            norm is None and self.x.dense
        ):  # the dense loads (after the prefetch) and the flag attempt
            in_regs = 4 * sum(_dense_chunks(sc.rows, ks * 128)) + 2
        elif norm is None:
            in_regs = _poll_regs(sc.rows, ks * 128)
        pre = _prefetch(
            g // wr,
            c * (g // wr),
            in_regs,
            (
                0
                if self.weight_format == "bf16" or self.row_scale
                else 4 * -(-g * ks * 4 // THREADS)
            ),
            wformat(self.weight_format).lane_words,
            -(-sc.rows // 16),
            1 if xchunk else None,
        )
        return _Plan(
            name=self.name,
            rows=sc.rows,
            n=self.n,
            k=self.k,
            g=g,
            k_split=kt,
            wr=wr,
            wk=wk,
            prefetch=pre,
            lead=max(1, pre // LEAD_DIV),
            kind=kind,
            x="" if norm else self.x.name,
            out_ch=out_ch,
            res_arg=out.arg or "" if kind == "residual" else "",
            res_ch=out.channel.name if kind == "residual" and out.channel else "",
            ssq=out.ssq.name if kind == "residual" and out.ssq is not None else "",
            out_arg=(
                out.out if kind == "residual" else (out.arg if kind == "argout" else "")
            ),
            part=f"{self.name}.part" if kt > 1 else "",
            ctr=f"{self.name}.ctr" if kt > 1 else "",
            norm_src=norm.src_name if norm else "",
            norm_ch=bool(norm and isinstance(norm.src, Channel)),
            gamma=norm.gamma if norm else "",
            nspec=norm.spec if norm else None,
            norm_rows=nrows if norm else 0,
            norm_pre=npre if norm else 0,
            x_proto="tagged" if norm else self.x.proto.value,
            out_proto=(
                self._out_channel().proto.value
                if self._out_channel() is not None
                else "tagged"
            ),
            wfmt=self.weight_format + (X16 if self.x16 else ""),
            tp=sc.tp if self.tp_partial else 1,
            kwave=self.kwave,
            pre_units=self.pre_units or 0,
            tp_partial=self.tp_partial,
            tp_sum=self.tp_sum,
            groups=self.groups,
            x_split=self.x_split,
            swi_round=_swi_round(out) if kind == "swiglu" else "qwen",
            swi_limit=out.limit if kind == "swiglu" else 0.0,
            compute=self.compute,
            seg=sc.spec,
            hcw=out if kind == "hcwrite" else None,
            prewiden=self.prewiden,
            bias=f"{self.name}.b" if self.bias else "",
            row_scale=f"{self.name}.rs" if self.row_scale else "",
            xchunk=xchunk,
            xlead=xlead if xchunk else 1,
        )

    def _out_channel(self) -> Channel | None:
        return (
            self.out.publish
            if isinstance(self.out, (Residual, HCWrite))
            else self.out.channel
        )

    def stages(self, sc: Scenario) -> list[Stage]:
        return self._stages(self.plan(sc))

    def _stages(self, p: _Plan) -> list[Stage]:
        writes = []
        ch = (
            self._out_channel() if p.tp_sum == "producer" else None
        )  # consumer: the partials region only
        if ch is not None:  # FLAG: one flag per row block (its last K slice puts it)
            writes += list(ch.regions(p.rows, (self.name,), flags=p.blocks))
        if p.part:
            writes.append(
                Region(p.part, p.k_split * p.rows * p.n, (self.name,), exchange=True)
            )
            writes.append(
                Region(p.ctr, p.blocks, (self.name,), exchange=True, single=True)
            )  # atomics: one copy
        if p.tp_partial:
            pairs = (peer.shot2_pairs(p.rows, p.n, p.tp, p.tp_partial) if p.tp_sum == "rs"
                     else peer.partial_pairs(p.tp_partial, p.rows, p.n))  # fmt: skip
            writes.append(
                Region(
                    p.tp_reg,
                    pairs,
                    (self.name,),
                    space=Space.PEER,
                    exchange=p.tp_sum == "producer",
                )
            )
        args = list(self.weight_args)
        return [
            Stage(
                self.name,
                p.tasks,
                partial(_linear, p=p),
                lds_words=p.lds_words,
                writes=tuple(writes),
                args=tuple(args),
                key=(p,),
                all_rows=True,  # (its row tiles cover every row: dead rows' outputs from clamped inputs, put)
            )
        ]

    def pack(self, layer) -> dict[str, torch.Tensor]:
        if self.weight_format == "bf16":
            return self._pack_bf16(layer)
        fmt = gemv.FORMATS[self.weight_format]
        mats = [layer.linears[n] for n in self.linears]
        for n, m in zip(self.linears, mats, strict=True):
            if getattr(m, "fmt", "mxfp4") != fmt.name:
                raise ValueError(
                    f"Linear {self.name}: {n} is {getattr(m, 'fmt', 'mxfp4')}, the block is {fmt.name}"
                )
        data = [
            m.packed.view(torch.uint8) for m in mats
        ]  # bytes: FP8 tensors stack and pack as uint8
        if isinstance(self.out, SwiGLU):
            gate, up = data
            g_codes, u_codes = (m.codes for m in mats)
            n2 = gate.shape[0]
            packed = torch.stack(
                (gate.reshape(n2 // 16, 16, -1), up.reshape(n2 // 16, 16, -1)), 1
            )
            codes = torch.stack(
                (g_codes.reshape(n2 // 16, 16, -1), u_codes.reshape(n2 // 16, 16, -1)),
                1,
            )
            packed, codes = packed.reshape(2 * n2, -1), codes.reshape(2 * n2, -1)
        else:
            packed = torch.cat(data)
            codes = torch.cat([m.codes for m in mats])
        if packed.shape != (self.n, fmt.stored_k(self.k)):
            raise ValueError(
                f"Linear {self.name}: stacked weight {tuple(packed.shape)} != ({self.n}, {fmt.stored_k(self.k)})"
            )
        w, s = gemv.pack_weight(packed, codes, fmt)
        out = dict(zip(self.weight_args, (w, s), strict=True))
        rs = [getattr(m, "row_scale", None) for m in mats]
        if self.row_scale or any(r is not None for r in rs):
            out[f"{self.name}.rs"] = self._pack_row_scale(rs)
        if isinstance(self.x, NormIn):
            out[self.x.gamma] = layer.retained[self.x.weight].contiguous()
        if self.bias:
            b = torch.cat([layer.retained[n].reshape(-1) for n in self.bias]).float()
            if b.shape != (self.n,):
                raise ValueError(
                    f"Linear {self.name}: bias {tuple(b.shape)} != ({self.n},)"
                )
            out[f"{self.name}.b"] = b.contiguous()
        return out

    def _pack_row_scale(self, rs: list) -> torch.Tensor:
        """``row_scale``: the linears' fp32 row scales in packed row order (SwiGLU: gate / up per 16-row group)."""
        if not self.row_scale or any(r is None for r in rs):
            raise ValueError(
                f"Linear {self.name}: row scales on {sum(r is not None for r in rs)} of {len(rs)} "
                f"linears with row_scale={self.row_scale}: every linear or none, and the block's option"
            )
        rs = [r.reshape(-1).float() for r in rs]
        if isinstance(self.out, SwiGLU):
            n2 = rs[0].shape[0]
            v = torch.stack(
                (rs[0].reshape(n2 // 16, 16), rs[1].reshape(n2 // 16, 16)), 1
            ).reshape(-1)
        else:
            v = torch.cat(rs)
        if v.shape != (self.n,):
            raise ValueError(
                f"Linear {self.name}: row scales {tuple(v.shape)} != ({self.n},)"
            )
        return v.contiguous()

    def _pack_bf16(self, layer) -> dict[str, torch.Tensor]:
        """bf16 weights: ``layer.linears[name]`` bf16 [N_i, K] tensors stacked along N (SwiGLU: gate / up interleaved
        per 16-row group, as the MX path) into ``gemv16.pack_bf16_mmajor``'s units; ``.ws`` a placeholder.
        """
        mats = [layer.linears[n] for n in self.linears]
        for n, m in zip(self.linears, mats, strict=True):
            if not isinstance(m, torch.Tensor) or m.dtype != torch.bfloat16:
                raise ValueError(
                    f"Linear {self.name}: {n} is not a bf16 tensor (weight_format 'bf16')"
                )
        if isinstance(self.out, SwiGLU):
            gate, up = mats
            n2 = gate.shape[0]
            w = torch.stack(
                (gate.reshape(n2 // 16, 16, -1), up.reshape(n2 // 16, 16, -1)), 1
            ).reshape(2 * n2, -1)
        else:
            w = torch.cat(mats)
        if w.shape != (self.n, self.k):
            raise ValueError(
                f"Linear {self.name}: stacked weight {tuple(w.shape)} != ({self.n}, {self.k})"
            )
        w_arg, s_arg = self.weight_args
        return {w_arg: gemv16.pack_bf16_mmajor(w.contiguous()),
                s_arg: torch.zeros(16, dtype=torch.uint8, device=w.device)}  # fmt: skip


# ------------------------------------------------------------------ device code


def _kstep(p, wk, i):
    """K step (within the slice) of K wave ``wk``'s i-th step: interleaved, so the
    CTA's waves read adjacent units at once (-0.15..0.25 us vs contiguous ranges);
    ``kwave`` "contiguous": wave ``wk`` takes steps ``wk c ..``, so ``wk`` K waves sum
    the same K ranges in the same order as ``wk`` K slices (bit-identical results)."""
    return wk + p.wk * i


def _issue(p, r_w, rg0, st0, wk, lane, u):
    """Wave unit ``u`` (K step outer, row group inner): its weight load."""
    i, gl = u // p.gw, u % p.gw
    unit = (rg0 + gl) * p.steps + st0 + _kstep(p, wk, i)
    valid = None
    if p.ks % p.wk != 0:
        valid = _kstep(p, wk, i) < p.ks
    return gemv.load_weight(r_w, unit, lane, valid, p.fmt)


def _scale_issue(k, p, r_s, b, s):
    """The task's weight scale codes, 16 B per thread per round, issued first:
    [(chunk vector, LDS word offset)]. Chunk ``ch`` of the task is unit
    ``ch // 4`` (row group ``u // ks``, K step ``u % ks`` of the slice), lanes
    ``16 (ch % 4) ..``."""
    out = []
    for c in range(-(-p.ws_chunks // THREADS)):
        ch = fx.min(fx.Int32(c * THREADS) + k.tid, fx.Int32(p.ws_chunks - 1))
        u = ch // 4
        g, st = u // p.ks, u % p.ks
        unit = (b * p.g + g) * p.steps + s * p.ks + st
        out.append((gemv.load_scales(r_s, unit * 4 + ch % 4), p.ws_base + ch * 4))
    return out


def _unit_scale(k, p, gt, st):
    """Lane's weight scale code of task row group ``gt``, slice K step ``st`` (from LDS; ``row_scale``: the constant
    1.0, never loaded)."""
    off = (p.ws_base * 4 + (gt * p.ks + st) * 64) + k.lane
    return fx.Int32(fx.ptr_load(fx.recast_iter(fx.Uint8, k.lds) + off))


def _unit_codes_bf16(k, p, gt, st):
    """The weight scale codes of the lane's two 16-value halves (``gemv.widen_bf16``): lane group j holds 32-blocks
    j // 2 and 2 + j // 2 of its row; the LDS bytes are [block][row] per K step (``_unit_scale``: byte = lane).
    """
    off = (p.ws_base * 4 + (gt * p.ks + st) * 64) + (k.lane >> 5) * 16 + k.lane % 16
    b = fx.recast_iter(fx.Uint8, k.lds)
    return fx.Int32(fx.ptr_load(b + off)), fx.Int32(fx.ptr_load(b + (off + 32)))


def _x_batches(k, p, s, rmap=None, xoff=0):
    """The input poll of K slice ``s`` (every row) in batches of at most POLL_MAX
    pair specs: [(specs, LDS word offsets)]. FP8 words go to LDS word 0 on (row
    pitch ``p.pitch``), codes to ``p.xs_base`` on (row pitch ``p.spitch``).
    ``rmap``: LDS row -> the channel row it is polled from (default: the same)."""
    rmap = rmap or (lambda r: r)
    words_row = p.kse // 4  # FP8 words per row in the slice
    specs_row = words_row // 2  # 2-pair specs
    nq, ns = p.rows * specs_row, p.rows * (p.kse // 32)
    aq, as_ = k.mb[f"{p.x}.q"], k.mb[f"{p.x}.s"]
    items = [("q", c) for c in range(-(-nq // THREADS))] + [
        ("s", c) for c in range(-(-ns // THREADS))
    ]
    out = []
    for b0 in range(0, len(items), mb.POLL_MAX):
        specs, dests = [], []
        for kind, c in items[b0 : b0 + mb.POLL_MAX]:
            if kind == "q":
                idx = fx.min(fx.Int32(c * THREADS) + k.tid, fx.Int32(nq - 1))
                r, w2 = idx // specs_row, idx % specs_row
                specs.append(
                    (aq, rmap(r) * (p.xk // 4) + xoff // 4 + s * words_row + w2 * 2, 2)
                )
                dests.append(r * p.pitch + w2 * 2)
            else:
                idx = fx.min(fx.Int32(c * THREADS) + k.tid, fx.Int32(ns - 1))
                r, gi = idx // (p.kse // 32), idx % (p.kse // 32)
                specs.append(
                    (
                        as_,
                        rmap(r) * (p.xk // 32) + xoff // 32 + s * (p.kse // 32) + gi,
                        1,
                    )
                )
                dests.append(p.xs_base + r * p.spitch + gi)
        out.append((specs, dests))
    return out


def _x_dense(k, p, s, xoff=0):
    """A FLAG input slice (every row): [(Addr, dense word, LDS word, is_code)] per 16-byte
    load of this thread (clamped duplicates past the end store the same bytes again)."""
    words_row, codes_row = p.kse // 4, p.kse // 32
    nq, ns = p.rows * words_row // 4, p.rows * codes_row // 4
    aq, as_ = k.mb[f"{p.x}.q"], k.mb[f"{p.x}.s"]
    out = []
    for c in range(-(-nq // THREADS)):
        idx = fx.min(fx.Int32(c * THREADS) + k.tid, fx.Int32(nq - 1))
        r, w4 = idx // (words_row // 4), idx % (words_row // 4)
        out.append(
            (
                aq,
                r * (p.xk // 4) + xoff // 4 + s * words_row + w4 * 4,
                r * p.pitch + w4 * 4,
                False,
            )
        )
    for c in range(-(-ns // THREADS)):
        idx = fx.min(fx.Int32(c * THREADS) + k.tid, fx.Int32(ns - 1))
        r, c4 = idx // (codes_row // 4), idx % (codes_row // 4)
        out.append(
            (
                as_,
                r * (p.xk // 32) + xoff // 32 + s * codes_row + c4 * 4,
                p.xs_base + r * p.spitch + c4 * 4,
                True,
            )
        )
    return out


def _land_dense(k, p, s, xoff=0):
    """After every flag was seen: the slice's dense words into the LDS activation layout."""
    items = _x_dense(k, p, s, xoff)
    _store_dense(k, items, [chan.load_words(a, w, 4) for a, w, _, _ in items])


def _store_dense(k, items, got):
    """``_x_dense``'s items' loaded words into LDS at their offsets."""
    for (_, _, off, code), ws in zip(items, got, strict=True):
        if code:  # odd row pitch: word stores
            for e in range(4):
                fx.ptr_store(ws[e], k.lds + (off + e))
        else:
            fx.ptr_store(fx.Vector.from_elements(ws, fx.Int32), k.lds + off)


def _try_batch(specs):
    """One poll attempt of a batch, issued now, not waited for: the raw (value, tag)
    words of every spec (device scope, as ``mailbox.poll``)."""
    out = []
    for a, i, n in specs:
        note(a.region, a.space, "poll")
        v = fx.Vector(mb.load_pairs(a, i, 2 * n))
        out.append([v[e] for e in range(2 * n)])
    return out


def _land_batch(k, p, specs, dests, words):
    """Store an attempt's values to LDS; a thread whose tags were not all current
    polls its specs again (spinning) and stores those."""
    bad = None
    for ws in words:
        for e in range(1, len(ws), 2):
            b = ws[e] != k.tag
            bad = b if bad is None else bad | b
    for ws, off in zip(words, dests, strict=True):
        for e in range(0, len(ws), 2):
            fx.ptr_store(ws[e], k.lds + (off + e // 2))
    _repoll(k, bad, specs, dests)


@traced
def _repoll(k, bad, specs, dests):
    if bad:
        got = mb.poll(specs, k.tag)
        for words, off in zip(got, dests, strict=True):
            for e in range_constexpr(len(words)):
                fx.ptr_store(words[e], k.lds + (off + e))


# ---- NormIn: the input rows normalized and quantized in the task's prologue


def _live_item(k, ok, n):
    """``ok`` and, in a ``Program.dyn_rows`` build, item row ``n`` live: a dead row's K-slice partials are neither put
    nor polled (its output, zeros, still goes out: the region stays whole). A static
    build: ``ok``."""
    return ok if isinstance(k.n, int) else ok & (n < k.n)


@traced
def _put_partial(k, p, ok, idx, vals):
    if ok:
        mb.put2(k.mb[p.part], idx, vals[0], vals[1], k.tag)
        mb.put2(k.mb[p.part], idx + 2, vals[2], vals[3], k.tag)


@traced
def _store_bf16(k, p, ok, e, vals):
    if ok:
        chan.put_bf16(k.mb[p.out_ch], e, vals, k.tag, p.out_proto == "flag")


@traced
def _store_mxfp8(k, p, ok, e, word, lead, code, width):
    """FP8 word of elements e .. e + 3 (row-major, row width ``width``); ``lead``
    lanes also store their group's scale code."""
    if ok:
        chan.put_mxfp8_word(
            k.mb[f"{p.out_ch}.q"], e // 4, word, k.tag, p.out_proto == "flag"
        )
        if lead:
            chan.put_mxfp8_scale(
                k.mb[f"{p.out_ch}.s"], e // 32, code, k.tag, p.out_proto == "flag"
            )


def _sum_red(k, p, base):
    """Sum over the K waves (wave order) of LDS partial words base + wk * stride .. + 3."""
    stride = p.g * p.rows * 16
    tot = None
    for w in range_constexpr(p.wk):
        v = fx.Vector(
            fx.ptr_load(
                k.ldsf + (base + w * stride),
                result_type=fx.Vector.make_type(4, fx.Float32),
            )
        )
        vals = [v[q] for q in range(4)]
        tot = vals if tot is None else [tot[q] + vals[q] for q in range(4)]
    return tot


@traced
def _poll_slices(k, p, ok, idx):
    """The total over the K slices of item ``idx`` (4 values), summed in slice
    order from the published partials (``ok`` False: zeros, no poll)."""
    t0, t1, t2, t3 = fx.Float32(0.0), fx.Float32(0.0), fx.Float32(0.0), fx.Float32(0.0)
    if ok:
        specs = []
        for s in range_constexpr(p.k_split):
            specs += [
                (k.mb[p.part], s * (p.rows * p.n) + idx, 2),
                (k.mb[p.part], s * (p.rows * p.n) + idx + 2, 2),
            ]
        got = mb.poll(specs, k.tag)
        parts = [
            [w.bitcast(fx.Float32) for w in got[2 * s] + got[2 * s + 1]]
            for s in range(p.k_split)
        ]
        acc = parts[0]
        for s in range_constexpr(1, p.k_split):
            acc = [acc[q] + parts[s][q] for q in range(4)]
        t0, t1, t2, t3 = acc[0], acc[1], acc[2], acc[3]
    return [t0, t1, t2, t3]


@traced
def _arrive(k, p, b):
    """Thread 0 counts this K slice in for row block ``b`` (its partials issued);
    LDS ``flag_word`` := 1 if it was the last slice to arrive. Every launch adds
    exactly ``k_split`` to each counter (zeroed scratch: 0), so the last arrival
    of a launch is the one that finds ``k_split - 1`` modulo ``k_split`` (a 32-bit
    counter for a power of two, which may wrap; else 64-bit). No tag in
    the counter: launches need distinct tags, not growing ones (a tag below the
    previous launch's, e.g. a graph replayed without a step bump, or the int32 tag
    wrapping, left a tag-keyed counter behind: no last slice, a hang downstream)."""
    if k.tid == 0:
        note(p.ctr, k.mb[p.ctr].space, "put")
        addr = k.mb[p.ctr].value + fx.Int64(b * 8)
        if const_expr(p.k_split & (p.k_split - 1) == 0):
            # a power of two: a 32-bit counter (2**32 is a multiple of k_split, so it may wrap)
            ptr = fx.inttoptr(
                fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4), addr
            )
            old = fx.atomic_add(ptr, fx.Int32(1), syncscope=fx.rocdl.SyncScope.Agent)
            last = (fx.Int32(old) & (p.k_split - 1)) == p.k_split - 1
        else:
            ptr = fx.inttoptr(
                fx.PointerType.get(fx.Int64.ir_type, fx.AddressSpace.Global, 8), addr
            )
            old = fx.atomic_add(ptr, fx.Int64(1), syncscope=fx.rocdl.SyncScope.Agent)
            last = fx.Int32(old % p.k_split) == p.k_split - 1
        fx.ptr_store(last.select(fx.Int32(1), fx.Int32(0)), k.lds + p.flag_word)


def _items(p, b, e_idx):
    """Epilogue item ``e_idx`` (clamped) of row block ``b``: (ok, row, [(LDS partial
    word, item index in [rows, N])] (gate then up for SwiGLU), output column)."""
    valid_e = e_idx < p.out_quads
    e = fx.min(e_idx, fx.Int32(p.out_quads - 1))
    f4 = e % 4
    n, gl = e // (p.g * 4), (e // 4) % p.g
    rg = b * p.g + gl
    ok = valid_e & (rg < p.n_rg)
    return (
        ok,
        n,
        [((gl * p.rows + n) * 16 + f4 * 4, n * p.n + rg * 16 + f4 * 4)],
        rg * 16 + f4 * 4,
        (gl % 2 == 0) & (f4 == 0),
    )


@traced
def _tp_reduce(k, p, ok, e, vals):
    """Row-parallel TP (module docstring): this rank's partial of elements e .. e + 3 to every
    rank, then (``tp_sum`` "producer") the ``tp`` ranks' partials summed in rank order (fp32);
    ``ok`` False: zeros. "consumer": the push only (the consumer sums: ``peer.sum_partials``).
    """
    t0, t1, t2, t3 = fx.Float32(0.0), fx.Float32(0.0), fx.Float32(0.0), fx.Float32(0.0)
    if ok:  # noqa: SIM102 (traced)
        if const_expr(
            p.tp_sum == "rs"
        ):  # two-shot: to the columns' owner only (a task's columns: one owner)
            peer.push_rs(k, p.tp_reg, e, vals, p.n, p.tp_partial)
    return [t0, t1, t2, t3]


def _emit_out(k, p, ok, n, col, lead, vals, res_words):
    """The output spec of one item's totals (``vals``: one or, SwiGLU, two lists of 4)."""
    if p.tp_partial:  # never SwiGLU (the constructor)
        vals = [_tp_reduce(k, p, ok, n * p.n + col, vals[0])]
        if const_expr(
            p.tp_sum != "producer"
        ):  # pushed: the consumer sums and owns the output
            return
    if p.kind == "bf16":
        _store_bf16(k, p, ok, n * p.n + col, vals[0])
    else:
        _quant_out(k, p, ok, n * p.n + col, [bf16_round(v) for v in vals[0]], lead, p.n)


def _finish(k, p, s, b, res_words):
    """The epilogue after the K waves' partials are in LDS. With K slices, every
    slice publishes its partial and arrives; the last to arrive sums all slices
    (slice order, so the result does not depend on who is last) and stores."""
    passes = -(-p.out_quads // THREADS)
    if p.k_split == 1:
        for ps in range(passes):
            ok, n, srcs, col, lead = _items(p, b, fx.Int32(ps * THREADS) + k.tid)
            _emit_out(
                k,
                p,
                ok,
                n,
                col,
                lead,
                [_sum_red(k, p, w) for w, _ in srcs],
                res_words[ps],
            )
        return
    for ps in range(passes):
        ok, n, srcs, col, lead = _items(p, b, fx.Int32(ps * THREADS) + k.tid)
        for w, idx in srcs:
            _put_partial(
                k, p, _live_item(k, ok, n), s * (p.rows * p.n) + idx, _sum_red(k, p, w)
            )
    # no wait for the partial stores before arriving: the last arrival's polls check
    # tags, so a partial still in flight is waited for there (-0.5 us at down B1S8)
    _arrive(k, p, b)
    gpu.barrier()
    _last_slice(k, p, b, fx.ptr_load(k.lds + p.flag_word) != 0, res_words)


@traced
def _last_slice(k, p, b, last, res_words):
    if last:
        _finish_last(k, p, b, res_words)


def _finish_last(k, p, b, res_words):
    for ps in range(-(-p.out_quads // THREADS)):
        ok, n, srcs, col, lead = _items(p, b, fx.Int32(ps * THREADS) + k.tid)
        live = _live_item(k, ok, n)
        _emit_out(
            k,
            p,
            ok,
            n,
            col,
            lead,
            [_poll_slices(k, p, live, idx) for _, idx in srcs],
            res_words[ps],
        )


def _quant_out(k, p, ok, e, vals, lead, width):
    """MXFP8 of four values per lane, groups of 32 over 8 adjacent lanes."""
    amax = gemv.group_amax(vals, 8)
    code = gemv.activation_code(amax)
    word = gemv.quant_word(vals, code)
    _store_mxfp8(k, p, ok, e, word, lead, code, width)


def _res_prefetch(k, p, b):
    """The residual's words for every epilogue pass, issued with the weights: a plain
    arg's two words, or a first poll attempt of the channel's two pairs (value,
    tag, value, tag; landed in the epilogue by ``_res_words``). With ``row_scale`` each
    pass's entry is (those words, the row scales)."""
    out = _res_issue(k, p, b)
    return out


def _res_issue(k, p, b):
    passes = -(-p.out_quads // THREADS)
    if p.kind != "residual":
        return [_bias_prefetch(k, p, b, ps) for ps in range(passes)]
    out = []
    for ps in range(passes):
        e = fx.min(fx.Int32(ps * THREADS) + k.tid, fx.Int32(p.out_quads - 1))
        n, gl, f4 = e // (p.g * 4), (e // 4) % p.g, e % 4
        rg = fx.min(b * p.g + gl, fx.Int32(p.n_rg - 1))
        idx = n * p.n + rg * 16 + f4 * 4
        if p.res_ch:
            out.append(_try_batch([(k.mb[p.res_ch], idx // 2, 2)])[0])
        else:
            v = fx.Vector(
                bo.buffer_load(
                    rsrc(k.args[p.res_arg]), idx // 2, vec_width=2, dtype=T.i32
                )
            )
            out.append([v[0], v[1]])
        out[-1] += _bias_prefetch(k, p, b, ps)
    return out


def _bias_prefetch(k, p, b, ps):
    """The bias of pass ``ps``'s four output columns (f32), issued with the weights ([] without a bias)."""
    if not p.bias:
        return []
    e = fx.min(fx.Int32(ps * THREADS) + k.tid, fx.Int32(p.out_quads - 1))
    gl, f4 = (e // 4) % p.g, e % 4
    rg = fx.min(b * p.g + gl, fx.Int32(p.n_rg - 1))
    v = fx.Vector(
        bo.buffer_load(rsrc(k.args[p.bias]), rg * 16 + f4 * 4, vec_width=4, dtype=T.f32)
    )
    return [fx.Float32(v[q]) for q in range(4)]


@traced
def _red_store(k, p, wr, wk, acc, chunk=0):
    """This wave's partials -> LDS [wk][row group][row][16 features] (useful lanes only); ``chunk``: the 16-row chunk
    ``acc`` holds (rows 16 chunk ..)."""
    n = k.lane % 16 + 16 * chunk if chunk else k.lane % 16
    if n < p.rows:
        for gl in range_constexpr(p.gw):
            off = ((wk * p.g + wr * p.gw + gl) * p.rows + n) * 16 + (k.lane // 16) * 4
            fx.ptr_store(acc[gl], k.ldsf + off)


def _linear(k, t, p):
    """One task: ``t`` = K slice * blocks + row block. A streamed input's task sees an opaque copy of the thread id
    (``_fresh_lanes``)."""
    if p.xchunk:
        k = _fresh_lanes(k)
    s = t // p.blocks if p.k_split > 1 else fx.Int32(0)
    b = t % p.blocks if p.k_split > 1 else t
    lane = k.lane
    wr, wk = k.wave // p.wk, k.wave % p.wk
    r_w = rsrc(k.args[f"{p.name}.w"], p.w_bytes)
    r_s = rsrc(k.args[f"{p.name}.ws"], p.s_bytes)
    rg0 = b * p.g + wr * p.gw
    st0 = s * p.ks
    # groups: the input columns of the task's group (its row groups lie in one: Linear.plan)
    xoff = (b * p.g) // (p.n // p.groups // 16) * p.k if p.groups > 1 else 0
    units = p.c * p.gw
    issue = partial(_issue, p, r_w, rg0, st0, wk, lane)
    wide = []  # prewiden: the first units' bf16 operands

    # 1. weights first, then the plain residual, then the input. The
    # input's first poll attempt goes out after the first ``p.lead`` units, the
    # rest of the prefetch after it: vmcnt is in order, so the attempt lands
    # (and, if the input is there, the MFMAs start) while most weights stream.
    n_pre = min(p.prefetch, units)
    lead = min(p.lead, n_pre)
    n_first = (
        max(lead, min(p.pre_units, n_pre))
        if p.pre_units and not (p.norm_src or p.cin_src)
        else n_pre
    )
    scales = _scale_issue(k, p, r_s, b, s)
    rocdl.sched_barrier(0)
    q = [issue(u) for u in range(lead)]
    res_words = _res_prefetch(k, p, b)
    if p.x_proto == "flag":
        # bulk + flag: one attempt of the flags after ``lead`` units (cheap), the rest of the
        # prefetch, then the flags' spin and a CTA barrier, then the dense words (they return
        # after the prefetch: vmcnt is in order)
        rocdl.sched_barrier(0)
        fspecs, fraw = chan.flag_try(k, f"{p.x}.f")
        rocdl.sched_barrier(0)
        q += [issue(u) for u in range(lead, n_first)]
        idle = idle_issue(k)
        chan.flag_land(k, fspecs, fraw)
        if p.xchunk:  # a streamed input: its first ring slot only
            _land_dense(k, *_xslot(p, s, 0), xoff)
        else:
            _land_dense(k, p, s, xoff)
    else:  # a streamed input: its first ring slot only (``_stream_ring`` polls the others)
        batches = (
            _x_batches(k, *_xslot(p, s, 0), xoff=xoff)
            if p.xchunk
            else _x_batches(k, p, s, xoff=xoff)
        )
        rocdl.sched_barrier(
            0
        )  # keep the issue order: LLVM would sink the attempt below the weights
        tries = [_try_batch(specs) for specs, _ in batches]
        rocdl.sched_barrier(0)
        q += [issue(u) for u in range(lead, n_first)]
        idle = idle_issue(
            k
        )  # the stage's idle work (Stage.idle), behind the prefetch and the attempt
        if (
            p.prewiden
        ):  # the scale codes to LDS and the first units widened while the input is out
            wide = _prewiden(k, p, q, scales, wr, wk, min(p.prewiden, n_first))
        for (specs, dests), words in zip(batches[:-1], tries[:-1], strict=True):
            _land_batch(k, p, specs, dests, words)
        _land_batch(k, p, *batches[-1], tries[-1])
    if not wide:
        for v, off in scales:
            fx.ptr_store(v, k.lds + off)
    gpu.barrier()

    # 2. stream, in this order per K step (sched barriers keep it: LLVM would sink
    # each load next to its MFMA and collapse the prefetch to one unit): the
    # next K step's LDS operands, then per row group the load ``prefetch`` units
    # ahead and this unit's MFMA.
    row = fx.min(lane % 16, fx.Int32(p.rows - 1))
    j = lane // 16

    def operands(i):
        st = _kstep(p, wk, i)
        if p.ks % p.wk != 0:
            st = fx.min(st, fx.Int32(p.ks - 1))
        x, xs = gemv.b_operand(k.lds, k.lds + p.xs_base, row, j, st, p.pitch, p.spitch)
        if (
            p.compute == "bf16"
        ):  # widened once a K step, shared by the wave's row groups
            x = gemv.widen_bf16(
                [x[e] for e in range(8)],
                gemv.b_codes_bf16(k.lds + p.xs_base, row, j, st, p.spitch),
            )
            return (
                x,
                xs,
                [_unit_codes_bf16(k, p, wr * p.gw + gl, st) for gl in range(p.gw)],
            )
        return x, xs, [_unit_scale(k, p, wr * p.gw + gl, st) for gl in range(p.gw)]

    if p.xchunk:  # the input through the LDS ring, polled while the weights stream
        _stream_ring(k, p, q, issue, units, wr, wk, lane, s, xoff)
        _finish(k, p, s, b, res_words)
        idle_sink(k, idle)
        gpu.barrier()
        return
    if (
        p.rchunks > 1
    ):  # more than 16 rows: one MFMA per 16-row chunk on each weight unit (loaded once)
        _stream_chunks(k, p, q, issue, units, wr, wk, lane)
        _finish(k, p, s, b, res_words)
        idle_sink(k, idle)
        gpu.barrier()
        return
    acc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(p.gw)]
    cur = operands(0)
    for i in range(p.c):
        nxt = operands(i + 1) if i + 1 < p.c else None
        x, xs, ws = cur
        for gl in range(p.gw):
            u = i * p.gw + gl
            rocdl.sched_barrier(0)
            if u + p.prefetch < units:
                q.append(issue(u + p.prefetch))
            if p.compute == "bf16":
                wv = (
                    wide[u]
                    if u < len(wide)
                    else gemv.widen_bf16([q[u][e] for e in range(8)], ws[gl])
                )
                acc[gl] = gemv.mfma_bf16_wide(acc[gl], wv, x)
            else:
                acc[gl] = gemv.mfma_wxa8(acc[gl], q[u], ws[gl], x, xs, p.fmt)
        rocdl.sched_barrier(0)
        cur = nxt

    # 3. K waves' partials through LDS (over the activation), then the epilogue
    gpu.barrier()
    _red_store(k, p, wr, wk, acc)
    gpu.barrier()
    _finish(k, p, s, b, res_words)
    idle_sink(k, idle)
    gpu.barrier()  # the next task (or stage) reuses the LDS


def _prewiden(k, p, q, scales, wr, wk, n):
    """bf16 compute: the task's weight scale codes into LDS (their loads were issued first), then this wave's first
    ``n`` units widened to bf16 (``gemv.widen_bf16``, the same operands the stream would build) while the input's
    attempt is still out: the widening leaves the path from the input's landing to the MFMAs.
    """
    for v, off in scales:
        fx.ptr_store(v, k.lds + off)
    gpu.barrier()
    out = []
    for u in range(n):
        i, gl = u // p.gw, u % p.gw
        st = _kstep(p, wk, i)
        out.append(
            gemv.widen_bf16(
                [q[u][e] for e in range(8)], _unit_codes_bf16(k, p, wr * p.gw + gl, st)
            )
        )
    return out


def _stream_chunks(k, p, q, issue, units, wr, wk, lane):
    """The stream of a task over ``p.rchunks`` 16-row chunks (more than 16 rows): per K step the chunks' LDS operands,
    then per row group the next load and one MFMA per chunk on the same weight registers; the chunks' partials to
    LDS (the epilogue reads ``rows`` rows). The scaled MX compute, the bf16 compute (each weight unit widened once
    a K step, shared by the chunks; each chunk's x widened as in one chunk), or bf16 weights (``a16``: each unit's
    operands built once, shared by the chunks; each chunk's bf16 B operands as in one chunk): a row's products and
    sums are those of the one-chunk stream (the MFMA's rows are independent), so its bits are the same at any row
    count."""
    j = lane // 16
    rows = [
        fx.min(lane % 16 + 16 * ch, fx.Int32(p.rows - 1)) for ch in range(p.rchunks)
    ]
    wide = p.compute == "bf16"

    def operands(i):
        st = _kstep(p, wk, i)
        if p.ks % p.wk != 0:
            st = fx.min(st, fx.Int32(p.ks - 1))
        xs = [
            gemv.b_operand(k.lds, k.lds + p.xs_base, r, j, st, p.pitch, p.spitch)
            for r in rows
        ]
        if wide:
            xs = [(gemv.widen_bf16([x[e] for e in range(8)], gemv.b_codes_bf16(k.lds + p.xs_base, r, j, st, p.spitch)),
                   xsc) for r, (x, xsc) in zip(rows, xs, strict=True)]  # fmt: skip
            return xs, [
                _unit_codes_bf16(k, p, wr * p.gw + gl, st) for gl in range(p.gw)
            ]
        return xs, [_unit_scale(k, p, wr * p.gw + gl, st) for gl in range(p.gw)]

    acc = [
        [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(p.gw)]
        for _ in range(p.rchunks)
    ]
    cur = operands(0)
    for i in range(p.c):
        nxt = operands(i + 1) if i + 1 < p.c else None
        xs, ws = cur
        for gl in range(p.gw):
            u = i * p.gw + gl
            rocdl.sched_barrier(0)
            if u + p.prefetch < units:
                q.append(issue(u + p.prefetch))
            w = gemv.widen_bf16([q[u][e] for e in range(8)], ws[gl]) if wide else None
            for ch in range(p.rchunks):
                x, xsc = xs[ch]
                if wide:
                    acc[ch][gl] = gemv.mfma_bf16_wide(acc[ch][gl], w, x)
                else:
                    acc[ch][gl] = gemv.mfma_wxa8(
                        acc[ch][gl], q[u], ws[gl], x, xsc, p.fmt
                    )
        rocdl.sched_barrier(0)
        cur = nxt
    gpu.barrier()
    for ch in range(p.rchunks):
        _red_store(k, p, wr, wk, acc[ch], ch)
    gpu.barrier()


def _fresh_lanes(k):
    """``k`` with the thread id and lane from an opaque copy of the thread id: LLVM would otherwise share the
    task's lane arithmetic with an earlier stage's and hold it across that stage (24 VGPRs of spills seen in a composed
    layer at 64 rows)."""
    from types import SimpleNamespace

    tid = fresh(k.tid)
    return SimpleNamespace(**{**vars(k), "tid": tid, "lane": tid % WAVE_LANES})


def _xslot(p, s, ci):
    """(the plan seen as K slices of one ring slot each, slot ``ci`` of K slice ``s`` as such a slice): what
    ``_x_batches`` polls into LDS from word 0 (the ring slot's own offset is added by the caller).
    """
    from dataclasses import replace

    per = p.ks // p.xchunk
    return replace(p, k_split=p.k_split * per, xchunk=0), s * per + ci


def _stream_ring(k, p, q, issue, units, wr, wk, lane, s, xoff):
    """The stream of a task whose input goes through the LDS ring (``p.xchunk`` K steps a slot, two slots): slot
    ``ci`` holds the slice's K steps ``ci xchunk ..``, i.e. iterations ``ci ipc ..`` of every K wave (interleaved).
    Slots ``ci + 1 .. ci + xlead`` have their poll attempts in flight (registers) while slot ``ci`` streams; at its
    end slot ``ci + 1``'s lands into the other LDS slot (last read by slot ``ci - 1``, behind the previous barrier) and
    one CTA barrier hands it over. vmcnt is in order: the landing waits for every load issued before that attempt, so
    the lead (not the slot) is what keeps ``xlead`` slots of weight units in flight across it. Per iteration: the
    weight loads ``prefetch`` units ahead, then per 16-row chunk its B operand (the next chunk's read one ahead) and
    one MFMA per row group on the same weight registers. The sums (per K wave, K steps in order) and the K waves'
    partials through LDS are the resident stream's: the same bits as the same shape with the slice resident. The
    lane's operand rows come from ``_fresh_lanes``'s thread id."""
    j = lane // 16
    rows = [
        fx.min(lane % 16 + 16 * ch, fx.Int32(p.rows - 1)) for ch in range(p.rchunks)
    ]
    nslots = p.ks // p.xchunk
    ipc = p.xchunk // p.wk  # iterations a slot
    nbuf = p.xbufs

    def operand(i, ch):
        sl = (i // ipc) % nbuf
        st = wk + p.wk * (i % ipc)
        base = k.lds + sl * p.xslot
        return gemv.b_operand(
            base, base + p.xs_base, rows[ch], j, st, p.pitch, p.spitch
        )

    def scale(gt, st):
        return _unit_scale(k, p, gt, st)

    def attempt(
        si,
    ):  # slot si's poll attempt (dense: its loads, the flags seen), issued now (LDS slot: si % nbuf)
        view, sv = _xslot(p, s, si)
        off = (si % nbuf) * p.xslot
        rocdl.sched_barrier(0)
        if p.x_proto == "flag":
            items = [
                (a, w, d + off, code) for a, w, d, code in _x_dense(k, view, sv, xoff)
            ]
            got = [chan.load_words(a, w, 4) for a, w, _, _ in items]
            rocdl.sched_barrier(0)
            return "dense", items, got
        nxt = [
            (specs, [d + off for d in dests])
            for specs, dests in _x_batches(k, view, sv, xoff=xoff)
        ]
        tries = [_try_batch(specs) for specs, _ in nxt]
        rocdl.sched_barrier(0)
        return "tagged", nxt, tries

    pending = [
        attempt(si) for si in range(1, min(p.xlead, nslots))
    ]  # slots 1 .. xlead - 1, behind the prefetch
    acc = [
        [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(p.gw)]
        for _ in range(p.rchunks)
    ]
    for ci in range(nslots):
        if (
            ci + p.xlead < nslots
        ):  # slot ci + xlead's attempt, behind the weights already in flight
            pending.append(attempt(ci + p.xlead))
        seq = [
            (i, ch) for i in range(ci * ipc, (ci + 1) * ipc) for ch in range(p.rchunks)
        ]
        cur = operand(*seq[0])
        for n, (i, ch) in enumerate(seq):
            nb = operand(*seq[n + 1]) if n + 1 < len(seq) else None
            if ch == 0:
                ws = [scale(wr * p.gw + gl, _kstep(p, wk, i)) for gl in range(p.gw)]
                for gl in range(p.gw):
                    u = i * p.gw + gl
                    rocdl.sched_barrier(0)
                    if u + p.prefetch < units:
                        q.append(issue(u + p.prefetch))
            x, xsc = cur
            for gl in range(p.gw):
                acc[ch][gl] = gemv.mfma_wxa8(
                    acc[ch][gl], q[i * p.gw + gl], ws[gl], x, xsc, p.fmt
                )
            rocdl.sched_barrier(0)
            cur = nb
        if ci + 1 < nslots:
            mode, nxt, tries = pending.pop(0)
            if mode == "dense":
                _store_dense(k, nxt, tries)
            else:
                for (specs, dests), words in zip(nxt, tries, strict=True):
                    _land_batch(k, p, specs, dests, words)
        gpu.barrier()  # slot ci + 1 landed for every wave; slot ci free for ci + 2
    for ch in range(p.rchunks):
        _red_store(k, p, wr, wk, acc[ch], ch)
    gpu.barrier()


# ------------------------------------------------------------------ a source of channels (tests, block evals)


def _source(k, t, ch, fmt, words, items, per_thread):
    """Copy plain arg ``<ch><fmt>`` into channel region ``<ch><fmt>`` as tagged
    pairs, two words per store (every load issued before the first store)."""
    got = _source_loads(k, t, ch, fmt, items, per_thread)
    for c in range(per_thread):
        it = (t * per_thread + c) * THREADS + k.tid
        _put_pair(k.mb[f"{ch}{fmt}"], it < items, it * 2, got[c], k.tag)


def _source_flag(k, t, ch, fmt, words, items, per_thread, slot0, dense):
    """The same (dense words for a FLAG channel), then this task's flag ``slot0 + t``."""
    got = _source_loads(k, t, ch, fmt, items, per_thread)
    for c in range(per_thread):
        it = (t * per_thread + c) * THREADS + k.tid
        _put_pair(k.mb[f"{ch}{fmt}"], it < items, it * 2, got[c], k.tag, dense)
    chan.put_flag(k, f"{ch}.f", t + slot0)


def _source_loads(k, t, ch, fmt, items, per_thread):
    got = []
    for c in range(per_thread):
        it = fx.min((t * per_thread + c) * THREADS + k.tid, fx.Int32(items - 1))
        got.append(
            fx.Vector(
                bo.buffer_load(
                    rsrc(k.args[f"{ch}{fmt}"]), it * 2, vec_width=2, dtype=T.i32
                )
            )
        )
    return got


@traced
def _put_pair(a, ok, i, w, tag, dense=False):
    if ok:
        chan.put_words(a, i, [w[0], w[1]], tag, dense)


def _source_tasks(items: int, per_thread: int) -> int:
    return -(-items // (THREADS * per_thread))


def source_stages(ch: Channel, rows: int, per_thread: int = 2) -> list[Stage]:
    """Stages that publish channel ``ch`` [rows, width] from plain args: the stand-in
    for its producer in block tests and evals. MXFP8 reads args ``<ch>.q`` (E4M3
    bytes) and ``<ch>.s`` (one int32 code per group), BF16 reads arg ``<ch>`` (bf16),
    F32 reads arg ``<ch>`` (f32). A FLAG channel's flags: one per source task, the
    ``.q`` stage's slots first."""
    if ch.fmt is Fmt.MXFP8:
        parts = [(".q", rows * ch.width // 4), (".s", rows * ch.width // 32)]
    out, slot0 = [], 0
    ntasks = sum(_source_tasks(w // 2, per_thread) for _, w in parts)
    for fmt, words in parts:
        if words % 2:
            raise ValueError("source stages copy two words per store")
        items = words // 2
        name = f"src:{ch.name}{fmt}"
        tasks = _source_tasks(items, per_thread)
        writes = [Region(f"{ch.name}{fmt}", words if not ch.dense else items, (name,))]
        if ch.flagged and slot0 == 0:
            writes.append(
                Region(
                    ch.flag_region, ntasks, tuple(f"src:{ch.name}{f}" for f, _ in parts)
                )
            )
        out.append(
            Stage(
                name,
                tasks,
                partial(
                    _source_flag if ch.flagged else _source,
                    ch=ch.name,
                    fmt=fmt,
                    words=words,
                    items=items,
                    per_thread=per_thread,
                    **({"slot0": slot0, "dense": ch.dense} if ch.flagged else {}),
                ),
                writes=tuple(writes),
                args=(f"{ch.name}{fmt}",),
                key=(ch.name, fmt, words, per_thread, ch.proto.value, slot0),
            )
        )
        slot0 += tasks
    return out

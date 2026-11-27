# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The DSpark draft's context KV write (``DeepseekV41DSpark.write_context_kv``)
in one launch, after the target's mono step, on the step's rows:

1. ``proj``: ``main_proj(concat(taps))``, its 5120 columns split over the
   attention TP group (``tp``; one rank under DP attention: every column of
   its own rows). A CTA task is 8 row groups of the rank's columns (a wave
   each) x a K slice; it quantizes the slice of the taps it reads (MXFP8,
   group 32, ``quantize_fp8``) into LDS a 16-row block at a time and writes
   its fp32 partials to PARTS [slice][row][column], then its flag (PF).
2. ``reduce``: a 16-row block x 128 columns: the slices' partials summed,
   rounded to bf16 (the GEMM's output), pushed to every group rank's CX as
   tagged pairs.
3. ``mnorm``: a CTA a row: its 5120 columns polled from CX, ``main_norm``
   (bf16), MXFP8 -> H8 / H8S, its flag (HF).
4. ``wkv``: the three stages' ``wkv`` (each the KV rows of its stage's own
   ``wqkv_a``, read in place: a [512, 5120] FP8 weight and its E8M0 rows, every
   rank all of it; one [1536, 5120] product), a task 8 row groups of one stage
   x a K slice, the rows' H8 in LDS -> fp32 partials KVP [slice][row][column],
   its flag (WF).
5. ``kvrow``: a wave a (row, stage): the slices summed -> bf16, ``kv_norm``
   -> bf16, RoPE of the last 64 columns -> bf16, the QAT (E4M3, group 32,
   ceil E8M0) into the stage's fp8 window ring by ``write_window``'s rule.

Rounding points and quantization formulas are the original chain's (main_norm
and kv_norm write bf16, RoPE writes bf16, the GEMMs output bf16); the sums run
in their own order. Every GEMV task loads its weights once and runs every
16-row block on them: nothing here grows with the rows but the trip counts.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from atom.models.deepseek_v41.mono.kernels.attn_pre import (
    _fp8_step_mfma,
    _gemv_steps,
    gemv_fp8_loads,
    ld_bf,
    ld_bf8,
    lds_row,
)
from atom.models.deepseek_v41.mono.kernels.dims import HIDDEN, KV_DIM, Dims
from atom.models.deepseek_v41.mono.kernels.mx import AMAX_FLOOR
from atom.models.deepseek_v41.mono.kernels.peer_layout import peer_offsets
from atom.models.deepseek_v41.mono.sources import SOURCES
from atom.mono.device.mx import FP8_MAX, clamp_fp8, code_ceil, rcp_pow2
from atom.mono.device.ops import (
    bf2_f32,
    bf16_round,
    butterfly,
    fp8_pack4,
    kernel_symbol,
    launcher,
    ld_i32,
    load_ptr64,
    rms_rstd,
    rsrc,
    traced,
    xshfl,
)
from atom.mono.device.ranks import peer_bases
from atom.mono.device.sync import Mailbox, preg, publish, sreg
from atom.mono.plan.arith import pick
from atom.mono.plan.build_key import key_tuple
from atom.mono.plan.execution import BLOCKS, THREADS, WAVES
from atom.mono.plan.layout import pair_layout
from atom.mono.runtime.abi import KernelAbi

_CURRENT_STREAM = fx.Stream(None)

TAPS = 3
STAGES = 3
K_PROJ = TAPS * HIDDEN
KV_ROWS = STAGES * KV_DIM
KSTEP = 128  # K an MFMA step
RG = 16  # weight rows a row group, step rows a block
SETS = WAVES  # row groups a GEMV task: a wave each
ROPE = 64
HALF = ROPE // 2
WKV_SLICE_STEPS = 2  # wkv: 40 K steps in 20 slices, 12 x 20 tasks
STAGE_SETS = KV_DIM // RG // SETS  # wkv tasks' row-group sets a stage
# the launch's mailbox tag: past every layer's and the column gathers'
CTX_TAG = 240

ABI = KernelAbi(
    (
        "tap0", "tap1", "tap2", "wp", "wps", "wk0", "wk1", "wk2", "wks0",
        "wks1", "wks2", "mnorm_w", "kvn0", "kvn1", "kvn2", "pos", "pos_w", "ring_pos", "ring_pos_w", "batch",
        "cu", "slots", "cached", "cached_w", "wptrs", "layer0", "layer1",
        "layer2", "ring", "cos", "sin", "max_pos", "parts", "h8", "kvp",
        "scratch", "sym", "peers", "rank",
    )
)  # fmt: skip


@dataclass(frozen=True)
class DraftCtxBuild:
    rows: int
    # the attention TP group the projection's columns are split over (1:
    # DP attention, or a single rank)
    tp: int
    mnorm_eps: float
    kv_eps: float

    def __post_init__(self):
        assert self.rows > 0 and self.tp in (1, 2, 4, 8)
        assert self.groups % SETS == 0

    @property
    def cols(self) -> int:
        """A rank's columns of the projection."""
        return HIDDEN // self.tp

    @property
    def groups(self) -> int:
        return self.cols // RG

    @property
    def sets(self) -> int:
        return self.groups // SETS

    @property
    def slice_steps(self) -> int:
        """K steps a projection task: 3 at TP8 (40 slices, 200 tasks), 5
        below (24 slices; a wave's weights stay in registers)."""
        return 3 if self.tp == 8 else 5

    @property
    def slices(self) -> int:
        return K_PROJ // KSTEP // self.slice_steps

    @property
    def proj_tasks(self) -> int:
        return self.sets * self.slices

    @property
    def blocks(self) -> int:
        return -(-self.rows // RG)

    @property
    def reduce_tasks(self) -> int:
        return self.blocks * (self.cols // 128)

    @property
    def wkv_sets(self) -> int:
        return KV_ROWS // RG // SETS

    @property
    def wkv_slices(self) -> int:
        return HIDDEN // KSTEP // WKV_SLICE_STEPS

    @property
    def wkv_tasks(self) -> int:
        return self.wkv_sets * self.wkv_slices

    @property
    def x_words(self) -> int:
        """An LDS row of a projection slice's MXFP8 words (padded)."""
        return lds_row(self.slice_steps * KSTEP // 4)

    @property
    def x_codes(self) -> int:
        return lds_row(self.slice_steps * KSTEP // 32)

    @property
    def lds_words(self) -> int:
        return RG * (self.x_words + self.x_codes) + WAVES


def scratch_layout(rows: int, tp: int) -> dict[str, tuple[int, int]]:
    """The launch's flags (device scope, zeroed at the step's begin): PF a
    projection task's, HF a row's normed MXFP8 row, WF a wkv task's."""
    p = DraftCtxBuild(rows, tp, 0.0, 0.0)
    return pair_layout([("pf", p.proj_tasks), ("hf", rows), ("wf", p.wkv_tasks)])


def scratch_bytes(rows: int, tp: int) -> int:
    return max(off + n for off, n in scratch_layout(rows, tp).values())


def parts_bytes(rows: int, tp: int) -> int:
    """PARTS: the projection's fp32 partials [slice][row][rank column]."""
    p = DraftCtxBuild(rows, tp, 0.0, 0.0)
    return p.slices * rows * p.cols * 4


def h8_bytes(rows: int) -> int:
    """H8 then H8S: the normed rows' E4M3 bytes [row][5120], their codes
    [row][160]."""
    return rows * (HIDDEN + HIDDEN // 32)


def kvp_bytes(rows: int) -> int:
    """KVP: the wkv's fp32 partials [slice][row][1536]."""
    return HIDDEN // KSTEP // WKV_SLICE_STEPS * rows * KV_ROWS * 4


def _mx_words(vs):
    """32-col group values -> (E8M0 code, their E4M3 bytes as 8 words)."""
    amax = fx.Float32(0.0)
    for v in vs:
        amax = fx.max(amax, abs(v))
    code = code_ceil(fx.max(amax, fx.Float32(AMAX_FLOOR)) * fx.Float32(1.0 / FP8_MAX))
    inv = rcp_pow2(code)
    words = [
        fp8_pack4(*[clamp_fp8(vs[4 * w + i] * inv) for i in range(4)])
        for w in range(len(vs) // 4)
    ]
    return code, words


SUM_BATCH = 8  # slices' loads in flight at once (their VGPRs)


def _slice_sum(r, slices, at, vecs):
    """Partials of ``slices`` slices (``vecs`` f32x4 from element ``at(q)``
    each) summed in slice order, ``SUM_BATCH`` slices' loads at a time."""
    s = None
    for q0 in range_constexpr(0, slices, SUM_BATCH):
        batch = [
            fx.Vector(bo.buffer_load(r, at(q) + 4 * h, vec_width=4, dtype=T.f32))
            for q in range(q0, min(slices, q0 + SUM_BATCH))
            for h in range(vecs)
        ]
        for v in batch_rows(batch, vecs):
            s = v if s is None else [s[i] + v[i] for i in range(4 * vecs)]
    return s


def batch_rows(batch, vecs):
    """A batch of loads as one list of values a slice."""
    return [
        [batch[i + h][e] for h in range(vecs) for e in range(4)]
        for i in range(0, len(batch), vecs)
    ]


def _acc(c, k, ops, rows):
    """The wave's 16 x 16 tile of its row group against the block in LDS."""
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i, (st, live) in enumerate(_gemv_steps(c, k, 1, True)):
        acc = _fp8_step_mfma(c, k, c["xl"], c["xsl"], st, ops[i], acc, live, rows)
    return acc


@traced
def quant_taps(c, b, n, q):
    """Rows 16 b .. + n of the taps, K slice q: MXFP8 into LDS, a 32-col
    group a thread."""
    p, a, tid = c["key"], c["args"], c["tid"]
    per = p.slice_steps * KSTEP // 32
    for it in range_constexpr(-(-RG * per // THREADS)):
        i = tid + it * THREADS
        r, g = i // per, i % per
        if i < n * per:
            kk = (q * per + g) * 32
            tap, col = kk // HIDDEN, kk % HIDDEN
            src = rsrc(pick([a["tap0"], a["tap1"], a["tap2"]], tap))
            at = ((b * RG + r) * HIDDEN + col) // 2
            vs = []
            for w in range_constexpr(4):
                v = fx.Vector(bo.buffer_load(src, at + 4 * w, vec_width=4, dtype=T.i32))
                for e in range_constexpr(4):
                    vs += list(bf2_f32(v[e]))
            code, words = _mx_words(vs)
            xl = c["xl"] + (r * p.x_words + g * 8)
            fx.ptr_store(fx.Vector.from_elements(words[0:4], fx.Int32), xl)
            fx.ptr_store(fx.Vector.from_elements(words[4:8], fx.Int32), xl + 4)
            fx.ptr_store(fx.Int32(code), c["xsl"] + (r * p.x_codes + g))


@traced
def proj(c, task):
    """Projection task ``task``: 8 row groups x K slice, every block."""
    p, a, lane, wave = c["key"], c["args"], c["lane"], c["wave"]
    j, q = task // p.slices, task % p.slices
    k = p.slice_steps * KSTEP
    rg0 = c["rank"] * p.groups + j * SETS
    ops = gemv_fp8_loads(
        c, a["wp"], a["wps"], k, rg0, split=1, tiled=True, k_row=K_PROJ,
        k0=q * k,
    )  # fmt: skip
    col = (j * SETS + wave) * RG + 4 * (lane // 16)
    for b in range_constexpr(p.blocks):
        n = min(RG, p.rows - RG * b)
        gpu.barrier()
        quant_taps(c, b, n, q)
        gpu.barrier()
        acc = _acc(c, k, ops, n)
        t = RG * b + lane % 16
        if lane % 16 < n:
            bo.buffer_store(acc, c["r_parts"], ((q * p.rows + t) * p.cols + col))
    publish(c["mb"].put, c["pf"], task, 1, c["tid"] == 0)


@traced
def reduce(c, task):
    """Block b x 128 columns: the slices' partials summed in slice order,
    rounded to bf16, pushed to every group rank's CX."""
    p, tid, lane = c["key"], c["tid"], c["lane"]
    per = p.cols // 128
    b, j = task // per, task % per
    if c["wave"] == 0:
        for k in range_constexpr(-(-p.slices // 64)):
            f = fx.min(lane + 64 * k, fx.Int32(p.slices - 1))
            c["mb"].poll([(c["pf"], j * p.slices + f, 1)])
    gpu.barrier()
    r, quad = tid // 32, tid % 32
    t = RG * b + r
    col = j * 128 + 4 * quad
    if t < p.rows:
        s = _slice_sum(
            c["r_parts"], p.slices, lambda q: (q * p.rows + t) * p.cols + col, 1
        )
        at = t * HIDDEN + c["rank"] * p.cols + col
        for dst in c["cx_dsts"]:
            c["mb"].put_bf(dst, at, s)


@traced
def mnorm(c, t):
    """Row t: main_norm -> bf16 -> MXFP8 into H8 / H8S, then HF."""
    p, a, tid, lane, wave = c["key"], c["args"], c["tid"], c["lane"], c["wave"]
    tt = fx.min(tid, fx.Int32(HIDDEN // 16 - 1))
    mine = tid < HIDDEN // 16
    c0 = 16 * tt
    w = ld_bf8(a["mnorm_w"], c0) + ld_bf8(a["mnorm_w"], c0 + 8)
    got = c["mb"].poll([(c["cx"], (t * HIDDEN + c0) // 2 + 2 * k, 2) for k in range(4)])
    vs = []
    for k in range_constexpr(4):
        for e in range_constexpr(2):
            vs += list(bf2_f32(got[k][e]))
    ss = fx.Float32(0.0)
    for v in vs:
        ss = ss + v * v
    ss = mine.select(ss, fx.Float32(0.0))
    ss = butterfly(ss, (1, 2, 4, 8, 16, 32))
    if lane == 0:
        fx.ptr_store(ss, c["red"] + wave)
    gpu.barrier()
    total = fx.ptr_load(c["red"])
    for k in range_constexpr(1, WAVES):
        total = total + fx.ptr_load(c["red"] + k)
    rstd = rms_rstd(total, HIDDEN, p.mnorm_eps)
    ys = [bf16_round((vs[i] * rstd) * w[i]) for i in range(16)]
    amax = fx.Float32(0.0)
    for y in ys:
        amax = fx.max(amax, abs(y))
    amax = fx.max(amax, xshfl(amax, 1))
    code = code_ceil(fx.max(amax, fx.Float32(AMAX_FLOOR)) * fx.Float32(1.0 / FP8_MAX))
    inv = rcp_pow2(code)
    words = [
        fp8_pack4(*[clamp_fp8(ys[4 * k + i] * inv) for i in range(4)]) for k in range(4)
    ]
    if mine:
        bo.buffer_store(
            fx.Vector.from_elements(words, fx.Int32), c["r_h8"], (t * HIDDEN + c0) // 4
        )
        if tt % 2 == 0:
            bo.buffer_store(
                fx.Int8(code), c["r_h8"], p.rows * HIDDEN + t * (HIDDEN // 32) + tt // 2
            )
    publish(c["mb"].put, c["hf"], t, 1, tid == 0)
    gpu.barrier()


@traced
def load_h8(c, b, n, q):
    """Rows 16 b .. + n of H8, K slice q of the wkv, into LDS."""
    p, tid = c["key"], c["tid"]
    words = WKV_SLICE_STEPS * KSTEP // 4
    codes = WKV_SLICE_STEPS * KSTEP // 32
    for it in range_constexpr(-(-RG * words // THREADS)):
        i = tid + it * THREADS
        r, wd = i // words, i % words
        if i < n * words:
            t = RG * b + r
            v = fx.Int32(
                bo.buffer_load(
                    c["r_h8"], (t * HIDDEN + q * WKV_SLICE_STEPS * KSTEP) // 4 + wd,
                    vec_width=1, dtype=T.i32,
                )
            )  # fmt: skip
            fx.ptr_store(v, c["xl"] + (r * c["kx_words"] + wd))
    if tid < n * codes:
        r, g = tid // codes, tid % codes
        t = RG * b + r
        byte = fx.Int8(
            bo.buffer_load(
                c["r_h8"],
                p.rows * HIDDEN + t * (HIDDEN // 32) + q * codes + g,
                vec_width=1, dtype=T.i8,
            )
        )  # fmt: skip
        fx.ptr_store(fx.Int32(byte) & 0xFF, c["xsl"] + (r * c["kx_codes"] + g))


@traced
def wkv(c, task):
    """wkv task ``task``: 8 row groups of the 1536 x K slice, every block.
    Its row groups are one stage's (512 rows: ``STAGE_SETS`` tasks a stage),
    read from that stage's own weight: the same bytes, in the same order, as
    the stages' rows concatenated."""
    p, a, lane, wave = c["key"], c["args"], c["lane"], c["wave"]
    j, q = task // p.wkv_slices, task % p.wkv_slices
    k = WKV_SLICE_STEPS * KSTEP
    stage = j // STAGE_SETS
    ops = gemv_fp8_loads(
        c, pick([a["wk0"], a["wk1"], a["wk2"]], stage),
        pick([a["wks0"], a["wks1"], a["wks2"]], stage), k,
        (j % STAGE_SETS) * SETS, split=1, tiled=True, k_row=HIDDEN, k0=q * k,
    )  # fmt: skip
    col = (j * SETS + wave) * RG + 4 * (lane // 16)
    for b in range_constexpr(p.blocks):
        n = min(RG, p.rows - RG * b)
        if wave == 0:
            f = fx.min(lane, fx.Int32(n - 1))
            c["mb"].poll([(c["hf"], RG * b + f, 1)])
        gpu.barrier()
        load_h8(c, b, n, q)
        gpu.barrier()
        acc = _acc(c, k, ops, n)
        t = RG * b + lane % 16
        if lane % 16 < n:
            bo.buffer_store(acc, c["r_kvp"], (q * p.rows + t) * KV_ROWS + col)
        gpu.barrier()
    publish(c["mb"].put, c["wf"], task, 1, c["tid"] == 0)


@traced
def kvrow(c, t):
    """Row t, a wave a stage: the stage's KV row into its fp8 window."""
    p, a, lane, s = c["key"], c["args"], c["lane"], c["wave"]
    if s < STAGES:
        cols = 8 * lane
        kvn = pick([a["kvn0"], a["kvn1"], a["kvn2"]], s)
        w = ld_bf8(kvn, cols)
        pos = ld_i32(a["pos"], t * a["pos_w"])
        pos = fx.min(fx.max(pos, fx.Int32(0)), a["max_pos"] - 1)
        jr = fx.max(cols - (KV_DIM - ROPE), fx.Int32(0)) // 2
        rot = cols >= KV_DIM - ROPE
        cs = [ld_bf(a["cos"], pos * HALF + jr + i) for i in range(4)]
        sn = [ld_bf(a["sin"], pos * HALF + jr + i) for i in range(4)]
        batch = ld_i32(a["batch"], t)
        layer = pick([a["layer0"], a["layer1"], a["layer2"]], s)
        per = c["key"].wkv_slices
        sets = KV_DIM // RG // SETS
        for k in range_constexpr(-(-sets * per // 64)):
            f = fx.min(lane + 64 * k, fx.Int32(sets * per - 1))
            c["mb"].poll([(c["wf"], (s * sets + f // per) * per + f % per, 1)])
        x = _slice_sum(
            c["r_kvp"], per, lambda q: (q * p.rows + t) * KV_ROWS + s * KV_DIM + cols, 2
        )
        x = [bf16_round(v) for v in x]
        ss = fx.Float32(0.0)
        for v in x:
            ss = ss + v * v
        ss = butterfly(ss, (1, 2, 4, 8, 16, 32))
        rstd = rms_rstd(ss, KV_DIM, p.kv_eps)
        kn = [bf16_round((x[i] * rstd) * w[i]) for i in range(8)]
        e = []
        for i in range_constexpr(4):
            k0, k1 = kn[2 * i], kn[2 * i + 1]
            e.append(rot.select(bf16_round(k0 * cs[i] - k1 * sn[i]), k0))
            e.append(rot.select(bf16_round(k0 * sn[i] + k1 * cs[i]), k1))
        amax = fx.Float32(0.0)
        for v in e:
            amax = fx.max(amax, abs(v))
        amax = butterfly(amax, (1, 2), fx.max)
        code = code_ceil(
            fx.max(amax, fx.Float32(AMAX_FLOOR)) * fx.Float32(1.0 / FP8_MAX)
        )
        inv = rcp_pow2(code)
        words = [
            fp8_pack4(*[clamp_fp8(e[4 * h + i] * inv) for i in range(4)])
            for h in range(2)
        ]
        ring = a["ring"]
        live = batch >= 0
        b = fx.max(batch, fx.Int32(0))
        live = live & (t >= ld_i32(a["cu"], b + 1) - ring)
        rp = ld_i32(a["ring_pos"], t * a["ring_pos_w"])
        live = live & (rp >= ld_i32(a["cached"], b * a["cached_w"]))
        if live:
            row = fx.Int64(ld_i32(a["slots"], b)) * ring + fx.Int64(rp % ring)
            values = load_ptr64(a["wptrs"], layer * 2)
            scales = load_ptr64(a["wptrs"], layer * 2 + 1)
            bo.buffer_store(
                fx.Vector.from_elements(words, fx.Int32),
                rsrc(values + row * KV_DIM),
                2 * lane,
            )
            if lane % 4 == 0:
                bo.buffer_store(
                    fx.Int8(code), rsrc(scales + row * (KV_DIM // 32)), lane // 4
                )


def build_draft_ctx(key: DraftCtxBuild):
    keyed = key_tuple(key, SOURCES)
    regions = {r: off for r, (off, _) in scratch_layout(key.rows, key.tp).items()}
    cx_off = peer_offsets(key.rows, Dims(key.tp))["cx"]
    name = kernel_symbol("v41_draft_ctx", s=key.rows, tp=key.tp)
    wkv_words = lds_row(WKV_SLICE_STEPS * KSTEP // 4)
    wkv_codes = lds_row(WKV_SLICE_STEPS * KSTEP // 32)
    assert RG * (wkv_words + wkv_codes) <= RG * (key.x_words + key.x_codes)

    @fx.struct
    class Smem:
        words: fx.Array[fx.Int32, key.lds_words, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def draft_ctx(
        tap0: Int64, tap1: Int64, tap2: Int64, wp: Int64, wps: Int64, wk0: Int64,
        wk1: Int64, wk2: Int64, wks0: Int64, wks1: Int64, wks2: Int64,
        mnorm_w: Int64, kvn0: Int64, kvn1: Int64, kvn2: Int64,
        pos: Int64, pos_w: Int32, ring_pos: Int64, ring_pos_w: Int32,
        batch: Int64, cu: Int64, slots: Int64, cached: Int64, cached_w: Int32,
        wptrs: Int64, layer0: Int32, layer1: Int32, layer2: Int32, ring: Int32,
        cos: Int64, sin: Int64, max_pos: Int32, parts: Int64, h8: Int64,
        kvp: Int64, scratch: Int64, sym: Int64, peers: Int64, rank: Int32,
    ):  # fmt: skip
        tid = fx.thread_idx.x
        smem = fx.SharedAllocator().allocate(Smem).peek()
        lds = smem.words.ptr
        xsl = lds + RG * key.x_words
        if const_expr(key.tp == 1):
            dsts = [preg(sym, cx_off, "cx")]
        else:
            bases = peer_bases(peers, key.tp)
            dsts = [preg(bases[r], cx_off, "cx") for r in range(key.tp)]
        c = {
            "key": key, "tid": tid, "lane": tid % 64, "wave": tid // 64,
            "rank": rank if key.tp > 1 else fx.Int32(0),
            "xl": lds, "xsl": xsl,
            "red": fx.recast_iter(fx.Float32, lds + RG * (key.x_words + key.x_codes)),
            "args": {
                "tap0": tap0, "tap1": tap1, "tap2": tap2, "wp": wp, "wps": wps,
                "wk0": wk0, "wk1": wk1, "wk2": wk2, "wks0": wks0,
                "wks1": wks1, "wks2": wks2, "mnorm_w": mnorm_w, "kvn0": kvn0,
                "kvn1": kvn1, "kvn2": kvn2, "pos": pos, "pos_w": pos_w,
                "ring_pos": ring_pos, "ring_pos_w": ring_pos_w, "batch": batch,
                "cu": cu, "slots": slots, "cached": cached, "cached_w": cached_w,
                "wptrs": wptrs, "layer0": layer0, "layer1": layer1,
                "layer2": layer2, "ring": ring, "cos": cos, "sin": sin,
                "max_pos": max_pos,
            },
            "r_parts": rsrc(parts), "r_h8": rsrc(h8), "r_kvp": rsrc(kvp),
            "mb": Mailbox(CTX_TAG - 1),
            "pf": sreg(scratch, regions["pf"], "pf"),
            "hf": sreg(scratch, regions["hf"], "hf"),
            "wf": sreg(scratch, regions["wf"], "wf"),
            "cx": preg(sym, cx_off, "cx"), "cx_dsts": dsts,
            "kx_words": wkv_words, "kx_codes": wkv_codes,
        }  # fmt: skip
        cta = fx.block_idx.x
        for k in range_constexpr(-(-key.proj_tasks // BLOCKS)):
            if cta + k * BLOCKS < key.proj_tasks:
                proj(c, cta + k * BLOCKS)
        for k in range_constexpr(-(-key.reduce_tasks // BLOCKS)):
            if cta + k * BLOCKS < key.reduce_tasks:
                reduce(c, cta + k * BLOCKS)
        for k in range_constexpr(-(-key.rows // BLOCKS)):
            if cta + k * BLOCKS < key.rows:
                mnorm(c, cta + k * BLOCKS)
        for k in range_constexpr(-(-key.wkv_tasks // BLOCKS)):
            if cta + k * BLOCKS < key.wkv_tasks:
                wkv(c, cta + k * BLOCKS)
        for k in range_constexpr(-(-key.rows // BLOCKS)):
            if cta + k * BLOCKS < key.rows:
                kvrow(c, cta + k * BLOCKS)
        _ = keyed

    @launcher(name)
    def launch(
        tap0: Int64, tap1: Int64, tap2: Int64, wp: Int64, wps: Int64, wk0: Int64,
        wk1: Int64, wk2: Int64, wks0: Int64, wks1: Int64, wks2: Int64,
        mnorm_w: Int64, kvn0: Int64, kvn1: Int64, kvn2: Int64,
        pos: Int64, pos_w: Int32, ring_pos: Int64, ring_pos_w: Int32,
        batch: Int64, cu: Int64, slots: Int64, cached: Int64, cached_w: Int32,
        wptrs: Int64, layer0: Int32, layer1: Int32, layer2: Int32, ring: Int32,
        cos: Int64, sin: Int64, max_pos: Int32, parts: Int64, h8: Int64,
        kvp: Int64, scratch: Int64, sym: Int64, peers: Int64, rank: Int32,
        stream: fx.Stream = _CURRENT_STREAM,
    ):  # fmt: skip
        _ = keyed
        draft_ctx(
            tap0, tap1, tap2, wp, wps, wk0, wk1, wk2, wks0, wks1, wks2, mnorm_w,
            kvn0, kvn1, kvn2, pos,
            pos_w, ring_pos, ring_pos_w, batch, cu, slots, cached, cached_w,
            wptrs, layer0, layer1, layer2, ring, cos, sin, max_pos, parts, h8,
            kvp, scratch, sym, peers, rank,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    ABI.check(draft_ctx, launch)
    return launch

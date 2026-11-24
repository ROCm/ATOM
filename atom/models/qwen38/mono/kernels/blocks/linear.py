# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``Linear``: y = x @ W^T for every row of the step, fp32 accumulation, as one stage.

The weight is the given HF linears stacked along N in order (e.g. q, k, v), packed by ``pack`` into the pointer-table
args ``<name>.w`` / ``<name>.ws`` (``gemv.pack_weight``: 16-row groups x 128-K units, aiter's 16x16 shuffle) and,
for FP8 rows, ``<name>.rs``:

- ``weight_format="mxfp8"`` with ``row_scale``: per-output-channel FP8 weights (Quark's ``fp8_e4m3 per_channel``):
  the E4M3 bytes packed as MXFP8 with every block code 127 (scale 1, exact) against an MXFP8 channel input, and the
  sums times the row's fp32 scale after every sum (K waves, K slices), before the output's rounding;
- ``weight_format="bf16"``: bf16 weights (``gemv16.pack_bf16_mmajor``'s units, 64 B a lane) against a BF16 channel
  input, four bf16 MFMAs a unit: a BF16 GEMM's exact products, fp32 sums (``.ws`` is a 16-byte placeholder).

**Outputs** (``out=``): ``BF16Out`` (a BF16 channel of width N, RNE) or ``HCWrite`` (the hyper-connection write of
the layer's residual streams, ``hc.write_store``).

**Tasks.** A task owns ``g`` row groups (16 output rows each) and one of ``k_split`` K slices of the input; its 8
waves split as ``wr`` row-group waves x ``wk`` K waves (K steps interleaved over the K waves). Each wave issues its
weight units first, ``prefetch`` deep (``lead`` of them before the input's first poll attempt, the rest after it;
``pre_units`` of them before the input has landed when given), then per K step the LDS operands and one scaled MFMA
per row group. The K waves' partials are summed through LDS (wave order); with K slices every slice publishes its
partials and the last to arrive (an atomic counter a row block) sums them in slice order and writes the output. The
task shape is frozen per scenario in the schedules (``shape``).
"""

from dataclasses import dataclass
from functools import partial

import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from atom.models.qwen38.mono.kernels.blocks.base import Block, Channel, Fmt
from atom.models.qwen38.mono.kernels.blocks.hc import HCWrite, write_issue, write_store
from atom.models.qwen38.mono.kernels.device import buffer_ops as bo
from atom.models.qwen38.mono.kernels.device import channels as chan
from atom.models.qwen38.mono.kernels.device import gemv, gemv16
from atom.models.qwen38.mono.kernels.device import mailbox as mb
from atom.models.qwen38.mono.kernels.device.ops import rsrc, traced
from atom.models.qwen38.mono.kernels.program import PlanError, Region, Scenario, Stage
from atom.models.qwen38.mono.kernels.target import MI355X
from atom.mono.plan.trace import note

WAVES = MI355X.waves
THREADS = MI355X.threads
VGPR_BUDGET = 256  # 512 threads, one CTA per CU: two waves per SIMD
LEAD_DIV = 3  # the input's first poll attempt follows 1 / LEAD_DIV of the prefetch
ZERO_CODE = 127  # E8M0 code of 1.0: every block code of row-scaled FP8 weights
# bf16 weights in ``gemv16.pack_bf16_mmajor``'s units
BF16W = gemv.WeightFormat("bf16", -1, 16)


def wformat(name: str) -> gemv.WeightFormat:
    return BF16W if name == "bf16" else gemv.FORMATS[name]


@dataclass(frozen=True)
class BF16Out:
    channel: Channel


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
    kind: str  # "bf16" | "hcwrite"
    x: str  # input channel name
    out_ch: str  # output channel name ("" if none)
    part: str  # exchange region of K-slice partials ("" if k_split == 1)
    ctr: str  # per row block arrival counters of the K slices ("" if k_split == 1)
    wfmt: str  # "mxfp8" | "bf16"
    # weight units a wave issues before its input has landed (0: the whole prefetch)
    pre_units: int = 0
    hcw: HCWrite | None = None  # kind "hcwrite": the combine's spec
    # the f32 per-output-row weight scale arg [N] (packed row order) ("": none)
    row_scale: str = ""

    @property
    def a16(self) -> bool:  # bf16 weights and input
        return self.wfmt == "bf16"

    @property
    def x16(
        self,
    ) -> (
        bool
    ):  # (``_x_batches``' name for a BF16 channel input with other weights: never)
        return False

    @property
    def xk(self):  # the input channel's width
        return self.k

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
    def pitch(
        self,
    ):  # LDS words per activation row (+4: bank spread; a bf16 input: two values a word)
        return self.kse // 2 + 4 if self.a16 else self.kse // 4 + 4

    @property
    def spitch(self):  # LDS words per row of scale codes (odd: bank spread)
        return self.kse // 32 + 1

    @property
    def xs_base(self):  # the codes, after the words
        return self.rows * self.pitch

    @property
    def flag_word(self):  # LDS word: "this task is its row block's last K slice"
        return max(
            self.rows * (self.pitch + self.spitch), self.wk * self.g * self.rows * 16
        )

    @property
    def lds_words(self):
        return -(-(-(-(self.flag_word + 1) // 4) * 4) // 4) * 4  # 16 B multiples

    @property
    def out_quads(self):  # epilogue items: (row, row group, 4 features)
        return self.rows * self.g * 4

    @property
    def w_bytes(self):
        return self.fmt.weight_bytes(self.n, self.k)

    @property
    def s_bytes(self):
        # (bf16 weights: a placeholder scales arg)
        return 16 if self.a16 else self.n * self.k // 32


class Linear(Block):
    """One projection (module docstring)."""

    def __init__(
        self,
        name: str,
        linears: tuple[str, ...],
        x: Channel,
        n: int,
        out,
        shape: tuple[int, int, int, int] | None = None,
        weight_format: str = "mxfp8",
        pre_units: int | None = None,
        row_scale: bool = False,
    ):
        if weight_format not in ("mxfp8", "bf16"):
            raise ValueError(
                f"Linear {name}: weight format {weight_format!r} not in ('mxfp8', 'bf16')"
            )
        want = Fmt.BF16 if weight_format == "bf16" else Fmt.MXFP8
        if not isinstance(x, Channel) or x.fmt is not want:
            raise ValueError(
                f"Linear {name}: {weight_format} weights take a tagged {want.name} channel input"
            )
        if row_scale != (weight_format == "mxfp8"):
            raise ValueError(
                f"Linear {name}: FP8 weights are row-scaled (E4M3 rows, codes 127), bf16 weights are not"
            )
        if not isinstance(out, (BF16Out, HCWrite)):
            raise TypeError(f"Linear {name}: the output is a BF16Out or an HCWrite")
        if isinstance(out, HCWrite):
            out.check(n)
        self.name, self.linears, self.x, self.n, self.k, self.out = (
            name,
            tuple(linears),
            x,
            n,
            x.width,
            out,
        )
        if self.k % 128 or n % 16:
            raise ValueError(
                f"Linear {name}: K % 128 and N % 16 must be 0 (N {n}, K {self.k})"
            )
        self.weight_format, self.row_scale, self.pre_units = (
            weight_format,
            row_scale,
            pre_units,
        )
        # the task shape (row groups per task, K slices, wave rows, prefetch depth): frozen in the schedules
        self.shape = shape
        if shape is not None:
            g, kt, wr, pre = shape
            if WAVES % wr or g % wr or (self.k // 128) % kt or pre < 1:
                raise ValueError(
                    f"Linear {name}: task shape {shape} does not fit N {n} / K {self.k}"
                )

    @property
    def weight_args(self) -> tuple[str, str]:
        return f"{self.name}.w", f"{self.name}.ws"

    def _out_channel(self) -> Channel | None:
        return self.out.publish if isinstance(self.out, HCWrite) else self.out.channel

    def plan(self, sc: Scenario) -> _Plan:
        """The stage's plan for ``sc`` from the frozen task shape (``shape``; ``schedules.KNOBS``)."""
        if self.shape is None:
            raise PlanError(f"Linear {self.name}: no frozen task shape at {sc.name}")
        if self.weight_format == "bf16" and sc.rows > 16:
            raise PlanError(
                f"Linear {self.name}: a bf16 input up to 16 rows (one MFMA chunk), got {sc.rows}"
            )
        g, kt, wr, pre = self.shape
        hcw = isinstance(self.out, HCWrite)
        out_ch = self._out_channel()
        p = _Plan(
            name=self.name,
            rows=sc.rows,
            n=self.n,
            k=self.k,
            g=g,
            k_split=kt,
            wr=wr,
            wk=WAVES // wr,
            prefetch=pre,
            lead=max(1, pre // LEAD_DIV),
            kind="hcwrite" if hcw else "bf16",
            x=self.x.name,
            out_ch=out_ch.name if out_ch is not None else "",
            part=f"{self.name}.part" if kt > 1 else "",
            ctr=f"{self.name}.ctr" if kt > 1 else "",
            wfmt=self.weight_format,
            pre_units=self.pre_units or 0,
            hcw=self.out if hcw else None,
            row_scale=f"{self.name}.rs" if self.row_scale else "",
        )
        if p.lds_words * 4 > MI355X.lds_bytes - 1024:
            raise PlanError(
                f"Linear {self.name}: task shape {self.shape} exceeds the LDS at {sc.rows} rows"
            )
        return p

    def stages(self, sc: Scenario) -> list[Stage]:
        p = self.plan(sc)
        writes = []
        ch = self._out_channel()
        if ch is not None:
            writes += list(ch.regions(p.rows, (self.name,)))
        if p.part:
            writes.append(
                Region(p.part, p.k_split * p.rows * p.n, (self.name,), exchange=True)
            )
            # atomics: one copy
            writes.append(
                Region(p.ctr, p.blocks, (self.name,), exchange=True, single=True)
            )
        args = list(self.weight_args)
        if p.hcw is not None:
            args += [a for a in p.hcw.args() if a not in args]
        if p.row_scale:
            args.append(p.row_scale)
        return [Stage(self.name, p.tasks, partial(_linear, p=p), lds_words=p.lds_words, writes=tuple(writes),
                      args=tuple(args), key=(p,))]  # fmt: skip

    def weight_specs(self) -> tuple:
        from atom.models.qwen38.mono.kernels.weights.source import U8, WeightSpec

        w, ws = self.weight_args
        if (
            self.weight_format == "bf16"
        ):  # ``_pack_bf16``: the units, and a placeholder for the scales arg
            return (
                WeightSpec(
                    w, "mono.bf16.gemv16m", (self.n * self.k,), (torch.bfloat16,)
                ),
                WeightSpec(ws, "bf16", (16,), U8),
            )
        return (
            WeightSpec(w, "mono.w8.gemv", (self.n * gemv.MXFP8.stored_k(self.k),), U8),
            WeightSpec(ws, "mono.e8m0.gemv", (self.n * self.k // 32,), U8),
            WeightSpec(f"{self.name}.rs", "f32", (self.n,), (torch.float32,), align=4),
        )

    def pack(self, layer) -> dict[str, torch.Tensor]:
        if self.weight_format == "bf16":
            return self._pack_bf16(layer)
        fmt = gemv.MXFP8
        mats = [layer.linears[n] for n in self.linears]
        for n, m in zip(self.linears, mats, strict=True):
            if getattr(m, "fmt", "mxfp4") != fmt.name:
                raise ValueError(
                    f"Linear {self.name}: {n} is {getattr(m, 'fmt', 'mxfp4')}, the block is {fmt.name}"
                )
        # bytes: FP8 tensors stack as uint8
        packed = torch.cat([m.packed.view(torch.uint8) for m in mats])
        codes = torch.cat([m.codes for m in mats])
        if packed.shape != (self.n, fmt.stored_k(self.k)):
            raise ValueError(
                f"Linear {self.name}: stacked weight {tuple(packed.shape)} != ({self.n}, {fmt.stored_k(self.k)})"
            )
        w, s = gemv.pack_weight(packed, codes, fmt)
        out = dict(zip(self.weight_args, (w, s), strict=True))
        rs = [getattr(m, "row_scale", None) for m in mats]
        if any(r is None for r in rs):
            raise ValueError(
                f"Linear {self.name}: row scales on {sum(r is not None for r in rs)} of {len(rs)} linears"
            )
        v = torch.cat([r.reshape(-1).float() for r in rs])
        if v.shape != (self.n,):
            raise ValueError(
                f"Linear {self.name}: row scales {tuple(v.shape)} != ({self.n},)"
            )
        out[f"{self.name}.rs"] = v.contiguous()
        return out

    def _pack_bf16(self, layer) -> dict[str, torch.Tensor]:
        """bf16 weights: ``layer.linears[name]`` bf16 [N_i, K] tensors stacked along N into
        ``gemv16.pack_bf16_mmajor``'s units; ``.ws`` a placeholder."""
        mats = [layer.linears[n] for n in self.linears]
        for n, m in zip(self.linears, mats, strict=True):
            if not isinstance(m, torch.Tensor) or m.dtype != torch.bfloat16:
                raise ValueError(
                    f"Linear {self.name}: {n} is not a bf16 tensor (weight_format 'bf16')"
                )
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
    """K step (within the slice) of K wave ``wk``'s i-th step: interleaved, so the CTA's waves read adjacent units."""
    return wk + p.wk * i


def _issue(p, r_w, rg0, st0, wk, lane, u):
    """Wave unit ``u`` (K step outer, row group inner): its weight load."""
    i, gl = u // p.gw, u % p.gw
    unit = (rg0 + gl) * p.steps + st0 + _kstep(p, wk, i)
    valid = None
    if p.ks % p.wk != 0:
        valid = _kstep(p, wk, i) < p.ks
    if (
        p.a16
    ):  # 64 B a lane: four 16 B loads, each instruction a contiguous 1 KiB (``gemv16.pack_bf16_mmajor``)
        return gemv16.load_bf16_mmajor(r_w, unit, lane, valid)
    return gemv.load_weight(r_w, unit, lane, valid, p.fmt)


def _unit_scale(k, p, gt, st):
    """Lane's weight scale code of task row group ``gt``, slice K step ``st``: row-scaled FP8, the constant 1.0."""
    return fx.Int32(ZERO_CODE)


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


@traced
def _store_bf16(k, p, ok, e, vals):
    if ok:
        chan.put_bf16(k.mb[p.out_ch], e, vals, k.tag)


def _emit_out(k, p, ok, n, col, lead, vals, res_words):
    """The output of one item's totals (``vals``: one list of 4)."""
    if p.row_scale:  # FP8 per-channel weights: acc x the row's scale, after every sum
        res_words, rsc = res_words
        vals = [
            [v * rsc[4 * h + q] for q, v in enumerate(vh)] for h, vh in enumerate(vals)
        ]
    if p.kind == "bf16":
        _store_bf16(k, p, ok, n * p.n + col, vals[0])
    else:
        write_store(k, p.hcw, p.n, ok, n, col, vals[0], res_words)


def _finish(k, p, s, b, res_words):
    """The epilogue after the K waves' partials are in LDS. With K slices, every slice publishes its partial and
    arrives; the last to arrive sums all slices (slice order, so the result does not depend on who is last).
    """
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
            _put_partial(k, p, ok, s * (p.rows * p.n) + idx, _sum_red(k, p, w))
    # no wait for the partial stores before arriving: the last arrival's polls check tags
    _arrive(k, p, b)
    gpu.barrier()
    _last_slice(k, p, b, fx.ptr_load(k.lds + p.flag_word) != 0, res_words)


def _finish_last(k, p, b, res_words):
    for ps in range(-(-p.out_quads // THREADS)):
        ok, n, srcs, col, lead = _items(p, b, fx.Int32(ps * THREADS) + k.tid)
        _emit_out(
            k,
            p,
            ok,
            n,
            col,
            lead,
            [_poll_slices(k, p, ok, idx) for _, idx in srcs],
            res_words[ps],
        )


def _res_prefetch(k, p, b):
    """Issued with the weights, per epilogue pass: HCWrite's residual words and injection attempt
    (``_hc_prefetch``); with ``row_scale`` each pass's entry is (those words, the row scales: ``_rs_prefetch``).
    """
    passes = -(-p.out_quads // THREADS)
    out = (
        [_hc_prefetch(k, p, b, ps) for ps in range(passes)]
        if p.kind == "hcwrite"
        else [[] for _ in range(passes)]
    )
    if p.row_scale:
        out = [(o, _rs_prefetch(k, p, b, ps)) for ps, o in enumerate(out)]
    return out


def _rs_prefetch(k, p, b, ps):
    """``row_scale``: the f32 scales of pass ``ps``'s four output rows (packed order: ``_items``' columns)."""
    e = fx.min(fx.Int32(ps * THREADS) + k.tid, fx.Int32(p.out_quads - 1))
    f4 = e % 4
    r = rsrc(k.args[p.row_scale])
    rows = [fx.min(b * p.g + (e // 4) % p.g, fx.Int32(p.n_rg - 1))]
    out = []
    for rg in rows:
        v = fx.Vector(bo.buffer_load(r, rg * 16 + f4 * 4, vec_width=4, dtype=T.f32))
        out += [fx.Float32(v[q]) for q in range(4)]
    return out


def _linear(k, t, p):
    """One task: ``t`` = K slice * blocks + row block."""
    s = t // p.blocks if p.k_split > 1 else fx.Int32(0)
    b = t % p.blocks if p.k_split > 1 else t
    lane = k.lane
    wr, wk = k.wave // p.wk, k.wave % p.wk
    r_w = rsrc(k.args[f"{p.name}.w"], p.w_bytes)
    # (never read: kept in its place in the build's IR order)
    _ = rsrc(k.args[f"{p.name}.ws"], p.s_bytes)
    rg0 = b * p.g + wr * p.gw
    st0 = s * p.ks
    xoff = 0
    units = p.c * p.gw
    issue = partial(_issue, p, r_w, rg0, st0, wk, lane)

    # 1. weights first, then the residual, then the input. The input's first poll attempt goes out after the first
    # ``p.lead`` units, the rest of the prefetch after it: vmcnt is in order, so the attempt lands (and, if the input
    # is there, the MFMAs start) while most weights stream.
    n_pre = min(p.prefetch, units)
    lead = min(p.lead, n_pre)
    n_first = max(lead, min(p.pre_units, n_pre)) if p.pre_units else n_pre
    rocdl.sched_barrier(0)
    q = [issue(u) for u in range(lead)]
    res_words = _res_prefetch(k, p, b)
    batches = _x_batches(k, p, s, xoff=xoff)
    # keep the issue order: LLVM would sink the attempt below the weights
    rocdl.sched_barrier(0)
    tries = [_try_batch(specs) for specs, _ in batches]
    rocdl.sched_barrier(0)
    q += [issue(u) for u in range(lead, n_first)]
    for (specs, dests), words in zip(batches[:-1], tries[:-1], strict=True):
        _land_batch(k, p, specs, dests, words)
    _land_batch(k, p, *batches[-1], tries[-1])
    gpu.barrier()
    if (
        n_first < n_pre
    ):  # the rest of the prefetch, once the input has landed (``pre_units``)
        rocdl.sched_barrier(0)
        q += [issue(u) for u in range(n_first, n_pre)]
        rocdl.sched_barrier(0)

    # 2. stream, in this order per K step (sched barriers keep it: LLVM would sink each load next to its MFMA and
    # collapse the prefetch to one unit): the next K step's LDS operands, then per row group the load ``prefetch``
    # units ahead and this unit's MFMA.
    row = fx.min(lane % 16, fx.Int32(p.rows - 1))
    j = lane // 16

    def operands(i):
        st = _kstep(p, wk, i)
        if p.ks % p.wk != 0:
            st = fx.min(st, fx.Int32(p.ks - 1))
        if p.a16:  # bf16 input words: the unit's four B operands
            return gemv16.b_operands(k.lds, row, j, st, p.pitch), None, None
        x, xs = gemv.b_operand(k.lds, k.lds + p.xs_base, row, j, st, p.pitch, p.spitch)
        return x, xs, [_unit_scale(k, p, wr * p.gw + gl, st) for gl in range(p.gw)]

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
            if p.a16:
                acc[gl] = gemv16.mfma_unit(acc[gl], gemv16.bf16_operands(q[u]), x)
            else:
                acc[gl] = gemv.mfma_wxa8(acc[gl], q[u], ws[gl], x, xs, p.fmt)
        rocdl.sched_barrier(0)
        cur = nxt

    # 3. K waves' partials through LDS (over the activation), then the epilogue
    gpu.barrier()
    _red_store(k, p, wr, wk, acc)
    gpu.barrier()
    _finish(k, p, s, b, res_words)
    gpu.barrier()  # the next task (or stage) reuses the LDS


# ------------------------------------------------------------------ shared with the MoE stages


def _x_batches(k, p, s, rmap=None, xoff=0):
    """The input poll of K slice ``s`` (every row) in batches of at most POLL_MAX
    pair specs: [(specs, LDS word offsets)]. FP8 words go to LDS word 0 on (row
    pitch ``p.pitch``), codes to ``p.xs_base`` on (row pitch ``p.spitch``).
    ``rmap``: LDS row -> the channel row it is polled from (default: the same)."""
    rmap = rmap or (lambda r: r)
    if getattr(p, "a16", False) or getattr(
        p, "x16", False
    ):  # a BF16 channel: two values a word, no codes
        return _x_batches16(k, p, s, rmap, xoff)
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


def _x_batches16(k, p, s, rmap, xoff):
    """``_x_batches`` of a BF16 channel input: the slice's bf16 words (2-pair specs) to LDS row ``r * pitch``."""
    words_row = p.kse // 2
    specs_row = words_row // 2
    nq = p.rows * specs_row
    a = k.mb[p.x]
    items = []
    for c in range(-(-nq // THREADS)):
        idx = fx.min(fx.Int32(c * THREADS) + k.tid, fx.Int32(nq - 1))
        r, w2 = idx // specs_row, idx % specs_row
        items.append(
            (
                (a, rmap(r) * (p.xk // 2) + xoff // 2 + s * words_row + w2 * 2, 2),
                r * p.pitch + w2 * 2,
            )
        )
    return [
        (
            [sp for sp, _ in items[b0 : b0 + mb.POLL_MAX]],
            [d for _, d in items[b0 : b0 + mb.POLL_MAX]],
        )
        for b0 in range(0, len(items), mb.POLL_MAX)
    ]


def _try_batch(specs):
    """One poll attempt of a batch, issued now, not waited for: the raw (value, tag)
    words of every spec (device scope, as ``mailbox.poll``)."""
    out = []
    for a, i, n in specs:
        note(a.region, a.space, "poll")
        v = fx.Vector(mb.load_pairs(a, i, 2 * n))
        out.append([v[e] for e in range(2 * n)])
    return out


@traced
def _put_partial(k, p, ok, idx, vals):
    if ok:
        mb.put2(k.mb[p.part], idx, vals[0], vals[1], k.tag)
        mb.put2(k.mb[p.part], idx + 2, vals[2], vals[3], k.tag)


@traced
def _store_mxfp8(k, p, ok, e, word, lead, code, width):
    """FP8 word of elements e .. e + 3 (row-major, row width ``width``); ``lead``
    lanes also store their group's scale code."""
    if ok:
        chan.put_mxfp8_word(k.mb[f"{p.out_ch}.q"], e // 4, word, k.tag)
        if lead:
            chan.put_mxfp8_scale(k.mb[f"{p.out_ch}.s"], e // 32, code, k.tag)


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
    if p.kind == "swiglu":
        gp = p.g // 2
        n, i = e // (gp * 4), (e // 4) % gp
        ok = valid_e & (b * gp + i < p.n_rg // 2)
        srcs = [
            (
                ((2 * i + h) * p.rows + n) * 16 + f4 * 4,
                n * p.n + (b * p.g + 2 * i + h) * 16 + f4 * 4,
            )
            for h in (0, 1)
        ]
        return ok, n, srcs, (b * gp + i) * 16 + f4 * 4, (i % 2 == 0) & (f4 == 0)
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
def _last_slice(k, p, b, last, res_words):
    if last:
        _finish_last(k, p, b, res_words)


def _quant_out(k, p, ok, e, vals, lead, width):
    """MXFP8 of four values per lane, groups of 32 over 8 adjacent lanes."""
    amax = gemv.group_amax(vals, 8)
    code = gemv.activation_code(amax)
    word = gemv.quant_word(vals, code)
    _store_mxfp8(k, p, ok, e, word, lead, code, width)


def _hc_prefetch(k, p, b, ps):
    """HCWrite: pass ``ps``'s residual words of every stream and an attempt at the row's
    injection multipliers (``hc.write_issue``), issued with the weights."""
    e = fx.min(fx.Int32(ps * THREADS) + k.tid, fx.Int32(p.out_quads - 1))
    n, gl, f4 = e // (p.g * 4), (e // 4) % p.g, e % 4
    rg = fx.min(b * p.g + gl, fx.Int32(p.n_rg - 1))
    return write_issue(k, p.hcw, p.n, n, rg * 16 + f4 * 4)


@traced
def _red_store(k, p, wr, wk, acc, chunk=0):
    """This wave's partials -> LDS [wk][row group][row][16 features] (useful lanes only); ``chunk``: the 16-row chunk
    ``acc`` holds (rows 16 chunk ..)."""
    n = k.lane % 16 + 16 * chunk if chunk else k.lane % 16
    if n < p.rows:
        for gl in range_constexpr(p.gw):
            off = ((wk * p.g + wr * p.gw + gl) * p.rows + n) * 16 + (k.lane // 16) * 4
            fx.ptr_store(acc[gl], k.ldsf + off)

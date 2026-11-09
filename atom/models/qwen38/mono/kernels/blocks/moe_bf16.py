# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""BF16 routed + shared experts for one-row steps: the Qwen3.8-Flash MTP drafter's MLP, which ATOM keeps in bf16.

The routing comes from a ``Router`` with its select stage (``ids`` I32 [rows, top_k], ``wts`` F32 [rows, top_k +
1]: the routed ranks' softmax weights, then the shared expert's ``sigmoid`` gate). The slots of the row are its
``top_k`` routed ranks, then the shared expert, stored as expert ``experts`` of the packed tensors.

- ``GateUpBF16`` (``<name>``): ``act[r, s, :] = bf16(w_s * silu(g) * u)`` with ``g``, ``u`` the fp32 sums of slot
  ``s``'s expert's gate / up rows against the row's bf16 ``x`` (the MLP HC's BF16 channel), every product exact
  (``mfma 16x16x32 bf16``), the wave's chain then the waves in order. Task = (slot, ``g`` 16-row groups of the
  expert's interleaved gate / up rows); the expert's weights come through a buffer resource at its own base.
  The routing weight is applied here (fp32, one bf16 rounding of the product), where ATOM's bf16 MoE applies it
  to the down's fp32 output: another rounding point (the drafter's arithmetic, not the target's).
- ``DownBF16`` (``<name>``): ``y = sum over slots of W2_e(s) act[s]`` as one K of ``slots x inter`` (each K step's
  expert looked up in LDS, 64-bit addresses), the waves' sums in wave order, then the hyper-connection combine
  (``HCWrite``: ``out = bf16(res + bf16(y) * inj)`` into every stream).

Weights (``pack``): the bf16 [N, K] rows in ``pack_bf16``'s units (16 rows x 32 K, 1 KiB), expert-major:
gate_up ``[(E + 1) * 2 I / 16][H / 32][64][8]`` (per expert gate / up interleaved per 16 rows), down ``[(E + 1) *
H / 16][I / 32][64][8]``. Rows (at most 16, one request each: the drafter's selected rows): every task serves one
row (``row = t // tasks a row``), its slots' experts; the MFMA's B operand carries that row alone (the others' x
words zero), so a row's sums are the one-row program's.
"""

from __future__ import annotations

from dataclasses import astuple, dataclass
from functools import partial

import flydsl.expr as fx
import torch
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from atom.models.qwen38.mono.kernels.blocks import hc as hcw
from atom.models.qwen38.mono.kernels.blocks.base import Block, Channel, Fmt
from atom.models.qwen38.mono.kernels.blocks.hc import _spin_from
from atom.models.qwen38.mono.kernels.blocks.moe import _store_if
from atom.models.qwen38.mono.kernels.device import buffer_ops as bo
from atom.models.qwen38.mono.kernels.device import channels as chan
from atom.models.qwen38.mono.kernels.device import mailbox as mb
from atom.models.qwen38.mono.kernels.device.ops import (
    CM_NT,
    global_load,
    mfma_bf16,
    rsrc,
    traced,
    uniform,
)
from atom.models.qwen38.mono.kernels.program import Scenario, Stage
from atom.models.qwen38.mono.kernels.target import MI355X

THREADS, WAVES = MI355X.threads, MI355X.waves
PRE = 16  # weight units a wave keeps in flight
XG = 3  # input poll batches in flight
XP = 4  # LDS pad words a row
XB = 6  # x pieces a thread polls a pass (a batch of 12 pairs)


def pack_bf16(w: torch.Tensor) -> torch.Tensor:
    """bf16 [N, K] -> the A-operand order [N / 16][K / 32][64 lanes][8] (lane 16 kg + r: row 16 g + r, K 32 st + 8 kg
    ..), contiguous."""
    n, k = w.shape
    t = w.reshape(n // 16, 16, k // 32, 4, 8)  # [g][r][st][kg][e]
    return t.permute(0, 2, 3, 1, 4).contiguous()  # [g][st][kg][r][e]: lane = 16 kg + r


def _load_batch(specs):
    """A poll batch's words (value, tag pairs) issued now (``mailbox.poll``'s loads)."""
    words = []
    for a, i, n in specs:
        w = fx.Vector(mb.load_pairs(a, i, 2 * n, mb._SCOPE_CM[a.space]))
        words += [w[e] for e in range(2 * n)]
    return fx.Vector.from_elements(words, fx.Int32)


def _pending(tag, v):
    bad = v[1] != tag
    for e in range(3, v.shape[0] if hasattr(v, "shape") else 4 * XB, 2):
        bad = bad | (v[e] != tag)
    return bad


@dataclass(frozen=True)
class _P:
    name: str
    x: str  # the input channel (gate_up: x [rows, H]; down: act [rows, slots x I])
    ids: str
    wts: str  # (gate_up only)
    out: str  # gate_up: the act channel ("" for the down)
    rows: int
    k: int  # K of the task's GEMV (gate_up: H; down: slots x I)
    n_e: int  # output rows an expert (gate_up: 2 I; down: H)
    ksteps_e: int  # 32-column K steps an expert (gate_up: H / 32; down: I / 32)
    top_k: int
    experts: int
    g: int
    inter: int

    @property
    def slots(self) -> int:
        return self.top_k + 1

    @property
    def gpe(self) -> int:  # 16-row groups an expert
        return self.n_e // 16

    @property
    def unit_bytes(self) -> int:  # an expert's packed bytes
        return self.gpe * self.ksteps_e * 1024

    @property
    def xpitch(self) -> int:
        return self.k // 2 + XP

    @property
    def a_red(self) -> int:
        return self.rows * self.xpitch

    @property
    def a_ids(self) -> int:  # (down) the slots' expert ids
        return self.a_red + WAVES * self.g * 64 * 4

    @property
    def lds(self) -> int:
        return self.a_ids + 16


def _land_x(k, p: _P):
    """The input rows into LDS (batches of XB two-word pieces a thread a pass, XG batches out at once, each checked
    against the step's tag)."""
    per = p.rows * p.k // 2
    batches = []
    for ps in range_constexpr(-(-per // (THREADS * 2 * XB))):
        specs, dst = [], []
        for h in range_constexpr(XB):
            i = fx.min((k.tid + (ps * XB + h) * THREADS) * 2, fx.Int32(per - 2))
            r, w = i // (p.k // 2), i % (p.k // 2)
            specs.append((k.mb[p.x], r * p.k // 2 + w, 2))
            dst.append(r * p.xpitch + w)
        batches.append((specs, dst))
    for g0 in range(0, len(batches), XG):
        grp = batches[g0 : g0 + XG]
        first = [_load_batch(specs) for specs, _ in grp]
        rocdl.sched_barrier(0)
        for (specs, dst), v in zip(grp, first, strict=True):
            v = _spin_from(v, partial(_load_batch, specs), partial(_pending, k.tag))
            for h in range_constexpr(XB):
                for e in range_constexpr(2):
                    fx.ptr_store(v[4 * h + 2 * e], k.lds + (dst[h] + e))
    gpu.barrier()


def _x_word(k, p: _P, st, live, r=0):
    """Row ``r``'s x words as the B operand's column ``r`` (lane ``16 kg + r``), every other column zero."""
    lm, kg = k.lane % 16, k.lane // 16
    row = fx.min(lm, fx.Int32(p.rows - 1))
    xb = fx.ptr_load(k.ldsf + (row * p.xpitch + st * 16 + kg * 4),
                     result_type=fx.Vector.make_type(4, fx.Float32)).bitcast(fx.Int32)  # fmt: skip
    mine = (lm < p.rows) if p.rows == 1 else (lm == r)
    return fx.Vector.from_elements(
        [(mine & live).select(xb[e], fx.Int32(0)) for e in range(4)], fx.Int32
    )


def _reduce(k, p: _P, acc):
    for gl in range_constexpr(p.g):
        fx.ptr_store(
            acc[gl], k.ldsf + (p.a_red + ((k.wave * p.g + gl) * 64 + k.lane) * 4)
        )
    gpu.barrier()


def _sums(k, p: _P, gl, l_):
    s = fx.ptr_load(
        k.ldsf + (p.a_red + (gl * 64 + l_) * 4),
        result_type=fx.Vector.make_type(4, fx.Float32),
    )
    vals = [fx.Float32(s[e]) for e in range(4)]
    for w in range_constexpr(1, WAVES):
        v = fx.ptr_load(k.ldsf + (p.a_red + ((w * p.g + gl) * 64 + l_) * 4),
                        result_type=fx.Vector.make_type(4, fx.Float32))  # fmt: skip
        vals = [vals[e] + fx.Float32(v[e]) for e in range(4)]
    return vals


@traced
def _put_act(k, ok, a, e, vals):
    if ok:
        chan.put_bf16(a, e, vals, k.tag)


def _silu(g):
    return g / (fx.Float32(1.0) + fmath.exp(fx.Float32(0.0) - g))


# ------------------------------------------------------------------ gate_up


def _gu_task(k, t0, p: _P):
    per = p.slots * p.gpe // p.g  # tasks a row
    r = t0 // per if p.rows > 1 else 0
    t = t0 % per if p.rows > 1 else t0
    vg0 = t * p.g
    s = vg0 // p.gpe  # the task's slot (g divides the expert's groups: one slot a task)
    lg0 = vg0 % p.gpe
    rid = chan.poll_i32(
        [(k.mb[p.ids], r * p.top_k + fx.min(s, fx.Int32(p.top_k - 1)))], k.tag
    )[0]
    rid = fx.max(fx.min(rid, fx.Int32(p.experts - 1)), fx.Int32(0))
    e = uniform((s < p.top_k).select(rid, fx.Int32(p.experts)))
    wr = rsrc(fx.Int64(k.args[f"{p.name}.w"]) + fx.Int64(e) * fx.Int64(p.unit_bytes))
    nst = -(-p.ksteps_e // WAVES)
    units = [(i, gl) for i in range(nst) for gl in range(p.g)]

    def issue(u):
        i, gl = units[u]
        st = fx.min(k.wave + i * WAVES, fx.Int32(p.ksteps_e - 1))
        return fx.Vector(bo.buffer_load(wr, (((lg0 + gl) * p.ksteps_e + st) * 64 + k.lane) * 4, vec_width=4,
                                        dtype=T.i32, cache_modifier=CM_NT))  # fmt: skip

    pre = min(PRE, len(units))
    q = [issue(u) for u in range(pre)]
    rocdl.sched_barrier(0)
    _land_x(k, p)
    acc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(p.g)]
    xw = None
    for u, (i, gl) in enumerate(units):
        if gl == 0:
            st = fx.min(k.wave + i * WAVES, fx.Int32(p.ksteps_e - 1))
            xw = _x_word(k, p, st, k.wave + i * WAVES < p.ksteps_e, r)
        rocdl.sched_barrier(0)
        if u + pre < len(units):
            q.append(issue(u + pre))
        acc[gl] = mfma_bf16(q[u].bitcast(fx.BFloat16), xw.bitcast(fx.BFloat16), acc[gl])
    rocdl.sched_barrier(0)
    _reduce(k, p, acc)
    w_s = chan.poll_f32([(k.mb[p.wts], r * p.slots + s)], k.tag)[0]
    pairs = p.g // 2
    for ps in range_constexpr(-(-pairs * 64 // THREADS)):
        it = fx.min(k.tid + ps * THREADS, fx.Int32(pairs * 64 - 1))
        ok0 = k.tid + ps * THREADS < pairs * 64
        j, l_ = it // 64, it % 64
        lm, kg = l_ % 16, l_ // 16
        gate = _sums(k, p, 2 * j, l_)
        up = _sums(k, p, 2 * j + 1, l_)
        vals = [w_s * _silu(gate[e]) * up[e] for e in range(4)]
        f0 = ((lg0 + 2 * j) // 2) * 16 + kg * 4
        mine = (lm < p.rows) if p.rows == 1 else (lm == r)
        _put_act(
            k,
            ok0 & mine,
            k.mb[p.out],
            (fx.min(lm, fx.Int32(p.rows - 1)) * p.slots + s) * p.inter + f0,
            vals,
        )
    gpu.barrier()


class GateUpBF16(Block):
    """The slots' gate / up GEMVs and the weighted SiLU (module docstring)."""

    def __init__(self, name: str, x: Channel, ids: Channel, wts: Channel, act: Channel, experts: int, inter: int,
                 top_k: int, g: int = 4):  # fmt: skip
        if (
            x.fmt is not Fmt.BF16
            or act.fmt is not Fmt.BF16
            or act.width != (top_k + 1) * inter
        ):
            raise ValueError(
                f"GateUpBF16 {name}: x BF16 [rows, H], act BF16 [rows, (top_k + 1) x inter]"
            )
        if (2 * inter // 16) % g or g % 2 or x.width % 256:
            raise ValueError(
                f"GateUpBF16 {name}: g {g} must be even and divide the expert's {2 * inter // 16} groups"
            )
        self.name, self.x, self.ids, self.wts, self.act = name, x, ids, wts, act
        self.experts, self.inter, self.top_k, self.g = experts, inter, top_k, g

    def plan(self, sc: Scenario) -> _P:
        if sc.rows > 16:
            raise ValueError(
                f"GateUpBF16 {self.name}: at most 16 rows a step (got {sc.rows})"
            )
        h = self.x.width
        return _P(self.name, self.x.name, self.ids.name, self.wts.name, self.act.name, sc.rows, h, 2 * self.inter,
                  h // 32, self.top_k, self.experts, self.g, self.inter)  # fmt: skip

    def stages(self, sc: Scenario) -> list[Stage]:
        p = self.plan(sc)
        tasks = p.rows * p.slots * p.gpe // p.g
        return [Stage(p.name, tasks, partial(_gu_task, p=p), lds_words=p.lds,
                      writes=tuple(self.act.regions(p.rows, (p.name,))), args=(f"{p.name}.w",), key=astuple(p))]  # fmt: skip

    def pack(self, layer) -> dict[str, torch.Tensor]:
        """gate / up of every routed expert (``mlp.experts.gate_proj`` / ``up_proj`` [E x I, H]) and the shared
        expert (slot ``E``), interleaved per 16 rows, in ``pack_bf16``'s units."""
        lin, i = layer.linears, self.inter
        gate = torch.cat([lin["mlp.experts.gate_proj"].reshape(-1, i, self.x.width),
                          lin["mlp.shared_expert.gate_proj"].reshape(1, i, self.x.width)])  # fmt: skip
        up = torch.cat([lin["mlp.experts.up_proj"].reshape(-1, i, self.x.width),
                        lin["mlp.shared_expert.up_proj"].reshape(1, i, self.x.width)])  # fmt: skip
        e1 = gate.shape[0]
        w = torch.stack(
            (gate.reshape(e1, i // 16, 16, -1), up.reshape(e1, i // 16, 16, -1)), 2
        )
        return {
            f"{self.name}.w": pack_bf16(w.reshape(e1 * 2 * i, -1).to(torch.bfloat16))
        }


# ------------------------------------------------------------------ down


def _dn_task(k, t0, p: _P, w: hcw.HCWrite):
    tid = k.tid
    per = p.n_e // 16 // p.g  # tasks a row
    r = t0 // per if p.rows > 1 else 0
    t = t0 % per if p.rows > 1 else t0
    # the slots' expert ids into LDS (the routed ranks polled, the shared slot fixed)
    rid = chan.poll_i32(
        [(k.mb[p.ids], r * p.top_k + fx.min(tid, fx.Int32(p.top_k - 1)))], k.tag
    )[0]
    rid = fx.max(fx.min(rid, fx.Int32(p.experts - 1)), fx.Int32(0))
    _store_if(
        tid < p.slots,
        (tid < p.top_k).select(rid, fx.Int32(p.experts)),
        k.lds + (p.a_ids + fx.min(tid, fx.Int32(p.slots - 1))),
    )
    gpu.barrier()
    base = fx.Int64(k.args[f"{p.name}.w"])
    kst = p.k // 32  # the K steps of every slot
    nst = -(-kst // WAVES)
    units = [(i, gl) for i in range(nst) for gl in range(p.g)]

    # (the wave, as a per-lane value: the units' slots and steps stay in VGPRs, not 28 SGPR sets)
    wv = tid // 64

    def issue(u):
        i, gl = units[u]
        st = fx.min(wv + i * WAVES, fx.Int32(kst - 1))
        s, ls = st // p.ksteps_e, st % p.ksteps_e
        e = fx.Int32(fx.ptr_load(k.lds + (p.a_ids + s)))
        unit = (t * p.g + gl) * p.ksteps_e + ls
        addr = (
            base
            + fx.Int64(e) * fx.Int64(p.unit_bytes)
            + fx.Int64((unit * 64 + k.lane) * 16)
        )
        return fx.Vector(global_load(addr, 4, nt=True))

    pre = min(PRE, len(units))
    q = [issue(u) for u in range(pre)]
    rocdl.sched_barrier(0)
    _land_x(k, p)
    acc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(p.g)]
    xw = None
    for u, (i, gl) in enumerate(units):
        if gl == 0:
            st = fx.min(k.wave + i * WAVES, fx.Int32(kst - 1))
            xw = _x_word(k, p, st, k.wave + i * WAVES < kst, r)
        rocdl.sched_barrier(0)
        if u + pre < len(units):
            q.append(issue(u + pre))
        acc[gl] = mfma_bf16(q[u].bitcast(fx.BFloat16), xw.bitcast(fx.BFloat16), acc[gl])
    rocdl.sched_barrier(0)
    _reduce(k, p, acc)
    hidden = p.n_e
    for ps in range_constexpr(-(-p.g * 64 // THREADS)):
        it = fx.min(tid + ps * THREADS, fx.Int32(p.g * 64 - 1))
        ok0 = tid + ps * THREADS < p.g * 64
        gl, l_ = it // 64, it % 64
        lm, kg = l_ % 16, l_ // 16
        vals = _sums(k, p, gl, l_)
        row = fx.min(lm, fx.Int32(p.rows - 1))
        col = (t * p.g + gl) * 16 + kg * 4
        mine = (lm < p.rows) if p.rows == 1 else (lm == r)
        issued = hcw.write_issue(k, w, hidden, row, col)
        hcw.write_store(k, w, hidden, ok0 & mine, row, col, vals, issued)
    gpu.barrier()


class DownBF16(Block):
    """Every slot's down GEMV as one K, then the hyper-connection combine (module docstring)."""

    def __init__(self, name: str, act: Channel, ids: Channel, out: hcw.HCWrite, experts: int, inter: int,
                 hidden: int, top_k: int, g: int = 1):  # fmt: skip
        if (
            act.fmt is not Fmt.BF16
            or act.width != (top_k + 1) * inter
            or inter % 32
            or hidden % 16
        ):
            raise ValueError(f"DownBF16 {name}: act BF16 [rows, (top_k + 1) x inter]")
        out.check(hidden)
        self.name, self.act, self.ids, self.out = name, act, ids, out
        self.experts, self.inter, self.hidden, self.top_k, self.g = (
            experts,
            inter,
            hidden,
            top_k,
            g,
        )

    def plan(self, sc: Scenario) -> _P:
        if sc.rows > 16:
            raise ValueError(
                f"DownBF16 {self.name}: at most 16 rows a step (got {sc.rows})"
            )
        return _P(self.name, self.act.name, self.ids.name, "", "", sc.rows, (self.top_k + 1) * self.inter,
                  self.hidden, self.inter // 32, self.top_k, self.experts, self.g, self.inter)  # fmt: skip

    def stages(self, sc: Scenario) -> list[Stage]:
        p = self.plan(sc)
        tasks = p.rows * (self.hidden // 16 // p.g)
        return [Stage(p.name, tasks, partial(_dn_task, p=p, w=self.out), lds_words=p.lds,
                      writes=tuple(self.out.writes(p.rows, p.name)), args=(f"{p.name}.w",) + self.out.args(),
                      key=astuple(p) + (self.out.inj.name, self.out.out, self.out.arg,
                                        self.out.channel.name if self.out.channel else None,
                                        self.out.publish.name if self.out.publish else None))]  # fmt: skip

    def pack(self, layer) -> dict[str, torch.Tensor]:
        """down of every routed expert (``mlp.experts.down_proj`` [E x H, I]) and the shared expert (slot ``E``)."""
        lin, h, i = layer.linears, self.hidden, self.inter
        w = torch.cat([lin["mlp.experts.down_proj"].reshape(-1, h, i),
                       lin["mlp.shared_expert.down_proj"].reshape(1, h, i)])  # fmt: skip
        return {f"{self.name}.w": pack_bf16(w.reshape(-1, i).to(torch.bfloat16))}

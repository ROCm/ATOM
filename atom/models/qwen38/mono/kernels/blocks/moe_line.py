# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``LineMoE``: the target's routed + shared experts (gate_up, SwiGLU, down, the weighted combine) as one
expert-major line of CTAs.

Every line CTA selects the step's experts itself from the Router's partial logits (``moe._route``: the union of the
step's experts in LDS), then streams an equal share of the step's actual expert bytes, gate_up first, then down.

**Units** (after the routing, ``S`` = the union's ``U`` routed experts + the fixed experts ``shared``, slot-major:
the routed union in first-occurrence order, then the fixed ones):

- gate_up unit = (slot, block of ``g`` interleaved row groups): ``g // 2`` MXFP8 groups of 32 intermediate features
  (``g`` = 4: one group). The waves split it into ``wk`` K waves (``_gu_waves``; K step ``wk + wk_n i``, partials
  summed in wave order). The epilogue (SwiGLU, MXFP8) goes into the channel ``act`` at (row, the row's rank of the
  slot).
- down unit = (slot, one 16-row group of the down weight): the chain over the expert's K steps in order, times the
  row's combine weight (fp32); published to the F32 channel ``ys`` [rows, slots x hidden] at (row, rank).

**The line** (``_deal``, the same integers on host and device): CTA ``t`` of ``N`` takes gate_up units
``[t G / N, (t + 1) G / N)`` and down units ``[D(t), D(t + 1))``, ``D`` the running maximum of ``(t W / N -
wg [t G / N]) / wd`` clipped to ``[0, Dn]``: ``wg`` / ``wd`` a unit's weight bytes in 1 KiB units, ``W`` every unit's;
every CTA's bytes equal to within one down unit unless its gate_up units alone exceed its share. A CTA's down units
are the slots its gate_up units are of, so the exchange stays between neighbours.

**Stages**: ``<name>.line`` (one task per line CTA: the routing, the gate_up units (statically ``gu_max`` a CTA,
the ones past its range read nothing), then the down units (wave ``w``: units ``w, w + 8, ..`` of its range), polling
the slots of ``act`` they read: an exchange region, put before polled); ``<name>.combine`` (a task per ``CG``
16-column groups of the output: polls ``ys`` of every (row, rank), sums them in rank order (fp32), then the
``HCWrite`` epilogue). Row chunks (``moe.ROW_BLOCK``) run one after the other on every line CTA.
"""

from dataclasses import dataclass, replace
from functools import partial
from types import SimpleNamespace

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl

from atom.models.qwen38.mono.kernels.blocks import moe as M
from atom.models.qwen38.mono.kernels.blocks.base import Block, Channel, Fmt
from atom.models.qwen38.mono.kernels.blocks.linear import _items, _quant_out, _sum_red
from atom.models.qwen38.mono.kernels.device import gemv
from atom.models.qwen38.mono.kernels.device import mailbox as mb
from atom.models.qwen38.mono.kernels.device.ops import fresh, traced, wave_max
from atom.models.qwen38.mono.kernels.geometry import MoEGeometry
from atom.models.qwen38.mono.kernels.program import PlanError, Scenario, Stage
from atom.models.qwen38.mono.kernels.target import MI355X

WAVES = MI355X.waves
THREADS = MI355X.threads
LANES = THREADS // WAVES
LINES = MI355X.cus  # CTAs on the line (from ``base``)
CG = 4  # combine: 16-column groups a task
G = 4  # gate_up unit: row groups at least (one MX group of the intermediate)
GU_PRE = 16  # gate_up weight units a wave keeps in flight
ACT_SLOTS = 2  # slots a CTA's down range spans at most (checked by ``LineMoE.plan``)


def _gu_lo(t, g, n):
    """The first gate_up unit of CTA ``t`` (``g`` units over ``n`` CTAs): ``t g / n`` (every CTA floor or ceil of
    g / n, interleaved). Host ints or traced Int32."""
    return t * g // n


def _deal(
    s: int, n: int, nb_gu: int, nb_dn: int, wg: int, wd: int
) -> list[tuple[int, int, int, int]]:
    """The line's ranges for ``s`` slots over ``n`` CTAs: [(gate_up lo, hi, down lo, hi)] per CTA (module
    docstring; ``_ranges`` computes the same on the device)."""
    g, d = nb_gu * s, nb_dn * s
    w = wg * g + wd * d
    m, lo = 0, [0]
    for t in range(1, n + 1):
        m = max(m, (t * w // n - wg * _gu_lo(t, g, n)) // wd)
        lo.append(min(max(m, 0), d))
    return [(_gu_lo(t, g, n), _gu_lo(t + 1, g, n), lo[t], lo[t + 1]) for t in range(n)]


def _gu_waves(chunks: int, utot: int, n_rg: int, ctas: int) -> int:
    """The gate_up unit's K waves: those of the expert-gathered gate_up task this line replaced (the same chains,
    so the same bits): its fewest row groups ``g`` (a multiple of 4) whose ``chunks x utot x n_rg / g`` tasks fit
    one round of ``ctas`` CTAs, over the fewest wave rows that leave a wave at most 4 row groups (else the most).
    The target's shapes: 2 at 4 rows (one chunk), else 1."""
    g = 4
    while g < n_rg and chunks * utot * -(-n_rg // g) > ctas:
        g += 4
    ws = [w for w in (1, 2, 4, 8) if g % w == 0]
    small = [w for w in ws if g // w <= 4]
    return WAVES // (small[0] if small else ws[-1])


@dataclass(frozen=True)
class _LPlan:
    name: str
    gu: M._GPlan  # the gate_up unit (g row groups, ``wk`` K waves)
    dn: M._GPlan  # the down unit (one row group)
    gu_max: int  # gate_up units a CTA at most (any union)
    dnw: int  # down units a wave at most
    dn_pre: int  # down weight units a wave keeps in flight
    w_cpol: int  # the expert weight loads' cache-policy bits (2 nt, 16 sc1, 1 sc0, or a sum)
    dn_issue: str  # the down's weights: "epi" (after the gate_up epilogue) | "gu" (after its MFMAs)
    ys: str  # the F32 channel [rows, slots x hidden]
    hcw: object  # the combine's HCWrite
    # row chunks: the chunk's first row in an SGPR ("s") or a VGPR ("v")
    chunk_reg: str = "s"

    @property
    def rows(self):  # rows of a chunk (the unit plans' ``rows``)
        return self.gu.rows

    @property
    def nrows(self):  # the step's rows
        return self.gu.nrows

    @property
    def chunks(self):  # row chunks, run one after the other on every line CTA
        return self.gu.chunks

    @property
    def slots(self):
        return self.gu.slots

    @property
    def hidden(self):
        return self.dn.n

    @property
    def nb_gu(self):  # gate_up units of a slot
        return self.gu.n_rg // self.gu.g

    @property
    def nb_dn(self):  # down units of a slot
        return self.dn.n_rg

    @property
    def wg(self):  # a gate_up unit's weight units (1 KiB)
        return self.gu.g * self.gu.eks

    @property
    def wd(self):
        return self.dn.eks

    @property
    def xoff(
        self,
    ):  # LDS: the gate_up input past the selection's logits (it lands while they are in use)
        return -(-(self.rows * self.gu.lg_total) // 4) * 4

    @property
    def part0(self):  # the gate_up partials, after its input
        return (
            -(
                -(self.gu.x0 + self.xoff + self.gu.xs_base + self.rows * self.gu.spitch)
                // 4
            )
            * 4
        )

    @property
    def gu_end(self):
        return self.part0 + self.gu.wk * self.gu_max * self.gu.g * self.rows * 16

    @property
    def lds_words(self):
        dn_end = self.dn.x0 + self.dn.xs_base + self.rows * self.dn.spitch
        lg_end = self.gu.x0 + self.rows * self.gu.lg_total
        return -(-max(self.gu_end, dn_end, lg_end) // 4) * 4

    @property
    def combine_tasks(self):
        return -(-self.nb_dn // CG)


class LineMoE(Block):
    """The target's experts as one expert-major line (module docstring): ``x`` the MXFP8 input rows, ``logits`` the
    Router's partials, ``hcw`` the combine's ``HCWrite``, ``args`` Atom's expert tensors (``moe.MoEArgs``, read in
    place). Knobs: ``dn_pre`` (down weight units in flight a wave), ``dn_issue`` (when the down's weights go out),
    ``w_cpol``, ``cbase`` (the combine's first CTA, default ``base``), ``chunk_reg``."""

    def __init__(
        self,
        name: str,
        geo: MoEGeometry,
        x: Channel,
        logits: M.RouterLogits,
        hcw,
        args: M.MoEArgs,
        base: int = 0,
        dn_pre: int = 20,
        dn_issue: str = "epi",
        w_cpol: int = 2,
        cbase: int | None = None,
        chunk_reg: str = "s",
    ):
        from atom.models.qwen38.mono.kernels.weights.aiter_moe import padded

        if dn_issue not in ("epi", "gu"):
            raise ValueError(f"LineMoE {name}: dn_issue {dn_issue!r}")
        slots, tslots, p_ = (
            geo.top_k + geo.shared,
            geo.experts + geo.shared,
            padded(geo.inter),
        )
        self.name, self.geo, self.x, self.logits, self.hcw = name, geo, x, logits, hcw
        self.w13 = M.Experts(
            args.w13,
            args.s13,
            tslots,
            2 * geo.inter,
            geo.hidden,
            2 * p_,
            geo.hidden,
            True,
        )
        self.w2 = M.Experts(
            args.w2, args.s2, tslots, geo.hidden, geo.inter, geo.hidden, p_, False
        )
        self.act = Channel(f"{name}.act", slots * geo.inter, Fmt.MXFP8)
        self.ys = Channel(f"{name}.ys", slots * geo.hidden, Fmt.F32)
        self.base, self.cbase = base, base if cbase is None else cbase
        self.dn_pre, self.dn_issue, self.w_cpol, self.chunk_reg = (
            dn_pre,
            dn_issue,
            w_cpol,
            chunk_reg,
        )

    def _units(self, sc: Scenario) -> tuple[M._GPlan, M._GPlan]:
        """The gate_up and down unit plans of the step: row chunks of at most ``moe.ROW_BLOCK`` rows."""
        geo, lg = self.geo, self.logits
        chunks = -(-sc.rows // M.ROW_BLOCK)
        rows = -(-sc.rows // chunks)  # a chunk's rows (the last chunk may hold fewer)
        if rows * geo.top_k > THREADS:  # (``moe._route``: one (row, rank) a thread)
            raise PlanError(f"LineMoE {self.name}: rows x top_k > {THREADS}")
        for e in (self.w13, self.w2):  # one buffer resource a tensor
            if (e.slots + 1) * e.w_bytes >= 1 << 32:
                raise PlanError(
                    f"LineMoE {self.name}: the expert tensor {e.w} exceeds one buffer resource"
                )
        umax = min(rows * geo.top_k, geo.experts)
        shared = tuple(range(geo.experts, geo.experts + geo.shared))
        # (the CTAs the gathered down's hidden / 32 tasks left the gathered gate_up)
        wk = _gu_waves(
            chunks,
            umax + len(shared),
            self.w13.n // 16,
            MI355X.cus - -(-geo.hidden // 32),
        )
        g = max(G, WAVES // wk)
        route = {"rows": rows, "top_k": geo.top_k, "experts": geo.experts, "shared": shared, "umax": umax, "lg": lg.part,
                     "lg_ks": lg.k_split, "lg_total": lg.total, "rows_all": sc.rows if chunks > 1 else 0}  # fmt: skip

        def unit(part, e, x, **kw):
            return M._GPlan(name=f"{self.name}.{part}", n=e.n, ek=e.k, k=x.width, x=x.name, tslots=e.slots,
                            n_stored=e.n_stored, k_stored=e.k_stored, w=e.w, s=e.s, **route, **kw)  # fmt: skip

        gu = unit(
            "gate_up",
            self.w13,
            self.x,
            g=g,
            wr=WAVES // wk,
            wk=wk,
            out_ch=self.act.name,
        )
        if g % 4 or gu.n_rg % g:
            raise PlanError(
                f"LineMoE {self.name}: {g} row groups a unit over {WAVES // wk} wave rows"
            )
        return gu, unit("down", self.w2, self.act, g=1, wr=1, wk=WAVES, out_ch="")

    def plan(self, sc: Scenario) -> _LPlan:
        gu, dn = self._units(sc)
        deals = [
            _deal(s, LINES, gu.n_rg // gu.g, dn.n_rg, gu.g * gu.eks, dn.eks)
            for s in range(1, gu.utot + 1)
        ]
        gu_max = max(hi - lo for d in deals for lo, hi, _, _ in d)
        dn_max = max(hi - lo for d in deals for _, _, lo, hi in d)
        spans = max(
            (hi - 1) // dn.n_rg - lo // dn.n_rg + 1
            for d in deals
            for _, _, lo, hi in d
            if hi > lo
        )
        if spans > ACT_SLOTS:
            raise PlanError(
                f"LineMoE {self.name}: a down range spans {spans} slots (> {ACT_SLOTS})"
            )
        p = _LPlan(
            name=self.name,
            gu=gu,
            dn=dn,
            gu_max=gu_max,
            dnw=-(-dn_max // WAVES),
            dn_pre=self.dn_pre,
            w_cpol=self.w_cpol,
            dn_issue=self.dn_issue,
            ys=self.ys.name,
            hcw=self.hcw,
            chunk_reg=self.chunk_reg,
        )
        if p.lds_words * 4 > MI355X.lds_bytes - 1024:
            raise PlanError(f"LineMoE {self.name}: LDS {p.lds_words * 4} B")
        return p

    def stages(self, sc: Scenario) -> list[Stage]:
        p = self.plan(sc)
        line = f"{self.name}.line"
        comb = f"{self.name}.combine"
        writes = [replace(r, exchange=True) for r in self.act.regions(p.nrows, (line,))]
        writes += list(self.ys.regions(p.nrows, (line,)))
        args = (p.gu.w, p.gu.s, p.dn.w, p.dn.s)
        p.hcw.check(p.hidden)
        return [
            Stage(
                line,
                LINES,
                partial(_line, p=p),
                lds_words=p.lds_words,
                writes=tuple(writes),
                args=args,
                base=self.base,
                key=(p,),
            ),
            Stage(
                comb,
                p.combine_tasks,
                partial(_combine, p=p),
                writes=tuple(p.hcw.writes(p.nrows, comb)),
                args=tuple(p.hcw.args()),
                base=self.cbase,
                key=(p,),
            ),
        ]

    def weight_specs(self) -> tuple:
        return self.w13.weight_specs() + self.w2.weight_specs()


# ------------------------------------------------------------------ device


def _ranges(k, p, t):
    """CTA ``t``'s (gate_up lo, hi, down lo, hi) for the step's slot count (``_deal``), wave-uniform."""
    s = M._lds_i(k, p.gu.r_cnt) + len(p.gu.shared)
    g, d = s * p.nb_gu, s * p.nb_dn
    w = g * p.wg + d * p.wd
    n = LINES

    def f(x):
        return (x * w // n - _gu_lo(x, g, n) * p.wg) // p.wd

    lane = fresh(k.lane)
    m = fx.Int32(0)
    for q in range_constexpr(-(-n // LANES)):
        x = lane + q * LANES
        m = fx.max(m, (x <= t).select(f(x), fx.Int32(0)))
    m = wave_max(m)
    lo = fx.min(m, d)
    hi = fx.min(fx.max(m, f(t + 1)), d)
    return (_gu_lo(t, g, n), _gu_lo(t + 1, g, n), lo, hi)


def _slot_of(k, p, s, ok):
    """Line slot ``s`` (routed union first, then the fixed experts) -> the union slot of ``moe``'s tables; a unit not
    on this CTA's range gets a slot past every slot (its expert reads nothing)."""
    cnt = M._lds_i(k, p.r_cnt)
    u = (s < cnt).select(s, fx.Int32(p.umax) + s - cnt)
    return ok.select(u, fx.Int32(p.utot))


def _view(k, off):
    return SimpleNamespace(**{**vars(k), "lds": k.lds + off, "ldsf": k.ldsf + off})


def _line(k, t, p):
    """One CTA of the line: per row chunk (one after the other, a loop: one chunk's registers), the chunk's routing,
    its gate_up units, then its down units (their weights issued after the gate_up epilogue, or with ``dn_issue``
    "gu" right after its MFMAs)."""
    if const_expr(p.chunks == 1):
        _line_chunk(k, t, p, 0)
    else:
        _line_chunks(k, t, p)


@traced
def _line_chunks(k, t, p):
    for c in range(p.chunks):
        row0 = fx.Int32(c) * p.rows
        if const_expr(
            p.chunk_reg == "v"
        ):  # (a VGPR: the composed qsa layer is at its SGPR budget)
            row0 = fresh(row0)
        _line_chunk(k, t, p, row0)


def _line_chunk(k, t, p, row0):
    gu = p.gu
    lane = k.lane
    # the gate_up input (published before the router ran) lands with the routing
    xin = M._x_issue(k, gu, row0)
    M._route(k, gu, row0)
    M._x_land(_view(k, p.xoff), gu, *xin)
    gpu.barrier()
    gu_lo, gu_hi, dn_lo, dn_hi = _ranges(k, p, t)
    dn = _gate_up_units(
        k, p, gu_lo, gu_hi, lane, partial(_down_issue, k, p, dn_lo, dn_hi, lane), row0
    )
    gpu.barrier()
    if dn is None:
        dn = _down_issue(k, p, dn_lo, dn_hi, lane)
    _down_compute(k, p, lane, dn, row0)
    gpu.barrier()


def _gate_up_units(k, p, lo, hi, lane, dn_issue, row0=0):
    """The CTA's gate_up units ``lo ..`` (``gu_max`` slots, those at or past ``hi`` read nothing): one weight stream
    over them, the partials of every unit, then their epilogues. Returns ``dn_issue()`` if it ran (``p.dn_issue``
    "gu": after the MFMAs), else None."""
    gu = p.gu
    w = fresh(k.wave)
    wr, wk = w // gu.wk, w % gu.wk
    r_w, r_s = M._tensor_rsrc(k, gu)
    units = []
    for j in range(p.gu_max):
        x = lo + j
        ok = x < hi
        u = _slot_of(k, gu, x // p.nb_gu, ok)
        units.append((u, x % p.nb_gu, ok, M._expert(k, gu, u)))
    order = [
        (j, i, gl) for j in range(p.gu_max) for i in range(gu.c) for gl in range(gu.gw)
    ]
    pair = gu.gw % 2 == 0
    dws = {}

    def unit(x):
        j, i, gl = order[x]
        _, b, _, e = units[j]
        rg = b * gu.g + wr * gu.gw + gl
        st = M._gu_kstep(gu, wk, i)
        if not pair or gl % 2 == 0:
            sst = fx.min(st, fx.Int32(gu.eks - 1))
            dws[j, i, gl // 2 if pair else gl] = M._s_issue(
                gu, r_s, e, rg, sst // 2, lane
            )
        ok = (st < gu.eks) if gu.eks % gu.wk else None
        return M._w_issue(gu, r_w, e, rg, st, lane, ok, p.w_cpol)

    n_units = len(order)
    n_pre = min(GU_PRE, n_units)
    q = [unit(x) for x in range(n_pre)]

    kx = _view(k, gu.x0 + p.xoff)
    row = fx.min(lane % 16, fx.Int32(gu.rows - 1))
    jl = lane // 16
    acc = {
        (j, gl): fx.Vector.filled(4, 0.0, fx.Float32)
        for j in range(p.gu_max)
        for gl in range(gu.gw)
    }
    for x_ in range(n_units):
        j, i, gl = order[x_]
        st = fx.min(M._gu_kstep(gu, wk, i), fx.Int32(gu.eks - 1))
        if gl == 0:
            xb, xs = gemv.b_operand(
                kx.lds, kx.lds + gu.xs_base, row, jl, st, gu.pitch, gu.spitch
            )
        rocdl.sched_barrier(0)
        if x_ + GU_PRE < n_units:
            q.append(unit(x_ + GU_PRE))
        rg = units[j][1] * gu.g + wr * gu.gw + gl
        ws = M._code(dws[j, i, gl // 2 if pair else gl], rg, st)
        acc[j, gl] = gemv.mfma_w4a8(acc[j, gl], q[x_], ws, xb, xs)
    rocdl.sched_barrier(0)
    dn = dn_issue() if p.dn_issue == "gu" else None

    # every unit's K-wave partials -> LDS [wk][unit][row group][row][16], then the epilogues
    gpu.barrier()
    _part_store(k, p, wr, wk, acc)
    gpu.barrier()
    kp = _view(k, p.part0)
    ps_ = replace(gu, g=p.gu_max * gu.g)  # ``_sum_red``'s K-wave stride over every unit
    total = p.gu_max * gu.out_quads
    for ps in range(-(-total // THREADS)):
        idx = fx.min(fx.Int32(ps * THREADS) + k.tid, fx.Int32(total - 1))
        j = idx // gu.out_quads
        x = lo + j
        okj = (x < hi) & (fx.Int32(ps * THREADS) + k.tid < total)
        u = _slot_of(k, gu, x // p.nb_gu, okj)
        ok, n, srcs, col, lead_ = _items(gu, x % p.nb_gu, idx % gu.out_quads)
        slot = M._slot(k, gu, u, fx.min(n, fx.Int32(gu.rows - 1)))
        ok = ok & okj & (slot >= 0)
        gn, real = M._grow(gu, row0, n)  # the chunk's row -> the step's
        if real is not None:
            ok = ok & real
        base = j * (gu.g * gu.rows * 16)
        gate, up = [_sum_red(kp, ps_, base + s) for s, _ in srcs]
        mid = [M.bf16_round(M._silu(gate[c]) * up[c]) for c in range(4)]
        width = gu.slots * gu.mid
        _quant_out(
            k,
            gu,
            ok,
            gn * width + fx.max(slot, fx.Int32(0)) * gu.mid + col,
            mid,
            lead_,
            width,
        )
    return dn


@traced
def _part_store(k, p, wr, wk, acc):
    """This wave's partials of every unit (useful lanes only)."""
    gu = p.gu
    n = k.lane % 16
    if n < gu.rows:
        for j in range_constexpr(p.gu_max):
            for gl in range_constexpr(gu.gw):
                rg = j * gu.g + wr * gu.gw + gl
                off = (
                    p.part0
                    + ((wk * p.gu_max * gu.g + rg) * gu.rows + n) * 16
                    + (k.lane // 16) * 4
                )
                fx.ptr_store(acc[j, gl], k.ldsf + off)


def _down_issue(k, p, lo, hi, lane):
    """The CTA's down units ``lo .. hi`` (wave ``w``: units ``lo + w + 8 m``): their experts and the first ``dn_pre``
    weight units issued -> the state ``_down_compute`` streams on."""
    dn = p.dn
    order = [(m, st) for m in range(p.dnw) for st in range(dn.eks)]
    dws, units, rs = {}, {}, []

    def get(
        m,
    ):  # unit m's (union slot, row group, on the range, expert), built at its first use (registers)
        if m not in units:
            x = lo + fresh(k.wave) + WAVES * m
            ok = x < hi
            u = _slot_of(k, dn, x // p.nb_dn, ok)
            units[m] = (u, x % p.nb_dn, ok, M._expert(k, dn, u))
        return units[m]

    def unit(x):
        m, st = order[x]
        _, hb, _, e = get(m)
        if not rs:
            rs.extend(M._tensor_rsrc(k, dn))
        if (m, st // 2) not in dws:
            dws[m, st // 2] = M._s_issue(dn, rs[1], e, hb, st // 2, lane)
        return M._w_issue(dn, rs[0], e, hb, st, lane, None, p.w_cpol)

    n_pre = min(p.dn_pre, len(order))
    q = [unit(x) for x in range(n_pre)]
    return SimpleNamespace(get=get, order=order, dws=dws, unit=unit, q=q, lo=lo, hi=hi)


def _down_compute(k, p, lane, s, row0=0):
    """The down units of ``_down_issue``'s state ``s``: ``act`` landed, the chains, ``ys`` (the chunk from ``row0``)."""
    dn = p.dn
    get, order, dws, unit, q = s.get, s.order, s.dws, s.unit, s.q
    n_units = len(order)
    _act_land(k, p, s.lo, s.hi, row0)
    gpu.barrier()

    kx = M._x_view(k, dn)
    row = fx.min(lane % 16, fx.Int32(dn.rows - 1))
    jl = lane // 16
    x_ = 0
    for m in range(p.dnw):
        u, hb, okm, _ = get(m)
        slot = M._slot(k, dn, u, row)
        sc = fx.max(slot, fx.Int32(0))
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for st in range(dn.eks):
            xb, xs = gemv.b_operand(
                kx.lds,
                kx.lds + dn.xs_base,
                row,
                jl,
                sc * dn.eks + st,
                dn.pitch,
                dn.spitch,
            )
            rocdl.sched_barrier(0)
            while len(q) < min(x_ + 1 + p.dn_pre, n_units):
                q.append(unit(len(q)))
            acc = gemv.mfma_w4a8(acc, q[x_], M._code(dws[m, st // 2], hb, st), xb, xs)
            x_ += 1
        rocdl.sched_barrier(0)
        wv = fx.Float32(fx.ptr_load(k.ldsf + (dn.r_wts + row * dn.slots + sc)))
        ok = okm & (slot >= 0) & (lane % 16 < dn.rows)
        grow, real = M._grow(dn, row0, row)
        if real is not None:
            ok = ok & real
        e0 = (grow * dn.slots + sc) * p.hidden + hb * 16 + jl * 4
        _put_ys(k, p, ok, e0, [wv * acc[c] for c in range(4)])


def _act_land(k, p, lo, hi, row0=0):
    """``act`` of the slots the CTA's down units ``lo .. hi`` read (at most ``ACT_SLOTS``), for the rows that picked
    them, into the down's input area (its layout: row-major, the row's ranks side by side): each (slot, row) is
    polled as soon as its gate_up units are out, not every slot's."""
    dn = p.dn
    words, codes = dn.ek // 4, dn.ek // 32  # a (row, rank)'s FP8 words / scale codes
    s0, s1 = lo // p.nb_dn, (hi - 1) // p.nb_dn
    qa, sa = k.mb[f"{dn.x}.q"], k.mb[f"{dn.x}.s"]
    for kind, per, n in (("q", words // 2, 2), ("s", codes, 1)):
        total = ACT_SLOTS * dn.rows * per
        for c in range_constexpr(-(-total // THREADS)):
            i0 = fx.Int32(c * THREADS) + k.tid
            i = fx.min(i0, fx.Int32(total - 1))
            w_, r, it = i // (dn.rows * per), (i // per) % dn.rows, i % per
            sl = s0 + w_
            ok = (i0 < total) & (lo < hi) & (sl <= s1)
            u = _slot_of(k, dn, sl, ok)
            rank = M._slot(k, dn, u, r)
            rk = fx.max(rank, fx.Int32(0))
            gr = M._grow(dn, row0, r)[
                0
            ]  # the step's row (the chunk's rows past the step: no rank, no poll)
            if kind == "q":
                src = gr * (dn.xk // 4) + rk * words + it * n
                dst = r * dn.pitch + rk * words + it * n
                _act_poll_if(k, ok & (rank >= 0), qa, src, n, dn.x0 + dst)
            else:
                src = gr * (dn.xk // 32) + rk * codes + it
                dst = dn.xs_base + r * dn.spitch + rk * codes + it
                _act_poll_if(k, ok & (rank >= 0), sa, src, n, dn.x0 + dst)


@traced
def _act_poll_if(k, ok, a, idx, n, dst):
    if ok:
        got = mb.poll([(a, idx, n)], k.tag)[0]
        for e in range_constexpr(n):
            fx.ptr_store(got[e], k.lds + (dst + e))


@traced
def _put_ys(k, p, ok, e0, y):
    if ok:
        mb.put2(k.mb[p.ys], e0, y[0], y[1], k.tag)
        mb.put2(k.mb[p.ys], e0 + 2, y[2], y[3], k.tag)


def _combine(k, t, p):
    """Output columns ``16 CG t ..``: four features a thread, every (row, rank)'s ``ys`` summed in rank order, then
    the ``HCWrite`` epilogue."""
    from atom.models.qwen38.mono.kernels.blocks.hc import write_issue, write_store

    quads = p.nrows * CG * 4
    h = p.hidden
    for ps in range(-(-quads // THREADS)):
        idx = fx.Int32(ps * THREADS) + k.tid
        ok = idx < quads
        idx = fx.min(idx, fx.Int32(quads - 1))
        r, qd = idx // (CG * 4), idx % (CG * 4)
        col = t * (CG * 16) + qd * 4
        ok = ok & (col < h)
        col = fx.min(col, fx.Int32(h - 4))
        issued = write_issue(k, p.hcw, h, r, col)
        specs = []
        for s in range(p.slots):
            e = (r * p.slots + s) * h + col
            specs += [(k.mb[p.ys], e, 2), (k.mb[p.ys], e + 2, 2)]
        got = mb.poll(specs, k.tag)
        tot = None
        for s in range(p.slots):
            vals = [v.bitcast(fx.Float32) for v in got[2 * s] + got[2 * s + 1]]
            tot = vals if tot is None else [tot[c] + vals[c] for c in range(4)]
        write_store(k, p.hcw, h, ok, r, col, tot, issued)

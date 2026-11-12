# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``TopK``: per row, the indices of the ``k`` largest of ``n`` fp32 values.

The rule is aiter's stable top-k (``top_k_per_row_decode(stable=True)``, ATOM's): the fp32 bits' u32 order
(``device.gfx950.topk.order_key``: -0 below +0, NaN above +inf), **ties to the lowest index**; the
winners come out ascending by index (position order, as the V4.1 indexer re-sorts them), -1 padded to ``k``
when a row has fewer values.

**Device.** A task's CTA runs ``device.gfx950.topk.select`` (the radix select over the 64-bit (key,
index), one implementation with the Indexer's) on keys in registers, thread i holding ``per`` consecutive
positions from i x per: ``select`` places the kept keys by one CTA prefix sum in thread order, so its slots come
out in position order, which is index order at every level; ``slot`` reads them (no sort). Every u32 is a key
(``valid``: a position below the list's count), bf16 values select on their top 16 bits (``u_bits`` 16), and keys
equal to the k-th best's are told apart by that prefix sum (``in_order``: no passes over the index).

**Inputs.** ``x``: an ``F32`` channel of width ``n`` (polled) or the name of a plain f32 arg [rows, n].
``bound`` (optional): a plain int32 arg [rows], the values a row has (<= n; e.g. the indexer's visible
compressed positions); a channel's producer must have put every element below the row's bound rounded
up to a multiple of 4. **Output** ``out``: a plain int32 arg [rows, k], or an ``F32`` channel of width
``k`` carrying the int32 indices' bits.

**Shape** (adapted from ATOM's V4.1 indexer, ``atom/models/deepseek_v41/mono/index_plan.py``). One CTA
ranks at most ``seg`` values (keys in LDS). Level 0: a task per (row, segment of ``seg`` columns) ->
its top-k list. Level l > 0: list j merges the level l - 1 lists [fan j, fan (j + 1)), fan = seg / k,
laid end to end; every list but a level's last holds exactly k valid entries, so the merged input is
dense and in index order. A global winner is in its own list's top-k (fewer than k beat it there, ties
included: lower indices come first in every list), so each level keeps every winner; the tree ends at
a level of one list. A row with fewer values than the build's ``n`` runs the same levels with fewer
inputs (a level-l task beyond the row's lists ranks nothing). Lists hand over as tagged pairs
(value bits, global index) in exchange regions ``<name>.l<level>``.

**Groups** (``groups`` g, with ``split`` g: one group a TP rank, e.g. index heads under context
parallelism): ``x`` holds g x rows rows, row ``h rows + r`` group h's values for row r (the same bound, ``bound[r]``);
each rank ranks every group's share of the values and sends group h's answer to rank h only; rank h's merge ranks
its own group's answers into the output (rows rows). A request ranked whole (``scap``) ranks this rank's group only.
"""

from dataclasses import dataclass
from functools import partial

import flydsl.expr as fx
from flydsl.expr import const_expr

from ...device.gfx950 import mailbox as mb
from ...device.gfx950 import peer, topk
from ...device.gfx950.ops import ld_i32, traced, uniform
from ...device.gfx950.topk import THREADS, key_bits, order_key
from ...plan.program import Region, Scenario, Space, Stage
from ..base import Block, Channel, Fmt

SEG = 8192  # values a CTA ranks by default (ATOM K2a's choice)
MAX_PER_THREAD = 64  # keys a thread holds in registers (32768 a CTA)
NEG_INF_BITS = -8388608  # 0xFF800000: a padding entry's value
PACK_SEG = 16384  # ``TopK(packed=True)``: segments up to this (a place in one fits 14 bits, below PACK_PAD's 0xFFFF)
PACK_PAD = -8323073  # 0xFF80FFFF: a packed list's padding entry (-inf, place 0xFFFF)


# ------------------------------------------------------------------ the tree, on ints or traced Int32


def _sel(c, a, b):
    return (a if c else b) if isinstance(c, bool) else c.select(a, b)


def _min(a, b):
    return min(a, b) if isinstance(a, int) and isinstance(b, int) else fx.min(a, b)


def _max(a, b):
    return max(a, b) if isinstance(a, int) and isinstance(b, int) else fx.max(a, b)


def _cdiv(a, b):
    return (a + b - 1) // b


def segments(bound, seg):
    return _cdiv(_max(bound, 0), seg)


def seg_len(bound, s, seg):
    return _max(_min(bound - s * seg, seg), 0)


def lists(bound, level, seg, k):
    n = segments(bound, seg)
    for _ in range(level):
        n = _cdiv(n, seg // k)
    return n


def list_inputs(bound, level, j, seg, k):
    """Values list j of ``level`` ranks (0 past the row's lists)."""
    if level == 0:
        return seg_len(bound, j, seg)
    fan = seg // k
    below = lists(bound, level - 1, seg, k)
    count = _min(below - j * fan, fan)
    last = j * fan + _max(count, 1) - 1
    n = _max(count - 1, 0) * k + _min(list_inputs(bound, level - 1, last, seg, k), k)
    return _sel(j < lists(bound, level, seg, k), n, 0)


def depth(n, seg, k) -> int:
    level, m = 0, segments(n, seg)
    while m > 1:
        m = _cdiv(m, seg // k)
        level += 1
    return level


def _per_thread(values: int) -> int:
    return max(4, -(-_cdiv(values, THREADS) // 4) * 4)


# ------------------------------------------------------------------ the plan


@dataclass(frozen=True)
class _Plan:
    name: str
    rows: int
    n: int
    k: int
    seg: int
    depth: int
    x: str
    x_arg: bool  # x is a plain arg (else a channel)
    out: str
    out_arg: bool
    bound: str  # "" if none
    bf16: bool = (
        False  # x holds bf16 values (a BF16 channel / plain bf16 arg): two radix passes (``key16``)
    )
    lift: str = (
        ""  # the final level's unit ids channel ("" : the columns are the output)
    )
    lift_unit: int = 0
    live: bool = (
        False  # a bounded tree's tasks: only those of the rows' largest bound (``TopK(live=)``)
    )
    # ``TopK(split=)``: this rank's share of the values (``n`` columns: global index r Q + column, Q from the request's
    # largest bound), its answer to every rank, then the merge of the ``split`` answers (``gn``: the global indices)
    split: int = 0
    unit: int = 0
    gn: int = 0
    spec: int = 1
    scap: int = (
        0  # split: requests whose last row's bound is at most this are ranked whole on every rank (no exchange)
    )
    packed: bool = (
        False  # bf16: level 0's lists one tagged word an entry (value's bits | index in the segment)
    )
    xhoist: bool = (
        False  # split: the final level's peer addresses (every rank's answer slot) before its select
    )
    groups: int = (
        1  # x's rows: groups x rows, group h's answers to rank h (``TopK(groups=)``)
    )
    wmerge: bool = False  # split: the merge as one wave's rank (``TopK(wmerge=)``)

    @property
    def vrows(self) -> int:  # x's rows (each group's rows)
        return self.groups * self.rows

    @property
    def index_bits(
        self,
    ) -> (
        int
    ):  # global indices below 2^bits - 1 (select's ``valid``: v = 0 marks the others)
        return self.n.bit_length()

    @property
    def merge_bits(self) -> int:  # the merge's global indices
        return self.gn.bit_length()

    @property
    def xreg(self) -> str:  # the split answers' peer region
        return f"{self.name}.x"

    @property
    def u_bits(self) -> int:
        return 16 if self.bf16 else 32

    def lists(self, level: int) -> int:
        return lists(self.n, level, self.seg, self.k)

    def per_thread(self, level: int) -> int:
        if level == 0:
            return _per_thread(min(self.n, self.seg))
        return _per_thread(min(self.lists(level - 1), self.seg // self.k) * self.k)

    def region(self, level: int) -> str:
        return f"{self.name}.l{level}"

    @property
    def lift_words(self) -> int:  # unit ids a row
        return self.n // self.lift_unit if self.lift else 0


class TopK(Block):
    def __init__(
        self,
        name: str,
        x: Channel | str,
        n: int,
        k: int,
        out: Channel | str,
        bound: str | None = None,
        seg: int = SEG,
        bf16: bool = False,
        lift: Channel | None = None,
        lift_unit: int = 0,
        live: bool = False,
        split: int = 0,
        unit: int = 4,
        gn: int = 0,
        scap: int = 0,
        packed: bool = False,
        xhoist: bool = False,
        groups: int = 1,
        wmerge: bool = False,
    ):
        xfmt = Fmt.BF16 if bf16 else Fmt.F32
        if isinstance(x, Channel) and (x.fmt is not xfmt or x.width != n or x.flagged):
            raise ValueError(
                f"TopK {name}: x must be a tagged {xfmt.name} channel of width {n}"
            )
        if bf16 and n % 4:
            raise ValueError(
                f"TopK {name}: bf16 values need n % 4 == 0 (quads are word pairs)"
            )
        if isinstance(out, Channel) and (
            out.fmt is not Fmt.F32 or out.width != k or out.flagged
        ):
            raise ValueError(
                f"TopK {name}: out must be a tagged F32 channel of width {k} (int32 bits)"
            )
        if seg % 4 or seg > THREADS * MAX_PER_THREAD or not 1 <= k:
            raise ValueError(
                f"TopK {name}: seg {seg} (a multiple of 4 up to {THREADS * MAX_PER_THREAD}), k {k}"
            )
        if n > seg and (k % 4 or seg // k < 2):
            raise ValueError(
                f"TopK {name}: a merge tree needs k % 4 == 0 and k <= seg / 2 (k {k}, seg {seg})"
            )
        if lift is not None and (
            lift_unit < 1 or n % lift_unit or lift.fmt is not Fmt.F32 or lift.width != n // lift_unit
            or lift.flagged or (n // lift_unit) % (2 * THREADS)
        ):  # fmt: skip
            raise ValueError(f"TopK {name}: lift is a tagged F32 channel of n / lift_unit unit ids, a multiple of "
                             f"{2 * THREADS}")  # fmt: skip
        self.name, self.x, self.n, self.k, self.out, self.bound, self.seg = (
            name,
            x,
            n,
            k,
            out,
            bound or "",
            seg,
        )
        self.bf16 = bf16
        self.lift, self.lift_unit = lift, lift_unit if lift is not None else 0
        # live tasks (``Stage.live_tasks``): a bounded tree runs, at each level, the tasks of the lists the rows'
        # largest bound has (level 0: one at least, the task that writes a short row's output); the others rank
        # nothing (``list_inputs`` 0) and are not run. The same lists, ranked by the same select: the same ids.
        self.live = bool(live and self.bound and depth(n, seg, k))
        # split (TP ranks): ``x`` holds this rank's share, positions [r Q, r Q + Q) of a request whose last row's
        # bound is m, Q = ceil(m / (split unit)) unit (a request's bounds rise over its rows); local bounds
        # clamp(bound - r Q, 0, Q). Each rank's answer (k (value, global index) pairs a row) goes to every rank; the
        # merge ranks the split x k: every winner is in its rank's answer (fewer than k beat it there, ties to the
        # lower index in both), so the same ids and order as one TopK over all values.
        if split > 1 and (not self.bound or lift is not None or unit % 4 or gn < n):
            raise ValueError(
                f"TopK {name}: split needs a bound, no lift, unit % 4 == 0 and gn >= n"
            )
        self.split, self.unit, self.gn = (split, unit, gn) if split > 1 else (0, 0, 0)
        # ``scap``: a request whose bounds are at most scap (<= n: this rank's columns hold them all) is ranked whole on
        # every rank, its answer the output directly: the exchange and the merge only for longer requests
        if self.split and scap > n:
            raise ValueError(f"TopK {name}: scap {scap} past this rank's {n} columns")
        self.scap = scap if self.split else 0
        # ``packed`` (bf16 values only, which rank on their top 16 bits): level 0's lists hand over one tagged word an
        # entry, the value's bits | its place in the task's segment (``PACK_PAD``: padding), instead of the (value,
        # global index) pair; level 1 recovers the index from the list's place in its input (its segment's first
        # column). The same keys and indices: the same ids. Level 1's input is half the bytes.
        if packed and (not bf16 or seg > PACK_SEG):
            raise ValueError(
                f"TopK {name}: packed lists need bf16 values and seg <= {PACK_SEG} (seg {seg})"
            )
        self.packed = bool(packed and depth(n, seg, k))
        # ``xhoist`` (split): the final level's answer puts to every rank address the peers' slots from base pointers
        # loaded at the task's start (before its keys and select), not at the puts: the same stores, the pointer loads'
        # latency hidden behind the select
        self.xhoist = bool(xhoist and self.split)
        # groups (module docstring): group h's answers go to rank h, whose merge ranks them
        if groups > 1 and (
            self.split != groups or lift is not None or packed or xhoist
        ):
            raise ValueError(f"TopK {name}: groups {groups} needs split {groups} (a group a rank), no lift / packed / "
                             "xhoist")  # fmt: skip
        self.groups = groups
        # ``wmerge`` (split): the merge's split x k answers (at most a wave's 64) ranked by one wave
        # (``device.gfx950.topk.wave_topk_won_thresh``: a bound and a rank count) instead of the CTA's radix select
        # (~3 µs of passes and barriers whatever the count): the same winners, put in index order
        if wmerge and (not self.split or self.split * k > 64):
            raise ValueError(
                f"TopK {name}: wmerge needs split with split x k <= 64 (split {self.split}, k {k})"
            )
        self.wmerge = bool(wmerge)

    def plan(self, sc: Scenario) -> _Plan:
        x_arg, out_arg = isinstance(self.x, str), isinstance(self.out, str)
        return _Plan(
            self.name, sc.rows, self.n, self.k, self.seg, depth(self.n, self.seg, self.k),
            self.x if x_arg else self.x.name, x_arg, self.out if out_arg else self.out.name, out_arg, self.bound,
            self.bf16, self.lift.name if self.lift else "", self.lift_unit,
            self.live, self.split, self.unit, self.gn, sc.spec, self.scap, self.packed, self.xhoist, self.groups,
            self.wmerge,
        )  # fmt: skip

    def stages(self, sc: Scenario) -> list[Stage]:
        p = self.plan(sc)
        out = []
        short = bool(
            p.bound and p.depth
        )  # a bounded tree: level 0 writes the output for rows of one segment
        names = [
            f"{self.name}.l{level}" if p.depth else self.name
            for level in range(p.depth + 1)
        ]
        for level in range(p.depth + 1):
            final = level == p.depth
            name = names[level]
            # (live: a row's answer comes from the first level with one list of it, any level)
            outputs = p.live or final or (short and level == 0)
            if not final:
                per = 1 if p.packed and level == 0 else 2  # pairs an entry
                writes = (
                    Region(
                        p.region(level),
                        p.vrows * p.lists(level) * p.k * per,
                        (name,),
                        exchange=True,
                    ),
                )
            else:
                writes = ()
            writers = (
                tuple(names)
                if p.live
                else ((f"{self.name}.l0",) if short else ()) + (name,)
            )
            if final and p.split:  # the rank's answers to every rank
                writes += (Region(p.xreg, p.rows * p.k * 2, writers, space=Space.PEER),)
                if (
                    p.scap and not p.out_arg
                ):  # (requests up to scap: the output from here, else the merge's)
                    writes += tuple(
                        self.out.regions(p.rows, writers + (f"{self.name}.m",))
                    )
            elif final and not p.out_arg:
                writes += tuple(self.out.regions(p.rows, writers))
            args = []
            if p.bound:
                args.append(p.bound)
            words = (
                -(
                    -topk.lds_words(
                        p.k, p.index_bits, p.u_bits, ranked=False, in_order=True
                    )
                    // 4
                )
                * 4
            )
            if outputs:
                words += p.lift_words  # the row's unit ids after select's words
            out.append(
                Stage(
                    name,
                    p.vrows * p.lists(level),
                    partial(_level, p=p, level=level),
                    lds_words=words,
                    writes=writes,
                    args=tuple(args),
                    key=(p,) + (level,),
                    live_tasks=(
                        partial(_live_tasks, p=p, level=level) if p.live else None
                    ),
                )
            )
        if p.split:
            mname = f"{self.name}.m"
            writes = (
                () if p.out_arg or p.scap else tuple(self.out.regions(p.rows, (mname,)))
            )
            words = (
                -(
                    -topk.lds_words(
                        p.k, p.merge_bits, p.u_bits, ranked=False, in_order=True
                    )
                    // 4
                )
                * 4
            )
            out.append(Stage(mname, p.rows, partial(_merge_if, p=p), lds_words=words, writes=writes,
                             args=((p.out,) if p.out_arg else ()) + ((p.bound,) if p.scap else ()),
                             key=(p,) + ("merge",),
                             live_tasks=partial(_merge_live, p=p) if p.scap else None))  # fmt: skip
        return out


# ------------------------------------------------------------------ device side


@traced
def _poll_quad(a, pair, tag, live, pairs_per_value):
    """4 values at pair ``pair`` (F32 channel: 1 pair a value; a list: 2, value then index) -> the
    value words and (lists) the index words; ``live`` False: zeros, no poll."""
    v0, v1, v2, v3 = fx.Int32(0), fx.Int32(0), fx.Int32(0), fx.Int32(0)
    i0, i1, i2, i3 = fx.Int32(-1), fx.Int32(-1), fx.Int32(-1), fx.Int32(-1)
    if live:
        if const_expr(pairs_per_value == 1):
            got = mb.poll([(a, pair, 2), (a, pair + 2, 2)], tag)
            v0, v1, v2, v3 = got[0][0], got[0][1], got[1][0], got[1][1]
        else:
            got = mb.poll([(a, pair + 2 * e, 2) for e in range(4)], tag)
            v0, v1, v2, v3 = got[0][0], got[1][0], got[2][0], got[3][0]
            i0, i1, i2, i3 = got[0][1], got[1][1], got[2][1], got[3][1]
    return [v0, v1, v2, v3], [i0, i1, i2, i3]


@traced
def _put(k, kk, pos, n_out, index_bits, u_bits, store):
    """Output position ``pos`` (< ``kk``): select's slot ``pos``, (padding, -1) past ``n_out``, to ``store``."""
    if pos < kk:
        u, gidx = topk.slot(k.lds, pos, n_out, kk, index_bits, u_bits, in_order=True)
        store(pos, (pos < n_out).select(key_bits(u), fx.Int32(NEG_INF_BITS)), gidx)


def _max_bound(k, p):
    """The rows' largest bound (clamped to 0 .. n; split: of this rank's share): what the live tasks follow."""
    m = fx.Int32(0)
    if p.split:
        for b in range(p.rows // p.spec):
            m = fx.max(
                m,
                _share(k, p, uniform(ld_i32(k.args[p.bound], b * p.spec + p.spec - 1)))[
                    1
                ],
            )
        return m
    for r in range(p.rows):
        m = fx.max(m, uniform(ld_i32(k.args[p.bound], r)))
    return fx.min(m, fx.Int32(p.n))


def _share(k, p, last, bound=None):
    """Split: (this rank's first global index r Q, its count of ``bound``'s values) for a request whose last row's
    bound is ``last`` (``bound`` None: ``last``'s own); ``scap``: a request up to it is whole on every rank (0, b).
    """
    q = (
        (fx.min(last, fx.Int32(p.gn)) + (p.split * p.unit - 1))
        // (p.split * p.unit)
        * p.unit
    )
    first = k.rank * q
    b = last if bound is None else bound
    n = fx.max(fx.min(fx.min(b, fx.Int32(p.gn)) - first, q), fx.Int32(0))
    if p.scap:
        whole = last <= p.scap
        first, n = whole.select(fx.Int32(0), first), whole.select(
            fx.max(fx.min(b, fx.Int32(p.n)), fx.Int32(0)), n
        )
    return first, n


def _direct(k, p, row):
    """Split with ``scap``: whether row's request is ranked whole here (its answer the output, no exchange)."""
    return (
        uniform(ld_i32(k.args[p.bound], row // p.spec * p.spec + p.spec - 1)) <= p.scap
    )


def _merge_live(k, p):
    """The merge's live tasks: every row if a request is past ``scap``, else none."""
    on = fx.Int32(0)
    for b in range(p.rows // p.spec):
        last = uniform(ld_i32(k.args[p.bound], b * p.spec + p.spec - 1))
        on = (last > p.scap).select(fx.Int32(1), on)
    return on * p.rows


@traced
def _merge_body_if(k, t, p, live):
    if live:
        _merge(k, t, p)


def _merge_if(k, t, p):
    if p.scap:
        _merge_body_if(
            k,
            t,
            p,
            uniform(ld_i32(k.args[p.bound], t // p.spec * p.spec + p.spec - 1))
            > p.scap,
        )


def _live_lists(k, p, level):
    """Lists of ``level`` the live tasks cover a row: level 0 one at least; a level above the first with one list
    of the largest bound none (every row's answer is below it)."""
    m = _max_bound(k, p)
    n = lists(m, level, p.seg, p.k)
    if level == 0:
        return fx.max(n, fx.Int32(1))
    return (lists(m, level - 1, p.seg, p.k) > 1).select(n, fx.Int32(0))


def _live_tasks(k, p, level):
    return fx.Int32(p.vrows) * _live_lists(k, p, level)


def _level(k, t, p, level):
    nl = (
        fx.max(k.live // p.vrows, fx.Int32(1)) if p.live else p.lists(level)
    )  # (live: ``_live_tasks``' lists)
    row, j = t // nl, t % nl
    brow = row % p.rows if p.groups > 1 else row  # (groups: the bound's row)
    b = fx.Int32(p.n)
    first = None
    if p.split:  # this rank's share of the row: its local bound, its first global index
        last = uniform(ld_i32(k.args[p.bound], brow // p.spec * p.spec + p.spec - 1))
        first, b = _share(k, p, last, uniform(ld_i32(k.args[p.bound], brow)))
    elif p.bound:
        b = fx.max(
            fx.min(uniform(ld_i32(k.args[p.bound], row)), fx.Int32(p.n)), fx.Int32(0)
        )
    n_in = list_inputs(b, level, j, p.seg, p.k)
    # a bounded tree: a row whose bound fits one segment has its answer in level 0's first list, which that task
    # writes as the output (lifted); the levels above skip the row, and lists past the row's bound are not ranked
    # (no level reads them: ``list_inputs``). The same select on the same values: the same ids.
    single, multi = segments(b, p.seg) <= 1, segments(b, p.seg) > 1
    if (
        p.live
    ):  # the answer at the first level with one list of the row (each level keeps every winner); above none
        if level == p.depth:  # one list a row (no list region): the answer
            final_now, not_final, run = True, False, n_in >= 0
        else:
            final_now, not_final = (lists(b, level, p.seg, p.k) <= 1) & (j == 0), lists(
                b, level, p.seg, p.k
            ) > 1
            run = final_now | (n_in > 0)
        if level:
            run = run & (lists(b, level - 1, p.seg, p.k) > 1)
        _body_if(k, p, level, row, j, n_in, final_now, not_final, run, first)
        return
    if level == 0:
        final_now, not_final = single & (j == 0), multi | (j != 0)
        run = final_now | (n_in > 0)
    else:
        final_now, not_final, run = True, False, multi
    _body_if(k, p, level, row, j, n_in, final_now, not_final, run, first)


@traced
def _body_if(k, p, level, row, j, n_in, final_now, not_final, run, first=None):
    if run:
        _body(k, p, level, row, j, n_in, n_in > 0, (final_now, not_final), first)


@traced
def _lift_if(ok, k, p, row, lift_at):
    if ok:
        _lift_load(k, p, row, lift_at)


@traced
def _store_if(ok, fn, pos, bits, gidx):
    if ok:
        fn(pos, bits, gidx)


def _body(k, p, level, row, j, n_in, live, final, first=None):
    """Rank list ``j`` of ``level`` for ``row`` (``n_in`` values); ``final``: a bool (write the output, lifted, instead
    of the level's list), or at a bounded tree's level 0 the traced pair (output, list).
    """
    final_now, not_final = final if isinstance(final, tuple) else (final, not final)
    per = p.per_thread(level)
    nl = p.lists(level)
    base = k.tid * per
    xdst = None  # (xhoist) every rank's answer slot, addressed before the keys' loads (above level 0: a split row's
    if (
        p.xhoist and final_now is not False and (level or not p.depth)
    ):  # level 0 answers only short, direct rows)
        xdst = [peer.to_rank(k, p.xreg, dst) for dst in range(p.split)]
    # quads past the last value's re-read it: in the row, never counted
    last = fx.max(n_in - 1, 0) & ~3
    us, idxs, valid = [], [], []
    for q in range(per // 4):
        i = fx.min(base + 4 * q, last)
        vals, gid = _load(k, p, level, row, j, i, live)
        for e in range(4):
            key = order_key(vals[e])
            us.append(key & fx.Int32(-65536) if p.bf16 else key)
            idxs.append(j * p.seg + i + e if level == 0 else gid[e])
            valid.append(base + (4 * q + e) < n_in)
    lift_at = (
        -(
            -topk.lds_words(p.k, p.index_bits, p.u_bits, ranked=False, in_order=True)
            // 4
        )
        * 4
    )
    lifts = p.lift and final_now is not False
    if (
        lifts
    ):  # the row's unit ids into LDS after select's words (select's barriers order them)
        if isinstance(final_now, bool):
            _lift_load(k, p, row, lift_at)
        else:
            _lift_if(final_now, k, p, row, lift_at)
    n_out = topk.select(
        k.lds,
        k.tid,
        k.lane,
        k.wave,
        us,
        idxs,
        p.k,
        p.index_bits,
        p.u_bits,
        valid,
        in_order=True,
    )

    def put_list(pos, bits, gidx):
        mb.put2(
            k.mb[p.region(level)], 2 * ((row * nl + j) * p.k + pos), bits, gidx, k.tag
        )

    def put_out(pos, bits, gidx):
        if lifts:  # column -> unit id x lift_unit + the column's place in its unit
            unit = fx.ptr_load(
                k.lds + (lift_at + fx.max(gidx, fx.Int32(0)) // p.lift_unit)
            )
            gidx = (gidx >= 0).select(unit * p.lift_unit + gidx % p.lift_unit, gidx)
        if (
            p.split
        ):  # (split) this rank's answer, global indices, to every rank's slot of it (scap: or the output)
            gidx = (gidx >= 0).select(gidx + first, fx.Int32(-1))
            if p.scap:
                _split_put_if(k, p, row, pos, bits, gidx, _direct(k, p, row), xdst)
        else:
            mb.put(k.mb[p.out], row * p.k + pos, gidx, k.tag)

    if isinstance(final_now, bool):
        store = put_out if final_now else put_list
    else:

        def store(pos, bits, gidx):
            _store_if(final_now, put_out, pos, bits, gidx)
            _store_if(not_final, put_list, pos, bits, gidx)

    for r in range(-(-p.k // THREADS)):
        _put(k, p.k, k.tid + THREADS * r, n_out, p.index_bits, p.u_bits, store)


@traced
def _split_put_if(k, p, row, pos, bits, gidx, direct, xdst=None):
    if direct:
        _out_put(k, p, row, pos, gidx)
    else:
        _split_put(k, p, row, pos, bits, gidx, xdst)


def _split_put(k, p, row, pos, bits, gidx, xdst):
    for dst in range(p.split):
        a = xdst[dst] if xdst else peer.to_rank(k, p.xreg, dst)
        mb.put2(a, 2 * (row * p.k + pos), bits, gidx, k.tag)


def _out_put(k, p, row, pos, gidx):
    mb.put(k.mb[p.out], row * p.k + pos, gidx, k.tag)


def _merge(k, t, p):
    _merge_cta(k, t, p)


def _merge_cta(k, t, p):
    """Split: row t's ``split`` rank answers (k (value, global index) pairs each, rank order = index order; padding
    -1) ranked into the output, as one TopK of every value would."""
    n_all = p.split * p.k
    exact = (
        n_all % (4 * THREADS) == 0
    )  # every thread whole quads; else (few answers: k 16) 4 a thread, the rest none
    per = n_all // THREADS if exact else 4
    base = k.tid * per
    us, idxs, valid = [], [], []
    for q in range(per // 4):
        i = base + 4 * q
        live = True if exact else i < n_all
        if not exact:
            i = fx.min(i, fx.Int32(n_all - 4))
        src, e = i // p.k, i % p.k  # (quads never straddle a rank: k % 4 == 0)
        vals, gid = _poll_quad(
            peer.from_rank(k, p.xreg, src), 2 * (t * p.k + e), k.tag, live, 2
        )
        for v, g in zip(vals, gid, strict=True):
            key = order_key(v)
            us.append(key & fx.Int32(-65536) if p.bf16 else key)
            idxs.append(g)
            valid.append(g >= 0)
    n_out = topk.select(
        k.lds,
        k.tid,
        k.lane,
        k.wave,
        us,
        idxs,
        p.k,
        p.merge_bits,
        p.u_bits,
        valid,
        in_order=True,
    )

    def store(pos, bits, gidx):
        mb.put(k.mb[p.out], t * p.k + pos, gidx, k.tag)

    for r in range(-(-p.k // THREADS)):
        _put(k, p.k, k.tid + THREADS * r, n_out, p.merge_bits, p.u_bits, store)


def _lift_load(k, p, row, lift_at):
    per = p.lift_words // THREADS
    got = mb.poll(
        [
            (k.mb[p.lift], row * p.lift_words + per * k.tid + 2 * e, 2)
            for e in range(per // 2)
        ],
        k.tag,
    )
    ids = [w for pr in got for w in pr]
    for e in range(0, per, 4):
        fx.ptr_store(
            fx.Vector.from_elements(ids[e : e + 4], fx.Int32),
            k.lds + (lift_at + per * k.tid + e),
        )


def _load(k, p, level, row, j, i, live):
    """The 4 values (fp32 bits) at positions i .. i + 3 of list j, and (level > 0) their global indices."""
    if level == 0:
        col = j * p.seg + i
        vals, _ = _poll_quad(k.mb[p.x], row * p.n + col, k.tag, live, 1)
        return vals, None
    fan = p.seg // p.k
    src = (row * p.lists(level - 1) + j * fan + i // p.k) * p.k + i % p.k
    return _poll_quad(k.mb[p.region(level - 1)], 2 * src, k.tag, live, 2)

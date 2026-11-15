# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``ArgmaxHead``: a greedy LM head. Per row of the step, the index of the largest logit of ``x @ W^T`` over a
vocab-sharded bf16 weight (exact bf16 products, fp32 sums), the argmax taken across the TP ranks exactly; no logits
leave the CTA that computed them.

Per row ``r`` of the step, with ``W`` this rank's ``n`` rows of the vocab (rank ``q`` holds ids ``q n .. q n + n - 1``,
the reference's ``ParallelHead`` split)::

    logit[r, v] = sum_k x[r, k] W[v, k]                        (fp32 sums of exact bf16 products)
    id[r]       = argmax_v key(logit[r, v]), ties to the lowest id over the whole vocab

``key`` (``key=``): "bf16" compares the logits rounded to bf16 (RNE), as a bf16 GEMM's output is compared by
``torch.argmax`` / ATOM's ``topk_select(tie="low")`` (the greedy sampler); "f32" compares the fp32 sums (the reference
model's fp32 head). -0 counts as +0 (equal keys, the lowest id wins), as in ``torch.argmax``.

**Inputs.** ``x``: a tagged BF16 channel of width K (the final-normed rows). The weight: arg ``<name>.w``, bf16,
``layout`` "packed" (``pack``: 16-row x 32-K units of 1 KiB, a lane's 16 B at 16 l: lane l holds row l % 16, K
8 (l // 16) .. + 7, the bf16 MFMA's A operand; units ordered [task][K step][the task's 8 row groups], so at each K
step a CTA's 8 waves read 8 KiB that are contiguous), "wave" (the same units ordered [task][row group][K step]: each
wave's 160 KiB contiguous), "kmajor" (ordered [K step][task][row group]: at each K step the grid reads one contiguous
range) or "rows" (the checkpoint's [n, K] row-major tensor as it is: no copy, a lane reads 16 B of one row).

**Outputs.** Plain args ``<name>.ids`` int32 [rows] (the global id) and, with ``vals``, ``<name>.vals`` f32 [rows]
(the winning key's value: the bf16-rounded logit for "bf16"); with ``out``, an F32 channel of width 1 carrying the
id's bits (a later stage of the launch, e.g. a verdict).

**Stages.**

1. ``<name>``: ``ceil(n / 128)`` tasks, a task 8 row groups of 16 vocab rows over the whole K, one a wave. A task
   issues its first ``lead`` weight units (nontemporal), polls x (an attempt), lands x [rows, K] in LDS and issues
   the rest of its ``prefetch`` (``late``; else before the landing), then each wave runs ``K / 32`` bf16 MFMAs
   (``mfma_f32_16x16x32_bf16``: A = 16 vocab rows, B = the step's rows as N), loading ``prefetch`` units ahead.
   Epilogue: per row, each lane's 4 logits in id order, then lanes ^ 16, ^ 32 (the 16 rows of the wave), then the
   8 waves in order -> the task's best (key, id) per row, put as two tagged pairs (key, global id) into every rank's
   ``<name>.top`` (TP: a peer region, [source rank][task][row]; TP1: scratch). Over 16 rows an MFMA per 16-row
   chunk on the same weight unit (a row's logits are those of one chunk: the same bits at any row count); x that
   does not fit the LDS lands in ``kparts`` K slices, one at a time (the next once every wave has streamed the K
   steps of the last: the same products and the same K order on the accumulator).
2. ``<name>.pick``: a task a row: polls the ``tp x tasks`` entries of its row, the best (largest key, then lowest
   id) by a CTA reduction -> the outputs. The comparison is a total order, so the order of the reduction does not
   matter and every rank, polling the same entries, picks the same id.

**MXFP4 weights** (``weight_format="mxfp4"``: a quantized head, the blog megakernel's precision): ``x`` is a tagged
MXFP8 channel of width K (a multiple of 128: the final norm's MXFP8 output), the weight MXFP4 (``pack_weight`` of an
MXFP4 weight [n, K], ``packed`` / ``codes``: ``gemv.pack_weight``'s units, args ``<name>.w`` / ``<name>.ws``, rows
padded to whole tasks), each wave's row group streamed over K in steps of 128 (``mfma_scale_f32_16x16x128_f8f6f4``,
its scale code per lane from ``<name>.ws``), the same top-1 epilogue and pick.

Numerics contract: each logit is the MFMA's 32-element sums added over K steps 0, 1, .. in order, on the fp32
accumulator (no split K; MXFP4: the 128-element scaled MFMA's). A bf16 GEMM elsewhere (hipBLASLt) sums in another
order, so a logit can round to a different bf16 value: ids differ from such a reference only on near-ties (two logits
within about an ulp).

**Sampling** (``sample``; ``key`` "bf16", no ``out``, a rank's ``n`` <= 65536: past 32768 the scans below run in two
slices, ``SCAN_SLICES``): per row, temperature 0 keeps the greedy pick above; temperature T > 0 draws one id from
softmax(b / T) restricted to the top-p nucleus, b the bf16 logits.
The nucleus is aiter's (FlashInfer's) rule, the one ATOM's sampler applies: id v is kept when the probability of the
ids strictly more likely than v is below top_p (equal logits are all kept or all dropped). As b is bf16 and p a
monotone function of it, the nucleus is {v : key16(b_v) >= tau}, key16 the 16-bit order key, for one tau a row.

- Arg ``<name>.par`` int32 [2 rows + 2]: per row (T, top_p) as f32 bits (T 0: greedy), then a 64-bit seed.
- The dot's epilogue also puts the task's bf16 logits into the rank's scratch ``<name>.lg`` [rows][n / 2] pairs.
- The pick's task of a sampled row then, with M the row's max logit (the greedy pick's, the same on every rank):
  masses m_v = round(exp((b_v - M) / T) 2^36) (fixed point: a row's sum stays below 2^53, so every sum below is
  exact, in any order, and the same on every rank); (1) scans the rank's row: masses by key16 >> 8 (LDS atomics into
  ``COPIES`` copies), into every rank's ``<name>.h1``; (2) gathers the ranks' histograms (every thread a bin and half
  the ranks): Z, and B1, the bin where the mass of the bins above first reaches top_p Z (none: top_p 1, every id
  kept); (3) scans again: the masses of B1's 256 keys, per key the best (score, id), score = m / E with E = -log2 U
  (the exponential race: as Gumbel-max, with one log a candidate), and the best of the ids above B1, into every
  rank's ``<name>.h2``; (4) gathers those: a key of B1 is kept when the mass above B1 plus the mass of B1's keys
  above it is below top_p Z; the id = the best (score, id) over the kept keys and the ranks' bests above B1 (ties:
  the lower id): the race's winner over the nucleus, a draw of the renormalized distribution. ``vals``: the
  smallest bf16 logit kept (-inf: every id).
- U: Philox2x32-10, key = seed word 0, counter (id / 2 | row << 16, launch tag ^ seed word 1), one 32-bit word an id
  (24 bits used). The launch tag is new every step and the same on every rank: every rank draws the same U, and
  every rank's id is the same. Only the ids in B1 or above draw (and score) at all.

**Race rows** (``race``, with ``sample``): a row with T > 0 and top_p >= 1 (every id kept) is drawn in the dot's
epilogue instead: each id's key is its race score s_v = b_v log2(e) / T - log2(E_v), E_v = -log2 U_v (U_v the same
Philox word as above, as (k + 1/2) 2^-24 exactly: ``_exp_var``), whose argmax is the Gumbel-max draw from
softmax(b / T) over the whole vocab. The task's best (key, id) per row then takes the greedy path (the waves,
``<name>.top``, the pick's reduction: ties to the lower id); its ``<name>.lg`` pairs are not written, the pick skips
the scans, and ``vals`` is -inf (every id kept, as the scans give at top_p 1). The dot reads ``<name>.par`` into its
LDS with x; rows with T 0 or top_p < 1 keep the greedy keys and the scans as above. ``race="check"``: the same, with
every row's ``<name>.lg`` pairs written (a check's host race over the head's own logits).

``race="only"``: every row with T > 0 races, whatever its top_p (not read: the caller serves top_p 1 rows only), and
no scans are built (no ``<name>.lg``, ``<name>.h1`` / ``.h2``), so neither of their limits applies: a vocab up to
2^18 ids (over 2^17 the Philox counter is (id / 2 | row << 17, ...): ids / 2 take 17 bits; below, as above).

**Balanced tasks** (``balance`` S > 0; layout "wave", rows <= 16, x whole in the LDS): a task a CTA (``tasks`` = the
CUs), CTA c the row groups [c G / tasks, (c + 1) G / tasks) of the rank's G, each split over K into S parts of K / S
steps, the CTA's parts spread evenly over its 8 waves in (row group, part) order. A wave streams its parts (each into
a fresh fp32 accumulator, K steps in order) and stores each part's sums into the LDS; then wave w sums row groups w,
w + 8, .. of the CTA over their parts in part order and takes their top-1 (the same epilogue, the same keys). Where
G / 8 row groups of a rank do not fill whole rounds of the grid (e.g. 3126 over 256 CTAs, 1.53 tasks a CTA
with whole row groups a wave) a wave's work is at most ceil(ceil(G / tasks) S / 8) parts instead of two tasks. The
logits are the same products summed in another order (S partial sums added in order): another rounding than the
unbalanced head's (a bf16 key can differ on a near-tie)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial

import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from ...device.gfx950 import buffer_ops as bo
from ...device.gfx950 import gemv
from ...device.gfx950 import mailbox as mb
from ...device.gfx950.dpp_utils import update_dpp_i32
from ...device.gfx950.gemv import OOB
from ...device.gfx950.ops import (
    CM_NT,
    bf16_pair,
    bf16_round,
    fresh,
    hw_exp2,
    hw_log2,
    hw_rcp,
    ld_i32,
    mfma_bf16,
    readlane,
    rsrc,
    traced,
    uniform,
    xshfl,
)
from ...device.gfx950.peer import from_rank, to_rank
from ...device.gfx950.topk import INT_MIN, key_bits, order_key, ult
from ...plan.program import Region, Scenario, Stage
from ...plan.trace import Space
from ...target import MI355X
from ..base import Block, Channel, Fmt
from .linear import _land_batch, _try_batch

THREADS, WAVES = MI355X.threads, MI355X.waves
RG = 16  # vocab rows a row group (the MFMA's M)
KSTEP = 32  # K a unit (the MFMA's K)
UNIT_WORDS = 256  # 32-bit words a unit (16 x 32 bf16 = 1 KiB)
KEYS = ("bf16", "f32")
LAYOUTS = ("packed", "kmajor", "wave", "rows")
WFMTS = ("bf16", "mxfp4")
MX_KSTEP = 128  # K an MXFP4 unit (the scaled MFMA's K)


@dataclass(frozen=True)
class _Plan:
    name: str
    rows: int
    k: int
    n: int  # this rank's vocab rows
    tp: int
    x: str
    key: str
    layout: str
    prefetch: int
    vals: bool
    out: str  # "" without an output channel
    lead: int = 4  # weight units a wave issues before x's poll attempt
    late: bool = False  # the rest of the prefetch after x lands (else before)
    sample: bool = (
        False  # per-row temperature / top-p sampling (``sample``: the module docstring's "Sampling")
    )
    wfmt: str = (
        "bf16"  # the weight's format ("mxfp4": MXFP8 x, the module docstring's "MXFP4 weights")
    )
    kparts: int = (
        1  # x lands in this many K slices, one at a time (``ArgmaxHead.plan``: the fewest that fit the LDS)
    )
    race: bool = (
        False  # ``sample``: rows with T > 0 and top_p >= 1 drawn in the dot's epilogue (module docstring)
    )
    race_lg: bool = (
        False  # ``race="check"``: their ``<name>.lg`` pairs written too (a check's reference)
    )
    race_only: bool = (
        False  # ``race="only"``: every T > 0 row races, no scans (module docstring)
    )
    balance: int = (
        0  # K parts a row group (``balance``: balanced tasks, module docstring); 0: a task 8 row groups
    )

    probe = False  # ``linear._land_batch``'s probe build: off

    @property
    def mx(self) -> bool:
        return self.wfmt == "mxfp4"

    @property
    def n_rg(self) -> int:
        return self.n // RG

    @property
    def tasks(self) -> int:
        return min(MI355X.cus, self.n_rg) if self.balance else -(-self.n_rg // WAVES)

    @property
    def bal_g(self) -> int:  # ``balance``: the most row groups a CTA
        return -(-self.n_rg // self.tasks)

    @property
    def bal_base(
        self,
    ) -> int:  # ``balance``: LDS word of the parts' sums (a part 64 lanes x 4 floats)
        return self.rpar + (2 * self.rows + 2 if self.race else 0)

    @property
    def steps(self) -> int:
        return self.k // (MX_KSTEP if self.mx else KSTEP)

    @property
    def pitch(
        self,
    ) -> (
        int
    ):  # LDS words a row of x (its K slice: ``kparts``; +4: the 16 rows of an operand load on
        # distinct banks)
        return self.k // 4 + 4 if self.mx else self.k // self.kparts // 2 + 4

    @property
    def spitch(
        self,
    ) -> int:  # MXFP8 x: LDS words a row of its scale codes (odd: bank spread)
        return self.k // 32 + 1

    @property
    def xs_base(self) -> int:  # MXFP8 x: LDS word of its codes, after the FP8 words
        return self.rows * self.pitch

    @property
    def kse(
        self,
    ) -> int:  # ``linear._x_batches``' slice width and channel width: all of K
        return self.k

    @property
    def xk(self) -> int:
        return self.k

    @property
    def psteps(self) -> int:  # K steps a slice of x
        return self.steps // self.kparts

    @property
    def rc(self) -> int:  # 16-row chunks (the MFMA's N)
        return -(-self.rows // RG)

    @property
    def red(self) -> int:  # LDS word of the waves' (key, id) per row, after x
        return (
            self.xs_base + self.rows * self.spitch
            if self.mx
            else self.rows * self.pitch
        )

    @property
    def rpar(
        self,
    ) -> int:  # ``race``: LDS word of ``<name>.par``'s copy, after the waves' (key, id)
        return self.red + WAVES * self.rows * 2

    @property
    def lds_words(self) -> int:
        return self.bal_base + (self.bal_g * self.balance * 256 if self.balance else 0)

    @property
    def cshift(
        self,
    ) -> (
        int
    ):  # the row's shift in the Philox counter's first word (module docstring: ids / 2 below it)
        return 16 if self.n * self.tp <= 2**17 else 17

    @property
    def top(self) -> str:
        return f"{self.name}.top"

    @property
    def top_pairs(self) -> int:  # a source rank's slot: (key, id) pairs [task][row]
        return self.tasks * self.rows * 2

    @property
    def n2(self) -> int:  # pairs a row of ``<name>.lg`` (two bf16 logits a pair)
        return self.n // 2

    @property
    def quads(
        self,
    ) -> int:  # 16-byte poll specs a row of ``<name>.lg`` (two pairs: four logits)
        return self.n // 4

    @property
    def scan_slices(
        self,
    ) -> (
        int
    ):  # the sampling scans' slices of at most SCAN_SPECS specs a thread (1 to 32768 ids)
        tasks = -(-self.quads // THREADS)
        return -(-tasks // SCAN_SPECS)

    @property
    def lg(self) -> str:
        return f"{self.name}.lg"

    @property
    def h1(self) -> str:
        return f"{self.name}.h1"

    @property
    def h2(self) -> str:
        return f"{self.name}.h2"

    @property
    def w_bytes(self) -> int:
        rows = self.n if self.layout == "rows" else -(-self.n_rg // WAVES) * WAVES * RG
        return rows * self.k // 2 if self.mx else rows * self.k * 2

    @property
    def s_bytes(self) -> int:  # MXFP4: the scale codes, one a row and 32 K
        return self.tasks * WAVES * RG * self.k // 32


class ArgmaxHead(Block):
    """Greedy LM head with a top-1 epilogue and the exact TP argmax (module docstring). ``n``: this rank's vocab
    rows (a multiple of 16)."""

    def __init__(self, name: str, x: Channel, n: int, key: str = "bf16", layout: str = "packed",
                 prefetch: int = 16, vals: bool = True, out: Channel | None = None, lead: int = 4,
                 late: bool = False, sample: bool = False, weight_format: str = "bf16",
                 race: bool | str = False, balance: int = 0, kparts: int = 0):  # fmt: skip
        if weight_format not in WFMTS:
            raise ValueError(f"ArgmaxHead {name}: weight_format in {WFMTS}")
        mx = weight_format == "mxfp4"
        if mx and (
            x.fmt is not Fmt.MXFP8
            or x.flagged
            or x.width % MX_KSTEP
            or layout != "packed"
            or sample
        ):
            raise ValueError(f"ArgmaxHead {name}: mxfp4 reads a tagged MXFP8 channel of width a multiple of "
                             f"{MX_KSTEP}, layout 'packed', no sampling")  # fmt: skip
        if not mx and (x.fmt is not Fmt.BF16 or x.flagged or x.width % (2 * KSTEP)):
            raise ValueError(
                f"ArgmaxHead {name}: x is a tagged BF16 channel, width a multiple of {2 * KSTEP}"
            )
        if n % RG or n >= 2**31 // 8:
            raise ValueError(f"ArgmaxHead {name}: n a multiple of {RG}")
        if key not in KEYS or layout not in LAYOUTS:
            raise ValueError(f"ArgmaxHead {name}: key in {KEYS}, layout in {LAYOUTS}")
        kstep = MX_KSTEP if mx else KSTEP
        if not 0 <= lead <= prefetch <= x.width // kstep:
            raise ValueError(f"ArgmaxHead {name}: 0 <= lead <= prefetch <= K / {kstep}")
        if out is not None and (
            out.fmt is not Fmt.F32 or out.width != 1 or out.flagged
        ):
            raise ValueError(
                f"ArgmaxHead {name}: out is a tagged F32 channel of width 1 (the id's bits)"
            )
        if sample and (key != "bf16" or out is not None or n % 4):
            raise ValueError(
                f"ArgmaxHead {name}: sample needs key 'bf16', no out channel, n a multiple of 4"
            )
        if race not in (False, True, "check", "only") or (race and not sample):
            raise ValueError(
                f"ArgmaxHead {name}: race (True / 'check' / 'only') is a sampling mode (sample=True)"
            )
        self.name, self.x, self.n, self.key, self.layout = name, x, n, key, layout
        self.prefetch, self.vals, self.out, self.lead, self.late = (
            prefetch,
            vals,
            out,
            lead,
            late,
        )
        if balance and (
            mx
            or layout != "wave"
            or (sample and race != "only")
            or x.width // KSTEP % balance
        ):
            raise ValueError(f"ArgmaxHead {name}: balance (K parts) with bf16 weights in layout 'wave', K / 32 a "
                             f"multiple of it, greedy or race 'only'")  # fmt: skip
        self.sample, self.weight_format, self.race, self.balance = (
            sample,
            weight_format,
            race,
            balance,
        )
        # x's K slices at least this many (bf16 weights; 0: the fewest that fit the LDS): smaller slices land with
        # fewer loads in flight beside the prefetch (the same products and K order: the same bits)
        if kparts < 0 or (kparts and mx):
            raise ValueError(f"ArgmaxHead {name}: kparts >= 0, bf16 weights only")
        self.kparts = kparts

    def plan(self, sc: Scenario) -> _Plan:
        p = _Plan(self.name, sc.rows, self.x.width, self.n, sc.tp, self.x.name, self.key, self.layout,
                  self.prefetch, self.vals, self.out.name if self.out is not None else "", self.lead,
                  self.late, self.sample, self.weight_format, race=bool(self.race),
                  race_lg=self.race == "check", race_only=self.race == "only", balance=self.balance)  # fmt: skip
        if (
            not p.mx
        ):  # x in K slices where it does not fit whole (the bf16 weights' path), ``kparts`` at least
            if self.kparts > 1:
                if p.steps % self.kparts:
                    raise ValueError(
                        f"ArgmaxHead {self.name}: kparts {self.kparts} does not divide {p.steps} K steps"
                    )
                p = replace(p, kparts=self.kparts)
            for kp in (q for q in range(2, 9) if p.steps % q == 0 and q > p.kparts):
                if p.lds_words * 4 <= MI355X.lds_bytes:
                    break
                p = replace(p, kparts=kp)
        if p.balance and (p.kparts > 1 or p.rc > 1):
            raise ValueError(
                f"ArgmaxHead {self.name}: balance: x of {sc.rows} rows whole in the LDS, at most 16 rows"
            )
        if p.race_only and p.n * p.tp > 2**18:
            raise ValueError(
                f"ArgmaxHead {self.name}: race 'only': a vocab below 2^18 ids (the sampler's counter)"
            )
        if (
            p.sample
            and not p.race_only
            and (p.n * p.tp > 2**17 or p.scan_slices > SCAN_SLICES)
        ):
            raise ValueError(f"ArgmaxHead {self.name}: sample: a vocab below 2^17 ids (the sampler's counter), at most "
                             f"{4 * SCAN_SLICES * SCAN_SPECS * THREADS} a rank (the scans' slices)")  # fmt: skip
        if p.lds_words * 4 > MI355X.lds_bytes:
            raise ValueError(
                f"ArgmaxHead {self.name}: x of {sc.rows} rows does not fit the LDS"
            )
        if (
            p.race and 2 * sc.rows + 2 > THREADS
        ):  # ``_race_par``: a word of ``<name>.par`` a thread
            raise ValueError(f"ArgmaxHead {self.name}: race reads its 2 rows + 2 parameter words one a thread, at most "
                             f"{THREADS}")  # fmt: skip
        return p

    def stages(self, sc: Scenario) -> list[Stage]:
        p = self.plan(sc)
        space = Space.PEER if p.tp > 1 else Space.SCRATCH
        top = Region(p.top, p.top_pairs, (self.name,), space=space)
        pname = f"{self.name}.pick"
        outs = (f"{self.name}.ids",) + ((f"{self.name}.vals",) if p.vals else ())
        writes = (
            tuple(self.out.regions(p.rows, (pname,))) if self.out is not None else ()
        )
        dot_writes, pick_lds = (top,), WAVES * 2
        if p.sample:
            dot_writes += (
                Region(p.lg, p.rows * p.n2, (self.name,), space=Space.SCRATCH),
            )
            writes += (Region(p.h1, p.rows * H1_ROW, (pname,), space=space, exchange=True),
                       Region(p.h2, p.rows * H2_ROW, (pname,), space=space, exchange=True))  # fmt: skip
            outs += (f"{self.name}.par",)
            pick_lds = S_WORDS
        w_args = (f"{self.name}.w",) + ((f"{self.name}.ws",) if p.mx else ())
        w_args += (
            (f"{self.name}.par",) if p.race else ()
        )  # (race: the dot reads the rows' T / top_p and the seed)
        return [
            Stage(self.name, p.tasks, partial(_dot, p=p), lds_words=p.lds_words, writes=dot_writes,
                  args=w_args, key=(p, "dot")),
            Stage(pname, p.rows, partial(_pick, p=p), lds_words=pick_lds, writes=writes, args=outs,
                  key=(p, "pick")),
        ]  # fmt: skip

    def pack_weight(self, w) -> dict[str, torch.Tensor]:
        """Arg ``<name>.w`` from this rank's bf16 [n, K] rows (``pack``); MXFP4: ``<name>.w`` / ``<name>.ws`` from an
        MXFP4 weight (``packed`` [n, K / 2], ``codes`` [n, K / 32], K the channel's width: ``pack_mx``).
        """
        if self.weight_format == "mxfp4":
            wb, sb = pack_mx(w.packed, w.codes)
            return {f"{self.name}.w": wb, f"{self.name}.ws": sb}
        return {f"{self.name}.w": pack(w, self.layout)}


def pack(w: torch.Tensor, layout: str = "packed") -> torch.Tensor:
    """This rank's bf16 [n, K] head rows in ``layout`` (module docstring); "packed" pads to whole tasks (zero rows,
    never read)."""
    n, k = w.shape
    if w.dtype != torch.bfloat16 or n % RG or k % KSTEP:
        raise ValueError("pack: bf16 [n, K], n a multiple of 16, K of 32")
    if layout == "rows":
        return w.contiguous()
    per = WAVES * RG
    tasks = -(-n // per)
    if tasks * per != n:
        w = torch.cat([w, w.new_zeros(tasks * per - n, k)])
    # [task, row group, row, K step, lane group, 8] -> [task, K step, row group, lane group, row, 8] ("kmajor": K step
    # outermost: at each K step the whole grid reads one contiguous range)
    u = w.view(tasks, WAVES, RG, k // KSTEP, 4, 8)
    order = {"kmajor": (3, 0, 1, 4, 2, 5), "wave": (0, 1, 3, 4, 2, 5)}.get(
        layout, (0, 3, 1, 4, 2, 5)
    )
    u = u.permute(*order)
    return u.contiguous().view(-1, k)


def pack_mx(
    packed: torch.Tensor, codes: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """An MXFP4 head [n, K] (``packed`` uint8 [n, K / 2], ``codes`` [n, K / 32]) as ``gemv.pack_weight``'s units and
    codes, rows padded to whole tasks (zero weights, code 127: never read past n)."""
    n = packed.shape[0]
    per = WAVES * RG
    pad = -(-n // per) * per - n
    if pad:
        packed = torch.cat(
            [
                packed.view(torch.uint8),
                packed.new_zeros(pad, packed.shape[1]).view(torch.uint8),
            ]
        )
        codes = torch.cat([codes, codes.new_full((pad, codes.shape[1]), 127)])
    return gemv.pack_weight(
        packed.view(torch.uint8), codes.view(torch.uint8), gemv.MXFP4
    )


# ------------------------------------------------------------------ device


def _better(ka, ia, kb, ib):
    """(ka, ia) beats (kb, ib): a larger key (u32 order), or the same key and a lower id."""
    return ult(kb, ka) | ((ka == kb) & (ia < ib))


def _take(ka, ia, kb, ib):
    """The better of two (key, id) entries (a total order: associative, commutative)."""
    b = _better(kb, ib, ka, ia)
    return b.select(kb, ka), b.select(ib, ia)


def _key(p, v):
    """The u32 order key (in an Int32) of the fp32 logit ``v``."""
    bits = (bf16_round(v) if p.key == "bf16" else fx.Float32(v)).bitcast(fx.Int32)
    bits = (bits == fx.Int32(INT_MIN)).select(fx.Int32(0), bits)  # -0 as +0
    return order_key(bits)


def _issue(p, r_w, t, wave, rg, lane, valid, i):
    """Lane ``lane``'s 16 B of the wave's unit at K step ``i``, nontemporal (a row group past n: read as 0, no
    memory touched)."""
    if p.layout == "wave":
        off = ((t * WAVES + wave) * p.steps + i) * UNIT_WORDS + lane * 4
    off = valid.select(off, fx.Int32(OOB // 4))
    return fx.Vector(
        bo.buffer_load(r_w, off, vec_width=4, dtype=T.i32, cache_modifier=CM_NT)
    )


def _x_batches(k, p, tid, threads, part: int = 0):
    """x [rows, K] polled into LDS (row r at word r * pitch) by ``threads`` threads (``tid`` this one's index among
    them): [(specs, LDS word offsets)] of at most POLL_MAX 2-pair specs a thread. ``kparts`` > 1: K slice ``part``.
    """
    words_row = p.k // 2 // p.kparts
    specs_row = words_row // 2
    total = p.rows * specs_row
    a = k.mb[p.x]
    items = []
    for c in range(-(-total // threads)):
        idx = fx.min(
            fx.max(fx.Int32(c * threads) + tid, fx.Int32(0)), fx.Int32(total - 1)
        )
        r, w2 = idx // specs_row, idx % specs_row
        src = (
            r * (p.k // 2) + part * words_row + w2 * 2
            if p.kparts > 1
            else r * words_row + w2 * 2
        )
        items.append(((a, src, 2), r * p.pitch + w2 * 2))
    return [
        (
            [s for s, _ in items[b : b + mb.POLL_MAX]],
            [d for _, d in items[b : b + mb.POLL_MAX]],
        )
        for b in range(0, len(items), mb.POLL_MAX)
    ]


def _dot(k, t, p):
    lane, wave = k.lane, k.wave
    rg = t * WAVES + wave
    valid = rg < p.n_rg
    r_w = rsrc(k.args[f"{p.name}.w"], p.w_bytes)
    issue = partial(_issue, p, r_w, t, wave, rg, lane, valid)
    n_pre = min(p.prefetch, p.steps)
    lead = min(p.lead, n_pre)
    if const_expr(
        p.race
    ):  # issued ahead of the weights, into the LDS after the lead units (vmcnt is in order: the
        # store waits for it alone), so its register is free before x's batches and the prefetch (12 rows: held
        # across x's landing it spilled one VGPR)
        par = _race_par(k, p)
        rocdl.sched_barrier(0)
    # 1. weights first, x's attempt after ``lead`` units, then x lands; the rest of the prefetch before the
    # landing (``late`` False: in flight through x's wait, but a stale attempt's re-poll then waits for all of it,
    # vmcnt being in order) or after it (``late``: the re-poll waits for the lead units only)
    q = [issue(i) for i in range(lead)]
    if const_expr(p.race):
        rocdl.sched_barrier(0)
        _race_put(k, p, par)
    batches = _x_batches(k, p, k.tid, THREADS)
    rocdl.sched_barrier(
        0
    )  # keep the issue order: LLVM would sink the attempt below the weights
    tries = [_try_batch(specs) for specs, _ in batches]
    rocdl.sched_barrier(0)
    if not p.late:
        q += [issue(i) for i in range(lead, n_pre)]
        rocdl.sched_barrier(0)
    for (specs, dests), words in zip(batches, tries, strict=True):
        _land_batch(k, p, specs, dests, words)
    if p.late:
        rocdl.sched_barrier(0)
        q += [issue(i) for i in range(lead, n_pre)]
    gpu.barrier()
    _stream(k, p, t, rg, lane, valid, issue, q)
    gpu.barrier()
    _task_put(k, p, t, k.rank * p.n if p.tp > 1 else fx.Int32(0))
    gpu.barrier()  # the next task reuses the LDS


def _stream(k, p, t, rg, lane, valid, issue, q):
    """After x landed (a CTA barrier): per K step the load ``prefetch`` units ahead, then this unit's MFMA (B: x rows
    from LDS; over 16 rows one a 16-row chunk; ``kparts`` > 1: the next K slice of x lands between its steps); then
    the wave's top-1 per row into LDS."""
    n_pre = len(q)
    if const_expr(p.rc == 1 and p.kparts == 1):
        row = fx.min(lane % 16, fx.Int32(p.rows - 1))
        xrow = k.ldsf + row * p.pitch + (lane // 16) * 4
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for i in range(p.steps):
            rocdl.sched_barrier(0)
            if i + n_pre < p.steps:
                q.append(issue(i + n_pre))
            b = fx.ptr_load(
                xrow + i * (KSTEP // 2), result_type=fx.Vector.make_type(4, fx.Float32)
            )
            acc = mfma_bf16(q[i].bitcast(fx.BFloat16), b.bitcast(fx.BFloat16), acc)
        accs = [acc]
    else:
        xrows = [k.ldsf + fx.min(lane % 16 + RG * ch, fx.Int32(p.rows - 1)) * p.pitch + (lane // 16) * 4
                 for ch in range(p.rc)]  # fmt: skip
        accs = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(p.rc)]
        for i in range(p.steps):
            rocdl.sched_barrier(0)
            if (
                i and i % p.psteps == 0
            ):  # the next K slice of x, once every wave is past the last one
                gpu.barrier()
                _land_x(k, p, i // p.psteps)
                gpu.barrier()
            if i + n_pre < p.steps:
                q.append(issue(i + n_pre))
            j = i % p.psteps
            for ch in range(p.rc):
                b = fx.ptr_load(
                    xrows[ch] + j * (KSTEP // 2),
                    result_type=fx.Vector.make_type(4, fx.Float32),
                )
                accs[ch] = mfma_bf16(
                    q[i].bitcast(fx.BFloat16), b.bitcast(fx.BFloat16), accs[ch]
                )
    rocdl.sched_barrier(0)
    if const_expr(
        p.race
    ):  # (after the barrier: the Philox and log math never above the MFMA loop's peak; the lane
        # an opaque copy, so none of the row math is hoisted out of the task loop and held across it: 18 rows spilled)
        seed0 = fx.Int32(fx.ptr_load(k.lds + (p.rpar + 2 * p.rows)))
        seed1 = fx.Int32(fx.ptr_load(k.lds + (p.rpar + 2 * p.rows + 1)))
        rlane = fresh(lane)
    for ch, acc in enumerate(accs):
        row = lane % 16 if ch == 0 else lane % 16 + RG * ch
        race = None
        if const_expr(p.race):
            rrow = rlane % 16 + RG * ch
            rr = fx.min(rrow, fx.Int32(p.rows - 1))
            t = fx.Int32(fx.ptr_load(k.lds + (p.rpar + rr * 2))).bitcast(fx.Float32)
            races = _races(
                t,
                fx.Int32(fx.ptr_load(k.lds + (p.rpar + rr * 2 + 1))).bitcast(
                    fx.Float32
                ),
            )
            race = (races, fx.Float32(1.0) / t * fx.Float32(LOG2E), rrow, seed0, seed1)
        if const_expr(
            p.sample and not p.race_only
        ):  # the logits as bf16 pairs into the rank's scratch (the scans)
            w01, w23 = bf16_pair(acc[0], acc[1]), bf16_pair(acc[2], acc[3])
            at = row * p.n2 + rg * (RG // 2) + (lane // 16) * 2
            ok = valid & (row < p.rows)
            if const_expr(
                p.race and not p.race_lg
            ):  # a race row's pick never scans them
                ok = ok & (races.select(fx.Int32(1), fx.Int32(0)) == 0)
            _lg_put(k, p, ok, at, w01, w23)
        _top1(k, p, rg, lane, valid, acc, ch, race)


def _top1(k, p, rg, lane, valid, acc, ch=0, race=None, put=True):
    """The wave's top-1 per row into LDS: the lane's 4 logits (ids 4 (l // 16) + e of the row group), lanes ^ 16, ^ 32
    (the waves: ``_task_put``); ``ch``: the rows' 16-row chunk. ``race``: the lane's row's (races, log2 e / T, row,
    seed words): a racing row's keys are the race scores (``_race_keys``)."""
    base = rg * RG + (lane // 16) * 4
    keys = _race_keys(k, p, acc, base, *race)
    kk, ii = keys[0], base
    for e in range_constexpr(1, 4):
        kk, ii = _take(kk, ii, keys[e], base + e)
    kk = valid.select(kk, fx.Int32(0))
    for off in (16, 32):
        kk, ii = _take(kk, ii, xshfl(kk, off), xshfl(ii, off))
    _wave_put(k, p, kk, ii, ch)


def _land_x(k, p, part):
    """K slice ``part`` of x into the LDS (``kparts`` > 1, after the first): a batch at a time (attempted, landed),
    beside the weight units in flight."""
    for specs, dests in _x_batches(k, p, k.tid, THREADS, part):
        _land_batch(k, p, specs, dests, _try_batch(specs))


@traced
def _wave_put(k, p, kk, ii, ch=0):
    """Lane l < 16 (after the lane reductions): row l of 16-row chunk ``ch``'s (key, id) -> LDS [wave][row]."""
    if const_expr(p.rc == 1):
        if k.lane < p.rows:
            fx.ptr_store(kk, k.lds + (p.red + (k.wave * p.rows + k.lane) * 2))
            fx.ptr_store(ii, k.lds + (p.red + (k.wave * p.rows + k.lane) * 2 + 1))
    else:
        if (k.lane < RG) & (k.lane + RG * ch < p.rows):
            fx.ptr_store(kk, k.lds + (p.red + (k.wave * p.rows + k.lane + RG * ch) * 2))
            fx.ptr_store(
                ii, k.lds + (p.red + (k.wave * p.rows + k.lane + RG * ch) * 2 + 1)
            )


@traced
def _task_put(k, p, t, first):
    """Thread r < rows: the task's best of row r over its waves (in order), (key, global id: ``first`` is this rank's
    first vocab id) to every rank."""
    if k.tid < p.rows:
        r = k.tid
        kk = fx.Int32(fx.ptr_load(k.lds + (p.red + r * 2)))
        ii = fx.Int32(fx.ptr_load(k.lds + (p.red + r * 2 + 1)))
        for w in range_constexpr(1, WAVES):
            kw = fx.Int32(fx.ptr_load(k.lds + (p.red + (w * p.rows + r) * 2)))
            iw = fx.Int32(fx.ptr_load(k.lds + (p.red + (w * p.rows + r) * 2 + 1)))
            kk, ii = _take(kk, ii, kw, iw)
        gid = ii + first
        pair = (t * p.rows + r) * 2
        if const_expr(p.tp > 1):
            for dst in range_constexpr(p.tp):
                mb.put2(to_rank(k, p.top, dst), pair, kk, gid, k.tag)


def _pick(k, r, p):
    """Row r: the best of the ``tp x tasks`` entries -> the outputs."""
    if const_expr(
        p.sample
    ):  # the row's temperature, loaded first: its latency under the poll, not after it
        t = uniform(ld_i32(k.args[f"{p.name}.par"], 2 * r)).bitcast(fx.Float32)
        race = None
        if const_expr(p.race):
            top_p = uniform(ld_i32(k.args[f"{p.name}.par"], 2 * r + 1)).bitcast(
                fx.Float32
            )
            race = _races(t, top_p).select(fx.Int32(1), fx.Int32(0))
    ents = p.tp * p.tasks
    a = (
        from_rank(k, p.top, 0) if p.tp > 1 else k.mb[p.top]
    )  # slot (parity, 0); slot src at src x pairs
    specs = []
    for c in range(-(-ents // THREADS)):
        e = fx.min(fx.Int32(c * THREADS) + k.tid, fx.Int32(ents - 1))
        src, j = e // p.tasks, e % p.tasks
        specs.append((a, src * p.top_pairs + (j * p.rows + r) * 2, 2))
    got = mb.poll(specs, k.tag)
    kk, ii = got[0][0], got[0][1]
    for g in got[1:]:
        kk, ii = _take(kk, ii, g[0], g[1])
    for off in (32, 16, 8, 4, 2, 1):
        kk, ii = _take(kk, ii, xshfl(kk, off), xshfl(ii, off))
    if const_expr(p.sample):
        _pick_out_s(k, p, r, kk, ii, t, race)
        if const_expr(not p.race_only):
            _sample_if(k, p, r, t, race)
    gpu.barrier()  # the next task reuses the LDS


# ------------------------------------------------------------------ sampling (``sample``)
#
# Pick-stage LDS (words): the waves' (key, id) pairs, then broadcast words, then 64-bit arrays.
S_BC = 2 * WAVES  # the row's max key, its id, B1, (pad), S_above(B1) f64, top_p Z f64
S_H1 = 32  # level 1: 256 u64 masses (by key16 >> 8)
S_H2 = S_H1 + 2 * 256  # level 2: 256 u64 masses (the keys of bin B1)
S_G = S_H2 + 2 * 256  # level 2: 256 i64 best (score, id) of the rank's tokens a key
S_GA = S_G + 2 * 256  # the best (score, id) of the rank's tokens in the bins above B1
SCAN_BATCH = 16  # 16-byte specs a poll batch of the scans (all: one round trip)
SCAN_SPECS = 16  # at most this many a thread (unrolled): 4 x 16 x 512 = 32768 ids a rank (V4.1 at TP4: 32320)
# a rank's row scanned in up to this many slices of SCAN_SPECS specs a thread, each slice's specs built where it is
# polled (not held across the scan): 65536 ids a rank (V4.1 at TP2: 64640); one slice: the scans as before
SCAN_SLICES = 2
COPIES = 16  # level 1 counts into copies (lane l: copy l % COPIES): its bins are sign and exponent, few take most
S_H1C = S_GA + 4  # the copies, 256 u64 each
S_JUNK = (
    S_H1C + COPIES * 2 * 256
)  # a word a lane: the zero masses' atomics (no conflicts)
S_WORDS = S_JUNK + 2 * 64
S_T = S_H1C  # the gathers of the ranks' records (``_gather``): the copies' words, free by then
H1_ROW = 2 * 256  # values a row of ``<name>.h1``: 256 x (mass lo, hi)
H2_ROW = (
    4 * 256 + 4
)  # values a row of ``<name>.h2``: 256 x (mass lo, hi, best lo, hi), the above-best, a pad pair
FIX_LOG2 = 36.0
FIX = (
    2.0**FIX_LOG2
)  # mass fixed point: a row's 129280 masses (each <= 1) sum below 2^53, so f64 sums of them are exact
LOG2E = 1.4426950408889634
I64_MIN_HI = INT_MIN  # an empty best: (lo 0, hi INT_MIN)
PHILOX_M, PHILOX_W = 0xD256D193, 0x9E3779B9 - 2**32


@traced
def _lg_put(k, p, ok, at, w01, w23):
    if ok:
        mb.put2(k.mb[p.lg], at, w01, w23, k.tag)


def _i64(lo, hi):
    return fx.Vector.from_elements([fx.Int32(lo), fx.Int32(hi)], fx.Int32).bitcast(
        fx.Int64
    )[0]


def _words(v):
    w = fx.Vector.from_elements([fx.Int64(v)], fx.Int64).bitcast(fx.Int32)
    return w[0], w[1]


def _f64(v):
    return fx.Int64(v).to(fx.Float64)


def _f64_words(x):
    w = fx.Vector.from_elements([fx.Float64(x)], fx.Float64).bitcast(fx.Int32)
    return w[0], w[1]


def _f64_of(lo, hi):
    return fx.Vector.from_elements([fx.Int32(lo), fx.Int32(hi)], fx.Int32).bitcast(
        fx.Float64
    )[0]


def _wave_prefix_f64(x):
    """Inclusive prefix sum of f64 ``x`` over the wave's lanes (lane 0 first), ``topk.wave_prefix``'s DPP steps on
    both words (shifted-in zeros are +0.0). Exact when every partial sum is an integer below 2^53 (the masses).
    """
    for ctrl, rmask in (
        (0x111, 0xF),
        (0x112, 0xF),
        (0x114, 0xF),
        (0x118, 0xF),
        (0x142, 0xA),
        (0x143, 0xC),
    ):
        lo, hi = _f64_words(x)
        slo = fx.Int32(update_dpp_i32(fx.Int32(0), lo, ctrl, rmask, 0xF, False))
        shi = fx.Int32(update_dpp_i32(fx.Int32(0), hi, ctrl, rmask, 0xF, False))
        x = x + _f64_of(slo, shi)
    return x


def _readlane_f64(x, lane):
    lo, hi = _f64_words(x)
    return _f64_of(readlane(lo, lane), readlane(hi, lane))


def _wave_max_i64(v):
    for off in (32, 16, 8, 4, 2, 1):
        lo, hi = _words(v)
        v = fx.max(v, _i64(xshfl(lo, off), xshfl(hi, off)))
    return v


def _key16(w, hi_half):
    """(bf16 value as f32, its 16-bit order key) of the low / high bf16 of word ``w`` (-0 as +0)."""
    bits = (w & fx.Int32(-65536)) if hi_half else (w << 16)
    bits = (bits == fx.Int32(INT_MIN)).select(fx.Int32(0), bits)
    return bits.bitcast(fx.Float32), (order_key(bits) >> 16) & 0xFFFF


def fma_f32(a, b, c):
    from flydsl.expr import math as fmath

    return fmath.fma(fx.Float32(a), fx.Float32(b), fx.Float32(c))


def _mass(b, m, c):
    """The fixed-point mass of bf16 logit ``b``: exp2((b - M) log2e / T) 2^36 (an f32: 24 significant bits, as an
    fp32 probability), to the nearest integer, as Int64 (f32 only: high word v / 2^32, low word the exact rest).
    """
    return _mass_v(b, m, c)[0]


def _mass_v(b, m, c):
    """(``_mass``, the f32 v it rounds)."""
    v = hw_exp2(fx.Float32(fma_f32((b - m), c, fx.Float32(FIX_LOG2))))
    hi = (v * fx.Float32(2.0**-32)).to(fx.Int32)  # < 2^4
    rest = (
        v - hi.to(fx.Float32) * fx.Float32(2.0**32) + fx.Float32(0.5)
    )  # exact, in [0, 2^32)
    return _i64(rest.to(fx.Uint32), hi), v


def _philox(c0, c1, key):
    """Philox2x32-10 of counter (c0, c1) under ``key`` -> two uniform words."""
    for _ in range_constexpr(10):
        prod = (fx.Int64(c0) & fx.Int64(0xFFFFFFFF)) * fx.Int64(PHILOX_M)
        hi, lo = fx.Int32(prod >> 32), fx.Int32(prod)
        c0, c1 = hi ^ key ^ c1, lo
        key = key + fx.Int32(PHILOX_W)
    return c0, c1


def _race_key(v, word, gid):
    """The (score, id) of a token as one signed Int64 (larger: better; equal scores: the lower id): the exponential
    race, score = v / E with v its mass (``_mass_v``) and E = -log2 U, U = (word >> 8 + 1/2) 2^-24 in (0, 1) (an
    exponential variable, scaled): argmax v / E draws id i with probability v_i / sum v, as the Gumbel-max
    argmax (b / T - ln(-ln U)) does, with one log instead of two."""
    u = ((word >> 8) & 0xFFFFFF).to(fx.Float32) * fx.Float32(2.0**-24) + fx.Float32(
        2.0**-25
    )
    score = v * hw_rcp(-hw_log2(u))
    return _i64(
        fx.Int32(0x7FFFFFFF) - gid,
        order_key(score.bitcast(fx.Int32)) ^ fx.Int32(INT_MIN),
    )


def _races(t, top_p):
    """A row drawn in the dot's epilogue (``race``): T > 0 and top_p >= 1 (the dot and the pick both decide by this)."""
    return (t > fx.Float32(0.0)) & (top_p >= fx.Float32(1.0))


def _race_par(k, p):
    """``race``: this thread's word of ``<name>.par`` (thread i < 2 rows + 2: word i; ``_race_put``)."""
    return ld_i32(k.args[f"{p.name}.par"], fx.min(k.tid, fx.Int32(2 * p.rows + 1)))


@traced
def _race_put(k, p, w):
    """``race``: ``<name>.par`` into the dot's LDS from word ``rpar`` (read after the stream's barrier)."""
    if k.tid < 2 * p.rows + 2:
        fx.ptr_store(w, k.lds + (p.rpar + k.tid))


def _exp_var(word):
    """E = -log2 U (an exponential variable, in log2 units), U = (k + 1/2) 2^-24 for the word's top 24 bits k: in
    (0, 1) exactly. An f32 U would round in the upper half (the top cell to 1.0: E 0), so the upper half works from
    x = 1 - U = (2^24 - 1 - k + 1/2) 2^-24 (exact, below 1/2): -log2(1 - x) by its series below 2^-5 (to x^5: within
    2^-25 relative), else the log of 1 - x rounded (within 2^-25 absolute: 2^-20 of x, relative).
    """
    kk = (word >> 8) & 0xFFFFFF
    up = kk >= fx.Int32(1 << 23)
    x = (
        up.select(fx.Int32(0xFFFFFF) - kk, kk).to(fx.Float32) + fx.Float32(0.5)
    ) * fx.Float32(2.0**-24)
    lg = -hw_log2(up.select(fx.Float32(1.0) - x, x))
    ser = fma_f32(x, fma_f32(x, fma_f32(x, fma_f32(x, fx.Float32(0.2), fx.Float32(0.25)), fx.Float32(1 / 3)),
                             fx.Float32(0.5)), fx.Float32(1.0))  # fmt: skip
    return (up & (x < fx.Float32(2.0**-5))).select(x * ser * fx.Float32(LOG2E), lg)


def _race_keys(k, p, acc, base, races, c, row, seed0, seed1):
    """The lane's 4 keys of row ``row`` (ids ``base`` + e of this rank): racing (``races``), the order key of the race
    score b log2(e) / T - log2(E), b the bf16 logit, ``c`` = log2(e) / T, E = ``_exp_var`` of the id's Philox word (the
    pick's counter mapping: ``_cand``); else the greedy key."""
    gid = base + (
        k.rank * p.n if p.tp > 1 else fx.Int32(0)
    )  # the global id of e = 0 (even: n is a multiple of 4)
    us = []
    for h in range_constexpr(2):
        u0, u1 = _philox(((gid >> 1) + h) | (row << p.cshift), k.tag ^ seed1, seed0)
        us += [u0, u1]
    keys = []
    for e in range_constexpr(4):
        bits = fma_f32(bf16_round(acc[e]), c, -hw_log2(_exp_var(us[e]))).bitcast(
            fx.Int32
        )
        bits = (bits == fx.Int32(INT_MIN)).select(fx.Int32(0), bits)  # -0 as +0
        keys.append(races.select(order_key(bits), _key(p, acc[e])))
    return keys


@traced
def _pick_out_s(k, p, r, kk, ii, t, race=None):
    if k.lane == 0:
        fx.ptr_store(kk, k.lds + k.wave * 2)
        fx.ptr_store(ii, k.lds + (k.wave * 2 + 1))
    gpu.barrier()
    if k.tid == 0:
        kb = fx.Int32(fx.ptr_load(k.lds + 0))
        ib = fx.Int32(fx.ptr_load(k.lds + 1))
        for w in range_constexpr(1, WAVES):
            kb, ib = _take(
                kb,
                ib,
                fx.Int32(fx.ptr_load(k.lds + w * 2)),
                fx.Int32(fx.ptr_load(k.lds + (w * 2 + 1))),
            )
        fx.ptr_store(kb, k.lds + S_BC)
        fx.ptr_store(ib, k.lds + (S_BC + 1))
        if t <= fx.Float32(0.0):
            bo.buffer_store(ib, rsrc(k.args[f"{p.name}.ids"]), r)
            if const_expr(p.vals):
                bo.buffer_store(key_bits(kb), rsrc(k.args[f"{p.name}.vals"]), r)
        if const_expr(race is not None):  # noqa: SIM102 (traced)
            if (
                race != 0
            ):  # the dot's race winner; vals -inf: every id kept (the scans' value at top_p 1)
                bo.buffer_store(ib, rsrc(k.args[f"{p.name}.ids"]), r)
                if const_expr(p.vals):
                    bo.buffer_store(
                        fx.Int32(-8388608), rsrc(k.args[f"{p.name}.vals"]), r
                    )
    gpu.barrier()


@traced
def _sample_if(k, p, r, t, race=None):
    if (t > fx.Float32(0.0)) & (race == 0):
        _sample_row(k, p, r, t)


def _lg_specs(k, p, r, s: int = 0):
    """This thread's 16-byte specs of row r of ``<name>.lg`` (quad q = tid + THREADS c, clamped) and their
    validity (Python True where every thread's quad is in range). ``s``: the scan's slice (``p.scan_slices`` > 1:
    columns c of slice s only, from an opaque copy of tid, so a slice's addresses are not hoisted and held).
    """
    a = k.mb[p.lg]
    specs, oks = [], []
    cols = -(-p.quads // THREADS)
    tid = k.tid if p.scan_slices == 1 else fresh(k.tid)
    for c in range(
        s * SCAN_SPECS, min(cols, (s + 1) * SCAN_SPECS) if p.scan_slices > 1 else cols
    ):
        q = fx.Int32(c * THREADS) + tid
        specs.append((a, r * p.n2 + fx.min(q, fx.Int32(p.quads - 1)) * 2, 2))
        oks.append(True if (c + 1) * THREADS <= p.quads else q < p.quads)
    return specs, oks


def _sample_row(k, p, r, t):
    lds64 = fx.recast_iter(fx.Int64, k.lds)
    tid, lane = k.tid, k.lane
    c = fx.Float32(1.0) / t * fx.Float32(LOG2E)
    m = key_bits(fx.Int32(fx.ptr_load(k.lds + S_BC))).bitcast(
        fx.Float32
    )  # the row's max bf16 logit (all ranks)
    first = k.rank * p.n if p.tp > 1 else fx.Int32(0)
    # 0. clear: the masses to 0, the bests to empty, the level-1 copies to 0
    for i in range_constexpr(-(-(2 * 256 + 1) // THREADS)):  # H2 (0), G and GA (empty)
        e = fx.min(fx.Int32(i * THREADS) + tid, fx.Int32(2 * 256))
        fx.ptr_store(
            (e < 256).select(fx.Int64(0), _i64(0, I64_MIN_HI)), lds64 + (S_H2 // 2 + e)
        )
    for i in range_constexpr(COPIES * 256 // THREADS):
        fx.ptr_store(fx.Int64(0), lds64 + (S_H1C // 2 + i * THREADS + tid))
    gpu.barrier()
    # 1. level 1: the rank's masses by key16 >> 8 (a zero mass: into the lane's junk word)
    specs, oks = _lg_specs(k, p, r)
    copy = S_H1C // 2 + (lane % COPIES) * 256
    for s in range_constexpr(p.scan_slices):
        for ci in range_constexpr(len(specs)):
            if const_expr(
                ci % SCAN_BATCH == 0
            ):  # a batch polled, then used (its words only live)
                got = mb.poll(specs[ci : ci + SCAN_BATCH], k.tag)
            for wi in range_constexpr(2):
                for h in range_constexpr(2):
                    b, key = _key16(got[ci % SCAN_BATCH][wi], h == 1)
                    q = _mass(b, m, c)
                    live = q != fx.Int64(0)
                    if oks[ci] is not True:
                        live = live & oks[ci]
                    at = live.select(copy + (key >> 8), fx.Int32(S_JUNK // 2) + lane)
                    fx.atomic_add(lds64 + at, q, syncscope="workgroup")
    gpu.barrier()
    _sum_copies(k)
    _put_h1(k, p, r)
    _level1(
        k, p, r, uniform(ld_i32(k.args[f"{p.name}.par"], 2 * r + 1)).bitcast(fx.Float32)
    )
    gpu.barrier()
    b1 = fx.Int32(fx.ptr_load(k.lds + (S_BC + 2)))
    seed0 = uniform(
        ld_i32(k.args[f"{p.name}.par"], 2 * p.rows)
    )  # (loaded here: not held across level 1)
    seed1 = uniform(ld_i32(k.args[f"{p.name}.par"], 2 * p.rows + 1))
    # 2. level 2: the rank's masses and bests of bin B1's keys, its best above B1
    best = _i64(0, I64_MIN_HI)
    for s in range_constexpr(p.scan_slices):
        for ci in range_constexpr(len(specs)):
            if const_expr(ci % SCAN_BATCH == 0):
                got = mb.poll(specs[ci : ci + SCAN_BATCH], k.tag)
            q4 = (specs[ci][1] - r * p.n2) // 2  # this quad's index in the row
            for wi in range_constexpr(2):
                w = got[ci % SCAN_BATCH][wi]
                b0, key0 = _key16(w, False)
                b1v, key1 = _key16(w, True)
                ok = True if oks[ci] is True else oks[ci]
                gid0 = first + q4 * 4 + wi * 2
                if ok is not True:  # an out-of-range quad (clamped): never a candidate
                    b1x = ok.select(b1, fx.Int32(1 << 20))
                else:
                    b1x = b1
                best = _cand(
                    k, b0, key0, b1v, key1, b1x, m, c, r, gid0, seed0, seed1, best
                )
    fx.atomic_max(lds64 + S_GA // 2, best, syncscope="workgroup")
    gpu.barrier()
    _put_h2(k, p, r)
    _level2(k, p, r)


@traced
def _cand(k, b0, key0, b1v, key1, b1, m, c, r, gid0, seed0, seed1, best):
    """A pair of ids (gid0, gid0 + 1), when either token is a candidate (its bin >= B1; else nothing): their
    (score, id) keys; a token in a bin above B1 into ``best`` (returned), one in B1 into its key's mass and best
    (LDS atomics; the other token's to the lane's junk word). The uniforms: Philox2x32-10, counter
    (gid0 / 2 | row << 16, tag ^ seed1), key seed0."""
    if ((key0 >> 8) >= b1) | ((key1 >> 8) >= b1):
        lds64 = fx.recast_iter(fx.Int64, k.lds)
        gid0 = fresh(
            gid0
        )  # its math here, not hoisted for every pair of the scan and held across it
        junk = fx.Int32(S_JUNK // 2) + (fresh(k.tid) & 63)
        u0, u1 = _philox((gid0 >> 1) | (r << 16), k.tag ^ seed1, seed0)
        for h, (b, key, u) in enumerate(((b0, key0, u0), (b1v, key1, u1))):
            q, v = _mass_v(b, m, c)
            g = _race_key(v, u, gid0 + h)
            bin_ = key >> 8
            best = fx.max(best, (bin_ > b1).select(g, _i64(0, I64_MIN_HI)))
            inb = bin_ == b1
            at = inb.select(fx.Int32(S_H2 // 2) + (key & 255), junk)
            fx.atomic_add(lds64 + at, inb.select(q, fx.Int64(0)), syncscope="workgroup")
            at = inb.select(fx.Int32(S_G // 2) + (key & 255), junk)
            fx.atomic_max(
                lds64 + at, inb.select(g, _i64(0, I64_MIN_HI)), syncscope="workgroup"
            )
    return best


@traced
def _sum_copies(k):
    """Threads 0 .. 255: bin ``tid``'s level-1 mass, the sum of its copies (integers: any order)."""
    if fresh(k.tid) < 256:  # (an opaque copy: the mask not hoisted to the stage start)
        lds64 = fx.recast_iter(fx.Int64, k.lds)
        s = fx.Int64(0)
        for i in range_constexpr(COPIES):
            s = s + fx.ptr_load(lds64 + (S_H1C // 2 + i * 256 + k.tid))
        fx.ptr_store(s, lds64 + (S_H1 // 2 + k.tid))


@traced
def _put_h1(k, p, r):
    """Threads 0 .. 255: bin ``tid``'s mass into every rank's ``<name>.h1`` (row r)."""
    if fresh(k.tid) < 256:  # (an opaque copy: the mask not hoisted to the stage start)
        lo, hi = _words(
            fx.ptr_load(fx.recast_iter(fx.Int64, k.lds) + (S_H1 // 2 + k.tid))
        )
        at = r * H1_ROW + k.tid * 2
        if const_expr(p.tp > 1):
            for dst in range_constexpr(p.tp):
                mb.put2(to_rank(k, p.h1, dst), at, lo, hi, k.tag)


def _src(k, p, region, src):
    return from_rank(k, region, src) if p.tp > 1 else k.mb[region]


@traced
def _level1(k, p, r, top_p):
    """Wave 0: the ranks' level-1 masses summed (lane l: bins 255 - 4 l - j, j = 0 .. 3), Z, and B1: the bin whose
    keys hold tau (the mass of the bins above it < top_p Z <= that plus its own). Broadcast words: B1 (-1: no bin,
    every token kept), S_above(B1), top_p Z."""
    _gather(k, p, p.h1, r * H1_ROW, H1_ROW * p.rows, 1)
    gpu.barrier()
    if k.wave == 0:
        lane = k.lane
        lds64 = fx.recast_iter(fx.Int64, k.lds)
        cnt = []
        for j in range_constexpr(4):  # bin 255 - 4 l - j
            b = fx.Int32(255 - j) - lane * 4
            cnt.append(
                _f64(
                    fx.ptr_load(lds64 + (S_T // 2 + b))
                    + fx.ptr_load(lds64 + (S_T // 2 + 256 + b))
                )
            )
        tot = (cnt[0] + cnt[1]) + (cnt[2] + cnt[3])
        incl = _wave_prefix_f64(tot)
        z = _readlane_f64(incl, 63)
        pz = top_p.to(fx.Float64) * z
        pz = (top_p >= fx.Float32(1.0)).select(
            fx.Float64(float("inf")), pz
        )  # top_p 1: no filter (ATOM's too)
        above = incl - tot
        hit_w, s_hit = fx.Int32(-1), fx.Float64(0.0)
        for j in range_constexpr(4):
            hit = (above < pz) & (above + cnt[j] >= pz)
            hit_w = hit.select(fx.Int32(255 - j) - lane * 4, hit_w)
            s_hit = hit.select(above, s_hit)
            above = above + cnt[j]
        b1 = uniform(_wave_max_i32(hit_w))
        who = uniform(_wave_max_i32((hit_w >= 0).select(lane, fx.Int32(-1))))
        s_b1 = _readlane_f64(s_hit, fx.max(who, fx.Int32(0)))
        if lane == 0:
            slo, shi = _f64_words(s_b1)
            plo, phi = _f64_words(pz)
            fx.ptr_store(
                fx.Vector.from_elements([b1, b1, slo, shi, plo, phi], fx.Int32),
                k.lds + (S_BC + 2),
            )


@traced
def _gather(k, p, region, base, slot, fields):
    """Every thread: entry e = tid % 256 of row record ``base`` of ``region`` (``fields`` u64 at value 2 fields e: 1:
    a mass; 2: a mass and a best) over the ranks of half tid // 256 (rank 2 j + half): the masses summed, the bests'
    max, into LDS ``S_T`` (masses [2][256] i64, then bests [2][256]). ``slot``: values a rank's slot holds. One poll
    batch a thread (the ranks' records spread over the CTA)."""
    half, e = k.tid // 256, k.tid % 256
    a0 = _src(k, p, region, 0)
    n = -(-p.tp // 2)
    specs, oks = [], []
    for j in range_constexpr(n):
        src = fx.min(fx.Int32(2 * j) + half, fx.Int32(p.tp - 1))
        oks.append(fx.Int32(2 * j) + half < p.tp)
        at = src * (slot if p.tp > 1 else 0) + base + e * 2 * fields
        for f in range_constexpr(fields):
            specs.append((a0, at + f * 2, 2))
    got = mb.poll(specs, k.tag)
    s, g = fx.Int64(0), _i64(0, I64_MIN_HI)
    for j in range_constexpr(n):
        w = got[j * fields]
        s = s + oks[j].select(_i64(w[0], w[1]), fx.Int64(0))
        if const_expr(fields == 2):
            w = got[j * fields + 1]
            g = fx.max(g, oks[j].select(_i64(w[0], w[1]), _i64(0, I64_MIN_HI)))
    lds64 = fx.recast_iter(fx.Int64, k.lds)
    fx.ptr_store(s, lds64 + (S_T // 2 + k.tid))
    if const_expr(fields == 2):
        fx.ptr_store(g, lds64 + (S_T // 2 + 512 + k.tid))


def _wave_max_i32(v):
    for off in (32, 16, 8, 4, 2, 1):
        v = fx.max(v, xshfl(v, off))
    return v


@traced
def _put_h2(k, p, r):
    """Threads 0 .. 255: key ``tid`` of bin B1, its (mass, best); thread 256: the best above B1; into every rank's
    ``<name>.h2`` (row r)."""
    lds64 = fx.recast_iter(fx.Int64, k.lds)
    if fresh(k.tid) < 257:  # (an opaque copy: the mask not hoisted to the stage start)
        e = fx.min(k.tid, fx.Int32(255))
        mlo, mhi = _words(fx.ptr_load(lds64 + (S_H2 // 2 + e)))
        glo, ghi = _words(fx.ptr_load(lds64 + (S_G // 2 + e)))
        alo, ahi = _words(fx.ptr_load(lds64 + S_GA // 2))
        last = k.tid == 256
        at = r * H2_ROW + last.select(fx.Int32(1024), e * 4)
        w0, w1 = last.select(alo, mlo), last.select(ahi, mhi)
        w2, w3 = last.select(fx.Int32(0), glo), last.select(fx.Int32(0), ghi)
        if const_expr(p.tp > 1):
            for dst in range_constexpr(p.tp):
                a = to_rank(k, p.h2, dst)
                mb.put2(a, at, w0, w1, k.tag)
                mb.put2(a, at + 2, w2, w3, k.tag)


@traced
def _level2(k, p, r):
    """Wave 0: the ranks' level-2 records summed (lane l: keys 255 - 4 l - j of bin B1); a key is in the nucleus
    when S_above(B1) + the mass of the bin's keys above it < top_p Z; the winner = the best (score, id) over the
    nucleus keys' bests and the ranks' bests above B1 -> the outputs."""
    _gather(k, p, p.h2, r * H2_ROW, H2_ROW * p.rows, 2)
    gpu.barrier()
    if k.wave == 0:
        lane = k.lane
        lds64 = fx.recast_iter(fx.Int64, k.lds)
        s_b1 = _f64_of(fx.ptr_load(k.lds + (S_BC + 4)), fx.ptr_load(k.lds + (S_BC + 5)))
        pz = _f64_of(fx.ptr_load(k.lds + (S_BC + 6)), fx.ptr_load(k.lds + (S_BC + 7)))
        # the ranks' bests above B1: lane q < tp polls rank q's
        q = fx.min(lane, fx.Int32(p.tp - 1))
        wa = mb.poll([(_src(k, p, p.h2, 0), q * (H2_ROW * p.rows if p.tp > 1 else 0) + r * H2_ROW + 1024, 2)],
                     k.tag)[0]  # fmt: skip
        win = _i64(wa[0], wa[1])
        cnt, bst = [], []
        for j in range_constexpr(4):  # key 255 - 4 l - j
            f = fx.Int32(255 - j) - lane * 4
            cnt.append(
                _f64(
                    fx.ptr_load(lds64 + (S_T // 2 + f))
                    + fx.ptr_load(lds64 + (S_T // 2 + 256 + f))
                )
            )
            bst.append(
                fx.max(
                    fx.ptr_load(lds64 + (S_T // 2 + 512 + f)),
                    fx.ptr_load(lds64 + (S_T // 2 + 768 + f)),
                )
            )
        tot = (cnt[0] + cnt[1]) + (cnt[2] + cnt[3])
        above = s_b1 + (_wave_prefix_f64(tot) - tot)
        b1 = fx.Int32(fx.ptr_load(k.lds + (S_BC + 2)))
        tau = fx.Int32(1 << 16)
        for j in range_constexpr(4):
            keep = above < pz
            win = fx.max(win, keep.select(bst[j], _i64(0, I64_MIN_HI)))
            tau = keep.select(
                fx.min(tau, (b1 << 8) + (fx.Int32(255 - j) - lane * 4)), tau
            )
            above = above + cnt[j]
        win = _wave_max_i64(win)
        tau = -_wave_max_i32(-tau)
        lo, _hi = _words(win)
        if lane == 0:
            bo.buffer_store(fx.Int32(0x7FFFFFFF) - lo, rsrc(k.args[f"{p.name}.ids"]), r)
            if const_expr(
                p.vals
            ):  # the nucleus: every bf16 logit >= this (-inf: every token)
                lowest = key_bits(tau << 16) & fx.Int32(-65536)
                lowest = (b1 < 0).select(fx.Int32(-8388608), lowest)
                bo.buffer_store(lowest, rsrc(k.args[f"{p.name}.vals"]), r)

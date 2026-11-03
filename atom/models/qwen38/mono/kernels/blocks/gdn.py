# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``GDNCore``: the gated-delta-net mixer between its two linears (Qwen3.8-Flash-Next).

For a ``GDNGeometry`` (``geo``: k_dim = v_dim = 128, a 4-tap conv, fp32 state; k-head ``h // rep``
serves v-head ``h``), per request ``b`` (rows ``b * S .. b * S + S - 1`` of the step,
consecutive tokens):

- causal depthwise conv (4 taps) over the 3-row history + this step's qkv rows: each tap
  product rounded to bf16, summed in fp32 in tap order (from +0), SiLU, bf16
  (``causal_conv1d_update``);
- q, k l2-normalized per 128-dim k-head (fp32, eps 1e-6 inside the root, the factors as
  ``1 / sqrt(.)`` correctly rounded), q's times fp32(128^-0.5);
- b, a: BF16 linears of the layer's mixed row (``ba``: that row's BF16 channel [rows, K], the
  weight ``ba_w`` bf16 [2 * v_heads, K], b rows then a rows). Every task computes its v-head's
  b, a for the request's rows (fp32 products of exact bf16 values, a fixed order: each lane's
  words, the wave butterfly, then the waves of the row in order; bf16 out). The poll of the
  mixed row goes out before ``proj``'s (the row is ready a projection earlier);
- beta = bf16(sigmoid(b)), g = -exp(A_log) * softplus(a + dt_bias) (fp32);
- per row t, in order (state S [K, V] per v-head): S *= exp(g_t); kv = k_t S;
  S += k_t (x) ((v_t - kv) beta_t); out_t = q_t S (fp32 -> bf16); the state after
  every row is stored into that row's snapshot slot;
- gated norm per v-head: bf16(bf16(w * bf16(out * rsqrt(mean(out^2) + eps))) * sigmoid(z));
- quantized to the MXFP8 channel ``core`` [rows, v_heads * 128] (``quantize_mxfp8``).

Input: the BF16 channel ``proj`` = in_proj_qkv (q | k | v) | z, HF row order.

State (plain memory, Atom's pools; every address goes through the
helpers under "state pools" so a layout change touches one place):

- ``ssm_pool`` fp32 [slots, v_heads, K=128, V=128] (V contiguous); ``ssm_read`` int32
  [batch]: each request's state before the step; ``ssm_write`` int32 [rows]: the
  slot receiving the state after each row (an accepted prefix commits by slot);
- ``conv_pool`` bf16 [slots, 2 + S, conv_dim] (token-major); ``conv_slot`` /
  ``conv_off`` int32 [batch]: the request's window and the offset of its 3-row
  history (oldest first). After the step the slot holds, in place, the window
  (history rows 1..2, the step's S pre-conv qkv rows).

A negative slot is no slot: never read (a negative ``ssm_read`` / ``conv_slot`` reads a
zero state / history), never written (``ssm_write`` per row, ``conv_slot`` for the
window). Atom's padded requests (graph batches) pass -1 everywhere, so they touch no
pool and their outputs stay finite.

In-place safety: a write slot may equal the read slot. Every thread reads its
own state elements before storing them; the q / k conv history (read by the
rep * splits tasks of a k-head) is rewritten only after all of them have flagged
it read (region ``<name>.hr``, put right after their input phase); v history is
read and written by one task.

Weights (``pack``): conv taps bf16 [conv_dim, 4], A_log and dt_bias bf16 [v_heads], the
norm weight bf16 [128], ``ba_w``.

Algorithm (the recurrence in its chunked form, exact in real arithmetic; fp32):
with ``A_t(x) = x S_t`` (raw, unnormalized k / q rows), ``fk_t``, ``fq_t`` the
norm factors and ``e_t = fk_t (v_t - dec_t fk_t A_{t-1}(k_t)) beta_t``:
``S_t = dec_t S_{t-1} + k_t (x) e_t``, ``A_t(x) = dec_t A_{t-1}(x) + (k_t . x) e_t``,
``out_t = fq_t A_t(q_t)``. So the step needs ``A_{-1}`` of every k_t and q_t and
the Gram matrix of the k / q rows (one small matmul with the initial state and
the rows themselves as B operands, ``v_mfma_f32_16x16x4_f32``), then a short
scalar recurrence per V column; snapshots are elementwise.

Task decomposition (``splits``: 2 for batch <= 2, else 1, ``frozen_splits``): task = (request,
v-head, V-slice of ``vs = 128 / splits`` columns); 512 threads. The state slice [128, vs]
lives in the MFMA B-operand layout: wave ``w`` owns K rows ``16 w .. 16 w + 15``
(lane ``l``: rows ``16 w + 4 (l // 16) .. + 3``) and V columns ``vw (l % 16) ..``
(``cw`` contiguous columns: one vector load / store per K row), so every wave holds
all ``vs`` columns. With ``splits > 1`` the gated norm's per-head sum of squares is
an exchange: every slice puts its partial (per row) into ``<name>.sq`` and sums all
slices' partials in slice order. Tasks are ordered (request, k-head, v-head of the
k-head, slice), ``group = rep * splits`` task slots per k-head, all in one round.

Phases of a task: (A) loads that do not depend on the step (slot indices first,
then everything not behind them, then state and history), (B) the poll of its
``proj`` elements, inputs into LDS; (C) conv + SiLU, gates, the v window; (D) the
matmul (8 K-groups = 8 waves); (R) the cross-wave reduction; then the tail:

- every wave runs the scalar recurrence for its lanes' columns and stores each
  row's snapshot as soon as its e_t is known (no barrier between rows);
- wave 0 (which shares SIMD 0 with wave 4) runs the recurrence without stores
  (out_t and e_t into LDS), then (E) gated norm + MXFP8 + ``core`` puts and (F) the
  q / k window: its polls never wait for stores of its own; its snapshots after (F);
- with 2 slices (``exchange_first``) wave 0 puts its partials before the other waves
  start storing: a put queued behind the snapshot stream reaches the siblings late.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import partial

import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import rocdl as _rocdl_ods
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.rocdl import _split_mfma_operands
from flydsl.expr.typing import T

from atom.models.qwen38.mono.kernels.blocks.base import Block, Channel, Fmt
from atom.models.qwen38.mono.kernels.device import buffer_ops as bo
from atom.models.qwen38.mono.kernels.device import channels as chan
from atom.models.qwen38.mono.kernels.device import mailbox as mb
from atom.models.qwen38.mono.kernels.device.ops import (
    CM_NT,
    bf16_pair,
    bf16_round,
    bf_hi,
    bf_lo,
    butterfly,
    fp8_pack4,
    fresh,
    fresh_s64,
    hw_rsq,
    lane_gather,
    ld_i32,
    readlane,
    rsrc,
    traced,
    uniform,
)
from atom.models.qwen38.mono.kernels.geometry import REPLAY_MAX, GDNGeometry
from atom.models.qwen38.mono.kernels.program import Region, Scenario, Stage
from atom.models.qwen38.mono.kernels.target import MI355X

THREADS = MI355X.threads
WAVE = MI355X.wave
WAVES = THREADS // WAVE
GROUP = 32  # MXFP8 group
ZERO_CODE = 127

# The geometry the device code supports (lane mappings, the 4-tap conv, fp32 state); head counts
# and eps come from the ``GDNGeometry`` (``GDNPlan.geo``).
KD = 128  # k_dim
VD = 128  # v_dim
HIST = 3  # history rows: conv_kernel 4
RSQRT_KD = KD**-0.5
RED_COL = 20  # column stride of a 16 x 16 partial tile (16 rows + 4 pad)
RED_TILE = 16 * RED_COL
STATE_CM = CM_NT  # cache policy of the snapshot stores
STATE_LD_CM = 0  # cache policy of the state loads
# the contract's rounding choices (``numerics.GDNNumerics`` fields conv_rounding, gate, qk_norm): each conv tap
# product rounded to bf16, the norm gated by sigmoid(z), the q / k norm factors as 1 / sqrt(.)
FLASH_NUMERICS = ("tap_bf16", "sigmoid", "div_sqrt")


# ------------------------------------------------------------------ state pools (the one place that knows the layout)


def check_geometry(geo: GDNGeometry) -> None:
    """Raise unless the device code supports ``geo`` (k_dim = v_dim = 128, a 4-tap conv, fp32 state)."""
    if (geo.k_dim, geo.v_dim, geo.conv_kernel) != (
        KD,
        VD,
        HIST + 1,
    ) or geo.state_dtype != torch.float32:
        raise ValueError(
            f"GDNCore supports k_dim = v_dim = {KD}, conv_kernel {HIST + 1}, fp32 state; got {geo}"
        )


def ssm_slot(c, pool, slot):
    """Device: Int64 address of ``slot``'s recurrent state."""
    return fx.Int64(pool) + fx.Int64(slot) * c.slot_nbytes


def slot_bytes(slot, nbytes):
    """``nbytes`` (a state slot's: ``GDNPlan.slot_nbytes``) for a slot >= 0, else 0: a buffer
    resource over a negative slot (a padded request's) has no bytes, so its loads return 0 and
    its stores are dropped."""
    return fx.Int64((uniform(slot) >= 0).select(fx.Int32(nbytes), fx.Int32(0)))


def ssm_index(h, kk, v):
    """Element index of state[h, kk, v] within a slot."""
    return (h * KD + kk) * VD + v


def load_state(base, idx, n, nbytes=None, cm=0):
    """``n`` fp32 state elements (4 or 8) at a slot base (Int64) + element index
    (``nbytes``: the resource's size, ``slot_bytes``; ``cm``: the loads' cache policy).
    """
    r = rsrc(base, nbytes)
    out = []
    for q in range(0, n, 4):
        w = min(4, n - q)
        v = fx.Vector(
            bo.buffer_load(r, idx + q, vec_width=w, dtype=T.i32, cache_modifier=cm)
        )
        out += [v[e].bitcast(fx.Float32) for e in range(w)]
    return out


def store_state(base, idx, vals, cm, nbytes=None):
    """f32 ``vals`` (a multiple of 4) at a slot base + element index."""
    r = rsrc(base, nbytes)
    for q in range(0, len(vals), 4):
        words = fx.Vector.from_elements(vals[q : q + 4], fx.Float32).bitcast(fx.Int32)
        bo.buffer_store(words, r, idx + q, cache_modifier=cm)


def conv_rows(spec: int) -> int:
    """Rows of a conv window slot: the 2 history rows kept + the step's rows."""
    return HIST - 1 + spec


def conv_word(c, slot, row, pair):
    """Word index (from the pool base) of window row ``row``, channels ``2 pair, 2 pair + 1``.
    32-bit: the pool must stay below 2^31 bytes."""
    return (slot * conv_rows(c.spec) + row) * (c.conv // 2) + pair


# ------------------------------------------------------------------ plan


def frozen_splits(batch: int) -> int:
    """V-slices per v-head of the served builds (every task in one round): 2 up to batch 2, else 1."""
    return 2 if batch <= 2 else 1


@dataclass(frozen=True)
class GDNPlan:
    """Plan-side constants of one build (bound into emit by value, part of the key)."""

    batch: int
    spec: int
    splits: int  # V-slices per v-head
    geo: GDNGeometry  # head counts, eps (``check_geometry``)
    ba_k: int  # b, a computed in the core from the BF16 mixed row of this width
    # ReplaySSM record capacity (ATOM's ``replayssm_cache_len``); 0: the classic verify (a state snapshot per row)
    replay: int = 0

    def __post_init__(self):
        if self.splits not in (1, 2):
            raise ValueError(f"splits {self.splits} not in (1, 2)")
        if not 1 <= self.spec <= 8:
            raise ValueError(f"spec {self.spec} not in 1..8")
        if self.ba_k % (4 * WAVE):
            raise ValueError(
                f"the mixed row width {self.ba_k} must be a multiple of {4 * WAVE}"
            )
        check_geometry(self.geo)
        if self.tasks > MI355X.cus:
            raise ValueError(f"{self.tasks} tasks: more than one round of CTAs")
        if self.replay and not 2 * self.spec <= self.replay <= REPLAY_MAX:
            raise ValueError(
                f"ReplaySSM records {self.replay} not in {2 * self.spec}..{REPLAY_MAX}"
            )

    # ---- the geometry (``geo``) the emit code reads
    @property
    def vh(self) -> int:  # v-heads
        return self.geo.v_heads

    @property
    def kh(self) -> int:  # k-heads
        return self.geo.k_heads

    @property
    def rep(self) -> int:  # v-heads per k-head
        return self.geo.rep

    @property
    def conv(self) -> int:  # conv channels q | k | v
        return self.geo.conv_dim

    @property
    def kw(self) -> int:  # q (and k) width
        return self.geo.key_width

    @property
    def outw(self) -> int:  # core width = v_heads x v_dim
        return self.geo.value_width

    @property
    def proj(self) -> int:  # proj width qkv | z
        return self.geo.conv_dim + self.geo.value_width

    # ---- b, a from the mixed row (``ba_k``): in round q, wave w serves row q * brows + w // bwr,
    # part w % bwr of it; lane l the 4-element units (part * bupl + i) * 64 + l, i < bupl
    @property
    def bwr(self) -> int:
        """Waves per mixed row: the most (as many rows as fit the CTA in one round) that split
        the row's 4-element units evenly over their lanes."""
        units = self.ba_k // 4 // WAVE
        return max(
            b for b in range(1, max(1, WAVES // self.spec) + 1) if units % b == 0
        )

    @property
    def bupl(self) -> int:  # 4-element units of the mixed row per lane
        return self.ba_k // 4 // WAVE // self.bwr

    @property
    def brows(self) -> int:  # mixed rows per round
        return WAVES // self.bwr

    @property
    def brounds(self) -> int:
        return -(-self.spec // self.brows)

    @property
    def z0(self) -> int:
        return self.conv

    @property
    def slot_nbytes(self) -> int:  # bytes of one fp32 state slot [v_heads, K, V]
        return self.vh * KD * VD * 4

    @property
    def vs(self) -> int:  # V columns per task
        return VD // self.splits

    @property
    def vw(self) -> int:  # V columns per lane over all MFMA chains (vs / 16)
        return self.vs // 16

    @property
    def jb(self) -> int:  # 16-row blocks of the 2S k | q rows
        return -(-2 * self.spec // 16)

    @property
    def nwk(self) -> int:
        """K-groups of waves: as many as the partials' LDS allows (the rest of the
        waves split the MFMA chains)."""
        return WAVES  # every wave holds all vs columns (the tail relies on it)

    @property
    def exchange_first(self) -> bool:
        return self.splits > 1 and self.spec <= 4

    @property
    def rmax(
        self,
    ) -> (
        int
    ):  # ReplaySSM: records a step replays at most (a flush keeps the cursor <= replay - spec)
        return self.replay - self.spec

    @property
    def rec_bytes(
        self,
    ) -> tuple[int, int, int]:  # ReplaySSM: one (slot, v-head) block of k, u, g records
        return self.replay * KD * 2, self.replay * VD * 2, self.replay * 4

    @property
    def rec_words(
        self,
    ) -> tuple[
        int, int, int
    ]:  # ReplaySSM: the words of the records a task folds (k, u, g)
        return self.rmax * KD // 2, self.rmax * self.vs // 2, self.rmax

    @property
    def fits(self) -> bool:
        # room for timeline stamps
        return self.L["_words"] * 4 <= MI355X.lds_bytes - 1024

    @property
    def nct(
        self,
    ) -> int:  # MFMA output chains per (K-group, row block): vw state + jb Gram
        return self.vw + self.jb

    def red_index(self, wk, jb, chain):
        """Word offset of a 16 x 16 partial tile in ``red``: column-major, column stride
        ``RED_COL`` (20: the b128 column writes and the row reads are bank-conflict free).
        """
        return ((wk * self.jb + jb) * self.nct + chain) * RED_TILE

    @property
    def cw(self) -> int:  # contiguous V columns per lane (MFMA chains per wave)
        return self.vw // (WAVES // self.nwk)

    @property
    def kpw(self) -> int:  # K rows per wave
        return KD // self.nwk

    @property
    def steps(self) -> int:  # MFMA k-steps per wave = K rows per lane
        return self.kpw // 4

    @property
    def group(self) -> int:
        """Task slots per k-head (the tasks take one round: no padding slots)."""
        return self.rep * self.splits

    @property
    def tasks(self) -> int:
        return self.batch * self.kh * self.group

    @property
    def npairs(self) -> int:  # conv channel pairs per task: q 64 | k 64 | v vs / 2
        return KD + self.vs // 2

    @property
    def nch(self) -> int:  # conv channels per task
        return 2 * self.npairs

    @property
    def upr(
        self,
    ) -> int:  # 16-byte poll units of proj per row: q 32 | k 32 | v vs / 4 | z vs / 4
        return KD // 2 + self.vs // 2

    @property
    def units(self) -> int:
        return self.spec * self.upr

    @property
    def qk_pairs(self) -> int:  # q / k window pairs each writer slice rewrites
        return KD // self.splits

    @property
    def dh(
        self,
    ) -> int:  # half-row stride of kkq / norm / dfin: S rounded up to 4 (16-byte rows)
        return -(-self.spec // 4) * 4

    @property
    def L(self) -> dict[str, int]:
        """LDS layout: block name -> 32-bit word offset (16-byte aligned blocks)."""
        S, a = self.spec, {}
        off = 0
        for name, words in (
            # the conv sequence: 3 history rows then the step's rows, one row stride
            # (``win`` starts right after ``hist``: npairs is a multiple of 4)
            ("hist", HIST * self.npairs),  # the history words (bf16 pairs)
            ("win", S * self.npairs),  # this step's qkv words
            ("z", S * self.vs // 2),  # z words
            # unused (b, a come from ``bap``); kept: it sets every later block's offset
            (
                "ba",
                2 * S,
            ),
            ("bap", 2 * S * self.bwr),  # b, a partials per wave of a row (``ba_k``)
            ("cv", S * self.nch),  # post-conv q | k | v (f32, raw: not normalized)
            ("gt", 2 * S),  # decay, beta (f32)
            ("kkq", S * 2 * self.dh),  # Gram rows: [t][k: k_t . k_j | q: k_t . q_j][j]
            ("norm", 2 * self.dh),  # [fk_j | fq_j], fq with 1 / sqrt(128)
            # matmul partials per K-group
            (
                "red",
                self.nwk * self.jb * self.nct * RED_TILE,
            ),
            # per column: [A_{-1}(k_j) | A_{-1}(q_j)][j]
            (
                "dfin",
                2 * self.dh * self.vs,
            ),
            ("out", S * self.vs),  # core out (f32, bf16-rounded)
            ("ev", S * self.vs),  # e_t per column (wave 0's deferred snapshots)
            ("trash", 4),  # target of stores that do not apply (branch-free)
        ):
            a[name] = off
            off += -(-words // 4) * 4
        if self.replay:
            a["glog"] = off  # the log decays g_t (f32): the g records
            off += -(-S // 4) * 4
            # the committed records, staged (``rec_words``): k [rmax][K / 2], u [rmax][vs / 2], g [rmax]
            a["rec"] = off
            off += -(-sum(self.rec_words) // 4) * 4
        a["_words"] = off
        return a


# ------------------------------------------------------------------ device helpers


def _arg(k, name):
    """Pointer-table entry ``name`` behind an opaque copy: its buffer descriptor is
    built at the use, not hoisted out of the task loop and held in SGPRs."""
    return fresh_s64(k.args[name])


def _mbox(k, region):
    """Region ``region``'s address, re-materialized at the use (see ``_arg``): the shared scratch resource
    (``mailbox.res``) holds only its base (every other field kept: a dual region's too).
    """
    a = k.mb[region]
    return dataclasses.replace(a, base=fresh_s64(a.base))


def _dv(x, d: int):
    """floor(x / d) for a traced x >= 0 (index math): shifts for powers of two, else an
    unsigned division (magic multiply), not the signed floor sequence of ``//``."""
    if d & (d - 1) == 0:
        return x >> (d.bit_length() - 1)
    return fx.Int32(fx.Uint32(x) // d)


def _md(x, d: int):
    """x mod d for a traced x >= 0."""
    if d & (d - 1) == 0:
        return x & (d - 1)
    return x - _dv(x, d) * d


def _lds_f4(ptr, off):
    v = fx.Vector(
        fx.ptr_load(ptr + off, result_type=fx.Vector.make_type(4, fx.Float32))
    )
    return [v[e] for e in range(4)]


def _lds_fn(ptr, off, n):
    """``n`` (>= 4) f32 from LDS at ``off``: 16-byte reads of ceil(n / 4) * 4 words (``off`` 16-byte
    aligned), sliced."""
    out = []
    for q in range(-(-n // 4)):
        out += _lds_f4(ptr, off + 4 * q)
    return out[:n]


def _half(word, odd):
    """The bf16 half ``odd`` (traced bool) of a packed word, as f32."""
    return odd.select(bf_hi(word), bf_lo(word))


# The gates and SiLUs are torch's own fp32 formulas (accurate exp / log1p, IEEE division), not the
# hardware approximations: bf16(silu(x)) and bf16(sigmoid(x)) then equal torch's on every bf16 input
# (the hardware versions differ on a few). The cost is a few VALU ops per element.


def _sigmoid(x):
    """torch's sigmoid: 1 / (1 + exp(-x))."""
    return fx.Float32(1.0) / (fx.Float32(1.0) + fmath.exp(fx.Float32(0.0) - x))


def _silu(x):
    """torch's silu: x / (1 + exp(-x))."""
    return x / (fx.Float32(1.0) + fmath.exp(fx.Float32(0.0) - x))


def _softplus(x):
    """torch softplus (beta 1, threshold 20): log1p(exp(x))."""
    return (x > 20.0).select(x, fx.Float32(fmath.log1p(fmath.exp(x))))


def _fma(a, b, c):
    return fx.Float32(fmath.fma(a, b, c))


def _dot(a, b):
    acc = a[0] * b[0]
    for e in range(1, len(a)):
        acc = _fma(a[e], b[e], acc)
    return acc


def _lane_offsets(n):
    return tuple(1 << i for i in range(n.bit_length() - 1))


def mfma_f32_4(a, b, c):
    """v_mfma_f32_16x16x4_f32: ``a``, ``b`` one f32 per lane, ``c`` a 4-vector."""
    a_, b_, c_, cbsz, abid, blgp = _split_mfma_operands([a, b, c, 0, 0, 0])
    return fx.Vector(
        _rocdl_ods.mfma_f32_16x16x4f32(
            T.vec(4, T.f32), a_, b_, c_, cbsz, abid, blgp
        ).result
    )


def e8m0_code(amax):
    """E8M0 code of ceil_pow2(amax / 448), clipped to [1, 254], 127 for 0 (exact:
    448 = 1.75 * 2^8, so the code follows from amax's exponent and mantissa)."""
    bits = amax.bitcast(fx.Int32)
    code = (
        ((bits >> 23) & 255)
        - 8
        + ((bits & 0x7FFFFF) > 0x600000).select(fx.Int32(1), fx.Int32(0))
    )
    code = fx.max(fx.min(code, fx.Int32(254)), fx.Int32(1))
    return fx.Int32((amax == 0.0).select(fx.Int32(ZERO_CODE), code))


@dataclass(frozen=True)
class _Task:
    """The traced per-task values every phase uses (attribute reads only)."""

    b: object  # request
    kh: object  # k-head
    jh: object  # v-head within the k-head group (0..rep - 1)
    s: object  # V-slice
    h: object  # v-head
    r0: object  # first row of the request


def _qk_pair(c: GDNPlan, task: _Task, p):
    """Global conv pair of task-local q | k pair ``p`` (0..127)."""
    return (p < KD // 2).select(
        task.kh * (KD // 2) + p, c.kw // 2 + task.kh * (KD // 2) + (p - KD // 2)
    )


def _v_pair0(c: GDNPlan, task: _Task):
    """Global conv pair of the task's first v pair."""
    return c.kw + _dv(task.h * VD + task.s * c.vs, 2)


def _seq_word(c: GDNPlan, lds, m, p):
    """Sequence word ``m`` (0..2 history, 3.. the step's rows) of task-local pair ``p``."""
    L = c.L
    assert L["win"] == L["hist"] + HIST * c.npairs
    return fx.ptr_load(lds + (L["hist"] + m * c.npairs + p))


# ---- small predicated helpers (a dynamic ``if`` only calls these)


@traced
def _store_unit(c: GDNPlan, lds, u, words):
    if u < c.units:
        j = _dv(u, c.upr)
        w = _md(u, c.upr)
        nqkv = KD // 2 + c.vs // 4
        dst = (w < nqkv).select(
            c.L["win"] + j * c.npairs + 2 * w,
            c.L["z"] + j * (c.vs // 2) + 2 * (w - nqkv),
        )
        fx.ptr_store(words[0], lds + dst)
        fx.ptr_store(words[1], lds + (dst + 1))


@traced
def _store_if(pred, val, ptr):
    if pred:
        fx.ptr_store(val, ptr)


@traced
def _gstore_word_if(pred, base, idx, word):
    if pred:
        bo.buffer_store(word, rsrc(base), idx)


@traced
def _put_if(pred, a, idx, val, tag):
    if pred:
        mb.put(a, idx, val, tag)


@traced
def _put_words_if(pred, a, idx, words, tag):
    if pred:
        chan.put_words(a, idx, words, tag)


# ---- phase C: conv + SiLU, gates, v window


def _conv_pair(win, taps):
    """Channels 2p, 2p + 1 of one row: ``win`` the 4 sequence words (oldest first) of pair
    p, ``taps`` its 4 tap words: each product rounded to bf16 and summed in fp32 from +0,
    SiLU, bf16."""
    w = [[bf_lo(taps[0]), bf_hi(taps[0]), bf_lo(taps[1]), bf_hi(taps[1])]]
    w.append([bf_lo(taps[2]), bf_hi(taps[2]), bf_lo(taps[3]), bf_hi(taps[3])])
    outs = []
    for ch, half in enumerate((bf_lo, bf_hi)):
        acc = fx.Float32(0.0)
        for tap in range(4):
            acc = acc + bf16_round(w[ch][tap] * half(win[tap]))
        outs.append(bf16_round(_silu(acc)))
    return outs


def _conv_item(c: GDNPlan, k, x, taps, p0=0, npairs=None):
    """Conv item ``x`` = (row j, task-local pair p0 + x % npairs): channels 2p, 2p + 1 of row j into ``cv``."""
    L = c.L
    npairs = npairs or c.npairs
    j = _dv(x, npairs)
    p = p0 + _md(x, npairs)
    win = [_seq_word(c, k.lds, j + t, p) for t in range(4)]
    outs = _conv_pair(win, taps)
    fx.ptr_store(
        fx.Vector.from_elements(outs, fx.Float32),
        k.ldsf + (L["cv"] + j * c.nch + 2 * p),
    )


@traced
def _conv_if(c: GDNPlan, k, x, taps):
    if x < c.spec * c.npairs:
        _conv_item(c, k, x, taps)


@traced
def _gates_if(c: GDNPlan, k, tid, a_log, dt_bias):
    if (tid >= WAVE) & (tid < WAVE + c.spec):
        L = c.L
        j = tid - WAVE
        b, a = _ba_sum(c, k, j)
        g = (fx.Float32(0.0) - fmath.exp(a_log)) * _softplus(a + dt_bias)
        fx.ptr_store(fx.Float32(fmath.exp(g)), k.ldsf + (L["gt"] + 2 * j))
        fx.ptr_store(bf16_round(_sigmoid(b)), k.ldsf + (L["gt"] + 2 * j + 1))
        if c.replay:
            fx.ptr_store(g, k.ldsf + (L["glog"] + j))


# ---- b, a from the mixed row (``GDNPlan.ba_k``)


def _ba_unit(c: GDNPlan, wave, lane, i):
    """Unit ``i`` (4 elements, 2 words) of this lane within its wave's part of a mixed row."""
    return (_md(wave, c.bwr) * c.bupl + i) * WAVE + lane


def _ba_weights(c: GDNPlan, k, task: _Task, wave, lane, units):
    """This lane's ``units`` (indices into its part) of the v-head's b and a weight rows
    (``ba_w`` rows h, v_heads + h): [side][unit] = 2 words."""
    r = rsrc(_arg(k, "ba_w"))
    out = []
    for side in range(2):
        base = (side * c.vh + task.h) * (c.ba_k // 2)
        out.append(
            [
                fx.Vector(
                    bo.buffer_load(
                        r,
                        base + 2 * _ba_unit(c, wave, lane, i),
                        vec_width=2,
                        dtype=T.i32,
                    )
                )
                for i in units
            ]
        )
    return out


def _ba_partials(c: GDNPlan, k, task: _Task, wave, lane, bw):
    """Poll the request's mixed rows (wave w: row q * brows + w // bwr of round q, its part
    w % bwr) and sum each wave's b, a partials: fp32 fma over the lane's units in order,
    then the wave butterfly. ``bw``: the weights (``_ba_weights``, all units, issued with the
    step-independent loads). -> [(store?, value, ``bap`` word)]."""
    S = c.spec
    xa = _mbox(k, k.names["ba"])
    rr = _dv(wave, c.bwr)
    part = _md(wave, c.bwr)
    out = []
    for q in range(c.brounds):
        row = q * c.brows + rr
        j = fx.min(row, S - 1)
        accs = [None, None]
        units = range(c.bupl)
        specs = [
            (xa, (task.r0 + j) * (c.ba_k // 2) + 2 * _ba_unit(c, wave, lane, i), 2)
            for i in units
        ]
        got = mb.poll(specs, k.tag)
        for side in range(2):
            acc = accs[side]
            for n in range(len(units)):
                for e in range(2):
                    x, wv = got[n][e], bw[side][n][e]
                    for half in (bf_lo, bf_hi):
                        acc = (
                            half(x) * half(wv)
                            if acc is None
                            else _fma(half(x), half(wv), acc)
                        )
            accs[side] = acc
        ok = (rr < c.brows) & (row < S)
        for side in range(2):
            acc = butterfly(accs[side], (32, 16, 8, 4, 2, 1))
            out.append((ok, acc, c.L["bap"] + (j * 2 + side) * c.bwr + part))
    return out


def _ba_store(c: GDNPlan, k, lane, parts):
    """Lane 0 of each wave: its partials into ``bap`` [row][side][part] (after the task's
    LDS-reuse barrier)."""
    for ok, acc, at in parts:
        _store_if(ok & (lane == 0), acc, k.ldsf + at)


def _ba_sum(c: GDNPlan, k, j):
    """(b, a) of row ``j``: the parts in order, rounded to bf16 (``bf16_linear``)."""
    L = c.L
    out = []
    for side in range(2):
        at = L["bap"] + (j * 2 + side) * c.bwr
        t = fx.ptr_load(k.ldsf + at)
        for p in range(1, c.bwr):
            t = t + fx.ptr_load(k.ldsf + (at + p))
        out.append(bf16_round(t))
    return out


def _window(c: GDNPlan, k, task: _Task, slot, x, p0, npairs, gpair):
    """Window item ``x`` = (window row n, pair p0 + x % npairs) := sequence word n + 1."""
    n = _dv(x, npairs)
    p = p0 + _md(x, npairs)
    word = _seq_word(c, k.lds, n + 1, p)
    return word, conv_word(c, slot, n, gpair(p))


# ---- phase D: state matmul (f32 MFMA) and Gram (bf16 MFMA)


def _kq_row_vals(c: GDNPlan, k, j, kk0, n):
    """``n`` contiguous values of k | q row ``j`` (j < S: k_j, < 2S: q_{j-S}, else 0) from K row ``kk0``."""
    L = c.L
    S = c.spec
    jc = fx.min(j, 2 * S - 1)
    isk = jc < S
    row = isk.select(jc, jc - S)
    off = L["cv"] + row * c.nch + isk.select(fx.Int32(KD), fx.Int32(0)) + kk0
    vals = _lds_fn(k.ldsf, off, n)
    live = j < 2 * S
    return [live.select(v, fx.Float32(0.0)) for v in vals]


def _matmul(c: GDNPlan, k, st, wk, wc, lane):
    """This wave's partials over its K rows, into ``red``: A_{-1} of the 2S k | q rows
    (chains 0 .. vw - 1: the state's V columns) and their Gram matrix (chains vw + jb2:
    the k | q rows themselves as B, the same registers as A), the A operand read from ``cv``.
    """
    L = c.L
    g = _dv(lane, 16)
    n = _md(lane, 16)
    kk0 = c.kpw * wk + c.steps * g
    a = [_kq_row_vals(c, k, jb * 16 + n, kk0, c.steps) for jb in range(c.jb)]
    for jb in range_constexpr(c.jb):
        for cc in range_constexpr(c.cw):
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for t in range_constexpr(c.steps):
                acc = mfma_f32_4(a[jb][t], st[t][cc], acc)
            fx.ptr_store(
                acc,
                k.ldsf
                + (
                    L["red"] + c.red_index(wk, jb, wc * c.cw + cc) + n * RED_COL + 4 * g
                ),
            )
        for jb2 in range_constexpr(jb, c.jb):
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for t in range_constexpr(c.steps):
                acc = mfma_f32_4(a[jb][t], a[jb2][t], acc)
            fx.ptr_store(
                acc,
                k.ldsf
                + (L["red"] + c.red_index(wk, jb, c.vw + jb2) + n * RED_COL + 4 * g),
            )


# ---- phase R: reduce the K-group partials; phase S: scalar recurrence per V column


def _red_sum(c: GDNPlan, k, jb, chain, col, row):
    L = c.L
    tot = fx.ptr_load(
        k.ldsf + (L["red"] + c.red_index(0, jb, chain) + col * RED_COL + row)
    )
    for wk in range(1, c.nwk):
        tot = tot + fx.ptr_load(
            k.ldsf + (L["red"] + c.red_index(wk, jb, chain) + col * RED_COL + row)
        )
    return tot


def _red_sum4(c: GDNPlan, k, jb, chain, col, row4):
    """Rows ``row4 .. row4 + 3`` of one partial-tile column summed over the K-groups (b128 reads)."""
    L = c.L
    tot = _lds_f4(k.ldsf, L["red"] + c.red_index(0, jb, chain) + col * RED_COL + row4)
    for wk in range(1, c.nwk):
        part = _lds_f4(
            k.ldsf, L["red"] + c.red_index(wk, jb, chain) + col * RED_COL + row4
        )
        tot = [a + b for a, b in zip(tot, part)]
    return tot


def _reduce_state(c: GDNPlan, k, x):
    """Item x = (V column vl, 4 k | q rows from 4 q4) -> dfin[vl][k | q half][j]: A(k_j), A(q_j)."""
    L = c.L
    S = c.spec
    vl = _md(x, c.vs)
    j0 = 4 * _dv(x, c.vs)
    tot = _red_sum4(c, k, _dv(j0, 16), _md(vl, c.vw), _dv(vl, c.vw), _md(j0, 16))
    for r in range(4):
        j = j0 + r
        slot = (j < S).select(vl * 2 * c.dh + j, (vl * 2 + 1) * c.dh + (j - S))
        fx.ptr_store(
            tot[r], k.ldsf + (j < 2 * S).select(L["dfin"] + slot, fx.Int32(L["trash"]))
        )


def _reduce_gram(c: GDNPlan, k, x):
    """Item x = (column col < 2S, 4 k rows from 4 iq) -> kkq[i][k | q half][j] (k_i . k_j,
    k_i . q_j); the diagonal k_i . k_i -> norm[0][i] = fk_i."""
    L = c.L
    S = c.spec
    col = _md(x, 2 * S)
    i0 = 4 * _dv(x, 2 * S)
    tot = _red_sum4(c, k, 0, c.vw + _dv(col, 16), _md(col, 16), i0)
    trash = fx.Int32(L["trash"])
    for r in range(4):
        i = i0 + r
        half = (col < S).select(i * 2 * c.dh + col, (i * 2 + 1) * c.dh + (col - S))
        fx.ptr_store(tot[r], k.ldsf + (i < S).select(L["kkq"] + half, trash))
        fx.ptr_store(
            _inv_norm(tot[r]),
            k.ldsf + ((i < S) & (col == i)).select(L["norm"] + i, trash),
        )


def _inv_norm(sq):
    """The l2 norm factor of a row with square sum ``sq`` (eps 1e-6 inside the root): ``1 / sqrt``
    correctly rounded."""
    return fx.Float32(1.0) / fx.Float32(fmath.sqrt(sq + 1e-6))


def _reduce_qdiag(c: GDNPlan, k, j):
    """q_j . q_j -> norm[1][j] = fq_j (with 1 / sqrt(128))."""
    i = c.spec + j
    tot = _red_sum(c, k, _dv(i, 16), c.vw + _dv(i, 16), _md(i, 16), _md(i, 16))
    fx.ptr_store(_inv_norm(tot) * RSQRT_KD, k.ldsf + (c.L["norm"] + c.dh + j))


@traced
def _reduce_if(c: GDNPlan, k, x):
    if x < c.vs * -(-2 * c.spec // 4):
        _reduce_state(c, k, x)


@traced
def _reduce_gram_if(c: GDNPlan, k, x):
    if (x >= 0) & (x < 2 * c.spec * -(-c.spec // 4)):
        _reduce_gram(c, k, x)


@traced
def _reduce_qdiag_if(c: GDNPlan, k, x):
    if (x >= 0) & (x < c.spec):
        _reduce_qdiag(c, k, x)


def _lane_cols(c: GDNPlan, lane):
    """The slice-local V columns a lane runs the scalar recurrence for (every wave holds
    all ``vs`` columns: lanes (g, n) of the 4 K-row groups share column set n), and for
    each of the lane's ``cw`` state columns the (source lane, index) holding its e_t."""
    g = _dv(lane, 16)
    n = _md(lane, 16)
    q = min(c.cw, 4)  # lane groups per column set
    own = [c.vw * n + _md(g, q) + 4 * i for i in range(-(-c.cw // 4))]
    src = []
    for cc in range(c.cw):
        gs = cc % 4
        src.append((16 * gs + n, cc // 4))
    return own, src


def _recurrence(c: GDNPlan, k, lane, st, sw, task: _Task, store: bool):
    """The scalar recurrence (module docstring) for this lane's columns, row by row. With
    ``store`` each row's snapshot is stored as soon as its e_t is known; otherwise out_t
    and e_t go to LDS (for the epilogue and the deferred snapshots)."""
    L = c.L
    S = c.spec
    own, src = _lane_cols(c, lane)
    al = [_lds_fn(k.ldsf, L["dfin"] + vl * 2 * c.dh, S) for vl in own]
    be = (
        [_lds_fn(k.ldsf, L["dfin"] + (vl * 2 + 1) * c.dh, S) for vl in own]
        if not store
        else None
    )
    fk = _lds_fn(k.ldsf, L["norm"], S)
    fq = _lds_fn(k.ldsf, L["norm"] + c.dh, S)
    gt = _lds_fn(k.ldsf, L["gt"], 2 * S)
    for t in range(S):
        dec = gt[2 * t]
        beta = gt[2 * t + 1]
        gk = _lds_fn(k.ldsf, L["kkq"] + t * 2 * c.dh, S)
        gq = _lds_fn(k.ldsf, L["kkq"] + (t * 2 + 1) * c.dh, S) if not store else None
        es = []
        for i, vl in enumerate(own):
            v = fx.ptr_load(k.ldsf + (L["cv"] + t * c.nch + 2 * KD + vl))
            u = (
                v - dec * fk[t] * al[i][t]
            ) * beta  # = u_t, the ReplaySSM record (normalized k)
            e = u * fk[t]
            es.append(e)
            if not store:
                o = fq[t] * _fma(e, gq[t], dec * be[i][t])
                # lanes sharing a column store the same value (no branch: rows stay one block)
                fx.ptr_store(bf16_round(o), k.ldsf + (L["out"] + t * c.vs + vl))
                fx.ptr_store(u if c.replay else e, k.ldsf + (L["ev"] + t * c.vs + vl))
            for j in range(t + 1, S):
                al[i][j] = _fma(e, gk[j], dec * al[i][j])
                if not store:
                    be[i][j] = _fma(e, gq[j], dec * be[i][j])
        if store:
            ecol = [
                lane_gather(es[which].bitcast(fx.Int32), sl).bitcast(fx.Float32)
                for sl, which in src
            ]
            _snapshot_row(c, k, lane, st, sw, task, t, ecol)


def _deferred_snapshots(c: GDNPlan, k, lane, st, sw, task: _Task):
    """Wave 0's snapshots, from the e_t it left in LDS (``ev``, its own writes)."""
    n = _md(lane, 16)
    for t in range(c.spec):
        rocdl.sched_barrier(0)
        ecol = _lds_fn(k.ldsf, c.L["ev"] + t * c.vs + c.vw * n, c.cw)
        _snapshot_row(c, k, lane, st, sw, task, t, ecol)


def _snapshot_row(c: GDNPlan, k, lane, st, sw, task: _Task, t, ecol):
    """Row t's state into its snapshot slot: S = dec_t S + k_t (x) e_t (``st`` updated in place)."""
    L = c.L
    g = _dv(lane, 16)
    n = _md(lane, 16)
    wk = uniform(_dv(fresh(k.tid), WAVE))  # nwk == 8: K-group = wave
    kk0 = c.kpw * wk + c.steps * g
    v0 = task.s * c.vs + c.vw * n
    dec = fx.ptr_load(k.ldsf + (L["gt"] + 2 * t))
    kv = _lds_fn(k.ldsf, L["cv"] + t * c.nch + KD + kk0, c.steps)
    slot = readlane(sw, t)
    base = ssm_slot(c, _arg(k, "ssm_pool"), slot)
    nbytes = slot_bytes(slot, c.slot_nbytes)
    for i in range(c.steps):
        st[i] = [_fma(kv[i], ecol[cc], dec * st[i][cc]) for cc in range(c.cw)]
        store_state(base, ssm_index(task.h, kk0 + i, v0), st[i], STATE_CM, nbytes)


# ---- phase E (wave 0): gated norm, MXFP8, core puts


def _epilogue(c: GDNPlan, k, lane, nw, task: _Task, put=True, finish=True):
    """Wave 0. Unit u = lane + 64 m: row u // (vs / 4), slice elements 4w .. 4w + 3 (one fp8
    word). ``put``: this slice's sum-of-squares partials; ``finish``: the siblings' partials,
    the gated norm, MXFP8 and the core puts (the two halves can run in separate regions:
    the units are recomputed from LDS)."""
    L = c.L
    per = c.vs // 4
    total = c.spec * per
    rounds = -(-total // WAVE)
    sq = _mbox(k, f"{k.names['block']}.sq")
    core = k.names["core"]
    units, parts = [], []
    for m in range(rounds):
        u = fx.min(lane + m * WAVE, total - 1)
        j = _dv(u, per)
        w = _md(u, per)
        o = _lds_f4(k.ldsf, L["out"] + j * c.vs + 4 * w)
        base = ((task.r0 + j) * c.vh + task.h) * c.splits
        if put:
            part = butterfly(_dot(o, o), _lane_offsets(per))
            _put_if(
                (lane + m * WAVE < total) & (w == 0), sq, base + task.s, part, k.tag
            )
        units.append((u, j, w, o, base))
    if not finish:
        return
    for m in range(rounds):
        u, j, w, o, base = units[m]
        specs = [(sq, base + q) for q in range(c.splits)]
        if m == 0:
            # the k-head group's "history read" flags ride along (lane l checks flag l % (rep splits)):
            # the q / k window writer then needs no poll round trip of its own
            specs.append(_hr_flag(c, k, task, _md(lane, c.rep * c.splits)))
        parts = mb.poll_f32(specs, k.tag)
        tot = parts[0]
        for q in range(1, c.splits):
            tot = tot + parts[q]
        rstd = hw_rsq(tot * (1.0 / VD) + c.geo.eps)
        zw = [
            fx.ptr_load(k.lds + (L["z"] + j * (c.vs // 2) + 2 * w + e))
            for e in range(2)
        ]
        z = [bf_lo(zw[0]), bf_hi(zw[0]), bf_lo(zw[1]), bf_hi(zw[1])]
        y = []
        for e in range(4):
            normed = bf16_round(o[e] * rstd)
            y.append(bf16_round(bf16_round(nw[m][e] * normed) * _sigmoid(z[e])))
        am = fmath.absf(y[0])
        for e in range(1, 4):
            am = fx.max(am, fmath.absf(y[e]))
        am = butterfly(am, (1, 2, 4), fx.max)
        code = e8m0_code(am)
        inv = ((fx.Int32(254) - code) << 23).bitcast(fx.Float32)  # 2^(127 - code)
        word = fp8_pack4(*[y[e] * inv for e in range(4)])
        e0 = (task.r0 + j) * c.outw + task.h * VD + task.s * c.vs + 4 * w
        ok = lane + m * WAVE < total
        _put_words_if(ok, _mbox(k, f"{core}.q"), _dv(e0, 4), [word], k.tag)
        _put_words_if(
            ok & (_md(w, GROUP // 4) == 0),
            _mbox(k, f"{core}.s"),
            _dv(e0, GROUP),
            [code],
            k.tag,
        )


# ---- phase F (wave 0 of the writer slices): the q / k window, after the k-head group consumed it


def _hr_flag(c: GDNPlan, k, task: _Task, i):
    """Spec of "history read" flag ``i`` (< rep splits) of the task's k-head group."""
    hr = _mbox(k, f"{k.names['block']}.hr")
    return (hr, (task.b * c.vh + task.kh * c.rep) * c.splits + i)


def _qk_window(c: GDNPlan, k, lane, slot, task: _Task):
    """The writer slices (v-head rep kh) rewrite the k-head's q / k window once all rep * splits
    tasks of the group flagged their history read (``<name>.hr``, put after their input
    phase); the flags were polled with the norm partials (``_epilogue``)."""
    n = (c.spec + 2) * c.qk_pairs
    for m in range(-(-n // WAVE)):
        x = fx.min(lane + m * WAVE, n - 1)
        word, idx = _window(
            c,
            k,
            task,
            slot,
            x,
            task.s * c.qk_pairs,
            c.qk_pairs,
            partial(_qk_pair, c, task),
        )
        _gstore_word_if(
            (lane + m * WAVE < n) & (slot >= 0), _arg(k, "conv_pool"), idx, word
        )


@traced
def _tail(c: GDNPlan, k, wave, lane, st, sw, nw, conv_slot, task: _Task):
    # each branch updates its own copy of the state list: a list mutated while tracing
    # one branch would hand that branch's values to the other
    if wave == 0:
        _wave0(c, k, lane, _copy(st), sw, nw, conv_slot, task)
    else:
        _others(c, k, wave, lane, _copy(st), sw, task)


@traced
def _others(c: GDNPlan, k, wave, lane, st, sw, task: _Task):
    _recurrence(c, k, lane, st, sw, task, True)


def _copy(st):
    return [list(x) for x in st]


@traced
def _w0_head(c: GDNPlan, k, wave, lane, st, sw, task: _Task):
    """``exchange_first``: wave 0's recurrence and partial puts, before any snapshot store."""
    if wave == 0:
        _recurrence(c, k, lane, _copy(st), sw, task, False)
        _epilogue(c, k, lane, None, task, put=True, finish=False)


@traced
def _tail_after_put(c: GDNPlan, k, wave, lane, st, sw, nw, conv_slot, task: _Task):
    if wave == 0:
        _epilogue(c, k, lane, nw, task, put=False, finish=True)
        _writer_if(c, k, lane, conv_slot, task)
        _deferred_snapshots(c, k, lane, _copy(st), sw, task)
    else:
        _others(c, k, wave, lane, _copy(st), sw, task)


def _wave0(c: GDNPlan, k, lane, st, sw, nw, conv_slot, task: _Task):
    """Wave 0: the recurrence without stores (out_t into LDS), the gated norm exchange and
    core puts, the q / k window, then its snapshots: its polls wait for no stores of its own.
    """
    _recurrence(c, k, lane, st, sw, task, False)
    _epilogue(c, k, lane, nw, task)
    _writer_if(c, k, lane, conv_slot, task)
    _deferred_snapshots(c, k, lane, st, sw, task)


@traced
def _writer_if(c: GDNPlan, k, lane, conv_slot, task: _Task):
    if task.jh == 0:
        _qk_window(c, k, lane, conv_slot, task)


# ---- ReplaySSM: the checkpoint plus the committed records, and this step's records


def _rec_block(c: GDNPlan, k, name, slot, h, nbytes):
    """Buffer resource over the (slot, v-head) block of record buffer ``name`` (``rec_k`` / ``rec_u`` / ``rec_g``:
    ATOM's ``replayssm_buf_*`` of the layer, [slots, v_heads, replay, ...]); no bytes for a negative slot.
    """
    base = (
        fx.Int64(_arg(k, name))
        + (fx.Int64(fx.max(slot, 0)) * c.vh + fx.Int64(h)) * nbytes
    )
    return rsrc(base, fx.Int64((slot >= 0).select(fx.Int32(nbytes), fx.Int32(0))))


def _record_blocks(c: GDNPlan, k, slot, task: _Task):
    kb, ub, gb = c.rec_bytes
    return (
        _rec_block(c, k, "rec_k", slot, task.h, kb),
        _rec_block(c, k, "rec_u", slot, task.h, ub),
        _rec_block(c, k, "rec_g", slot, task.h, gb),
    )


def _rec_items(c: GDNPlan, task: _Task):
    """(word count, the global word of item x within the record block) of k, u (this slice's columns) and g."""
    nk, nu, ng = c.rec_words
    half = c.vs // 2
    return (
        (nk, lambda x: x),
        (nu, lambda x: _dv(_dv(x, half) * VD + task.s * c.vs, 2) + _md(x, half)),
        (ng, lambda x: x),
    )


def _replay_loads(c: GDNPlan, k, slot, task: _Task):
    """Phase A, ReplaySSM: the cursor and the records a step can replay (``rmax``: k rows, u over this slice's
    columns, g), one word an item over the CTA's threads (staged into LDS by ``_records_to_lds``).
    """
    tid = fresh(k.tid)
    h = ld_i32(_arg(k, "write_pos"), fx.max(slot, 0))
    got = []
    for r, (n, gword) in zip(_record_blocks(c, k, slot, task), _rec_items(c, task)):
        for m in range(-(-n // THREADS)):
            x = fx.min(tid + m * THREADS, n - 1)
            got.append(fx.Int32(bo.buffer_load(r, gword(x), vec_width=1, dtype=T.i32)))
    return h, got


def _records_to_lds(c: GDNPlan, k, got):
    """Phase B: the loaded record words into ``rec`` (k, then u, then g)."""
    tid = fresh(k.tid)
    i, off = 0, c.L["rec"]
    for n in c.rec_words:
        for m in range(-(-n // THREADS)):
            x = tid + m * THREADS
            _store_if(x < n, got[i], k.lds + (off + fx.min(x, n - 1)))
            i += 1
        off += n


def _replay_fold(c: GDNPlan, k, st, slot, h_raw, task: _Task, kk0, v0):
    """The state after the committed records, over this lane's elements (``st``, from the checkpoint S_0), from the
    records staged in LDS: S <- e^{g_j} S + k_j (x) u_j for j < h (ATOM's ``replayssm``: the records hold the
    normalized k and the delta-corrected u, so the chain needs no state; a record at or past the cursor folds as
    e^0 S + 0). A step with ``h + 2 S > replay`` stores the result as the checkpoint and appends its records from 0.
    Returns (st, the step's first record row)."""
    # -1: a prefill's sentinel, nothing to fold
    h = uniform((slot >= 0).select(fx.max(h_raw, 0), fx.Int32(0)))
    nk, nu, _ = c.rec_words
    base = c.L["rec"]
    half = c.vs // 2
    for j in range(c.rmax):
        valid = fx.Int32(j) < h
        g = fx.ptr_load(k.ldsf + (base + nk + nu + j))
        dec = valid.select(fx.Float32(fmath.exp(g)), fx.Float32(1.0))
        kj = _bf16_words(
            _lds_words(k, base + j * (KD // 2) + _dv(kk0, 2), c.steps // 2)
        )
        uj = _bf16_words(
            _lds_words(k, base + nk + j * half + _dv(v0 - task.s * c.vs, 2), c.cw // 2)
        )
        for i in range(c.steps):
            # (a stale record is never read into the sum)
            kv = valid.select(kj[i], fx.Float32(0.0))
            st[i] = [
                _fma(kv, valid.select(uj[cc], fx.Float32(0.0)), dec * st[i][cc])
                for cc in range(c.cw)
            ]
    flush = h + 2 * c.spec > c.replay
    _flush_if(c, k, flush, st, slot, task, kk0, v0)
    return st, flush.select(fx.Int32(0), h)


def _lds_words(k, off, n):
    """``n`` 32-bit words of LDS at word ``off``."""
    return [fx.ptr_load(k.lds + (off + e)) for e in range(n)]


def _bf16_words(words):
    """bf16 pairs (i32 words) -> f32 values, low half first."""
    out = []
    for w in words:
        out += [bf_lo(w), bf_hi(w)]
    return out


@traced
def _flush_if(c: GDNPlan, k, flush, st, slot, task: _Task, kk0, v0):
    if flush:
        base = ssm_slot(c, _arg(k, "ssm_pool"), slot)
        nbytes = slot_bytes(slot, c.slot_nbytes)
        for i in range_constexpr(c.steps):
            store_state(base, ssm_index(task.h, kk0 + i, v0), st[i], STATE_CM, nbytes)


def _records(c: GDNPlan, k, lane, slot, base, task: _Task):
    """Wave 0, ReplaySSM: this step's records at rows ``base`` .. ``base + S - 1``: u over this slice's columns
    (``ev``), and from slice 0, once every slice has folded the old records (``<name>.rr``), the normalized k and g.
    """
    L = c.L
    S = c.spec
    _kb, ub, _gb = c.rec_bytes
    ru = _rec_block(c, k, "rec_u", slot, task.h, ub)
    half = c.vs // 2
    for m in range(-(-(S * half) // WAVE)):
        x = fx.min(lane + m * WAVE, S * half - 1)
        t = _dv(x, half)
        w = _md(x, half)
        e = L["ev"] + t * c.vs + 2 * w
        word = bf16_pair(
            fx.ptr_load(k.ldsf + e), fx.ptr_load(k.ldsf + (e + 1))
        ).bitcast(fx.Int32)
        _rstore_if(
            lane + m * WAVE < S * half,
            ru,
            _dv((base + t) * VD + task.s * c.vs, 2) + w,
            word,
        )
    _records_kg_if(c, k, lane, slot, base, task)


@traced
def _records_kg_if(c: GDNPlan, k, lane, slot, base, task: _Task):
    if task.s == 0:
        L = c.L
        S = c.spec
        kb, _ub, gb = c.rec_bytes
        if (
            c.splits > 1
        ):  # the other slices still fold the records this step overwrites after a flush
            rr = _mbox(k, f"{k.names['block']}.rr")
            specs = []
            for q in range_constexpr(1, c.splits):
                specs.append((rr, (task.b * c.vh + task.h) * c.splits + q, 1))
            mb.poll(specs, k.tag)
        rk = _rec_block(c, k, "rec_k", slot, task.h, kb)
        fk = _lds_fn(k.ldsf, L["norm"], S)
        for t in range_constexpr(S):  # k_t / |k_t|, K / 2 words a row: one a lane
            e = L["cv"] + t * c.nch + KD + 2 * lane
            word = bf16_pair(
                fx.ptr_load(k.ldsf + e) * fk[t], fx.ptr_load(k.ldsf + (e + 1)) * fk[t]
            )
            _rstore_if(
                lane < KD // 2,
                rk,
                (base + t) * (KD // 2) + lane,
                word.bitcast(fx.Int32),
            )
        rg = _rec_block(c, k, "rec_g", slot, task.h, gb)
        g = fx.ptr_load(k.ldsf + (L["glog"] + fx.min(lane, S - 1)))
        _rstore_if(lane < S, rg, base + lane, g.bitcast(fx.Int32))


@traced
def _rstore_if(pred, r, idx, word):
    if pred:
        bo.buffer_store(word, r, idx)


@traced
def _tail_replay(c: GDNPlan, k, wave, lane, st, nw, conv_slot, base, task: _Task):
    """ReplaySSM: no snapshots (a flush stored the checkpoint in phase A). Wave 0: the recurrence (out_t, u_t into
    LDS), the gated norm and core puts, the q / k window, the records."""
    if wave == 0:
        _recurrence(c, k, lane, _copy(st), None, task, False)
        _epilogue(c, k, lane, nw, task)
        _writer_if(c, k, lane, conv_slot, task)
        _records(c, k, lane, conv_slot, base, task)


# ---- the task


def _task(c: GDNPlan, k, task: _Task):
    L = c.L
    S = c.spec
    tid = fresh(k.tid)
    lane = _md(tid, WAVE)
    wave = uniform(_dv(tid, WAVE))
    wk = _md(wave, c.nwk)
    wc = _dv(wave, c.nwk)
    # (A) loads that do not depend on the step's input. The slot indices first; then
    # everything that does not need them (so the index latency overlaps their issue);
    # then the state and history behind the indices.
    if c.replay:
        # ReplaySSM: the request's one slot (its conv window's) holds the checkpoint
        cslot_v = ld_i32(_arg(k, "conv_slot"), task.b)
        rd_ssm_v = cslot_v
    else:
        rd_ssm_v = ld_i32(_arg(k, "ssm_read"), task.b)
        cslot_v = ld_i32(_arg(k, "conv_slot"), task.b)
    coff_v = ld_i32(_arg(k, "conv_off"), task.b)
    # row t's slot in lane t
    sw = (
        None
        if c.replay
        else ld_i32(_arg(k, "ssm_write"), task.r0 + fx.min(lane, S - 1))
    )
    taps = []
    nconv = S * c.npairs
    for m in range(-(-nconv // THREADS)):
        p = _md(fx.min(tid + m * THREADS, nconv - 1), c.npairs)
        gp = (p < KD).select(_qk_pair(c, task, p), _v_pair0(c, task) + (p - KD))
        tv = fx.Vector(
            bo.buffer_load(rsrc(_arg(k, "conv_w")), 4 * gp, vec_width=4, dtype=T.i32)
        )
        taps.append([tv[e] for e in range(4)])
    a_log_w = ld_i32(_arg(k, "a_log"), _dv(task.h, 2))
    dt_bias_w = ld_i32(_arg(k, "dt_bias"), _dv(task.h, 2))
    # b, a weights with the step-independent loads (one round of mixed rows)
    bw = _ba_weights(c, k, task, wave, lane, range(c.bupl))
    per = c.vs // 4
    nw = []
    for m in range(-(-(S * per) // WAVE)):
        we = task.s * c.vs + 4 * _md(fx.min(lane + m * WAVE, S * per - 1), per)
        nwv = fx.Vector(
            bo.buffer_load(
                rsrc(_arg(k, "norm_w")), _dv(we, 2), vec_width=2, dtype=T.i32
            )
        )
        nw.append([bf_lo(nwv[0]), bf_hi(nwv[0]), bf_lo(nwv[1]), bf_hi(nwv[1])])
    g = _dv(lane, 16)
    n = _md(lane, 16)
    rd_ssm = uniform(rd_ssm_v)
    src = ssm_slot(c, _arg(k, "ssm_pool"), rd_ssm)
    v0 = task.s * c.vs + c.vw * n + wc * c.cw
    kk0 = c.kpw * wk + c.steps * g
    nbytes = slot_bytes(rd_ssm, c.slot_nbytes)
    st = [
        load_state(src, ssm_index(task.h, kk0 + i, v0), c.cw, nbytes, STATE_LD_CM)
        for i in range(c.steps)
    ]
    if c.replay:
        rec_h, rec_got = _replay_loads(c, k, rd_ssm, task)
    cslot = uniform(cslot_v)
    coff = uniform(coff_v)
    hist = []
    nh = HIST * c.npairs
    # conv_word: < 2 GiB
    hist_r = rsrc(_arg(k, "conv_pool"), slot_bytes(cslot, 0x7FFFFFFF))
    for m in range(-(-nh // THREADS)):
        x = fx.min(tid + m * THREADS, nh - 1)
        p = _md(x, c.npairs)
        gp = (p < KD).select(_qk_pair(c, task, p), _v_pair0(c, task) + (p - KD))
        w = conv_word(c, cslot, coff + _dv(x, c.npairs), gp)
        hist.append(fx.Int32(bo.buffer_load(hist_r, w, vec_width=1, dtype=T.i32)))
    odd = _md(task.h, 2) == 1
    a_log = _half(a_log_w, odd)
    dt_bias = _half(dt_bias_w, odd)

    # (B) the step's proj elements: 16-byte units
    proj = _mbox(k, k.names["proj"])
    specs = []
    nqk = KD // 2
    nv = c.vs // 4
    for m in range(-(-c.units // THREADS)):
        u = fx.min(tid + m * THREADS, c.units - 1)
        j = _dv(u, c.upr)
        w = _md(u, c.upr)
        e = (w < nqk // 2).select(
            task.kh * KD + 4 * w,
            (w < nqk).select(
                c.kw + task.kh * KD + 4 * (w - nqk // 2),
                (w < nqk + nv).select(
                    2 * c.kw + task.h * VD + task.s * c.vs + 4 * (w - nqk),
                    c.z0 + task.h * VD + task.s * c.vs + 4 * (w - nqk - nv),
                ),
            ),
        )
        specs.append((proj, (task.r0 + j) * (c.proj // 2) + _dv(e, 2), 2))
    # b, a from the mixed row: polled first (it is ready a projection before proj)
    ba_parts = _ba_partials(c, k, task, wave, lane, bw)
    got = mb.poll(specs, k.tag)
    gpu.barrier()  # earlier tasks' LDS readers are done
    tid = fresh(k.tid)
    for m in range(len(got)):
        _store_unit(c, k.lds, tid + m * THREADS, got[m])
    _ba_store(c, k, _md(tid, WAVE), ba_parts)
    for m in range(len(hist)):
        x = tid + m * THREADS
        _store_if(x < nh, hist[m], k.lds + (L["hist"] + fx.min(x, nh - 1)))
    if c.replay:
        _records_to_lds(c, k, rec_got)
    gpu.barrier()
    # this task's conv history is consumed (in LDS): the q / k writer may overwrite it
    hr = _mbox(k, f"{k.names['block']}.hr")
    _put_if(
        fresh(k.tid) == 0,
        hr,
        (task.b * c.vh + task.h) * c.splits + task.s,
        fx.Int32(1),
        k.tag,
    )
    if c.replay:
        if c.splits > 1:
            # this slice's copy of the records is in LDS (before the barrier): slice 0 may now overwrite them
            rr = _mbox(k, f"{k.names['block']}.rr")
            _put_if(
                fresh(k.tid) == 0,
                rr,
                (task.b * c.vh + task.h) * c.splits + task.s,
                fx.Int32(1),
                k.tag,
            )
        st, rbase = _replay_fold(c, k, st, rd_ssm, rec_h, task, kk0, v0)

    # (C) conv + SiLU, gates, the v window
    tid = fresh(k.tid)
    for m in range(len(taps)):
        _conv_if(c, k, tid + m * THREADS, taps[m])
    _gates_if(c, k, tid, a_log, dt_bias)
    nvw = (S + 2) * (c.vs // 2)
    for m in range(-(-nvw // THREADS)):
        x = fx.min(tid + m * THREADS, nvw - 1)
        word, idx = _window(
            c, k, task, cslot, x, KD, c.vs // 2, lambda p: _v_pair0(c, task) + (p - KD)
        )
        _gstore_word_if(
            (tid + m * THREADS < nvw) & (cslot >= 0), _arg(k, "conv_pool"), idx, word
        )
    gpu.barrier()  # the matmul reads q / k from cv

    # (D) state matmul + Gram
    lane = _md(fresh(k.tid), WAVE)
    _matmul(c, k, st, wk, wc, lane)
    gpu.barrier()
    # (R) cross-wave reduction
    tid = fresh(k.tid)
    for m in range(-(-(c.vs * -(-2 * S // 4)) // THREADS)):
        _reduce_if(c, k, tid + m * THREADS)
    # Gram items and the q diagonal start on the next wave boundaries (else at small S all
    # three item kinds land in wave 0 and run one after the other)
    g0 = -(-(c.vs * -(-2 * S // 4)) // WAVE) * WAVE
    gitems = 2 * S * -(-S // 4)
    for m in range(-(-(g0 + gitems) // THREADS)):
        _reduce_gram_if(c, k, tid + m * THREADS - g0)
    _reduce_qdiag_if(c, k, tid - (-(-(g0 + gitems) // WAVE) * WAVE) % THREADS)
    gpu.barrier()
    # (S, W) the recurrence and the snapshots; wave 0 also (E, F)
    lane = _md(fresh(k.tid), WAVE)
    if c.replay:
        _tail_replay(c, k, wave, lane, st, nw, cslot, rbase, task)
    elif c.exchange_first:
        # small S: the snapshot stores wait for wave 0's partial puts (a put queued behind
        # them takes longer to reach its siblings than the short wait costs)
        _w0_head(c, k, wave, lane, st, sw, task)
        gpu.barrier()
        _tail_after_put(c, k, wave, lane, st, sw, nw, cslot, task)
    else:
        _tail(c, k, wave, lane, st, sw, nw, cslot, task)
    # The waves leave the task at very different times (wave 0's exchange tail vs the snapshot
    # streams), and the next stage on this CTA reuses the same LDS arena without a barrier:
    # no wave may leave while another still reads this task's LDS (out / z / sequence / cv).
    gpu.barrier()


@traced
def _gdn(k, t, c: GDNPlan, names: dict):
    k = _with_names(k, names)
    per_req = c.kh * c.group
    b = _dv(t, per_req)
    rem = _md(t, per_req)
    kh = _dv(rem, c.group)
    m = _md(rem, c.group)
    jh = _dv(m, c.splits)
    s = _md(m, c.splits)
    if (
        m < c.rep * c.splits
    ):  # (always: group == rep * splits; the emitted guard is kept)
        _task(c, k, _Task(b=b, kh=kh, jh=jh, s=s, h=kh * c.rep + jh, r0=b * c.spec))


def _with_names(k, names):
    from types import SimpleNamespace

    d = dict(vars(k))
    d["names"] = names
    d["args"] = {local: k.args[glob] for local, glob in names["args"].items()}
    return SimpleNamespace(**d)


# ------------------------------------------------------------------ the block


ARGS = ("conv_pool", "ssm_pool", "conv_slot", "conv_off", "ssm_read", "ssm_write")
# ReplaySSM: the checkpoint is the conv window's slot; ATOM's record buffers of the layer and the shared cursor
REPLAY_ARGS = (
    "conv_pool",
    "ssm_pool",
    "conv_slot",
    "conv_off",
    "rec_k",
    "rec_u",
    "rec_g",
    "write_pos",
)
WEIGHTS = ("conv_w", "a_log", "dt_bias", "norm_w")
# ``QuantizedLayer.retained`` names of the weights (HF Qwen3.5 / Qwen3-Next ``linear_attn``)
RETAINED = {
    "conv_w": "linear_attn.conv1d",
    "a_log": "linear_attn.A_log",
    "dt_bias": "linear_attn.dt_bias",
    "norm_w": "linear_attn.norm",
    "in_proj_b": "linear_attn.in_proj_b",  # BF16 b, a linears (``ba``)
    "in_proj_a": "linear_attn.in_proj_a",
}
BA_WEIGHTS = ("ba_w",)  # bf16 [2 * v_heads, K], in_proj_b rows then in_proj_a rows


class GDNCore(Block):
    """The GDN mixer core: ``proj`` (BF16 channel) and ``ba`` (the mixed row's BF16 channel) -> ``core``
    (MXFP8 channel).

    ``geo``: the ``GDNGeometry`` (head counts, fp32 state, eps; ``check_geometry``).
    ``args``: local arg name -> pointer-table name, for ``ARGS`` (state pools and
    slots, plain memory) and ``WEIGHTS`` + ``BA_WEIGHTS`` (``pack``); defaults ``<name>.<local>``.
    ``retained``: local weight name -> ``QuantizedLayer.retained`` name (default ``RETAINED``).
    ``splits``: V-slices per v-head; None (or the same value) = ``frozen_splits``.
    ``numerics``: the contract's rounding choices (``numerics.GDNNumerics``); only ``FLASH_NUMERICS``.
    ``ba``: the BF16 channel of the mixed row (b, a computed in the core with the BF16 weight ``ba_w``;
    ``proj`` is qkv | z; module docstring).
    ``replay``: ATOM's ReplaySSM record capacity when the verify runs on it (one checkpoint a request plus
    ``(k, u, g)`` records; ``REPLAY_ARGS``), 0 for the classic verify (a state snapshot per row; ``ARGS``).
    """

    def __init__(
        self,
        name="gdn",
        proj="proj",
        core="core",
        splits: int | None = None,
        args: dict | None = None,
        geo: GDNGeometry | None = None,
        retained: dict | None = None,
        numerics=None,
        ba: Channel | None = None,
        replay: int = 0,
    ):
        if geo is None:
            raise ValueError(
                f"GDNCore {name}: give its geometry (geo=GDNGeometry(...))"
            )
        check_geometry(geo)
        if ba is None or ba.fmt is not Fmt.BF16:
            raise ValueError(
                f"GDNCore {name}: b, a are computed from the mixed row's BF16 channel, got {ba}"
            )
        if (
            numerics is not None
            and (numerics.conv_rounding, numerics.gate, numerics.qk_norm)
            != FLASH_NUMERICS
        ):
            raise ValueError(
                f"GDNCore {name}: only the numerics {FLASH_NUMERICS} are built; got {numerics}"
            )
        self.name = name
        self.geo = geo
        self.ba = ba
        self.proj = Channel(proj, geo.conv_dim + geo.value_width, Fmt.BF16)
        self.core = (
            core
            if isinstance(core, Channel)
            else Channel(core, geo.value_width, Fmt.MXFP8)
        )
        self.splits = splits
        self.replay = replay
        self.args = {
            a: f"{name}.{a}"
            for a in (REPLAY_ARGS if replay else ARGS) + WEIGHTS + BA_WEIGHTS
        }
        self.args.update(args or {})
        self.retained = dict(RETAINED, **(retained or {}))

    def plan(self, sc: Scenario) -> GDNPlan:
        splits = frozen_splits(sc.batch)
        if self.splits not in (None, splits):
            raise ValueError(
                f"GDNCore {self.name}: batch {sc.batch} is built with {splits} splits, not {self.splits}"
            )
        return GDNPlan(sc.batch, sc.spec, splits, self.geo, self.ba.width, self.replay)

    def stages(self, sc: Scenario) -> list[Stage]:
        c = self.plan(sc)
        if not c.fits:
            raise ValueError(
                f"{c}: LDS {c.L['_words'] * 4} B does not fit; use more splits"
            )
        names = {
            "block": self.name,
            "proj": self.proj.name,
            "core": self.core.name,
            "args": dict(self.args),
            "ba": self.ba.name,
        }
        writes = self.core.regions(sc.rows, (self.name,)) + (
            Region(
                f"{self.name}.sq",
                sc.rows * c.vh * c.splits,
                (self.name,),
                exchange=True,
            ),
            Region(
                f"{self.name}.hr",
                sc.batch * c.vh * c.splits,
                (self.name,),
                exchange=True,
            ),
        )
        if c.replay and c.splits > 1:  # "records folded" flags (``_records_kg_if``)
            writes += (
                Region(
                    f"{self.name}.rr",
                    sc.batch * c.vh * c.splits,
                    (self.name,),
                    exchange=True,
                ),
            )
        emit = partial(_gdn, c=c, names=names)
        return [
            Stage(
                self.name,
                c.tasks,
                emit,
                lds_words=c.L["_words"],
                writes=writes,
                args=tuple(self.args.values()),
                key=(
                    c,
                    tuple(sorted((k, v) for k, v in names.items() if k != "args")),
                    tuple(sorted(self.args.items())),
                ),
            )
        ]

    def weight_specs(self) -> tuple:
        from atom.models.qwen38.mono.kernels.weights.source import WeightSpec

        g, bf = self.geo, (torch.bfloat16,)
        return (
            WeightSpec(self.args["conv_w"], "bf16", (g.conv_dim, g.conv_kernel), bf),
            WeightSpec(self.args["a_log"], "bf16", (g.v_heads,), bf, align=4),
            WeightSpec(self.args["dt_bias"], "bf16", (g.v_heads,), bf, align=4),
            WeightSpec(self.args["norm_w"], "bf16", (g.v_dim,), bf),
            WeightSpec(self.args["ba_w"], "bf16", (2 * g.v_heads, self.ba.width), bf),
        )

    def pack(self, layer) -> dict[str, torch.Tensor]:
        r, n, g = layer.retained, self.retained, self.geo
        return {
            self.args["conv_w"]: r[n["conv_w"]]
            .to(torch.bfloat16)
            .reshape(g.conv_dim, g.conv_kernel)
            .contiguous(),
            self.args["a_log"]: r[n["a_log"]].to(torch.bfloat16).contiguous(),
            self.args["dt_bias"]: r[n["dt_bias"]].to(torch.bfloat16).contiguous(),
            self.args["norm_w"]: r[n["norm_w"]].to(torch.bfloat16).contiguous(),
            self.args["ba_w"]: torch.cat((r[n["in_proj_b"]], r[n["in_proj_a"]]))
            .to(torch.bfloat16)
            .contiguous(),
        }

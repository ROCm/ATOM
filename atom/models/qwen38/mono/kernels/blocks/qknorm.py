# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``QKNormRope``: per-head q / k RMSNorm (zero-centred ``1 + w``) and partial RoPE of a fused
attention projection, with the step's K / V appended to Atom's token-major paged cache, for any
``AttnGeometry`` with head_dim 256 and a rotary_dim with a power-of-two lane partner
(``check_geometry``).

    proj [rows, proj_width] bf16 (q_proj per head [q, gate] | k_proj | v_proj from column
    ``col``)  ->  qn [rows, q_width] bf16 (q normed, rotated),
                  kv [rows, 2 kv_width] bf16 (k normed, rotated | v),
                  K / V of every step row appended to the paged cache

bit-exact: one fp32 square sum per head, ``v_rsq``, one rounding; RoPE ``bf16(bf16(x cos) +
bf16(x_pair sin))``. The step rows' own K / V reach a consumer in this launch through ``kv``
(never read back from the cache).

The caches (``kv_block`` tokens a block-table entry): bf16 ``k_cache`` / ``v_cache`` [pages, kv_block,
kv_heads, head_dim] (the drafter's), or with ``fp8`` ATOM Qwen's fp8 row layout (``_quant_atom``):
E4M3 ``k_cache`` / ``v_cache`` [pages, kv_block, kv_heads, head_dim] and fp32 ``k_scale`` / ``v_scale``
[pages, kv_block, kv_heads]; ``kv`` then carries the values the cache holds (code x scale).

One stage ``<prefix>.qk``: task = (row, group of 8 heads), wave ``w`` = head ``8 group + w``
of the ``q_heads + kv_heads`` (q heads first): it polls its head's ``head_dim`` values
(``head_dim / 64`` a lane), normalizes, rotates and publishes; a K wave also polls V,
publishes both and appends them at position ``seq_lens[b] + s`` (64-bit addresses; a negative
block-table entry, e.g. a padded request's, is no page: nothing is stored). RoPE rotates at
``seq_lens[b] + s + rope_off[b]``: ATOM's position of the row (an image request's MRoPE offset; 0 for text).

Args: ``seq_lens`` int32 [batch], ``rope_off`` int32 [batch], ``block_table`` int32 [batch, width], ``k_cache`` /
``v_cache`` (+ ``k_scale`` / ``v_scale``), ``<prefix>_w`` (``pack``: bf16 q_norm | k_norm) and ``rope``
([positions, rotary_dim] bf16: cos then sin of the rotary_dim / 2 frequencies,
``rope_table``; shareable by every block and layer with the same RoPE).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from atom.models.qwen38.mono.kernels.blocks.base import Block, Channel, Fmt
from atom.models.qwen38.mono.kernels.blocks.polls import mpoll_finish, mpoll_start
from atom.models.qwen38.mono.kernels.device import buffer_ops as bo
from atom.models.qwen38.mono.kernels.device import channels as chan
from atom.models.qwen38.mono.kernels.device.ops import (
    bf2_f32,
    bf16_pair,
    bf16_round,
    fp8_pack4,
    global_store,
    ld_i32,
    row_sum_torch,
    rsqrt_rn,
    rsrc,
    traced,
    uniform,
    wave_max,
    xshfl,
)
from atom.models.qwen38.mono.kernels.geometry import AttnGeometry
from atom.models.qwen38.mono.kernels.program import Scenario, Stage
from atom.models.qwen38.mono.kernels.target import MI355X

WAVES = MI355X.waves
LANES = MI355X.wave


def check_geometry(geo: AttnGeometry) -> None:
    """Raise unless the lane mapping supports ``geo``: ``head_dim / 64`` = 4 dims a lane and the
    rotary partner ``rotary_dim / 2`` dims away on a power-of-two lane offset."""
    dpl = geo.head_dim // LANES
    if geo.head_dim % LANES or dpl != 4:
        raise ValueError(f"QKNormRope: head_dim must be 256; got {geo}")
    off = (geo.rotary_dim // 2) // dpl
    if (geo.rotary_dim // 2) % dpl or off not in (1, 2, 4, 8, 16, 32):
        raise ValueError(
            f"QKNormRope: unsupported rotary_dim {geo.rotary_dim} for head_dim {geo.head_dim}"
        )


def rope_table(
    max_pos: int, rotary_dim: int, theta: float, device="cuda"
) -> torch.Tensor:
    """[max_pos, rotary_dim] bf16: cos then sin of the rotary_dim / 2 frequencies (dims
    rotary_dim / 2 .. repeat them)."""
    from atom.models.qwen38.mono.kernels.rope import rope_cos_sin

    cos, sin = rope_cos_sin(torch.arange(max_pos, device=device), rotary_dim, theta)
    half = rotary_dim // 2
    return torch.cat((cos[:, :half], sin[:, :half]), -1).contiguous()


# ------------------------------------------------------------------ device helpers (any head)


def head_norm(x, w, d: int, eps: float, lane=None):
    """Zero-centred RMSNorm of one head held by a wave (``len(x)`` dims a lane): the square sum
    in torch's order (``row_sum_torch``) and torch's correctly rounded ``rsqrt`` (``rsqrt_rn``);
    bf16-exact f32 out."""
    sq = [x[e] * x[e] for e in range(len(x))]
    inv = rsqrt_rn(row_sum_torch(sq, lane) * (1.0 / d) + eps)
    return [bf16_round(x[e] * inv * (1.0 + w[e])) for e in range(len(x))]


def head_rope(y, lane, cs, sn, rot: int, dpl: int):
    """HF partial RoPE (rotate-half on dims 0 .. rot - 1) in bf16 steps; lane ``l`` holds dims
    ``dpl l ..``, its partner (``± rot / 2``) lane ``l ^ (rot / 2 dpl)``."""
    off = (rot // 2) // dpl
    out = []
    for e in range(len(y)):
        partner = xshfl(y[e], off)
        r = ((lane & off) == 0).select(-partner, partner)
        v = bf16_round(bf16_round(y[e] * cs[e]) + bf16_round(r * sn[e]))
        out.append((lane < rot // dpl).select(v, y[e]))
    return out


def _ld_words(r, word, n):
    """``n`` (2 or 4) i32 words at ``word`` of buffer resource ``r``."""
    v = fx.Vector(bo.buffer_load(r, word, vec_width=n, dtype=T.i32))
    return [v[q] for q in range(n)]


def rope_pos(pos, max_positions: int):
    """The RoPE table row of position ``pos`` (traced): ``pos``, clamped to the table's rows (``max_positions``
    rows). In range it is ``pos`` (every output keeps its bits); past the table, the rows of a step nobody reads (an
    engine's graph-capture steps at the cache's end) read the last row instead of past the table's end (a fault when
    the table ends where the allocation does)."""
    return fx.max(fx.min(pos, fx.Int32(max_positions - 1)), fx.Int32(0))


def ld_rope(table, pos, lane, rot: int, dpl: int):
    """cos, sin (f32) of this lane's dims ``dpl lane ..`` (mod rot / 2) at ``pos`` from a
    ``rope_table`` at device address ``table``."""
    r = rsrc(table)
    half_words = rot // 4  # i32 words of cos (or sin) per position
    row = pos * (rot // 2) + (dpl * lane // 2) % half_words
    return [
        unpack_words(_ld_words(r, row + half * half_words, dpl // 2)) for half in (0, 1)
    ]


def ld_bf16(ptr, word, n_words):
    """``2 n_words`` bf16 at i32 ``word`` of a buffer -> f32 list."""
    return unpack_words(_ld_words(rsrc(ptr), word, n_words))


def unpack_words(words):
    vals = []
    for w in words:
        vals += list(bf2_f32(w))
    return vals


def pack_words(vals):
    return [
        bf16_pair(vals[2 * q], vals[2 * q + 1]).bitcast(fx.Int32)
        for q in range(len(vals) // 2)
    ]


# ------------------------------------------------------------------ the stage


@dataclass(frozen=True)
class QKCfg:
    """Every Python constant of the stage (its build key)."""

    batch: int
    spec: int
    qh: int
    kvh: int
    d: int
    rot: int
    eps: float
    proj_w: int  # elements per proj row
    col: int  # first column of q_proj in proj
    width: int  # block-table entries per request
    proj: str
    qn: str
    kv: str
    k_cache: str
    v_cache: str
    block_table: str
    seq_lens: str
    weights: str
    rope: str
    kv_block: int  # tokens a block-table entry
    # the caches: ATOM Qwen's fp8 row layout (module docstring), else bf16
    fp8: bool = False
    k_scale: str = ""
    v_scale: str = ""
    # int32 [batch]: the request's RoPE position minus its cache row (ATOM's MRoPE offset, 0 for text)
    rope_off: str = "rope_off"

    @property
    def page(self) -> int:
        """Tokens a block-table entry."""
        return self.kv_block

    @property
    def qstride(self) -> int:
        """Columns of q_proj a head: [q, gate]."""
        return 2 * self.d

    @property
    def kv_out(self) -> int:
        """``kv`` channel width: k | v."""
        return 2 * self.kv_width

    @property
    def rows(self) -> int:
        return self.batch * self.spec

    @property
    def heads(self) -> int:
        return self.qh + self.kvh

    @property
    def groups(self) -> int:
        return -(-self.heads // WAVES)

    @property
    def tasks(self) -> int:
        return self.rows * self.groups

    @property
    def dpl(self) -> int:
        return self.d // LANES

    @property
    def kv_width(self) -> int:
        return self.kvh * self.d


def _qk(k, t, c: QKCfg):
    _qk_body(k, t, c)


@traced
def _qk_body(k, t, c: QKCfg):
    r = t // c.groups
    hd = (t % c.groups) * WAVES + k.wave
    if hd < c.heads:
        _qk_head(k, c, r, hd)


def _qk_head(k, c: QKCfg, r, hd):
    """Wave work for head ``hd`` of row ``r`` (straight-line: every load goes out before the
    first wait, the length's first)."""
    lane = k.lane
    dpl = c.dpl
    b = r // c.spec
    n_raw = ld_i32(k.args[c.seq_lens], b)
    off = ld_i32(k.args[c.rope_off], b)
    is_q = hd < c.qh
    kh = fx.max(hd - c.qh, 0)
    base = r * c.proj_w + c.col + dpl * lane
    xcol = is_q.select(base + hd * c.qstride, base + c.qh * c.qstride + kh * c.d)
    vcol = base + c.qh * c.qstride + c.kv_width + kh * c.d
    a = k.mb[c.proj]
    specs = [(a, xcol // 2, dpl // 2, None)] + [(a, vcol // 2, dpl // 2, None)]
    first = mpoll_start(specs)
    w = ld_bf16(
        k.args[c.weights],
        is_q.select(fx.Int32(0), fx.Int32(c.d // 2)) + dpl // 2 * lane,
        dpl // 2,
    )
    pos = uniform(n_raw) + r % c.spec
    rp = rope_pos(pos + uniform(off), c.width * c.page)
    cs, sn = ld_rope(k.args[c.rope], rp, lane, c.rot, dpl)
    page = ld_i32(
        k.args[c.block_table], b * c.width + fx.min(pos // c.page, c.width - 1)
    )
    got = mpoll_finish(specs, first, k.tag)
    x, v = unpack_words(got[0]), unpack_words(got[1])
    y = head_rope(head_norm(x, w, c.d, c.eps, lane), lane, cs, sn, c.rot, dpl)
    _qk_out(k, c, r, hd, kh, is_q, page, pos, y, v)


@traced
def _qk_out(k, c: QKCfg, r, hd, kh, is_q, page, pos, y, v):
    """Publish a q head, or a k head with its v (and append both to the cache)."""
    dpl = c.dpl
    if is_q:
        chan.put_bf16(k.mb[c.qn], r * c.qh * c.d + hd * c.d + dpl * k.lane, y, k.tag)
    elif const_expr(c.fp8):
        _fp8_out(k, c, r, kh, page, pos, y, v)
    else:
        e0 = r * c.kv_out + kh * c.d + dpl * k.lane
        chan.put_bf16(k.mb[c.kv], e0, y, k.tag)
        chan.put_bf16(k.mb[c.kv], e0 + c.kv_width, v, k.tag)
        _append(k, c, page, pos % c.page, kh, y, v)


@traced
def _append(k, c: QKCfg, page, slot, kh, y, v):
    """This lane's dims of the step row's K (normed, rotated) and V at (page, slot, head kh);
    a negative page stores nothing."""
    if page >= 0:
        off = (fx.Int64(page) * c.page + fx.Int64(slot)) * (c.kvh * c.d * 2) + fx.Int64(
            kh
        ) * (c.d * 2)
        off = off + fx.Int64(k.lane * c.dpl * 2)
        _store_row(fx.Int64(k.args[c.k_cache]) + off, y)
        _store_row(fx.Int64(k.args[c.v_cache]) + off, v)


def _quant_atom(vals):
    """ATOM Qwen's FP8 cache row (``atom/model_ops/qwen4_exp/ops/qsa.py``): over the head's dims (the wave's lanes)
    ``amax = max(|x|, 1e-12)``, codes ``E4M3(clamp(x * (448 / amax)))``, scale ``amax / 448`` (fp32). -> (codes as
    one i32, the cached values ``code * scale`` (f32), the scale); every lane ends with the same scale.
    """
    am = fx.Float32(0.0)
    for x in vals:
        am = fx.max(am, fx.Float32(fmath.absf(x)))
    am = fx.max(wave_max(am), fx.Float32(1.0e-12))
    inv = fx.Float32(448.0) / am
    s = am / fx.Float32(448.0)
    q = [fx.min(fx.max(x * inv, fx.Float32(-448.0)), fx.Float32(448.0)) for x in vals]
    q = q + [fx.Float32(0.0)] * (4 - len(q))
    word = fp8_pack4(q[0], q[1], q[2], q[3])
    back = [
        fx.Float32(rocdl.cvt_f32_fp8(T.f32, word.ir_value(), e)) * s
        for e in range(len(vals))
    ]
    return word, back, s


def _fp8row_append(k, c: QKCfg, kh, page, pos, kw, ks, vw, vs):
    """This lane's ``dpl`` codes of the step row's key and value in ATOM Qwen's row layout: ``k_cache`` /
    ``v_cache`` [pages, block, kv_heads, d] E4M3 (a (token, head) row of ``d`` bytes), ``k_scale`` / ``v_scale``
    [pages, block, kv_heads] fp32."""
    row = (fx.Int64(page) * c.page + fx.Int64(pos % c.page)) * c.kvh + fx.Int64(kh)
    d0 = fx.Int64(k.lane * c.dpl)
    _store_word(fx.Int64(k.args[c.k_cache]) + row * c.d + d0, kw)
    _store_word(fx.Int64(k.args[c.v_cache]) + row * c.d + d0, vw)
    global_store(fx.Int64(k.args[c.k_scale]) + row * 4, ks)
    global_store(fx.Int64(k.args[c.v_scale]) + row * 4, vs)


@traced
def _fp8_out(k, c: QKCfg, r, kh, page, pos, y, v):
    """fp8 caches (module docstring): quantize, publish the cache's values, append. Every lane stores the same scales
    (no lane-conditional store after the wave maximum)."""
    dpl, lane = c.dpl, k.lane
    e0 = r * c.kv_out + kh * c.d + dpl * lane
    kw, kb, ks = _quant_atom(y)
    chan.put_bf16(k.mb[c.kv], e0, kb, k.tag)
    vw, vb, vs = _quant_atom(v)
    chan.put_bf16(k.mb[c.kv], e0 + c.kv_width, vb, k.tag)
    if page >= 0:
        _fp8row_append(k, c, kh, page, pos, kw, ks, vw, vs)


def _store_word(addr, word):
    """The 4 bytes (a lane's ``dpl`` = 4 codes) of ``word`` at ``addr``."""
    w = fx.Int32(word)
    global_store(addr, w)


def _store_row(addr, vals):
    words = pack_words(vals)
    global_store(
        addr, fx.Vector.from_elements(words, fx.Int32) if len(words) > 1 else words[0]
    )


class QKNormRope(Block):
    """proj (bf16 [rows, proj_width]) -> qn, kv (bf16), appending the step's K / V.

    ``geo``: the ``AttnGeometry``; ``proj_width`` / ``col``: the proj channel's row width and
    the column of q_proj in it (default: the attention projection alone); ``q_norm`` /
    ``k_norm``: the ``QuantizedLayer.retained`` names of the norm weights; ``rope``: the RoPE
    table's arg name (``max_positions`` rows); ``fp8`` (``fp8_row`` must equal it: the row layout is
    the only fp8 one) / ``kv_block``: the caches (module docstring)."""

    name = "qknorm"

    def __init__(
        self,
        geo: AttnGeometry,
        proj: str = "proj",
        qn: str = "qn",
        kv: str = "kv",
        proj_width: int | None = None,
        col: int = 0,
        table_width: int = 1024,
        prefix: str = "qk",
        k_cache: str = "k_cache",
        v_cache: str = "v_cache",
        block_table: str = "block_table",
        seq_lens: str = "seq_lens",
        weights: str | None = None,
        rope: str = "rope",
        rope_off: str = "rope_off",
        q_norm: str = "self_attn.q_norm",
        k_norm: str = "self_attn.k_norm",
        fp8: bool = False,
        k_scale: str = "k_scale",
        v_scale: str = "v_scale",
        kv_block: int = 0,
        fp8_row: bool = False,
    ):
        check_geometry(geo)
        if fp8_row != fp8 or not kv_block:
            raise ValueError(f"QKNormRope: fp8 caches are the row layout (fp8_row) and kv_block is given; got "
                             f"fp8={fp8}, fp8_row={fp8_row}, kv_block={kv_block}")  # fmt: skip
        self.fp8 = fp8
        self.kv_block = kv_block
        self.fp8_names = {"k_scale": k_scale, "v_scale": v_scale} if fp8 else {}
        self.geo, self.prefix, self.table_width = geo, prefix, table_width
        self.q_norm, self.k_norm = q_norm, k_norm
        self.proj_width = proj_width or (2 * geo.q_width + 2 * geo.kv_width)
        self.col = col
        self.names = {
            "proj": proj,
            "qn": qn,
            "kv": kv,
            "k_cache": k_cache,
            "v_cache": v_cache,
            "block_table": block_table,
            "seq_lens": seq_lens,
            "weights": weights or f"{prefix}_w",
            "rope": rope,
            "rope_off": rope_off,
        }
        self.in_ch = Channel(proj, self.proj_width, Fmt.BF16)
        self.qn_ch = Channel(qn, geo.q_width, Fmt.BF16)
        self.kv_ch = Channel(kv, 2 * geo.kv_width, Fmt.BF16)

    def cfg(self, sc: Scenario) -> QKCfg:
        g = self.geo
        return QKCfg(
            sc.batch,
            sc.spec,
            g.q_heads,
            g.kv_heads,
            g.head_dim,
            g.rotary_dim,
            g.eps,
            self.proj_width,
            self.col,
            self.table_width,
            **self.names,
            kv_block=self.kv_block,
            fp8=self.fp8,
            **self.fp8_names,
        )

    def stages(self, sc: Scenario) -> list[Stage]:
        c = self.cfg(sc)
        name = f"{self.prefix}.qk"
        n = self.names
        caches = (n["k_cache"], n["v_cache"]) + tuple(self.fp8_names.values())
        return [
            Stage(
                name,
                c.tasks,
                partial(_qk, c=c),
                writes=self.qn_ch.regions(sc.rows, (name,))
                + self.kv_ch.regions(sc.rows, (name,)),
                args=(
                    n["seq_lens"],
                    n["block_table"],
                    *caches,
                    n["weights"],
                    n["rope"],
                    n["rope_off"],
                ),
                key=(c,),
            )
        ]

    @property
    def max_positions(self) -> int:
        return self.table_width * self.kv_block

    def pack(self, layer) -> dict[str, torch.Tensor]:
        qn, kn = layer.retained[self.q_norm], layer.retained[self.k_norm]
        w = torch.cat(
            (qn.to(torch.bfloat16).flatten(), kn.to(torch.bfloat16).flatten())
        ).contiguous()
        rope = rope_table(
            self.max_positions, self.geo.rotary_dim, self.geo.rope_theta, qn.device
        )
        return {self.names["weights"]: w, self.names["rope"]: rope}

    def weight_specs(self) -> tuple:
        from atom.models.qwen38.mono.kernels.weights.source import WeightSpec

        g = self.geo
        return (
            WeightSpec(
                self.names["weights"], "bf16", (2 * g.head_dim,), (torch.bfloat16,)
            ),
            WeightSpec(
                self.names["rope"],
                "bf16",
                (self.max_positions, g.rotary_dim),
                (torch.bfloat16,),
            ),
        )

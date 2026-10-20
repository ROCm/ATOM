"""Tagged-pair mailboxes for the resident layer kernel.

Every value one CTA hands to another is an aligned 8-byte ``(u32 payload, u32
tag)`` pair written by a single 8- or 16-byte store. ``tag = step * LAYER_SLOTS
+ layer + 1`` is unique per launch, so the data is its own ready flag: no
separate flag, no fence, no clearing between launches. A consumer re-loads its
whole batch of pairs until every tag matches.

Cache rules, measured on gfx950 (coarse-grained VRAM, no tgsplit):

- Polls use sc1 loads. Plain and sc0 loads hit the vL1D, and ``buffer_inv sc0``
  does not drop it, so a plain-load poll can spin forever on a stale line.
- AGENT scope (``CM_DEV`` store + ``CM_DEV`` poll) is visible on every XCD,
  ~1.0 us round trip. It is used everywhere.
- Bulk data (KV rows, index lists): plain stores, then ``vmcnt(0)``, a
  barrier, and thread 0 alone releases with ``buffer_wbl2 sc1`` and stores one
  ready pair; on the consumer side thread 0 polls and acquires with
  ``buffer_inv sc1``, then a barrier, before plain loads. Fence in one wave
  only: 8 waves each writing back the L2 more than doubles the hand-off.

Adapted from FlyDSL ``kernels/monokernel/glm/kernel.py`` (``put``, ``put2``,
``put_bf``, ``poll``, ``pre_poll``) at tag v0.4.0.dev963.
"""

from __future__ import annotations

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from atom.models.qwen4_exp.mono.kernels import buffer_ops as bo
from atom.models.qwen4_exp.mono.kernels.device_ops import rsrc, spin_pause, uniform

# One launch per decoder layer; the 48 layers of a step fit with room to spare,
# and a launch fusing several layers would still give each its own slot.
LAYER_SLOTS = 64

# Buffer instruction cache-policy bits on gfx94x/gfx950.
CM_SC1 = 16
CM_DEV = CM_SC1  # agent scope

POLL_MAX = 12  # pairs loaded per retry; bounds live registers


def launch_tag(step_addr, layer):
    """This launch's tag from the device step counter and the layer id."""
    step_value = uniform(bo.buffer_load(rsrc(step_addr), 0, vec_width=1, dtype=T.i32))
    return step_value * LAYER_SLOTS + layer + 1


def put(addr, i, v, tag, cm=CM_DEV):
    """Pair ``i`` := (v, tag); ``v`` is an f32 or int32 bits."""
    bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
    bo.buffer_store(
        fx.Vector.from_elements([bits, tag], fx.Int32),
        rsrc(addr),
        fx.Int32(i) * 2,
        cache_modifier=cm,
    )


def put2(addr, i, v0, v1, tag, cm=CM_DEV):
    """Pairs ``i`` and ``i + 1`` (``i`` even) in one 16-byte store."""
    words = []
    for v in (v0, v1):
        words += [
            v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v),
            tag,
        ]
    bo.buffer_store(
        fx.Vector.from_elements(words, fx.Int32),
        rsrc(addr),
        fx.Int32(i) * 2,
        cache_modifier=cm,
    )


def bf16_pair(a, b):
    """Two f32 -> one int32 word holding (bf16(a), bf16(b)), RNE."""
    return (
        fx.Vector.from_elements([a, b], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0]
    )


def put_bf(addr, i, vs, tag, cm=CM_DEV):
    """Elements ``i .. i + len(vs)`` (2 or 4, ``i`` aligned) as packed bf16 pairs:
    pair ``i // 2 + j`` := (bf16(vs[2j]) | bf16(vs[2j + 1]) << 16, tag)."""
    words = []
    for j in range_constexpr(len(vs) // 2):
        words += [bf16_pair(vs[2 * j], vs[2 * j + 1]), tag]
    bo.buffer_store(
        fx.Vector.from_elements(words, fx.Int32),
        rsrc(addr),
        fx.Int32(i),
        cache_modifier=cm,
    )


def bf2_f32(w):
    """Packed bf16 pair word -> (f32 low, f32 high)."""
    return (w << 16).bitcast(fx.Float32), (w & fx.Int32(-65536)).bitcast(fx.Float32)


def load_pairs(specs, cm=CM_DEV):
    """Load ``specs`` = [(base_addr, pair index, npairs in {1, 2})] with no wait
    between them; returns one int32 vector of (value, tag) words."""
    words = []
    for b, i, n in specs:
        w = fx.Vector(
            bo.buffer_load(
                rsrc(b),
                fx.Int32(i) * 2,
                vec_width=2 * n,
                dtype=T.i32,
                cache_modifier=cm,
            )
        )
        words += [w[e] for e in range_constexpr(2 * n)]
    return fx.Vector.from_elements(words, fx.Int32)


def pending(v, tag):
    """True while any tag in a :func:`load_pairs` vector differs from ``tag``."""
    bad = v[1] != tag
    for e in range_constexpr(3, len(v), 2):
        bad = bad | (v[e] != tag)
    return bad


def split_values(v, specs):
    """Value words of a :func:`load_pairs` vector, one list per spec."""
    outs, e = [], 0
    for _, _, n in specs:
        outs.append([v[e + 2 * q] for q in range_constexpr(n)])
        e += 2 * n
    return outs


# Comment emitted inside every poll loop so the loops can be found in the
# final ISA; it assembles to nothing.
POLL_MARK = "mono-poll"


def _poll_mark():
    llvm.InlineAsmOp(None, [], f"; {POLL_MARK}", "", has_side_effects=True)


def _poll_spin(specs, tag, cm, pause):
    v = load_pairs(specs, cm)
    while pending(v, tag):
        pause()
        _poll_mark()
        v = load_pairs(specs, cm)
    return v


_poll_spin = ASTRewriter.transform(_poll_spin)


def poll_raw(specs, tag, cm=CM_DEV):
    """Spin until every pair in ``specs`` (at most ``POLL_MAX`` of them) carries
    ``tag``; returns the :func:`load_pairs` vector. A side-effecting ``s_nop``
    between retries keeps the loads inside the loop."""
    assert 0 < len(specs) <= POLL_MAX
    return _poll_spin(specs, tag, cm, spin_pause)


def poll(specs, tag, cm=CM_DEV):
    """Batched poll: one list of int32 value words per spec. Batches larger than
    ``POLL_MAX`` are split, each costing one round trip after its last producer."""
    if const_expr(len(specs) == 0):
        return []
    out = []
    for first in range_constexpr(0, len(specs), POLL_MAX):
        batch = specs[first : first + POLL_MAX]
        out += split_values(poll_raw(batch, tag, cm), batch)
    return out


def _pre_poll_wave0(hint, n, tag):
    lane = fx.thread_idx.x % 64
    # A wave-uniform branch: a divergent one turns buffer resources that GVN
    # merges across it (the hint's region polled again after) into VGPR phis
    # and their loads into waterfall loops.
    if uniform(fx.thread_idx.x // 64) == 0:
        specs = []
        for r in range_constexpr((n + 63) // 64):
            b, i = hint(fx.min(lane + r * 64, n - 1))
            specs.append((b, i, 1))
        poll_raw(specs, tag, CM_DEV)
    gpu.barrier()


_pre_poll_wave0 = ASTRewriter.transform(_pre_poll_wave0)


def pre_poll(hint, n, tag):
    """Wave 0 spins on one hint pair per producer -- lane ``l`` on producers
    ``l, l + 64, ...`` with ``hint(m) -> (base_addr, pair index)`` -- then the CTA
    barriers. Every thread must call this.

    Only a throttle: callers still :func:`poll` every payload pair. Measured on
    gfx950: with all 512 threads of all 256 CTAs spinning on 8 or more pairs
    each, the sc1 load flood starves the producers' stores and the grid
    livelocks. Pre-polling first leaves 64 lanes per CTA spinning, and the
    payload poll then normally succeeds on its first load.
    """
    assert 0 < n <= 64 * POLL_MAX
    _pre_poll_wave0(hint, n, tag)

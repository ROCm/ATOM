# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 tagged mailboxes, cache scopes and steady-clock diagnostics.

Shared gfx950 address/trace primitives are reused unchanged. The gfx1250
scope encoding, wave32 bounded polling and store-counter publication live
with this backend rather than extending the common mono APIs.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from atom.mono.device.ops import CM_DEV, CM_SYS, ballot, bf16_pair, rsrc, traced
from atom.mono.device.sync import Addr, DIAG_TICKS, POLL_MAX, _noted, _spin
from atom.mono.plan.layout import DIAG_WORDS
from atom.mono.plan.trace import Space

# gfx12 scopes occupy bits[3:4]; gfx950's 17 means device+NT here.
CM_SYS_GFX1250 = 24
# gfx12 TH_LOAD_NT is 1; gfx950's 2 would select high-temporal loads.
CM_NT_GFX1250 = 1


def memrealtime(steady=False):
    """The 100 MHz wall clock; gfx1250 selects GET_REALTIME via LLVM."""
    intrinsic = "llvm.readsteadycounter" if steady else "llvm.amdgcn.s.memrealtime"
    return fx.Int64(llvm.call_intrinsic(T.i64, intrinsic, [], [], []))


@traced
def _spin_bounded(
    load_all,
    pending,
    diag,
    region_id,
    index,
    tag,
    steady=False,
    failure=None,
    wait_ticks=DIAG_TICKS,
):
    """``_spin`` giving up after wait_ticks: the CTA's record in ``diag`` names the
    wait (the runner raises with it after the step), and the kernel runs on."""
    t0 = memrealtime(steady)
    v = load_all()
    late = fx.Int32(0)

    def waiting(words):
        bad = pending(words)
        if const_expr(steady):
            # All wave32 lanes reissue loads and leave the wait together.
            bad = ballot(bad) != 0
        return bad

    while waiting(v) & (late == 0):
        if const_expr(steady):
            # On gfx1250 the original one-unit polling interval can delay
            # producer progress when many wave32 consumers are waiting.
            # Retain the same wait set/publication protocol, but back off
            # stale reads. Verified with the full front and 216 consumers.
            rocdl.s_sleep(64)
        else:
            rocdl.s_sleep(1)
        v = load_all()
        timed_out = memrealtime(steady) - t0 > wait_ticks
        if const_expr(failure is not None):
            stopped = (
                fx.generic_load(
                    failure, memory_order=fx.AtomicOrdering.Monotonic, syncscope="agent"
                )
                != 0
            )
            timed_out = timed_out | (ballot(stopped) != 0)
        late = timed_out.select(fx.Int32(1), fx.Int32(0))
    if late != 0:
        bid = fx.block_idx.x
        if const_expr(failure is not None):
            fx.atomic_cas(
                failure,
                fx.Int32(0),
                fx.Int32(30_000_000) + fx.Int32(bid),
                success_ordering=fx.AtomicOrdering.Release,
                failure_ordering=fx.AtomicOrdering.Monotonic,
                syncscope="agent",
            )
        # the CTA's first give-up of the step: the later ones wait on its fallout
        seen = bo.buffer_load(
            rsrc(diag),
            bid * DIAG_WORDS,
            vec_width=1,
            dtype=T.i32,
            cache_modifier=CM_SYS_GFX1250 if steady else CM_SYS,
        )
        if fx.Int32(seen) == 0:
            at = fx.Int32(memrealtime(steady) & 0x7FFFFFFF)
            record = [1, region_id, index, tag, v[1], bid, at, 0]
            for h in range_constexpr(2):  # 16 B a store
                bo.buffer_store(
                    fx.Vector.from_elements(
                        [fx.Int32(w) for w in record[4 * h : 4 * h + 4]], fx.Int32
                    ),
                    rsrc(diag),
                    bid * DIAG_WORDS + 4 * h,
                    cache_modifier=CM_SYS_GFX1250 if steady else CM_SYS,
                )
    return v


class Mailbox:
    """Tagged-pair hand-off between CTAs (device scope) and GPUs (system scope).

    Every 32-bit value is stored next to this launch's tag; a consumer
    polls the payload until all tags match, so a hand-off costs one round trip
    (no store drain, no separate flag). The tag is ``layer + 1``: it tells the
    layers of one step apart, and the runner zeroes every mailbox at a step's
    start (``StepMailboxes.begin_step``), so a pair from an earlier step never
    carries a live tag. Every address is an ``Addr``: its space picks the scope.
    """

    def __init__(
        self,
        layer,
        diag=None,
        region_ids=None,
        *,
        steady=False,
        failure=None,
        wait_ticks=DIAG_TICKS,
    ):
        """``diag`` (a debug build): the address of the per-CTA wait records, every
        wait bounded (``_spin_bounded``); ``region_ids``: region name -> the id a
        record names it by."""
        self.tag = layer + 1
        self.diag = diag
        self.region_ids = region_ids or {}
        self.steady = steady
        self.failure = failure
        if not isinstance(wait_ticks, int) or wait_ticks <= 0:
            raise ValueError("mailbox wait_ticks must be a positive integer")
        self.wait_ticks = wait_ticks
        self.scope_cm = {
            Space.SCRATCH: CM_DEV,
            Space.PEER: CM_SYS_GFX1250,
        }
        if steady and diag is None:
            raise ValueError("gfx1250 peer waits require bounded diagnostics")

    def put(self, a, i, v):
        bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
        bo.buffer_store(
            fx.Vector.from_elements([bits, self.tag], fx.Int32),
            rsrc(_noted(a, "put")),
            i * 2,
            cache_modifier=self.scope_cm[a.space],
        )

    def put_bf(self, a, i, vs):
        """Elements i .. i+len(vs) (2 or 4, i aligned) as packed bf16 pairs, one store."""
        words = []
        for j in range_constexpr(len(vs) // 2):
            words += [bf16_pair(vs[2 * j], vs[2 * j + 1]).bitcast(fx.Int32), self.tag]
        bo.buffer_store(
            fx.Vector.from_elements(words, fx.Int32),
            rsrc(_noted(a, "put")),
            i,
            cache_modifier=self.scope_cm[a.space],
        )

    def put_words(self, a, i, words):
        """Raw 32-bit words (1 or 2) at pairs i.. as (word, tag). Integer payloads
        (fp8 words) must come this way: through an f32 value a NaN bit pattern is
        not kept."""
        vals = []
        for w in words:
            vals += [fx.Int32(w), self.tag]
        bo.buffer_store(
            fx.Vector.from_elements(vals, fx.Int32),
            rsrc(_noted(a, "put")),
            i * 2,
            cache_modifier=self.scope_cm[a.space],
        )

    def poll(self, specs, batch=POLL_MAX):
        """Batched poll of pairs ``specs`` = [(Addr, pair index, npairs in {1, 2})],
        each loaded at its own space's scope.

        All pairs are re-loaded together while any tag is stale, so a batch costs
        one round trip after its last producer lands. Returns the value words."""
        if const_expr(len(specs) == 0):
            return []
        if const_expr(len(specs) > batch):
            return self.poll(specs[:batch], batch) + self.poll(specs[batch:], batch)
        cms = [self.scope_cm[a.space] for a, _, _ in specs]
        bases = [_noted(a, "poll") for a, _, _ in specs]
        tag = self.tag

        def load_all():
            words = []
            for b, cm, (_, i, n) in zip(bases, cms, specs):
                w = fx.Vector(
                    bo.buffer_load(
                        rsrc(b),
                        fx.Int32(i) * 2,
                        vec_width=2 * n,
                        dtype=T.i32,
                        cache_modifier=cm,
                    )
                )
                words += [w[e] for e in range(2 * n)]
            return fx.Vector.from_elements(words, fx.Int32)

        nw = sum(2 * n for _, _, n in specs)

        def pending(v):
            bad = v[1] != tag
            for e in range_constexpr(3, nw, 2):
                bad = bad | (v[e] != tag)
            return bad

        if const_expr(self.diag is None):
            v = _spin(load_all, pending)
        else:
            rid = self.region_ids.get(specs[0][0].region, -1)
            v = _spin_bounded(
                load_all,
                pending,
                self.diag,
                rid,
                specs[0][1],
                tag,
                self.steady,
                self.failure,
                self.wait_ticks,
            )
        outs, e = [], 0
        for _, _, n in specs:
            outs.append(
                [
                    (
                        (v[e + 2 * q + 1] == tag).select(v[e + 2 * q], fx.Int32(0))
                        if self.steady
                        else v[e + 2 * q]
                    )
                    for q in range(n)
                ]
            )
            e += 2 * n
        return outs


WAIT_TICKS = 10_000_000  # 100 ms at the validated 100 MHz steady clock.


def scalar_address(address):
    """Restore a wave-uniform base address at its consumer stage boundary.

    Earlier stages keep addresses in VGPRs to bound their scalar live set.
    Passing those opaque values to a buffer resource makes LLVM emit a
    waterfall and load-counter drain at every access. These base addresses
    are uniform by contract; materialize that scalar value when it is used.
    The side-effecting move keeps it out of the earlier stages' live set.
    """
    scalar = rocdl.readfirstlane(T.i64, fx.Int64(address).ir_value())
    return fx.Int64(
        llvm.inline_asm(
            T.i64,
            [scalar],
            "s_mov_b64 $0, $1",
            "=s,s",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def uniform_stage_context(c, regions=(), args=(), roots=()):
    """Restore only explicitly named uniform bases for the consuming stage.

    The context and its argument mapping exist only during kernel compilation.
    Keep unrelated bases in their VGPR form to bound descriptor live ranges.
    Callers must name immutable, wave-uniform allocation or parameter bases;
    lane-dependent row addresses, indices and payloads are never eligible.
    Addr region/space metadata and all per-lane offsets remain unchanged.
    """
    out = dict(c)
    for name in regions:
        if name in c:
            old = c[name]
            out[name] = Addr(old.region, old.space, scalar_address(old.value))
    if args:
        out["args"] = dict(c["args"])
        for name in args:
            out["args"][name] = scalar_address(c["args"][name])
    for name in roots:
        out[name] = scalar_address(c[name])
    return out


def reference_mailbox(layer, diag, regions, failure, *, diagnostic=False):
    """Mother normal polling, with gfx1250's required system-scope encoding.

    Clock, wave-wide pending checks and failure propagation belong to the
    explicitly requested diagnostic variant. Normal polling uses the original
    per-lane pending condition and s_sleep(1), without result masking.
    """
    if diagnostic:
        return Mailbox(
            layer, diag, regions, steady=True, failure=failure, wait_ticks=WAIT_TICKS
        )
    return Mailbox(layer)


@traced
def publish_plain(put, addr, index, value, who):
    """Drain each writer wave's stores, join the CTA, then publish a tagged pair.

    gfx1250 separates the load and store counters. The mother's vmcnt drain
    therefore needs a store-counter instruction here; a load-counter wait
    alone cannot fulfill its plain-payload-before-readiness contract.
    """
    llvm.inline_asm(
        None,
        [],
        "s_wait_storecnt 0",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    gpu.barrier()
    if who:
        put(addr, index, value)

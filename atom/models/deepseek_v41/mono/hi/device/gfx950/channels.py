# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Device-side channel access (formats and protocols: ``blocks.base``).

Thin, format-aware wrappers over the mailbox so blocks agree on layouts. ``a``
is the channel's ``Addr`` (``k.mb[name]``; MXFP8 channels have ``name.q`` and
``name.s``); indices are element indices within the row-major [rows, width]
tensor unless noted. Producer helpers take ``dense``: True for a ``FLAG``
channel (the words without tags; the task then calls ``put_flag``).

Bulk + flag (``Proto.FLAG``) in device terms:

- producer: dense words stored at device scope (``store_words``: sc1, past the
  XCD L2s, as the tagged puts), then ``put_flag``: every
  storing wave drains its stores (``s_waitcnt vmcnt(0)``: the stores are acknowledged
  at the device coherence point), the CTA meets (multi-wave producers), and one
  lane puts the tagged flag: a flag never overtakes its words;
- consumer: ``flag_try`` / ``flag_land`` (every flag current, then a CTA barrier, so no thread
  issues a word load before *all* flags were seen), then ``load_words`` at device
  scope. A word load issued before its flag is seen current may return stale bytes
  (no tag to catch it): never issue one speculatively. Loads return in issue order
  (vmcnt), so a word load issued after a weight prefetch waits for the prefetch.
"""

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl

from ...plan.trace import note
from . import mailbox as mb
from .ops import CM_DEV, spin_pause, traced

# ---- dense words (FLAG channels)


def store_words(a, w, words):
    """Dense words ``w .. w + len(words)`` (1, 2 or 4; ``w`` aligned to the count) at
    device scope; recorded as a put of ``a``'s region."""
    note(a.region, a.space, "put")
    data = (
        fx.Vector.from_elements([fx.Int32(x) for x in words], fx.Int32)
        if len(words) > 1
        else fx.Int32(words[0])
    )
    mb.store(a, data, w, CM_DEV)


def load_words(a, w, n, cm=CM_DEV):
    """Dense words ``w .. w + n`` (n in 1, 2, 4) -> list of Int32, issued now (call only
    after ``flag_land``); recorded as a poll of ``a``'s region."""
    note(a.region, a.space, "poll")
    v = mb.load(a, w, n, cm)
    if n == 1:
        return [fx.Int32(v)]
    v = fx.Vector(v)
    return [v[e] for e in range(n)]


def put_flag(k, region, slot, waves=8):
    """End of a producer task (every thread of the CTA calls it): drain this task's
    stores, meet, then thread 0 puts flag ``slot`` of ``region`` := (1, tag)."""
    _put_flag(k.mb[region], slot, k.tag, k.tid, waves > 1)


@traced
def _put_flag(a, slot, tag, tid, meet):
    rocdl.s_waitcnt(vmcnt=0)
    if const_expr(meet):
        gpu.barrier()
    if tid == 0:
        mb.put(a, slot, fx.Int32(1), tag)


def flag_count(k, region) -> int:
    """Flag slots of a FLAG channel's flag region (from the program: the producer declared it)."""
    return {r.name: r.pairs for r in k.program.regions}[region]


def flag_try(k, region, threads=None):
    """One poll attempt of this thread's share of every flag of ``region`` (issued now, not
    waited for): (specs, raw words). Thread ``i`` polls flags ``i, i + T, ...`` (clamped).
    """
    n = flag_count(k, region)
    threads = threads or k.threads
    a = k.mb[region]
    specs = [
        (a, mb.clamp_live(a, fx.min(fx.Int32(c * threads) + k.tid, fx.Int32(n - 1))), 1)
        for c in range(-(-n // threads))
    ]  # (dyn_rows: a dead row's task puts no flag, mb.clamp_live)
    raw = []
    for sa, i, cnt in specs:
        note(sa.region, sa.space, "poll")
        v = fx.Vector(mb.load_pairs(sa, i, 2))
        raw += [v[0], v[1]]
    return specs, raw


def flag_land(k, specs, raw):
    """Spin until this thread's flags carry the tag (from ``flag_try``'s attempt), then a
    CTA barrier: after it every flag of the channel was seen current by some thread."""

    def load_all():
        words = []
        for a, i, _n in specs:
            v = fx.Vector(mb.load_pairs(a, i, 2))  # flag_try's words
            words += [v[0], v[1]]
        return fx.Vector.from_elements(words, fx.Int32)

    def pending(v):
        bad = v[1] != k.tag
        for e in range_constexpr(3, 2 * len(specs), 2):
            bad = bad | (v[e] != k.tag)
        return bad

    v0 = fx.Vector.from_elements(raw, fx.Int32)
    _spin_from(v0, load_all, pending)
    gpu.barrier()


@traced
def _spin_from(v, load_all, pending):
    while pending(v):
        spin_pause()
        v = load_all()
    return v


# ---- BF16: two elements per pair


def put_bf16(a, e, vals, tag, dense=False):
    """Elements ``e .. e + len(vals)`` (2 or 4; ``e`` a multiple of len(vals))."""
    mb.put_bf(a, e, vals, tag)


# ---- F32: one element per pair


# ---- I32: one int32 per pair


# ---- MXFP8: four E4M3 per pair in ``.q``, one E8M0 code per pair in ``.s``


def put_mxfp8_word(aq, w, word, tag, dense=False):
    """FP8 word ``w`` (elements 4w .. 4w + 3, row-major) := ``word`` (Int32)."""
    mb.put_words(aq, w, [word], tag)


def put_mxfp8_scale(as_, g, code, tag, dense=False):
    """Scale code of 32-group ``g`` (row-major over rows) := ``code`` (Int32, low byte)."""
    mb.put_words(as_, g, [fx.Int32(code)], tag)


def put_words(a, i, words, tag, dense=False):
    """Raw words (1 or 2) at word / pair index ``i`` (``mailbox.put_words`` or dense)."""
    if dense:
        store_words(a, i, list(words))
    else:
        mb.put_words(a, i, list(words), tag)

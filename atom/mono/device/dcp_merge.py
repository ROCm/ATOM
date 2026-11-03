# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A DCP decode's cross-rank attention merge inside a mono kernel: each rank
holds one KV shard's partial output and its softmax statistics, and the owner
of a head weights the shards' partials into the full attention output.

Two forms, each the arithmetic of the path its kernel has to agree with:

* ``lse_*`` -- one LSE a source, its sums one source at a time in rank order
  with explicit FMAs: ``dcp_ops._dcp_a2a_unpack_combine_rank_order_kernel`` op
  for op, so the Kimi-K3 K2 merge is bit-identical to the steps it falls back
  on.
* ``maxsum_*`` -- a source's (max score, sum of exp) pair, normalized weights
  from a lane a source: the GLM-5 MonoKernel's latent merge, whose attention
  is its own and has no production path to match.

Only the arithmetic is here: who pushes, the regions, the tags and where the
merged row goes are the kernel's.
"""

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from atom.mono.device.ops import exp, hw_exp2, rcp

NEG_INF = float("-inf")
LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453


# What Triton 3.8 lowers ``tl.exp2`` / ``tl.log2`` to on gfx950 (checked on its
# LLVM IR): ``llvm.amdgcn.exp2.f32`` (no denormal fix-up, ``ops.hw_exp2``) and
# ``llvm.log2.f32``. Another Triton could lower them otherwise; Kimi-K3's check
# mode (``ATOM_MONO_CHECK``) compares the two paths bit for bit.
exp2 = hw_exp2


def log2(x):
    return fx.Float32(
        llvm.call_intrinsic(T.f32, "llvm.log2.f32", [fx.Float32(x).ir_value()], [], [])
    )


# ------------------------------------------------------------------- LSE form
def clean_lse(lse):
    """-inf for a non-finite LSE: a rank owning no KV for the row reports
    -inf (or NaN), and contributes nothing."""
    bad = (lse != lse) | (lse == fx.Float32(float("inf")))  # noqa: PLR0124
    return bad.select(fx.Float32(NEG_INF), lse)


def lse_total(lses):
    """The merged LSE of the sources' (clean) LSEs, in rank order. Every source
    empty: the max is -inf, taken as 0 so the subtraction stays finite."""
    m = fx.Float32(NEG_INF)
    for lse in lses:
        m = fx.max(m, lse)
    m = (m == fx.Float32(NEG_INF)).select(fx.Float32(0.0), m)
    total = fx.Float32(0.0)
    for lse in lses:
        total = total + exp2((lse - m) * fx.Float32(LOG2E))
    return fx.Float32(fmath.fma(log2(total), fx.Float32(LN2), m))


def lse_factor(lse, total):
    """A source's weight: exp(lse - total), 0 where that is NaN (every source
    empty, so ``total`` is -inf)."""
    f = exp2((lse - total) * fx.Float32(LOG2E))
    return (f != f).select(fx.Float32(0.0), f)  # noqa: PLR0124


def lse_accumulate(acc, vals, f):
    """``acc`` + one source's ``vals`` weighted by ``f``, elementwise. A source
    with weight 0 adds an exact 0: its partial (NaN from a rank with no KV)
    is dropped before the multiply, as NaN * 0 is NaN."""
    out = []
    for a, v in zip(acc, vals, strict=True):
        v = (f == fx.Float32(0.0)).select(fx.Float32(0.0), v)
        out.append(fx.Float32(fmath.fma(v, f, a)))
    return out


# -------------------------------------------------------------- max/sum form
def maxsum_weight(m, local_sum, wave_max, wave_sum):
    """A lane's source weight from the sources' (max, sum of exp) pairs, a lane
    a source (a lane with no source passes a max below every score, sum 0): its share of the
    normalizer, 0 when every source is empty. ``wave_max`` / ``wave_sum`` are
    the kernel's own wave reductions."""
    m_all = wave_max(m)
    z = local_sum * exp(m - m_all)
    den = wave_sum(z)
    return z * (den > 0.0).select(rcp(den), fx.Float32(0.0))


def maxsum_accumulate(acc, vals, weight):
    """``acc`` + one source's ``vals`` times ``weight``, elementwise. A source
    with weight 0 adds an exact 0: an empty rank's partial is read from whatever
    its padding slot holds, and NaN * 0 is NaN."""
    out = []
    for a, v in zip(acc, vals, strict=True):
        v = (weight == fx.Float32(0.0)).select(fx.Float32(0.0), v)
        out.append(a + v * weight)
    return out

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Lossless row transport with 32-byte wave headers, shared through LDS.

The format of ar_add_rmsnorm_pull_lossless: 6144 sign/mantissa bytes,
3072 delta bytes and twelve 32-byte headers. A separate caller-owned raw arena
supports every row count, without aliasing staged input or reduced rows.
The caller drains stores before its existing publication and keeps that raw
arena alive until all peers acknowledge reads through their next arrivals.
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from amd_kernels.common import buffer_ops
from .common import SYS
from .lossless_bf16 import enc, _dec_fast, _dec_fix, RAWF, SPECF

H = 6144
ROWB = 12288


@flyc.jit
def store32(value, base, raw, tid, mask):
    lane, wave = tid % 64, tid // 64
    a0, a1, nib, gmax, cwv, ev, idx, has, rawi = enc(value, lane)
    rs = buffer_ops.create_buffer_resource_from_addr(base, num_records_bytes=ROWB)
    rr = buffer_ops.create_buffer_resource_from_addr(raw, num_records_bytes=ROWB)
    buffer_ops.buffer_store(value, rr, tid * 4, mask=mask & (rawi != 0), cache_modifier=SYS)
    buffer_ops.buffer_store(Vec.from_elements([a0, a1], fx.Int32), rs, tid * 2, mask=mask, cache_modifier=SYS)
    buffer_ops.buffer_store(nib, rs, H // 4 + tid, mask=mask, cache_modifier=SYS)
    buffer_ops.buffer_store(
        gmax.to(fx.Int8), rs, 9216 + 32 * wave + lane // 4,
        mask=mask & (lane % 4 == 0), cache_modifier=SYS, offset_is_bytes=True,
    )
    buffer_ops.buffer_store(cwv, rs, 2304 + 8 * wave + 4, mask=mask & (lane == 0), cache_modifier=SYS)
    buffer_ops.buffer_store(
        ev.to(fx.Int16), rs, 9216 + 32 * wave + 20 + 2 * (idx - 1),
        mask=mask & has & (rawi == 0), cache_modifier=SYS, offset_is_bytes=True,
    )


@flyc.jit
def decode32(base, raw, tid, mask, hw):
    lane = tid % 64
    rs = buffer_ops.create_buffer_resource_from_addr(base, num_records_bytes=ROWB)
    pa = Vec(buffer_ops.buffer_load(rs, tid * 2, vec_width=2, dtype=fx.Int32, mask=mask, cache_modifier=SYS))
    pb = fx.Int32(buffer_ops.buffer_load(rs, H // 4 + tid, vec_width=1, dtype=fx.Int32, mask=mask, cache_modifier=SYS))
    pw = [fx.Int32(pa[0]), fx.Int32(pa[1])]
    cw = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(hw[4])))
    d0, d1, d2, d3 = _dec_fast(pw, pb, hw, lane)
    if (cw & SPECF) != 0:
        d0, d1, d2, d3 = _dec_fix([d0, d1, d2, d3], pw, pb, hw)
    if (cw & RAWF) != 0:
        rr = buffer_ops.create_buffer_resource_from_addr(raw, num_records_bytes=ROWB)
        r = Vec(buffer_ops.buffer_load(rr, tid * 4, vec_width=4, dtype=fx.Int32, mask=mask, cache_modifier=SYS))
        d0, d1, d2, d3 = [fx.Int32(r[t]) for t in range_constexpr(4)]
    return Vec.from_elements([d0, d1, d2, d3], fx.Int32)

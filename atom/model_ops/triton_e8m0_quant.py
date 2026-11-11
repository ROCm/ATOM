# SPDX-License-Identifier: MIT
"""E8M0 group scales bit-identical to ``per_1x128_e8m0_quant`` (aiter
``per_group_quant_hip``), for producers that fuse the consumer's quant.

FP8_MAX is passed in: ``aiter.dtypes.fp8`` needs a GPU to resolve.
"""

import triton
import triton.language as tl

# aiter's amax floor (kAmaxFloor) and its gfx1250 cap for inf groups.
E8M0_AMAX_FLOOR = tl.constexpr(1e-10)
E8M0_AMAX_CAP = tl.constexpr(448.0 * 2.0**119)


@triton.jit
def fp8_e8m0_group_scale(amax, FP8_MAX: tl.constexpr):
    """(E8M0 byte, quant multiplier 2^(127 - byte)) for a group amax.

    RoundUp of ``amax * (1 / FP8_MAX)`` as ``aiter::fp_f32_to_e8m0_scale``;
    the power-of-two multiply is exact, so the FP8 cast rounds as aiter's does.
    """
    amax = tl.minimum(tl.maximum(amax, E8M0_AMAX_FLOOR), E8M0_AMAX_CAP)
    bits = (amax * (1.0 / FP8_MAX)).to(tl.uint32, bitcast=True)
    code = (bits >> 23) & 0xFF
    code += ((code < 0xFF) & ((bits & 0x7FFFFF) != 0)).to(tl.uint32)
    multiplier = ((254 - code) << 23).to(tl.float32, bitcast=True)
    return code.to(tl.uint8), multiplier

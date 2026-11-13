# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM's MXFP8 KV row at the hi boundary; the attention operands stay bf16.

One row is 512 E4M3 bytes followed by 16 E8M0 scale bytes, in groups of 32.
Use ATOM's amax floor and ceil-scale rule, including the all-zero group.
"""

import flydsl.expr as fx

from atom.mono.device.mx import FP8_MAX, code_ceil, pow2
from atom.mono.device.ops import mxfp8_to_bf16x8

from . import mailbox as mb
from . import paged
from .ops import global_load, traced

DIM = 512
GROUP = 32
ROW_BYTES = DIM + DIM // GROUP
# ATOM's blockscale quantizer floor; checked against that module by the
# boundary tests. Keep this device package independent of engine modules.
AMAX_FLOOR = 1e-10


def code(amax):
    """ATOM ``fp8_groups``' E8M0 scale code for a finite group amax."""
    return code_ceil(fx.max(amax, fx.Float32(AMAX_FLOOR)) * (1.0 / FP8_MAX))


def load8(row_addr, element):
    """Eight aligned elements of an MXFP8 row as four packed bf16 dwords.

    Both addresses are 64-bit: compressed owners can sit beyond 4 GiB.
    A dword load for the scale stays inside the row's 16-byte scale tail.
    """
    packed = fx.Vector(global_load(row_addr + fx.Int64(element), 2))
    group = element // GROUP
    scales = fx.Int32(global_load(row_addr + DIM + fx.Int64((group // 4) * 4)))
    scale = pow2((scales >> ((group % 4) * 8)) & 255)
    return mxfp8_to_bf16x8(packed[0], packed[1], scale).bitcast(fx.Int32)


@traced
def store_encoded(k, channel: str, row, addr):
    """Store KVPrep's original quantized row, without quantizing its QAT output again.

    Re-quantization can halve a scale when E4M3 rounding brings the group's amax
    down to a power-of-two boundary, so the original bytes travel beside bf16.
    """
    if k.tid < DIM // 4:
        word = mb.poll([(k.mb[f"{channel}.q"], row * (DIM // 4) + k.tid, 1)], k.tag)[0][
            0
        ]
        paged.global_store(addr + fx.Int64(k.tid * 4), word)
    if k.tid < DIM // GROUP:
        scale = mb.poll(
            [(k.mb[f"{channel}.s"], row * (DIM // GROUP) + k.tid, 1)], k.tag
        )[0][0]
        paged.global_store(addr + DIM + fx.Int64(k.tid), scale, 1)

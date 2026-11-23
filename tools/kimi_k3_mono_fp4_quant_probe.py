# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The replicated K2 front's latent FP4 quant two ways, exhaustively: every
BF16 value times every E8M0 code 0..254, eight values a thread. The software
RNE E2M1 conversion (``moe_gfx1250._emit_f32_to_e2m1`` of the value times
``2^(127 - code)``, the front's today) against one gfx1250
``v_cvt_scalef32_pk8_fp4_bf16`` of the eight by the code's scale
``2^(code - 127)``, the software one kept where the code is 254 (only an
infinite amax rounds there; FIXED=0: the instruction's there too). Prints the
mismatched nibbles, those with a finite value and a code a group holding it
can get (its own amax's or larger, below 254) apart.

    python tools/kimi_k3_mono_fp4_quant_probe.py
"""

import json
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir.dialects import rocdl as rocdl_dialect
from flydsl.expr.typing import Int64, T, as_ir_value

from atom.models.kimi_k3.mono.kernels.stages import moe_gfx1250
from atom.mono.device.ops import rsrc

FIXED = os.environ.get("FIXED", "1") == "1"
THREADS = 256
VALUES = 1 << 16
CODES = 255


def build():
    @flyc.kernel(name="fp4_quant_probe", known_block_size=[THREADS, 1, 1])
    def kernel(out: Int64):
        item = fx.Int32(fx.block_idx.x) * THREADS + fx.thread_idx.x
        code = item // (VALUES // 8)
        base = item % (VALUES // 8) * 8
        words = [(base + 2 * k) | ((base + 2 * k + 1) << 16) for k in range(4)]
        values = fx.Vector.from_elements(words, fx.Int32).bitcast(fx.BFloat16)
        reciprocal = ((fx.Int32(254) - code) << 23).bitcast(fx.Float32)
        soft = fx.Int32(0)
        for k in range(8):
            nibble = fx.Int32(
                moe_gfx1250._emit_f32_to_e2m1(fx.Float32(values[k]) * reciprocal)
            )
            soft = soft | (nibble << (4 * k))
        hard = fx.Int32(
            rocdl_dialect.cvt_scalef32_pk8_fp4_bf16(
                T.i32,
                as_ir_value(values),
                as_ir_value((code << 23).bitcast(fx.Float32)),
            )
        )
        if FIXED:
            # the front's rule: the software conversion where the code is 254
            hard = (code == 254).select(soft, hard)
        bo.buffer_store(
            fx.Vector.from_elements([soft, hard], fx.Int32), rsrc(out), item * 2
        )

    @flyc.jit
    def launch(out: Int64, stream: fx.Stream = fx.Stream(None)):  # noqa: B008
        kernel(out).launch(
            grid=(CODES * VALUES // 8 // THREADS,), block=(THREADS,), stream=stream
        )

    return launch


def own_codes():
    """Each BF16 value's code as a group's amax (``_e8m0_roundup`` of |v|);
    a value sits with any code from its own up (a larger amax), and with 0
    or 254 in a group holding a NaN or an infinity."""
    bits = (torch.arange(VALUES, dtype=torch.int32) & 0x7FFF) << 16
    scaled = (bits.view(torch.float32) * torch.tensor(1.0 / 6.0)).view(torch.int32)
    code = ((scaled.long() + 0x7FFFFF) & 0xFFFFFFFF) >> 23 & 0xFF
    return code.clamp(max=254)


out = torch.zeros(CODES * VALUES // 8 * 2, dtype=torch.int32, device="cuda")
build()(out.data_ptr(), stream=torch.cuda.current_stream())
torch.cuda.synchronize()
got = out.view(CODES, VALUES // 8, 2).cpu().long() & 0xFFFFFFFF
shifts = torch.arange(8) * 4
soft = (got[..., 0:1] >> shifts) & 15
hard = (got[..., 1:2] >> shifts) & 15
soft, hard = soft.reshape(CODES, VALUES), hard.reshape(CODES, VALUES)
bits = torch.arange(VALUES)
own = own_codes()
codes = torch.arange(CODES).view(CODES, 1)
bad = soft != hard
finite = ((bits & 0x7F80) != 0x7F80).view(1, -1)
ordinary = bad & finite & (codes >= own.view(1, -1)) & (codes < 254)
print(
    json.dumps(
        {
            "nibbles": CODES * VALUES,
            "mismatched": int(bad.sum()),
            "ordinary_mismatched": int(ordinary.sum()),
            "mismatched_codes": sorted(set(bad.nonzero()[:, 0].tolist())),
            "examples": [
                {
                    "code": c,
                    "bf16": hex(v),
                    "soft": int(soft[c, v]),
                    "hard": int(hard[c, v]),
                }
                for c, v in (ordinary if ordinary.any() else bad)
                .nonzero()
                .tolist()[:16]
            ],
        }
    ),
    flush=True,
)

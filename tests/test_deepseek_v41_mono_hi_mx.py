# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The mono hi MX codecs (``mono/hi/weights/mx.py``) read every E8M0 scale code exactly.

``code_values`` turns a scale code into its fp32 power of two. If that value is off by one
ulp, a weight made of MXFP4 values does not survive its own round trip and
``pack_mxfp4_exact`` rejects it. The scales below cover the ranges where the old fp32
``torch.ldexp`` build of the value was one ulp low on gfx950 (torch 2.10 / ROCm 7.2.4),
plus a few ordinary ones. The tests run on the CPU and, when there is one, on the GPU.
"""

import math

import pytest
import torch

from atom.models.deepseek_v41.mono.hi.weights import mx

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
# scales 2^e: the ranges holding the exponents the old fp32 torch.ldexp was not exact
# at on the GPU, the single one above them, and a few ordinary ones
EXPONENTS = list(range(-46, -33)) + list(range(16, 33)) + [49] + [-20, -8, -1, 0, 1, 7]


@pytest.mark.parametrize("device", DEVICES)
def test_every_code_reads_as_its_exact_power_of_two(device):
    codes = torch.arange(255, dtype=torch.uint8, device=device)  # 0..254; 255 is NaN
    got = mx.code_values(codes)
    assert got.dtype == torch.float32
    want = torch.tensor(
        [math.ldexp(1.0, c - 127) for c in range(255)], dtype=torch.float32
    )
    assert torch.equal(got.cpu(), want)


def _mxfp4_weight(device, rows=4):
    """bf16 [rows * len(EXPONENTS), 64] of MXFP4 values: row block i holds groups at scale
    2^EXPONENTS[i], each reaching +-6 so that scale is the one the quantizer picks."""
    g = torch.Generator().manual_seed(0)
    values = torch.tensor(mx._FP4_VALUES + tuple(-v for v in mx._FP4_VALUES))
    blocks = []
    for e in EXPONENTS:
        v = values[torch.randint(0, len(values), (rows, 64), generator=g)]
        v[:, 0], v[:, mx.GROUP] = 6.0, -6.0  # every group's amax is 6
        blocks.append(v * 2.0**e)
    return torch.cat(blocks).to(torch.bfloat16).to(device)


@pytest.mark.parametrize("device", DEVICES)
def test_mxfp4_round_trip_is_exact_at_every_scale(device):
    w = _mxfp4_weight(device)
    packed, codes = mx.quantize_mxfp4(w.float())
    want = torch.tensor([e + 127 for e in EXPONENTS]).repeat_interleave(4)
    assert torch.equal(codes.cpu().long(), want[:, None].expand(-1, 2))
    assert torch.equal(mx.dequantize_mxfp4(packed, codes), w.float())


@pytest.mark.parametrize("device", DEVICES)
def test_mxfp8_round_trip_is_exact_at_every_scale(device):
    g = torch.Generator().manual_seed(1)
    blocks = []
    for e in EXPONENTS:
        bits = torch.randint(0, 256, (4, 64), dtype=torch.uint8, generator=g)
        bits[(bits & 0x7F) == 0x7F] = 0  # no NaN
        v = bits.view(torch.float8_e4m3fn).float()
        v[:, 0], v[:, mx.GROUP] = 448.0, -448.0  # every group's amax is 448
        blocks.append(v * 2.0**e)
    w = torch.cat(blocks).to(device)
    values, codes = mx.quantize_mxfp8(w)
    want = torch.tensor([e + 127 for e in EXPONENTS]).repeat_interleave(4)
    assert torch.equal(codes.cpu().long(), want[:, None].expand(-1, 2))
    assert torch.equal(mx.dequantize_mxfp8(values, codes), w)


@pytest.mark.parametrize("device", DEVICES)
def test_pack_mxfp4_exact_accepts_mxfp4_weights_at_every_scale(device):
    pytest.importorskip("flydsl")
    from atom.models.deepseek_v41.mono.hi.blocks.gfx950.hc import pack_mxfp4_exact

    w = _mxfp4_weight(device)
    packed, codes = pack_mxfp4_exact(w)
    assert torch.equal(mx.dequantize_mxfp4(packed, codes), w.float())


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
def test_code_values_takes_uint8_codes_only(dtype):
    with pytest.raises(TypeError, match="uint8"):
        mx.code_values(torch.full((4,), 127, dtype=dtype))

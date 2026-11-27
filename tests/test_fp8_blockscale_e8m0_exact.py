# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``quantize_weight_to_fp8_128x128_blockscale`` with E8M0 scales gives every block the
exact power of two it rounds up to.

The scale used to be built with an fp32 ``torch.ldexp``, which is one ulp low at some
exponents on gfx950 (torch 2.10 / ROCm 7.2.4). The exponents below cover those ranges
plus a few ordinary ones. The test runs on the CPU and, when there is one, on the GPU.
"""

import pytest
import torch
from aiter_stub import atom_without_aiter

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
EXPONENTS = list(range(-46, -33)) + list(range(16, 33)) + [49] + [-20, -8, -1, 0, 1, 7]
FP8 = torch.float8_e4m3fn
E8M0 = torch.float8_e8m0fnu


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("top", [448.0, 256.0], ids=["exact", "rounded_up"])
def test_e8m0_block_scales_are_exact_powers_of_two(monkeypatch, device, top):
    assert torch.finfo(FP8).max == 448.0
    # one 128x128 block per exponent with amax top * 2^e: amax / fmax is 2^e itself, or
    # 4/7 * 2^e, which rounds up to 2^e; either way the block amax quantizes to top
    amax = [top * 2.0**e for e in EXPONENTS]
    g = torch.Generator().manual_seed(0)
    w = torch.rand(128, 128 * len(EXPONENTS), generator=g) * 2 - 1
    for i, a in enumerate(amax):
        block = w[:, 128 * i : 128 * (i + 1)]
        block.mul_(a)
        block[0, 0] = a
        # halfway between the E4M3 values 1 and 1.125: an exact scale rounds it to
        # even (1); a scale one ulp low pushes it past the midpoint (1.125)
        block[1, 0] = 1.0625 * 2.0 ** EXPONENTS[i]
    with atom_without_aiter() as load:
        utils = load("atom.quantization.quark.utils")
        from aiter import dtypes

        # the stand-in dtypes map E8M0 to uint8; the real ones already use E8M0
        monkeypatch.setattr(dtypes, "fp8_e8m0", E8M0)
        q, scale = utils.quantize_weight_to_fp8_128x128_blockscale(
            w.to(device), FP8, scale_dtype=E8M0
        )

    assert scale.dtype == E8M0 and tuple(scale.shape) == (1, len(EXPONENTS))
    want = torch.tensor([e + 127 for e in EXPONENTS], dtype=torch.uint8)
    assert torch.equal(scale.view(torch.uint8).cpu()[0], want)
    # each block is divided by its exact scale, so its amax lands exactly on top and
    # the midpoint rounds to even
    q = q.float().cpu()
    assert torch.equal(q[0, ::128], torch.full_like(q[0, ::128], top))
    assert torch.equal(q[1, ::128], torch.ones_like(q[1, ::128]))

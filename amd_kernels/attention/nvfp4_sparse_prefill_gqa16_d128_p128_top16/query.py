# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Input preparation of the fp8 path in torch: the E4M3 query operand and the NVFP4 -> E4M3 lookup table."""

import torch

# within each eight bytes the FP8 kernel reads Q elements in this order
FP8_Q_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)

_NVFP4_E4M3_LUT: dict[torch.device, torch.Tensor] = {}


def pack_query(q: torch.Tensor) -> torch.Tensor:
    """bf16 ``q`` -> the fp8 kernels' uint8 Q: each element clamped to [-448, 448] and rounded once to E4M3 (RNE; NaN
    stays NaN with its sign), then each eight bytes reordered to ``FP8_Q_ORDER``."""
    codes = q.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)
    return codes.reshape(-1, 8)[:, list(FP8_Q_ORDER)].reshape(q.shape).contiguous()


def nvfp4_e4m3_lut(device: torch.device) -> torch.Tensor:
    """[256] int64: byte j of entry s is e4m3_rn_satfinite(m_j * e4m3(s)) for the E2M1 magnitudes m = 0, 0.5, 1, 1.5,
    2, 3, 4, 6: the fp8 K / V value of code magnitude j under block scale byte s."""
    lut = _NVFP4_E4M3_LUT.get(device)
    if lut is None:
        scales = torch.arange(256, dtype=torch.uint8, device=device)
        scales = scales.view(torch.float8_e4m3fn).float()
        # unwritten tokens can carry NaN scale bytes; P = 0 must not multiply a NaN V
        scales = torch.nan_to_num(scales, nan=0.0)
        code = torch.arange(8, device=device)
        exp, man = code // 2, (code % 2).float()
        mags = torch.where(
            exp == 0, man * 0.5, torch.exp2((exp - 1).float()) * (1.0 + man * 0.5)
        )
        # e2m1 * e4m3 is exact in fp32; saturate, then round once to E4M3
        values = (scales[:, None] * mags[None, :]).clamp(-448.0, 448.0)
        lut = values.to(torch.float8_e4m3fn).view(torch.int64).reshape(256)
        _NVFP4_E4M3_LUT[device] = lut
    return lut

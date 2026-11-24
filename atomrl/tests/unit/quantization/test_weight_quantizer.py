# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from __future__ import annotations

import torch
from atomrl.quantization.fp8_config import FP8Config
from atomrl.quantization.weight_quantizer import WeightQuantizer


def test_quantize_tensor_roundtrip() -> None:
    cfg = FP8Config(use_weight_pow2_scale=False)
    wq = WeightQuantizer(cfg)
    x = torch.randn(2, 256, dtype=torch.float32)
    q, scales = wq.quantize_tensor(x, block_size=128)
    dq = wq.dequantize_tensor(q, scales, block_size=128)
    err = (x - dq).abs().max().item()
    assert err < 0.25


def test_quantize_state_dict() -> None:
    cfg = FP8Config()
    wq = WeightQuantizer(cfg)
    sd = {"w": torch.randn(64, 128, dtype=torch.float32)}
    out = wq.quantize_state_dict(sd)
    assert "w" in out
    assert "w_fp8_scales" in out
    assert "w_fp8_meta" in out
    back = wq.dequantize_state_dict(out)
    assert back["w"].shape == (64, 128)
    assert back["w"].dtype == torch.bfloat16


def test_blockwise_shape() -> None:
    cfg = FP8Config()
    wq = WeightQuantizer(cfg)
    x = torch.randn(3, 4, 256, dtype=torch.float32)
    q, scales = wq.quantize_tensor(x, block_size=128)
    assert q.shape == x.shape
    assert scales.shape == (3, 4, 2)


def test_round_scale_pow2_rounds_up() -> None:
    """Pow-2 scale rounding must be ceil, not round-to-nearest.

    This direction is what makes AtomRL's scales agree with TransformerEngine's;
    round-to-nearest differs on ~half of all blocks by exactly one power of two.
    Previously this helper had no test at all and used torch.round.
    """
    from atomrl.quantization.weight_quantizer import _round_scale_pow2

    # 1.1 and 1.9 straddle the midpoint between 2^0 and 2^1: nearest would send
    # 1.1 -> 1.0, ceil sends both -> 2.0.
    s = torch.tensor([1.1, 1.9, 0.6, 3.0], dtype=torch.float32)
    out = _round_scale_pow2(s)

    assert torch.equal(out, torch.tensor([2.0, 2.0, 1.0, 4.0]))
    assert (out >= s).all(), "ceil must never round a scale down"
    assert torch.equal(out, torch.exp2(torch.ceil(torch.log2(s))))
    # exact powers of two are fixed points
    p = torch.tensor([0.25, 1.0, 8.0])
    assert torch.equal(_round_scale_pow2(p), p)


def test_weight_quantizer_pow2_flag_is_reachable() -> None:
    """use_weight_pow2_scale=True must actually produce power-of-two scales."""
    from atomrl.quantization.weight_quantizer import WeightQuantizer

    x = torch.randn(4, 256, dtype=torch.float32)
    for flag in (False, True):
        wq = WeightQuantizer(FP8Config(use_weight_pow2_scale=flag))
        _, scales = wq.quantize_tensor(x, block_size=128)
        mant = scales.float().view(torch.int32) & 0x007FFFFF
        all_pow2 = int(mant.abs().sum()) == 0
        assert all_pow2 is flag, f"use_weight_pow2_scale={flag} gave pow2={all_pow2}"

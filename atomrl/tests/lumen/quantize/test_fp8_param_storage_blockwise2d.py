# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""FP8 param-storage (blockwise2d): storing the frozen weight as FP8 and feeding
it via FP8StoredLinearFunction must match the per-step in-place-quant path."""

import pytest
import torch
from torch import nn

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _fp8():
    return (
        torch.float8_e4m3fnuz
        if "gfx94" in torch.cuda.get_device_properties(0).gcnArchName
        else torch.float8_e4m3fn
    )


@_CUDA
def test_fp8_param_storage_matches_inplace_quant():
    from atomrl.lumen.quantize import disable, enable, store_weights_fp8_blockwise2d
    from atomrl.lumen.quantize.config import QuantConfig
    from conftest import compute_snr  # type: ignore

    torch.manual_seed(0)
    w = (torch.randn(256, 256) * 0.05).cuda().to(torch.bfloat16)
    x0 = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)

    outs, gins = {}, {}
    for store in (False, True):
        m = nn.Linear(256, 256, bias=False).cuda().to(torch.bfloat16)
        m.weight.requires_grad_(False)  # frozen base (LoRA recipe)
        m.weight.data.copy_(w)
        enable(m, config=QuantConfig.from_str(scaling="blockwise2d", block_size=128))
        if store:
            n = store_weights_fp8_blockwise2d(m, fp8_dtype=_fp8(), block_size=128)
            assert n == 1 and m.weight.dtype == _fp8()
            assert getattr(m, "_fp8_weight_data", None) is not None
        x = x0.clone().requires_grad_(True)
        y = m(x)
        y.sum().backward()
        outs[store] = y.detach().float()
        gins[store] = x.grad.float()
        disable(m)

    # Same quant function (once vs per-step) → forward should be ~bit-identical;
    # DGrad goes through FP8StoredLinearFunction.backward but uses the same stored FP8.
    assert compute_snr(outs[False], outs[True]) > 40, "stored fwd != in-place quant fwd"
    assert compute_snr(gins[False], gins[True]) > 20, "stored DGrad != in-place DGrad"


@_CUDA
def test_fp8_param_storage_state_dict_dequantizes():
    """state_dict must emit BF16 (dequantized), not raw FP8."""
    from atomrl.lumen.quantize import disable, enable, store_weights_fp8_blockwise2d
    from atomrl.lumen.quantize.config import QuantConfig

    m = nn.Linear(256, 256, bias=False).cuda().to(torch.bfloat16)
    m.weight.requires_grad_(False)
    enable(m, config=QuantConfig.from_str(scaling="blockwise2d", block_size=128))
    store_weights_fp8_blockwise2d(m, fp8_dtype=_fp8(), block_size=128)
    sd = m.state_dict()
    wkey = next(k for k in sd if k.endswith("weight"))
    assert (
        sd[wkey].dtype == torch.bfloat16
    ), "checkpoint weight should be dequantized BF16"
    assert m.weight.dtype == _fp8(), "runtime weight should stay FP8 after state_dict"
    disable(m)

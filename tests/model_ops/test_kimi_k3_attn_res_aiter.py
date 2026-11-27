# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""AITER's gfx1250 attention-residual path against ATOM's inline Triton kernel."""

from unittest.mock import MagicMock, Mock

import pytest
import torch

pytest.importorskip("aiter")
pytest.importorskip("triton")

from aiter.ops.triton.fusions import attn_res as aiter_attn_res

from atom.model_ops.kimi_k3 import attention_residual as attn_res

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ROCm GPU required"
)


def _inputs(rows, blocks, add_count, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, device="cuda")

    prefix = randn(rows, 7168).bfloat16()
    block = randn(rows, blocks, 7168).bfloat16()
    score_weight = randn(7168) * 0.01
    norm_weight = randn(7168).bfloat16()
    adds = [randn(rows, 7168).bfloat16() if i < add_count else None for i in range(2)]
    return prefix, block, score_weight, norm_weight, adds


@pytest.fixture
def gfx1250():
    if attn_res.arch_info.get_arch() != "gfx1250":
        pytest.skip("AITER Gluon TDM attn-res needs gfx1250")


# B=1..3 cover the prefetch pipeline's tail (fewer rows than are kept in flight).
@pytest.mark.parametrize("blocks", [1, 2, 3, 8])
@pytest.mark.parametrize("add_count", [0, 1, 2])
@pytest.mark.parametrize("quant", [False, True])
@torch.inference_mode()
def test_aiter_matches_inline_triton(monkeypatch, gfx1250, blocks, add_count, quant):
    prefix, block, score_weight, norm_weight, adds = _inputs(
        301, blocks, add_count, 20261029 + 10 * blocks + add_count
    )
    kwargs = {
        "add_hidden": adds[0],
        "out_norm_weight": norm_weight,
        "out_eps": 1e-5,
        "add_hidden2": adds[1],
        "quant_dtype": torch.float8_e4m3fn if quant else None,
    }
    args = (prefix, block, score_weight, 1e-5)
    gate = Mock(wraps=attn_res.attn_res_gate)
    tdm_launch = Mock(wraps=aiter_attn_res._launch_gate_tdm)
    triton_launch = Mock(wraps=attn_res._attn_res_fused_kernel[(301,)])
    triton_kernel = MagicMock()
    triton_kernel.__getitem__.return_value = triton_launch
    monkeypatch.setattr(attn_res, "attn_res_gate", gate)
    monkeypatch.setattr(aiter_attn_res, "_launch_gate_tdm", tdm_launch)
    monkeypatch.setattr(attn_res, "_attn_res_fused_kernel", triton_kernel)

    monkeypatch.setattr(attn_res, "_USE_AITER_ATTN_RES", True)
    gluon = attn_res._apply_attn_res_impl(*args, **kwargs)
    gate.assert_called_once()
    assert gate.call_args.kwargs["backend"] == "arch"
    tdm_launch.assert_called_once()
    triton_launch.assert_not_called()

    gate.reset_mock()
    tdm_launch.reset_mock()
    monkeypatch.setattr(attn_res, "_USE_AITER_ATTN_RES", False)
    triton = attn_res._apply_attn_res_impl(*args, **kwargs)
    triton_kernel.__getitem__.assert_called_once_with((301,))
    triton_launch.assert_called_once()
    gate.assert_not_called()
    tdm_launch.assert_not_called()

    # The folded prefix is the same BF16 sum on both paths.
    assert torch.equal(gluon[-1], triton[-1])
    if quant:
        (y_g, s_g), (y_t, s_t) = gluon[:2], triton[:2]
        torch.testing.assert_close(s_g, s_t, rtol=1e-5, atol=0)
        y_g, y_t = y_g.float() * s_g, y_t.float() * s_t
    else:
        y_g, y_t = gluon[0].float(), triton[0].float()
    # Same softmax mix; only the FP32 accumulation order differs (the prefix
    # seeds the Gluon kernel's online softmax instead of closing it).
    row_error = (y_g - y_t).norm(dim=-1) / y_t.norm(dim=-1)
    assert row_error.max() < (1e-2 if quant else 5e-3)

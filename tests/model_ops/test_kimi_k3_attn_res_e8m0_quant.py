# SPDX-License-Identifier: MIT
"""Attention residual with E8M0 scales: byte-identical to its bf16 output +
the consumer's per_1x128_e8m0_quant: fused on the gfx1250 Gluon kernel, two kernels
with the Triton one."""

import pytest
import torch

pytest.importorskip("aiter")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("ROCm GPU required", allow_module_level=True)

from aiter import dtypes

from atom.model_ops.kimi_k3 import attention_residual as attn_res
from atom.model_ops.linear import per_1x128_e8m0_quant


def _run(gluon_tdm, monkeypatch, *args, **kwargs):
    monkeypatch.setitem(
        attn_res.envs.environment_variables,
        "ATOM_KIMI_K3_ATTN_RES_GLUON_TDM",
        lambda: gluon_tdm,
    )
    attn_res._attn_res_tdm_module.cache_clear()
    if gluon_tdm and attn_res._attn_res_tdm_module() is None:
        pytest.skip("Gluon TDM attn-res needs gfx1250")
    try:
        return attn_res._apply_attn_res_impl(*args, **kwargs)
    finally:
        attn_res._attn_res_tdm_module.cache_clear()


@pytest.mark.parametrize("gluon_tdm", [False, True])
@pytest.mark.parametrize("transpose_scale", [True, False])
@pytest.mark.parametrize("add_count", [0, 2])
@pytest.mark.parametrize(
    ("rows", "blocks"), [(1, 1), (15, 3), (135, 8), (1009, 2), (4100, 2)]
)
@torch.inference_mode()
def test_attn_res_e8m0_matches_bf16_then_consumer_quant(
    monkeypatch, rows, blocks, add_count, transpose_scale, gluon_tdm
):
    g = torch.Generator(device="cuda").manual_seed(20261105 + rows + blocks)

    def randn(*shape):
        return torch.randn(*shape, generator=g, device="cuda")

    H = 7168
    args = (
        randn(rows, H).bfloat16(),
        randn(rows, blocks, H).bfloat16(),
        randn(H) * 0.01,
        1e-5,
    )
    adds = [randn(rows, H).bfloat16() if i < add_count else None for i in range(2)]
    kwargs = {
        "add_hidden": adds[0],
        "out_norm_weight": randn(H).bfloat16(),
        "out_eps": 1e-5,
        "add_hidden2": adds[1],
    }
    y, prefix = _run(gluon_tdm, monkeypatch, *args, **kwargs)
    q_ref, s_ref = per_1x128_e8m0_quant(y, dtypes.fp8, transpose_scale)
    q, s, prefix_e8m0 = _run(
        gluon_tdm,
        monkeypatch,
        *args,
        **kwargs,
        quant_dtype=dtypes.fp8,
        e8m0_scale=True,
        transpose_scale=transpose_scale,
    )
    assert torch.equal(prefix_e8m0, prefix)
    assert s.dtype == dtypes.fp8_e8m0 and s.shape == s_ref.shape
    assert torch.equal(s.view(torch.uint8), s_ref.view(torch.uint8))
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))

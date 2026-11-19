"""gfx1250 Gluon TDM attn-res kernel against the Triton kernel it replaces."""

import pytest
import torch

pytest.importorskip("aiter")
pytest.importorskip("triton")

from atom.model_ops.kimi_k3 import attention_residual as attn_res

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ROCm GPU required"
)


def _run(enabled, monkeypatch, *args, **kwargs):
    monkeypatch.setitem(
        attn_res.envs.environment_variables,
        "ATOM_KIMI_K3_ATTN_RES_GLUON_TDM",
        lambda: enabled,
    )
    attn_res._attn_res_tdm_module.cache_clear()
    try:
        return attn_res._apply_attn_res_impl(*args, **kwargs)
    finally:
        attn_res._attn_res_tdm_module.cache_clear()


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
def gluon_tdm(monkeypatch):
    monkeypatch.setitem(
        attn_res.envs.environment_variables,
        "ATOM_KIMI_K3_ATTN_RES_GLUON_TDM",
        lambda: True,
    )
    attn_res._attn_res_tdm_module.cache_clear()
    if attn_res._attn_res_tdm_module() is None:
        pytest.skip("Gluon TDM attn-res needs gfx1250")


# B=1..3 cover the prefetch pipeline's tail (fewer rows than are kept in flight).
@pytest.mark.parametrize("blocks", [1, 2, 3, 8])
@pytest.mark.parametrize("add_count", [0, 1, 2])
@pytest.mark.parametrize("quant", [False, True])
@pytest.mark.usefixtures("gluon_tdm")
@torch.inference_mode()
def test_gluon_tdm_matches_triton(monkeypatch, blocks, add_count, quant):
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
    gluon = _run(True, monkeypatch, *args, **kwargs)
    triton = _run(False, monkeypatch, *args, **kwargs)
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

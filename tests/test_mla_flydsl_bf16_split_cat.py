# SPDX-License-Identifier: MIT
"""DeepSeek-R1 BF16 prefill K/V written by aiter's FlyDSL split-cat GEMM."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from aiter_stub import atom_under_stubs


def _case(mla, m=2, device="cpu", weight=None):
    if weight is None:
        weight = torch.empty((128 * 256, 512), dtype=torch.bfloat16, device=device)
    projector = SimpleNamespace(
        weight=weight,
        bias=None,
        quant_type=mla.QuantType.No,
        params_dtype=torch.bfloat16,
        is_output_padded=False,
        reduce_results=False,
    )
    impl = SimpleNamespace(
        use_flydsl_bf16_prefill_kv_split_cat=True,
        _flydsl_fp8_mha_ok=False,
        dtype=torch.bfloat16,
        dcp_world_size=1,
        kv_lora_rank=512,
        num_heads=128,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        qk_head_dim=192,
        v_head_dim=128,
        kv_b_proj=projector,
    )
    bf16 = {"dtype": torch.bfloat16, "device": device}
    x = torch.randn((m, 512), **bf16)
    return impl, torch.empty((m, 128, 192), **bf16), x, torch.randn((m, 1, 64), **bf16)


def _gfx1250_split_cat():
    if not torch.cuda.is_available():
        return False
    from aiter.jit.utils.chip_info import get_gfx_runtime

    from atom.model_ops import attention_mla

    return (
        get_gfx_runtime() == "gfx1250" and attention_mla._FLYDSL_KV_SPLIT_CAT_AVAILABLE
    )


@pytest.mark.parametrize("m", [1, 300])
def test_split_cat_matches_kv_b_proj_then_split_cat(m):
    if not _gfx1250_split_cat():
        pytest.skip("gfx1250 and aiter's FlyDSL split-cat (ROCm/aiter#8261) only")
    from atom.model_ops import attention_mla as mla

    weight = torch.randn((128 * 256, 512), dtype=torch.bfloat16, device="cuda") / 16
    impl, q, x, rope = _case(mla, m, "cuda", weight)
    k, v = mla.MLAAttention._try_flydsl_bf16_prefill_kv_split_cat(impl, q, x, rope)
    kv = (x.float() @ weight.float().T).view(m, 128, 256)
    torch.testing.assert_close(
        k[..., :128].float(), kv[..., :128], atol=2e-2, rtol=1e-2
    )
    torch.testing.assert_close(v.float(), kv[..., 128:], atol=2e-2, rtol=1e-2)
    assert torch.equal(k[..., 128:], rope.expand(-1, 128, -1))


@pytest.mark.parametrize("reason", ["flag_off", "other_arch", "fp8_proj", "dcp"])
def test_unsupported_paths_keep_kv_b_proj(monkeypatch, reason):
    with atom_under_stubs() as load:
        mla = load("atom.model_ops.attention_mla")
        launch = Mock(side_effect=lambda *a, **kw: (kw["k_out"], kw["v_out"]))
        monkeypatch.setattr(mla, "_FLYDSL_KV_SPLIT_CAT_AVAILABLE", True)
        monkeypatch.setattr(mla, "_get_gfx_runtime", lambda: "gfx1250")
        monkeypatch.setattr(mla, "get_pcp_world_size", lambda: 1)
        monkeypatch.setattr(mla, "_flydsl_hgemm", launch)
        split_cat = mla.MLAAttention._try_flydsl_bf16_prefill_kv_split_cat
        assert split_cat(*_case(mla)) is not None and launch.call_count == 1
        impl, q, x, rope = _case(mla)
        if reason == "flag_off":
            impl.use_flydsl_bf16_prefill_kv_split_cat = False
        elif reason == "other_arch":
            monkeypatch.setattr(mla, "_get_gfx_runtime", lambda: "gfx950")
        elif reason == "fp8_proj":
            impl.kv_b_proj.quant_type = mla.QuantType.per_1x128
        elif reason == "dcp":
            impl.dcp_world_size = 2
        assert split_cat(impl, q, x, rope) is None and launch.call_count == 1

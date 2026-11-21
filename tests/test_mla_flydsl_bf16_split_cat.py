# SPDX-License-Identifier: MIT
"""CPU/mock coverage for DeepSeek-R1 BF16 prefill split-cat dispatch."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("triton", reason="attention_mla defines @triton.jit kernels")
pytest.importorskip("aiter", reason="attention_mla imports the AITER runtime")

from atom.model_ops import attention_mla as mla


@pytest.fixture(scope="module")
def bf16_weight():
    # This is the real DeepSeek-R1 TP1 kv_b_proj shape. The launch is mocked.
    return torch.empty((128 * 256, 512), dtype=torch.bfloat16)


def _case(weight):
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
        kv_cache_dtype="auto",
        rope_is_zero_pad=False,
        is_sparse_mla=False,
        dcp_world_size=1,
        kv_lora_rank=512,
        num_heads=128,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        qk_head_dim=192,
        v_head_dim=128,
        kv_b_proj=projector,
    )
    q = torch.empty((2, 128, 192), dtype=torch.bfloat16)
    x = torch.empty((2, 512), dtype=torch.bfloat16)
    rope = torch.arange(128, dtype=torch.bfloat16).view(2, 1, 64)
    return impl, q, x, rope


def test_dispatch_preallocates_final_kv_and_wires_validated_config(
    monkeypatch, bf16_weight
):
    impl, q, x, rope = _case(bf16_weight)
    # The MLPerf recipe caches FP8 for decode while this plain prefill stays BF16.
    impl.kv_cache_dtype = "fp8"
    calls = []

    def launch(a, b, **kwargs):
        calls.append((a, b, kwargs))
        kwargs["k_out"][..., :128].fill_(3)
        kwargs["k_out"][..., 128:].copy_(rope.expand(-1, 128, -1))
        kwargs["v_out"].fill_(5)
        return kwargs["k_out"], kwargs["v_out"]

    monkeypatch.setattr(mla, "_FLYDSL_KV_SPLIT_CAT_AVAILABLE", True)
    monkeypatch.setattr(mla, "_get_gfx_runtime", lambda: "gfx1250")
    monkeypatch.setattr(mla, "get_pcp_world_size", lambda: 1)
    monkeypatch.setattr(mla, "_flydsl_hgemm", launch)

    result = mla.MLAAttention._try_flydsl_bf16_prefill_kv_split_cat(impl, q, x, rope)

    assert result is not None
    k_out, v_out = result
    assert k_out.shape == (2, 128, 192)
    assert v_out.shape == (2, 128, 128)
    assert k_out.dtype == v_out.dtype == torch.bfloat16
    assert torch.equal(k_out[..., 128:], rope.expand(-1, 128, -1))
    assert torch.count_nonzero(k_out[..., :128] != 3) == 0
    assert torch.count_nonzero(v_out != 5) == 0

    a, b, kwargs = calls.pop()
    assert a is x and b is bf16_weight
    assert kwargs["k_out"] is k_out and kwargs["v_out"] is v_out
    assert kwargs["k_rope"] is rope
    assert kwargs["split_sizes"] == (128, 128)
    assert kwargs["out_dtype"] == torch.bfloat16
    for name, value in mla._DEEPSEEK_R1_KV_SPLIT_CAT_CONFIG.items():
        assert kwargs[name] == value


def test_plain_prefill_uses_fused_buffers_without_linear_fallback(
    monkeypatch, bf16_weight
):
    impl, q, x, rope = _case(bf16_weight)
    seen = {}

    def launch(_a, _b, **kwargs):
        kwargs["k_out"].fill_(2)
        kwargs["v_out"].fill_(4)
        return kwargs["k_out"], kwargs["v_out"]

    def attend(q_arg, k_arg, v_arg, **_kwargs):
        seen.update(q=q_arg, k=k_arg, v=v_arg)
        return v_arg

    impl._check_flydsl_fp8_mha = lambda _q: None
    impl._drop_rope_pad = lambda *tensors: tensors
    impl._flash_attn_prefill = attend
    impl.o_proj = lambda output: output
    meta = SimpleNamespace(
        cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 2], dtype=torch.int32),
        max_seqlen_q=2,
        max_seqlen_k=2,
        min_seqlen_q=2,
        dropout_p=0.0,
    )

    monkeypatch.setattr(mla, "_FLYDSL_KV_SPLIT_CAT_AVAILABLE", True)
    monkeypatch.setattr(mla, "_get_gfx_runtime", lambda: "gfx1250")
    monkeypatch.setattr(mla, "get_pcp_world_size", lambda: 1)
    monkeypatch.setattr(mla, "_flydsl_hgemm", launch)
    monkeypatch.setattr(
        mla,
        "use_triton_gemm",
        Mock(side_effect=AssertionError("linear fallback was consulted")),
    )

    output = mla.MLAAttention._forward_prefill_mha(impl, q, x, rope, None, meta)

    assert seen["q"] is q
    assert seen["k"].shape == (2, 128, 192)
    assert seen["v"].shape == (2, 128, 128)
    assert output.shape == (2, 128 * 128)


@pytest.mark.parametrize(
    "reason",
    [
        "flag_off",
        "api_missing",
        "other_arch",
        "fp8_attention",
        "fp8_projection",
        "fp4_projection",
        "dcp",
        "pcp",
    ],
)
def test_unsupported_paths_fall_back_before_launch(monkeypatch, bf16_weight, reason):
    impl, q, x, rope = _case(bf16_weight)
    api_available, arch, pcp_size = True, "gfx1250", 1
    if reason == "flag_off":
        impl.use_flydsl_bf16_prefill_kv_split_cat = False
    elif reason == "api_missing":
        api_available = False
    elif reason == "other_arch":
        arch = "gfx950"
    elif reason == "fp8_attention":
        impl._flydsl_fp8_mha_ok = True
    elif reason == "fp8_projection":
        impl.kv_b_proj.quant_type = mla.QuantType.per_1x128
        impl.kv_b_proj.params_dtype = mla.dtypes.fp8
    elif reason == "fp4_projection":
        impl.kv_b_proj.quant_type = mla.QuantType.per_1x32
        impl.kv_b_proj.params_dtype = mla.dtypes.fp4x2
    elif reason == "dcp":
        impl.dcp_world_size = 2
    elif reason == "pcp":
        pcp_size = 2

    launch = Mock(side_effect=AssertionError("unsupported path launched"))
    monkeypatch.setattr(mla, "_FLYDSL_KV_SPLIT_CAT_AVAILABLE", api_available)
    monkeypatch.setattr(mla, "_get_gfx_runtime", lambda: arch)
    monkeypatch.setattr(mla, "get_pcp_world_size", lambda: pcp_size)
    monkeypatch.setattr(mla, "_flydsl_hgemm", launch)

    assert (
        mla.MLAAttention._try_flydsl_bf16_prefill_kv_split_cat(impl, q, x, rope) is None
    )
    launch.assert_not_called()

    # The original materialization remains the BF16 fallback.
    k_nope = torch.randn((2, 3, 128), dtype=torch.bfloat16)
    fallback, quant_rope = mla.MLAAttention._prepare_prefill_k(
        SimpleNamespace(_flydsl_fp8_mha_ok=False), k_nope, rope
    )
    assert quant_rope is None
    assert torch.equal(fallback, torch.cat((k_nope, rope.expand(-1, 3, -1)), dim=-1))

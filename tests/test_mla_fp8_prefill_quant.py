# SPDX-License-Identifier: MIT
"""Exercise FP8 prefill callers and dynamic K/V fallback on the real GPU path."""

from types import MethodType, SimpleNamespace

import pytest
import torch

if not torch.version.hip or not torch.cuda.is_available():
    pytest.skip("requires a ROCm GPU", allow_module_level=True)
if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"):
    pytest.skip("FlyDSL FP8 attention requires gfx950", allow_module_level=True)

from atom.model_ops import attention_mla as mla
from atom.model_ops.triton_fused_qkv_quant import fused_qkv_per_tensor_quant
from atom.model_ops.utils import quant_fp8_per_tensor


def _impl():
    impl = SimpleNamespace(
        use_flydsl_fp8_prefill_attn=True,
        rope_is_zero_pad=False,
        num_heads=12,
        qk_nope_head_dim=128,
        v_head_dim=128,
        scale=192**-0.5,
        layer_num=0,
        o_proj=lambda x: x,
    )
    for name in ("_prepare_prefill_k", "_drop_rope_pad", "_flash_attn_prefill"):
        setattr(impl, name, MethodType(getattr(mla.MLAAttention, name), impl))
    return impl


def _attend(qkv, cq, ck, *, causal, return_lse=False):
    q, k, v, qs, ks, vs = qkv[:6]
    return mla.flydsl_flash_attn_fp8_func(
        q,
        k,
        v,
        q_descale=qs,
        k_descale=ks,
        v_descale=vs,
        softmax_scale=192**-0.5,
        cu_seqlens_q=cq,
        cu_seqlens_kv=ck,
        max_seqlen_q=8,
        max_seqlen_kv=int(ck.diff().max()),
        cross_seqlen=True,
        causal=causal,
        return_lse=return_lse,
    )


def _no_quant(*args, **kwargs):
    raise AssertionError("prequantized tensors must not be quantized again")


@pytest.mark.parametrize("path", ["plain", "plain_triton", "cached_new"])
def test_prefill_callers_fuse_k_concat(monkeypatch, path):
    torch.manual_seed(41)
    q = torch.randn((8, 12, 192), device="cuda", dtype=torch.bfloat16)
    kv = torch.randn((8, 12, 256), device="cuda", dtype=q.dtype)
    rope = torch.randn((8, 64), device="cuda", dtype=q.dtype)
    k, v = kv.split([128, 128], -1)
    full_k = torch.cat((k, rope[:, None, :].expand(-1, 12, -1)), -1)
    cu = torch.tensor([0, 8], device="cuda", dtype=torch.int32)
    reference = _attend(fused_qkv_per_tensor_quant(q, full_k, v), cu, cu, causal=True)
    impl = _impl()
    impl.kv_b_proj = lambda x: kv
    impl.kv_b_proj.weight = torch.empty(1, device="cuda", dtype=q.dtype)
    impl.kv_b_proj.weight_scale = None
    metadata = SimpleNamespace(
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=8,
        max_seqlen_k=8,
        min_seqlen_q=8,
        dropout_p=0,
        total_kv=8,
    )
    monkeypatch.setattr(mla, "use_triton_gemm", lambda: path == "plain_triton")
    calls = []
    original = mla.fused_qkv_per_tensor_quant

    def quant(q, k, v, *, k_rope=None):
        assert k.shape[-1] == 128 and k_rope.shape == (8, 1, 64)
        calls.append(k_rope)
        return original(q, k, v, k_rope=k_rope)

    monkeypatch.setattr(mla, "fused_qkv_per_tensor_quant", quant)
    monkeypatch.setattr(mla, "quant_fp8_per_tensor", _no_quant)
    monkeypatch.setattr(mla, "fused_kv_per_tensor_quant", _no_quant)
    # Any remaining BF16 concat in these callers would undo the optimization.
    monkeypatch.setattr(torch, "cat", _no_quant)
    if path == "cached_new":
        # An all-empty cached iteration takes the new-token branch without a merge.
        chunks = SimpleNamespace(num_chunks=0, k_workspace=None, v_workspace=None)
        actual = mla.MLAAttention._forward_prefill_cached_chunked(
            impl,
            q,
            None,
            rope,
            None,
            metadata,
            chunks,
        )
    else:
        actual = mla.MLAAttention._forward_prefill_mha(
            impl,
            q,
            None,
            rope,
            None,
            metadata,
        )
    assert len(calls) == 1
    torch.testing.assert_close(actual, reference.flatten(1), rtol=0, atol=0)


@pytest.mark.parametrize("strided", [False, True])
def test_bf16_kv_fallback_matches_hip_attention(monkeypatch, strided):
    torch.manual_seed(42)
    q = torch.randn((8, 12, 192), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((16, 12, 256 if strided else 192), device="cuda", dtype=q.dtype)[
        ..., :192
    ]
    v = torch.randn((16, 12, 256 if strided else 128), device="cuda", dtype=q.dtype)[
        ..., :128
    ]
    q8, qs = quant_fp8_per_tensor(q)
    k8, ks = quant_fp8_per_tensor(k)
    v8, vs = quant_fp8_per_tensor(v)
    cq = torch.tensor([0, 3, 8], device="cuda", dtype=torch.int32)
    ck = torch.tensor([0, 16, 16], device="cuda", dtype=torch.int32)
    reference = _attend((q8, k8, v8, qs, ks, vs), cq, ck, causal=False, return_lse=True)
    calls = []
    original = mla.fused_kv_per_tensor_quant

    def quant(k, v):
        assert k.is_contiguous() and v.is_contiguous()
        calls.append((k, v))
        return original(k, v)

    monkeypatch.setattr(mla, "fused_kv_per_tensor_quant", quant)
    monkeypatch.setattr(mla, "quant_fp8_per_tensor", _no_quant)
    monkeypatch.setattr(mla, "fused_qkv_per_tensor_quant", _no_quant)
    impl = _impl()
    kwargs = {
        "cu_seqlens_q": cq,
        "cu_seqlens_k": ck,
        "max_seqlen_q": 8,
        "max_seqlen_k": 16,
        "min_seqlen_q": 3,
        "dropout_p": 0,
        "causal": False,
        "return_lse": True,
        "q_fp8": (q8, qs),
    }
    actual = impl._flash_attn_prefill(q, k, v, **kwargs)
    assert len(calls) == 1
    for a, b in zip(actual, reference):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert torch.count_nonzero(actual[0][3:]) == 0
    assert torch.isneginf(actual[1][:, 3:]).all()
    # Direct FP8 gather supplies the tuple and must still skip dynamic K/V quant.
    monkeypatch.setattr(mla, "fused_kv_per_tensor_quant", _no_quant)
    direct = impl._flash_attn_prefill(q, k, v, kv_fp8=(k8, v8, ks, vs), **kwargs)
    for a, b in zip(direct, reference):
        torch.testing.assert_close(a, b, rtol=0, atol=0)

# SPDX-License-Identifier: MIT
"""Sparse prefill writes directly to caller-owned token-row slices."""

import importlib

import pytest
import torch

pytest.importorskip("aiter")
pytest.importorskip("triton")

from atom.model_ops.v4_kernels import paged_2src


@pytest.mark.parametrize("backend", ["opus", "asm", "asm_fallback"])
@pytest.mark.parametrize("supplied_output", [False, True])
def test_fp8_prefill_passes_output_to_opaque_api(monkeypatch, backend, supplied_output):
    expected = torch.full((2, 128, 4), 3.0)
    parent = torch.full((4, 128, 4), -9.0)
    out = parent[1:3] if supplied_output else None
    calls = []

    # An opaque wrapper must receive out= without advertising its signature
    # or private implementation. This also tests the ordinary out=None path.
    def kernel(*args, **kwargs):
        assert kwargs["out"] is out
        calls.append(out)
        return expected.clone() if out is None else out.copy_(expected)

    def unavailable_asm(*args, **kwargs):
        assert kwargs["out"] is out
        raise RuntimeError("unsupported ASM shape")

    opus = importlib.import_module("aiter.ops.pa_sparse_prefill_opus")
    monkeypatch.setattr(opus, "pa_sparse_prefill_fp8_opus", kernel)
    monkeypatch.setattr(paged_2src, "_HAS_PREFILL_ASM", backend != "opus")
    monkeypatch.setattr(paged_2src, "get_gfx_runtime", lambda: "gfx1250")
    monkeypatch.setattr(paged_2src.envs, "ATOM_FORCE_V4_PREFILL_OPUS", False)
    monkeypatch.setattr(
        paged_2src,
        "mla_sparse_prefill_fp8_asm",
        unavailable_asm if backend == "asm_fallback" else kernel,
    )
    q = torch.zeros_like(expected)
    k = torch.zeros(2, 4)
    indices = torch.zeros(0, dtype=torch.int32)
    indptr = torch.zeros(3, dtype=torch.int32)
    result = paged_2src.sparse_attn_v4_paged_2src(
        q,
        k,
        indices,
        indptr,
        k,
        indices,
        indptr,
        torch.zeros(128),
        0.5,
        out=out,
        unified_kv_rope=k,
        q_packed=q,
        q_rope=q,
        k_packed=k,
        k_rope=k,
    )
    torch.testing.assert_close(result, expected)
    assert len(calls) == 1 and calls[0] is out
    if out is not None:
        assert result is out
    assert torch.all(parent[[0, 3]] == -9)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires ROCm GPU")
@pytest.mark.parametrize("two_sources", [False, True])
@pytest.mark.parametrize("alias_query", [False, True])
def test_bf16_triton_prefill_reuses_output_on_device(
    monkeypatch, two_sources, alias_query
):
    supports_two_sources = paged_2src.get_gfx().startswith("gfx1250")
    if two_sources and not supports_two_sources:
        pytest.skip("AITER two-source prefill requires gfx1250")
    monkeypatch.setattr(paged_2src, "_HAS_OPUS", False)
    # Force ATOM's fallback dispatch while AITER selects the actual GPU kernel.
    monkeypatch.setattr(paged_2src, "get_gfx", lambda: "gfx1250")
    torch.manual_seed(41)
    n, heads, dim = 3, 16, 512
    q = torch.randn(n, heads, dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(15, dim, device="cuda", dtype=q.dtype)
    extend = torch.randn(n, dim, device="cuda", dtype=q.dtype)
    indices = torch.arange(15, device="cuda", dtype=torch.int32)
    indptr = torch.tensor([0, 3, 8, 15], device="cuda", dtype=torch.int32)
    ext_indices = torch.tensor([0, 0, 1, 0, 1, 2], device="cuda", dtype=torch.int32)
    ext_indptr = torch.tensor([0, 1, 3, 6], device="cuda", dtype=torch.int32)
    if not two_sources:
        if supports_two_sources:
            extend = extend[:0]
            ext_indices = ext_indices[:0]
            ext_indptr.zero_()
        else:
            extend = ext_indices = ext_indptr = None
    sink = torch.randn(heads, device="cuda", dtype=torch.float32)
    args = (q, kv, indices, indptr, extend, ext_indices, ext_indptr, sink, dim**-0.5)
    expected = paged_2src.sparse_attn_v4_paged_2src(*args)
    parent = torch.full((n + 2, heads, dim), -99.0, device="cuda", dtype=q.dtype)
    out = parent[1:-1]
    if alias_query:
        out.copy_(q)
        args = (out, *args[1:])
    result = paged_2src.sparse_attn_v4_paged_2src(*args, out=out)
    assert result is out
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert torch.all(parent[[0, -1]] == -99)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires ROCm GPU")
@pytest.mark.parametrize("heads", [8, 128])
def test_fp8_prefill_reuses_output_on_device(heads):
    from atom.model_ops.v4_kernels.v4_quant import quantize_bf16_to_v4_2buff

    torch.manual_seed(41)
    n, dim = 3, 512
    q = torch.randn(n, heads, dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(15, dim, device="cuda", dtype=q.dtype)
    extend = torch.randn(n, dim, device="cuda", dtype=q.dtype)
    qp, qr = quantize_bf16_to_v4_2buff(q)
    kp, kr = quantize_bf16_to_v4_2buff(kv)
    ep, er = quantize_bf16_to_v4_2buff(extend)
    indices = torch.arange(15, device="cuda", dtype=torch.int32)
    indptr = torch.tensor([0, 3, 8, 15], device="cuda", dtype=torch.int32)
    ext_indices = torch.tensor([0, 0, 1, 0, 1, 2], device="cuda", dtype=torch.int32)
    ext_indptr = torch.tensor([0, 1, 3, 6], device="cuda", dtype=torch.int32)
    sink = torch.randn(heads, device="cuda", dtype=torch.float32)
    args = (None, kp, indices, indptr, None, ext_indices, ext_indptr, sink, dim**-0.5)
    kwargs = {
        "unified_kv_rope": kr,
        "q_packed": qp,
        "q_rope": qr,
        "k_packed": ep,
        "k_rope": er,
    }
    expected = paged_2src.sparse_attn_v4_paged_2src(*args, **kwargs)
    parent, out = _output_slice(n, heads, dim, q.dtype)
    out.zero_()
    untouched = parent == -99
    result = paged_2src.sparse_attn_v4_paged_2src(*args, **kwargs, out=out)
    assert result is out
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert torch.all(parent[untouched] == -99)


def _output_slice(n, heads, dim, dtype):
    parent = torch.full((n + 2, heads, dim), -99.0, device="cuda", dtype=dtype)
    return parent, parent[1:-1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires ROCm GPU")
@pytest.mark.parametrize("alias_query", [False, True])
def test_bf16_opus_prefill_output_on_device(monkeypatch, alias_query):
    if not paged_2src._HAS_OPUS:
        pytest.skip("requires BF16 OPUS")
    monkeypatch.setattr(paged_2src.envs, "ATOM_FORCE_ATTN_TRITON", False)
    opus = paged_2src.pa_sparse_prefill_opus
    calls = []

    def single_split(*args, **kwargs):
        calls.append(kwargs.get("out"))
        return opus(*args, **kwargs, num_kv_splits=1)

    def no_fallback(*args, **kwargs):
        pytest.fail("BF16 OPUS test must not pass through another backend")

    monkeypatch.setattr(paged_2src, "pa_sparse_prefill_opus", single_split)
    monkeypatch.setattr(paged_2src, "_sparse_attn_v4_paged_2src_triton", no_fallback)
    monkeypatch.setattr(paged_2src, "pa_prefill_sparse", no_fallback)
    torch.manual_seed(41)
    n, heads, dim = 3, 16, 512
    q = torch.randn(n, heads, dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(15, dim, device="cuda", dtype=q.dtype)
    extend = torch.randn(n, dim, device="cuda", dtype=q.dtype)
    indices = torch.arange(15, device="cuda", dtype=torch.int32)
    indptr = torch.tensor([0, 3, 8, 15], device="cuda", dtype=torch.int32)
    ext_indices = torch.tensor([0, 0, 1, 0, 1, 2], device="cuda", dtype=torch.int32)
    ext_indptr = torch.tensor([0, 1, 3, 6], device="cuda", dtype=torch.int32)
    sink = torch.randn(heads, device="cuda", dtype=torch.float32)
    args = (kv, indices, indptr, extend, ext_indices, ext_indptr, sink, dim**-0.5)
    expected = paged_2src.sparse_attn_v4_paged_2src(q.contiguous(), *args)
    parent, out = _output_slice(n, heads, dim, q.dtype)
    out.zero_()
    untouched = parent == -99
    if alias_query:
        out.copy_(q)
        q = out
    result = paged_2src.sparse_attn_v4_paged_2src(q, *args, out=out)
    assert result is out and len(calls) == 2 and calls[1] is out
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert torch.all(parent[untouched] == -99)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires ROCm GPU")
def test_native_triton_prefill_preserves_output_on_device(monkeypatch):
    if paged_2src.get_gfx() == "gfx1250":
        pytest.skip("gfx1250 dispatches to the AITER fallback")
    monkeypatch.setattr(paged_2src.envs, "ATOM_FORCE_ATTN_TRITON", True)
    native = paged_2src._sparse_attn_v4_paged_2src_triton
    seen = []

    def record_destination(*args, **kwargs):
        seen.append(kwargs.get("out"))
        return native(*args, **kwargs)

    monkeypatch.setattr(
        paged_2src, "_sparse_attn_v4_paged_2src_triton", record_destination
    )
    torch.manual_seed(41)
    n, heads, dim = 3, 16, 512
    q = torch.randn(n, heads, dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(15, dim, device="cuda", dtype=q.dtype)
    extend = torch.randn(n, dim, device="cuda", dtype=q.dtype)
    indices = torch.arange(15, device="cuda", dtype=torch.int32)
    indptr = torch.tensor([0, 3, 8, 15], device="cuda", dtype=torch.int32)
    ext_indices = torch.tensor([0, 0, 1, 0, 1, 2], device="cuda", dtype=torch.int32)
    ext_indptr = torch.tensor([0, 1, 3, 6], device="cuda", dtype=torch.int32)
    sink = torch.randn(heads, device="cuda", dtype=torch.float32)
    args = (q, kv, indices, indptr, extend, ext_indices, ext_indptr, sink, dim**-0.5)
    expected = paged_2src.sparse_attn_v4_paged_2src(*args)
    parent, out = _output_slice(n, heads, dim, q.dtype)
    out.zero_()
    untouched = parent == -99
    result = paged_2src.sparse_attn_v4_paged_2src(*args, out=out)
    assert result is out and len(seen) == 2 and seen[1] is out
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert torch.all(parent[untouched] == -99)

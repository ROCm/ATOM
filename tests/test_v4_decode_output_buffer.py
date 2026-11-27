# SPDX-License-Identifier: MIT
"""Caller-owned decode outputs, including sliced and padded-head destinations."""

import pytest
import torch

pytest.importorskip("aiter")
from atom.model_ops.v4_kernels import paged_decode
from atom.model_ops.v4_kernels.v4_quant import quantize_bf16_to_v4_2buff


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires ROCm GPU")
@pytest.mark.parametrize(
    "fp8,alias_query", [(False, False), (False, True), (True, False)]
)
@pytest.mark.parametrize("heads", [8, 16])
def test_decode_writes_only_the_supplied_slice(monkeypatch, fp8, heads, alias_query):
    monkeypatch.setenv("ATOM_USE_TRITON_ATTN", "1")
    monkeypatch.setenv("ATOM_USE_V4_PREFILL_ASM_FOR_DECODE", "0")
    torch.manual_seed(41)
    n, dim = 3, 512
    q = torch.randn(n, heads, dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(15, dim, device="cuda", dtype=torch.bfloat16)
    indices = torch.arange(15, device="cuda", dtype=torch.int32)
    indptr = torch.tensor([0, 3, 8, 15], device="cuda", dtype=torch.int32)
    sink = torch.randn(heads, device="cuda", dtype=torch.float32)
    kwargs = {}
    if fp8:
        qp, qr = quantize_bf16_to_v4_2buff(q)
        kv, kr = quantize_bf16_to_v4_2buff(kv)
        kwargs = {
            "unified_kv_rope": kr,
            "q_packed_in": qp,
            "q_rope_in": qr,
            "qo_indptr": torch.arange(n + 1, device="cuda", dtype=torch.int32),
        }
        q = None
    args = (q, kv, indices, indptr, sink, dim**-0.5)
    expected = paged_decode.sparse_attn_v4_paged_decode(*args, **kwargs)
    parent = torch.full(
        (n + 2, heads, dim),
        -99.0,
        device="cuda",
        dtype=torch.bfloat16,
    )
    out = parent[1:-1]
    if alias_query:
        out.copy_(q)
        args = (out, *args[1:])
    actual = paged_decode.sparse_attn_v4_paged_decode(*args, **kwargs, out=out)
    assert actual is out
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.all(parent[[0, -1]] == -99)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("layout", ["allocate", "contiguous"])
def test_prefill_asm_decode_output(monkeypatch, device, layout):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    from aiter.ops import mla_sparse_prefill

    monkeypatch.setenv("ATOM_USE_V4_PREFILL_ASM_FOR_DECODE", "1")
    monkeypatch.setattr(paged_decode, "get_gfx", lambda: "gfx1250")
    n, heads, dim = 3, 128, 512
    expected = torch.arange(n * heads * dim, device=device, dtype=torch.float32)
    expected = (expected % 37).to(torch.bfloat16).reshape(n, heads, dim)
    calls = []

    def current(**kwargs):
        out = kwargs["out"]
        assert out is None or out.is_contiguous()
        assert kwargs["kv_indptr_prefix"].numel() == n + 1
        assert kwargs["kv_indptr_extend"].numel() == n + 1
        calls.append(out)
        return expected.clone() if out is None else out.copy_(expected)

    monkeypatch.setattr(mla_sparse_prefill, "mla_sparse_prefill_fp8_asm", current)
    parent = torch.full(
        (n + 2, heads, dim),
        -99.0,
        dtype=torch.bfloat16,
        device=device,
    )
    out = None
    if layout == "contiguous":
        out = parent[1:-1]
    result = paged_decode.sparse_attn_v4_paged_decode(
        None,
        torch.zeros(15, 512, device=device, dtype=torch.uint8),
        torch.zeros(15, device=device, dtype=torch.int32),
        torch.zeros(n + 4, device=device, dtype=torch.int32),
        torch.zeros(heads, device=device),
        dim**-0.5,
        unified_kv_rope=torch.zeros(15, 64, device=device, dtype=torch.bfloat16),
        q_packed_in=torch.zeros(n, heads, 512, device=device, dtype=torch.uint8),
        q_rope_in=torch.zeros(n, heads, 64, device=device, dtype=torch.bfloat16),
        empty_kv_indptr=torch.zeros(n + 4, device=device, dtype=torch.int32),
        out=out,
    )
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert len(calls) == 1
    assert calls[0] is out
    if out is not None:
        assert result is out
    else:
        assert torch.all(parent == -99)
    assert torch.all(parent[[0, -1]] == -99)


def _decode_output_slice(q):
    n, heads, dim = q.shape
    parent = torch.full((n + 2, heads, dim), -99.0, dtype=q.dtype, device=q.device)
    out = parent[1:-1]
    out.zero_()
    return parent, out, parent == -99


@pytest.mark.parametrize("gfx", ["gfx942", "gfx1250"])
@pytest.mark.parametrize("supplied_output", [False, True])
def test_aiter_bf16_decode_output_dispatch(monkeypatch, gfx, supplied_output):
    monkeypatch.setattr(paged_decode, "get_gfx", lambda: gfx)
    q = torch.arange(24.0).reshape(3, 2, 4)
    seen = []

    def current(*args, **kwargs):
        out = kwargs["out"]
        assert kwargs["has_invalid"] is False and kwargs["kv_scales"] is None
        seen.append(out)
        return q + 1 if out is None else torch.add(q, 1, out=out)

    monkeypatch.setattr(paged_decode, "pa_decode_sparse", current)
    parent, out, untouched = _decode_output_slice(q)
    out = out if supplied_output else None
    result = paged_decode.sparse_attn_v4_paged_decode(
        q,
        q[0],
        torch.empty(0, dtype=torch.int32),
        torch.zeros(4, dtype=torch.int32),
        torch.zeros(2),
        0.5,
        out=out,
    )
    assert len(seen) == 1 and seen[0] is out
    if out is not None:
        assert result is out
    torch.testing.assert_close(result, q + 1)
    assert torch.all(parent[untouched] == -99)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires ROCm GPU")
@pytest.mark.parametrize("kv_splits", [1, 2])
@pytest.mark.parametrize("alias_query", [False, True])
def test_aiter_bf16_decode_direct_output_on_device(monkeypatch, kv_splits, alias_query):
    gfx = paged_decode.get_gfx()
    if not (gfx.startswith("gfx94") or gfx in ("gfx950", "gfx1250")):
        pytest.skip("requires a supported ROCm architecture")
    kernel = paged_decode.pa_decode_sparse
    seen = []

    def record_destination(*args, out=None, **kwargs):
        seen.append(out)
        # Explicit block_h selects the shared Triton BF16 kernel on gfx950,
        # where ATOM normally uses its native decode path instead of AITER.
        return kernel(*args, out=out, block_h=16, kv_splits=kv_splits, **kwargs)

    monkeypatch.setattr(paged_decode, "pa_decode_sparse", record_destination)
    if gfx == "gfx950":
        monkeypatch.setattr(paged_decode, "get_gfx", lambda: "gfx942")
    torch.manual_seed(41)
    n, heads, dim = 3, 16, 512
    q = torch.randn(n, heads, dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(15, dim, device="cuda", dtype=q.dtype)
    indices = torch.arange(15, device="cuda", dtype=torch.int32)
    indptr = torch.tensor([0, 3, 8, 15], device="cuda", dtype=torch.int32)
    sink = torch.randn(heads, device="cuda", dtype=torch.float32)
    args = (q, kv, indices, indptr, sink, dim**-0.5)
    expected = paged_decode.sparse_attn_v4_paged_decode(*args)
    parent, out, untouched = _decode_output_slice(q)
    if alias_query:
        out.copy_(q)
        args = (out, *args[1:])
    result = paged_decode.sparse_attn_v4_paged_decode(*args, out=out)
    assert result is out and len(seen) == 2 and seen[1] is out
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert torch.all(parent[untouched] == -99)

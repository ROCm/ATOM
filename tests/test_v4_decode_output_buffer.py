# SPDX-License-Identifier: MIT
"""Caller-owned decode outputs, including sliced and padded-head destinations."""

import pytest
import torch

pytest.importorskip("aiter")
from atom.model_ops.v4_kernels import paged_decode
from atom.model_ops.v4_kernels.v4_quant import quantize_bf16_to_v4_2buff


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires ROCm GPU")
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.parametrize("strided", [False, True])
def test_decode_writes_only_the_supplied_slice(monkeypatch, fp8, heads, strided):
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
        (n + 2, heads, dim * (2 if strided else 1)),
        -99.0,
        device="cuda",
        dtype=torch.bfloat16,
    )
    out = parent[1:-1, :, ::2] if strided else parent[1:-1]
    actual = paged_decode.sparse_attn_v4_paged_decode(*args, **kwargs, out=out)
    assert actual is out
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.all(parent[[0, -1]] == -99)
    if strided:
        assert torch.all(parent[:, :, 1::2] == -99)


@pytest.mark.parametrize("legacy", [False, True])
def test_output_api_is_compile_safe(legacy):
    from atom.model_ops.v4_kernels.output_buffer import call_with_output

    def old(x):
        return x + 1

    def new(x, *, out=None):
        return x + 1 if out is None else torch.add(x, 1, out=out)

    kernel = old if legacy else new

    def run(x, out):
        return call_with_output(kernel, (x,), out)

    out = torch.empty(4)
    torch.compile(run, fullgraph=True, backend="eager")(torch.ones(4), out)
    torch.testing.assert_close(out, torch.full((4,), 2.0))

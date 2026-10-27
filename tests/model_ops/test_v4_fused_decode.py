# SPDX-License-Identifier: MIT
"""Single-pass V4 decode: ragged CSR, attention sinks and live graph inputs."""

import pytest
import torch

pytest.importorskip("triton")
pytest.importorskip("aiter")

from atom.model_ops.v4_kernels import paged_decode
from atom.model_ops.v4_kernels.paged_decode import (
    _sparse_attn_v4_paged_decode_triton,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ROCm GPU required"
)


@pytest.fixture(params=[False, True], ids=["native", "cdna3_layout"])
def fused_arch(request, monkeypatch):
    if request.param:
        if paged_decode.get_gfx() != "gfx950":
            pytest.skip("CDNA3 layout coverage on CDNA4 requires gfx950")
        # CDNA4 can execute the older BF16 MFMA. Exercise its layout and K32
        # tiling against the same oracle; this is not gfx942 hardware validation.
        monkeypatch.setattr(paged_decode, "get_gfx", lambda: "gfx942")


def _inputs(heads, dim, dtype, quantized=False):
    torch.manual_seed(20260922)
    tokens = 512
    lengths = torch.tensor(
        [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 640, 1152],
        device="cuda",
        dtype=torch.int32,
    ).repeat(tokens // 16)
    ptr = torch.cat((lengths.new_zeros(1), lengths.cumsum(0).to(torch.int32)))
    indices = torch.randint(4096, (int(ptr[-1]),), device="cuda", dtype=torch.int32)
    # Exercise strides independently of the contiguous production layout.
    query = torch.randn(tokens, heads, dim + 16, device="cuda", dtype=dtype)[..., :dim]
    cache = torch.randn(4096, dim + 16, device="cuda", dtype=dtype)[:, :dim]
    sink = torch.linspace(-12, 12, heads, device="cuda")
    scales = None
    if quantized:
        cache = cache.to(torch.float8_e4m3fnuz)
        scales = torch.rand(4096, dim // 64, device="cuda") + 0.25
    return query, cache, indices, ptr, sink, scales


def _reference(query, cache, indices, ptr, sink, scales):
    if scales is not None:
        cache = cache.to(query.dtype) * scales.to(query.dtype).repeat_interleave(64, 1)
    lengths = ptr[1:] - ptr[:-1]
    expected = torch.zeros_like(query)
    dim = query.shape[-1]
    for length in lengths.unique().tolist():
        if length == 0:
            continue
        rows = torch.where(lengths == length)[0]
        positions = ptr[rows, None] + torch.arange(length, device=query.device)
        values = cache[indices[positions].long()].double()
        logits = torch.bmm(query[rows].double(), values.transpose(1, 2)) * dim**-0.5
        sink_logits = sink.double()[None, :, None].expand(rows.numel(), -1, 1)
        probabilities = torch.cat((logits, sink_logits), -1).softmax(-1)[..., :-1]
        expected[rows] = torch.bmm(probabilities, values).to(query.dtype)
    return expected


def _check(actual, expected, ptr):
    fp16 = actual.dtype == torch.float16
    # PV rounds probabilities to the operand dtype. Use each head's output
    # magnitude for cancellation-sensitive elements, plus an L2 bound below.
    peak = expected.float().abs().amax(-1, keepdim=True).clamp_min(1e-30)
    torch.testing.assert_close(
        actual.float() / peak,
        expected.float() / peak,
        rtol=0,
        atol=1 / 256 if fp16 else 1 / 64,
    )
    error = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert error < (3e-4 if fp16 else 3e-3)
    assert torch.count_nonzero(actual[ptr[1:] == ptr[:-1]]) == 0


@pytest.mark.parametrize(
    "heads,dim,dtype,quantized",
    [
        (16, 512, torch.bfloat16, False),
        (32, 512, torch.bfloat16, False),
        (64, 512, torch.bfloat16, False),
        (128, 512, torch.bfloat16, False),
        (17, 512, torch.bfloat16, False),
        (32, 96, torch.bfloat16, False),
        (32, 512, torch.float16, False),
        (32, 512, torch.bfloat16, True),
    ],
)
def test_fused_decode_matches_fp64_with_ragged_rows(
    heads, dim, dtype, quantized, fused_arch
):
    query, cache, indices, ptr, sink, scales = _inputs(heads, dim, dtype, quantized)
    actual = _sparse_attn_v4_paged_decode_triton(
        query, cache, indices, ptr, sink, dim**-0.5, scales, kv_splits=1
    )
    expected = _reference(query, cache, indices, ptr, sink, scales)
    _check(actual, expected, ptr)


@pytest.mark.parametrize("heads", [16, 32])
def test_fused_decode_graph_reads_changed_csr_and_sink(heads, fused_arch):
    query, cache, indices, ptr, sink, scales = _inputs(heads, 512, torch.bfloat16)

    def forward():
        return _sparse_attn_v4_paged_decode_triton(
            query, cache, indices, ptr, sink, 512**-0.5, kv_splits=1
        )

    forward()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = forward()
    for _ in range(3):
        lengths = (ptr[1:] - ptr[:-1]).roll(1)
        ptr[1:].copy_(lengths.cumsum(0))
        indices.copy_(indices.roll(17))
        query.copy_(torch.randn_like(query))
        cache.mul_(-1)
        sink.neg_()
        graph.replay()
        expected = _reference(query, cache, indices, ptr, sink, scales)
        _check(actual, expected, ptr)


@pytest.mark.parametrize("heads", [16, 32])
def test_fused_decode_addresses_past_int32_element_offsets(heads, fused_arch):
    torch.manual_seed(20260922)
    tokens, dim = 512, 512
    first_slot = 1 << 22
    # Only these two rows are read; their element offsets exceed int32.
    cache = torch.empty(first_slot + 2, dim, device="cuda", dtype=torch.bfloat16)
    cache[-2:].normal_()
    query = torch.randn(tokens, heads, dim, device="cuda", dtype=cache.dtype)
    indices = torch.tensor(
        [first_slot, first_slot + 1], device="cuda", dtype=torch.int32
    ).repeat(tokens)
    ptr = torch.arange(tokens + 1, device="cuda", dtype=torch.int32) * 2
    sink = torch.randn(heads, device="cuda")
    actual = _sparse_attn_v4_paged_decode_triton(
        query, cache, indices, ptr, sink, dim**-0.5, kv_splits=1
    )
    expected = _reference(query, cache, indices, ptr, sink, None)
    _check(actual, expected, ptr)


@pytest.mark.parametrize("heads", [16, 32])
def test_fused_decode_strided_channels(heads, fused_arch):
    query, cache, indices, ptr, sink, scales = _inputs(heads, 512, torch.bfloat16)
    q_storage = torch.empty(*query.shape[:-1], 1024, device="cuda", dtype=query.dtype)
    kv_storage = torch.empty(cache.shape[0], 1024, device="cuda", dtype=cache.dtype)
    q_storage[..., ::2].copy_(query)
    kv_storage[:, ::2].copy_(cache)
    query, cache = q_storage[..., ::2], kv_storage[:, ::2]
    actual = _sparse_attn_v4_paged_decode_triton(
        query, cache, indices, ptr, sink, 512**-0.5, kv_splits=1
    )
    _check(actual, _reference(query, cache, indices, ptr, sink, scales), ptr)


@pytest.mark.parametrize("block_k,arch", [(16, None), (None, "gfx90a")])
def test_fused_decode_fallback_matches_reference(block_k, arch, monkeypatch):
    if arch is not None:
        monkeypatch.setattr(paged_decode, "get_gfx", lambda: arch)
    # Explicit tile overrides and unsupported devices must remain usable
    # without invoking the Gluon implementation.
    monkeypatch.setattr(paged_decode, "_paged_decode_fused_gluon_kernel", None)
    query, cache, indices, ptr, sink, scales = _inputs(16, 512, torch.bfloat16)
    actual = _sparse_attn_v4_paged_decode_triton(
        query, cache, indices, ptr, sink, 512**-0.5, kv_splits=1, block_k=block_k
    )
    _check(actual, _reference(query, cache, indices, ptr, sink, scales), ptr)

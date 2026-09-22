# SPDX-License-Identifier: MIT
"""Single-pass V4 decode: ragged CSR, attention sinks and live graph inputs."""

import pytest
import torch

pytest.importorskip("triton")
pytest.importorskip("aiter")

from atom.model_ops.v4_kernels.paged_decode import (
    _sparse_attn_v4_paged_decode_triton,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ROCm GPU required"
)


def _inputs(heads, dim, dtype, quantized=False):
    torch.manual_seed(20260922)
    tokens = 512
    lengths = torch.tensor(
        [0, 1, 15, 16, 17, 63, 128, 129, 640], device="cuda", dtype=torch.int32
    ).repeat(57)[:tokens]
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
def test_fused_decode_matches_fp64_with_ragged_rows(heads, dim, dtype, quantized):
    query, cache, indices, ptr, sink, scales = _inputs(heads, dim, dtype, quantized)
    actual = _sparse_attn_v4_paged_decode_triton(
        query, cache, indices, ptr, sink, dim**-0.5, scales, kv_splits=1
    )
    expected = _reference(query, cache, indices, ptr, sink, scales)
    _check(actual, expected, ptr)


def test_fused_decode_graph_reads_changed_csr_and_sink():
    query, cache, indices, ptr, sink, scales = _inputs(32, 512, torch.bfloat16)

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

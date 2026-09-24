# SPDX-License-Identifier: MIT
"""Projection contracts at small-chunk and batched-decode boundaries."""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("aiter", reason="the projections call AITER GEMMs")

from atom.model_ops.deepseek_v41.projections import (
    grouped_output_projection,
    hc_projection,
)


def test_cpu_projections_preserve_native_arithmetic():
    torch.manual_seed(2)
    hidden = torch.randn(1, 2, 2, 128, dtype=torch.bfloat16)
    weight = torch.randn(2, 32, 128, dtype=torch.bfloat16)
    coefficients = torch.randn(1, 2, 128) * 1e-8
    fn = torch.randn(24, 128)
    assert torch.equal(
        grouped_output_projection(hidden, weight),
        torch.einsum("bsgd,grd->bsgr", hidden, weight),
    )
    assert torch.equal(hc_projection(coefficients, fn), F.linear(coefficients, fn))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("batch,tokens", [(1, 1), (1, 2), (3, 1), (4, 6), (2, 16)])
def test_small_rows_match_fp64_grouped_projection(batch, tokens):
    """V4's small-row GEMM at actual TP4 shapes, including verify batches."""
    torch.manual_seed(314)
    hidden = torch.randn(batch, tokens, 2, 4096, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2, 1024, 4096, device="cuda", dtype=torch.bfloat16)
    expected = torch.einsum("bsgd,grd->bsgr", hidden.double(), weight.double())
    actual = grouped_output_projection(hidden, weight)
    assert actual.is_contiguous()
    assert actual.shape == expected.shape
    assert actual.dtype == hidden.dtype
    # FP32 dot accumulation followed by one BF16 rounding; cancellation needs
    # an absolute allowance as well as the one-ULP relative bound.
    torch.testing.assert_close(actual, expected.bfloat16(), rtol=1 / 128, atol=2**-9)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("rows", [33, 63, 64, 65, 127, 128, 189])
def test_larger_rows_preserve_native_projection(rows):
    torch.manual_seed(27)
    hidden = torch.randn(1, rows, 2, 4096, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2, 1024, 4096, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(
        grouped_output_projection(hidden, weight),
        torch.einsum("bsgd,grd->bsgr", hidden, weight),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("rows", [1, 2, 24, 64, 65])
def test_reference_hc_projection_preserves_row_policy(rows):
    coefficients = torch.randn(128, 20480, device="cuda")
    fn = torch.randn(24, 20480, device="cuda")
    expected = F.linear(coefficients if 1 < rows <= 64 else coefficients[:rows], fn)[
        :rows
    ]
    actual = hc_projection(coefficients[:rows], fn)
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_graph_replay_reads_updated_projection_inputs():
    torch.manual_seed(20)
    x = torch.randn(1, 3, 2, 4096, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2, 1024, 4096, device="cuda", dtype=torch.bfloat16)
    hc = torch.randn(1, 3, 20480, device="cuda")
    fn = torch.randn(24, 20480, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            grouped_output_projection(x, weight)
            hc_projection(hc, fn)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = grouped_output_projection(x, weight)
        mixes = hc_projection(hc, fn)
    for _ in range(3):
        x.normal_()
        hc.normal_()
        graph.replay()
        assert torch.equal(output, grouped_output_projection(x, weight))
        assert torch.equal(mixes, hc_projection(hc, fn))

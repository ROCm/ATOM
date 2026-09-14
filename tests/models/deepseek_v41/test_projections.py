# SPDX-License-Identifier: MIT
"""Projection contracts at small-chunk and batched-decode boundaries."""

import pytest
import torch
import torch.nn.functional as F

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
@pytest.mark.parametrize("batch,tokens", [(1, 2), (3, 1), (2, 31), (1, 63), (4, 16)])
def test_small_rows_match_shared_prefix_of_native128(batch, tokens):
    """Real TP4 shapes; unrelated later rows must not affect the prefix."""
    torch.manual_seed(314)
    rows = batch * tokens
    hidden = torch.randn(1, 128, 2, 4096, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2, 1024, 4096, device="cuda", dtype=torch.bfloat16)
    coefficients = torch.randn(128, 20480, device="cuda")
    fn = torch.randn(24, 20480, device="cuda")
    expected = torch.einsum("bsgd,grd->bsgr", hidden, weight)[:, :rows]
    actual = grouped_output_projection(
        hidden[:, :rows].view(batch, tokens, 2, 4096), weight
    )
    assert torch.equal(actual.flatten(0, 1), expected.flatten(0, 1))
    expected_mix = F.linear(coefficients, fn)[:rows]
    actual_mix = hc_projection(coefficients[:rows].view(batch, tokens, -1), fn)
    assert torch.equal(actual_mix.flatten(0, 1), expected_mix)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("rows", [1, 65, 127, 128, 189])
def test_other_rows_preserve_native_projection(rows):
    torch.manual_seed(27)
    hidden = torch.randn(1, rows, 2, 4096, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2, 1024, 4096, device="cuda", dtype=torch.bfloat16)
    coefficients = torch.randn(1, rows, 20480, device="cuda")
    fn = torch.randn(24, 20480, device="cuda")
    assert torch.equal(
        grouped_output_projection(hidden, weight),
        torch.einsum("bsgd,grd->bsgr", hidden, weight),
    )
    assert torch.equal(hc_projection(coefficients, fn), F.linear(coefficients, fn))


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

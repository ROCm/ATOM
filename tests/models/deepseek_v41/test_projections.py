# SPDX-License-Identifier: MIT
"""Projection contracts at small-chunk and batched-decode boundaries."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

pytest.importorskip("aiter", reason="the projections call AITER GEMMs")

from aiter import QuantType
from aiter.jit.utils.chip_info import get_gfx_runtime

from atom.model_ops.deepseek_v41.draft_block import rotate_rows
from atom.model_ops.deepseek_v41.projections import (
    grouped_output_projection,
    hc_projection,
)
from atom.model_ops.deepseek_v41.rotary import RotaryEmbedding
from atom.models.deepseek_v41.attention import Attention


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


# --- wo_a: fp8 e8m0 mxscale BMM against the BF16 dequant it replaces --------
#
# TP2 local shapes: G=4, N=1024, K = heads * head_dim / G = 4096.
_HEADS, _HEAD_DIM, _GROUPS, _RANK = 32, 512, 4, 1024
_K = _HEADS * _HEAD_DIM // _GROUPS
_BLOCK = 32

_needs_mxscale = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx_runtime() != "gfx950",
    reason="the preshuffled 32x32 mxscale BMM is a gfx950 kernel",
)


class _WoA(nn.Module):
    """The fields `process_weights_after_loading` reads off the linear."""

    def __init__(self, weight, scale):
        super().__init__()
        self.weight = nn.Parameter(weight.clone(), requires_grad=False)
        self.weight_scale = nn.Parameter(scale.clone(), requires_grad=False)
        self.quant_type = QuantType.per_1x128
        self.need_normalize_e4m3fn_to_e4m3fnuz = False


def _attention(weight, scale, *, gfx950):
    """An `Attention` reduced to what the wo_a projection touches.

    Built without `__init__` on purpose: the real one allocates every
    projection in the layer, and none of them reach this path.
    """
    attention = Attention.__new__(Attention)
    nn.Module.__init__(attention)
    attention.spec = SimpleNamespace(layer_id=0)
    attention.heads, attention.head_dim = _HEADS, _HEAD_DIM
    attention.groups, attention.o_rank = _GROUPS, _RANK
    attention._is_gfx950 = gfx950
    attention._wo_a_mxscale = False
    attention._wo_a_w_fp8 = attention._wo_a_w_scale = None
    attention.wo_a = _WoA(weight, scale)
    attention.wo_b = nn.Identity()
    attention.process_weights_after_loading()
    return attention


@pytest.fixture(scope="module")
def wo_a_pair():
    torch.manual_seed(0)
    weight = (torch.randn(_GROUPS * _RANK, _K, device="cuda") * 0.5).to(
        torch.float8_e4m3fn
    )
    # Exact powers of two: anything else has no e8m0 form and would send both
    # sides down the BF16 branch, testing nothing.
    exponents = torch.randint(
        -10, -4, (_GROUPS * _RANK // _BLOCK, _K // _BLOCK), device="cuda"
    )
    scale = torch.exp2(exponents.float()).to(torch.float8_e8m0fnu)
    mxscale = _attention(weight, scale, gfx950=True)
    bf16 = _attention(weight, scale, gfx950=False)
    assert mxscale._wo_a_mxscale, "the mxscale path was not selected"
    assert not bf16._wo_a_mxscale and bf16.wo_a.weight.dtype == torch.bfloat16
    rope = RotaryEmbedding(
        64, 1 << 17, base=160000, original_length=65536, factor=16,
        beta_fast=32, beta_slow=1,
    ).to("cuda")
    return mxscale, bf16, rope


@_needs_mxscale
@pytest.mark.parametrize("rows", [1, 2, 5, 16, 33, 64, 300, 2048, 8192])
def test_wo_a_mxscale_tracks_the_bf16_projection(wo_a_pair, rows):
    """One FP8 rounding of the activation, so this is a closeness bound."""
    mxscale, bf16, rope = wo_a_pair
    torch.manual_seed(rows)
    positions = torch.randint(0, 100000, (rows,), device="cuda", dtype=torch.int64)
    source = torch.randn(
        1, rows, _HEADS, _HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )

    expected = bf16._project_out(rope(source.clone(), positions, inverse=True))
    actual = mxscale._project_out(source.clone(), rope=rope, positions=positions)

    assert actual.shape == expected.shape
    reference, got = expected.float(), actual.float()
    assert ((got - reference).norm() / reference.norm()).item() < 0.08
    similarity = F.cosine_similarity(got.flatten(), reference.flatten(), dim=0)
    assert similarity.item() > 0.997


@_needs_mxscale
@pytest.mark.parametrize("batch,tokens", [(1, 6), (3, 6), (8, 6)])
def test_wo_a_mxscale_covers_draft_batches(wo_a_pair, batch, tokens):
    """Draft rows arrive as [B, T] positions against B*T rows.

    The BF16 side goes through `rotate_rows` rather than the rope directly:
    `RotaryEmbedding` cannot broadcast a 2D position tensor over the batch,
    which is the reason the draft block owns that flattening.
    """
    mxscale, bf16, rope = wo_a_pair
    torch.manual_seed(batch * tokens)
    positions = torch.randint(
        0, 100000, (batch, tokens), device="cuda", dtype=torch.int64
    )
    source = torch.randn(
        batch, tokens, _HEADS, _HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )

    expected = bf16._project_out(
        rotate_rows(rope, source.clone(), positions, inverse=True)
    )
    actual = mxscale._project_out(source.clone(), rope=rope, positions=positions)

    assert actual.shape == expected.shape
    reference, got = expected.float(), actual.float()
    assert ((got - reference).norm() / reference.norm()).item() < 0.08
    similarity = F.cosine_similarity(got.flatten(), reference.flatten(), dim=0)
    assert similarity.item() > 0.997

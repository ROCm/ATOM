# SPDX-License-Identifier: MIT
"""The real row-range hyper-connection kernels the SDMA sequence-parallel path
uses: `mix_fused_rows` and the `out=` / `h_out=` of `hc_gated_mean` /
`hc_combine_norm` write exactly their rows of a larger buffer, bit-identical
to the allocating calls."""

import itertools
import types

import pytest
import torch
from conftest import gpu_stack_import

pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires a ROCm GPU"
)

# Qwen3.8-Flash-Next.
HIDDEN, HC, LOWRANK = 2560, 4, 320
BF16 = torch.bfloat16


def _hc(monkeypatch, has_block_inject=True):
    linear = gpu_stack_import("atom.model_ops.linear")
    hyperconnection = gpu_stack_import("atom.model_ops.qwen4_exp.hyperconnection")
    group = types.SimpleNamespace(rank_in_group=0, world_size=1)
    monkeypatch.setattr(linear, "get_tp_group", lambda: group)
    torch.manual_seed(0)
    hc = hyperconnection.Qwen4ExpHyperConnection(
        HIDDEN, HC, LOWRANK, has_block_inject=has_block_inject
    ).to(device="cuda", dtype=BF16)
    with torch.no_grad():
        hc.hc_norm.weight.normal_(0, 0.1)
        for p in hc.parameters():
            if p.dim() == 2:
                p.normal_(0, 0.02)
    hc.process_weights_after_loading()
    return hc


def _nan(*shape):
    return torch.full(shape, float("nan"), dtype=BF16, device="cuda")


def _same(a, b):
    assert a.shape == b.shape
    assert torch.equal(a.view(torch.int16), b.view(torch.int16))


# Chunks across `hc_rows` (<= 2 rows), both `hc_combine_norm` kernels (below
# and above 256 rows) and the tiled `hc_gated_mean`.
BOUNDS = [0, 2, 3, 200, 457, 1700, 3001]


@pytest.mark.parametrize("combine", [False, True])
def test_mix_fused_rows_writes_its_rows(monkeypatch, combine):
    hc = _hc(monkeypatch)
    t = BOUNDS[-1]
    streams = torch.randn(t, HC * HIDDEN, dtype=BF16, device="cuda")
    block_out = torch.randn(t, HIDDEN, dtype=BF16, device="cuda")
    raw_prev = torch.randn(t, HC, dtype=BF16, device="cuda")
    h_out, raw_out = _nan(t, HC * HIDDEN), _nan(t, HC)
    mixed_all = _nan(t, HIDDEN)
    for r0, r1 in itertools.pairwise(BOUNDS):
        rows = slice(r0, r1)
        pending = (block_out[rows], raw_prev) if combine else None
        mixed_out = mixed_all[rows]
        hc.mix_fused_rows(
            streams,
            pending,
            rows,
            mixed_out=mixed_out,
            h_out=h_out,
            raw_out=raw_out,
        )
        # Rows past this chunk are untouched.
        assert mixed_all[r1:].isnan().all() and raw_out[r1:].isnan().all()
        if combine:
            assert h_out[r1:].isnan().all()
        ref_pending = (block_out[rows], raw_prev[rows]) if combine else None
        ref_mixed, (ref_h, ref_raw) = hc.mix_fused(streams[rows], ref_pending)
        _same(mixed_out, ref_mixed)
        _same(raw_out[rows], ref_raw)
        if combine:
            _same(h_out[rows], ref_h)
    assert not h_out.isnan().any() if combine else h_out.isnan().all()
    # Chunked vs one call: the HC GEMMs pick per-M kernels, so close, not equal.
    pending = (block_out, raw_prev) if combine else None
    ref_mixed, _ = hc.mix_fused(streams, pending)
    torch.testing.assert_close(mixed_all, ref_mixed, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("m", [3, 200, 1700])
@pytest.mark.parametrize("hidden", [HIDDEN, 96])  # tiled kernel / torch fallback
def test_hc_gated_mean_out_rows(m, hidden):
    ops = gpu_stack_import("atom.model_ops.qwen4_exp.ops.hc_fused")
    normed = torch.randn(m, HC * hidden, dtype=BF16, device="cuda")
    gate = torch.randn(m, HC * hidden, dtype=BF16, device="cuda")
    ref = ops.hc_gated_mean(normed, gate, HC)
    # Rows of a wider buffer: the row stride is not `hidden`.
    big = _nan(m + 7, hidden + 64)
    out = ops.hc_gated_mean(normed, gate, HC, out=big[5 : 5 + m, :hidden])
    assert out.data_ptr() == big[5:].data_ptr()
    _same(big[5 : 5 + m, :hidden], ref)
    assert big[:5].isnan().all() and big[5 + m :].isnan().all()
    assert big[:, hidden:].isnan().all()


@pytest.mark.parametrize("m", [3, 200, 1700])  # both combine-norm kernels
def test_hc_combine_norm_h_out_rows(m):
    ops = gpu_stack_import("atom.model_ops.qwen4_exp.ops.hc_fused")
    full = torch.randn(m + 7, HC * HIDDEN, dtype=BF16, device="cuda")
    h = full[5 : 5 + m]
    weight = torch.randn(HC * HIDDEN, dtype=BF16, device="cuda") * 0.1
    block_out = torch.randn(m, HIDDEN, dtype=BF16, device="cuda")
    raw = torch.randn(m, HC, dtype=BF16, device="cuda")
    ref_h, ref_normed = ops.hc_combine_norm(h, weight, 1e-6, HC, block_out, raw)
    big = _nan(m + 7, HC * HIDDEN)
    h_new, normed = ops.hc_combine_norm(
        h, weight, 1e-6, HC, block_out, raw, h_out=big[5 : 5 + m]
    )
    assert h_new.data_ptr() == big[5:].data_ptr()
    _same(big[5 : 5 + m], ref_h)
    _same(normed, ref_normed)
    assert big[:5].isnan().all() and big[5 + m :].isnan().all()
    with pytest.raises(AssertionError, match="strides"):
        ops.hc_combine_norm(
            h, weight, 1e-6, HC, block_out, raw, h_out=_nan(m, 2 * HC * HIDDEN)[:, ::2]
        )

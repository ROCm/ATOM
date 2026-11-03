# SPDX-License-Identifier: MIT
"""The FP8 pool: 528-byte MXFP8 rows, row addressed, read by AITER's v4.1
sparse attention."""

from dataclasses import replace

import pytest
import torch

from atom.model_ops.attentions.pool_layout.entry_arena import field_extents
from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry

# V4.1-Flash's owners at the served page (128 tokens), FP4 index plane
FLASH = {
    "layers": 43,
    "owners": ((2, 2), (8, 2), (14, 2), (20, 1), (24, 2), (30, 1)),
    "block_size": 128,
    "window_size": 128,
    "head_dim": 512,
    "index_dim": 128,
    "speculative_tokens": 5,
    "index_block_rows": 8,
    "index_fp4": True,
}


def test_fp8_rows_are_whole_rows_of_the_pool():
    geo = V41PoolGeometry(**FLASH, kv_dtype="fp8")
    bf16 = replace(geo, kv_dtype="bf16")
    assert (geo.row_bytes, geo.main_row_bytes, geo.window_row_bytes) == (528,) * 3
    assert geo.alignment % 528 == 0 and geo.alignment % 256 == 0
    # every region a row id reaches starts on a whole row of the pool
    assert geo.page_bytes % 528 == 0 and geo.state_bytes % 528 == 0
    for field, start, _ in field_extents(geo.page_fields):
        assert start % 528 == 0, field.name
    pages = 1000
    for layer in (0, geo.layers - 1):
        rows, raw = geo.window(layer, pages), geo.window_bytes(layer, pages)
        assert rows.ring_start * 528 == raw.ring_start
        assert rows.slot_rows * 528 == raw.slot_rows
    # the padding to whole rows costs nothing at the served page
    assert geo.page_bytes == sum(f.bytes_per_entry for f in geo.page_fields)
    assert geo.page_bytes / bf16.page_bytes == pytest.approx(528 / 1024)
    assert geo.layout_id != bf16.layout_id


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("heads", [32, 64])
def test_v41_attention_matches_its_dequantized_reference(heads):
    from aiter.jit.utils.chip_info import get_gfx

    if get_gfx() != "gfx950":
        pytest.skip("mla_sparse_prefill_v41 is gfx950 only")
    from atom.model_ops.attentions.deepseek_v41.fp8_attention import (
        sparse_attn_v41_paged_2src,
        sparse_attn_v41_paged_2src_reference,
    )
    from atom.model_ops.attentions.deepseek_v41.packed_rows import pack_rows
    from atom.model_ops.blockscale import quantize_fp8

    torch.manual_seed(41)
    device, dim, tokens = "cuda", 512, 7

    def rows(n):
        return pack_rows(*quantize_fp8(torch.randn(n, dim, device=device).bfloat16()))

    pool, extend = rows(4096), rows(tokens + 128)
    prefix_counts = [0, 1, 37, 512, 300, 64, 129]
    extend_counts = [128, 5, 0, 1, 128, 77, 33]
    prefix = torch.cat(
        [torch.randperm(4096, device=device)[:n] for n in prefix_counts]
    ).int()
    second = torch.cat(
        [torch.randperm(tokens + 128, device=device)[:n] for n in extend_counts]
    ).int()

    def indptr(counts):
        return torch.tensor([0, *counts], device=device).cumsum(0).int()

    query = quantize_fp8(torch.randn(tokens, heads, dim, device=device).bfloat16())
    sink = torch.randn(heads, device=device)
    args = (
        query,
        pool,
        prefix,
        indptr(prefix_counts),
        extend,
        second,
        indptr(extend_counts),
        sink,
        dim**-0.5,
    )
    actual = sparse_attn_v41_paged_2src(*args)
    expected = sparse_attn_v41_paged_2src_reference(*args)
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)

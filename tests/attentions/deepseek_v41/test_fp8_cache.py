# SPDX-License-Identifier: MIT
"""The FP8 cache: contiguous source planes read by AITER sparse attention."""

import functools
from dataclasses import replace

import pytest
import torch
from aiter_stub import atom_without_aiter

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


def test_fp8_source_rows_preserve_budget_and_use_local_addresses():
    geo = V41PoolGeometry(**FLASH, kv_dtype="fp8")
    bf16 = replace(geo, kv_dtype="bf16")
    assert (geo.row_bytes, geo.main_row_bytes, geo.window_row_bytes) == (528,) * 3
    assert geo.alignment % 528 == 0 and geo.alignment % 256 == 0
    # Keep the existing logical PAGE/STATE budget and its alignment.
    assert geo.page_bytes % 528 == 0 and geo.state_bytes % 528 == 0
    for field, start, _ in field_extents(geo.page_fields):
        assert start % 528 == 0, field.name
    pages = 1000
    for layer in (0, geo.layers - 1):
        rows = geo.window(layer, pages)
        assert rows.ring_start == 0
        assert rows.slot_rows == rows.ring_slots == geo.ring_slots
        with pytest.raises(ValueError, match="separate data/scale"):
            geo.window_bytes(layer, pages)
    # the padding to whole rows costs nothing at the served page
    assert geo.page_bytes == sum(f.bytes_per_entry for f in geo.page_fields)
    assert geo.page_bytes / bf16.page_bytes == pytest.approx(528 / 1024)
    assert geo.layout_id != bf16.layout_id


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("planar", [False, True])
def test_v41_attention_matches_its_dequantized_reference(heads, planar):
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
        values = quantize_fp8(torch.randn(n, dim, device=device).bfloat16())
        return tuple(v.contiguous() for v in values) if planar else pack_rows(*values)

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("heads", [8, 16, 32, 64])
@pytest.mark.parametrize(
    "path",
    [
        "entry_split",
        "entry_whole",
        "in_kernel",
        "in_kernel_whole",
        "unique",
        "per_entry",
        "ring",
    ],
)
def test_v41_bf16_query_attention_matches_its_dequantized_reference(
    monkeypatch, heads, path
):
    """The BF16-query attention over MXFP8 planes (the path gfx942 takes)
    against V4's reference on the dequantized rows: the in-register kernel
    split and whole (the in-kernel sink, which graph capture reaches once a
    bucket fills the CUs), the prefill gather with its rows decoded once, the
    per-entry gather past the scratch bound, a decode-sized ring gathered by
    row, and the BF16 entry, which takes the in-register kernel when V4 would
    split the keys and the gather otherwise. Eight heads is one TP8 rank of
    the 64-head Flash model."""
    from atom.model_ops.attentions.deepseek_v41 import fp8_attention
    from atom.model_ops.blockscale import quantize_fp8
    from atom.model_ops.v4_kernels import sparse_attn_v4_paged_2src_reference

    torch.manual_seed(42)
    device, dim, tokens = "cuda", 512, 7

    def rows(n):
        values = quantize_fp8(torch.randn(n, dim, device=device).bfloat16())
        return tuple(v.contiguous() for v in values)

    # A source no larger than its selection is decoded whole (prefill); a
    # larger one is a ring, gathered like the CSA pool (decode).
    extend_rows = 4096 if path == "ring" else tokens + 128
    pool, extend = rows(4096), rows(extend_rows)
    prefix_counts = [0, 1, 37, 512, 300, 64, 129]
    extend_counts = [128, 5, 0, 1, 128, 77, 33]
    prefix = torch.cat(
        [torch.randperm(4096, device=device)[:n] for n in prefix_counts]
    ).int()
    second = torch.cat(
        [torch.randperm(extend_rows, device=device)[:n] for n in extend_counts]
    ).int()

    def indptr(counts):
        return torch.tensor([0, *counts], device=device).cumsum(0).int()

    if path == "per_entry":
        # Fewer scratch rows than the step references: no unique decode,
        # query ranges tiled to the bound.
        monkeypatch.setattr(fp8_attention, "_SCRATCH_ROWS", 256)
    query = torch.randn(tokens, heads, dim, device=device).bfloat16()
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
    taken = []
    if path.startswith("entry"):
        # Seven tokens split on any real device; the split count is what the
        # entry's choice turns on.
        splits = 5 if path == "entry_split" else 1
        monkeypatch.setattr(fp8_attention, "_csa_num_splits", lambda *_: splits)
        for name in (
            "sparse_attn_v41_paged_2src_fp8kv",
            "sparse_attn_v41_paged_2src_gather",
        ):
            impl = getattr(fp8_attention, name)
            monkeypatch.setattr(
                fp8_attention,
                name,
                lambda *a, impl=impl, name=name, **kw: (
                    taken.append(name) or impl(*a, **kw)
                ),
            )
        attend = fp8_attention.sparse_attn_v41_paged_2src_bf16
    elif path == "in_kernel":
        attend = fp8_attention.sparse_attn_v41_paged_2src_fp8kv
    elif path == "in_kernel_whole":
        attend = functools.partial(
            fp8_attention.sparse_attn_v41_paged_2src_fp8kv, num_splits=1
        )
    else:
        attend = fp8_attention.sparse_attn_v41_paged_2src_gather
    actual = attend(*args)
    if path.startswith("entry"):
        assert taken == [
            (
                "sparse_attn_v41_paged_2src_fp8kv"
                if path == "entry_split"
                else "sparse_attn_v41_paged_2src_gather"
            )
        ]
    expected = sparse_attn_v4_paged_2src_reference(
        query.float(),
        fp8_attention.dequantize_rows(*fp8_attention._split(pool, dim)),
        prefix,
        indptr(prefix_counts),
        fp8_attention.dequantize_rows(*fp8_attention._split(extend, dim)),
        second,
        indptr(extend_counts),
        sink,
        dim**-0.5,
    )
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("backend", ["opus", None])
def test_native_fp8_attention_asks_aiter_once_per_device_and_shape(
    monkeypatch, backend
):
    """AITER decides from the live device, not from the build's `GPU_ARCHS`;
    a device it does not serve, gfx942 or any other, takes the BF16 query."""
    with atom_without_aiter() as load:
        fp8_attention = load("atom.model_ops.attentions.deepseek_v41.fp8_attention")
    asked = []

    def ask(device, q_shape, **dtypes):
        asked.append((device, q_shape, dtypes))
        return backend

    monkeypatch.setattr(fp8_attention, "get_mla_sparse_prefill_v41_backend", ask)
    fp8_attention._native_fp8_attention.cache_clear()
    try:
        for rows in ((5, 8, 512), (1, 7, 8, 512)):
            query = torch.empty(rows, dtype=torch.bfloat16)
            assert fp8_attention.native_fp8_attention(query) is (backend is not None)
    finally:
        fp8_attention._native_fp8_attention.cache_clear()
    ((device, q_shape, dtypes),) = asked
    assert (device, q_shape) == (torch.device("cpu"), (1, 8, 512))
    assert dtypes["layout"] == "contiguous"
    assert dtypes["unified_kv_dtype"] == dtypes["q_dtype"] == torch.float8_e4m3fn
    assert dtypes["unified_kv_scale_dtype"] == torch.float8_e8m0fnu

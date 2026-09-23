# SPDX-License-Identifier: MIT
"""Native row bytes, QAT round trips and bounded paged index reads."""

from dataclasses import replace

import pytest
import torch

if not torch.cuda.is_available():
    # Ahead of the imports below, not after them: every one of them reaches
    # Triton, which a CPU runner does not have, and an import that raises
    # during collection takes the whole session down rather than one file.
    pytest.skip(
        "reads a packed pool through a Triton kernel; needs a real GPU",
        allow_module_level=True,
    )

import triton
import triton.language as tl

from atom.model_ops.attentions.deepseek_v41.packed_rows import (
    load_mixed_rows,
    pack_rows,
)
from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry
from atom.model_ops.blockscale import quantize_fp4, quantize_fp8


@triton.jit
def _read_mixed(pool, addresses, out, D: tl.constexpr):
    row = tl.program_id(0)
    ids = tl.load(addresses + row + tl.arange(0, 1))
    d = tl.arange(0, D)
    value = load_mixed_rows(pool, ids, d, tl.full((1,), True, tl.int1), D)
    tl.store(out + row * D + d[None, :], value)


def test_native_byte_accounting():
    geo = V41PoolGeometry(
        40,
        ((2, 2), (8, 2), (14, 2), (20, 1)),
        32,
        128,
        512,
        128,
        packed=True,
    )
    bf16 = replace(geo, packed=False)
    assert (geo.main_row_bytes, geo.index_row_bytes, geo.window_row_bytes) == (
        288,
        132,  # 128 B of data, one fp32 scale
        528,
    )
    # The index rows are a region apart, so only the main share of them is a
    # page field. `packed` is the main and window planes' alone -- the index
    # plane has one format, so it weighs the same on both sides here.
    assert sum(f.bytes_per_entry for f in geo.page_fields) == 32 * 720
    # 2.5 index rows per token -- three ratio-2 owners and one ratio-1.
    assert geo.paged_bytes == 32 * (720 + 2.5 * 132)
    assert geo.state_fields[0].bytes_per_entry == 40 * 128 * 528
    assert geo.page_bytes < bf16.page_bytes / 2
    assert geo.state_bytes < bf16.state_bytes * 0.52
    assert geo.layout_id != bf16.layout_id
    # A ratio-2 owner halves the PAGE, so the tile bound is a statement about
    # twice it: 16 tokens leave that owner 8 rows, which no block id names.
    with pytest.raises(ValueError, match="16-row tiles"):
        replace(geo, block_size=16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("magnitude", [0.0, 2**-20, 0.03, 1.0, 10.0])
def test_native_main_window_rows_restore_exact_qat(magnitude):
    torch.manual_seed(54)
    x = (torch.randn(13, 512, device="cuda") * magnitude).bfloat16()
    main = quantize_fp4(x, group_size=16, scale_dtype=torch.float8_e4m3fn)
    window = quantize_fp8(x)
    main_rows, window_rows = pack_rows(*main), pack_rows(*window)
    pool = torch.cat((main_rows.flatten(), window_rows.flatten()))
    addresses = torch.cat(
        (
            torch.arange(13, device="cuda") * 288 * 2,
            ((main_rows.numel() + torch.arange(13, device="cuda") * 528) * 2) | 1,
        )
    )
    actual = torch.empty((26, 512), dtype=torch.bfloat16, device="cuda")
    _read_mixed[(26,)](pool, addresses, actual, 512)
    expected = torch.cat(
        (
            quantize_fp4(
                x, group_size=16, scale_dtype=torch.float8_e4m3fn, dequantize=True
            ),
            quantize_fp8(x, dequantize=True),
        )
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("batch", [1, 4, 8])
def test_packed_decode_reuses_v4_with_ragged_rows(batch):
    from atom.model_ops.attentions.deepseek_v41.packed_attention import packed_decode
    from atom.model_ops.attentions.deepseek_v41.packed_rows import gather_prefix_rows
    from atom.model_ops.v4_kernels import sparse_attn_v4_paged_decode

    torch.manual_seed(71)
    main = torch.randn(1024, 512, device="cuda", dtype=torch.bfloat16) * 0.2
    window = torch.randn(batch * 128, 512, device="cuda", dtype=torch.bfloat16) * 0.2
    main_values = quantize_fp4(main, group_size=16, scale_dtype=torch.float8_e4m3fn)
    main_bytes = pack_rows(*main_values).flatten()
    window_bytes = pack_rows(*quantize_fp8(window)).flatten()
    pool = torch.cat((main_bytes, window_bytes))
    choices, lengths = [], []
    for i in range(batch):
        selected = torch.randperm(1024, device="cuda")[: 512 - i * 23]
        history = torch.arange(i * 128, (i + 1) * 128, device="cuda")
        choices.append(
            torch.cat(
                (selected * 288 * 2, ((main_bytes.numel() + history * 528) * 2) | 1)
            )
        )
        lengths.append(choices[-1].numel())
    indices = torch.full((batch * 640,), 2**45, dtype=torch.int64, device="cuda")
    selected = torch.cat(choices)
    indices[: selected.numel()] = selected
    ptr = torch.tensor([0, *lengths], device="cuda", dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )
    q = torch.randn(batch, 16, 512, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(16, device="cuda")
    bf16 = torch.cat(
        (
            quantize_fp4(
                main, group_size=16, scale_dtype=torch.float8_e4m3fn, dequantize=True
            ),
            quantize_fp8(window, dequantize=True),
        )
    )
    bf16_indices = torch.where(
        (indices & 1) != 0,
        1024 + ((indices >> 1) - main_bytes.numel()) // 528,
        (indices >> 1) // 288,
    ).to(torch.int32)
    expected = sparse_attn_v4_paged_decode(q, bf16, bf16_indices, ptr, sink, 512**-0.5)
    actual = packed_decode(q, pool.view(-1, 1), indices, ptr, sink, 512**-0.5)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Capacity padding contains invalid addresses; the gather must use indptr.
    scratch = torch.empty((indices.numel(), 512), device="cuda", dtype=torch.bfloat16)
    gather_prefix_rows(pool, indices, ptr, scratch, 0, batch)
    assert scratch[selected.numel() :].count_nonzero() == 0


@pytest.mark.parametrize("ratio", [1, 2])
def test_packed_plan_scatter_graph_replay_preserves_other_fields(ratio):
    from types import SimpleNamespace

    from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache

    geo = V41PoolGeometry(3, ((0, 2), (1, 1), (2, 2)), 32, 128, 512, 128, packed=True)
    cache = PagedAttentionCache(geo, 6, 2, "cuda")
    owner = 1 if ratio == 1 else 0
    pages = cache.pages.view(f"main_{owner}")[0]
    per_page = geo.rows_per_page(ratio)
    tables = torch.tensor([[3, 1], [5, 2]], device="cuda", dtype=torch.int32)
    plan = torch.full((4, 4), -1, device="cuda", dtype=torch.int32)
    step = SimpleNamespace(
        block_tables=tables,
        plans={ratio: SimpleNamespace(compress_plan_gpu=plan)},
    )
    x = torch.randn(1, 4, 512, device="cuda", dtype=torch.bfloat16)
    value = quantize_fp4(x, group_size=16, scale_dtype=torch.float8_e4m3fn)
    packed = pack_rows(*value)[0]
    # Compile and warm up before capture. All-sentinel capture must still
    # produce writes when live rows arrive later at the same plan address.
    cache._scatter_rows(pages, step, value, ratio)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cache._scatter_rows(pages, step, value, ratio)
    for rows in (
        [(0, 0), (1, per_page), (-1, -1), (0, per_page + 1)],
        [(-1, -1), (1, 3), (0, per_page - 1), (-1, -1)],
        [(-1, -1)] * 4,
    ):
        cache.page_bytes.fill_(97)
        before = cache.backing.clone()
        expected = cache.page_bytes.clone()
        plan_cpu = torch.full((4, 4), -1, dtype=torch.int32, device="cpu")
        for i, (batch, row) in enumerate(rows):
            plan_cpu[i, 1:3] = torch.tensor([batch, row * ratio])
            if batch >= 0:
                page = int(tables[batch, row // per_page])
                offset = pages.storage_offset() + (row % per_page) * pages.stride(1)
                expected[page, offset : offset + packed.shape[-1]] = packed[i]
        plan.copy_(plan_cpu)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(cache.page_bytes, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            cache.backing[cache.page_bytes.numel() :],
            before[cache.page_bytes.numel() :],
            rtol=0,
            atol=0,
        )

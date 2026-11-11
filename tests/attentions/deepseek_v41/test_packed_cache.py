# SPDX-License-Identifier: MIT
"""Native row bytes, QAT round trips and bounded paged index reads."""

from dataclasses import replace

import numpy as np
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
        kv_dtype="fp4",
    )
    bf16 = replace(geo, kv_dtype="bf16")
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
    # A ratio-2 owner halves the PAGE, so the block bound is a statement about
    # twice it: 16 tokens leave that owner 8 rows, which no block id names --
    # at this geometry's 16-row blocks. Declaring 8-row ones makes the same
    # PAGE legal, which is what says the gate is about the block and not the
    # PAGE.
    with pytest.raises(ValueError, match="16-row blocks"):
        replace(geo, block_size=16)
    replace(geo, block_size=16, index_block_rows=8)


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


def _compress_inputs(ratio, rows):
    """A compressor, its projections and a step whose `rows` plan slots (each
    `(request, compressed row)` or `(-1, -1)`) each pool `ratio` fresh rows."""
    from types import SimpleNamespace

    torch.manual_seed(ratio)
    compressor = SimpleNamespace(
        ratio=ratio,
        ape=torch.zeros(ratio, 512, device="cuda"),
        norm=SimpleNamespace(weight=torch.randn(512, device="cuda"), eps=1e-6),
    )
    values = torch.randn(1, len(rows) * ratio, 512, device="cuda") * 3
    scores = None if ratio == 1 else torch.randn_like(values)
    rope = SimpleNamespace(
        cos_cache=torch.randn(512, 32, device="cuda").to(torch.bfloat16),
        sin_cache=torch.randn(512, 32, device="cuda").to(torch.bfloat16),
        rope_dim=64,
    )
    plan = torch.full((len(rows), 4), -1, device="cuda", dtype=torch.int32)
    step = SimpleNamespace(
        block_tables=torch.tensor([[3, 1], [5, 2]], device="cuda", dtype=torch.int32),
        slots=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        max_q_len=ratio,
        plans={
            ratio: SimpleNamespace(
                compress_plan_gpu=plan,
                write_plan_gpu=torch.full_like(plan, -1),
                num_compress=len(rows),
            )
        },
    )
    return compressor, values, scores, rope, step


def _plan_rows(rows, ratio):
    """`rows` as plan rows: slot i a boundary pooling ragged rows i ratio ..
    + ratio - 1, all from this forward's projections."""
    plan = torch.full((len(rows), 4), -1, dtype=torch.int32)
    for i, (batch, row) in enumerate(rows):
        if batch >= 0:
            plan[i] = torch.tensor(
                [(i + 1) * ratio - 1, batch, (row + 1) * ratio - 1, 0]
            )
    return plan


@pytest.mark.parametrize("kv_dtype", ["fp4", "fp8"])
@pytest.mark.parametrize("ratio", [1, 2])
def test_compressed_rows_land_packed_under_graph_replay(kv_dtype, ratio):
    """`compress` writes a quantized pool's main rows in the kernel itself:
    each the BF16 pool's row for the same inputs, quantized by the pool's rule
    (`quantize_fp4` group16 / `quantize_fp8`) and packed, byte for byte; every
    other byte of the backing as it was. Captured with sentinel rows only, then
    replayed with live ones at the same plan address."""
    from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache

    owners = ((0, 2), (1, 1), (2, 2))
    cache = PagedAttentionCache(
        V41PoolGeometry(3, owners, 32, 128, 512, 128, kv_dtype=kv_dtype), 6, 2, "cuda"
    )
    reference = PagedAttentionCache(
        V41PoolGeometry(3, owners, 32, 128, 512, 128), 6, 2, "cuda"
    )
    owner = 1 if ratio == 1 else 0
    per_page = cache.geometry.rows_per_page(ratio)
    pages = cache.pages.view(f"main_{owner}")[0]
    compressor, values, scores, rope, step = _compress_inputs(ratio, [(-1, -1)] * 4)
    plan = step.plans[ratio].compress_plan_gpu

    def run(target):
        target.compress(owner, compressor, values, scores, step, rope)

    run(cache)  # compile and warm up before the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(cache)
    for rows in (
        [(0, 0), (1, per_page), (-1, -1), (0, per_page + 1)],
        [(-1, -1), (1, 3), (0, per_page - 1), (-1, -1)],
        [(-1, -1)] * 4,
    ):
        plan.copy_(_plan_rows(rows, ratio))
        run(reference)
        ref_pages = reference.pages.view(f"main_{owner}")[0]
        cache.page_bytes.fill_(97)
        before = cache.backing.clone()
        expected = cache.page_bytes.clone()
        for batch, row in rows:
            if batch < 0:
                continue
            page = int(step.block_tables[batch, row // per_page])
            latent = ref_pages[page, row % per_page][None]
            packed = pack_rows(
                *(
                    quantize_fp4(latent, group_size=16, scale_dtype=torch.float8_e4m3fn)
                    if kv_dtype == "fp4"
                    else quantize_fp8(latent)
                )
            )[0]
            offset = pages.storage_offset() + (row % per_page) * pages.stride(1)
            expected[page, offset : offset + packed.numel()] = packed
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(cache.page_bytes, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            cache.backing[cache.page_bytes.numel() :],
            before[cache.page_bytes.numel() :],
            rtol=0,
            atol=0,
        )


def test_packed_window_keeps_rows_below_cached_until():
    """The packed ring write leaves a request's rows below `swa_cached_until`
    as they are (V4.1 decoder bounded replay recomputes rows its ring holds),
    and writes the rest exactly as without it."""
    from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
    from atom.model_ops.blockscale import quantize_fp8
    from tests.attentions.deepseek_v41.helpers import PagedRequest, begin_step

    geo = V41PoolGeometry(
        3, ((0, 2), (1, 1), (2, 2)), 32, 128, 512, 128, kv_dtype="fp4"
    )
    requests = (
        PagedRequest(1, 32, 0, 40, 0, (0, 1, 2)),
        PagedRequest(2, 0, 40, 10, 1, (3,)),
    )
    x = torch.randn(50, 512, device="cuda", dtype=torch.bfloat16)
    rows = quantize_fp8(x)
    caches = []
    for cached in (None, [48, 0]):
        cache = PagedAttentionCache(geo, 8, 2, "cuda")
        cache.backing.zero_()
        step = begin_step(
            cache,
            requests,
            floors=None if cached is None else np.asarray([32, 0], np.int32),
            cached_until=None if cached is None else np.asarray(cached, np.int32),
        )
        cache.write_window(1, rows, step)
        caches.append(cache.backing.clone())
    torch.cuda.synchronize()
    whole, kept = caches
    window = geo.window(1, 8)
    row_bytes = 512 + 512 // 32
    expected = whole.clone()
    for pos in range(32, 48):
        start = window.ring_start + 0 * window.slot_rows + (pos % 128) * row_bytes
        expected[start : start + row_bytes] = 0
    assert not torch.equal(whole, expected)
    torch.testing.assert_close(kept, expected, rtol=0, atol=0)


@pytest.mark.parametrize("ratio", [1, 2])
def test_ring_update_in_the_compress_launch_matches_its_own_launch(ratio, monkeypatch):
    """A decode-sized step folds the ring update into the boundary kernel's
    launch; the backing it leaves -- main rows, ring, everything -- is the one
    the two launches leave, with boundaries reading the ring (ratio 2: each
    request's first row closes a group the previous forward opened)."""
    from types import SimpleNamespace

    import atom.model_ops.deepseek_v41.compressor as compressor_module
    from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache

    owners = ((0, 2), (1, 1), (2, 2))
    # DSpark's 5 drafts: the ring the server runs (7 rows at ratio 2)
    geo = V41PoolGeometry(
        3, owners, 32, 128, 512, 128, kv_dtype="fp8", speculative_tokens=5
    )
    caches = [PagedAttentionCache(geo, 6, 2, "cuda") for _ in range(2)]
    owner = 1 if ratio == 1 else 0
    torch.manual_seed(7)
    caches[0].page_bytes.copy_(
        torch.randint(0, 255, caches[0].page_bytes.shape, dtype=torch.uint8)
    )
    for state in caches[0].compress_state(owner):
        state.copy_(torch.randn_like(state))
    caches[1].backing.copy_(caches[0].backing)
    compressor, values, scores, rope, _ = _compress_inputs(ratio, [(-1, -1)] * 4)
    # Two requests, two rows each, at positions (5, 6) and (9, 10)
    rows = [(0, 5), (0, 6), (1, 9), (1, 10)]
    boundaries = [
        (i, batch, position, ratio - 1 - min(i % 2, ratio - 1))
        for i, (batch, position) in enumerate(rows)
        if (position + 1) % ratio == 0
    ]
    plan = torch.full((4, 4), -1, dtype=torch.int32)
    plan[: len(boundaries)] = torch.tensor(boundaries, dtype=torch.int32)
    write = torch.full((4, 4), -1, dtype=torch.int32)
    write[:] = torch.tensor([(i, b, p, 0) for i, (b, p) in enumerate(rows)])

    def step(max_q_len):
        return SimpleNamespace(
            block_tables=torch.tensor(
                [[3, 1], [5, 2]], device="cuda", dtype=torch.int32
            ),
            slots=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
            max_q_len=max_q_len,
            plans={
                ratio: SimpleNamespace(
                    compress_plan_gpu=plan.cuda(),
                    write_plan_gpu=write.cuda(),
                    num_compress=len(boundaries),
                )
            },
        )

    caches[1].compress(owner, compressor, values, scores, step(64), rope)

    def separate_launch(*args, **kwargs):
        raise AssertionError("a decode step's ring update ran as its own launch")

    monkeypatch.setattr(compressor_module, "update_compressor_states", separate_launch)
    caches[0].compress(owner, compressor, values, scores, step(2), rope)
    torch.cuda.synchronize()
    torch.testing.assert_close(caches[0].backing, caches[1].backing, rtol=0, atol=0)

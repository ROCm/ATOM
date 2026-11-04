# SPDX-License-Identifier: MIT
"""The NoPE index writer, the BF16 output gate and the direct prefill slot
conversion against the separate operators and CPU FP32 arithmetic."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires a ROCm GPU"
)


@pytest.mark.parametrize(
    "rows,page,stride", [(0, 16, 132), (1, 16, 144), (17, 16, 132), (129, 128, 144)]
)
def test_nope_writer_matches_separate_ops_and_preserves_unowned_slots(
    rows, page, stride
):
    from aiter import QuantType, dtypes, get_hip_quant, indexer_k_quant_and_cache

    from atom.model_ops.indexer.nope import indexer_nope_quant_and_cache
    from atom.model_ops.layernorm import LayerNorm

    generator = torch.Generator(device="cuda").manual_seed(3147)
    # A fused projection's output: other columns, the keys, the head weights.
    packed = torch.randn(
        rows, 1024 + 128 + 32, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    k, weights = packed[:, 1024:1152], packed[:, 1152:]
    q = torch.randn(
        rows, 33, 128, generator=generator, device="cuda", dtype=torch.bfloat16
    )[:, :32]
    if rows:
        q[0, 0].zero_()
        q[0, 1].fill_(1e-14)
        k[0].fill_(2.0)  # zero variance, exercising eps and the K scale floor
    norm = LayerNorm(128, eps=1e-6, dtype=torch.float32).cuda()
    norm.weight.data.copy_(torch.randn(128, generator=generator, device="cuda") / 7 + 1)
    norm.bias.data.copy_(torch.randn(128, generator=generator, device="cuda") / 31)
    if rows == 1:
        norm.bias.data.zero_()  # constant K now exercises the 1e-4 scale floor
    pages = max(3, (rows + page - 1) // page + 2)
    slots = torch.randperm(pages * page, generator=generator, device="cuda")[:rows]
    if rows > 1:
        slots[1::3] = -1  # DCP non-owners still need every Q/head-weight output
        slots = slots[:-2]  # graph padding can also have no slot entry at all
    cache = torch.full((pages, page, stride), 0x5A, dtype=torch.uint8, device="cuda")
    expected_cache = cache.clone()
    q8, scaled_weights = indexer_nope_quant_and_cache(
        q, k, weights, cache, slots, norm.weight, norm.bias, norm.eps, 1 / 64
    )
    assert q8.shape == q.shape and q8.dtype == dtypes.fp8
    assert (
        scaled_weights.shape == weights.shape and scaled_weights.dtype == torch.float32
    )
    if not rows:
        assert torch.equal(cache, expected_cache)
        return
    q_ref, scale = get_hip_quant(QuantType.per_1x128)(
        q.reshape(-1, 128), quant_dtype=dtypes.fp8
    )
    expected_weights = weights.float() * scale.reshape(rows, 32) / 64
    indexer_k_quant_and_cache(
        norm(k.contiguous())[: slots.numel()],
        expected_cache,
        slots,
        128,
        "ue8m0",
        preshuffle=True,
    )
    assert torch.equal(q8.view(torch.uint8), q_ref.reshape_as(q8).view(torch.uint8))
    torch.testing.assert_close(scaled_weights, expected_weights, rtol=2e-6, atol=1e-12)
    assert torch.equal(cache, expected_cache)


@pytest.mark.parametrize("layout", ["contiguous", "row_strided", "alias", "empty"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_output_gate_matches_fp32_without_mutating_inputs(layout, dtype):
    from atom.model_ops.triton_fused_sigmoid_mul_quant import (
        fused_sigmoid_mul_maybe_quant,
    )

    generator = torch.Generator(device="cuda").manual_seed(114)
    rows, width = (0 if layout == "empty" else 17), 1536
    storage = torch.randn(
        rows, width + 32, generator=generator, device="cuda", dtype=dtype
    )
    x = storage[:, :width]
    gate = torch.randn(
        rows, width + 64, generator=generator, device="cuda", dtype=dtype
    )[:, :width]
    if layout != "row_strided":
        x, gate = x.contiguous(), gate.contiguous()
    if layout == "alias":
        gate = x
    before_x, before_gate = x.clone(), gate.clone()
    out, scale = fused_sigmoid_mul_maybe_quant(x, gate, quant=False)
    assert out.shape == x.shape and out.dtype == x.dtype and scale is None
    assert torch.equal(x, before_x) and torch.equal(gate, before_gate)
    if rows:
        assert out.data_ptr() != x.data_ptr() and out.data_ptr() != gate.data_ptr()
        reference = before_x.cpu().float() * torch.sigmoid(before_gate.cpu().float())
        # BF16/FP16 keep their original output-rounding budget. For FP32,
        # both CPU and GPU sigmoid/multiply paths round independently; allow
        # one epsilon from each instead of treating the CPU result as exact.
        tolerance = max(torch.finfo(dtype).eps, 2 * torch.finfo(torch.float32).eps)
        torch.testing.assert_close(
            out.cpu().float(), reference, rtol=tolerance, atol=2e-7
        )


def test_direct_prefill_slots_match_separate_conversion_with_request_bases():
    from aiter import QuantType, dtypes, get_hip_quant, indexer_k_quant_and_cache

    from atom.model_ops.attention_mla import (
        triton_convert_req_index_to_global_index_dsa_prefill,
    )
    from atom.model_ops.indexer import IndexCache, concat_rows, score_topk
    from atom.model_ops.indexer.rows import ConcatKeys

    generator = torch.Generator(device="cuda").manual_seed(8701)
    lengths, ends = [17, 2100, 63], [1, 17, 1999, 2100, 1, 63]
    rows, page, topk = 6, 128, 2048
    blocks = (max(lengths) + page - 1) // page
    table = (
        torch.randperm(3 * blocks, generator=generator, device="cuda")
        .to(torch.int32)
        .reshape(3, blocks)
    )
    cache = torch.zeros(3 * blocks, page, 144, dtype=torch.uint8, device="cuda")
    for req, length in enumerate(lengths):
        keys = torch.randn(
            length, 128, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        pos = torch.arange(length, device="cuda")
        slots = table[req, pos // page].long() * page + pos % page
        indexer_k_quant_and_cache(keys, cache, slots, 128, "ue8m0", preshuffle=True)
    query = torch.randn(
        rows, 32, 128, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    q8, scales = get_hip_quant(QuantType.per_1x128)(
        query.reshape(-1, 128), quant_dtype=dtypes.fp8
    )
    q8 = q8.reshape_as(query)
    weights = (
        torch.randn(rows, 32, generator=generator, device="cuda")
        * scales.reshape(rows, 32)
        / 64
    )
    ends = torch.tensor(ends, dtype=torch.int32, device="cuda")
    batch_ids = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, 17, 2117, 2180], dtype=torch.int32, device="cuda")
    # Sparse MLA treats each query token as its own CSR row. Request-level
    # [0, 2, 4, 6] would truncate the conversion grid to only three tokens.
    cu_q = torch.arange(rows + 1, dtype=torch.int32, device="cuda")
    indptr = torch.cat([ends.new_zeros(1), ends.clamp_max(topk).cumsum(0).int()])
    index_rows = concat_rows(
        ends, ConcatKeys(table, cu_k, 2180, cu_k[batch_ids.long()])
    )
    index_cache = IndexCache(cache, None, page)
    selected = score_topk(q8, weights, index_cache, index_rows, topk=topk, stable=True)
    separate = torch.full((rows * topk,), -77, dtype=torch.int32, device="cuda")
    triton_convert_req_index_to_global_index_dsa_prefill(
        cu_q,
        indptr,
        batch_ids,
        selected,
        table,
        cu_k,
        PAGE_SIZE=page,
        NUM_TOPK_TOKENS=topk,
        out=separate,
    )
    # Gathered-run ids, the request base subtracted inside the conversion.
    gathered = score_topk(
        q8,
        weights,
        index_cache,
        index_rows,
        topk=topk,
        stable=True,
        request_local=False,
    )
    fused = torch.full_like(separate, -77)
    triton_convert_req_index_to_global_index_dsa_prefill(
        cu_q,
        indptr,
        batch_ids,
        gathered,
        table,
        cu_k,
        PAGE_SIZE=page,
        NUM_TOPK_TOKENS=topk,
        out=fused,
        row_starts=index_rows.concat.row_starts,
    )
    assert torch.equal(fused, separate)
    for row, end in enumerate(ends.tolist()):
        count = min(end, topk)
        local = selected[row, :count].long()
        expected = table[batch_ids[row], local // page].long() * page + local % page
        assert torch.equal(
            fused[int(indptr[row]) : int(indptr[row + 1])].long(), expected
        )


def test_nope_and_bf16_gate_graph_replay_reads_new_values_and_slots():
    from aiter import QuantType, dtypes, get_hip_quant, indexer_k_quant_and_cache

    from atom.model_ops.indexer.nope import indexer_nope_quant_and_cache
    from atom.model_ops.layernorm import LayerNorm
    from atom.model_ops.triton_fused_sigmoid_mul_quant import (
        fused_sigmoid_mul_maybe_quant,
    )

    generator = torch.Generator(device="cuda").manual_seed(510)
    q = torch.randn(
        17, 32, 128, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    packed = torch.randn(
        17, 1024 + 128 + 32, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    k, weights = packed[:, 1024:1152], packed[:, 1152:]
    gate = torch.randn(
        17, 1024, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    norm = LayerNorm(128, dtype=torch.float32).cuda()
    slots = torch.arange(17, device="cuda")
    cache = torch.full((3, 16, 144), 0x5A, dtype=torch.uint8, device="cuda")

    def forward():
        return (
            indexer_nope_quant_and_cache(
                q, k, weights, cache, slots, norm.weight, norm.bias, norm.eps, 1 / 64
            ),
            fused_sigmoid_mul_maybe_quant(packed[:, :1024], gate, quant=False)[0],
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            forward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        (q8, scaled), output = forward()
    for shift in (3, 19):
        q.normal_(generator=generator)
        packed.normal_(generator=generator)
        gate.normal_(generator=generator)
        slots.copy_((torch.arange(17, device="cuda") + shift) % 48)
        slots[::3] = -1
        cache.fill_(0x5A)
        expected_cache = cache.clone()
        graph.replay()
        q_ref, scales = get_hip_quant(QuantType.per_1x128)(
            q.reshape(-1, 128), quant_dtype=dtypes.fp8
        )
        indexer_k_quant_and_cache(
            norm(k.contiguous()), expected_cache, slots, 128, "ue8m0", preshuffle=True
        )
        assert torch.equal(q8.view(torch.uint8), q_ref.reshape_as(q8).view(torch.uint8))
        assert torch.equal(cache, expected_cache)
        torch.testing.assert_close(
            scaled, weights.float() * scales.reshape(17, 32) / 64, rtol=2e-6, atol=1e-12
        )
        expected_gate = packed[:, :1024].cpu().float() * gate.cpu().float().sigmoid()
        torch.testing.assert_close(
            output.cpu().float(),
            expected_gate,
            rtol=torch.finfo(torch.bfloat16).eps,
            atol=2e-7,
        )


def test_shared_nope_dispatch_writes_real_fp8_pool_bytes(monkeypatch):
    """The bound pool is FP8; the fused writer consumes a uint8 storage view."""
    import numpy as np
    from aiter import dtypes, indexer_k_quant_and_cache

    import atom.config as config_module
    from atom.model_ops.attentions.mla_kv_pool import MlaKvPool
    from atom.model_ops.indexer import indexer_forward
    from atom.model_ops.layernorm import LayerNorm
    from atom.utils.forward_context import (
        AttentionMetaData,
        Context,
        reset_forward_context,
        set_forward_context,
    )

    page, rows, dim = 128, 17, 144
    pool = MlaKvPool(
        layers=1,
        block_size=page,
        entry_dim=576,
        kv_dtype=torch.bfloat16,
        index_layers=1,
        index_rows_per_block=page,
        index_dim=dim,
        index_dtype=dtypes.fp8,
    )
    pool.allocate(3, "cuda")
    cache = pool.layer("index", 0)
    cache.view(torch.uint8).fill_(0x5A)
    expected = cache.view(torch.uint8).clone()
    generator = torch.Generator(device="cuda").manual_seed(741)
    q = torch.randn(
        rows, 32, 128, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    k = torch.randn(rows, 128, dtype=torch.bfloat16, device="cuda", generator=generator)
    weights = torch.randn(
        rows, 32, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    norm = LayerNorm(128, eps=1e-6, dtype=torch.float32).cuda()
    slots = torch.arange(page + 5, page + 5 + rows, dtype=torch.int64, device="cuda")
    slots[::3] = -1
    positions = torch.arange(rows, device="cuda")
    indexer = SimpleNamespace(_indexer_fp4=False, use_nope_cache_fusion=True)
    cfg = SimpleNamespace(
        # A prefill within top-k returns right after the cache write.
        hf_config=SimpleNamespace(),
        kv_cache_block_size=page,
        dcp_config=SimpleNamespace(interleave_size=1),
        parallel_config=SimpleNamespace(data_parallel_size=1),
        compilation_config=SimpleNamespace(
            static_forward_context={"test.indexer": indexer}
        ),
    )
    monkeypatch.setattr(config_module, "_current_atom_config", cfg)
    monkeypatch.setattr(indexer_forward, "get_dcp_world_size", lambda: 1)
    monkeypatch.setattr(indexer_forward, "pcp_is_enabled", lambda: False)
    md = AttentionMetaData(slot_mapping=slots, max_seqlen_k=rows, max_seqlen_q=rows)
    ctx = Context(
        positions,
        is_prefill=True,
        scheduled_bs=1,
        scheduled_tokens=rows,
        running_bs=1,
        running_tokens=rows,
    )
    empty = torch.empty(0, dtype=torch.int32, device="cuda")
    try:
        set_forward_context(md, cfg, ctx, np.array([rows]))
        indexer_forward.sparse_attn_indexer(
            k,
            "test.indexer.k_cache",
            cache.view(-1, 1, dim),
            q,
            k,
            weights,
            128,
            "ue8m0",
            2048,
            128,
            4096,
            rows,
            empty,
            empty,
            empty,
            norm.weight,
            norm.bias,
            norm.eps,
            positions,
            None,
            None,
            1 / 64,
            False,
            False,
            True,
        )
    finally:
        reset_forward_context()
    indexer_k_quant_and_cache(norm(k), expected, slots, 128, "ue8m0", preshuffle=True)
    assert cache.dtype == dtypes.fp8
    assert torch.equal(cache.view(torch.uint8), expected)
    assert torch.count_nonzero(pool.layer("kv", 0)) == 0

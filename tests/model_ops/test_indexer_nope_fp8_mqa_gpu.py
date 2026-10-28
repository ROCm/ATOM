# SPDX-License-Identifier: MIT
"""NoPE FP8 index writes and native MQA vs independently dequantized scores."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires a ROCm GPU"
)


@pytest.mark.parametrize("length", [2047, 2048, 2049, 4096])
def test_fp8_mqa_paged_and_cached_prefill_agree_with_dequantized_reference(length):
    from aiter import QuantType, dtypes, get_hip_quant, indexer_k_quant_and_cache
    from aiter.ops.triton.attention.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits

    from atom.model_ops.indexer import IndexCache, concat_rows, fp8_rows, score_topk
    from atom.model_ops.indexer.rows import ConcatKeys

    gen = torch.Generator().manual_seed(5931)
    block, heads, dim, topk = 128, 32, 128, 2048
    pages = (length + block - 1) // block
    keys = torch.randn(length, dim, generator=gen).bfloat16().cuda()
    q = torch.randn(4, heads, dim, generator=gen).bfloat16().cuda()
    w = torch.randn(4, heads, generator=gen).bfloat16().cuda()
    ends = torch.tensor([1, 127, length - 1, length], device="cuda", dtype=torch.int32)
    table = torch.randperm(pages, generator=gen).int()[None, :].cuda()
    tokens = torch.arange(length, device="cuda")
    slots = (table[0, tokens // block].long() * block + tokens % block).long()
    cache = torch.zeros(pages, block, 144, dtype=torch.uint8, device="cuda")
    # The two writes model crossing into cached/chunked prefill.
    for start, end in ((0, 127), (127, length)):
        indexer_k_quant_and_cache(
            keys[start:end], cache, slots[start:end], dim, "ue8m0", preshuffle=True
        )
    before = cache.clone()
    indexer_k_quant_and_cache(
        keys[:1],
        cache,
        torch.tensor([-1], device="cuda"),
        dim,
        "ue8m0",
        preshuffle=True,
    )
    assert torch.equal(cache, before)

    # Invert the documented 16-token × 16-column tile layout, without using
    # the production gather or logits kernels as the oracle.
    flat = cache.view(pages, -1)
    data = flat[:, : block * dim].reshape(pages, block // 16, dim // 16, 16, 16)
    data = (
        data.permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(dtypes.fp8)
        .float()
        .reshape(pages, block, dim)
    )
    scale = (
        flat[:, block * dim : block * dim + block * 4]
        .contiguous()
        .view(torch.float32)
        .reshape(pages, block, 1)
    )
    decoded = (data * scale)[table[0, tokens // block].long(), tokens % block]
    expected_scale = torch.pow(
        2.0,
        torch.ceil(
            torch.log2(
                keys.float().abs().amax(-1, keepdim=True).clamp_min(1e-4)
                / torch.finfo(dtypes.fp8).max
            )
        ),
    )
    expected_keys = (keys.float() / expected_scale).to(
        dtypes.fp8
    ).float() * expected_scale
    torch.testing.assert_close(decoded, expected_keys, rtol=0, atol=0)

    q8, qs = get_hip_quant(QuantType.per_1x128)(
        q.reshape(-1, dim), quant_dtype=dtypes.fp8
    )
    q8 = q8.view_as(q)
    weights = w.float() * qs.view(4, heads) / 64
    scores = torch.full((4, pages * block), -torch.inf, device="cuda")
    tables = table.expand(4, -1).contiguous()
    deepgemm_fp8_paged_mqa_logits(
        q8[:, None],
        cache[:, :, None],
        weights,
        scores,
        ends,
        tables,
        pages * block,
        KVBlockSize=block,
        Preshuffle=True,
    )
    reference = torch.einsum(
        "rhd,kd->rhk", q8.cpu().float(), decoded.cpu().float()
    ).relu()
    reference = (reference * weights.cpu().float()[:, :, None]).sum(1)
    # Bound error between native and CPU FP32 reductions. Signed head weights
    # can cancel, so a relative tolerance on the final score alone is invalid.
    unit_roundoff = torch.finfo(torch.float32).eps / 2
    operations = dim + heads + 3
    gamma = operations * unit_roundoff / (1 - operations * unit_roundoff)
    absolute_terms = torch.einsum(
        "rhd,kd->rhk", q8.cpu().float().abs(), decoded.cpu().float().abs()
    )
    bound = gamma * (absolute_terms * weights.cpu().float().abs()[:, :, None]).sum(1)
    for row, end in enumerate(ends.tolist()):
        error = (scores[row, :end].cpu().float() - reference[row, :end]).abs()
        assert (error <= bound[row, :end]).all()
        assert error.norm() / reference[row, :end].norm() < 1e-4

    paged = fp8_rows(ends, pages * block, None, tables)
    concat = concat_rows(
        ends,
        ConcatKeys(
            table,
            torch.tensor([0, length], dtype=torch.int32, device="cuda"),
            length,
            torch.zeros(4, device="cuda", dtype=torch.int32),
        ),
    )
    for rows in (paged, concat):
        actual = score_topk(
            q8, weights, IndexCache(cache, None, block), rows, topk=topk, stable=True
        ).cpu()
        for row, end in enumerate(ends.tolist()):
            expected = (
                torch.argsort(reference[row, :end], descending=True, stable=True)[:topk]
                .cpu()
                .int()
            )
            selected = actual[row][actual[row] >= 0]
            assert torch.equal(selected.sort().values, expected.sort().values)
            assert (actual[row, len(expected) :] == -1).all()

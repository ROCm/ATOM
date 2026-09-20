# SPDX-License-Identifier: MIT
"""Unmodified V4 BF16 kernels with V4.1's dimensions and sparse inputs."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter", reason="the V4 kernels import the AITER runtime")

from atom.model_ops.v4_kernels import (
    sparse_attn_v4_paged_decode,
    sparse_attn_v4_paged_prefill,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ROCm GPU required"
)


@pytest.mark.parametrize("length", [1, 2])
def test_v4_bf16_counts_sink_once_for_swa_and_global(small_config, length):
    """A row visible to both the window and the selection is attended once.

    With a zero query every score is equal, so the output is the mean of the
    attended values against one sink: 2 rows give 6*2/3 and 3 give 6*3/4. A
    row double-counted would move it.
    """
    from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
    from atom.model_ops.attentions.deepseek_v41.metadata import RequestSpan
    from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry

    config = small_config
    config.head_dim = 512
    geo = V41PoolGeometry(
        1,
        ((0, 1),),
        32,
        config.sliding_window,
        512,
        32,
        layer_ratios=(1,),
        index_topk=1,
    )
    cache = PagedAttentionCache(geo, 8, 2, "cuda")
    cache.pages.view("main_0").fill_(6)
    spans = (
        RequestSpan(1, 0, 0, length, 0, (0, 1)),
        RequestSpan(2, 0, length, length, 1, (2, 3)),
    )
    step = cache.begin_step(spans)
    spec = SimpleNamespace(layer_id=0, ratio=1, kv_owner=0, topk_owner=0)
    # Index row 0 for every query row: the same row its window already holds.
    step.selected[0] = torch.zeros(1, step.width, 1, device="cuda", dtype=torch.int32)
    q = torch.zeros(2 * length, 8, 512, dtype=torch.bfloat16, device="cuda")
    kv = torch.full((1, 2 * length, 512), 6.0, dtype=torch.bfloat16, device="cuda")
    sink = torch.zeros(8, device="cuda")
    prefix, pptr, extend, eptr = cache.attention_indices(spec, step)
    if length == 1:
        cache.write_window(spec.layer_id, kv, step)
        output = sparse_attn_v4_paged_decode(
            q, cache.pool, prefix, pptr, sink, 512**-0.5
        )
    else:
        output = sparse_attn_v4_paged_prefill(
            q, cache.pool, prefix, pptr, kv.flatten(0, 1), extend, eptr, sink, 512**-0.5
        )
    expected = torch.tensor(
        [4.0] if length == 1 else [4.0, 4.5], device="cuda", dtype=torch.bfloat16
    )
    expected = expected.repeat(2)[:, None, None].expand_as(output)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("length", [1, 3])
@pytest.mark.parametrize("count", [192, 640])
def test_v4_bf16_sparse_attention_matches_independent_oracle(length, count):
    from .oracle_kernels import sparse_attn as oracle

    torch.manual_seed(512)
    batch, heads, dim = 2, 8, 512
    q = torch.randn(batch, length, heads, dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(batch, count, dim, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(heads, device="cuda", dtype=torch.float32)
    selected = (
        torch.arange(count, device="cuda", dtype=torch.int32)
        .view(1, 1, -1)
        .expand(batch, length, -1)
        .contiguous()
    )
    expected = oracle(q, kv, sink, selected, dim**-0.5)
    bids = torch.arange(batch, device="cuda", dtype=torch.int32).repeat_interleave(
        length
    )
    prefix_count = count if length == 1 else count // 2
    prefix = (
        bids[:, None] * count
        + torch.arange(prefix_count, device="cuda", dtype=torch.int32)
    ).flatten()
    pptr = (
        torch.arange(batch * length + 1, device="cuda", dtype=torch.int32)
        * prefix_count
    )
    if length == 1:
        actual = sparse_attn_v4_paged_decode(
            q.flatten(0, 1), kv.flatten(0, 1), prefix, pptr, sink, dim**-0.5
        )
    else:
        extend = (
            bids[:, None] * count
            + torch.arange(prefix_count, count, device="cuda", dtype=torch.int32)
        ).flatten()
        eptr = torch.arange(batch * length + 1, device="cuda", dtype=torch.int32) * (
            count - prefix_count
        )
        actual = sparse_attn_v4_paged_prefill(
            q.flatten(0, 1),
            kv.flatten(0, 1),
            prefix,
            pptr,
            kv.flatten(0, 1),
            extend,
            eptr,
            sink,
            dim**-0.5,
        )
    actual = actual.view_as(q)
    torch.testing.assert_close(actual, expected, rtol=1 / 64, atol=2**-10)
    assert (actual.float() - expected.float()).norm() / expected.float().norm() < 3e-3

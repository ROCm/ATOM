# SPDX-License-Identifier: MIT
"""Paging, sparse causality and exact state recovery across request lifetimes."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from atom.model_engine.page_unit_checkpoint import (
    CheckpointRestoreOp,
    CheckpointStoreOp,
    PagedStateCheckpointSpec,
)
from atom.model_ops.attentions.deepseek_v41.cache import (
    PagedAttentionCache,
    PagedIndexKeys,
)
from atom.model_ops.attentions.deepseek_v41.checkpoints import StateCopies
from atom.model_ops.attentions.deepseek_v41.metadata import RequestSpan
from atom.model_ops.attentions.deepseek_v41_state import EagerAttentionCache
from atom.model_ops.deepseek_v41.indexer import select_indices
from atom.model_ops.v4_kernels import (
    sparse_attn_v4_paged_decode,
    sparse_attn_v4_paged_prefill,
)
from atom.models.deepseek_v41.config import build_attention_topology
from tests.attentions.deepseek_v41.helpers import geometry


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_checkpoint_fork_rollback_relocation_and_slot_reuse(
    small_config, device, packed
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("ROCm GPU required")
    geo = replace(geometry(small_config, block=2), packed=packed)
    cache = PagedAttentionCache(geo, 40, 4, device)
    assert cache.backing.numel() == 40 * geo.page_bytes + 4 * geo.state_bytes
    spec = PagedStateCheckpointSpec(
        geo.page_bytes, geo.state_bytes, geo.layout_id, geo.state_bytes
    )
    assert spec.units_per_checkpoint > 1
    copies = StateCopies(cache, spec, 4)
    copies.warmup()
    # Fill ALL bytes, including padding and FP32 compressor tails. The cursor
    # remains interpretable while the rest proves byte-exact recovery.
    cache.state_bytes[2].copy_(
        torch.randint(
            0, 256, cache.state_bytes[2].shape, dtype=torch.uint8, device=device
        )
    )
    cache.cursor[2] = torch.tensor([3, 19, -1, 27], device=device)
    original = copies.entry(2).clone()
    units = tuple(range(1, 2 * spec.units_per_checkpoint, 2))
    store = CheckpointStoreOp(2, units, spec.image_bytes, spec.layout_id)
    restore = CheckpointRestoreOp(1, units, spec.image_bytes, spec.layout_id)
    copies.execute([store], [restore])
    torch.testing.assert_close(copies.entry(1), original, rtol=0, atol=0)
    cache.state_bytes[1].fill_(93)  # rejected / cancelled tentative suffix
    copies.execute([], [restore])
    torch.testing.assert_close(copies.entry(1), original, rtol=0, atol=0)
    cache.state_bytes[0].fill_(71)
    old_zero = copies.entry(0).clone()
    copies.relocate([(1, 0), (0, 1)])
    torch.testing.assert_close(copies.entry(0), original, rtol=0, atol=0)
    torch.testing.assert_close(copies.entry(1), old_zero, rtol=0, atol=0)
    span = RequestSpan(27, 3, 0, 1, 0, (32, 33))
    step = cache.begin_step([span])
    np.testing.assert_array_equal(cache.prepare_state(step), [[19, -1, 27]])
    # Recycled slot begins at zero and drops every old state field.
    fresh = cache.begin_step([replace(span, request_id=28, position=0)])
    np.testing.assert_array_equal(cache.prepare_state(fresh), [[-1, -1, -1]])
    assert cache.state.view("window")[:, 0].count_nonzero() == 0
    assert cache.state.view("tail_values")[:, 0].count_nonzero() == 0
    assert cache.cursor[0, 0] == 0
    with pytest.raises(ValueError, match="recoverable boundary"):
        cache.prepare_state(step)
    for bad in (
        replace(restore, layout_id="wrong"),
        replace(restore, total_bytes=3),
        replace(restore, unit_ids=(0,) * len(units)),
    ):
        with pytest.raises(ValueError):
            copies.execute([], [bad])
    with pytest.raises(IndexError):
        copies.entry(-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("tie", ["small_position", "large_position"])
def test_paged_index_tiles_and_candidates_equal_contiguous(tie):
    torch.manual_seed(13)
    blocks = torch.tensor([7, 2, 5, 1, 8, 3], device="cuda")
    pages = torch.zeros(10, 8, 32, device="cuda", dtype=torch.bfloat16)
    keys = torch.randint(-1, 2, (1, 45, 32), device="cuda").bfloat16()
    ids = torch.arange(45, device="cuda")
    pages[blocks[ids // 8], ids % 8] = keys[0]
    paged = PagedIndexKeys(pages, blocks, 45)
    q = torch.randint(-1, 2, (1, 6, 4, 32), device="cuda").bfloat16()
    weights = torch.ones(1, 6, 4, device="cuda", dtype=torch.bfloat16)
    visible = torch.tensor([1, 9, 18, 27, 36, 45], device="cuda")
    opts = {
        "topk": 8,
        "make_candidates": True,
        "block_size": 4,
        "topk_blocks": 3,
        "key_tile": 16,
        "tie_break": tie,
    }
    expected = select_indices(q, weights, keys, visible, **opts)
    actual = select_indices(q, weights, paged, visible, **opts)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    opts.update(make_candidates=False, candidate_blocks=expected[1])
    torch.testing.assert_close(
        select_indices(q, weights, paged, visible, **opts)[0],
        select_indices(q, weights, keys, visible, **opts)[0],
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_ragged_swa_wrap_csr_and_attention_match_private_caches(small_config):
    torch.manual_seed(42)
    config = small_config
    config.max_position_embeddings = 64
    topology = build_attention_topology(config)
    geo = geometry(config)
    paged = PagedAttentionCache(geo, 70, 5, "cuda")
    private = {
        r: EagerAttentionCache(config, topology, 1, 64, "cuda") for r in (11, 29)
    }
    slots = {11: 4, 29: 1}
    blocks = {11: tuple(range(0, 32, 2)), 29: tuple(range(1, 33, 2))}
    sentinel = paged.state_bytes[0].clone()
    for schedule in (
        ((11, 5), (29, 3)),
        ((29, 8), (11, 1)),
        ((11, 19), (29, 2)),
        ((29, 1), (11, 1)),
    ):
        spans, at = [], 0
        for request, length in schedule:
            spans.append(
                RequestSpan(
                    request,
                    private[request].position,
                    at,
                    length,
                    slots[request],
                    blocks[request],
                )
            )
            at += length
        step = paged.begin_step(spans)
        histories = paged.prepare_state(step)
        eager_steps = [
            private[s.request_id].begin_step(s.position, s.length, 1) for s in spans
        ]
        for spec in topology:
            q = torch.randn(1, at, 8, 512, dtype=torch.bfloat16, device="cuda")
            kv = torch.randn(1, at, 512, dtype=torch.bfloat16, device="cuda")
            sink = torch.randn(8, device="cuda")
            if spec.ratio:
                count = max(s.end // spec.ratio for s in spans)
                values = torch.randn(1, count, 512, device="cuda", dtype=torch.bfloat16)
                for i, (span, local) in enumerate(zip(spans, step.request_steps)):
                    eager = private[span.request_id]
                    eager.main[spec.kv_owner][:, :count] = values
                    view = next(
                        v
                        for v, _, rows in paged.requests(step)
                        if rows.start == span.offset
                    )
                    view.write_global(
                        spec.kv_owner, 0, values, values[..., : config.index_head_dim]
                    )
                    if spec.topk_owner == spec.layer_id:
                        chosen = (
                            torch.arange(min(count, 6), device="cuda")
                            .expand(1, span.length, -1)
                            .int()
                            .clone()
                        )
                        visible = (local.positions + 1) // spec.ratio
                        chosen.masked_fill_(chosen >= visible[None, :, None], -1)
                        local.indices[spec.layer_id] = chosen
                        eager_steps[i].indices[spec.layer_id] = chosen
            if step.decode:
                paged.write_window(spec.layer_id, kv, step)
            prefix, pptr, extend, eptr = paged.attention_indices(spec, step)
            actual = (
                sparse_attn_v4_paged_decode(
                    q.flatten(0, 1), paged.pool, prefix, pptr, sink, 512**-0.5
                )
                if step.decode
                else sparse_attn_v4_paged_prefill(
                    q.flatten(0, 1),
                    paged.pool,
                    prefix,
                    pptr,
                    kv.flatten(0, 1),
                    extend,
                    eptr,
                    sink,
                    512**-0.5,
                )
            )
            for span, eager_step in zip(spans, eager_steps):
                eager = private[span.request_id]
                local_kv = kv[:, span.token_slice].contiguous()
                if eager_step.decode:
                    eager.write_window(spec.layer_id, local_kv, eager_step)
                a, b, c, d = eager.attention_indices(spec, eager_step)
                query = q[:, span.token_slice].flatten(0, 1).contiguous()
                expected = (
                    sparse_attn_v4_paged_decode(
                        query, eager.pool, a, b, sink, 512**-0.5
                    )
                    if eager_step.decode
                    else sparse_attn_v4_paged_prefill(
                        query,
                        eager.pool,
                        a,
                        b,
                        local_kv.flatten(0, 1),
                        c,
                        d,
                        sink,
                        512**-0.5,
                    )
                )
                torch.testing.assert_close(
                    actual[span.token_slice], expected, rtol=0.01, atol=0.008
                )
                if not eager_step.decode:
                    eager.write_window(spec.layer_id, local_kv, eager_step)
            if not step.decode:
                paged.write_window(spec.layer_id, kv, step)
        paged.finish_step(step, histories)
        for span, eager_step in zip(spans, eager_steps):
            private[span.request_id].finish_step(eager_step)
    torch.testing.assert_close(paged.state_bytes[0], sentinel, rtol=0, atol=0)
    before = paged.backing.clone()
    empty = paged.begin_step([])
    paged.prepare_state(empty)
    paged.finish_step(empty, np.empty((0, 3), dtype=np.int64))
    torch.testing.assert_close(paged.backing, before, rtol=0, atol=0)

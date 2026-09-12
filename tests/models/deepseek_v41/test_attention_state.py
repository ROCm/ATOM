# SPDX-License-Identifier: MIT
"""Physical KV ownership and causal CSR contracts for the V4 BF16 caller."""

import pytest
import torch

from atom.model_ops.attentions.deepseek_v41_state import EagerAttentionCache
from atom.models.deepseek_v41.config import build_attention_topology

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ROCm GPU required"
)


def test_shared_pool_ring_wrap_and_two_source_causality(small_config):
    config = small_config
    config.max_position_embeddings = 64
    topology = build_attention_topology(config)
    cache = EagerAttentionCache(config, topology, 2, 64, "cuda")
    cache.pool.fill_(torch.nan)
    assert set(cache.main) == {1, 3} and set(cache.index) == {1, 3}
    for owner, main in cache.main.items():
        assert (
            main.untyped_storage().data_ptr() == cache.pool.untyped_storage().data_ptr()
        )
        for b in range(2):
            main[b] = (
                -1 - 64 * b - 16 * owner - torch.arange(main.shape[1], device="cuda")
            )[:, None]

    def window_value(b, layer, pos):
        return b * 128 + layer * 16 + pos

    for length in (5, 3, 1, 20, 1, 24):
        position = cache.position
        step = cache.begin_step(position, length, 2)
        for spec in topology:
            kv = (
                torch.tensor(
                    [
                        [
                            window_value(b, spec.layer_id, pos)
                            for pos in range(position, position + length)
                        ]
                        for b in range(2)
                    ],
                    dtype=torch.bfloat16,
                    device="cuda",
                )[:, :, None]
                .expand(-1, -1, config.head_dim)
                .contiguous()
            )
            if spec.ratio and spec.topk_owner == spec.layer_id:
                width = min(8, (position + length) // spec.ratio)
                selected = torch.full((2, length, width), -1, dtype=torch.int32)
                for b in range(2):
                    for offset in range(length):
                        visible = (position + offset + 1) // spec.ratio
                        chosen = list(range(visible))[-width:] if width else []
                        if spec.layer_id == 4:
                            chosen = chosen[::2]
                        if b == 1:
                            chosen = chosen[1:]
                        selected[b, offset, : len(chosen)] = torch.tensor(
                            chosen, dtype=torch.int32
                        )
                step.indices[spec.layer_id] = selected.cuda()
            if length == 1:
                cache.write_window(spec.layer_id, kv, step)
            prefix, pptr, extend, eptr = cache.attention_indices(spec, step)
            prefix, pptr, extend, eptr = (x.cpu() for x in (prefix, pptr, extend, eptr))
            selected = step.indices[spec.topk_owner].cpu() if spec.ratio else None
            for b in range(2):
                for offset in range(length):
                    row = b * length + offset
                    last = position + offset
                    first = max(0, last - config.sliding_window + 1)
                    history_end = last + 1 if length == 1 else position
                    global_ids = (
                        []
                        if selected is None
                        else [i for i in selected[b, offset].tolist() if i >= 0]
                    )
                    expected = [
                        -1 - 64 * b - 16 * spec.kv_owner - i for i in global_ids
                    ]
                    expected += [
                        window_value(b, spec.layer_id, pos)
                        for pos in range(first, history_end)
                    ]
                    rows = prefix[pptr[row] : pptr[row + 1]].long().cuda()
                    assert cache.pool[rows, 0].tolist() == expected
                    if length > 1:
                        rows = extend[eptr[row] : eptr[row + 1]].long().cuda()
                        expected = [
                            window_value(b, spec.layer_id, pos)
                            for pos in range(max(position, first), last + 1)
                        ]
                        assert kv.flatten(0, 1)[rows, 0].tolist() == expected
            if length > 1:
                cache.write_window(spec.layer_id, kv, step)
            for b in range(2):
                for pos in range(
                    max(0, position + length - config.sliding_window), position + length
                ):
                    assert cache.pool[
                        cache.window[spec.layer_id].index(b, pos), 0
                    ].item() == window_value(b, spec.layer_id, pos)
        cache.finish_step(step)
    assert cache.position == 54


def test_request_cache_isolation_and_contiguous_steps(small_config):
    topology = build_attention_topology(small_config)
    first = EagerAttentionCache(small_config, topology, 1, 8, "cuda")
    second = EagerAttentionCache(small_config, topology, 1, 8, "cuda")
    second.pool.fill_(17)
    step = first.begin_step(0, 1, 1)
    first.write_window(
        0,
        torch.ones(1, 1, small_config.head_dim, device="cuda", dtype=torch.bfloat16),
        step,
    )
    first.finish_step(step)
    assert torch.all(second.pool == 17)
    assert first.position == 1 and second.position == 0
    for position, length, batch in ((0, 1, 1), (2, 1, 1), (1, 1, 2), (1, 8, 1)):
        with pytest.raises(ValueError):
            first.begin_step(position, length, batch)

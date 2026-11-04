# SPDX-License-Identifier: MIT
"""Mixed startup profiling uses actual token/query/sequence limits."""

import math
from types import SimpleNamespace

import pytest

from atom.model_engine.kv_block import STATE_SLOT_CLASS
from atom.model_engine.mixed_profile import _batch, mixed_profile_shapes


@pytest.mark.parametrize("width", [1, 8])
@pytest.mark.parametrize("budget", [2048, 8192, 32768])
@pytest.mark.parametrize("max_seqs", [2, 256, 1024])
@pytest.mark.parametrize("max_len", [8192, 1048576])
def test_profiles_are_schedulable_and_cover_decode_heavy_shapes(
    width, budget, max_seqs, max_len
):
    shapes = mixed_profile_shapes(budget, max_seqs, width, max_len)
    assert shapes
    for prefill, decodes, q in shapes:
        assert prefill > 0 and decodes > 0 and q == width
        assert prefill + decodes * q <= budget
        assert math.ceil(prefill / max_len) + decodes <= max_seqs
    assert shapes[0][1] == 1
    assert shapes[-1][1] == min(max_seqs - 1, (budget - max(1, budget // 8)) // width)


def test_impossible_mixed_shape_is_not_profiled():
    assert mixed_profile_shapes(8, 10, 8, 100) == []
    assert mixed_profile_shapes(2048, 1, 8, 4096) == []


@pytest.mark.parametrize("width", [1, 8])
@pytest.mark.parametrize("long_context", [False, True])
def test_profile_addresses_include_full_draft_lookahead(width, long_context):
    runner = SimpleNamespace(
        config=SimpleNamespace(max_model_len=257, max_num_seqs=8),
        block_size=64,
        drafter=SimpleNamespace(mtp_k=7),
        pool_plan=SimpleNamespace(entries_per_req={STATE_SLOT_CLASS: 2}),
    )
    for prefill, decodes, q in mixed_profile_shapes(400, 8, width, 251):
        batch, pages = _batch(runner, prefill, decodes, q, long_context=long_context)
        assert batch.num_spec_step == 7
        assert batch.total_tokens_num <= 400
        assert batch.total_seqs_num <= 8
        # The allocated table has room for the draft's lookahead and fits the
        # configured max-context metadata even when verification uses q=1.
        private_pages = []
        for table in batch.block_tables:
            assert len(table) <= math.ceil(257 / runner.block_size)
            private_pages.extend(p for p in table if p)
        assert len(set(private_pages)) == len(private_pages)
        assert pages == max(private_pages) + 1
        assert all(
            len(draft) == 7 for draft in batch.scheduled_spec_decode_tokens[-decodes:]
        )

# SPDX-License-Identifier: MIT
"""Mixed startup profiling uses actual token/query/sequence limits."""

import math
from contextlib import contextmanager
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


def test_million_token_profile_does_not_materialize_history(monkeypatch):
    from atom.model_engine import mixed_profile

    original = mixed_profile.Sequence
    rows = []

    def bounded_sequence(tokens, **kwargs):
        # Fail before an accidental full-context allocation can multiply by 1024.
        assert len(tokens) <= 16384
        seq = original(tokens, **kwargs)
        rows.append(seq)
        return seq

    monkeypatch.setattr(mixed_profile, "Sequence", bounded_sequence)
    runner = SimpleNamespace(
        config=SimpleNamespace(
            max_model_len=1048576, max_num_seqs=1024, speculative_config=None
        ),
        block_size=256,
        drafter=SimpleNamespace(mtp_k=7),
        pool_plan=SimpleNamespace(entries_per_req={STATE_SLOT_CLASS: 2}),
    )
    batch, _ = _batch(runner, 8192, 1023, 8, long_context=True)
    assert batch.total_seqs_num == 1024
    assert batch.total_tokens_num == 8192 + 1023 * 8
    assert batch.context_lens.tolist() == [1048570] * 2 + [8192] * 1022
    assert not batch.scheduled_tokens.any()
    assert all(isinstance(seq.token_ids, mixed_profile._ProfileTokens) for seq in rows)
    assert all(
        len(seq.token_ids) == seq.num_tokens == context
        for seq, context in zip(rows, batch.context_lens)
    )
    assert all(seq.token_ids[-1] == 0 for seq in rows)
    assert list(rows[-1].token_ids[-8:]) == [0] * 8


@pytest.mark.parametrize("width", [1, 8])
@pytest.mark.parametrize("long_context", [False, True])
@pytest.mark.parametrize("block_drafter", [False, True])
@pytest.mark.parametrize("max_len", [256, 257])
def test_profile_addresses_include_full_draft_lookahead(
    width, long_context, block_drafter, max_len
):
    lookahead = 7 if block_drafter else 6
    runner = SimpleNamespace(
        config=SimpleNamespace(
            max_model_len=max_len,
            max_num_seqs=8,
            speculative_config=SimpleNamespace(use_dspark=lambda: block_drafter),
        ),
        block_size=64,
        drafter=SimpleNamespace(mtp_k=7),
        pool_plan=SimpleNamespace(entries_per_req={STATE_SLOT_CLASS: 2}),
    )
    for prefill, decodes, q in mixed_profile_shapes(400, 8, width, max_len - lookahead):
        batch, pages = _batch(runner, prefill, decodes, q, long_context=long_context)
        assert batch.num_spec_step == 7
        assert batch.total_tokens_num <= 400
        assert batch.total_seqs_num <= 8
        # The allocated table has room for the draft's lookahead and fits the
        # configured max-context metadata even when verification uses q=1.
        private_pages = []
        for context, table in zip(batch.context_lens, batch.block_tables):
            assert len(table) <= math.ceil(max_len / runner.block_size)
            assert len(table) * runner.block_size >= context + lookahead
            private_pages.extend(p for p in table if p)
        assert len(set(private_pages)) == len(private_pages)
        assert pages == max(private_pages) + 1
        assert all(
            len(draft) == 7 for draft in batch.scheduled_spec_decode_tokens[-decodes:]
        )


@pytest.mark.parametrize("max_seqs", [512, 1024])
@pytest.mark.parametrize("budget", [2048, 16384])
@pytest.mark.parametrize("width", [1, 8])
def test_long_context_work_is_bounded_without_reducing_query_shapes(
    max_seqs, budget, width
):
    max_len = 1048576 - 7
    runner = SimpleNamespace(
        config=SimpleNamespace(
            max_model_len=1048576,
            max_num_seqs=max_seqs,
            speculative_config=SimpleNamespace(use_dspark=lambda: True),
        ),
        block_size=256,
        drafter=SimpleNamespace(mtp_k=7),
        pool_plan=SimpleNamespace(entries_per_req={STATE_SLOT_CLASS: 2}),
    )
    for prefill, decodes, q in reversed(
        mixed_profile_shapes(budget, max_seqs, width, max_len)
    ):
        short, _ = _batch(runner, prefill, decodes, q, long_context=False)
        long, _ = _batch(runner, prefill, decodes, q, long_context=True)
        # The full token and sequence envelopes still reach the model/drafter.
        assert long.total_tokens_num == short.total_tokens_num == prefill + decodes * q
        assert long.total_seqs_num == short.total_seqs_num == 1 + decodes
        assert long.num_spec_step == short.num_spec_step == 7
        # At c=1024 this is about 10M logical positions, rather than 1B.
        assert sum(long.context_lens) <= 2 * max_len + (max_seqs - 2) * 8192
        assert long.context_lens[:2].tolist() == [max_len, max_len]
        assert all(context == 8192 for context in long.context_lens[2:])


@pytest.mark.parametrize("tbo_flags", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("fail_unsplit", [False, True])
def test_profile_budgets_unsplit_fallback_and_restores_config(
    monkeypatch, tbo_flags, fail_unsplit
):
    from atom.model_engine import mixed_profile

    config = SimpleNamespace(
        enable_tbo=tbo_flags[0],
        enable_tbo_decode=tbo_flags[1],
        max_num_seqs=4,
        max_model_len=128,
        speculative_config=None,
    )
    memory = {"allocated": 80, "reserved": 80, "peak": 85, "reserved_peak": 85}
    observed = []
    pool_open = False

    @contextmanager
    def temporary_pool(pages):
        nonlocal pool_open
        pool_open = True
        memory.update(allocated=100, reserved=100)
        try:
            yield
        finally:
            pool_open = False
            memory["allocated"] = 80

    def forward(batch):
        flags = (config.enable_tbo, config.enable_tbo_decode)
        observed.append((flags, memory["reserved"]))
        if not config.enable_tbo and fail_unsplit:
            raise RuntimeError("unsplit forward failed")
        # An unsplit forward needs a larger workspace at the SAME token budget.
        allocated, reserved = (112, 116) if config.enable_tbo else (128, 140)
        memory.update(peak=allocated, reserved_peak=reserved, reserved=reserved)

    runner = SimpleNamespace(
        config=config,
        max_bs=4,
        block_size=16,
        warmup_max_tokens=lambda: 64,
        attn_metadata_builder=SimpleNamespace(temporary_profile_cache=temporary_pool),
        tokenID_processor=SimpleNamespace(clean=lambda: None),
        forward=forward,
    )
    monkeypatch.setattr(
        mixed_profile,
        "_batch",
        lambda *args, **kwargs: (SimpleNamespace(context_lens=[64, 64]), 1),
    )
    cuda = mixed_profile.torch.cuda
    monkeypatch.setattr(cuda, "synchronize", lambda: None)
    monkeypatch.setattr(cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(
        cuda, "empty_cache", lambda: memory.update(reserved=memory["allocated"])
    )
    monkeypatch.setattr(cuda, "memory_allocated", lambda: memory["allocated"])
    monkeypatch.setattr(cuda, "memory_reserved", lambda: memory["reserved"])
    monkeypatch.setattr(cuda, "max_memory_allocated", lambda: memory["peak"])
    monkeypatch.setattr(cuda, "max_memory_reserved", lambda: memory["reserved_peak"])
    monkeypatch.setattr(
        cuda, "mem_get_info", lambda: (1000 - memory["reserved"] - 7, 1000)
    )

    if fail_unsplit:
        with pytest.raises(RuntimeError, match="unsplit forward failed"):
            mixed_profile.profile_mixed_memory(runner)
    else:
        # Include the unsplit reserved peak, excluding the temporary pool.
        assert mixed_profile.profile_mixed_memory(runner) == (80 + (140 - 100), 7)
    assert not pool_open
    assert (config.enable_tbo, config.enable_tbo_decode) == tbo_flags
    assert {flags for flags, _ in observed} == {tbo_flags, (False, False)}
    if any(tbo_flags):
        # Transitioning out of TBO must retain the preceding allocator cache.
        first_unsplit = next(reserved for flags, reserved in observed if not flags[0])
        assert first_unsplit == 116

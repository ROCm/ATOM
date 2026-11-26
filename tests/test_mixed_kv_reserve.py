# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Mixed startup profiling uses actual token/query/sequence limits."""

import gc
import math
import weakref
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from aiter_stub import atom_without_aiter

from atom.model_engine.kv_block import STATE_SLOT_CLASS
from atom.model_engine.mixed_profile import _batch, mixed_profile_shapes


@pytest.fixture
def runner_module():
    with atom_without_aiter() as load:
        yield load("atom.model_engine.model_runner")


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
    assert shapes[-1][1] == min(max_seqs - 1, (budget - 1) // width)


@pytest.mark.parametrize("width", [1, 8])
def test_profile_covers_short_prefill_with_unused_token_budget(width):
    shapes = mixed_profile_shapes(100 * width, 101, width, 8192)
    assert (width, 99, width) in shapes


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


@pytest.mark.parametrize("fail_unsplit", [False, True])
@pytest.mark.parametrize("confidence", [False, True])
@pytest.mark.parametrize("dp_attention", [False, True])
@pytest.mark.parametrize("token_budget,spec_bs", [(16, 2), (64, 4)])
def test_profile_budgets_reserved_peak_and_releases_scratch(
    monkeypatch, fail_unsplit, confidence, dp_attention, token_budget, spec_bs
):
    from atom.model_engine import mixed_profile
    from atom.spec_decode.dspark_verify import VerifyScheduler

    verify = VerifyScheduler.__new__(VerifyScheduler)
    verify._ell_map = {}
    verify._ell_pending = []

    config = SimpleNamespace(
        max_num_seqs=4,
        max_model_len=128,
        speculative_config=None,
        enable_dp_attention=dp_attention,
    )
    memory = {
        "allocated": 80,
        "reserved": 80,
        "peak": 85,
        "reserved_peak": 85,
        "non_torch": 7,
    }
    observed = []
    shapes = []
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
        assert runner.gpu_forward_metrics is None
        observed.append(memory["reserved"])
        shapes.append((batch.prefill, batch.decode, batch.width))
        if confidence:
            verify._ell_map = {42: 1}
            verify._ell_pending.append(object())
        if fail_unsplit:
            raise RuntimeError("unsplit forward failed")
        allocated, reserved = (148, 170) if batch.prefill == 0 else (128, 140)
        memory.update(peak=allocated, reserved_peak=reserved, reserved=reserved)
        memory["non_torch"] = 13 if batch.prefill == 0 else 7

    runner = SimpleNamespace(
        config=config,
        gpu_forward_metrics=object(),
        max_bs=4,
        block_size=16,
        warmup_max_tokens=lambda: token_budget,
        attn_metadata_builder=SimpleNamespace(temporary_profile_cache=temporary_pool),
        tokenID_processor=SimpleNamespace(clean=lambda: None),
        forward=forward,
    )
    if confidence:
        runner.drafter = SimpleNamespace(mtp_k=7, verify_scheduler=verify)
    monkeypatch.setattr(
        mixed_profile,
        "_batch",
        lambda runner, prefill, decode, width, **kwargs: (
            SimpleNamespace(
                context_lens=[64, 64], prefill=prefill, decode=decode, width=width
            ),
            1,
        ),
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
        cuda,
        "mem_get_info",
        lambda: (1000 - memory["reserved"] - memory["non_torch"], 1000),
    )

    original_metrics = runner.gpu_forward_metrics
    if fail_unsplit:
        with pytest.raises(RuntimeError, match="unsplit forward failed"):
            mixed_profile.profile_mixed_memory(runner)
    else:
        # Include the unsplit reserved peak, excluding the temporary pool.
        expected_reserved = 170 if dp_attention else 140
        assert mixed_profile.profile_mixed_memory(runner) == (
            80 + (expected_reserved - 100),
            13 if dp_attention else 7,
        )
        decode_shapes = {(d, q) for p, d, q in shapes if p == 0}
        assert decode_shapes == (
            {(spec_bs, 1), (spec_bs, 8)}
            if dp_attention and confidence
            else {(4, 1)} if dp_attention else set()
        )
    assert not pool_open
    assert runner.gpu_forward_metrics is original_metrics
    assert observed
    assert verify._ell_pending == []
    assert verify.ell_by_req == {}


@pytest.fixture(scope="module")
def builder_cls():
    with atom_without_aiter() as load:
        yield load(
            "atom.model_ops.attentions.deepseek_v4_attn"
        ).DeepseekV4AttentionMetadataBuilder


@pytest.mark.parametrize("fail_forward", [False, True])
def test_profile_drops_region_layout_when_serving_reuses_addresses(
    monkeypatch, builder_cls, fail_forward
):
    storage = torch.zeros(128, dtype=torch.uint8)
    plane = torch.zeros(64, dtype=torch.uint8)
    serving = storage.view(2, 4, 16)
    plan = SimpleNamespace(entries={})
    plan.with_paged_entries = lambda blocks: plan
    runner = SimpleNamespace(
        model=torch.nn.Module(),
        pool_plan=plan,
        config=SimpleNamespace(speculative_config=None),
        index_pool=serving,
    )
    builder = SimpleNamespace(
        model_runner=runner,
        pool_geometry=SimpleNamespace(envelope_rows=4),
        num_blocks=4,
        csa_layers=[0, 1],
        _page_unit_region_owners=(),
        _kv_planes=lambda: [plane],
        _plane_row_widths=lambda: [8],
        _indexer_page_pools=lambda: [(runner.index_pool, None)],
        allocate_kv_cache_tensors=lambda **_: {
            "index_pool": storage[:64].view(2, 2, 16)
        },
        allocate_per_req_cache=lambda _: {},
        build_kv_cache_tensor=lambda _: None,
    )
    runner.attn_metadata_builder = builder
    builder._invalidate_pool_caches = lambda: builder_cls._invalidate_pool_caches(
        builder
    )
    builder.warmup_per_req_cache = lambda: builder_cls._page_unit_regions(builder)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    def profile():
        with builder_cls.temporary_profile_cache(builder, 2):
            bases, _ = builder_cls._page_unit_regions(builder)
            assert bases[-1] == storage.data_ptr() + 32
            if fail_forward:
                raise RuntimeError("profile failed")

    if fail_forward:
        with pytest.raises(RuntimeError, match="profile failed"):
            profile()
    else:
        profile()
    assert runner.index_pool is serving
    bases, _ = builder_cls._page_unit_regions(builder)
    assert bases[-1] == serving.data_ptr() + serving.stride(0)


@pytest.mark.parametrize("fail_forward", [False, True])
@pytest.mark.parametrize("independent_draft", [False, True])
def test_temporary_pool_restores_registered_buffers(
    monkeypatch, fail_forward, independent_draft, builder_cls
):
    module = torch.nn.Module()
    placeholder = torch.ones(1)
    module.register_buffer("kv_cache", placeholder)
    module.scale = placeholder
    old_plan = SimpleNamespace()
    plan = SimpleNamespace(entries={})
    old_plan.with_paged_entries = lambda blocks: plan
    runner = SimpleNamespace(
        model=module,
        pool_plan=old_plan,
        config=SimpleNamespace(speculative_config=None),
    )
    weak_pools = []

    def allocate(**kwargs):
        pool = torch.zeros(64)
        weak_pools.append(weakref.ref(pool))
        return {"test_pool": pool}

    def bind(layer):
        layer.kv_cache = runner.test_pool[:8]
        layer.scale = runner.test_pool[8:16]
        layer.new_view = runner.test_pool[16:24]

    builder = SimpleNamespace(
        model_runner=runner,
        pool_geometry="original",
        num_blocks=0,
        allocate_kv_cache_tensors=allocate,
        allocate_per_req_cache=lambda entries: {},
        warmup_per_req_cache=lambda: None,
        build_kv_cache_tensor=bind,
        _invalidate_pool_caches=lambda: None,
    )
    runner.attn_metadata_builder = builder
    draft = torch.nn.Module()
    draft.register_buffer("kv_cache", placeholder)
    if independent_draft:
        runner.config.speculative_config = object()
        runner.drafter = SimpleNamespace(model=draft)

        def bind_draft(layer):
            layer.kv_cache = runner.test_pool[24:32]
            return layer.kv_cache

        runner.draft_kv_builder = SimpleNamespace(build_kv_cache_tensor=bind_draft)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    def profile():
        with builder_cls.temporary_profile_cache(builder, 64):
            assert module.kv_cache.numel() == 8
            assert module.scale.numel() == 8
            if independent_draft:
                assert draft.kv_cache.numel() == 8
                assert draft.kv_cache.data_ptr() == runner.test_pool[24:].data_ptr()
            if fail_forward:
                raise RuntimeError("forward failed")

    if fail_forward:
        with pytest.raises(RuntimeError, match="forward failed"):
            profile()
    else:
        profile()
    gc.collect()
    assert module.kv_cache is placeholder
    assert module.scale is placeholder
    assert draft.kv_cache is placeholder
    assert not hasattr(module, "new_view")
    assert not hasattr(runner, "test_pool")
    assert runner.pool_plan is old_plan
    assert all(ref() is None for ref in weak_pools)


def test_binding_falls_back_only_for_unrecognized_draft_modules():
    from atom.model_engine.kv_cache_binding import kv_cache_bindings

    target, draft, shared = object(), object(), object()
    target_calls, draft_calls = [], []

    def target_bind(module):
        target_calls.append(module)
        return "target" if module is target else "shared"

    def draft_bind(module):
        draft_calls.append(module)
        return "draft" if module is draft else None

    runner = SimpleNamespace(
        model=SimpleNamespace(modules=lambda: [target]),
        drafter=SimpleNamespace(model=SimpleNamespace(modules=lambda: [draft, shared])),
        config=SimpleNamespace(speculative_config=object()),
        attn_metadata_builder=SimpleNamespace(build_kv_cache_tensor=target_bind),
        draft_kv_builder=SimpleNamespace(build_kv_cache_tensor=draft_bind),
    )
    assert list(kv_cache_bindings(runner)) == [
        (target, "target"),
        (draft, "draft"),
        (shared, "shared"),
    ]
    assert target_calls == [target, shared]
    assert draft_calls == [draft, shared]


@pytest.mark.parametrize("paged_state", [False, True])
@pytest.mark.parametrize("daemon_extra_gib", [0, 3])
def test_mixed_profile_is_not_charged_to_graph_pools(
    monkeypatch, caplog, paged_state, daemon_extra_gib, runner_module
):
    import torch

    from atom.model_engine.kv_block import STATE_SLOT_CLASS
    from atom.model_engine.state_runtime import DEFAULT_STATE_RUNTIME, StateTransfer
    from atom.model_ops.attentions.pool_layout.sub_pool_spec import (
        PoolPlan,
        page_pool,
        state_pool,
    )

    gib = 1 << 30
    stats = {
        "allocated_bytes.all.peak": 102 * gib,
        "allocated_bytes.all.current": 100 * gib,
    }
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda: stats)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda: ((100 - daemon_extra_gib) * gib, 200 * gib)
    )
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 100 * gib)
    monkeypatch.setattr(
        runner_module, "daemon_load_extra", lambda: daemon_extra_gib * gib
    )
    monkeypatch.setattr(
        runner_module.KVConnectorFactory, "kv_budget_reserve_bytes", lambda config: 0
    )
    runner = runner_module.ModelRunner.__new__(runner_module.ModelRunner)
    runner.device = "cpu"
    runner.block_size = 4
    runner.pool_plan = PoolPlan.empty()
    runner.state_runtime = DEFAULT_STATE_RUNTIME
    runner.config = SimpleNamespace(
        hf_config=SimpleNamespace(head_dim=128),
        enable_mixed_prefill_decode=True,
        enforce_eager=False,
        max_num_seqs=4,
        max_model_len=64,
        gpu_memory_utilization=0.9,
        pipeline_parallel_size=1,
        is_multinode_tp=False,
        enable_rapidserve=False,
        kv_transfer_config=None,
    )
    runner.drafter = SimpleNamespace(uses_confidence_schedule=True, mtp_k=7)
    runner._tr_hybrid_active = lambda: False
    runner._dspark_capture_q_buckets = lambda q: [1, 2, 4, 8]
    runner._sub_pool_specs = lambda: [page_pool(gib)] + (
        [state_pool(STATE_SLOT_CLASS, 512, entries_per_req=1)] if paged_state else []
    )
    runner.attn_metadata_builder = SimpleNamespace(
        state_transfer_slots=lambda: 0,
        state_transfer=lambda: (
            StateTransfer.copy("test-layout")
            if paged_state
            else DEFAULT_STATE_RUNTIME.transfer
        ),
        checkpoint_image_bytes=lambda: 256,
        checkpoint_page_unit_bytes=lambda: gib,
        sub_pool_specs=runner._sub_pool_specs,
    )

    def profile(runner):
        spec = runner.state_runtime.checkpoint_spec
        if paged_state:
            assert spec is not None
            assert (spec.page_unit_bytes, spec.slot_bytes, spec.image_bytes) == (
                gib,
                512,
                256,
            )
            assert spec.layout_id == "test-layout"
        else:
            assert spec is None
        stats["allocated_bytes.all.peak"] = 100 * gib
        return 112 * gib, (2 + daemon_extra_gib) * gib

    profile = Mock(side_effect=profile)
    monkeypatch.setattr(runner_module, "profile_mixed_memory", profile)
    with caplog.at_level("INFO", logger="atom"):
        result = runner.get_num_blocks()
    # 180 GiB budget - 112 live - 2 non-torch - 1.6 graphs - 4 safety.
    assert result["num_kvcache_blocks"] == 60
    assert profile.call_count == 1
    assert sum("Memory budget:" in r.message for r in caplog.records) == 1

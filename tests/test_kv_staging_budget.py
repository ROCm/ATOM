# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Exercise the runner's real KV budget path with CPU memory statistics."""

import logging
from types import SimpleNamespace

import pytest
import torch

from atom.model_engine.state_runtime import StateTransfer
from atom.model_ops.attentions.pool_layout.sub_pool_spec import page_pool, state_pool

try:
    from atom.model_engine import model_runner
    from atom.model_ops.attentions.backends import AttentionMetadataBuilder
except (ImportError, RuntimeError) as exc:
    pytest.skip(f"ModelRunner dependencies unavailable: {exc}", allow_module_level=True)


@pytest.fixture
def budget_runner(monkeypatch):
    runner = object.__new__(model_runner.ModelRunner)
    runner.device = torch.device("cpu")
    runner.block_size = 16
    runner.world_size = 1
    runner.model = SimpleNamespace(modules=lambda: iter(()))
    runner.config = SimpleNamespace(
        hf_config=SimpleNamespace(head_dim=64, num_key_value_heads=1),
        gpu_memory_utilization=0.8,
        max_num_seqs=4,
        max_model_len=128,
        pipeline_parallel_size=1,
        decode_context_parallel_size=1,
        speculative_config=None,
    )
    runner.attn_metadata_builder = SimpleNamespace(
        sub_pool_specs=lambda: [page_pool(256)],
        kv_transfer_staging_bytes=lambda: 512,
        state_transfer=StateTransfer.none,
        allocate_per_req_cache=lambda entries: {},
        warmup_per_req_cache=lambda: None,
        get_kv_transfer_tensors=lambda: None,
    )
    runner._estimate_cudagraph_overhead = lambda: 0
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (9000, 10000))
    monkeypatch.setattr(
        torch.cuda,
        "memory_stats",
        lambda: {
            "allocated_bytes.all.peak": 1000,
            "allocated_bytes.all.current": 1000,
        },
    )
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 1000)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    return runner


@pytest.mark.parametrize("with_draft", [False, True])
@pytest.mark.parametrize(
    "free,staging,blocks,draft_blocks",
    [
        (9000, 0, 26, 13),
        (9000, 512, 24, 12),
        (3000, 0, 11, 5),
        (3000, 512, 9, 4),
    ],
)
def test_staging_is_reserved_under_both_budget_limits(
    monkeypatch, budget_runner, with_draft, free, staging, blocks, draft_blocks
):
    runner = budget_runner
    runner.attn_metadata_builder.kv_transfer_staging_bytes = lambda: staging
    if with_draft:
        # Independent draft KV adds per-block bytes but no staging allocation.
        runner.draft_kv_builder = SimpleNamespace(
            sub_pool_specs=lambda: [page_pool(256)]
        )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (free, 10000))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 10000 - free)
    result = runner.get_num_blocks()
    assert result["num_kvcache_blocks"] == (draft_blocks if with_draft else blocks)
    assert runner.kv_transfer_staging_reserved_bytes == staging
    assert runner.pool_plan.total_reserved_bytes + staging <= free
    # 8000 utilization bytes minus 1000 profile peak and 200 safety bytes.
    assert runner.pool_plan.total_reserved_bytes + staging <= 6800


def test_default_backend_does_not_reserve_transfer_staging():
    assert AttentionMetadataBuilder.kv_transfer_staging_bytes(SimpleNamespace()) == 0


@pytest.mark.parametrize(
    "free,staging,state_bytes,hint",
    [
        (9000, 2048, 1400, "--gpu-memory-utilization >= 0.89"),
        (3000, 512, 750, "will NOT help"),
    ],
)
def test_staging_reservation_is_included_in_insufficient_budget_diagnostics(
    monkeypatch, budget_runner, free, staging, state_bytes, hint
):
    runner = budget_runner
    runner.attn_metadata_builder.kv_transfer_staging_bytes = lambda: staging
    runner.attn_metadata_builder.sub_pool_specs = lambda: [
        page_pool(256),
        state_pool("state", state_bytes, entries_per_req=1),
    ]
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (free, 10000))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 10000 - free)
    with pytest.raises(RuntimeError) as exc:
        runner.get_num_blocks()
    assert hint in str(exc.value)


@pytest.mark.parametrize("extra_allocation", [0, 512])
def test_allocation_reconciliation_includes_reserved_staging(
    monkeypatch, budget_runner, caplog, extra_allocation
):
    runner = budget_runner
    blocks = runner.get_num_blocks()["num_kvcache_blocks"]
    allocation = runner.pool_plan.total_reserved_bytes + 512 + extra_allocation
    readings = iter([1000, 1000 + allocation])
    monkeypatch.setattr(
        torch.cuda,
        "memory_stats",
        lambda: {"allocated_bytes.all.current": next(readings)},
    )
    runner._back_paged_pools = lambda blocks: torch.empty(0, dtype=torch.uint8)
    monkeypatch.setattr(model_runner, "set_kv_cache_data", lambda *args, **kwargs: None)
    with caplog.at_level(logging.WARNING, logger="atom"):
        assert runner.allocate_kv_cache(blocks)
    assert ("KV cache allocation mismatch" in caplog.text) == bool(extra_allocation)

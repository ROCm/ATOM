# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""GLM dual-CTA dispatch and its phase-reused scratch layout."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

pytest.importorskip("flydsl")

from atom.mono.dsa_indexer import kernel, runner, select
from atom.mono.dsa_indexer.bind import IndexerBinding
from atom.mono.dsa_indexer.plan import IndexerShape


def shape(page=128):
    return IndexerShape(32, 128, 2048, 2048, "bf16", "fp8", page, 64)


@pytest.mark.parametrize("page", [16, 128])
@pytest.mark.parametrize("context", [8192, 163840])
def test_glm_dual_bucket_properties(page, context):
    binding = IndexerBinding({}, shape(page), 8, context, dual_score=True)
    for rows in range(1, 257):
        key = binding.key(rows)
        dual = rows in (128, 160, 192, 224, 256)
        assert key.score_blocks == (512 if dual else 256)
        assert key.split_q == dual
        assert key.reuse_q_scratch == dual
        assert not kernel.unfit(key)
        expected_radix = select.DEFAULT_RADIX
        if dual and context <= 16384:
            expected_radix = select.SHORT_WIDE_RADIX
        elif dual and rows > 128:
            expected_radix = select.WIDE_RADIX
        assert key.radix == expected_radix
    key = binding.key(128)
    assert kernel.scratch_layout(key)["qrdy"][1] == 512 * 8
    assert kernel.scratch_layout(replace(key, score_blocks=256))["qrdy"][1] == 256 * 8
    assert not binding.key(64, 2).split_q
    assert IndexerBinding({}, shape(page), 8, context).key(128).score_blocks == 256


@pytest.mark.parametrize("page", [16, 128])
def test_glm_shape_guard_and_unfit(page):
    good = shape(page)
    assert kernel.dual_score_shape(good)
    for bad in (
        replace(good, q_lora=1536),
        replace(good, rope_dim=0),
        replace(good, rope_neox=True),
        replace(good, block_rows=64),
        replace(good, kv_dtype="fp4"),
        replace(good, wq_dtype="fp8"),
        replace(good, heads=64),
        replace(good, topk=1024),
    ):
        assert not kernel.dual_score_shape(bad)
    base = kernel.IxBuild(good, 128, 8, 163840, score_blocks=512)
    assert not kernel.unfit(base)
    for bad in (
        replace(base, tokens=64),
        replace(base, tokens=129),
        replace(base, world=4),
        replace(base, qlen=2),
        replace(base, replicated=True),
        replace(base, score_blocks=384),
    ):
        assert kernel.unfit(bad)


@pytest.mark.parametrize("context", [8192, 163840])
@pytest.mark.parametrize("capacity", [128, 129, 160, 161, 192, 193, 224, 225, 255, 256])
def test_glm_runner_sizes_every_reachable_layout(monkeypatch, context, capacity):
    import torch

    module = SimpleNamespace(wq_b=SimpleNamespace(weight=torch.empty(1)))
    binding = IndexerBinding({0: module}, shape(), 8, context, dual_score=True)
    monkeypatch.setattr(runner, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(runner, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(runner, "bind_indexers", lambda *args: binding)
    monkeypatch.setattr(
        runner,
        "get_current_atom_config",
        lambda: SimpleNamespace(max_num_seqs=capacity),
    )
    instance = runner.IndexerRunner(None, None, peer_offset=4096)
    for rows in range(1, capacity + 1):
        key = binding.key(rows)
        assert instance.scratch_bytes >= kernel.scratch_bytes(key)
        assert instance.plain_bytes >= kernel.plain_bytes(key)
        assert instance.peer_bytes >= 4096 + kernel.peer_bytes(key)

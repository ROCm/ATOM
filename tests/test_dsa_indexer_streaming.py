# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Streaming-score admission and the dual-resident-grid safety limits."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

pytest.importorskip("flydsl")

from atom.mono.dsa_indexer.bind import IndexerBinding
from atom.mono.dsa_indexer.kernel import (
    IxBuild,
    dual_score_device,
    dual_score_resources_fit,
    unfit,
)
from atom.mono.dsa_indexer.plan import IndexerShape


def key(rows=128, **kwargs):
    shape = IndexerShape(32, 128, 2048, 1536, "bf16", "fp8", 128, 0)
    return IxBuild(shape, rows, 8, 163840, **kwargs)


def test_streaming_only_changes_mid_size_dcp_plain_decode():
    for rows in range(1, 257):
        assert key(rows).stream_scores == (64 <= rows <= 128)
    assert not key(64, qlen=2).stream_scores
    assert not key(64, replicated=True).stream_scores
    assert not replace(key(64), world=1).stream_scores


def test_dual_score_grid_admission():
    for rows in (128, 160, 192, 224, 256):
        assert not unfit(key(rows, score_blocks=512))
    for other in (
        key(64, score_blocks=512),
        key(129, score_blocks=512),
        key(144, score_blocks=512),
        key(score_blocks=384),
        key(score_blocks=512, replicated=True),
        key(score_blocks=512, qlen=2),
        replace(key(score_blocks=512), world=4),
        replace(key(score_blocks=512), shape=replace(key().shape, q_lora=2048)),
        replace(key(score_blocks=512), shape=replace(key().shape, rope_dim=64)),
    ):
        assert unfit(other)


def test_dual_grid_does_not_change_data_or_mailbox_layouts():
    from atom.mono.dsa_indexer import kernel

    for layout in (kernel.scratch_layout, kernel.plain_layout, kernel.peer_layout):
        assert layout(key()) == layout(key(score_blocks=512))


def test_binding_uses_extra_ctas_only_at_the_tested_width():
    binding = IndexerBinding({}, key().shape, 8, 163840, dual_score=True)
    for rows in (1, 64, 80, 96, 112, 127, 128, 129, 160, 256):
        assert binding.key(rows).score_blocks == (
            512 if rows in (128, 160, 256) else 256
        )
    assert binding.key(128, 2).score_blocks == 256
    assert IndexerBinding({}, key().shape, 8, 163840).key(128).score_blocks == 256


def test_wide_q_split_and_radix_are_isolated_to_dual_buckets():
    from atom.mono.dsa_indexer import select

    binding = IndexerBinding({}, key().shape, 8, 8192, dual_score=True)
    assert binding.key(128).score_blocks == 256
    for rows in range(1, 257):
        built = binding.key(rows)
        wide = rows in (160, 192, 224, 256)
        assert built.split_q == wide
        assert built.stream_scores == (64 <= rows <= 128 or wide)
        assert built.radix == (
            select.SHORT_WIDE_RADIX if wide else select.DEFAULT_RADIX
        )
        if wide:
            assert replace(built, max_ctx=163840).radix == select.WIDE_RADIX
            assert replace(built, score_blocks=256).radix == select.DEFAULT_RADIX
        assert built.timeline_points == 7 + 3 * built.radix.passes + 4


@pytest.mark.parametrize("context", [8192, 163840])
@pytest.mark.parametrize("capacity", [128, 129, 160, 161, 192, 193, 224, 225, 255, 256])
def test_runner_covers_nonmonotonic_radix_layouts(monkeypatch, context, capacity):
    import torch

    from atom.mono.dsa_indexer import kernel, runner

    module = SimpleNamespace(wq_b=SimpleNamespace(weight=torch.empty(1)))
    binding = IndexerBinding({0: module}, key().shape, 8, context, dual_score=True)
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
        built = binding.key(rows)
        assert instance.scratch_bytes >= kernel.scratch_bytes(built)
        assert instance.plain_bytes >= kernel.plain_bytes(built)
        assert instance.peer_bytes >= 4096 + kernel.peer_bytes(built)
    if capacity == 256:
        assert kernel.peer_bytes(binding.key(255)) > kernel.peer_bytes(binding.key(256))


def test_radix_partitions_preserve_all_32_ordering_bits():
    import random

    from atom.mono.dsa_indexer import select

    rng = random.Random(1701)
    values = [0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF]
    values += [rng.getrandbits(32) for _ in range(4096)]
    for radix in (select.DEFAULT_RADIX, select.WIDE_RADIX, select.SHORT_WIDE_RADIX):
        assert sum(radix.bits) == 32
        digits = lambda v, radix=radix: tuple(
            select.digit(v, p, radix) for p in range(radix.passes)
        )
        assert sorted(values, key=digits) == sorted(values)
        for value in values:
            prefix = 0
            for p, bits in enumerate(radix.bits):
                assert select.under(value, p, prefix, radix)
                prefix = (prefix << bits) | select.digit(value, p, radix)
            assert prefix == value


@pytest.mark.parametrize(
    "vgpr,lds,scratch,expected",
    [
        (119, 63840, 0, True),
        (128, 81920, 0, True),
        (129, 63840, 0, False),
        (119, 81921, 0, False),
        (119, 63840, 4, False),
    ],
)
def test_dual_cta_compiled_resource_limits(vgpr, lds, scratch, expected):
    assert (
        dual_score_resources_fit([{"vgpr": vgpr, "lds": lds, "scratch": scratch}])
        == expected
    )
    assert not dual_score_resources_fit([])


@pytest.mark.parametrize(
    "arch,cus,expected",
    [
        ("gfx950:sramecc+:xnack-", 256, True),
        ("gfx942", 256, False),
        ("gfx950", 255, False),
        ("gfx950", 304, False),
    ],
)
def test_dual_grid_device_guard(monkeypatch, arch, cus, expected):
    import torch

    for name in ("ROC_GLOBAL_CU_MASK", "HSA_CU_MASK", "HIP_CU_MASK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(gcnArchName=arch, multi_processor_count=cus),
    )
    assert dual_score_device() == expected
    monkeypatch.setenv("ROC_GLOBAL_CU_MASK", "0:0-127")
    assert not dual_score_device()

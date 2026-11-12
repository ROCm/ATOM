# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Indexer admission, buffer capacity, and persistent-grid safety boundaries."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("flydsl")

from atom.mono.dsa_indexer import kernel, runner
from atom.mono.dsa_indexer.bind import IndexerBinding
from atom.mono.dsa_indexer.plan import IndexerShape


def _shape(q_lora=2048, cache="fp8"):
    return IndexerShape(
        32,
        128,
        2048,
        q_lora,
        "bf16",
        cache,
        64 if cache == "fp4" else 128,
        64 if q_lora == 2048 else 0,
        cache_layout="mfma" if cache == "fp4" else "preshuffle",
    )


@pytest.mark.parametrize("replicated", [False, True])
@pytest.mark.parametrize("cache", ["fp8", "fp4"])
def test_supported_widths_and_query_lengths(replicated, cache):
    key = kernel.IxBuild(_shape(cache=cache), 1, 8, 163840, replicated)
    for rows in range(1, 257):
        assert not kernel.unfit(replace(key, tokens=rows))
    for rows in range(6, 253, 6):
        assert not kernel.unfit(replace(key, tokens=rows, qlen=6))
    assert not kernel.unfit(replace(key, tokens=64, qlen=8))
    for rows, qlen in ((0, 1), (257, 1), (64, 6), (256, 6), (258, 6), (72, 8)):
        assert kernel.unfit(replace(key, tokens=rows, qlen=qlen))
    factor = 1 if replicated else key.world
    assert not kernel.unfit(replace(key, tokens=256, max_ctx=factor * ((1 << 22) - 1)))
    assert kernel.unfit(replace(key, tokens=256, max_ctx=factor * (1 << 22)))


def test_short_projection_requires_supported_shapes_and_replicated_cache():
    key = kernel.IxBuild(
        _shape(), 96, 8, 163840, replicated=True, qlen=6, short_gemm=True
    )
    for q_lora in (1024, 1536, 2048):
        assert not kernel.unfit(replace(key, shape=_shape(q_lora)))
    for changes in (
        {"world": 1},
        {"replicated": False},
        {"max_ctx": 262145},
        {"shape": _shape(1280)},
    ):
        assert kernel.unfit(replace(key, **changes))


@pytest.mark.parametrize(
    "q_lora,cache,replicated",
    [
        (1536, "fp8", False),
        (2048, "fp8", False),
        (1536, "fp4", False),
        (1536, "fp4", True),
        (2048, "fp8", True),
    ],
)
@pytest.mark.parametrize("context", [8192, 163840])
@pytest.mark.parametrize(
    "scheduled,cap,expected", [(2, 256, 64), (256, 128, 128), (256, 256, 256)]
)
def test_runner_buffers_cover_every_reachable_layout(
    monkeypatch, q_lora, cache, replicated, context, scheduled, cap, expected
):
    module = SimpleNamespace(wq_b=SimpleNamespace(weight=torch.empty(1)))
    binding = IndexerBinding(
        {0: module},
        _shape(q_lora, cache),
        8,
        context,
        replicated=replicated,
        dual_score=cache == "fp8" and not replicated,
        short_gemm=replicated,
    )
    monkeypatch.setattr(runner, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(runner, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(runner, "bind_indexers", lambda *args: binding)
    monkeypatch.setattr(
        runner, "logits_plane", lambda *_: SimpleNamespace(numel=lambda: 256 * context)
    )
    monkeypatch.setattr(
        runner,
        "get_current_atom_config",
        lambda: SimpleNamespace(max_num_seqs=scheduled),
    )
    instance = runner.IndexerRunner(None, None, max_rows=cap, peer_offset=4096)
    assert instance.rows == expected
    assert instance.serves(expected, 1)
    assert not instance.serves(expected + 1, 1)
    verify_rows = min(252, scheduled * 6)
    assert instance.verify_rows == verify_rows
    assert instance.serves(verify_rows, 6)
    assert not instance.serves(verify_rows + 6, 6)
    # Radix buckets can need less storage than smaller widths.
    keys = [binding.key(r) for r in range(1, expected + 1)]
    keys += [binding.key(r, 6) for r in range(6, verify_rows + 1, 6)]
    assert instance.plain_bytes == max(map(kernel.plain_bytes, keys))
    assert instance.scratch_bytes == max(map(kernel.scratch_bytes, keys))
    assert instance.peer_bytes == 4096 + max(map(kernel.peer_bytes, keys))
    if replicated:
        monkeypatch.setattr(
            runner,
            "logits_plane",
            lambda *_: SimpleNamespace(numel=lambda: 64 * context),
        )
        limited = runner.IndexerRunner(None, None)
        assert limited.rows == 64
        assert limited.verify_rows == min(60, scheduled * 6)
        assert not limited.serves(65, 1)
        assert not limited.serves(66, 6)


@pytest.mark.parametrize("q_lora", [1536, 2048])
def test_dual_grid_rejects_unsupported_launches(q_lora):
    key = kernel.IxBuild(_shape(q_lora), 256, 8, 163840, score_blocks=512)
    assert not kernel.unfit(key)
    for changes in (
        {"tokens": 64},
        {"tokens": 129},
        {"world": 4},
        {"qlen": 2},
        {"replicated": True},
        {"score_blocks": 384},
        {"shape": _shape(q_lora, "fp4")},
    ):
        assert kernel.unfit(replace(key, **changes))


@pytest.mark.parametrize(
    "vgpr,lds,scratch,expected",
    [
        (128, 81920, 0, True),
        (129, 63840, 0, False),
        (119, 81921, 0, False),
        (119, 63840, 4, False),
    ],
)
def test_dual_cta_resource_limits(vgpr, lds, scratch, expected):
    assert (
        kernel.dual_score_resources_fit(
            [{"vgpr": vgpr, "lds": lds, "scratch": scratch}]
        )
        == expected
    )
    assert not kernel.dual_score_resources_fit([])


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
    for name in ("ROC_GLOBAL_CU_MASK", "HSA_CU_MASK", "HIP_CU_MASK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(gcnArchName=arch, multi_processor_count=cus),
    )
    assert kernel.dual_score_device() == expected
    monkeypatch.setenv("ROC_GLOBAL_CU_MASK", "0:0-127")
    assert not kernel.dual_score_device()

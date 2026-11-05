# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Admission boundaries for the shared indexer's narrow and wide planners."""

from dataclasses import replace
from functools import lru_cache
from itertools import accumulate
from types import SimpleNamespace

import pytest

pytest.importorskip("flydsl")

from atom.mono.dsa_indexer.kernel import IxBuild, unfit
from atom.mono.dsa_indexer.plan import IndexerShape


def _key(rows, **kwargs):
    shape = IndexerShape(32, 128, 2048, 2048, "bf16", "fp8", 16, 64)
    return IxBuild(shape, rows, 4, 65536, **kwargs)


def test_dcp_plain_decode_accepts_every_integer_width_through_256():
    for rows in range(1, 257):
        assert not unfit(_key(rows)), rows


@pytest.mark.parametrize("rows", [0, 257, 512])
def test_out_of_range_widths_are_rejected(rows):
    assert "built for 1 .. 256" in unfit(_key(rows))


@pytest.mark.parametrize("rows", [65, 96, 127, 128, 129, 160, 192, 224, 255, 256])
def test_wide_rows_require_supported_qlen_and_a_valid_cache_layout(rows):
    assert not unfit(_key(rows, replicated=True))
    assert not unfit(replace(_key(rows, replicated=True), world=1))
    assert "requires qlen 1 or 6" in unfit(replace(_key(rows), world=1))
    assert "requires qlen 1 or 6" in unfit(_key(rows, qlen=2))
    assert "requires qlen 1 or 6" in unfit(_key(rows, replicated=True, qlen=2))


def test_legacy_replicated_and_uniform_verify_widths_are_unchanged():
    for rows in range(1, 65):
        assert not unfit(_key(rows, replicated=True))
        for qlen in range(1, rows + 1):
            reason = unfit(_key(rows, qlen=qlen))
            assert bool(reason) == bool(rows % qlen)


@pytest.mark.parametrize("replicated", [False, True])
@pytest.mark.parametrize("cache", ["fp8", "fp4"])
def test_qlen6_admits_complete_requests_through_252(replicated, cache):
    shape = replace(
        _key(6).shape,
        kv_dtype=cache,
        block_rows=64,
        cache_layout="mfma" if cache == "fp4" else "preshuffle",
    )
    for rows in range(6, 253, 6):
        assert not unfit(IxBuild(shape, rows, 8, 163840, replicated, qlen=6))
    for rows in (64, 256, 258, 384):
        assert unfit(IxBuild(shape, rows, 8, 163840, replicated, qlen=6))
    assert unfit(IxBuild(shape, 72, 8, 163840, replicated, qlen=8))


@pytest.mark.parametrize("rows", [8, 64, 128, 160, 192, 224, 256])
@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_peer_consumption_acknowledgements_only_use_replicated_storage(rows, world):
    from atom.mono.dsa_indexer import kernel

    key = replace(_key(rows), world=world)
    assert kernel.peer_layout(key)["rdone"][1] == 0
    replica = replace(key, replicated=True)
    assert kernel.peer_layout(replica)["rdone"][1] == (
        rows * world * 8 if world > 1 else 0
    )


@pytest.mark.parametrize("rows", [128, 192, 256])
def test_histogram_count_limit_is_still_enforced_for_wide_steps(rows):
    key = replace(_key(rows), max_ctx=4 * ((1 << 22) - 1))
    assert not unfit(key)
    assert "row histograms count" in unfit(replace(key, max_ctx=4 * (1 << 22)))


@pytest.mark.parametrize(
    "scheduled,cap,expected",
    [
        (2, 256, 64),
        (64, 256, 64),
        (128, 256, 128),
        (160, 256, 160),
        (192, 256, 192),
        (256, 256, 256),
        (512, 256, 256),
        (256, 128, 128),
    ],
)
def test_runner_buffers_follow_reachable_widths(monkeypatch, scheduled, cap, expected):
    import torch

    from atom.mono.dsa_indexer import runner
    from atom.mono.dsa_indexer.bind import IndexerBinding

    module = SimpleNamespace(wq_b=SimpleNamespace(weight=torch.empty(1)))
    binding = IndexerBinding({0: module}, _key(1).shape, 4, 65536)
    monkeypatch.setattr(runner, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(runner, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(runner, "bind_indexers", lambda *args: binding)
    monkeypatch.setattr(
        runner,
        "get_current_atom_config",
        lambda: SimpleNamespace(max_num_seqs=scheduled),
    )
    instance = runner.IndexerRunner(None, None, max_rows=cap)
    assert instance.rows == expected
    assert instance.serves(expected, 1)
    assert not instance.serves(expected + 1, 1)
    verify_rows = min(252, scheduled * 6)
    assert instance.verify_rows == verify_rows
    assert instance.serves(verify_rows, 6)
    assert not instance.serves(verify_rows + 6, 6)
    keys = [binding.key(r) for r in range(1, expected + 1)]
    keys += [binding.key(r, 6) for r in range(6, verify_rows + 1, 6)]
    assert instance.plain_bytes == max(map(runner.kix.plain_bytes, keys))
    assert instance.scratch_bytes == max(map(runner.kix.scratch_bytes, keys))
    assert instance.peer_bytes == max(map(runner.kix.peer_bytes, keys))


@lru_cache(None)
def _planner(rows, world, replicated=False):
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from aiter.ops.flydsl.kernels import buffer_ops as bo
    from flydsl.expr.typing import Int32, Int64

    from atom.mono.device.ops import rsrc
    from atom.mono.dsa_indexer import select

    s = rows

    @fx.struct
    class Smem:
        rows: fx.Array[fx.Int32, 3 * s, 16]
        plan: fx.Array[fx.Int32, 4 + 2 * s, 16]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def planner(ctx: Int64, out: Int64, rank: Int32):
        tid = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(Smem).peek()
        select.step_table(
            {
                "S": rows,
                "shape": _key(rows).shape,
                "tid": tid,
                "lane": tid % 64,
                "rank": rank,
                "world": world,
                "inter": 128 if replicated else 1,
                "own_rows": replicated and world > 1 and rows >= world,
                "qlen": 1,
                "rg": 1,
                "ppr": 1,
                "args": {"ctx": ctx},
                "rows": lds.rows.ptr,
                "plan": lds.plan.ptr,
            }
        )
        for i in range(tid, 3 * rows, 512):
            bo.buffer_store(fx.ptr_load(lds.rows.ptr + i), rsrc(out), i)
        for i in range(tid, 4 + 2 * rows, 512):
            bo.buffer_store(fx.ptr_load(lds.plan.ptr + i), rsrc(out), 3 * rows + i)

    @flyc.jit
    def launch(ctx: Int64, out: Int64, rank: Int32):
        planner(ctx, out, rank).launch(grid=(1,), block=(512,))

    return launch


def _expected_table(contexts, rank, world, replicated=False):
    from atom.mono.dsa_indexer.plan import local_len

    own_rows = replicated and world > 1 and len(contexts) >= world
    total = sum(max(ctx, 0) for ctx in contexts)
    split = [
        not own_rows or (ctx > 262144 and 2 * world * ctx > total) for ctx in contexts
    ]
    n = [
        (
            local_len(max(ctx, 0), rank, world, 128 if replicated else 1)
            if split[r]
            else (max(ctx, 0) if r % world == rank else 0)
        )
        for r, ctx in enumerate(contexts)
    ]
    long = [ctx > 2048 and count > 0 for ctx, count in zip(contexts, n)]
    units = [(count + 127) // 128 if active else 0 for count, active in zip(n, long)]
    chunks = [(count + 4095) // 4096 if active else 0 for count, active in zip(n, long)]
    lg = next(
        (w for w in range(3) if sum((u + (1 << w) - 1) >> w for u in units) <= 256),
        3,
    )
    slg = 0
    if sum(chunks) > 256:
        room = 256 - sum(long)
        slg = next((w for w in range(1, 11) if sum(chunks) <= (room << w)), 10)
        if len(contexts) <= 64:
            slg = min(slg, 8)
    scores = [(u + (1 << lg) - 1) >> lg for u in units]
    segments = [(ch + (1 << slg) - 1) >> slg for ch in chunks]
    assert sum(segments) <= 256
    flags = [
        int(active) | (2 * int(split[r]) if own_rows else 0)
        for r, active in enumerate(long)
    ]
    descriptors = [v for row in zip(contexts, n, flags) for v in row]
    plan = [sum(scores), lg, sum(segments), slg]
    plan += list(accumulate(scores, initial=0))[:-1]
    plan += list(accumulate(segments, initial=0))[:-1]
    return descriptors + plan


@pytest.mark.parametrize(
    "rows", [64, 65, 96, 127, 128, 129, 160, 191, 192, 193, 224, 255, 256]
)
@pytest.mark.parametrize("world", [1, 2, 4, 8])
@pytest.mark.parametrize("replicated", [False, True])
def test_gpu_step_table_matches_cpu_oracle(rows, world, replicated):
    import torch

    if not torch.cuda.is_available() or not torch.version.hip:
        pytest.skip("requires an AMD GPU and FlyDSL")
    maximum = (1 if replicated else world) * ((1 << 22) - 1)
    patterns = [
        [0] * rows,
        [2048] * rows,
        [2049] * rows,
        [65536] * rows,
        [163840] * rows,
        # Upper tile alone must contribute to totals and row prefixes.
        [0] * 64 + [65536] * (rows - 64),
        [maximum] * rows,
        [[0, -1, 1, 2048, 2049, 16385, 65536, maximum][r % 8] for r in range(rows)],
    ]
    for start in (128, 192):
        if rows > start:
            patterns.append([0] * start + [163840] * (rows - start))
    if replicated:
        patterns.extend(
            [
                [0] * (rows - 1) + [262145],
                [0] * (rows - 1) + [maximum],
                [262144] * rows,
                [262145] * rows,
                [0] * 64 + [maximum] * (rows - 64),
            ]
        )
    launch = _planner(rows, world, replicated)
    out = torch.empty(5 * rows + 4, dtype=torch.int32, device="cuda")
    for rank in range(world):
        for contexts in patterns:
            ctx = torch.tensor(contexts, dtype=torch.int32, device="cuda")
            out.fill_(-123)
            launch(ctx.data_ptr(), out.data_ptr(), rank)
            assert out.tolist() == _expected_table(contexts, rank, world, replicated), (
                rows,
                world,
                rank,
                replicated,
                contexts,
            )


@pytest.mark.parametrize("world", [1, 2, 4, 8])
@pytest.mark.parametrize("context", [8192, 163840, 1048576])
@pytest.mark.parametrize("capacity", [64, 128, 160, 192, 224, 256])
def test_replicated_runner_respects_logits_capacity(
    monkeypatch, world, context, capacity
):
    import torch

    from atom.mono.dsa_indexer import runner
    from atom.mono.dsa_indexer.bind import IndexerBinding

    module = SimpleNamespace(wq_b=SimpleNamespace(weight=torch.empty(1)))
    binding = IndexerBinding(
        {0: module}, _key(1).shape, world, context, replicated=True
    )
    monkeypatch.setattr(runner, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(runner, "get_tensor_model_parallel_world_size", lambda: world)
    monkeypatch.setattr(runner, "bind_indexers", lambda *args: binding)
    monkeypatch.setattr(
        runner,
        "logits_plane",
        lambda *_: SimpleNamespace(numel=lambda: capacity * context),
    )
    monkeypatch.setattr(
        runner, "get_current_atom_config", lambda: SimpleNamespace(max_num_seqs=256)
    )
    instance = runner.IndexerRunner(None, None)
    assert instance.rows == capacity
    assert instance.serves(capacity, 1)
    assert not instance.serves(capacity + 1, 1)
    for rows in range(1, capacity + 1):
        key = binding.key(rows)
        assert key.score_blocks == 256 and not key.split_q
        assert instance.scratch_bytes >= runner.kix.scratch_bytes(key)
        assert instance.plain_bytes >= runner.kix.plain_bytes(key)
        assert instance.peer_bytes >= runner.kix.peer_bytes(key)


@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_replicated_wide_histogram_bound(world):
    key = replace(_key(256, replicated=True), world=world, max_ctx=(1 << 22) - 1)
    assert not unfit(key)
    assert "row histograms count" in unfit(replace(key, max_ctx=1 << 22))

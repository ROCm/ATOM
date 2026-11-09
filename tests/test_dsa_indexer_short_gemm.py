# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host and GPU checks for the graph-local short-query specialization."""

from dataclasses import replace
from functools import lru_cache
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("flydsl")

from atom.mono.dsa_indexer import bind
from atom.mono.dsa_indexer.bind import pack_short_q
from atom.mono.dsa_indexer.kernel import IxBuild, unfit
from atom.mono.dsa_indexer.plan import IndexerShape


def key(rows=96, context=163840):
    shape = IndexerShape(32, 128, 2048, 2048, "bf16", "fp8", 128, 64)
    return IxBuild(shape, rows, 8, context, True, qlen=6, short_gemm=True)


@pytest.mark.parametrize("rank", [1024, 1536, 2048])
def test_packed_q_preserves_every_checkpoint_bit(rank):
    weight = (
        (torch.arange(4096 * rank, dtype=torch.int32) % 251)
        .to(torch.bfloat16)
        .view(4096, rank)
    )
    original = weight.clone()
    packed = pack_short_q(weight)
    restored = packed.permute(0, 2, 4, 1, 3, 5).contiguous().view_as(weight)
    assert torch.equal(restored.view(torch.int16), original.view(torch.int16))
    assert torch.equal(weight, original)
    assert packed.is_contiguous() and packed.dtype == torch.bfloat16
    for head, step, wave, lane, element in (
        (0, 0, 0, 0, 0),
        (17, 19, 4, 37, 3),
        (31, rank // 32 - 1, 7, 63, 7),
    ):
        assert (
            packed[head, step, wave, lane // 16, lane % 16, element]
            == weight[
                head * 128 + wave * 16 + lane % 16, step * 32 + lane // 16 * 8 + element
            ]
        )


@pytest.mark.parametrize(
    "changes", [{"world": 1}, {"replicated": False}, {"max_ctx": 262145}]
)
def test_short_gemm_refuses_other_execution_layouts(changes):
    assert unfit(replace(key(), **changes))


@pytest.mark.parametrize("rank", [1024, 1536, 2048])
def test_short_gemm_requires_a_supported_projection_shape(rank):
    assert not unfit(replace(key(), shape=replace(key().shape, q_lora=rank)))
    assert unfit(replace(key(), shape=replace(key().shape, q_lora=1280)))


@pytest.mark.parametrize("cache", ["fp8", "fp4"])
@pytest.mark.parametrize("dcp", [1, 8])
def test_k3_1_pro_binding_packs_nope_q_only_for_replicated_short_path(
    monkeypatch, cache, dcp
):
    shape = replace(
        key().shape,
        q_lora=1536,
        rope_dim=0,
        kv_dtype=cache,
        block_rows=64 if cache == "fp4" else 128,
        cache_layout="mfma" if cache == "fp4" else "preshuffle",
    )
    module = SimpleNamespace(
        wq_b=SimpleNamespace(weight=torch.zeros(4096, 1536, dtype=torch.bfloat16))
    )
    cfg = SimpleNamespace(
        decode_context_parallel_size=dcp,
        kv_cache_block_size=shape.block_rows,
        max_model_len=163840,
    )
    monkeypatch.setattr(bind, "indexer_modules", lambda inner: {3: module})
    monkeypatch.setattr(bind, "get_current_atom_config", lambda: cfg)
    monkeypatch.setattr(bind, "_refusal", lambda *args: "")
    monkeypatch.setattr(bind, "shape_of", lambda *args: shape)
    monkeypatch.setattr(bind.kix, "dual_score_device", lambda device: True)
    binding = bind.bind_indexers(None, 8, 0)
    assert binding.short_gemm == (dcp == 1)
    assert hasattr(module, "_mono_indexer_wq_packed") == (dcp == 1)
    for rows in (6, 48, 96, 144, 192, 240, 252):
        assert not unfit(binding.key(rows, 6))


@lru_cache(None)
def planner(rows, fast):
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from aiter.ops.flydsl.kernels import buffer_ops as bo
    from flydsl.expr.typing import Int64

    from atom.mono.device.ops import rsrc
    from atom.mono.dsa_indexer import select

    shape = key(rows).shape
    size = rows

    @fx.struct
    class Smem:
        rows: fx.Array[fx.Int32, 3 * size, 16]
        plan: fx.Array[fx.Int32, 4 + 2 * size, 16]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel(ctx: Int64, bids: Int64, out: Int64):
        tid = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(Smem).peek()
        c = {
            "S": rows,
            "shape": shape,
            "tid": tid,
            "lane": tid % 64,
            "rank": fx.Int32(0),
            "world": 8,
            "inter": 128,
            "own_rows": True,
            "short_owned_verify": True,
            "local_short": True,
            "fast_short": fast,
            "short_score": rows == 6,
            "max_ctx": 163840,
            "qlen": 6,
            "rg": 6,
            "ppr": 1,
            "args": {"ctx": ctx, "bids": bids},
            "rows": lds.rows.ptr,
            "plan": lds.plan.ptr,
        }
        select.step_table(c)
        for i in range(tid, 3 * rows, 512):
            bo.buffer_store(fx.ptr_load(lds.rows.ptr + i), rsrc(out), i)
        if tid < rows:
            bo.buffer_store(
                fx.ptr_load(select._seg_firsts(c) + tid), rsrc(out), 3 * rows + tid
            )
        if tid < 2:
            bo.buffer_store(
                fx.ptr_load(lds.plan.ptr + 2 + tid), rsrc(out), 4 * rows + tid
            )

    @flyc.jit
    def launch(ctx: Int64, bids: Int64, out: Int64, stream: fx.Stream):
        kernel(ctx, bids, out).launch(grid=(1,), block=(512,), stream=stream)

    return launch


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip,
    reason="AMD GPU planner check",
)
@pytest.mark.parametrize("batch", [1, 8, 42])
@pytest.mark.parametrize(
    "pattern", ["8k", "4k", "uneven", "boundary", "long", "padded"]
)
def test_fast_planner_matches_rows_and_schedules_one_local_segment(batch, pattern):
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    lengths = [8192] * batch
    if pattern == "4k":
        lengths = [4096] * batch
    elif pattern == "uneven":
        lengths = [[2054, 4097, 8191][i % 3] for i in range(batch)]
    elif pattern == "boundary":
        lengths[-1] = 8193
    elif pattern == "long":
        lengths[-1] = 163840
    elif pattern == "padded":
        lengths[-1] = 0
    ctx = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    bids = torch.arange(batch, device="cuda", dtype=torch.int32).repeat_interleave(6)
    outputs = []
    for fast in (False, True):
        output = torch.full((4 * batch * 6 + 2,), -1, device="cuda", dtype=torch.int32)
        _run_compiled(
            planner(batch * 6, fast),
            ctx.data_ptr(),
            bids.data_ptr(),
            output.data_ptr(),
            torch.cuda.current_stream(),
        )
        outputs.append(output)
    rows = batch * 6
    assert torch.equal(outputs[0][: 3 * rows], outputs[1][: 3 * rows])
    if all(2054 <= length <= 8192 for length in lengths):
        # The generic plan can reserve multiple 4096-key segments, but the
        # local selector executes only segment zero and handles all 8192 keys.
        assert torch.equal(
            outputs[1][3 * rows : 4 * rows],
            torch.arange(rows, device="cuda", dtype=torch.int32),
        )
        assert outputs[1][-2:].tolist() == [rows, 0]
    else:
        assert torch.equal(outputs[0], outputs[1])

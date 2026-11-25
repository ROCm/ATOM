# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""NVFP4 decode replayed from a CUDA graph matches the same step run eagerly.

Two caches start identical. One runs `try_nvfp4_decode` eagerly; the other
replays a graph captured once over padding-only inputs, then fed real rows
through the same fixed buffers, the way ATOM's builder does between replays.
Two steps: the second swaps the requests' batch rows. Needs gfx1250 and AITER's
NVFP4 MLA ops; skipped otherwise.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter.ops.mla_nvfp4")
if (
    not torch.cuda.is_available()
    or "gfx1250" not in torch.cuda.get_device_properties(0).gcnArchName
):
    pytest.skip("AITER's NVFP4 MLA decoder is gfx1250-only", allow_module_level=True)

from aiter.mla import mla_decode_prepare_page_csr

from atom.model_ops.attentions.nvfp4_mla import (
    Nvfp4MlaCache,
    try_nvfp4_decode,
)

DIM, RANK, ROPE, HEADS = 576, 512, 64, 128
ROWS, PAGES, COLS, MAX_REQS = 4, 8, 2, 4
DEV = "cuda"


def _attn(cache):
    return SimpleNamespace(
        nvfp4=cache,
        nvfp4_layer=0,
        num_heads=HEADS,
        num_kv_heads=1,
        kv_lora_rank=RANK,
        qk_rope_head_dim=ROPE,
        scale=DIM**-0.5,
        _q_scale=None,
        # RoPE is outside what this test checks; identity keeps both sides equal.
        rotary_emb=lambda positions, q, k: None,
        _v_up_proj_and_o_proj=lambda o: o,
    )


class StepBuffers:
    """The fixed-address inputs a captured decode reads."""

    def __init__(self):
        bf16 = {"dtype": torch.bfloat16, "device": DEV}
        self.q_nope = torch.zeros(ROWS, HEADS, RANK, **bf16)
        self.q_rope = torch.zeros(ROWS, HEADS, ROPE, **bf16)
        self.k_nope = torch.zeros(ROWS, RANK, **bf16)
        self.k_rope = torch.zeros(ROWS, 1, ROPE, **bf16)
        self.positions = torch.zeros(ROWS, dtype=torch.long, device=DEV)
        self.slots = torch.full((ROWS,), -1, dtype=torch.long, device=DEV)
        self.tail = torch.zeros(ROWS, dtype=torch.int32, device=DEV)
        self.context_lens = torch.zeros(ROWS, dtype=torch.int32, device=DEV)
        self.block_tables = torch.zeros(ROWS, COLS, dtype=torch.int32, device=DEV)
        self.csr = (
            torch.zeros(ROWS + 1, dtype=torch.int32, device=DEV),
            torch.zeros(ROWS * COLS, dtype=torch.int32, device=DEV),
            torch.zeros(ROWS, dtype=torch.int32, device=DEV),
        )

    def meta(self):
        return SimpleNamespace(
            max_seqlen_q=1,
            nvfp4_tail_slots=self.tail,
            nvfp4_csr=self.csr,
            slot_mapping=self.slots,
        )

    def load(self, step, tail_rows):
        """Copy one step's rows in, then rebuild the CSR (the builder's job)."""
        for name in ("q_nope", "q_rope", "k_nope", "k_rope", "positions", "slots"):
            getattr(self, name).copy_(step[name])
        self.context_lens.copy_(step["context_lens"])
        self.block_tables.copy_(step["block_tables"])
        self.tail.copy_(torch.as_tensor(tail_rows, dtype=torch.int32))
        mla_decode_prepare_page_csr(self.context_lens, self.block_tables, *self.csr)

    def run(self, attn):
        return try_nvfp4_decode(
            attn,
            self.q_nope,
            self.q_rope,
            self.k_nope,
            self.k_rope,
            self.positions,
            self.meta(),
        )


def _step(gen, reqs):
    """`reqs` is ``[(page, position)]`` per live row; the rest are padding."""

    def rand(*shape):
        return (torch.randn(*shape, generator=gen, device=DEV) * 0.3).to(torch.bfloat16)

    step = {
        "q_nope": rand(ROWS, HEADS, RANK),
        "q_rope": rand(ROWS, HEADS, ROPE),
        "k_nope": rand(ROWS, RANK),
        "k_rope": rand(ROWS, 1, ROPE),
        "positions": torch.zeros(ROWS, dtype=torch.long, device=DEV),
        "slots": torch.full((ROWS,), -1, dtype=torch.long, device=DEV),
        "context_lens": torch.zeros(ROWS, dtype=torch.int32, device=DEV),
        "block_tables": torch.zeros(ROWS, COLS, dtype=torch.int32, device=DEV),
    }
    for row, (page, pos) in enumerate(reqs):
        step["positions"][row] = pos
        step["slots"][row] = page * 64 + pos
        step["context_lens"][row] = pos + 1
        step["block_tables"][row, 0] = page
    return step


@pytest.mark.parametrize("fused", [True, False])
def test_graph_replay_matches_eager_across_a_row_swap(fused):
    eager_cache = Nvfp4MlaCache(1, PAGES, MAX_REQS, DIM, device=DEV, fused=fused)
    graph_cache = Nvfp4MlaCache(1, PAGES, MAX_REQS, DIM, device=DEV, fused=fused)
    for cache in (eager_cache, graph_cache):
        cache.kv_nope.zero_()
        cache.kv_aux.zero_()
        cache.params(DIM**-0.5)

    eager_bufs, graph_bufs = StepBuffers(), StepBuffers()
    graph_attn = _attn(graph_cache)

    # Capture over padding only, as ATOM does: no page is touched.
    pad = [graph_cache.pad_row] * ROWS
    graph_bufs.load(_step(torch.Generator(DEV).manual_seed(0), []), pad)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        graph_bufs.run(graph_attn)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = graph_bufs.run(graph_attn)
    assert graph_cache.kv_nope.abs().sum() == 0

    gen = torch.Generator(DEV).manual_seed(1234)
    # Step 1: request 11 on page 2, request 22 on page 5, then two padding rows.
    # Step 2: same requests, one token later, in swapped batch rows.
    schedule = [
        ([11, 22], [(2, 0), (5, 0)]),
        ([22, 11], [(5, 1), (2, 1)]),
    ]
    for ids, reqs in schedule:
        step = _step(gen, reqs)
        eager_bufs.load(step, eager_cache.bind_rows(ids, pad_to=ROWS))
        eager_out = eager_bufs.run(_attn(eager_cache)).clone()
        graph_bufs.load(step, graph_cache.bind_rows(ids, pad_to=ROWS))
        graph.replay()
        torch.cuda.synchronize()

        assert torch.equal(graph_out, eager_out)
        assert torch.isfinite(eager_out).all()
        # Padding rows have no context: the shader returns zero for them.
        assert eager_out[len(ids) :].abs().sum() == 0
        assert eager_out[: len(ids)].abs().sum() > 0
        assert torch.equal(graph_cache.kv_nope, eager_cache.kv_nope)
        assert torch.equal(graph_cache.kv_aux, eager_cache.kv_aux)
        live = slice(0, MAX_REQS)
        assert torch.equal(graph_cache.ring.buf[live], eager_cache.ring.buf[live])

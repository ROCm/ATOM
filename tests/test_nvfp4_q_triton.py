# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The Triton Q quantization matches mla_nvfp4_quantize_q byte for byte."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _query(batch, seed, strided=False):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    ch = torch.exp(torch.randn(576, generator=gen, device="cuda") * 1.5)
    head = torch.exp(torch.randn(1, 128, 1, generator=gen, device="cuda"))
    q = (
        torch.randn(batch, 128, 576, generator=gen, device="cuda") * ch * head * 0.1
    ).to(torch.bfloat16)
    q[torch.rand(q.shape, generator=gen, device="cuda") < 0.02] = 0
    q[torch.rand(q.shape, generator=gen, device="cuda") < 0.005] = -0.0
    q[:, :3, 512:528] = 0  # whole zero blocks
    nope = q[..., :512].contiguous()
    if strided:
        # RoPE as it leaves q_proj: a [B, 128, 64] view of a wider row.
        wide = torch.zeros(batch, 128, 192, dtype=torch.bfloat16, device="cuda")
        wide[..., 128:] = q[..., 512:]
        rope = wide[..., 128:]
    else:
        rope = q[..., 512:].contiguous()
    return q, nope, rope


@pytest.mark.parametrize(
    "batch,seed,strided", [(1, 0, False), (3, 1, True), (77, 2, False), (512, 3, True)]
)
def test_quantize_q_matches_the_aiter_producer(batch, seed, strided):
    pytest.importorskip("aiter.ops.mla_nvfp4")
    from aiter.ops.mla_nvfp4 import mla_nvfp4_quantize_q

    from atom.model_ops.attentions.nvfp4_q_triton import quantize_q

    q, nope, rope = _query(batch, seed, strided)
    want_n, want_a = mla_nvfp4_quantize_q(q.contiguous())
    got_n, got_a = quantize_q(nope, rope)
    torch.cuda.synchronize()
    bad = (got_n != want_n).nonzero()
    assert bad.numel() == 0, f"q_nope differs at {bad[:8].tolist()}"
    bad = (got_a != want_a).nonzero()
    assert bad.numel() == 0, f"q_aux differs at {bad[:8].tolist()}"


def test_quantize_q_replays_in_a_graph():
    from atom.model_ops.attentions.nvfp4_q_triton import quantize_q

    _, nope, rope = _query(64, 5)
    eager = [t.clone() for t in quantize_q(nope, rope)]
    static_n, static_r = torch.zeros_like(nope), torch.zeros_like(rope)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = quantize_q(static_n, static_r)
    static_n.copy_(nope)
    static_r.copy_(rope)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out[0], eager[0]) and torch.equal(out[1], eager[1])

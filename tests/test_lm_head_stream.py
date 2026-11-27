# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The streamed LM head (``lm_head_stream.mm``): bit-identical to ``tgemm.mm``
at decode rows on gfx1250, and ``ParallelLMHead.local_logits`` falls back to
``tgemm.mm`` wherever it is not served."""

import pytest
import torch

embed_head = pytest.importorskip(
    "atom.model_ops.embed_head",
    reason="embed_head requires the Triton/AITER runtime",
    exc_type=ImportError,
)
from atom.model_ops import lm_head_stream


def _gfx1250() -> bool:
    return (
        torch.cuda.is_available()
        and "gfx1250" in torch.cuda.get_device_properties(0).gcnArchName
    )


@pytest.mark.skipif(not _gfx1250(), reason="the streamed LM head is a gfx1250 kernel")
@pytest.mark.parametrize("m", [8, 7, 4, 1])
def test_bit_identical_to_tgemm(m):
    from aiter.tuned_gemm import tgemm

    # one TP-4 vocab shard of a 200K vocabulary, hidden 6144
    n, k = 50016, 6144
    g = torch.Generator(device="cuda").manual_seed(m)
    w = (torch.randn(n, k, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    x = (torch.randn(m, k, device="cuda", generator=g) * 4).to(torch.bfloat16)
    ref = tgemm.mm(x, w, None)
    out = torch.empty_like(ref)
    got = lm_head_stream.mm(x, w, None, out=out)
    assert got is out
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16))
    # refused: more rows than a decode tile, a bias, fp16
    x9 = torch.randn(9, k, device="cuda").to(torch.bfloat16)
    assert lm_head_stream.mm(x9, w, None) is None
    assert (
        lm_head_stream.mm(x, w, torch.zeros(n, device="cuda", dtype=torch.bfloat16))
        is None
    )
    assert lm_head_stream.mm(x.half(), w.half(), None) is None
    # a weight shape it was not measured on
    w2 = w[:32000]
    assert lm_head_stream.mm(x, w2, None) is None


@pytest.mark.parametrize("served", [True, False])
def test_local_logits_takes_the_stream_else_tgemm(monkeypatch, served):
    head = embed_head.ParallelLMHead.__new__(embed_head.ParallelLMHead)
    torch.nn.Module.__init__(head)
    head.weight = torch.nn.Parameter(torch.zeros(32, 128), requires_grad=False)
    head.bias = None
    x, out = torch.zeros(2, 128), torch.empty(2, 32)
    calls = []

    def stream_mm(x_, w_, bias=None, out=None):
        calls.append("stream")
        return out if served else None

    def tgemm_mm(x_, w_, bias=None, out=None):
        calls.append("tgemm")
        return out

    monkeypatch.setattr(lm_head_stream, "mm", stream_mm)
    monkeypatch.setattr(embed_head.tgemm, "mm", tgemm_mm)
    assert head.local_logits(x, out=out) is out
    assert calls == (["stream"] if served else ["stream", "tgemm"])

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The MTP draft step's front end folded on the mono path (``DraftFront``):
which rows take it, which predictors it refuses, and on a GPU its streams
against ATOM's ops."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("aiter")

from atom.model_ops.qwen4_exp import hyperconnection as hc
from atom.models.qwen4_exp.mono import front as F

H, HC, VOCAB = 2560, 4, 1024
BF16 = torch.bfloat16


class _Norm(nn.Module):
    """Stands in for ``Qwen4ExpGroupedRMSNorm`` over whole rows."""

    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(width, dtype=BF16), requires_grad=False)
        self.variance_epsilon = eps

    def forward(self, x):
        return hc.gemma_rmsnorm_triton(
            x, self.weight, self.variance_epsilon, None, x.shape[-1]
        )


def _predictor(device="cpu", hidden=H, dtype=BF16, bias=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    p = nn.Module()
    p.hidden_size, p.hc_count = hidden, HC
    p.embed_tokens = nn.Embedding(VOCAB, hidden, dtype=BF16)
    p.pre_fc_norm_embedding = _Norm(hidden)
    p.pre_fc_norm_hidden = _Norm(hidden * HC)
    p.fc_embedding = nn.Linear(hidden, hidden, bias=bias, dtype=dtype)
    p.fc_hidden = nn.Linear(hidden, hidden, bias=bias, dtype=dtype)
    with torch.no_grad():
        for w in (p.embed_tokens.weight, p.fc_embedding.weight, p.fc_hidden.weight):
            w.copy_(torch.randn(w.shape, generator=g) * hidden**-0.5)
        for n in (p.pre_fc_norm_embedding, p.pre_fc_norm_hidden):
            n.weight.copy_(torch.randn(n.weight.shape, generator=g) * 0.1)
    # as a loaded model's: the ops' out= adds refuse operands that need grad
    return p.requires_grad_(False).to(device)


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"dtype": torch.float32}, "bias-free bf16"),
        ({"bias": True}, "bias-free bf16"),
        ({"hidden": 2560 + 32}, "not a multiple"),
    ],
)
def test_predictors_the_front_does_not_take_are_refused(kwargs, reason):
    assert reason in F.refusal(_predictor(**kwargs))


def test_the_drafters_front_is_taken():
    assert F.refusal(_predictor()) is None


@pytest.mark.parametrize("rows", [1, F.DraftFront.MAX_ROWS, F.DraftFront.MAX_ROWS + 1])
def test_the_front_takes_its_rows_and_the_ops_the_rest(rows):
    calls = []

    class Front:
        MAX_ROWS = F.DraftFront.MAX_ROWS

        def __call__(self, ids, hidden, out):
            calls.append(ids.numel())
            return out

    p = SimpleNamespace(
        hc_count=HC,
        hidden_size=8,
        embed_tokens=lambda ids: torch.ones(ids.numel(), 8),
        pre_fc_norm_embedding=lambda x: x,
        fc_embedding=lambda x: x * 2,
        pre_fc_norm_hidden=lambda x: x,
        fc_hidden=lambda x: x,
    )
    ids = torch.zeros(rows, dtype=torch.long)
    hidden = torch.ones(rows, HC * 8)
    out = torch.zeros(rows, HC * 8)
    assert F.draft_front(Front(), p, ids, hidden, out) is out
    if rows <= F.DraftFront.MAX_ROWS:
        assert calls == [rows] and (out == 0).all()
    else:
        assert calls == [] and (out == 3).all()
    # without a front every row count takes the ops
    out.zero_()
    F.draft_front(None, p, ids, hidden, out)
    assert (out == 3).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 8])
def test_the_front_is_the_ops_but_for_summation_order(rows):
    p = _predictor("cuda", seed=rows)
    front = F.DraftFront(p)
    g = torch.Generator().manual_seed(100 + rows)
    ids = torch.randint(0, VOCAB, (rows,), generator=g).cuda()
    hidden = torch.randn(rows, HC * H, generator=g).to(BF16).cuda()
    got = torch.empty(rows, HC * H, dtype=BF16, device="cuda")
    ref = torch.empty_like(got)
    assert front(ids, hidden, got) is got
    F.draft_front(None, p, ids, hidden, ref)
    got, ref = got.float().view(rows * HC, H), ref.float().view(rows * HC, H)
    cosine = torch.nn.functional.cosine_similarity(got, ref, dim=-1)
    assert cosine.min() > 0.9999
    assert (got - ref).abs().max() <= ref.abs().max() * 2**-6

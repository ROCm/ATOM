# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The Qwen3.8-Flash-Next mono's MXFP4 lm_head: the exact packing, which heads
it takes, and which calls it serves. The kernel test needs a GPU: Triton's
interpreter reads a bf16 ``dot_scaled`` operand through its fp8 path, so it does
not reproduce the product."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("aiter")

from atom.models.qwen4_exp.mono import head as fp4

N, K = 256, 512


def _mxfp4_weight(n=N, k=K, seed=0):
    """A bf16 weight of MXFP4 values, with its codes and scales: every block of
    32 holds a +-6, so its scale is the one the packing picks."""
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (n, k), generator=g, dtype=torch.uint8)
    codes[codes == 8] = 0  # -0 is not a value the converter writes
    codes.view(n, k // 32, 32)[:, :, 0] = 7 | (
        torch.randint(0, 2, (n, k // 32), generator=g, dtype=torch.uint8) << 3
    )
    scales = torch.randint(110, 135, (n, k // 32), generator=g, dtype=torch.uint8)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return fp4._unpack(packed, scales), packed, scales


class _Head(nn.Module):
    """Stands in for a TP1 ``ParallelLMHead``."""

    def __init__(self, weight, bias=None, tp_size=1):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = bias
        self.tp_size = tp_size
        self.marker = "head"

    def forward(self, x):
        return ("bf16", x)

    def compute_argmax_token(self, x, *, out):
        return ("bf16-argmax", x)


def test_an_mxfp4_valued_weight_packs_exactly():
    weight, packed, scales = _mxfp4_weight()
    got = fp4.pack_exact(weight, rows=64)
    assert got is not None
    assert torch.equal(got[0], packed)
    assert torch.equal(got[1], scales)


def test_a_weight_off_the_mxfp4_grid_is_not_packed():
    weight = _mxfp4_weight()[0].clone()
    weight[200, 3] = weight[200, 3] * 1.25 + 1e-3
    assert fp4.pack_exact(weight, rows=64) is None


@pytest.mark.parametrize(
    "head, reason",
    [
        (nn.Identity(), "not a bf16"),
        (_Head(torch.zeros(N, K)), "not a bf16"),
        (_Head(torch.zeros(N, K, dtype=torch.bfloat16), bias=torch.zeros(N)), "bias"),
        (_Head(torch.zeros(N, K, dtype=torch.bfloat16), tp_size=2), "vocab-parallel"),
        (_Head(torch.zeros(N + 8, K, dtype=torch.bfloat16)), "tile"),
        (_Head(torch.zeros(N, K + 32, dtype=torch.bfloat16)), "tile"),
    ],
)
def test_heads_the_kernel_does_not_take_are_refused(head, reason):
    assert reason in fp4.head_refusal(head)


def test_a_tp1_bf16_head_is_taken():
    assert fp4.head_refusal(_Head(torch.zeros(N, K, dtype=torch.bfloat16))) is None


def _models(shared=True, weight=None):
    weight = _mxfp4_weight()[0] if weight is None else weight
    model = nn.Module()
    model.lm_head = _Head(weight)
    draft = nn.Module()
    draft.lm_head = model.lm_head if shared else _Head(weight.clone())
    return model, SimpleNamespace(model=draft)


def test_install_swaps_the_shared_head_of_target_and_drafter():
    model, drafter = _models()
    head = model.lm_head
    assert fp4.install_fp4_head(model, drafter)
    assert isinstance(model.lm_head, fp4.Mxfp4LMHead)
    assert drafter.model.lm_head is model.lm_head
    assert model.lm_head.head is head
    # every other attribute is the original head's
    assert model.lm_head.marker == "head"
    assert model.lm_head.weight is head.weight


def test_a_drafter_with_its_own_head_keeps_it():
    model, drafter = _models(shared=False)
    own = drafter.model.lm_head
    assert fp4.install_fp4_head(model, drafter)
    assert isinstance(model.lm_head, fp4.Mxfp4LMHead)
    assert drafter.model.lm_head is own


@pytest.mark.parametrize(
    "drafter", [None, SimpleNamespace(), SimpleNamespace(model=nn.Identity())]
)
def test_install_without_a_drafter_head(drafter):
    model = _models()[0]
    assert fp4.install_fp4_head(model, drafter)
    assert isinstance(model.lm_head, fp4.Mxfp4LMHead)


def test_a_head_off_the_grid_is_kept():
    weight = _mxfp4_weight()[0].clone()
    weight[0, 0] = 0.1
    model, drafter = _models(weight=weight)
    head = model.lm_head
    assert not fp4.install_fp4_head(model, drafter)
    assert model.lm_head is head and drafter.model.lm_head is head


@pytest.fixture
def swapped(monkeypatch):
    model = _models()[0]
    fp4.install_fp4_head(model, None)
    calls = []

    def fake_head(x, weight, scales):
        calls.append(x.shape[0])
        return torch.zeros(x.shape[0], N)

    monkeypatch.setattr(fp4, "mxfp4_head", fake_head)
    return model.lm_head, calls


def _context(monkeypatch, prefill=False, draft=False):
    context = SimpleNamespace(is_prefill=prefill, is_draft=draft)
    monkeypatch.setattr(
        fp4, "get_forward_context", lambda: SimpleNamespace(context=context)
    )


@pytest.mark.parametrize(
    "prefill, draft, rows, mxfp4",
    [
        (False, False, 1, True),
        (False, False, fp4.MAX_HEAD_ROWS, True),
        (False, False, fp4.MAX_HEAD_ROWS + 1, False),
        # a target prefill's head picks each sequence's last row first
        (True, False, 4, False),
        # a draft's rows are its own
        (True, True, 4, True),
        (False, True, 4, True),
    ],
)
def test_forward_routes(swapped, monkeypatch, prefill, draft, rows, mxfp4):
    head, calls = swapped
    _context(monkeypatch, prefill, draft)
    out = head(torch.zeros(rows, K, dtype=torch.bfloat16))
    assert (calls == [rows]) == mxfp4
    assert isinstance(out, tuple) != mxfp4


@pytest.mark.parametrize("rows, mxfp4", [(4, True), (fp4.MAX_HEAD_ROWS + 1, False)])
def test_argmax_routes(swapped, monkeypatch, rows, mxfp4):
    head, _ = swapped
    picks = []

    def fake_argmax(x, weight, scales, keys, out):
        picks.append((x.shape[0], keys is head.argmax_keys))
        return out.fill_(7)

    monkeypatch.setattr(fp4, "mxfp4_argmax", fake_argmax)
    out = torch.zeros(rows, dtype=torch.int32)
    got = head.compute_argmax_token(torch.zeros(rows, K, dtype=torch.bfloat16), out=out)
    if mxfp4:
        assert got is out and (out == 7).all()
        assert picks == [(rows, True)]
    else:
        assert got[0] == "bf16-argmax" and not picks


@pytest.mark.parametrize("rows", [1, 3, 4, 5, 16])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_the_kernel_is_the_bf16_product(rows):
    device = "cuda"
    weight = _mxfp4_weight()[0].to(device)
    packed, scales = fp4.pack_exact(weight)
    g = torch.Generator().manual_seed(rows)
    x = torch.randn(rows, K, generator=g).to(torch.bfloat16).to(device)
    y = fp4.mxfp4_head(x, packed, scales)
    ref = x.float() @ weight.float().T
    assert y.dtype == torch.bfloat16 and y.shape == (rows, N)
    # one bf16 rounding of an fp32 sum: the order differs from the reference's
    assert (y.float() - ref).abs().max() <= ref.abs().max() * 2**-8


@pytest.mark.parametrize("rows", [1, 3, 4, 5, 16])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_the_argmax_is_the_logits_lowest_largest(rows):
    device = "cuda"
    weight = _mxfp4_weight()[0].to(device)
    packed, scales = fp4.pack_exact(weight)
    keys = fp4.argmax_keys(packed)
    g = torch.Generator().manual_seed(rows)
    x = torch.randn(rows, K, generator=g).to(torch.bfloat16).to(device)
    out = torch.full((rows,), -1, dtype=torch.int32, device=device)
    for tie in (False, True):
        if tie:
            # the first row's winner copied to a lower and a higher column
            j = int(fp4.mxfp4_head(x, packed, scales)[0].float().argmax())
            for i in {0, j // 2, N - 1} - {j}:
                packed[i], scales[i] = packed[j], scales[j]
        logits = fp4.mxfp4_head(x, packed, scales)
        assert fp4.mxfp4_argmax(x, packed, scales, keys, out) is out
        # torch.argmax: the first of the largest
        assert torch.equal(out, logits.float().argmax(-1).to(torch.int32))

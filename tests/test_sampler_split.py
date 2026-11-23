# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM_SAMPLER_SPLIT_V: ``sampler_split.split_sample`` against a float64 torch
Gumbel-max reference, on a GPU (the Triton interpreter cannot run the kernels:
``tl.argmax``'s tie-break combine is itself a jitted function, which it refuses).

Sampled rows: argmax of ``softmax(x / max(T, 1e-5)) / (u + eps)``, lowest index
on ties; rows at or below ``greedy_eps``: the lowest-index argmax of the raw
logits (``topk_select(..., tie="low")``). The aiter-kernel comparison on gfx1250
(same noise, 1-256 rows of the 200064 vocabulary) is a GPU script, see the
track C4D notes."""

import importlib
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

EPS = 1e-10


@pytest.fixture(scope="module")
def split_mod():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU (see the module docstring)")
    sys.modules.pop("atom.model_ops.sampler_split", None)
    return importlib.import_module("atom.model_ops.sampler_split")


def _reference(logits, expo, temps):
    x = logits.double()
    t = temps.double().clamp_min(1e-5).unsqueeze(1)
    p = torch.softmax(x / t, dim=1)
    q = p / (expo.double() + EPS)
    out = []
    for r in range(x.shape[0]):
        row = x[r] if temps[r] <= EPS else q[r]
        best = row.max()
        out.append(int((row == best).nonzero()[0]))
    return torch.tensor(out, dtype=torch.int32)


@pytest.mark.parametrize("M,V,block", [(1, 300, 128), (3, 1000, 256), (5, 4100, 1024)])
@pytest.mark.parametrize("shared", [False, True])
def test_split_sample_matches_reference(split_mod, M, V, block, shared):
    dev = "cuda"
    g = torch.Generator(device="cpu").manual_seed(M * 1000 + V)
    logits = (torch.randn((M, V), generator=g) * 3).to(torch.bfloat16)
    # exact ties of the row max: the greedy rows must take the lowest index
    mx = logits.float().max(dim=1).values.to(torch.bfloat16)
    logits[:, V // 2] = mx
    logits[:, V - 1] = mx
    temps = torch.tensor(
        [[1.0, EPS, 0.7][i % 3] for i in range(M)], dtype=torch.float32
    )
    if shared:
        expo = torch.empty((1, V)).exponential_(1, generator=g).expand(M, V)
    else:
        expo = torch.empty((M, V)).exponential_(1, generator=g)
    got = split_mod.split_sample(
        logits.to(dev),
        expo.to(dev),
        temps.to(dev),
        eps=EPS,
        greedy_eps=EPS,
        block=block,
    ).cpu()
    assert got.dtype == torch.int32
    assert torch.equal(got, _reference(logits, expo, temps))


def test_split_sample_empty(split_mod):
    dev = "cuda"
    out = split_mod.split_sample(
        torch.empty((0, 64), dtype=torch.bfloat16, device=dev),
        torch.empty((0, 64), device=dev),
        torch.empty(0, device=dev),
        eps=EPS,
        greedy_eps=EPS,
    )
    assert out.shape == (0,)

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The one-launch DSpark Markov chain serves only its shape (one request,
K <= 8, bf16 rank-256 tables, gfx1250) and declines the rest without building
its kernel, so those steps keep the two-stage path."""

import pytest
import torch

from atom.model_ops import dspark_markov_chain as chain

V, K = 64, 7


def _args(bs=1, k=K, dtype=torch.bfloat16, rank=chain.RANK):
    base = torch.zeros(bs, k, V, dtype=dtype)
    ids = torch.zeros(bs, k + 1, dtype=torch.int32)
    w = torch.zeros(V, rank, dtype=torch.bfloat16)
    return base, ids, w, w.clone()


@pytest.mark.parametrize(
    "args",
    [
        _args(),  # CPU tensors: not a gfx1250 device
        _args(bs=2),  # two requests
        _args(k=chain.MAX_STEPS + 1),  # more steps than the launch holds
        _args(dtype=torch.float32),  # fp32 logits
        _args(rank=128),  # tables of another rank
    ],
)
def test_declines_outside_its_shape_without_building(monkeypatch, args):
    def no_build():
        raise AssertionError("built the kernel for a shape it does not serve")

    monkeypatch.setattr(chain, "_load", no_build)
    assert chain.usable(*args) is False
    assert chain.markov_chain_bs1(*args) is None


def test_poll_errors_is_free_before_any_launch(monkeypatch):
    monkeypatch.setattr(chain, "_state", {})
    monkeypatch.setattr(chain, "_calls", 0)
    for _ in range(1024):
        chain.poll_errors()  # no state: never reads an error word

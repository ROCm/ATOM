# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The one-launch DSpark Markov chain serves only its shape (one request,
K <= 8, bf16 rank-256 tables, gfx1250) and declines the rest without asking
aiter, so those steps keep the two-stage path; an aiter without the op keeps
it too; a bounded wait that gave up raises at the next check."""

import sys
from types import SimpleNamespace

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
def test_declines_outside_its_shape_without_aiter(monkeypatch, args):
    def no_aiter():
        raise AssertionError("asked aiter about a shape it does not serve")

    monkeypatch.setattr(chain, "_aiter_op", no_aiter)
    assert chain.usable(*args) is False
    assert chain.markov_chain_bs1(*args) is None


def test_an_aiter_without_the_op_keeps_the_two_stage_path(monkeypatch):
    # A None entry in sys.modules makes the import raise ImportError.
    monkeypatch.setitem(sys.modules, "aiter.ops.markov_argmax_chain", None)
    monkeypatch.setattr(chain, "_op", chain._UNSET)
    assert chain._aiter_op() is None


def test_check_errors_is_free_before_any_launch(monkeypatch):
    monkeypatch.setattr(chain, "_state", {})
    monkeypatch.setattr(chain, "_calls", 0)
    for _ in range(1024):
        chain.check_errors()  # no state: never reads an error word


def test_a_wait_that_gave_up_raises(monkeypatch):
    """A set error word (here: step 2 waited out) stops the step instead of
    letting it draft on."""
    copied = SimpleNamespace(query=lambda: True)
    err_host = torch.tensor([1 << 2], dtype=torch.int32)
    state = SimpleNamespace(err_host=err_host, err_copied=copied)
    monkeypatch.setattr(chain, "_state", {0: state})
    monkeypatch.setattr(chain, "_calls", 0)
    with pytest.raises(RuntimeError, match="steps 0x4"):
        for _ in range(512):
            chain.check_errors()

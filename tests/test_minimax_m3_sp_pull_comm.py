# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""SP prefill hand-offs over the pull collectives: staging, sizes and fallback.

A fake pull object (a byte buffer on the CPU) stands in for aiter's
``PullCollective``; the real ops are checked in aiter's own tests and in the GPU
parity check of the PR description.
"""

import pytest
import torch

pytest.importorskip("aiter")

from atom.models.minimax_m3 import sp_prefill


class _FakePull:
    def __init__(self, max_bytes):
        self.max_bytes = max_bytes
        self.storage = torch.zeros(max_bytes, dtype=torch.uint8)
        self.calls = []

    def acquire(self):
        self.calls.append(("acquire",))

    def buffer(self, shape, dtype, byte_offset=0):
        n = 1
        for s in shape:
            n *= s
        nbytes = n * dtype.itemsize
        return self.storage[byte_offset : byte_offset + nbytes].view(dtype).view(*shape)

    def all_gather(self, fields):
        self.calls.append(("all_gather", [(o, n) for o, n, _ in fields]))

    def all_to_all(self, fields):
        self.calls.append(("all_to_all", [(o, n) for o, n, _ in fields]))

    def reduce_scatter(self, out, numel):
        self.calls.append(("reduce_scatter", numel))

    def all_to_all_mxfp8(self, out_c, out_s, rows, d):
        self.calls.append(("a2a_mxfp8", rows, d))


class _Rccl:
    def __init__(self):
        self.calls = []

    def all_gather(self, outs, ins):
        self.calls.append("all_gather")

    def all_to_all(self, outs, ins):
        self.calls.append("all_to_all")

    def reduce_scatter(self, out, x):
        self.calls.append("reduce_scatter")


def _coll(max_bytes=1 << 20, world=4, staged=True):
    pull, rccl = _FakePull(max_bytes), _Rccl()
    return sp_prefill._PullColl(pull, rccl, world, staged=staged), pull, rccl


def test_default_is_rccl(monkeypatch):
    monkeypatch.setattr(sp_prefill, "_pull_state", {})
    monkeypatch.delenv("ATOM_M3_SP_COMM", raising=False)
    assert sp_prefill._pull() is None


def test_all_gather_stages_fields_back_to_back():
    coll, pull, rccl = _coll()
    a = torch.arange(64 * 16, dtype=torch.uint8).view(64, 16)
    b = torch.arange(64 * 8, dtype=torch.uint8).view(64, 8)
    outs = [
        torch.empty(256, 16, dtype=torch.uint8),
        torch.empty(256, 8, dtype=torch.uint8),
    ]
    coll.all_gather(outs, [a, b])
    assert pull.calls[-1] == ("all_gather", [(0, 1024), (1024, 512)])
    assert torch.equal(pull.storage[:1024].view(64, 16), a)
    assert torch.equal(pull.storage[1024:1536].view(64, 8), b)
    assert not rccl.calls


def test_all_gather_too_large_falls_back():
    coll, pull, rccl = _coll(max_bytes=1024)
    a = torch.zeros(64, 32, dtype=torch.uint8)
    coll.all_gather([torch.empty(256, 32, dtype=torch.uint8)], [a])
    assert rccl.calls == ["all_gather"]
    assert not pull.calls


def test_all_to_all_needs_world_multiple():
    coll, _, rccl = _coll(world=4)
    x = torch.zeros(5, 8, dtype=torch.uint8)  # 40 bytes: not a multiple of 8 x world
    coll.all_to_all([torch.empty_like(x)], [x])
    assert rccl.calls == ["all_to_all"]


def test_reduce_scatter_dtype_and_size_fallback():
    coll, pull, rccl = _coll()
    x = torch.zeros(64, 32, dtype=torch.float32)
    coll.reduce_scatter(torch.empty(16, 32, dtype=torch.float32), x)
    assert rccl.calls == ["reduce_scatter"]
    y = torch.zeros(64, 32, dtype=torch.bfloat16)
    coll.reduce_scatter(torch.empty(16, 32, dtype=torch.bfloat16), y)
    assert pull.calls[-1] == ("reduce_scatter", 64 * 32)


def test_a2a_mxfp8_quants_into_the_buffer():
    coll, pull, _ = _coll(max_bytes=1 << 20)
    d, rows, w = 256, 8, 4
    x = torch.zeros(w * rows, d, dtype=torch.bfloat16)
    seen = {}

    def quant(inp, out):
        codes, scales = out
        seen["codes"], seen["scales"] = codes, scales
        codes.view(torch.uint8).fill_(7)
        scales.view(torch.uint8).fill_(3)

    assert coll.can_a2a_mxfp8(w * rows, d)
    x_q, x_s = coll.a2a_mxfp8(quant, x, rows)
    # the quant wrote straight into the pull buffer: codes, then scales
    n = w * rows * d
    assert (pull.storage[:n] == 7).all()
    assert (pull.storage[n : n + n // 32] == 3).all()
    assert pull.calls[-1] == ("a2a_mxfp8", rows, d)
    assert x_q.shape == (rows, w * d) and x_s.shape == (rows, w * d // 32)
    assert not coll.can_a2a_mxfp8(10**6, d)


def test_unstaged_default_keeps_copy_ops_on_rccl():
    coll, pull, rccl = _coll(staged=False)
    a = torch.zeros(64, 16, dtype=torch.uint8)
    coll.all_gather([torch.empty(256, 16, dtype=torch.uint8)], [a])
    coll.all_to_all(
        [torch.empty(256, 16, dtype=torch.uint8)],
        [torch.zeros(256, 16, dtype=torch.uint8)],
    )
    coll.reduce_scatter(
        torch.empty(16, 32, dtype=torch.bfloat16),
        torch.zeros(64, 32, dtype=torch.bfloat16),
    )
    assert rccl.calls == ["all_gather", "all_to_all", "reduce_scatter"]
    assert not pull.calls
    # the zero-copy MXFP8 all-to-all still goes through the pull buffer
    coll.a2a_mxfp8(
        lambda x, out: None, torch.zeros(4 * 8, 256, dtype=torch.bfloat16), 8
    )
    assert pull.calls[-1][0] == "a2a_mxfp8"


def test_reduce_scatter_from_the_pull_buffer_skips_the_stage_copy():
    coll, pull, rccl = _coll(staged=False)
    x = pull.buffer((64, 32), torch.bfloat16)  # what _pull_rs_buffer hands the producer
    coll.reduce_scatter(torch.empty(16, 32, dtype=torch.bfloat16), x)
    assert pull.calls[-1] == ("reduce_scatter", 64 * 32)
    assert not rccl.calls

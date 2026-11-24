# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 indexer CP: the one-launch exchange builds and loads every
variant a decode can reach before serving (CPU; the kernel is stubbed)."""

import pytest
import torch

pytest.importorskip("aiter", reason="requires the AITER Python package")
triton = pytest.importorskip("triton", reason="requires Triton")

from atom.model_ops.minimax_m3 import indexer_candidate_exchange as ex


def _build_key(local, topk):
    cls = 1 if local == 1 else 16 if local % 16 == 0 else 0
    return ex._scan_width(local, topk), cls


@pytest.mark.parametrize("max_local", [1, 7, 16, 300, 2048])
@pytest.mark.parametrize("topk", [16, 64])
def test_variants_cover_every_local_once(max_local, topk):
    picks = ex.exchange_variants(max_local, topk)
    keys = [_build_key(local, topk) for local in picks]
    assert len(set(keys)) == len(keys)
    assert {_build_key(n, topk) for n in range(1, max_local + 1)} == set(keys)
    assert all(1 <= local <= max_local for local in picks)


def test_m31_variant_count():
    # 1M context, TP4: 2048 local blocks, top-16 -> widths 16..512.
    assert len(ex.exchange_variants(2048, 16)) == 13


class _Built:
    def __init__(self, log, kw):
        self.log, self.kw = log, kw

    def _init_handles(self):
        self.log.append(("load", self.kw["QUERY_LEN"], self.kw["LOCAL_BLOCKS"]))


class _Kernel:
    def __init__(self):
        self.log = []
        self.launches = []

    def warmup(self, *args, grid, **kw):
        assert grid == (1,)
        assert kw["BLOCK_SIZE_K"] == ex._scan_width(kw["LOCAL_BLOCKS"], kw["TOPK"])
        self.log.append(("build", kw["QUERY_LEN"], kw["LOCAL_BLOCKS"]))
        return _Built(self.log, kw)

    def __getitem__(self, grid):
        def launch(*args, **kw):
            self.launches.append((grid, kw["QUERY_LEN"], kw["LOCAL_BLOCKS"]))

        return launch


class _Mailbox:
    MAX_TOPK = 64
    MAX_POLLS = 10
    max_rows = 64

    def __init__(self):
        self.peers = torch.zeros(4, dtype=torch.int64)
        self.send_calls = torch.zeros(4 * self.max_rows, dtype=torch.int32)
        self.recv_calls = torch.zeros(self.max_rows, dtype=torch.int32)
        self.timeouts = torch.zeros(1, dtype=torch.int32)

    def fits(self, tokens, topk):
        return tokens <= self.max_rows and topk <= self.MAX_TOPK


@pytest.fixture
def kernel(monkeypatch):
    stub = _Kernel()
    monkeypatch.setattr(ex, "_select_exchange_merge", stub)
    monkeypatch.setattr(ex, "_prewarmed", set())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return stub


def _call(q, batch, context_blocks, table_blocks=8192, mailbox=None):
    local = triton.cdiv(context_blocks, 4)
    scores = torch.zeros((4, batch * q, local), dtype=torch.float32)
    lengths = torch.full((batch,), 128 * context_blocks, dtype=torch.int32)
    table = torch.zeros((batch, table_blocks), dtype=torch.int32)
    return ex.select_exchanged_candidates(
        scores, lengths, table, 16, 2, 4, q, 1, 4, mailbox or _Mailbox()
    )


def test_first_eager_call_builds_and_loads_q_and_q1(kernel):
    _call(q=8, batch=2, context_blocks=100)
    variants = ex.exchange_variants(2048, 16)
    builds = [e for e in kernel.log if e[0] == "build"]
    loads = [e for e in kernel.log if e[0] == "load"]
    expected = {(q, local) for q in (8, 1) for local in variants}
    assert {e[1:] for e in builds} == expected
    assert {e[1:] for e in loads} == expected  # every build is loaded too
    # the call itself still launches once, with its own shape
    assert kernel.launches == [((2 * 8 * 4,), 8, 25)]


def test_prewarm_runs_once_per_geometry(kernel):
    mailbox = _Mailbox()
    _call(q=8, batch=2, context_blocks=100, mailbox=mailbox)
    n = len(kernel.log)
    # the producer's q=1 decodes, other batches and contexts: nothing new
    _call(q=1, batch=3, context_blocks=7000, mailbox=mailbox)
    _call(q=8, batch=1, context_blocks=5, mailbox=mailbox)
    assert len(kernel.log) == n
    # a new query length (a DSpark bucket) builds only its own set
    _call(q=4, batch=2, context_blocks=100, mailbox=mailbox)
    assert {e[1] for e in kernel.log[n:]} == {4}
    assert len(kernel.log) - n == 2 * len(ex.exchange_variants(2048, 16))


def test_no_prewarm_inside_capture(kernel, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    _call(q=8, batch=2, context_blocks=100)
    assert kernel.log == []
    assert len(kernel.launches) == 1


def test_empty_call_still_prewarms(kernel):
    _call(q=1, batch=0, context_blocks=100)
    assert kernel.launches == []
    assert {e[1] for e in kernel.log} == {1}

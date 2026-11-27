# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""`shard_gemm_pad`: the pad is measured per shape, agreed over the group,
cached, and None (keep the replicated head) wherever no pad is exact.

Plain torch on CPU: the fake head's sharded GEMM is off by one ulp unless its
``M`` reaches a threshold, the way a GEMM backend's kernel choice is.
"""

import pytest
import torch

from atom.model_ops import dp_vocab_shard_pad as pad_mod

V, DIM = 16, 8


class _FakeHead:
    """Replicated ``x @ W.T``; the sharded GEMM is exact only at ``M >= exact_m``.

    Integer operands (the probe rows rounded) keep every sum exact, so CPU
    BLAS's own shape-dependent order cannot add a difference of its own.
    """

    def __init__(self, exact_m):
        self.weight = torch.randint(-3, 4, (V, DIM)).float()
        self.num_embeddings = V
        self.exact_m = exact_m
        self.shard_calls = 0

    def local_logits(self, x):
        return x.round() @ self.weight.t()

    def _vocab_shard_logits(self, x, rank, world):
        self.shard_calls += 1
        vs = V // world
        out = x.round() @ self.weight[rank * vs : (rank + 1) * vs].t()
        if self.exact_m is None or x.shape[0] < self.exact_m:
            out = torch.nextafter(out, torch.full_like(out, float("inf")))
        return out


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    monkeypatch.setattr(pad_mod, "_PADS", {})


def test_the_smallest_exact_pad_is_picked_and_cached():
    head = _FakeHead(exact_m=2 * 3 + 16)  # world 2, rows 3 -> needs pad 16
    assert pad_mod.shard_gemm_pad(head, 3, 0, 2, None) == 16
    calls = head.shard_calls
    assert pad_mod.shard_gemm_pad(head, 3, 0, 2, None) == 16
    assert head.shard_calls == calls, "a measured shape is not measured again"


def test_no_exact_pad_keeps_the_replicated_head(caplog):
    head = _FakeHead(exact_m=None)
    assert pad_mod.shard_gemm_pad(head, 3, 1, 2, None) is None
    assert "keeps the replicated head" in caplog.text
    # The verdict is cached too: no second measurement.
    calls = head.shard_calls
    assert pad_mod.shard_gemm_pad(head, 3, 1, 2, None) is None
    assert head.shard_calls == calls


def test_the_pad_is_the_smallest_exact_on_every_rank(monkeypatch):
    """A pad exact here but not on a peer is skipped for the next one."""
    head = _FakeHead(exact_m=2 * 3 + 16)
    seen = {}

    def peers_reject_16(values, group):
        seen["group"] = group
        agreed = list(values)
        agreed[pad_mod.PAD_CANDIDATES.index(16)] = 0
        return tuple(agreed)

    monkeypatch.setattr(pad_mod, "group_min_each", peers_reject_16)
    assert pad_mod.shard_gemm_pad(head, 3, 0, 2, "cpu-group") == 24
    assert seen["group"] == "cpu-group"


def test_an_unmeasured_shape_under_capture_stays_replicated(monkeypatch):
    """No measuring inside a capture, and nothing cached: the next eager
    pass measures it."""
    head = _FakeHead(exact_m=0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert pad_mod.shard_gemm_pad(head, 3, 0, 2, None) is None
    assert head.shard_calls == 0 and not pad_mod._PADS
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert pad_mod.shard_gemm_pad(head, 3, 0, 2, None) == 0


def test_shapes_are_cached_apart():
    head = _FakeHead(exact_m=2 * 4 + 8)
    assert pad_mod.shard_gemm_pad(head, 3, 0, 2, None) == 16
    assert pad_mod.shard_gemm_pad(head, 4, 0, 2, None) == 8
    assert pad_mod.shard_gemm_pad(head, 4, 0, 4, None) == 0


def test_candidates_start_unpadded_and_ascend():
    c = pad_mod.PAD_CANDIDATES
    assert c[0] == 0 and list(c) == sorted(set(c))
    assert max(c) >= 192

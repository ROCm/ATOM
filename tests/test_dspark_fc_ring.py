# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""``DSparkDraftModel.fc_taps`` takes aiter's concat-then-GEMM ring when ``ATOM_DSPARK_FC_RING`` is on.

The kernel is aiter's (numerics and perf are tested there, on a gfx950); what is ATOM's, and tested here with a stand-in
for it, is when the draft takes it: only with the env on, a built shuffled copy, and five bf16 taps of 1..128 rows with
one aligned row stride -- never otherwise. Runs on the CPU-only CI job (no aiter import).
"""

import types

import pytest
import torch

from atom.models import dspark_draft

SEG, N = 64, 64
TAPS = 5


class _Ring:
    def __init__(self, supported=True):
        self.supported = supported
        self.calls, self.warmed = [], []

    def namespace(self):
        return types.SimpleNamespace(
            cat_x_ring_supported=lambda m, n, seg, device=None: self.supported,
            shuffle_weight_ring=lambda w: w.clone(),
            warmup_cat_x_ring=lambda m, n, seg, w: self.warmed.append((m, n, seg)),
            gemm_a16w16_cat_x_ring=self._gemm,
        )

    def _gemm(self, xs, w):
        self.calls.append(xs[0].shape[0])
        return torch.cat(xs, dim=-1) @ w.T


class _Fc(torch.nn.Module):
    def __init__(self, weight, bias=None):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.bias = bias

    def forward(self, x):
        return (
            x.to(self.weight.dtype) @ self.weight.T
        ) + 1.0  # +1: the fallback is recognisable


def _draft(monkeypatch, ring, enabled=True, bias=None, dtype=torch.bfloat16):
    draft = dspark_draft.DSparkDraftModel.__new__(dspark_draft.DSparkDraftModel)
    torch.nn.Module.__init__(draft)
    draft.target_layer_ids = (1, 2, 3, 4, 5)
    draft.fc = _Fc(torch.randn(N, TAPS * SEG).to(dtype), bias)
    monkeypatch.setattr(
        dspark_draft,
        "_fc_ring_module",
        lambda: None if ring is None else ring.namespace(),
    )
    monkeypatch.setattr(
        dspark_draft.envs, "ATOM_DSPARK_FC_RING", enabled, raising=False
    )
    monkeypatch.setattr(dspark_draft, "linear_out", lambda y: y)
    return draft


def _taps(rows, dtype=torch.bfloat16):
    return [torch.randn(rows, SEG).to(dtype) for _ in range(TAPS)]


def test_ring_takes_small_rows_after_load(monkeypatch):
    ring = _Ring()
    draft = _draft(monkeypatch, ring)
    draft.process_weights_after_loading()
    assert ring.warmed == [(128, N, SEG)]
    taps = _taps(7)
    assert torch.equal(
        draft.fc_taps(taps), torch.cat(taps, -1) @ draft._fc_ring_weight.T
    )
    assert ring.calls == [7]


@pytest.mark.parametrize("rows", [129, 1000])
def test_large_rows_keep_fc(monkeypatch, rows):
    ring = _Ring()
    draft = _draft(monkeypatch, ring)
    draft.process_weights_after_loading()
    taps = _taps(rows)
    assert torch.equal(draft.fc_taps(taps), draft.fc(torch.cat(taps, -1)))
    assert ring.calls == []


def test_padded_row_stride_is_taken(monkeypatch):
    ring = _Ring()
    draft = _draft(monkeypatch, ring)
    draft.process_weights_after_loading()
    buf = torch.randn(TAPS, 8, SEG + 8).bfloat16()
    draft.fc_taps([buf[i, :, :SEG] for i in range(TAPS)])
    assert ring.calls == [8]


def test_odd_taps_keep_fc(monkeypatch):
    ring = _Ring()
    draft = _draft(monkeypatch, ring)
    draft.process_weights_after_loading()
    taps = _taps(8)
    for bad in (
        taps[:4],  # not five taps
        [torch.cat(taps, -1)],  # already joined (pipeline-parallel relay)
        [t.float() for t in taps],  # wrong dtype
        [t[:, : SEG - 8] for t in taps],  # width the weight does not span
    ):
        assert not dspark_draft._taps_fit_ring(bad, draft._fc_ring_weight)
    strided = torch.randn(TAPS, SEG, 8).bfloat16().transpose(1, 2)  # column stride 8
    assert not dspark_draft._taps_fit_ring(list(strided), draft._fc_ring_weight)
    assert ring.calls == []


def test_misaligned_row_stride_keeps_fc(monkeypatch):
    ring = _Ring()
    draft = _draft(monkeypatch, ring)
    draft.process_weights_after_loading()
    buf = torch.randn(
        TAPS, 8, SEG + 4
    ).bfloat16()  # 68 * 2 B = 136 B: not 16-byte aligned
    taps = [buf[i, :, :SEG] for i in range(TAPS)]
    assert torch.equal(draft.fc_taps(taps), draft.fc(torch.cat(taps, -1)))
    assert ring.calls == []


@pytest.mark.parametrize(
    "enabled,ring,bias,dtype",
    [
        (False, _Ring(), None, torch.bfloat16),
        (True, None, None, torch.bfloat16),
        (True, _Ring(supported=False), None, torch.bfloat16),
        (True, _Ring(), torch.zeros(N), torch.bfloat16),
        (True, _Ring(), None, torch.float32),
    ],
)
def test_no_ring_weight_unless_everything_fits(monkeypatch, enabled, ring, bias, dtype):
    draft = _draft(monkeypatch, ring, enabled=enabled, bias=bias, dtype=dtype)
    draft.process_weights_after_loading()
    assert draft._fc_ring_weight is None
    taps = _taps(8, dtype)
    assert torch.equal(draft.fc_taps(taps), draft.fc(torch.cat(taps, -1)))


def test_draft_without_fc_is_left_alone(monkeypatch):
    draft = _draft(monkeypatch, _Ring())
    del draft.fc
    draft.process_weights_after_loading()
    assert draft._fc_ring_weight is None


def test_draft_not_tapping_five_layers_is_left_alone(monkeypatch):
    draft = _draft(monkeypatch, _Ring())
    draft.target_layer_ids = (1, 2, 3)
    draft.process_weights_after_loading()
    assert draft._fc_ring_weight is None

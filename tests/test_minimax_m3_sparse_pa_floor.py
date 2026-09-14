# SPDX-License-Identifier: MIT
"""The row floor that routes MiniMax-M3 sparse selection to ASM.

`_gluon_one_pass_max_rows` is where split-KV stops buying parallelism, so it has
to agree with the residency target aiter sizes its own split ladder against --
`multi_processor_count * get_occupancy()`, not the bare CU count. Getting that
factor wrong halves the floor silently: the dispatch still works, the answers
still match, and only a row sweep in the 33..64 band shows the loss.

#2205 shipped without a test here because the floor "needs aiter and a live CUDA
device, neither of which CI has". The device is only read for one integer, so
stubbing `get_device_properties` turns this into arithmetic that runs anywhere
aiter imports -- which is what makes the factor testable at all.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("triton", reason="sparse_attn defines @triton.jit kernels")
pytest.importorskip("aiter", reason="the floor reads aiter's own split ladder")

import torch
from aiter.ops.triton.gluon import pa_decode_gluon

from atom.model_ops.minimax_m3 import sparse_attn


def _floor(monkeypatch, *, cus, occupancy, cap):
    """Evaluate the floor against a stubbed device and a stubbed aiter."""
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _i=None: SimpleNamespace(multi_processor_count=cus),
    )
    monkeypatch.setattr(pa_decode_gluon, "get_occupancy", lambda: occupancy)
    monkeypatch.setattr(
        pa_decode_gluon, "get_recommended_splits", lambda *_a, **_k: cap
    )
    # functools.cache keyed on device_index alone, so every case must clear it.
    sparse_attn._gluon_one_pass_max_rows.cache_clear()
    try:
        return sparse_attn._gluon_one_pass_max_rows(0)
    finally:
        sparse_attn._gluon_one_pass_max_rows.cache_clear()


class TestOccupancyIsInTheProduct:
    def test_mi355x_production_shape(self, monkeypatch):
        """256 CU, aiter's hardcoded occupancy 2, its cap 8 -> 64, not 32.

        32 is what the bare CU count gives, and it is half the measured
        crossover on the pinned triton 3.7 image (gluon still wins at 64 rows:
        11.2us against ASM's 14.7).
        """
        assert _floor(monkeypatch, cus=256, occupancy=2, cap=8) == 64

    def test_occupancy_scales_the_floor(self, monkeypatch):
        """Doubling occupancy doubles it. Drop the factor and this goes flat."""
        singly = _floor(monkeypatch, cus=256, occupancy=1, cap=8)
        doubly = _floor(monkeypatch, cus=256, occupancy=2, cap=8)
        assert doubly == 2 * singly, (
            "the floor ignored get_occupancy(); it is sizing against raw CUs "
            "while aiter sizes its split ladder against CUs * occupancy"
        )

    @pytest.mark.parametrize(
        "cus, occupancy, cap", [(256, 2, 8), (304, 2, 4), (64, 1, 8)]
    )
    def test_matches_aiters_own_residency_target(
        self, monkeypatch, cus, occupancy, cap
    ):
        """The floor IS aiter's resident-workgroup budget divided by its cap."""
        assert _floor(monkeypatch, cus=cus, occupancy=occupancy, cap=cap) == (
            cus * occupancy // cap
        )

    def test_never_zero(self, monkeypatch):
        """A cap larger than the budget must still leave one row on gluon.

        Zero would send even a single row to ASM, where it is measurably worse
        (12.7us against 9.0) and where run_pa_fwd_asm's envelope is tighter.
        """
        assert _floor(monkeypatch, cus=8, occupancy=1, cap=64) == 1


def test_real_device_agrees_with_aiter(monkeypatch):
    """No stubs: whatever this box is, the floor must equal aiter's own maths."""
    if not torch.cuda.is_available():
        pytest.skip("needs a device to read multi_processor_count")
    sparse_attn._gluon_one_pass_max_rows.cache_clear()
    cus = torch.cuda.get_device_properties(0).multi_processor_count
    budget = cus * pa_decode_gluon.get_occupancy()
    expected = max(1, budget // pa_decode_gluon.get_recommended_splits(1, 1))
    assert sparse_attn._gluon_one_pass_max_rows(0) == expected

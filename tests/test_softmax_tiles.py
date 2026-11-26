# SPDX-License-Identifier: MIT
"""The shared exponential race (`softmax_tiles.race_key`) and its Philox key
words.

Statistical checks use fixed seeds and tolerances several standard errors
wide, so they are deterministic on a given platform and fail only on a real
bias. The properties that hold for every coin are asserted exactly.
"""

import pytest
import torch

# The CPU unit job has neither aiter nor triton; both imports stay below these
# guards, or the job fails to collect this file and aborts the whole run.
pytest.importorskip("aiter", reason="the samplers are AITER/Triton ops")
if not torch.cuda.is_available():
    pytest.skip("ROCm GPU required", allow_module_level=True)

import triton
import triton.language as tl
from spec_lossless_reference import (
    DEV,
)

from atom.model_ops.spec_sampling.ratio_verify import (
    _RATIO_ACCEPT_KEY,
    _RATIO_RESIDUAL_KEY,
)
from atom.model_ops.spec_sampling.softmax_tiles import race_key
from atom.model_ops.tile_argmax import argmax_lowest_id


@triton.jit
def _uniforms(out_ptr, seed, base, WIDE: tl.constexpr, N: tl.constexpr):
    offs = tl.arange(0, N)
    if WIDE:
        u = tl.rand(seed, base + offs.to(tl.int64))
    else:
        u = tl.rand(seed, offs)
    tl.store(out_ptr + offs, u)


@pytest.mark.parametrize("wide", [False, True], ids=["int32-offset", "int64-offset"])
def test_the_key_purpose_word_separates_streams_at_any_offset_width(wide):
    """Draft noise, accept coins and residual race differ only in the key's
    high word (0, 1, 2); that word must reach Philox whether the offset is
    32-bit (the coins) or 64-bit (the draft noise and the residual race)."""

    step, n = 12345, 1024
    base = 5 * 2**31 if wide else 0
    streams = []
    for purpose in (0, _RATIO_ACCEPT_KEY, _RATIO_RESIDUAL_KEY):
        out = torch.empty(n, device=DEV)
        _uniforms[(1,)](out, purpose | step, base, WIDE=wide, N=n)
        streams.append(out)
    for i in range(3):
        for j in range(i + 1, 3):
            assert not torch.equal(streams[i], streams[j]), (i, j)


@triton.jit
def _race(
    mass_ptr,
    mask_ptr,
    key_ptr,
    best_ptr,
    id_ptr,
    seed,
    ROWS: tl.constexpr,
    N: tl.constexpr,
    TILE_2D: tl.constexpr,
):
    """`race_key` and `argmax_lowest_id` over ``[ROWS, N]``: as one 2-D tile
    (the GEMV tiles' shape), or one 1-D row per iteration (the races')."""
    offs = tl.arange(0, N)
    if TILE_2D:
        rows = tl.arange(0, ROWS)
        flat = rows[:, None] * N + offs[None, :]
        mask = tl.load(mask_ptr + flat) != 0
        key = race_key(
            tl.load(mass_ptr + flat), seed, rows[:, None].to(tl.int64) * N, offs, mask
        )
        best, best_id = argmax_lowest_id(key, offs[None, :], mask, 2147483647)
        tl.store(key_ptr + flat, key)
        tl.store(best_ptr + rows, best)
        tl.store(id_ptr + rows, best_id)
    else:
        for r in range(ROWS):
            mask = tl.load(mask_ptr + r * N + offs) != 0
            key = race_key(
                tl.load(mass_ptr + r * N + offs), seed, r * N, offs.to(tl.int64), mask
            )
            best, best_id = argmax_lowest_id(key, offs, mask, 2147483647)
            tl.store(key_ptr + r * N + offs, key)
            tl.store(best_ptr + r, best)
            tl.store(id_ptr + r, best_id)


@pytest.mark.parametrize("tile_2d", [False, True], ids=["1d", "2d"])
def test_the_shared_race_keys_and_picks_the_lowest_id(tile_2d):
    """The race every sampler here shares: masked lanes key -1, zero mass keys
    exactly 0, positive mass keys positive; the pick is the lowest id holding
    the max over kept lanes, and a row with no such lane answers the sentinel.
    Both layouts draw the same uniform per (row, id)."""
    torch.manual_seed(5)
    rows, n = 8, 256
    mass = torch.rand(rows, n, device=DEV)
    mass[mass < 0.3] = 0.0
    mask = torch.rand(rows, n, device=DEV) < 0.9
    mass[1] = 0.0  # no mass: every kept key 0, the lowest kept id wins
    mask[1, :3] = False
    mask[2] = False  # nothing kept: the sentinel
    mass[3] = float("nan")  # nothing equals a NaN max: the sentinel
    mass[4, 7] = float("nan")  # one NaN lane: the max skips it
    key = torch.empty(rows, n, device=DEV)
    best = torch.empty(rows, device=DEV)
    ids = torch.empty(rows, dtype=torch.int32, device=DEV)
    _race[(1,)](mass, mask.int(), key, best, ids, 77, ROWS=rows, N=n, TILE_2D=tile_2d)

    assert torch.all(key[~mask] == -1.0)
    kept = mask & ~mass.isnan()
    assert torch.all(key[kept & (mass == 0)] == 0.0)
    assert torch.all(key[kept & (mass > 0)] > 0.0)
    ref = torch.where(kept, key, float("-inf"))
    for r in range(rows):
        if r in (2, 3):
            assert ids[r].item() == 2147483647, r
            continue
        top = ref[r].max()
        assert best[r].item() == top.item(), r
        lowest = int(torch.nonzero(ref[r] == top)[0])
        assert ids[r].item() == lowest, r
    assert ids[1].item() == int(torch.nonzero(mask[1])[0])

    other = torch.empty_like(key)
    _race[(1,)](
        mass, mask.int(), other, best, ids, 77, ROWS=rows, N=n, TILE_2D=not tile_2d
    )
    assert torch.equal(key.isnan(), other.isnan())
    assert torch.equal(key.nan_to_num(), other.nan_to_num())

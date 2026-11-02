# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`ATOM_SPEC_BONUS_FUSED`: the two-launch draw against the race it replaces.

Runs on CPU tensors under Triton's interpreter (`triton_interpreter`), so it
needs Triton but no GPU and no AITER.

The reference is float64 `argmax_i softmax(x / max(T, 1e-5))_i / (E_i + eps)` on
the noise the kernel is given -- the definition of
`mixed_sample_outer_exponential` -- and, for a greedy row (T at or below the
epsilon), the lowest-id argmax of the raw logits (`topk_select(tie="low")`).
Greedy rows must match exactly, ties planted. A sampling row must match unless
the reference's two best keys are a near tie (relative gap < 1e-5), which the
fp32 rounding of either kernel may order either way.
"""

import pytest
import torch
from triton_interpreter import in_the_interpreter, interpreter_test

EPS = 1e-10

test_under_the_interpreter = interpreter_test(__file__)


def _bonus_sample():
    # Imported inside the child only: with the interpreter off, Triton would
    # build compiled kernels that nothing here can launch on CPU tensors.
    from atom.model_ops import bonus_sample

    return bonus_sample


def _reference_pick(logits_row, temp, noise_row):
    """(token, relative gap between the two best keys) for one sampling row."""
    t = max(float(temp), 1e-5)
    p = torch.softmax(logits_row.double() / t, dim=0)
    key = p / (noise_row.double() + EPS)
    top2 = torch.topk(key, 2)
    gap = float((top2.values[0] - top2.values[1]) / top2.values[0])
    return int(top2.indices[0]), int(top2.indices[1]), gap


@in_the_interpreter
@pytest.mark.parametrize("seed,vocab", [(1, 163840), (2, 50000), (3, 4097), (4, 1000)])
def test_matches_the_race_it_replaces(seed, vocab):
    gen = torch.Generator().manual_seed(seed)
    rows = 6
    logits = (torch.randn(rows, vocab, generator=gen) * 4).to(torch.bfloat16)
    temps = torch.tensor([1.0, 0.7, EPS, 1.3, EPS, 0.05], dtype=torch.float32)
    # Exact ties on the greedy rows: the lowest id must win.
    logits[2, 777 % vocab] = logits[2].float().max()
    logits[2, 12345 % vocab] = logits[2].float().max()
    logits[4, 5] = 50.0
    logits[4, vocab - 1] = 50.0
    noise = torch.empty(rows, vocab).exponential_(1, generator=gen)

    out = _bonus_sample().bonus_temperature_sample(logits, temps, noise, EPS, EPS)

    assert out.dtype == torch.int32 and out.shape == (rows,)
    for r in range(rows):
        if temps[r] <= EPS:
            expected = int(torch.nonzero(logits[r] == logits[r].max())[0])
            assert int(out[r]) == expected
            continue
        best, runner_up, gap = _reference_pick(logits[r], temps[r], noise[r])
        if int(out[r]) != best:
            assert gap < 1e-5 and int(out[r]) == runner_up, (r, gap)


@in_the_interpreter
def test_every_row_uses_its_own_temperature_and_noise():
    vocab = 4097
    gen = torch.Generator().manual_seed(7)
    logits = torch.randn(3, vocab, generator=gen).to(torch.bfloat16)
    logits[:] = logits[0]  # identical logits: only T and the noise differ
    temps = torch.tensor([1.0, 1.0, EPS])
    noise = torch.empty(3, vocab).exponential_(1, generator=gen)
    out = _bonus_sample().bonus_temperature_sample(logits, temps, noise, EPS, EPS)
    assert int(out[2]) == int(logits[0].float().argmax())
    for r in (0, 1):
        best, runner_up, gap = _reference_pick(logits[r], temps[r], noise[r])
        assert int(out[r]) == best or (gap < 1e-5 and int(out[r]) == runner_up)


@in_the_interpreter
def test_a_row_below_the_clamp_samples_like_the_clamped_temperature():
    # aiter clamps T to 1e-5 inside the race; a row just above the greedy
    # epsilon is a sampling row whose draw is (almost surely) the argmax.
    vocab = 4097
    gen = torch.Generator().manual_seed(11)
    logits = torch.randn(1, vocab, generator=gen).to(torch.bfloat16)
    noise = torch.empty(1, vocab).exponential_(1, generator=gen)
    temps = torch.tensor([1e-7])
    out = _bonus_sample().bonus_temperature_sample(logits, temps, noise, EPS, EPS)
    assert int(out[0]) == int(logits[0].float().argmax())


@in_the_interpreter
def test_no_rows_returns_an_empty_result():
    out = _bonus_sample().bonus_temperature_sample(
        torch.empty(0, 4097, dtype=torch.bfloat16),
        torch.empty(0),
        torch.empty(0, 4097),
        EPS,
        EPS,
    )
    assert out.shape == (0,) and out.dtype == torch.int32


@in_the_interpreter
def test_same_inputs_give_the_same_output():
    # The draw is in `Sampler.forward`; the kernels carry no state of their own,
    # so the same (logits, temps, noise) must give the same token every call.
    vocab = 50000
    gen = torch.Generator().manual_seed(3)
    logits = (torch.randn(5, vocab, generator=gen) * 4).to(torch.bfloat16)
    temps = torch.tensor([1.0, 0.7, 0.9, 1.2, 0.5])
    noise = torch.empty(5, vocab).exponential_(1, generator=gen)
    f = _bonus_sample().bonus_temperature_sample
    first = f(logits, temps, noise, EPS, EPS)
    assert torch.equal(first, f(logits, temps, noise, EPS, EPS))


@in_the_interpreter
def test_winner_in_a_late_tile_survives_the_reduction():
    # One dominating race key in the final vocab tile must beat tile 0: the
    # per-tile running max and the `(row,)` kernel's k * exp(m - mm) rescale
    # have to reconstruct the global key across tiles.
    vocab = 163840  # 40 tiles of 4096
    gen = torch.Generator().manual_seed(5)
    logits = (torch.randn(1, vocab, generator=gen) * 2).to(torch.bfloat16)
    noise = torch.full((1, vocab), 10.0)
    win = vocab - 3
    logits[0, win] = 60.0
    noise[0, win] = 1e-3  # huge key exp(0) / ~1e-3 against a flat field
    temps = torch.tensor([1.0])
    out = _bonus_sample().bonus_temperature_sample(logits, temps, noise, EPS, EPS)
    assert int(out[0]) == win

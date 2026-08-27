# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Reproducible sampling: the OpenAI ``seed`` parameter.

Covers the three pieces a seeded request passes through:

* :class:`SamplingParams` carries and validates it;
* ``build_seed_rows`` pairs each seeded sequence with the position it is
  generating, and mixes the two into a per-token seed;
* ``Sampler`` draws that row from its own generator instead of from the batch's
  shared noise.

Everything here runs on CPU tensors -- no GPU required.
"""

import types

import pytest
import torch

from atom.model_engine.sampling_control import build_seed_rows, mix_seed
from atom.model_ops.sampler import Sampler
from atom.sampling_params import SamplingParams


def _seq(tokens, seed=None, completed=None):
    """Minimal stand-in for the fields build_seed_rows reads off a Sequence."""
    seq = types.SimpleNamespace()
    seq.seed = seed
    seq.num_completion_tokens = len(tokens) if completed is None else completed
    return seq


# ---------------------------------------------------------------------------
# SamplingParams
# ---------------------------------------------------------------------------


class TestSamplingParams:
    def test_seed_defaults_to_none(self):
        assert SamplingParams().seed is None

    def test_seed_must_be_an_integer(self):
        with pytest.raises(ValueError, match="seed"):
            SamplingParams(seed="42")

    def test_seed_must_fit_in_64_bits(self):
        with pytest.raises(ValueError, match="64-bit"):
            SamplingParams(seed=2**63)

    def test_bool_is_not_a_seed(self):
        # bool is an int subclass; accepting it would silently seed with 0/1.
        with pytest.raises(ValueError, match="seed"):
            SamplingParams(seed=True)


# ---------------------------------------------------------------------------
# Seed derivation
# ---------------------------------------------------------------------------


class TestSeedDerivation:
    def test_is_a_pure_function(self):
        assert mix_seed(42, 7) == mix_seed(42, 7)

    def test_position_changes_the_draw(self):
        assert mix_seed(42, 0) != mix_seed(42, 1)

    def test_neighbouring_seeds_decorrelate(self):
        # A plain seed+position scheme would collide here: seed 42 at position 1
        # and seed 43 at position 0 would derive the same value.
        assert mix_seed(42, 1) != mix_seed(43, 0)

    def test_fits_a_torch_generator(self):
        for seed in (0, 1, -1, 2**62, -(2**62)):
            value = mix_seed(seed, 3)
            assert 0 <= value < 2**63
            torch.Generator().manual_seed(value)


# ---------------------------------------------------------------------------
# Seeded rows
# ---------------------------------------------------------------------------


class TestSeedRows:
    def test_no_seeds_means_no_payload(self):
        assert build_seed_rows([_seq([1, 2]), _seq([3])]) is None

    def test_only_seeded_rows_appear(self):
        rows, values = build_seed_rows(
            [_seq([1], seed=None), _seq([2], seed=7), _seq([3], seed=None)]
        )
        assert rows == [1]
        assert values == [mix_seed(7, 1)]

    def test_seed_is_paired_with_the_position_being_generated(self):
        # Three tokens already produced -> the next draw is position 3.
        rows, values = build_seed_rows([_seq([1, 2, 3], seed=99)])
        assert rows == [0]
        assert values == [mix_seed(99, 3)]

    def test_seed_zero_is_a_seed(self):
        # `if seed:` would drop it; the check has to be `is not None`.
        rows, _ = build_seed_rows([_seq([1], seed=0)])
        assert rows == [0]


# ---------------------------------------------------------------------------
# Seeded sampling
# ---------------------------------------------------------------------------


VOCAB = 64


@pytest.fixture
def sampler():
    return Sampler()


@pytest.fixture
def logits():
    generator = torch.Generator().manual_seed(0)
    return torch.randn(3, VOCAB, generator=generator)


def _draw(sampler, logits, rows, seeds, top_ks=None, top_ps=None, temps=None):
    sampled = torch.zeros(logits.shape[0], dtype=torch.int)
    if temps is None:
        temps = torch.ones(logits.shape[0])
    sampler._overwrite_seeded_rows(
        sampled, logits, temps, top_ks, top_ps, (rows, seeds)
    )
    return sampled.tolist()


class TestSeededSampling:
    def test_same_seed_and_position_replay(self, sampler, logits):
        seeds = [mix_seed(42, 0)] * 3
        assert _draw(sampler, logits, [0, 1, 2], seeds) == _draw(
            sampler, logits, [0, 1, 2], seeds
        )

    def test_different_seed_diverges(self, sampler, logits):
        first = _draw(sampler, logits, [0, 1, 2], [mix_seed(42, 0)] * 3)
        second = _draw(sampler, logits, [0, 1, 2], [mix_seed(43, 0)] * 3)
        assert first != second

    def test_next_position_diverges(self, sampler, logits):
        first = _draw(sampler, logits, [0, 1, 2], [mix_seed(42, 0)] * 3)
        second = _draw(sampler, logits, [0, 1, 2], [mix_seed(42, 1)] * 3)
        assert first != second

    def test_unseeded_rows_are_untouched(self, sampler, logits):
        sampled = torch.full((3,), -7, dtype=torch.int)
        sampler._overwrite_seeded_rows(
            sampled, logits, torch.ones(3), None, None, ([1], [mix_seed(1, 0)])
        )
        assert sampled[0].item() == -7
        assert sampled[2].item() == -7
        assert sampled[1].item() != -7

    def test_greedy_rows_are_left_to_argmax(self, sampler, logits):
        # temperature 0 is already deterministic; a seed must not perturb it.
        sampled = torch.full((3,), -7, dtype=torch.int)
        sampler._overwrite_seeded_rows(
            sampled, logits, torch.zeros(3), None, None, ([0], [mix_seed(1, 0)])
        )
        assert sampled[0].item() == -7

    def test_empty_seed_list_is_a_no_op(self, sampler, logits):
        sampled = torch.full((3,), -7, dtype=torch.int)
        sampler._overwrite_seeded_rows(
            sampled, logits, torch.ones(3), None, None, ([], [])
        )
        assert sampled.tolist() == [-7, -7, -7]

    def test_rows_beyond_the_batch_are_ignored(self, sampler, logits):
        sampled = torch.zeros(3, dtype=torch.int)
        sampler._overwrite_seeded_rows(
            sampled, logits, torch.ones(3), None, None, ([99], [mix_seed(1, 0)])
        )
        assert sampled.tolist() == [0, 0, 0]

    def test_top_k_one_is_argmax(self, sampler, logits):
        top_ks = torch.tensor([1, 1, 1], dtype=torch.int32)
        drawn = _draw(sampler, logits, [0, 1, 2], [mix_seed(7, 0)] * 3, top_ks=top_ks)
        assert drawn == logits.argmax(-1).tolist()

    def test_top_p_draws_stay_in_the_nucleus(self, sampler, logits):
        top_ps = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        probs = logits[0].softmax(-1)
        sorted_probs, sorted_idx = probs.sort(descending=True)
        nucleus = set(
            sorted_idx[(sorted_probs.cumsum(0) - sorted_probs) <= 0.5].tolist()
        )
        nucleus.add(sorted_idx[0].item())
        for trial in range(40):
            drawn = _draw(sampler, logits, [0], [mix_seed(trial, 0)], top_ps=top_ps)
            assert drawn[0] in nucleus

    def test_collapsed_filter_buffers_are_tolerated(self, sampler, logits):
        # prepare_sample collapses a uniform buffer to one element; the helper
        # must read that as "this value for every row".
        top_ks = torch.tensor([1], dtype=torch.int32)
        drawn = _draw(sampler, logits, [0, 1], [mix_seed(7, 0)] * 2, top_ks=top_ks)
        assert drawn[:2] == logits.argmax(-1).tolist()[:2]

    def test_matches_the_intended_distribution(self, sampler):
        # Seeded draws must sample the temperature-scaled distribution, not just
        # be reproducible: a constant would pass every test above.
        generator = torch.Generator().manual_seed(0)
        row = torch.randn(1, VOCAB, generator=generator)
        probs = row[0].softmax(-1)
        draws = 20000
        counts = torch.zeros(VOCAB)
        for trial in range(draws):
            drawn = _draw(sampler, row, [0], [mix_seed(trial, 0)])
            counts[drawn[0]] += 1
        distance = 0.5 * ((counts / draws) - probs).abs().sum().item()
        # torch.multinomial itself lands around 0.017 at this sample count.
        assert distance < 0.05

    def test_temperature_is_honoured(self, sampler):
        # A low temperature should concentrate the seeded draws on the argmax.
        generator = torch.Generator().manual_seed(0)
        row = torch.randn(1, VOCAB, generator=generator)
        best = row.argmax(-1).item()
        hits = sum(
            _draw(sampler, row, [0], [mix_seed(t, 0)], temps=torch.tensor([0.1]))[0]
            == best
            for t in range(50)
        )
        assert hits >= 45

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Tests for atomrl.lumen.models.llama31.dataset.

Covers:
  - chunking of a jsonl corpus into seq_length + 1 token samples
  - max_samples capping one pass over the corpus
  - allow_repeat serving a full step budget from a short corpus by wrapping,
    which is what keeps a fixed-length run from ending in StopIteration
  - check_sample_budget deciding, before step 0, whether a corpus is too short
"""

import json
import tempfile
from pathlib import Path

import pytest
from atomrl.lumen.models.llama31.dataset import PretrainTextDataset, check_sample_budget


class FakeTokenizer:
    """HF-style tokenizer emitting one id per character."""

    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def write_corpus(tmpdir, n_docs, doc_len):
    path = Path(tmpdir) / "corpus.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps({"text": "a" * doc_len}) + "\n" for _ in range(n_docs))
    return str(path)


def build(tmpdir, seq_length=4, n_docs=10, doc_len=9, **kwargs):
    return PretrainTextDataset(
        write_corpus(tmpdir, n_docs, doc_len),
        seq_length,
        FakeTokenizer(),
        is_hf_tokenizer=True,
        **kwargs,
    )


class TestChunking:
    def test_splits_corpus_into_fixed_length_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # 10 docs x (9 chars + 1 eos) = 100 ids -> 20 chunks of 5
            ds = build(tmpdir, seq_length=4)
            assert len(ds) == 20
            sample = ds[0]
            assert sample["input_ids"].shape[0] == 4
            assert sample["labels"].shape[0] == 4

    def test_labels_are_inputs_shifted_by_one(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = build(tmpdir, seq_length=4)
            sample = ds[0]
            assert sample["input_ids"][1:].tolist() == sample["labels"][:-1].tolist()


class TestMaxSamples:
    def test_caps_below_corpus_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = build(tmpdir, seq_length=4, max_samples=5)
            assert len(ds) == 5

    def test_does_not_extend_past_corpus_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = build(tmpdir, seq_length=4, max_samples=100)
            assert len(ds) == 20


class TestAllowRepeat:
    def test_serves_full_budget_from_short_corpus(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = build(tmpdir, seq_length=4, max_samples=100, allow_repeat=True)
            assert len(ds) == 100

    def test_wraps_back_to_the_first_chunk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = build(tmpdir, seq_length=4, max_samples=100, allow_repeat=True)
            assert ds[20]["input_ids"].tolist() == ds[0]["input_ids"].tolist()

    def test_still_caps_when_corpus_is_larger(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = build(tmpdir, seq_length=4, max_samples=5, allow_repeat=True)
            assert len(ds) == 5

    def test_empty_corpus_stays_empty(self):
        ds = PretrainTextDataset(
            None,
            4,
            FakeTokenizer(),
            is_hf_tokenizer=True,
            max_samples=100,
            allow_repeat=True,
        )
        assert len(ds) == 0


class TestNSamplesRead:
    """``n_samples_read`` is how a caller tells a short corpus from a full one.

    It counts distinct samples, not what the sampler is served, so a caller can
    compare it against the step budget and decide before step 0.
    """

    def test_falls_short_of_the_budget_on_a_short_corpus(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = build(tmpdir, seq_length=4, max_samples=100, allow_repeat=True)
            # Repeat hides the shortfall from __len__; this still shows it.
            assert len(ds) == 100
            assert ds.n_samples_read == 20

    def test_meets_the_budget_when_the_corpus_is_long_enough(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Reading stops once the budget is covered, so the count lands at
            # or just above max_samples -- never below, which would misreport a
            # sufficient corpus as short.
            ds = build(tmpdir, seq_length=4, max_samples=5)
            assert ds.n_samples_read >= 5

    def test_is_zero_without_a_corpus(self):
        ds = PretrainTextDataset(
            None, 4, FakeTokenizer(), is_hf_tokenizer=True, max_samples=100
        )
        assert ds.n_samples_read == 0


class TestCheckSampleBudget:
    """The gate between a short corpus and a run that silently repeats data."""

    def test_says_nothing_when_the_corpus_covers_the_budget(self):
        assert check_sample_budget(100, 100, allow_repeat=False) is None
        assert check_sample_budget(200, 100, allow_repeat=True) is None

    def test_rejects_a_short_corpus_by_default(self):
        with pytest.raises(ValueError) as excinfo:
            check_sample_budget(20, 100, allow_repeat=False)
        message = str(excinfo.value)
        # The error has to carry both numbers and the way out, or it sends the
        # reader looking for a bug instead of at their own configuration.
        assert "20" in message and "100" in message
        assert "--lumen-repeat-corpus" in message

    def test_reports_the_repeat_factor_when_asked_to_repeat(self):
        note = check_sample_budget(20, 100, allow_repeat=True)
        assert note is not None
        assert "5.00 times" in note

    def test_stays_quiet_on_an_empty_corpus(self):
        # A rank whose shard tokenized nothing is not evidence about the
        # corpus; Megatron's own empty-dataset handling owns that case.
        assert check_sample_budget(0, 100, allow_repeat=False) is None

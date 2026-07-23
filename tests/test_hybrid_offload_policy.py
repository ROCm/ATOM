# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for DSV4 scheduler-side offload policy (pure decisions)."""

from __future__ import annotations

import pytest

from atom.kv_transfer.offload.hybrid.policy import (
    candidate_boundaries,
    checkpoint_key,
    select_resume_boundary,
    should_save_at,
)


def test_should_save_at_requires_full_and_aligned():
    assert should_save_at(16384, 16384)  # full + aligned
    assert should_save_at(16384, 99999)  # over-computed still fine
    assert not should_save_at(16384, 16383)  # not fully prefilled
    assert not should_save_at(16130, 16130)  # not 128-aligned
    assert not should_save_at(64, 64)  # below min window


def test_candidate_boundaries_descending_and_strict():
    assert candidate_boundaries(512) == [384, 256, 128]
    assert candidate_boundaries(513) == [512, 384, 256, 128]
    assert candidate_boundaries(128) == []  # strict: B < prompt_len
    assert candidate_boundaries(0) == []


def test_candidate_boundaries_cap():
    got = candidate_boundaries(10_000, max_probes=3)
    assert got == [9984, 9856, 9728]  # 3 largest 128-multiples < 10000


def test_select_resume_boundary_picks_largest_below_len():
    stored = {128, 256, 512, 1024}
    assert select_resume_boundary(stored, 900) == 512
    assert select_resume_boundary(stored, 1024) == 512  # strict < len
    assert select_resume_boundary(stored, 1025) == 1024
    assert select_resume_boundary(stored, 100) is None  # nothing below
    assert select_resume_boundary(set(), 4096) is None


def test_select_resume_boundary_ignores_misaligned():
    assert select_resume_boundary({130, 200}, 4096) is None  # not 128-aligned


def test_checkpoint_key_stable_and_prefix_sensitive():
    fp = b"\x01" * 16
    toks = list(range(1000))
    k1 = checkpoint_key(toks, 512, fingerprint=fp)
    k2 = checkpoint_key(list(range(1000)), 512, fingerprint=fp)
    assert k1 == k2  # deterministic
    assert k1.startswith("dsv4:512:")
    # Different prefix contents -> different key.
    toks2 = list(range(1000))
    toks2[10] = 99999
    assert checkpoint_key(toks2, 512, fingerprint=fp) != k1
    # Different boundary -> different key.
    assert checkpoint_key(toks, 384, fingerprint=fp) != k1
    # Different geometry fingerprint -> different key.
    assert checkpoint_key(toks, 512, fingerprint=b"\x02" * 16) != k1


def test_checkpoint_key_validates():
    fp = b"\x00" * 16
    with pytest.raises(ValueError):
        checkpoint_key([1, 2, 3], 130, fingerprint=fp)  # not aligned
    with pytest.raises(ValueError):
        checkpoint_key([1, 2, 3], 128, fingerprint=fp)  # too few tokens


def test_checkpoint_key_only_depends_on_prefix_not_suffix():
    fp = b"\x07" * 16
    base = list(range(600))
    longer = base + [12345, 6789]  # same [0,512) prefix
    assert checkpoint_key(base, 512, fingerprint=fp) == checkpoint_key(
        longer, 512, fingerprint=fp
    )

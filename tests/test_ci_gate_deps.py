# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Declared PR dependencies of the heavy-CI gate (ci_gate/deps.py)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / ".github" / "scripts"))

from ci_gate.deps import declared_dependencies

REPO = "ROCm/ATOM"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Stacked on #3410; this PR's diff is its own change only.", (3410,)),
        ("Depends on: #12", (12,)),
        ("depends-on #12", (12,)),
        ("Depends on #1, #2, and #3", (1, 2, 3)),
        ("Blocked by #4 and #5", (4, 5)),
        ("Stacked on ROCm/ATOM#3199", (3199,)),
        ("Depends on rocm/atom#12", (12,)),  # GitHub resolves names case-insensitively
        ("Depends on https://github.com/ROCm/ATOM/pull/34", (34,)),
        (
            "Depends on #3187, ROCm/aiter#7788 and https://github.com/ROCm/ATOM/pull/5",
            (5, 3187),
        ),
        # Accepted false positive: history reads as a declaration; the hold says how to lift it.
        ("…gate_up input) and was stacked on #3198. That part…", (3198,)),
        # However it is written: words, emphasis, a list, a line break, a link.
        ("Depends on PR #12", (12,)),
        ("**Depends on** #12", (12,)),
        ("Depends on: **#12**", (12,)),
        ("Depends on:\n- #12\n- #13", (12, 13)),
        ("Depends on:\n\n- #12\n- #13", (12, 13)),  # a list after a blank line
        ("Depends on\n#12", (12,)),
        ("depends on [#12](https://github.com/ROCm/ATOM/pull/12)", (12,)),
        ("Depends on two open PRs (#3187, ROCm/aiter#7788)", (3187,)),
    ],
)
def test_declared(text, expected):
    assert declared_dependencies(text, 9999, REPO) == expected


def test_a_declaration_ends_with_its_paragraph():
    text = "Depends on #12.\n\nFixes #20, see #21."
    assert declared_dependencies(text, 9999, REPO) == (12,)


def test_a_keyword_with_words_after_it_does_not_reach_the_next_paragraph():
    text = "Depends on nothing open any more.\n\nFixes #20."
    assert declared_dependencies(text, 9999, REPO) == ()


@pytest.mark.parametrize(
    "text",
    [
        "Depends on ROCm/aiter#7788",
        "Stacked on https://github.com/ROCm/aiter/pull/5",
        "Fixes #12",
        "stacked on #9999",  # the PR itself
        "Requires MORI #1074",
    ],
)
def test_not_declared(text):
    assert declared_dependencies(text, 9999, REPO) == ()


def test_another_repository_in_a_list_is_dropped():
    assert declared_dependencies("Depends on #3187, ROCm/aiter#7788", 1, REPO) == (
        3187,
    )

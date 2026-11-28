# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""PR references in text (ci_gate/refs.py), as GitHub writes them."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / ".github" / "scripts"))

from ci_gate.refs import Reference, references


def test_the_three_forms_in_order():
    text = "ROCm/ATOM#1, then #2 and https://github.com/rocm/aiter/pull/3."
    assert references(text) == [
        Reference("ROCm/ATOM", 1),
        Reference(None, 2),
        Reference("rocm/aiter", 3),
    ]


@pytest.mark.parametrize("text", ["abc#6", "a/b/c#7", "##8", "#9x", "C#"])
def test_not_references(text):
    assert references(text) == []


def test_owner_repo_form_names_another_repository():
    assert references("see path/x#7") == [Reference("path/x", 7)]


def test_repository_names_compare_case_insensitively():
    assert Reference("rocm/atom", 1).is_in("ROCm/ATOM")
    assert Reference(None, 1).is_in("ROCm/ATOM")
    assert not Reference("ROCm/aiter", 1).is_in("ROCm/ATOM")

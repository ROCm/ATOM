# SPDX-License-Identifier: MIT
"""Guards the prompt construction of `scripts/run_longctx_needle.py`.

The script itself needs eight GPUs and a 328 GB checkpoint, so CI cannot run
it. What CI *can* guard is the part that makes its verdict mean anything: the
control prompt has to be identical to the real one apart from the secret, and
the digits-only match has to be incapable of finding a secret that was never
generated. If either of those slips, the script still prints PASSED -- for the
wrong reason -- and that is exactly the failure mode it exists to prevent.
"""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_longctx_needle.py"


@pytest.fixture(scope="module")
def needle():
    spec = importlib.util.spec_from_file_location("run_longctx_needle", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_control_differs_only_in_the_secret(needle):
    """Same length, same everything else -- so a differing answer is retrieval."""
    for depth in (0.0, 0.25, 0.5, 0.9, 1.0):
        a = needle.build_prompt(needle.SECRET_A, depth, 64)
        b = needle.build_prompt(needle.SECRET_B, depth, 64)
        assert len(a) == len(b)
        assert a.replace(needle.SECRET_A, needle.SECRET_B) == b


def test_the_only_digits_in_a_prompt_are_the_secret(needle):
    """What makes the digits-only match safe: nothing else can supply digits."""
    assert not any(c.isdigit() for c in needle.FILLER)
    prompt = needle.build_prompt(needle.SECRET_A, 0.5, 64)
    assert "".join(c for c in prompt if c.isdigit()) == needle.SECRET_A


def test_secrets_are_digit_disjoint(needle):
    """Cross-contamination has to be unambiguous, not a shared-substring fluke."""
    assert not set(needle.SECRET_A) & set(needle.SECRET_B)


def test_found_tolerates_separators_but_not_a_different_secret(needle):
    assert needle.found(needle.SECRET_A, "the code is 48213.")
    assert needle.found(needle.SECRET_A, "The code is **48,213**")
    assert needle.found(needle.SECRET_A, "code: 48 213")
    assert not needle.found(needle.SECRET_A, f"the code is {needle.SECRET_B}")
    assert not needle.found(needle.SECRET_A, "I do not know.")


def test_needle_lands_at_the_requested_depth(needle):
    """A needle pinned to one end would test position, not retrieval."""
    n_units = 100
    early = needle.build_prompt(needle.SECRET_A, 0.1, n_units)
    late = needle.build_prompt(needle.SECRET_A, 0.9, n_units)
    assert early.index(needle.SECRET_A) < late.index(needle.SECRET_A)
    # Both are inside the passage rather than adjacent to the question.
    assert late.index(needle.SECRET_A) < late.index("Question:")

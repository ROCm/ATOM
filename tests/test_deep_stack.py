# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""`run_on_deep_stack` (atom/utils/deep_stack.py): the call's arguments, result and
exceptions pass through, a deep call chain runs beneath the anchor, the anchor
reserves its value stack on CPython 3.11 / 3.12, and ATOM_DEEP_FRAME_STACK=0
calls straight through."""

import sys

import pytest

from atom.utils import deep_stack


def test_arguments_and_result_pass_through():
    assert deep_stack.run_on_deep_stack(lambda a, b=0: (a, b), 1, b=2) == (1, 2)


def test_exceptions_propagate():
    def boom():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        deep_stack.run_on_deep_stack(boom)


def test_a_deep_call_chain_runs_beneath_the_anchor():
    def descend(n):
        return n if n == 0 else 1 + descend(n - 1)

    assert deep_stack.run_on_deep_stack(descend, 600) == 600


@pytest.mark.skipif(
    sys.version_info[:2] not in ((3, 11), (3, 12)),
    reason="the frame chunks of CPython 3.11 / 3.12",
)
def test_the_anchor_reserves_its_value_stack():
    assert deep_stack._anchored.__code__.co_stacksize == deep_stack._RESERVED_SLOTS


def test_switched_off_it_calls_straight_through(monkeypatch):
    monkeypatch.setenv("ATOM_DEEP_FRAME_STACK", "0")

    def caller_name():
        return sys._getframe(1).f_code.co_name

    assert deep_stack.run_on_deep_stack(caller_name) == "run_on_deep_stack"

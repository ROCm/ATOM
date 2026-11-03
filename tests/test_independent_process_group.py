# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The vendored group helper passes its options under the name torch takes."""

import functools

import pytest
import torch

import atom.utils.independent_process_group as ipg


def _torch_2_5_helper(
    group_size, group_rank, global_ranks, backend, store, group_name, pg_options=None
):
    raise AssertionError("not called")


def _torch_2_6_helper(
    group_size,
    group_rank,
    global_ranks,
    backend,
    store,
    group_name,
    backend_options=None,
):
    raise AssertionError("not called")


@pytest.mark.parametrize(
    ("version", "helper", "keyword"),
    [
        ("2.5.1+rocm6.2", _torch_2_5_helper, "pg_options"),
        ("2.5a0+gitdeadbee", _torch_2_5_helper, "pg_options"),
        ("2.6.0.dev20241112+rocm6.2", _torch_2_6_helper, "backend_options"),
        ("2.10.0", _torch_2_6_helper, "backend_options"),
    ],
)
def test_options_go_under_the_keyword_the_helper_takes(
    monkeypatch, version, helper, keyword
):
    """Read off the version, a tag such as 2.5a0 failed to parse and was taken
    for 2.6 naming, so the options went to a 2.5 helper under a keyword it does
    not take."""
    monkeypatch.setattr(torch, "__version__", version)
    monkeypatch.setattr(ipg, "_new_process_group_helper", helper)

    assert ipg._options_keyword() == keyword


def test_a_wrapped_helper_is_read_through_its_wrapper(monkeypatch):
    @functools.wraps(_torch_2_5_helper)
    def timed(*args, **kwargs):
        return _torch_2_5_helper(*args, **kwargs)

    monkeypatch.setattr(ipg, "_new_process_group_helper", timed)

    assert ipg._options_keyword() == "pg_options"


def test_a_helper_taking_neither_keyword_is_refused(monkeypatch):
    def renamed_again(group_size, group_rank, global_ranks, backend, store, options):
        raise AssertionError("not called")

    monkeypatch.setattr(ipg, "_new_process_group_helper", renamed_again)

    with pytest.raises(RuntimeError, match="neither backend_options nor pg_options"):
        ipg._options_keyword()


def test_this_torch_is_one_the_helper_supports():
    assert ipg._options_keyword() in {"backend_options", "pg_options"}

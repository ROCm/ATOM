# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""`ModelRunner.state_builder`: the builder whose per-request state the
runner sizes, allocates, checkpoints and hands to transfers."""

import pytest
from aiter_stub import atom_without_aiter

from atom.model_ops.attentions.pool_layout.sub_pool_spec import (
    page_pool,
    state_pool,
)


@pytest.fixture(scope="module")
def ModelRunner():
    with atom_without_aiter() as load:
        yield load("atom.model_engine.model_runner").ModelRunner


class _Builder:
    def __init__(self, *specs):
        self._specs = list(specs)

    def sub_pool_specs(self):
        return self._specs


def _runner(ModelRunner, target, draft=None):
    runner = ModelRunner.__new__(ModelRunner)
    runner.attn_metadata_builder = target
    if draft is not None:
        runner.draft_kv_builder = draft
    return runner


def _state(entry_bytes):
    return state_pool("state", entry_bytes, entries_per_req=1)


def test_the_target_owns_state_without_a_stateful_draft(ModelRunner):
    target = _Builder(page_pool(64), _state(128))
    assert _runner(ModelRunner, target).state_builder is target
    draft = _Builder(page_pool(16))
    assert _runner(ModelRunner, target, draft).state_builder is target


def test_a_draft_declaring_state_owns_it(ModelRunner):
    target, draft = _Builder(page_pool(64)), _Builder(_state(128))
    assert _runner(ModelRunner, target, draft).state_builder is draft


def test_both_declaring_state_is_refused(ModelRunner):
    runner = _runner(
        ModelRunner,
        _Builder(page_pool(64), _state(8)),
        _Builder(_state(128)),
    )
    with pytest.raises(RuntimeError, match="both declare per-request state"):
        _ = runner.state_builder

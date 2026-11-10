# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""With TBO on, a mixed batch takes the mixed split -- or refuses through `can_split`.

`can_split` is AND-reduced across DP ranks so that one rank's inability to
split turns TBO off for the whole group. A rank that reported can_split=True
and then declined to build slices would run one ubatch while its peers ran two,
and the per-ubatch collectives would deadlock. So whatever the mixed split
cannot handle -- a cut that would leave a pure-decode ubatch -- must surface
here as can_split=False.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from atom.utils import envs
from atom.utils.tbo.ubatching import mixed_tbo_vote, prefill_tbo_vote


@pytest.fixture(autouse=True)
def _min_tokens(monkeypatch):
    """Pin the min-token bar at its default so `_PREFILL_TOKENS` clears it."""
    monkeypatch.setenv("ATOM_TBO_PREFILL_MIN_TOKENS", "8192")


class _Config:
    def __init__(self, enable_tbo=True, enable_tbo_decode=True):
        self.enable_tbo = enable_tbo
        self.enable_tbo_decode = enable_tbo_decode


class _Batch:
    """Only the attributes the runner vote reads."""

    def __init__(
        self,
        *,
        is_mixed=False,
        total_seqs_num_prefill=0,
        total_seqs_num_decode=0,
        total_tokens_num=0,
        total_tokens_num_prefill=0,
        is_dummy_run=False,
    ):
        self.has_prefill = total_seqs_num_prefill > 0
        self.is_mixed = is_mixed
        self.total_seqs_num_prefill = total_seqs_num_prefill
        self.total_seqs_num_decode = total_seqs_num_decode
        self.total_seqs_num = total_seqs_num_prefill + total_seqs_num_decode
        self.total_tokens_num = total_tokens_num
        self.total_tokens_num_prefill = total_tokens_num_prefill
        self.is_dummy_run = is_dummy_run


# Well above ATOM_TBO_PREFILL_MIN_TOKENS (8192), so the min-token bar is never
# what decides these cases.
_PREFILL_TOKENS = [8192, 8192]


def test_pure_prefill_can_split():
    meets_min, can_split, ub0, ub1 = prefill_tbo_vote(sum(_PREFILL_TOKENS))
    assert meets_min and can_split and ub0 + ub1 == sum(_PREFILL_TOKENS)


@pytest.mark.parametrize(
    "prefill,decode,can_split", [(16384, 4, True), (16, 128, False)]
)
def test_mixed_vote_keeps_decode_in_the_last_child(prefill, decode, can_split):
    meets_min, actual, ub0, ub1 = mixed_tbo_vote(
        np.asarray([prefill, decode]), 2, prefill, 8192
    )
    assert actual == can_split
    if can_split:
        assert meets_min and 0 < ub0 < prefill and ub0 + ub1 == prefill + decode
    else:
        assert (ub0, ub1) == (0, 0)


def test_tbo_off_reports_no_split(runner_tbo_eligibility):
    runner = SimpleNamespace(config=_Config(enable_tbo=False))
    batch = _Batch(is_mixed=True, total_seqs_num_prefill=2, total_seqs_num_decode=2)
    assert runner_tbo_eligibility(runner, batch) == (False, False, 0, 0)


@pytest.fixture(scope="module")
def runner_tbo_eligibility():
    # Execute the real CPU-only method and splitter without importing the
    # runner's GPU kernels. This keeps the backstop covered in non-GPU CI.
    path = Path(__file__).resolve().parents[1] / "atom/model_engine/model_runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ModelRunner"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_local_tbo_eligibility"
    )
    scope = {
        "ScheduledBatch": SimpleNamespace,
        "mixed_tbo_vote": mixed_tbo_vote,
        "prefill_tbo_vote": prefill_tbo_vote,
        "envs": envs,
    }
    exec(  # noqa: S102 - execute the trusted repository method, not GPU imports
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope
    )
    return scope["_local_tbo_eligibility"]


@pytest.mark.parametrize("pcp_size,is_mixed", [(1, True), (2, False)])
def test_runner_sizes_supported_mixed_and_pcp_collectives(
    runner_tbo_eligibility, pcp_size, is_mixed
):
    """Unsupported combinations are rejected by Config before scheduling."""
    config = _Config()
    config.prefill_context_parallel_size = pcp_size
    runner = SimpleNamespace(config=config)
    tokens = _PREFILL_TOKENS + ([1, 1] if is_mixed else [])
    batch = _Batch(
        is_mixed=is_mixed,
        total_seqs_num_prefill=2,
        total_seqs_num_decode=2 if is_mixed else 0,
        total_tokens_num=sum(tokens),
        total_tokens_num_prefill=sum(_PREFILL_TOKENS),
    )
    batch.num_scheduled_tokens = np.asarray(tokens, dtype=np.int64)
    meets_min, can_split, ub0, ub1 = runner_tbo_eligibility(runner, batch)
    assert meets_min and can_split
    if pcp_size == 1:
        assert ub0 + ub1 == sum(tokens)
    else:
        assert (ub0, ub1) == (0, 0)  # PCP cuts request groups after voting.

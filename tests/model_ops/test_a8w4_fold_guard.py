# SPDX-License-Identifier: MIT
# ATOM_QWEN_MOE_A8W4 turns off the two folds that hand MXFP4 rows to the routed
# experts (ATOM_FOLD_TOPK_ROUTER, ATOM_FOLD_POSTNORM): A8W4 experts take MXFP8.

import importlib

import pytest

from atom.utils import envs


@pytest.mark.parametrize("a8w4, expected", [("0", True), ("1", False)])
def test_folds_follow_a8w4(monkeypatch, a8w4, expected):
    monkeypatch.setenv("ATOM_FOLD_TOPK_ROUTER", "1")
    monkeypatch.setenv("ATOM_FOLD_POSTNORM", "1")
    monkeypatch.setenv("ATOM_QWEN_MOE_A8W4", a8w4)
    assert envs.ATOM_FOLD_TOPK_ROUTER is expected
    from atom.model_ops import fold_postnorm

    # fold_postnorm reads its flag at import.
    assert importlib.reload(fold_postnorm).ENABLED is expected


def test_folds_stay_off_by_default(monkeypatch):
    monkeypatch.delenv("ATOM_FOLD_TOPK_ROUTER", raising=False)
    monkeypatch.delenv("ATOM_FOLD_POSTNORM", raising=False)
    monkeypatch.setenv("ATOM_QWEN_MOE_A8W4", "0")
    assert envs.ATOM_FOLD_TOPK_ROUTER is False
    from atom.model_ops import fold_postnorm

    assert importlib.reload(fold_postnorm).ENABLED is False

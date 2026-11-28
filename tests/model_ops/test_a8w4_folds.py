# SPDX-License-Identifier: MIT
# ATOM_QWEN_MOE_A8W4 keeps the two folds (ATOM_FOLD_TOPK_ROUTER, ATOM_FOLD_POSTNORM): the
# folded router kernel writes the MXFP8 rows A8W4 experts take (routed_quant_mode).

import importlib

import pytest

from atom.utils import envs


@pytest.mark.parametrize("a8w4", ["0", "1"])
def test_folds_follow_their_flags_under_a8w4(monkeypatch, a8w4):
    monkeypatch.setenv("ATOM_FOLD_TOPK_ROUTER", "1")
    monkeypatch.setenv("ATOM_FOLD_POSTNORM", "1")
    monkeypatch.setenv("ATOM_QWEN_MOE_A8W4", a8w4)
    assert envs.ATOM_FOLD_TOPK_ROUTER is True
    from atom.model_ops import fold_postnorm

    # fold_postnorm reads its flag at import.
    assert importlib.reload(fold_postnorm).ENABLED is True


def test_folds_stay_off_by_default(monkeypatch):
    monkeypatch.delenv("ATOM_FOLD_TOPK_ROUTER", raising=False)
    monkeypatch.delenv("ATOM_FOLD_POSTNORM", raising=False)
    monkeypatch.setenv("ATOM_QWEN_MOE_A8W4", "1")
    assert envs.ATOM_FOLD_TOPK_ROUTER is False
    from atom.model_ops import fold_postnorm

    assert importlib.reload(fold_postnorm).ENABLED is False

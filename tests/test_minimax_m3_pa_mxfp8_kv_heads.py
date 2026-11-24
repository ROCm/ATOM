# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The sparse PA -> o_proj MXFP8 input (ATOM_MINIMAX_M3_PA_MXFP8_OUT) with more
than one local KV head (PP4 x TP1: 4 KV heads, 64 query heads a GPU). Runs on a
CPU runner: aiter, Triton and FlyDSL are stand-ins."""

import importlib
import importlib.abc
import importlib.util
import sys
from types import SimpleNamespace

import pytest
import torch

from tests.aiter_stub import _ExternalModuleLoader, atom_under_stubs


class _FlydslFinder(importlib.abc.MetaPathFinder):
    """model.py imports the mono runner's FlyDSL kernels; serve them stand-ins."""

    def find_spec(self, fullname, path, target=None):
        if fullname == "flydsl" or fullname.startswith("flydsl."):
            return importlib.util.spec_from_loader(
                fullname, _ExternalModuleLoader(), is_package=True
            )
        return None


@pytest.fixture(scope="module")
def mods():
    finder = _FlydslFinder()
    saved = {n: m for n, m in sys.modules.items() if n.split(".")[0] == "flydsl"}
    sys.meta_path.insert(0, finder)
    try:
        with atom_under_stubs() as load:
            yield SimpleNamespace(
                model=load("atom.models.minimax_m3.model"),
                nvfp4=load("atom.model_ops.minimax_m3.sparse_attn_nvfp4"),
                base=load("atom.model_ops.base_attention"),
                sparse=load("atom.model_ops.minimax_m3.sparse_attn"),
            )
    finally:
        sys.meta_path.remove(finder)
        for name in [n for n in sys.modules if n.split(".")[0] == "flydsl"]:
            sys.modules.pop(name)
        sys.modules.update(saved)


def _conditions(mods, monkeypatch, kv_heads):
    monkeypatch.setattr(mods.model.envs, "ATOM_MINIMAX_M3_PA_MXFP8_OUT", True)
    monkeypatch.setattr(mods.model, "nvfp4_reads_page_table", lambda: True)
    attn = SimpleNamespace(
        num_kv_heads=kv_heads, impl=SimpleNamespace(kv_format="nvfp4")
    )
    o_proj = SimpleNamespace(native_a8_group_rows=1)
    return mods.model._sparse_pa_mxfp8_conditions(attn, o_proj)


@pytest.mark.parametrize("kv_heads", [1, 4])
def test_mxfp8_out_conditions_by_local_kv_heads(mods, monkeypatch, kv_heads):
    """Decided at construction: a layer the attention cannot serve keeps the
    BF16 output and o_proj's own quant instead of failing at the first forward."""
    assert all(_conditions(mods, monkeypatch, kv_heads).values()) == (kv_heads == 1)

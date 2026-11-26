# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""MegatronNativeEngine's choice of TransformerEngine attention backend."""

from __future__ import annotations

import importlib.util

import pytest
import torch
from atomrl.engine.training import megatron_native_engine as eng


@pytest.fixture
def rocm_with_flash_attn(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.2.53211", raising=False)
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: object() if name == "flash_attn" else real(name, *a),
    )


def test_rocm_with_flash_attn_defaults_to_fused(rocm_with_flash_attn):
    # TE's auto would take flash_attn's backend here, half as fast as fused.
    assert eng._te_attention_backend({}) == "fused"
    assert eng._te_attention_backend({"te_attention_backend": None}) == "fused"


def test_explicit_choice_wins(rocm_with_flash_attn):
    assert eng._te_attention_backend({"te_attention_backend": "auto"}) is None
    assert eng._te_attention_backend({"te_attention_backend": "flash"}) == "flash"


def test_auto_without_rocm(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    assert eng._te_attention_backend({}) is None

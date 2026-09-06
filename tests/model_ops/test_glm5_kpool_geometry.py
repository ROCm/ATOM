# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU contracts shared by GLM-5.3's indexer and metadata builder."""

from atom.model_ops.glm5_next.geometry import (
    effective_kpool_size,
    pooled_path_enabled,
    topk_output_width,
)


def test_pooled_width_includes_tail_and_alignment(monkeypatch):
    monkeypatch.setenv("ATOM_GLM5_KPOOL", "1")

    assert pooled_path_enabled(4)
    assert effective_kpool_size(4) == 4
    assert topk_output_width(2048, 4) == 2176


def test_off_switch_restores_token_granular_geometry(monkeypatch):
    monkeypatch.setenv("ATOM_GLM5_KPOOL", "0")

    assert not pooled_path_enabled(4)
    assert effective_kpool_size(4) == 1
    assert topk_output_width(2048, 4) == 2048


def test_nonpooled_model_ignores_switch(monkeypatch):
    monkeypatch.setenv("ATOM_GLM5_KPOOL", "1")

    assert not pooled_path_enabled(1)
    assert effective_kpool_size(1) == 1
    assert topk_output_width(2048, 1) == 2048

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Attention backend selection for the diffusion DiT stack.

The backend is a product decision, not a fallback ladder: ASM is fastest on
gfx942, Triton is the only one that reproduces the sglang reference bit-for-bit.
These tests pin the *selection* logic, which is what a parity run depends on.
"""

import os

import pytest
import torch

from atom.diffusion.attention import (
    ATTENTION_BACKEND_ENV,
    AttentionBackend,
    packed_varlen_attention,
    resolve_attention_backend,
)
from atom.diffusion.models.minimax_h3.dit import MiniMaxH3DiTModel
from tests.test_diffusion_minimax_h3 import tiny_arch


def test_default_backend_is_asm(monkeypatch):
    monkeypatch.delenv(ATTENTION_BACKEND_ENV, raising=False)
    assert resolve_attention_backend() is AttentionBackend.ASM


def test_env_selects_backend(monkeypatch):
    monkeypatch.setenv(ATTENTION_BACKEND_ENV, "triton")
    assert resolve_attention_backend() is AttentionBackend.TRITON


def test_explicit_choice_beats_env(monkeypatch):
    monkeypatch.setenv(ATTENTION_BACKEND_ENV, "triton")
    assert resolve_attention_backend("sdpa") is AttentionBackend.SDPA


def test_string_and_enum_are_equivalent():
    assert resolve_attention_backend("ASM ") is resolve_attention_backend(
        AttentionBackend.ASM
    )


def test_unknown_backend_names_the_valid_ones():
    with pytest.raises(ValueError, match="asm"):
        resolve_attention_backend("flash3")


def test_cpu_matches_sdpa_reference():
    """On CPU every backend must route to SDPA rather than into aiter."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(12, 2, 8) for _ in range(3))
    cu = torch.tensor([0, 7, 12], dtype=torch.int32)
    ref = packed_varlen_attention(
        q, k, v, cu_seqlens=cu, max_seqlen=7, softmax_scale=0.35, backend="sdpa"
    )
    for backend in AttentionBackend:
        got = packed_varlen_attention(
            q, k, v, cu_seqlens=cu, max_seqlen=7, softmax_scale=0.35, backend=backend
        )
        assert torch.equal(got, ref), backend


def test_segments_do_not_leak_into_each_other():
    """A packed sequence is multiple independent segments, not one long one."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(9, 2, 8) for _ in range(3))
    cu = torch.tensor([0, 4, 9], dtype=torch.int32)
    packed = packed_varlen_attention(
        q, k, v, cu_seqlens=cu, max_seqlen=5, softmax_scale=0.35, backend="sdpa"
    )
    first = packed_varlen_attention(
        q[:4],
        k[:4],
        v[:4],
        cu_seqlens=torch.tensor([0, 4], dtype=torch.int32),
        max_seqlen=4,
        softmax_scale=0.35,
        backend="sdpa",
    )
    assert torch.allclose(packed[:4], first)


def test_empty_trailing_segment_is_skipped():
    """H3 pads the packed block with a zero-length segment on some shapes."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(6, 2, 8) for _ in range(3))
    cu = torch.tensor([0, 6, 6], dtype=torch.int32)
    out = packed_varlen_attention(
        q, k, v, cu_seqlens=cu, max_seqlen=6, softmax_scale=0.35, backend="sdpa"
    )
    assert out.shape == q.shape


def test_model_propagates_backend_to_every_attention():
    model = MiniMaxH3DiTModel(tiny_arch(), attn_backend="sdpa")
    assert model.attn_backend is AttentionBackend.SDPA
    seen = [m.attn_backend for m in model.modules() if hasattr(m, "attn_backend")]
    # model + 1 block + 1 refiner block, all agreeing.
    assert len(seen) >= 3
    assert set(seen) == {AttentionBackend.SDPA}


def test_model_reads_the_env_when_unset(monkeypatch):
    monkeypatch.setenv(ATTENTION_BACKEND_ENV, "triton")
    model = MiniMaxH3DiTModel(tiny_arch())
    assert model.attn_backend is AttentionBackend.TRITON


def test_env_is_not_consulted_once_constructed(monkeypatch):
    """Backend is frozen at construction; a later env flip must not split
    the model across two kernels mid-run."""
    monkeypatch.setenv(ATTENTION_BACKEND_ENV, "sdpa")
    model = MiniMaxH3DiTModel(tiny_arch())
    os.environ[ATTENTION_BACKEND_ENV] = "triton"
    assert model.attn_backend is AttentionBackend.SDPA
    assert all(
        m.attn_backend is AttentionBackend.SDPA
        for m in model.modules()
        if hasattr(m, "attn_backend")
    )

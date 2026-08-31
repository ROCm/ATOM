# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Packed varlen attention: backend selection and cross-backend parity."""

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
from tests.diffusion.test_h3_model import tiny_arch


@pytest.mark.parametrize(
    ("env", "explicit", "expected"),
    [
        (None, None, AttentionBackend.ASM),  # default
        ("triton", None, AttentionBackend.TRITON),  # env selects
        ("triton", "sdpa", AttentionBackend.SDPA),  # explicit beats env
        (None, "ASM ", AttentionBackend.ASM),  # string == enum, trimmed
        (None, AttentionBackend.ASM, AttentionBackend.ASM),
    ],
)
def test_backend_resolution(monkeypatch, env, explicit, expected):
    if env is None:
        monkeypatch.delenv(ATTENTION_BACKEND_ENV, raising=False)
    else:
        monkeypatch.setenv(ATTENTION_BACKEND_ENV, env)
    assert resolve_attention_backend(explicit) is expected


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


def test_pad_from_is_ignored_when_there_is_no_padding():
    torch.manual_seed(0)
    q, k, v = (torch.randn(12, 2, 8) for _ in range(3))
    cu = torch.tensor([0, 12], dtype=torch.int32)
    kw = {"cu_seqlens": cu, "max_seqlen": 12, "softmax_scale": 0.35, "backend": "sdpa"}
    assert torch.equal(
        packed_varlen_attention(q, k, v, **kw),
        packed_varlen_attention(q, k, v, pad_from=12, **kw),
    )


def test_dropping_a_trailing_pad_segment_preserves_the_real_rows():
    """The premise of the optimisation: padding lives in its own segment, so
    no real token attends to it and removing it cannot change their output."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(16, 2, 8) for _ in range(3))
    used = 12
    with_pad = packed_varlen_attention(
        q,
        k,
        v,
        cu_seqlens=torch.tensor([0, used, 16], dtype=torch.int32),
        max_seqlen=used,
        softmax_scale=0.35,
        backend="sdpa",
    )
    without_pad = packed_varlen_attention(
        q[:used],
        k[:used],
        v[:used],
        cu_seqlens=torch.tensor([0, used], dtype=torch.int32),
        max_seqlen=used,
        softmax_scale=0.35,
        backend="sdpa",
    )
    assert torch.equal(with_pad[:used], without_pad)


def test_the_dit_passes_used_len_down_to_attention(monkeypatch):
    """pad_from has to survive the whole chain -- packed_seq_params, the block
    stack and the refiner -- or the optimisation silently does nothing."""
    from atom.diffusion.models.minimax_h3 import dit as dit_mod
    from tests.diffusion.test_h3_model import make_inputs

    arch = tiny_arch()
    model = MiniMaxH3DiTModel(arch, attn_backend="sdpa").eval()
    seen = []
    original = dit_mod.packed_varlen_attention

    def spy(*args, **kwargs):
        seen.append(kwargs.get("pad_from"))
        return original(*args, **kwargs)

    monkeypatch.setattr(dit_mod, "packed_varlen_attention", spy)
    inputs = make_inputs(arch)
    inputs["packed_seq_params"] = {**inputs["packed_seq_params"], "used_len": 14}
    with torch.no_grad():
        model(**inputs)
    # Refiner blocks see None (their own cu_seqlens), the DiT blocks see 14.
    assert 14 in seen, seen

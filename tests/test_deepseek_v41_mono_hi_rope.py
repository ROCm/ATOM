# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The V4.1 mono hi programs read ATOM's own RoPE caches in place (``hi_weights.rope_args``: ``RotaryEmbedding``'s bf16
``cos_cache`` / ``sin_cache`` [positions, 32], the hi device code's ``rope_cs``): the pointer-table entries are those
caches, any other layout is refused, and the caches are the official ``precompute_freqs_cis`` tables rounded to bf16
(the numerics the hi path now shares with ATOM's own V4.1 kernels: the window layers' table without YaRN, the
compressing layers' with YaRN at ``compress_rope_theta``)."""

import math
from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_without_aiter

# DeepSeek-V4.1-Flash's config.json values
CFG = SimpleNamespace(
    qk_rope_head_dim=64,
    rope_theta=10000,
    compress_rope_theta=160000,
    rope_scaling={
        "rope_type": "yarn",
        "factor": 16,
        "beta_fast": 32,
        "beta_slow": 1,
        "original_max_position_embeddings": 65536,
    },
)


def official_freqs(dim, original_seq_len, base, factor, beta_fast, beta_slow):
    """The frequencies of the checkpoint's ``inference/model.py`` ``precompute_freqs_cis`` (its arithmetic as
    written there)."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:

        def corrected_dim(rotations):
            return (
                dim
                * math.log(original_seq_len / (rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = (
            (torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)
        ).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return freqs


def official_cis(seqlen, **kw):
    freqs = torch.outer(torch.arange(seqlen), official_freqs(**kw))
    return torch.polar(torch.ones_like(freqs), freqs)


WINDOW = {
    "dim": 64,
    "original_seq_len": 0,
    "base": 10000.0,
    "factor": 1,
    "beta_fast": 32,
    "beta_slow": 1,
}
YARN = {
    "dim": 64,
    "original_seq_len": 65536,
    "base": 160000,
    "factor": 16,
    "beta_fast": 32,
    "beta_slow": 1,
}


@pytest.fixture(scope="module")
def mods():
    """``hi_weights`` (pure torch) and ATOM's rotary module (AITER stubbed where it is not installed)."""
    with atom_without_aiter() as load:
        yield load("atom.models.deepseek_v41.mono.hi_weights"), load(
            "atom.model_ops.deepseek_v41.rotary"
        )


def _rotary(rotary, n, kw):
    """ATOM's ``RotaryEmbedding`` without its module init (its caches as ``__init__`` builds them)."""
    emb = rotary.RotaryEmbedding.__new__(rotary.RotaryEmbedding)
    torch.nn.Module.__init__(emb)
    freqs = rotary.rope_frequencies(
        kw["dim"],
        base=kw["base"],
        original_length=kw["original_seq_len"],
        factor=kw["factor"],
        beta_fast=kw["beta_fast"],
        beta_slow=kw["beta_slow"],
    )
    cis = torch.polar(
        torch.ones(n, kw["dim"] // 2), torch.outer(torch.arange(n), freqs)
    )
    emb.register_buffer("cos_cache", cis.real.to(torch.bfloat16).contiguous())
    emb.register_buffer("sin_cache", cis.imag.to(torch.bfloat16).contiguous())
    return emb


def test_the_tables_are_atoms_caches(mods):
    """``rope_args``: the programs' ``rope`` / ``rope.sin`` entries are the module's own caches (no copy)."""
    mh, rotary = mods
    emb = _rotary(rotary, 4096, YARN)
    args = mh.rope_args(emb)
    assert args["rope"] is emb.cos_cache and args["rope.sin"] is emb.sin_cache


def test_other_layouts_are_refused(mods):
    mh, rotary = mods
    emb = _rotary(rotary, 256, WINDOW)
    for bad in (
        SimpleNamespace(
            cos_cache=emb.cos_cache.float(), sin_cache=emb.sin_cache.float()
        ),
        SimpleNamespace(
            cos_cache=emb.cos_cache[:, :16], sin_cache=emb.sin_cache[:, :16]
        ),
        SimpleNamespace(cos_cache=emb.cos_cache.t(), sin_cache=emb.sin_cache.t()),
    ):
        with pytest.raises(mh.MonoUnsupported, match="RoPE caches"):
            mh.rope_args(bad)


@pytest.mark.parametrize("kw", [WINDOW, YARN], ids=["window", "yarn"])
def test_the_caches_are_the_official_tables_in_bf16(mods, kw):
    """ATOM's frequencies are the official ones, so its bf16 caches are the official f32 tables rounded to bf16 (the
    hi programs read these where they read the f32 tables before: the RoPE inputs differ by that rounding).
    """
    _, rotary = mods
    n = 65536 + 64
    emb = _rotary(rotary, n, kw)
    want = official_cis(n, **kw)
    assert torch.equal(
        rotary.rope_frequencies(
            kw["dim"],
            base=kw["base"],
            original_length=kw["original_seq_len"],
            factor=kw["factor"],
            beta_fast=kw["beta_fast"],
            beta_slow=kw["beta_slow"],
        ),
        official_freqs(**kw),
    )
    assert torch.equal(emb.cos_cache, want.real.to(torch.bfloat16))
    assert torch.equal(emb.sin_cache, want.imag.to(torch.bfloat16))

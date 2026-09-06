# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The chunked-prefill K workspace must be as wide as the MLA kernels expect.

A NoPE model keeps ``qk_rope_head_dim == 0`` in its config so the indexer
stays NoPE, and materializes the rope block as a zero pad on the MLA side.
Anything sized from the raw config is then too narrow by exactly that pad.
"""

from types import SimpleNamespace

import pytest

# `aiter_mla` imports AITER at module scope, so this module cannot even be
# collected on the CPU-only Pre Checkin runner without the guard.
pytest.importorskip("aiter")

from atom.model_ops.attentions.aiter_mla import mla_kv_entry_dim, mla_qk_head_dim


def _deepseek_like():
    # Real rope: nothing to pad, and the raw sum is already correct.
    return SimpleNamespace(
        kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128
    )


def _glm5_flash_like():
    # NoPE: config keeps rope at 0, mla_kv_entry_dim declares the padded 576.
    return SimpleNamespace(
        kv_lora_rank=512,
        qk_nope_head_dim=256,
        qk_rope_head_dim=0,
        v_head_dim=256,
        mla_kv_entry_dim=512 + 64,
    )


def test_real_rope_is_unchanged_by_the_helper():
    cfg = _deepseek_like()
    assert mla_kv_entry_dim(cfg) == 576
    # Exactly the raw sum it replaces, so no shipped model moves.
    assert mla_qk_head_dim(cfg) == cfg.qk_nope_head_dim + cfg.qk_rope_head_dim == 192


def test_nope_model_gets_the_padded_width_not_the_config_sum():
    cfg = _glm5_flash_like()
    assert cfg.qk_nope_head_dim + cfg.qk_rope_head_dim == 256  # what the raw sum gives
    assert mla_qk_head_dim(cfg) == 320  # what the kernels actually need


def test_the_width_leaves_a_nonzero_kv_pedim():
    """The property that actually matters, stated the way the kernel sees it.

    `gather_kv_b_proj` derives the rope width as
    ``k_prefix.shape[-1] - qk_nope_head_dim`` and emits
    ``tl.arange(0, KV_PeDim)``, which Triton refuses to compile at 0.
    """
    for cfg in (_deepseek_like(), _glm5_flash_like()):
        kv_pe_dim = mla_qk_head_dim(cfg) - cfg.qk_nope_head_dim
        assert kv_pe_dim > 0, "tl.arange(0, 0) -> Triton compile error"
        # It is the pad, and the pad is what the cache entry carries.
        assert kv_pe_dim == mla_kv_entry_dim(cfg) - cfg.kv_lora_rank

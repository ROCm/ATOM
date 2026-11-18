# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""_qkv_a_norm_fp4_supported: the gate for ATOM_DS_QKV_A_NORM_FP4.

Only the gating logic is covered here; the fused kernel itself needs a real
gfx1250 and aiter build (verified separately against the unfused path on
hardware). A stub attention object, not a real DeepseekV2MLAAttention, since
constructing the real module needs a loaded checkpoint.
"""

from types import SimpleNamespace

import pytest
from import_guard import skip_if_dependency_missing

try:
    # aiter/triton absent under bare non-GPU pytest
    from atom.models import deepseek_v2
except ImportError as exc:
    skip_if_dependency_missing(exc, "needs the aiter/triton stack")


def _attn(**overrides):
    base = {
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "indexer": None,
        "q_a_layernorm": SimpleNamespace(x_pad_to_multiple=0),
        "kv_a_layernorm": SimpleNamespace(x_pad_to_multiple=0),
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_supported_on_a_matching_shape():
    assert deepseek_v2._qkv_a_norm_fp4_supported(_attn())


@pytest.mark.parametrize("q_lora_rank", [None, 1024, 1537, 1536 + 128, 1152])
def test_refuses_a_q_lora_rank_outside_1024_1536_multiple_of_128(q_lora_rank):
    assert not deepseek_v2._qkv_a_norm_fp4_supported(_attn(q_lora_rank=q_lora_rank))


@pytest.mark.parametrize("kv_lora_rank", [513, 1024, 511])
def test_refuses_a_kv_lora_rank_over_512_or_not_a_multiple_of_8(kv_lora_rank):
    assert not deepseek_v2._qkv_a_norm_fp4_supported(_attn(kv_lora_rank=kv_lora_rank))


def test_refuses_an_indexer():
    assert not deepseek_v2._qkv_a_norm_fp4_supported(_attn(indexer=object()))


def test_refuses_a_padded_q_a_layernorm():
    attn = _attn(q_a_layernorm=SimpleNamespace(x_pad_to_multiple=8))
    assert not deepseek_v2._qkv_a_norm_fp4_supported(attn)


def test_refuses_a_padded_kv_a_layernorm():
    attn = _attn(kv_a_layernorm=SimpleNamespace(x_pad_to_multiple=8))
    assert not deepseek_v2._qkv_a_norm_fp4_supported(attn)


def test_env_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("ATOM_DS_QKV_A_NORM_FP4", raising=False)
    from atom.utils import envs

    assert envs.ATOM_DS_QKV_A_NORM_FP4 is False

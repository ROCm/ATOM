"""Qwen3.5 plugin configurations that would otherwise fail silently."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from atom.plugin.sglang.models.qwen3_5 import _validate_qwen35_runtime_contract


def _args(**overrides):
    fields = {
        "disable_radix_cache": False,
        "mamba_radix_cache_strategy": "extra_buffer",
        "page_size": 64,
        "speculative_algorithm": None,
        "attention_backend": "aiter",
        "decode_attention_backend": None,
        "prefill_attention_backend": None,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_extra_buffer_at_page_64_is_accepted():
    _validate_qwen35_runtime_contract(_args())


@pytest.mark.parametrize("page_size", [1, 128, 256])
def test_extra_buffer_rejects_other_page_sizes(page_size):
    with pytest.raises(ValueError, match="--page-size 64"):
        _validate_qwen35_runtime_contract(_args(page_size=page_size))


def test_page_size_is_free_without_the_radix_cache():
    _validate_qwen35_runtime_contract(_args(disable_radix_cache=True, page_size=128))


def test_dflash_on_aiter_requires_unified_attention():
    args = _args(speculative_algorithm="DFLASH")
    with pytest.raises(ValueError, match="SGLANG_USE_AITER_UNIFIED_ATTN=1"):
        _validate_qwen35_runtime_contract(args, unified_attn_enabled=False)
    _validate_qwen35_runtime_contract(args, unified_attn_enabled=True)


def test_dflash_uses_per_mode_overrides_instead_of_default_backend():
    args = _args(
        speculative_algorithm="DFLASH",
        attention_backend="aiter",
        prefill_attention_backend="triton",
        decode_attention_backend="triton",
    )
    _validate_qwen35_runtime_contract(args, unified_attn_enabled=False)


def test_dflash_on_effective_prefill_aiter_requires_unified_attention():
    args = _args(
        speculative_algorithm="DFLASH",
        attention_backend="triton",
        prefill_attention_backend="aiter",
    )
    with pytest.raises(ValueError, match="SGLANG_USE_AITER_UNIFIED_ATTN=1"):
        _validate_qwen35_runtime_contract(args, unified_attn_enabled=False)


@pytest.mark.parametrize(
    ("prefill_backend", "decode_backend", "raises"),
    [("triton", "aiter", True), ("aiter", "triton", False)],
)
def test_dflash_decode_mode_uses_decode_backend(
    prefill_backend, decode_backend, raises
):
    args = _args(
        speculative_algorithm="DFLASH",
        attention_backend="triton",
        speculative_attention_mode="decode",
        prefill_attention_backend=prefill_backend,
        decode_attention_backend=decode_backend,
    )
    if raises:
        with pytest.raises(ValueError, match="SGLANG_USE_AITER_UNIFIED_ATTN=1"):
            _validate_qwen35_runtime_contract(args, unified_attn_enabled=False)
    else:
        _validate_qwen35_runtime_contract(args, unified_attn_enabled=False)


def test_dflash_off_aiter_does_not_need_unified_attention():
    args = _args(speculative_algorithm="DFLASH", attention_backend="triton")
    _validate_qwen35_runtime_contract(args, unified_attn_enabled=False)

"""The gfx942 PTPC tuned MoE row must not keep block_m=16."""

from __future__ import annotations

import functools
import os
from functools import partial
from types import SimpleNamespace

from atom.plugin.sglang.patches.ck_block_m16_fallback import (
    bypass_unsupported_ck_block_m,
)


def _meta(block_m: int, kernel_name: str, *, run_1stage: bool = False):
    stage1 = partial(lambda: None, kernelName=kernel_name)
    return SimpleNamespace(
        block_m=block_m,
        run_1stage=run_1stage,
        full_impl=None,
        stage1=stage1,
    )


_CK16 = "moe_ck2stages_gemm1_256x16x64x256_1x4_MulABScale_silu_F8"


def test_block_m16_ck_row_requeries_without_tuned_config():
    seen = []

    def original(*_args, **_kwargs):
        seen.append(os.environ.get("AITER_BYPASS_TUNE_CONFIG"))
        if len(seen) == 1:
            return _meta(16, _CK16)
        return _meta(32, "untuned", run_1stage=True)

    result = bypass_unsupported_ck_block_m(original, 256, 2048, 512)

    assert result.block_m == 32
    assert result.run_1stage is True
    assert seen == [None, "1"]
    assert os.environ.get("AITER_BYPASS_TUNE_CONFIG") is None


def test_supported_block_m_is_not_requeried():
    calls = {"n": 0}

    def original(*_args, **_kwargs):
        calls["n"] += 1
        return _meta(32, "moe_ck2stages_gemm1_256x32x64x256")

    result = bypass_unsupported_ck_block_m(original, 256)

    assert result.block_m == 32
    assert calls["n"] == 1


def test_asm_one_stage_block_m16_is_kept():
    calls = {"n": 0}

    def original(*_args, **_kwargs):
        calls["n"] += 1
        return _meta(16, _CK16, run_1stage=True)

    result = bypass_unsupported_ck_block_m(original, 8)

    assert result.run_1stage is True
    assert calls["n"] == 1


def test_non_ck_kernel_with_block_m16_is_kept():
    calls = {"n": 0}

    def original(*_args, **_kwargs):
        calls["n"] += 1
        return _meta(16, "fmoe_int8_g1u1")

    result = bypass_unsupported_ck_block_m(original, 16)

    assert result.block_m == 16
    assert calls["n"] == 1


def test_lru_cached_lookup_is_cleared_before_requery():
    seen = []

    @functools.lru_cache(maxsize=8)
    def original(token):
        seen.append(os.environ.get("AITER_BYPASS_TUNE_CONFIG"))
        if os.environ.get("AITER_BYPASS_TUNE_CONFIG") == "1":
            return _meta(32, "untuned", run_1stage=True)
        return _meta(16, _CK16)

    result = bypass_unsupported_ck_block_m(original, 256)

    assert result.block_m == 32
    assert seen == [None, "1"]
    assert os.environ.get("AITER_BYPASS_TUNE_CONFIG") is None


def test_bypass_env_is_restored():
    os.environ["AITER_BYPASS_TUNE_CONFIG"] = "0"

    def original(*_args, **_kwargs):
        if os.environ.get("AITER_BYPASS_TUNE_CONFIG") == "0":
            return _meta(16, _CK16)
        return _meta(64, "untuned", run_1stage=True)

    try:
        result = bypass_unsupported_ck_block_m(original, 256)
        assert result.block_m == 64
        assert os.environ["AITER_BYPASS_TUNE_CONFIG"] == "0"
    finally:
        os.environ.pop("AITER_BYPASS_TUNE_CONFIG", None)

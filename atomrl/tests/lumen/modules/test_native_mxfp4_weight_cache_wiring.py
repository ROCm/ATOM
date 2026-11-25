# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""The native linear path hands its MXFP4 weight cache to the GEMM.

The cache and its optimizer-step invalidation hook were only ever reached from
``quantize._replace_forward``, the patched-Megatron path. ``--lumen-linear``
became the default later and ``_do_gemm`` called the GEMM with the BF16 weight,
so every gradient-accumulation micro-batch re-quantized and re-transposed an
unchanged weight -- 992 convert and transpose calls per step on Qwen3-8B where
124 suffice.

Nothing about that fails loudly: the cache exists, the hook fires, and the
numerics are identical either way, so tests covering the cache function and the
hook all pass while the path being measured never consults them. This asserts
the wiring itself.
"""

from importlib import import_module

import atomrl.lumen.ops.quantize.linear as linear_ops
import pytest
import torch
from atomrl.lumen.modules.parallel_linear import _do_gemm

_SENTINEL_DATA = object()
_SENTINEL_SCALE = object()


@pytest.fixture
def captured_gemm_kwargs(monkeypatch):
    """Record what ``_do_gemm`` passes to ``quantized_linear``."""
    captured = {}

    def _spy(*args, **kwargs):
        captured.update(kwargs)
        return torch.zeros(1)

    monkeypatch.setattr(linear_ops, "quantized_linear", _spy)
    return captured


@pytest.fixture
def cache_calls(monkeypatch):
    """Stub the cache so the wiring is testable without a GPU or AITER."""
    calls = []

    def _stub(
        module,
        weight,
        wcache,
        wscale,
        scaling_type,
        fp8_dtype,
        block_size,
        gemm_rows=None,
    ):
        calls.append({"scaling_type": scaling_type, "gemm_rows": gemm_rows})
        return _SENTINEL_DATA, _SENTINEL_SCALE

    quantize = import_module("atomrl.lumen.quantize")
    monkeypatch.setattr(quantize, "_mxfp4_cached_weight", _stub)
    return calls


def _call_do_gemm(scaling_type, rows=6, in_features=8, out_features=4):
    return _do_gemm(
        torch.zeros(rows, in_features),
        torch.zeros(out_features, in_features),
        None,
        scaling_manager=None,
        scaling_type=scaling_type,
        fp8_dtype=torch.float8_e4m3fn,
        block_size=32,
    )


def test_mxfp4_passes_the_cached_weight_into_the_gemm(
    captured_gemm_kwargs, cache_calls
):
    _call_do_gemm("mxfp4")

    assert cache_calls, "mxfp4 must consult the per-optimizer-step weight cache"
    assert captured_gemm_kwargs["fp8_weight_cache"] is _SENTINEL_DATA
    assert captured_gemm_kwargs["fp8_weight_scale"] is _SENTINEL_SCALE


def test_cache_is_told_the_row_count_of_the_gemm_it_serves(
    captured_gemm_kwargs, cache_calls
):
    """``gemm_rows`` decides whether an operand can be stored pre-shuffled."""
    _call_do_gemm("mxfp4", rows=6)

    assert cache_calls[0]["gemm_rows"] == 6


def test_row_count_counts_tokens_not_the_leading_dim(captured_gemm_kwargs, cache_calls):
    """Megatron hands (seq, batch, hidden); the GEMM sees seq*batch rows."""
    _do_gemm(
        torch.zeros(4, 3, 8),
        torch.zeros(4, 8),
        None,
        scaling_manager=None,
        scaling_type="mxfp4",
        fp8_dtype=torch.float8_e4m3fn,
        block_size=32,
    )

    assert cache_calls[0]["gemm_rows"] == 12


@pytest.mark.parametrize("scaling_type", ["delayed", "dynamic", "blockwise"])
def test_fp8_scaling_types_do_not_consult_the_mxfp4_cache(
    scaling_type, captured_gemm_kwargs, cache_calls
):
    """The cache is round-to-nearest MXFP4 only; FP8 recipes rescale per step."""
    _call_do_gemm(scaling_type)

    assert not cache_calls
    assert captured_gemm_kwargs["fp8_weight_cache"] is None
    assert captured_gemm_kwargs["fp8_weight_scale"] is None

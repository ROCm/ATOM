# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""MXFP4 weight operands built directly in the GEMM's layout.

Both operands the weight cache hands out -- the forward's and DGrad's -- are read
by nothing but an MXFP4 GEMM, so the quantizer and the transpose store them in
that GEMM's order and the separate permuting passes go away. A wrong permutation
does not raise: it multiplies the right numbers in the wrong places, so these
compare the GEMM's output against the operands built the two-pass way.
"""

import pytest
import torch
from torch import nn


def _is_gfx950():
    if not torch.cuda.is_available():
        return False
    return "gfx950" in torch.cuda.get_device_properties(0).gcnArchName


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(not _is_gfx950(), reason="gfx950 operand layout"),
]

BLOCK = 32


def _build(weight, gemm_rows):
    from atomrl.lumen.quantize import _mxfp4_cached_weight

    module = nn.Module()
    data, scale = _mxfp4_cached_weight(
        module,
        weight,
        None,
        None,
        "mxfp4",
        None,
        BLOCK,
        gemm_rows=gemm_rows,
    )
    data_t, scale_t = data._mxfp4_wt_cached
    return (data, scale), (data_t, scale_t)


def _operand_cache_inputs():
    w = torch.zeros(64, 64, device="cuda", dtype=torch.uint8)
    scale = torch.zeros(64, 2, device="cuda", dtype=torch.uint8)
    return w, scale


def test_operand_cache_refuses_to_memoize_a_view_of_the_weight():
    """A reshape of an already-shuffled weight is a view, not a copy.

    Memoizing it on the weight makes the weight reference itself. The cache
    guarded against being handed the weight back, but not against being handed a
    view of it, which leaked one copy of every quantized weight per iteration.
    """
    from atomrl.lumen.ops.quantize.linear import _cached_weight_operands

    w, scale = _operand_cache_inputs()
    key = "_test_alias_operands"

    data, _ = _cached_weight_operands(
        w, scale, key, lambda: (w.reshape(32, 128), scale)
    )

    assert data.data_ptr() == w.data_ptr(), "the build result should still be returned"
    assert getattr(w, key, None) is None


def test_operand_cache_lets_the_weight_die_by_refcount_alone():
    """The leak was only visible as a leak because GPU bytes are invisible to gc.

    With the cyclic collector off, a weight that the cache has put in a cycle
    never goes away. That is the training behaviour: the collector's thresholds
    count Python allocations, so a 2 GiB tensor never triggers one.
    """
    import gc
    import weakref

    from atomrl.lumen.ops.quantize.linear import _cached_weight_operands

    def build_and_drop():
        w, scale = _operand_cache_inputs()
        _cached_weight_operands(
            w, scale, "_test_alias_lifetime", lambda: (w.reshape(32, 128), scale)
        )
        return weakref.ref(w)

    gc_was_on = gc.isenabled()
    gc.disable()
    try:
        ref = build_and_drop()
        assert ref() is None
    finally:
        if gc_was_on:
            gc.enable()


def test_weight_cache_rebuilds_when_the_operand_layout_changes():
    """The layout the cache holds depends on the row count that asked for it.

    Whether either operand may be stored pre-shuffled is decided from
    ``gemm_rows``, since only some backends read that order. Keyed on the module
    alone, the first micro-batch's row count fixed the layout for the whole
    step, and a later GEMM dispatching to a backend that reads the other order
    got bytes in the wrong one -- which does not raise for the operand that has
    no quantizer/dispatch cross-check, it just multiplies the right numbers in
    the wrong places.

    It also means the fusion the docstring promises ("the first micro-batch
    writes row-major and every later one fuses") only actually arrived after the
    next optimizer step dropped the cache.
    """
    from atomrl.lumen.ops.quantize import mxfp4_autotune
    from atomrl.lumen.quantize import _mxfp4_cached_weight

    n_out, k_in = 4096, 4096
    weight = torch.randn(n_out, k_in, device="cuda", dtype=torch.bfloat16) * 0.02
    module = nn.Module()

    unfused_rows, fused_rows = 2048, 4096
    mxfp4_autotune.clear()
    try:
        # The unfused row count has no measured backend, so no operand may be
        # stored shuffled; the other is measured as one that reads that order.
        mxfp4_autotune._choice[(fused_rows, n_out, k_in)] = "asm"
        mxfp4_autotune._choice[(fused_rows, k_in, n_out)] = "asm"

        _mxfp4_cached_weight(
            module,
            weight,
            None,
            None,
            "mxfp4",
            None,
            BLOCK,
            gemm_rows=unfused_rows,
        )
        first_layout = module._mxfp4_w_cache[0]
        assert first_layout == (False, False)

        _mxfp4_cached_weight(
            module,
            weight,
            None,
            None,
            "mxfp4",
            None,
            BLOCK,
            gemm_rows=fused_rows,
        )
        assert (
            module._mxfp4_w_cache[0] != first_layout
        ), "kept the row-major operands for a row count whose backend reads shuffled"

        # A row count that agrees on both fusions reuses the entry.
        data_before = module._mxfp4_w_cache[1]
        _mxfp4_cached_weight(
            module,
            weight,
            None,
            None,
            "mxfp4",
            None,
            BLOCK,
            gemm_rows=fused_rows,
        )
        assert (
            module._mxfp4_w_cache[1] is data_before
        ), "rebuilt for an identical layout"
    finally:
        mxfp4_autotune.clear()


def test_mxfp4_refuses_a_weight_whose_output_width_is_ragged():
    """The quantizer pads those rows and nothing carries the count forward.

    Forward's ``output.view(..., weight.shape[0])`` then meets a wider tensor
    and dies on the row count, several frames from the cause and with no
    fallback. The reduction dim is free -- both operands are padded along it.
    """
    from atomrl.lumen.ops.quantize.linear import quantize_input, quantized_linear

    with pytest.raises(ValueError, match="multiple of 32"):
        quantize_input(
            torch.randn(48, 64, device="cuda", dtype=torch.bfloat16),
            "mxfp4",
            None,
            BLOCK,
            None,
            None,
            is_weight=True,
        )

    x = torch.randn(64, 48, device="cuda", dtype=torch.bfloat16) * 0.05
    w = torch.randn(64, 48, device="cuda", dtype=torch.bfloat16) * 0.02
    assert quantized_linear(x, w, scaling_type="mxfp4").shape == (
        64,
        64,
    ), "a ragged reduction dim is fine and must keep working"


def test_mxfp4_weight_shape_check_leaves_ragged_layers_alone():
    """The patch-time gate, which is where a static shape belongs.

    N is hidden size, vocab or a TP shard, so it cannot change during a run.
    """
    from atomrl.lumen.models.megatron import _mxfp4_weight_shape_supported

    ok = nn.Linear(64, 64)
    ragged = nn.Linear(64, 48)
    assert _mxfp4_weight_shape_supported(ok)
    assert not _mxfp4_weight_shape_supported(ragged)

    grouped = nn.Module()
    grouped.register_parameter("weight0", nn.Parameter(torch.empty(64, 64)))
    grouped.register_parameter("weight1", nn.Parameter(torch.empty(48, 64)))
    assert not _mxfp4_weight_shape_supported(
        grouped
    ), "grouped linears expose weight0..weightN rather than .weight"
    # No plain 2-D weight (grouped/MoE experts keep theirs elsewhere): not ours
    # to veto.
    assert _mxfp4_weight_shape_supported(nn.Module())


def test_operand_cache_does_not_hit_on_a_recycled_scale_address():
    """A freed scale's address is not proof the cached operands still match it.

    The cache keyed on ``(scale.data_ptr(), scale._version)``. The caching
    allocator readily returns a freed pointer for the next allocation of that
    size, and a freshly built scale starts at ``_version == 0``, so a dead
    tensor's stamp compared equal to a live unrelated one: a stale-operand hit
    with nothing to raise. Keyed on a weakref, a dead scale reads as dead.
    """
    from atomrl.lumen.ops.quantize.linear import _cached_weight_operands

    w = torch.zeros(64, 64, device="cuda", dtype=torch.uint8)
    key = "_test_recycled_scale"

    first_scale = torch.zeros(64, 2, device="cuda", dtype=torch.uint8)
    addr = first_scale.data_ptr()
    version = first_scale._version
    # The build result must not close over the scale, or the cache's strong
    # reference to it keeps the address alive and there is nothing to recycle.
    built_first = _cached_weight_operands(
        w,
        first_scale,
        key,
        lambda: (torch.full_like(w, 1),),
    )
    assert int(built_first[0][0, 0]) == 1

    del first_scale, built_first
    # Same size and dtype, so the allocator hands back the block just freed.
    second_scale = torch.ones(64, 2, device="cuda", dtype=torch.uint8)
    if second_scale.data_ptr() != addr or second_scale._version != version:
        pytest.skip("allocator did not recycle the address this test is about")

    built_second = _cached_weight_operands(
        w,
        second_scale,
        key,
        lambda: (torch.full_like(w, 2),),
    )
    assert (
        int(built_second[0][0, 0]) == 2
    ), "returned the operands built for a scale tensor that no longer exists"


@pytest.mark.parametrize(
    "N_out,K_in",
    [(6144, 4096), (4096, 12288)],
    ids=["qkv", "fc2"],
)
def test_mxfp4_cached_weight_fused_operands_match_two_pass(N_out, K_in):
    from atomrl.lumen.ops.quantize.linear import (
        _mxfp4_can_fuse_b_shuffle,
        gemm_mxfp4_dispatch,
    )
    from atomrl.lumen.ops.quantize.ops import convert_to_mxfp4

    torch.manual_seed(23)
    M = 2048
    weight = torch.randn(N_out, K_in, device="cuda", dtype=torch.bfloat16) * 0.05
    x = torch.randn(M, K_in, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(M, N_out, device="cuda", dtype=torch.bfloat16)
    a_fp4, a_scale = convert_to_mxfp4(x, block_size=BLOCK, axis=-1, use_sr=False)
    g_fp4, g_scale = convert_to_mxfp4(g, block_size=BLOCK, axis=-1, use_sr=False)

    # gemm_rows=None keeps both operands row-major, which is also what the first
    # micro-batch of a run gets, before the backend for the shape is measured.
    (w_ref, sw_ref), (wt_ref, swt_ref) = _build(weight, None)
    fwd_ref = gemm_mxfp4_dispatch(a_fp4, w_ref, a_scale, sw_ref)
    dgrad_ref = gemm_mxfp4_dispatch(g_fp4, wt_ref, g_scale, swt_ref)

    if not (
        _mxfp4_can_fuse_b_shuffle((M, N_out, K_in), N_out, K_in // 2)
        and _mxfp4_can_fuse_b_shuffle((M, K_in, N_out), K_in, N_out // 2)
    ):
        pytest.skip("these shapes dispatch to the row-major kernel, which cannot fuse")

    (w, sw), (wt, swt) = _build(weight, M)
    torch.testing.assert_close(
        gemm_mxfp4_dispatch(a_fp4, w, a_scale, sw), fwd_ref, atol=0, rtol=0
    )
    torch.testing.assert_close(
        gemm_mxfp4_dispatch(g_fp4, wt, g_scale, swt), dgrad_ref, atol=0, rtol=0
    )

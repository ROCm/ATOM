# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU checks for the V4-Pro TP8 + DSpark analytic estimate."""

import pytest

from atom.compass.dsv4_tp8_dspark import (
    V4_PRO_RATIOS,
    DSparkShape,
    V4ProShape,
    estimate,
    gemm_shapes,
    kernel_shapes,
    kv_bytes,
    weight_bytes,
)
from atom.compass.price_list import MissingPrice, Price, PriceList
from atom.model_ops.attentions.v4_pool_geometry import UnifiedPoolGeometry


def test_pro_trunk_matches_the_pool_geometry_fixture():
    assert len(V4_PRO_RATIOS) == 61
    assert sum(r == 4 for r in V4_PRO_RATIOS) == 30
    assert sum(r == 128 for r in V4_PRO_RATIOS) == 31
    assert 0 not in V4_PRO_RATIOS


def test_expert_weights_shard_and_indexer_does_not():
    shape = V4ProShape()
    draft = DSparkShape()
    tp1 = weight_bytes(shape, draft, tp=1)
    tp8 = weight_bytes(shape, draft, tp=8)
    assert tp8["moe.experts"] * 8 == tp1["moe.experts"]
    assert tp8["indexer.wq_b"] == tp1["indexer.wq_b"]
    assert tp8["compressor.csa"] == tp1["compressor.csa"]
    assert tp8["dspark.markov"] == tp1["dspark.markov"]
    assert tp8["attn.wq_b"] * 8 == tp1["attn.wq_b"]


def test_kv_pool_bytes_follow_unified_pool_geometry():
    shape = V4ProShape()
    got = kv_bytes(shape, num_blocks=4, num_slots=3, num_speculative_tokens=7)
    geo = UnifiedPoolGeometry(
        list(shape.ratios),
        num_blocks=4,
        num_slots=3,
        ring_slots=shape.window_size + 7,
        block_size=shape.block_size,
    )
    row = shape.nope_row_bytes + shape.rope_row_bytes
    assert got["pool"] == geo.plane_bytes(row)
    bigger = kv_bytes(shape, num_blocks=8, num_slots=3, num_speculative_tokens=7)
    assert bigger["pool"] > got["pool"]
    assert bigger["indexer"] == got["indexer"] * 2
    assert bigger["compressor_state"] == got["compressor_state"]


def test_decode_step_emits_the_declared_acceptance_length():
    slow = estimate(
        batch=4,
        prefill_tokens=128,
        num_speculative_tokens=7,
        acceptance_length=1,
        num_slots=4,
        num_blocks=4,
        context=1024,
    )
    fast = estimate(
        batch=4,
        prefill_tokens=128,
        num_speculative_tokens=7,
        acceptance_length=8,
        num_slots=4,
        num_blocks=4,
        context=1024,
    )
    # The forward is padded to K+1 either way, so the step time matches and
    # the throughput moves only with the declared acceptance length.
    assert slow.decode_s == pytest.approx(fast.decode_s)
    assert fast.output_tokens_per_s == pytest.approx(slow.output_tokens_per_s * 8)
    assert fast.tpot_s == pytest.approx(slow.tpot_s / 8)
    assert slow.provenance == "analytical"
    assert slow.draft.seconds > 0
    assert slow.verify.seconds > slow.draft.seconds


def test_a_pool_larger_than_the_gpu_does_not_fit():
    result = estimate(
        batch=1,
        prefill_tokens=16,
        acceptance_length=2,
        num_slots=8,
        num_blocks=10**9,
        context=128,
    )
    assert result.fits is False


def test_acceptance_length_outside_the_block_is_refused():
    with pytest.raises(ValueError, match="acceptance_length"):
        estimate(
            batch=1,
            prefill_tokens=16,
            num_speculative_tokens=7,
            acceptance_length=9,
            num_slots=2,
            num_blocks=2,
        )


def _priced_everywhere(microseconds: float) -> PriceList:
    """A price for every GEMM shape the default step looks up."""
    shape = V4ProShape()
    draft = DSparkShape()
    entries = {}
    for dtype, k, n in gemm_shapes(shape, draft, tp=8):
        for tokens in (8, 56, 64, 128):
            price = Price(dtype, tokens, k, n, microseconds * 1e-6, kernel="test")
            entries[price.key()] = price
    for op, m, k, n in kernel_shapes(
        shape,
        draft,
        8,
        batch=8,
        prefill_tokens=128,
        context=1024,
        num_speculative_tokens=7,
    ):
        price = Price(op, m, k, n, 0.0, kernel="test")
        entries[price.key()] = price
    return PriceList(entries, device="test", method="test", provenance="measured")


def test_a_measured_gemm_replaces_the_roofline_for_that_kernel():
    analytic = estimate(
        batch=8,
        prefill_tokens=128,
        num_speculative_tokens=7,
        acceptance_length=5,
        num_slots=4,
        num_blocks=4,
        context=1024,
    )
    priced = estimate(
        batch=8,
        prefill_tokens=128,
        num_speculative_tokens=7,
        acceptance_length=5,
        num_slots=4,
        num_blocks=4,
        context=1024,
        price_list=_priced_everywhere(1000.0),
    )
    assert priced.provenance == "mixed"
    # 1000 us per GEMM, and a verify step issues dozens of them per layer.
    assert priced.verify.seconds > analytic.verify.seconds
    assert priced.verify.seconds > 61 * 1000e-6


def test_a_missing_gemm_shape_is_refused():
    empty = PriceList({}, device="test", method="test")
    with pytest.raises(MissingPrice):
        estimate(
            batch=8,
            prefill_tokens=128,
            acceptance_length=2,
            num_slots=2,
            num_blocks=2,
            price_list=empty,
        )


def test_tp_that_does_not_divide_the_heads_is_refused():
    with pytest.raises(ValueError, match="n_heads"):
        estimate(
            batch=1,
            prefill_tokens=16,
            acceptance_length=2,
            num_slots=2,
            num_blocks=2,
            tp=3,
        )

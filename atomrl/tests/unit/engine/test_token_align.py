# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""``_token_align``: every token-count divisibility the forward has to pad for.

RL token counts are whatever the rollout produced, so each requirement below is
padded for rather than assumed. Getting this wrong is not a slowdown -- TE
raises, or DSv4's compressor asserts, on the first unaligned microbatch.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("megatron", reason="Megatron backend is optional in CPU CI")

from atomrl.engine.training.megatron_native_engine import MegatronNativeEngine


def _engine(
    *,
    tp=1,
    sp=False,
    cp=1,
    dsv4_align=1,
    has_spec_align=False,
    fp8=None,
    fp8_recipe="tensorwise",
):
    """A bare engine carrying only what ``_token_align`` reads."""
    e = object.__new__(MegatronNativeEngine)
    e._tp, e._sp, e._cp = tp, sp, cp
    e._dsv4_align = dsv4_align
    e._spec = SimpleNamespace(sequence_alignment=(object() if has_spec_align else None))
    e._tfcfg = SimpleNamespace(fp8=fp8, fp8_recipe=fp8_recipe)
    return e


def test_plain_bf16_run_needs_no_padding():
    assert _engine()._token_align() == 1


def test_sequence_parallel_requires_a_multiple_of_tp():
    assert _engine(tp=8, sp=True)._token_align() == 8


def test_tp_without_sequence_parallel_does_not_scatter_so_imposes_nothing():
    assert _engine(tp=8, sp=False)._token_align() == 1


def test_dsv4_compressor_ratio_applies():
    assert _engine(dsv4_align=128, has_spec_align=True)._token_align() == 128


def test_dsv4_ratio_doubles_under_cp():
    assert _engine(dsv4_align=128, has_spec_align=True, cp=2)._token_align() == 256


def test_dsv4_align_ignored_when_the_family_declares_no_alignment():
    # _dsv4_align is class-level, so a non-DSv4 family must not inherit a stale one.
    assert _engine(dsv4_align=128, has_spec_align=False)._token_align() == 1


@pytest.mark.parametrize(
    "recipe,expected",
    [
        ("tensorwise", 16),
        ("delayed", 16),
        ("blockwise", 16),
        ("mxfp8", 32),
    ],
)
def test_fp8_imposes_megatrons_alignment_per_recipe(recipe, expected):
    e = _engine(fp8="e4m3", fp8_recipe=recipe)
    assert e._token_align() == expected


def test_bf16_run_is_unaffected_by_a_stale_recipe_field():
    # fp8=None is the switch; the recipe string alone must not start padding.
    assert _engine(fp8=None, fp8_recipe="mxfp8")._token_align() == 1


def test_requirements_combine_as_an_lcm_not_a_max():
    # TP=8 and fp8=16 -> 16; adding DSv4's 128 subsumes both.
    assert _engine(tp=8, sp=True, fp8="e4m3")._token_align() == 16
    assert _engine(tp=8, sp=True, fp8="e4m3", fp8_recipe="mxfp8")._token_align() == 32
    e = _engine(tp=8, sp=True, fp8="e4m3", dsv4_align=128, has_spec_align=True)
    assert e._token_align() == 128


def test_a_coprime_pair_multiplies_rather_than_dropping_one():
    # TP=3 is not a real topology, but it is the case that distinguishes lcm
    # from max: a max would silently drop the TP requirement.
    assert _engine(tp=3, sp=True, fp8="e4m3")._token_align() == 48


def test_the_observed_failing_length_becomes_aligned():
    # dims=[1941, 2048] is what TE rejected on the first FP8 smoke.
    align = _engine(fp8="e4m3")._token_align()
    padded = -(-1941 // align) * align
    assert padded == 1952 and padded % 8 == 0 and padded % 16 == 0

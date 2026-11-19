# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU/static proofs for the gfx1250 Kimi K2 front-half substrate."""

import math

import pytest

from atom.models.kimi_k3.mono.kernels import execution_plan as ep
from atom.models.kimi_k3.mono.kernels.stages import moe_layout as ml
from atom.mono.plan.architecture import GFX950, GFX1250


def _plan(**changes):
    values = {
        "threads": 256,
        "blocks": 256,
        "lds_bytes": 320 * 1024,
        "tokens": 16,
        "nblocks": 6,
    }
    values.update(changes)
    return ep.Gfx1250KimiBuildPlan(**values)


@pytest.mark.parametrize("architecture", [GFX950, GFX1250])
def test_router_lane_ownership_is_complete_without_overlap(architecture):
    contract = ml.router_contract_for(architecture)
    owned = [contract.experts_for_lane(lane) for lane in range(architecture.wave_size)]
    flattened = [expert for lane in owned for expert in lane]
    assert len(flattened) == len(set(flattened))
    assert sorted(flattened) == list(range(ml.EXPERTS))
    assert contract.experts_per_lane == ml.EXPERTS // architecture.wave_size


def test_router_topk_ties_nans_signed_zero_and_order_are_explicit():
    scores = [-100.0] * ml.EXPERTS
    sigmoid = [float(expert + 1) for expert in range(ml.EXPERTS)]
    for expert in range(32):
        scores[expert] = 3.0
    scores[0] = float("nan")
    sigmoid[0] = float("nan")
    scores[1] = -0.0
    scores[2] = 0.0

    reference = ml.router_contract_for(GFX950).select(scores, sigmoid)
    wave32 = ml.router_contract_for(GFX1250).select(scores, sigmoid)
    assert wave32 == reference
    assert reference[0] == tuple(range(3, 19))
    assert ml.ordered_f32_key(0.0) > ml.ordered_f32_key(-0.0)
    assert all(math.isfinite(weight) for weight in reference[1])


def test_router_equal_keys_choose_lower_experts_and_sum_in_rank_order():
    scores = [1.0] * ml.EXPERTS
    sigmoid = [0.1 + expert * 1.0e-5 for expert in range(ml.EXPERTS)]
    experts, weights = ml.router_contract_for(GFX1250).select(scores, sigmoid)
    assert experts == tuple(range(ml.TOPK))
    total = ml.f32_round(0.0)
    for expert in experts:
        total = ml.f32_round(total + ml.f32_round(sigmoid[expert]))
    inverse = ml.f32_round(1.0 / total)
    assert weights == tuple(
        ml.f32_round(ml.f32_round(sigmoid[expert]) * inverse) for expert in experts
    )


@pytest.mark.parametrize(
    ("architecture", "threads"),
    [(GFX950, 512), (GFX1250, 32), (GFX1250, 128), (GFX1250, 256)],
)
def test_route_union_thread_ownership_covers_every_expert(architecture, threads):
    contract = ml.RouteUnionContract(architecture, threads, 16)
    owned = [contract.experts_for_thread(tid) for tid in range(threads)]
    flattened = [expert for thread in owned for expert in thread]
    assert flattened == list(range(ml.EXPERTS))
    assert len(flattened) == len(set(flattened))
    assert contract.scan_words == threads // architecture.wave_size


@pytest.mark.parametrize("tokens", range(1, 17))
def test_route_union_slots_and_kof_are_complete_and_nonoverlapping(tokens):
    routes = tuple(
        tuple((token * 7 + topk * 3) % ml.EXPERTS for topk in range(ml.TOPK))
        for token in range(tokens)
    )
    contract = ml.RouteUnionContract(GFX1250, 256, tokens)
    union = contract.build(routes)
    assert union.experts == tuple(sorted(set().union(*map(set, routes))))
    assert len(union.experts) == len(set(union.experts))
    for slot, expert in enumerate(union.experts):
        assert union.slot_of[expert] == slot
    for token, row in enumerate(routes):
        for topk, expert in enumerate(row):
            assert union.kof[union.slot_of[expert]][token] == topk


def test_route_union_refuses_duplicate_or_out_of_range_picks():
    contract = ml.RouteUnionContract(GFX1250, 256, 1)
    with pytest.raises(ValueError, match="unique"):
        contract.build((tuple([0] * ml.TOPK),))
    with pytest.raises(ValueError, match="outside"):
        contract.build((tuple(range(ml.TOPK - 1)) + (ml.EXPERTS,),))


@pytest.mark.parametrize("tokens", range(1, 17))
@pytest.mark.parametrize("threads", [32, 128, 256, 512])
def test_latent_quant_group_ownership_and_layout(tokens, threads):
    contract = ml.LatentQuantContract(GFX1250, threads, tokens)
    owned = [contract.blocks_for_thread(tid) for tid in range(threads)]
    flattened = [block for thread in owned for block in thread]
    assert len(flattened) == len(set(flattened))
    assert sorted(flattened) == list(range(contract.blocks))

    payload = []
    scales = []
    for block in range(contract.blocks):
        words, scale = contract.output_offsets(block)
        assert len(words) == ml.MX_PACKED_WORDS
        payload.extend(words)
        scales.append(scale)
    assert sorted(payload) == list(range(tokens * ml.LATENT // 8))
    assert sorted(scales) == list(range(tokens * ml.LATENT // ml.MX_GROUP))


def test_latent_quant_scale_contract_is_capped_roundup():
    contract = ml.LatentQuantContract(GFX1250, 256, 1)
    assert contract.e8m0_roundup(0.0) == 0
    assert contract.e8m0_roundup(6.0) == 127
    assert contract.e8m0_roundup(6.01) == 128
    assert contract.e8m0_roundup(float("inf")) == 254


@pytest.mark.parametrize("tokens", range(1, 17))
@pytest.mark.parametrize("threads", [32, 128, 256])
def test_k2_candidate_lds_is_bounded_across_wave32_inputs(tokens, threads):
    plan = _plan(tokens=tokens, threads=threads)
    lds = plan.lds_plan()
    assert lds.used_bytes <= lds.capacity_bytes
    route = ml.RouteUnionContract(GFX1250, threads, tokens)
    latent = ml.LatentQuantContract(GFX1250, threads, tokens)
    assert route.scan_words == threads // 32
    assert latent.blocks == tokens * ml.LATENT // 32


@pytest.mark.parametrize("rows", (1, 16, 32, 33, 40, 48, 49, 52, 53, 56, 60, 64))
def test_the_fused_moe_keeps_a_ring_of_four_to_64_rows(rows):
    from atom.models.kimi_k3.mono.kernels.stages import moe_tdm_gfx1250 as mt

    ring, resident = mt._moe_fit(rows, None, None)
    assert ring == mt.RING
    assert mt.moe_lds_bytes(rows, ring, resident) <= GFX1250.max_workgroup_lds_bytes
    # to 48 rows every latent K tile is resident: the build is the one before
    k_tiles = ml.LATENT // mt.K_TILE
    assert (resident == k_tiles) == (rows <= 48)
    assert resident >= 5
    name = mt.build_moe(rows).name
    assert name.startswith(f"k3_gfx1250_moe_s{rows}_p1_u15")
    assert (f"_k{resident}" in name) == (resident < k_tiles)


def test_a_three_item_ring_holds_the_whole_64_row_latent():
    from atom.models.kimi_k3.mono.kernels.stages import moe_tdm_gfx1250 as mt

    assert mt._moe_fit(64, 3, None) == (3, ml.LATENT // mt.K_TILE)
    assert "_r3" in mt.build_moe(64, ring=3).name
    with pytest.raises(ValueError, match="does not fit"):
        mt.build_moe(64, resident=ml.LATENT // mt.K_TILE)


@pytest.mark.parametrize("rows", (10, 40))
@pytest.mark.parametrize("rolled", (False, True))
def test_a_rolled_fused_moe_is_a_build_of_its_own(rows, rolled):
    from atom.models.kimi_k3.mono.kernels.stages.moe_tdm_gfx1250 import build_moe

    kernel = build_moe(rows, rolled=rolled)
    assert kernel.name.startswith(f"k3_gfx1250_moe_s{rows}_")
    assert kernel.name.endswith("_rl") == rolled


@pytest.mark.parametrize("rows", (1, 16, 40, 48, 56, 64))
def test_the_fused_moe_runs_wide_on_zeroed_padding_by_default(rows, monkeypatch):
    from atom.models.kimi_k3.mono.kernels.stages.moe_tdm_gfx1250 import build_moe

    monkeypatch.delenv("ATOM_KIMI_MONO_FUSED_MOE_WIDE", raising=False)
    monkeypatch.delenv("ATOM_KIMI_MONO_FUSED_MOE_ZERO_PAD", raising=False)
    monkeypatch.delenv("ATOM_KIMI_MONO_ROLLED_MOE", raising=False)
    name = build_moe(rows).name
    assert name.endswith("_w_z_rl")
    assert name == build_moe(rows, wide=True, zero_pad=True).name


@pytest.mark.parametrize("wide", ("0", "1"))
@pytest.mark.parametrize("zero_pad", ("0", "1"))
def test_the_fused_moe_switches_turn_wide_and_zeroed_padding_off(
    wide, zero_pad, monkeypatch
):
    from atom.models.kimi_k3.mono.kernels.stages.moe_tdm_gfx1250 import build_moe

    monkeypatch.setenv("ATOM_KIMI_MONO_FUSED_MOE_WIDE", wide)
    monkeypatch.setenv("ATOM_KIMI_MONO_FUSED_MOE_ZERO_PAD", zero_pad)
    name = build_moe(40).name
    assert ("_w_" in name) == (wide == "1")
    assert ("_z_" in name) == (zero_pad == "1")


def test_an_unrolled_fused_moe_zeroes_padding_on_six_waves():
    from atom.models.kimi_k3.mono.kernels.stages.moe_tdm_gfx1250 import build_moe

    name = build_moe(40, rolled=False, wide=True, zero_pad=True).name
    assert "_w" not in name
    assert name.endswith("_z")

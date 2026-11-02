# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU/static proofs for the gfx1250 Kimi K2 front-half substrate."""

import math
from pathlib import Path

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


@pytest.mark.parametrize("fused", [False, True])
def test_early_task_placement_is_unique_complete_and_in_grid(fused):
    schedule = ml.K2TaskSchedule(256, fused)
    schedule.validate_early_placement()
    tasks = schedule.early_tasks
    assert len(tasks) == 28 + 56 + 96
    assert len(tasks) == len({(task.kind, task.group, task.part) for task in tasks})
    owners = [schedule.early_owner(task) for task in tasks]
    assert all(0 <= owner < schedule.blocks for owner in owners)
    assert len(owners) == len(set(owners))
    route_owners = [schedule.route_owner(token, 16) for token in range(16)]
    assert len(route_owners) == len(set(route_owners))
    assert route_owners == list(range(0 if fused else 180, 16 if fused else 196))


@pytest.mark.parametrize("blocks", [1, 37, 256])
@pytest.mark.parametrize("total", [0, 1, 255, 256, 3072])
def test_reversed_ug_schedule_has_complete_unique_ownership(blocks, total):
    schedule = ml.K2TaskSchedule(blocks, False)
    owned = [schedule.ug_tasks_for_block(total, block) for block in range(blocks)]
    flattened = [task for block in owned for task in block]
    assert len(flattened) == len(set(flattened))
    assert sorted(flattened) == list(range(total))


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


def test_k2_dependency_map_and_gfx1250_device_evidence_are_explicit():
    stages = {entry.stage for entry in ml.K2_FRONT_HALF_DEPENDENCIES}
    assert stages == {
        "fused placement",
        "early router/latent/gate-up GEMV",
        "router and top-16",
        "route union",
        "latent MXFP4 quantization",
        "gate/up task placement",
        "expert gate/up and down",
        "post-expert tail stages",
    }
    assert {item.primitive for item in ml.K2_GFX1250_DEVICE_EVIDENCE} == {
        "wave32 ballot",
        "wave32 shuffle_xor",
        "portable E2M1 quantization",
        "BF16 WMMA early GEMV",
    }
    assert all(item.source and item.lines for item in ml.K2_GFX1250_DEVICE_EVIDENCE)
    root = Path(__file__).parents[2]
    config = (root / "atom/models/kimi_k3/mono/config.py").read_text()
    assert 'supported_gfx=("gfx950", "gfx1250")' in config

    gfx1250 = (
        root / "atom/models/kimi_k3/mono/kernels/stages/moe_gfx1250.py"
    ).read_text()
    assert "rocdl.ballot(T.i32" in gfx1250
    assert ".shuffle_xor(offset, GFX1250.wave_size)" in gfx1250
    assert "emit_f32_to_e2m1" in gfx1250
    assert "mfma_scaled" not in gfx1250

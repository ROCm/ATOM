import pytest

from atom.models.kimi_k3.mono.kernels.stages.k2_resident_protocol import (
    FINISH_PHASES,
    FRONT_PHASES,
    MAX_ROUTE_UNION_PER_TILE,
    PROJECTION_TASKS,
    ROUTED_GATE_UP_GROUPS,
    AttnResRoleSchedule,
    AttnResRoleTraffic,
    FinishIntervalSchedule,
    FrontProjectionSchedule,
    ResidentGridAssumption,
    ResidentGridProtocol,
)


@pytest.mark.parametrize("rows", (1, 16, 48))
def test_attnres_role_schedule_progress_and_removed_traffic(rows):
    schedule = AttnResRoleSchedule(rows, 5)
    schedule.prove_progress()
    assert schedule.tiles == (rows + 15) // 16
    assert len(schedule.slice_roles) == schedule.tiles * 56
    assert len(set(schedule.slice_roles)) == len(schedule.slice_roles)
    traffic = AttnResRoleTraffic.for_shape(rows, 5)
    assert traffic.removed_values_bytes == rows * 5 * 7168 * 4
    assert traffic.removed_mixed_bytes == rows * 7168 * 4
    assert traffic.compact_partials_bytes == 56 * rows * 5 * 2 * 4
    assert traffic.compact_weights_bytes == rows * 6 * 4
    assert traffic.compact_mean_squares_bytes == 56 * rows * 4
    assert traffic.compact_rstd_bytes == rows * 4
    assert traffic.net_removed_global_bytes > 0


def test_attnres_role_schedule_rejects_rows_above_contract():
    with pytest.raises(ValueError, match=r"\[1, 48\]"):
        AttnResRoleSchedule(49, 5).prove_progress()


@pytest.mark.parametrize(
    ("rows", "tiles", "partitions", "work"),
    (
        (1, 1, 2, 262),
        (16, 1, 2, 262),
        (32, 2, 1, 262),
        (48, 3, 1, 393),
    ),
)
def test_projection_schedule_exact_coverage(rows, tiles, partitions, work):
    schedule = FrontProjectionSchedule.for_shape(rows, 256)
    assert schedule.tiles == tiles
    assert schedule.k_partitions == partitions
    assert len(schedule.work) == work
    schedule.prove_coverage()
    assigned = [
        item for block in range(schedule.blocks) for item in schedule.for_block(block)
    ]
    assert len(assigned) == len(set(assigned)) == work
    assert {item.tile for item in assigned} == set(range(tiles))
    assert {item.k_partition for item in assigned} == set(range(partitions))
    assert {item.task for item in assigned} == set(range(PROJECTION_TASKS))


def test_projection_schedule_rejects_rows_above_contract():
    with pytest.raises(ValueError, match=r"\[1, 48\]"):
        FrontProjectionSchedule.for_shape(49, 256)


@pytest.mark.parametrize("rows", (1, 4, 8, 16, 17, 32, 33, 48))
def test_finish_interval_schedule_exact_coverage(rows):
    schedule = FinishIntervalSchedule(rows, 256)
    schedule.prove_coverage()
    assert schedule.tiles == (rows + 15) // 16
    assert schedule.router_roles == tuple(range(rows))
    assert len(FINISH_PHASES) == 5
    assert {item.kind for item in schedule.independent_work} == {
        "shared_down",
        "routed_gate_up",
    }
    assert {item.kind for item in schedule.routed_down_work} == {"routed_down"}


@pytest.mark.parametrize("union_size", (0, 1, 7, MAX_ROUTE_UNION_PER_TILE))
def test_finish_interval_runtime_union_mask_is_safe(union_size):
    schedule = FinishIntervalSchedule(48, 256)
    active = schedule.active_gate_up_tasks((union_size,) * schedule.tiles)
    gate_up = [item for item in active if item.kind == "routed_gate_up"]
    assert len(gate_up) == (schedule.tiles * union_size * ROUTED_GATE_UP_GROUPS)
    assert all(item.task < union_size * ROUTED_GATE_UP_GROUPS for item in gate_up)


@pytest.mark.parametrize("rows", (0, 49))
def test_finish_interval_schedule_rejects_invalid_rows(rows):
    with pytest.raises(ValueError, match=r"rows must be in \[1, 48\]"):
        FinishIntervalSchedule(rows, 256)


@pytest.mark.parametrize("blocks", (1, 256))
def test_front_protocol_has_exact_arrival_and_reset_ownership(blocks):
    protocol = ResidentGridProtocol(blocks, 256)
    proof = protocol.prove_progress(
        ResidentGridAssumption(all_declared_ctas_resident=True)
    )
    assert proof.phases == FRONT_PHASES
    assert proof.barriers == len(FRONT_PHASES) - 1
    assert len(proof.arrivals) == blocks * proof.barriers
    assert {(arrival.phase, arrival.block) for arrival in proof.arrivals} == {
        (phase, block) for phase in range(proof.barriers) for block in range(blocks)
    }
    assert tuple(proof.reset_owners).count(0) == proof.barriers
    assert protocol.scratch_bytes == 16 + blocks * proof.barriers * 8


def test_front_protocol_rejects_unproven_residency():
    protocol = ResidentGridProtocol(256, 256)
    with pytest.raises(ValueError, match="all declared CTAs"):
        protocol.prove_progress(
            ResidentGridAssumption(all_declared_ctas_resident=False)
        )
    with pytest.raises(ValueError, match="cannot validate hardware"):
        protocol.prove_progress(
            ResidentGridAssumption(
                all_declared_ctas_resident=True,
                hardware_validated=True,
            )
        )

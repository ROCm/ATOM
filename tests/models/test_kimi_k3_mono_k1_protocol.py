# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU state-machine proofs for Kimi K1's persistent mailbox protocol."""

import pytest

from atom.models.kimi_k3.mono.kernels.attention.protocol import (
    ATTENTION_CTAS,
    HEADS,
    PROJECTION_TASKS,
    STATE_CHUNKS,
    K1Protocol,
    ResidencyAssumption,
    StepResetState,
)


def _protocol(**changes):
    values = {
        "blocks": 256,
        "threads": 256,
        "tokens": 16,
        "qlen": 4,
        "sources": 7,
    }
    values.update(changes)
    return K1Protocol(**values)


@pytest.mark.parametrize("kda0", [0, 7, 255])
def test_every_cta_role_and_task_has_unique_complete_ownership(kda0):
    protocol = _protocol(kda0=kda0)
    assert protocol.timeline_points == 15
    assert protocol.timeline_observational_only
    assignments = protocol.assignments
    assert len(assignments) == protocol.blocks

    slices = [
        assignment.attention_slice
        for assignment in assignments
        if assignment.attention_slice is not None
    ]
    assert slices == list(range(ATTENTION_CTAS))

    gates = [
        assignment.gate_token
        for assignment in assignments
        if assignment.gate_token is not None
    ]
    assert gates == list(range(protocol.tokens))

    projection = [
        task for assignment in assignments for task in assignment.projection_tasks
    ]
    assert projection == list(range(PROJECTION_TASKS))
    assert len(projection) == len(set(projection))

    kda = [task for assignment in assignments for task in assignment.kda_tasks]
    assert sorted(kda) == list(range(protocol.kda_task_count))
    assert len(kda) == len(set(kda))


def test_each_head_chunk_group_runs_on_distinct_ctas():
    protocol = _protocol()
    owner = {
        task: assignment.block
        for assignment in protocol.assignments
        for task in assignment.kda_tasks
    }
    for request in range(protocol.requests):
        for head in range(HEADS):
            tasks = [
                (request * HEADS + head) * STATE_CHUNKS + chunk
                for chunk in range(STATE_CHUNKS)
            ]
            assert len({owner[task] for task in tasks}) == STATE_CHUNKS


@pytest.mark.parametrize(
    ("tokens", "qlen"),
    [
        (1, 1),
        (4, 1),
        (4, 2),
        (4, 4),
        (8, 4),
        (16, 1),
        (16, 4),
        (16, 8),
        (16, 16),
    ],
)
def test_all_resident_state_machine_reaches_completion(tokens, qlen):
    protocol = _protocol(tokens=tokens, qlen=qlen)
    assumption = ResidencyAssumption(all_declared_ctas_resident=True)
    proof = protocol.prove_progress(assumption)
    assert proof.actions_executed == sum(
        len(queue) for queue in protocol.action_queues()
    )
    assert protocol.all_parts <= proof.published
    assert protocol.all_weights <= proof.published
    assert protocol.all_msq <= proof.published
    assert protocol.all_xrdy <= proof.published
    assert protocol.all_projection <= proof.published
    assert proof.assumption is assumption
    assert not proof.assumption.hardware_validated


def test_progress_proof_does_not_claim_residency():
    protocol = _protocol()
    with pytest.raises(ValueError, match="every declared CTA resident"):
        protocol.prove_progress(ResidencyAssumption(all_declared_ctas_resident=False))
    with pytest.raises(ValueError, match="cannot mark hardware residency"):
        protocol.prove_progress(
            ResidencyAssumption(
                all_declared_ctas_resident=True,
                hardware_validated=True,
            )
        )


def test_every_wait_has_a_unique_reachable_producer():
    protocol = _protocol()
    actions = [action for queue in protocol.action_queues() for action in queue]
    producer_counts = {}
    waits = set()
    for action in actions:
        waits.update(action.waits)
        for resource in action.publishes:
            producer_counts[resource] = producer_counts.get(resource, 0) + 1
    assert waits <= producer_counts.keys()
    assert all(count == 1 for count in producer_counts.values())


def test_reset_chunk_ownership_is_complete_and_nonoverlapping():
    protocol = _protocol()
    for region, pairs in protocol.mailbox_pairs.items():
        owners = protocol.reset_chunk_owners(region)
        expected_chunks = (pairs * 8 + 15) // 16
        assert len(owners) == expected_chunks
        assert len(owners) == len(set(owners))
        assert all(
            0 <= block < protocol.blocks and 0 <= thread < protocol.threads
            for block, thread in owners
        )


def test_epoch_reset_and_peer_fence_gate_publication():
    state = StepResetState(blocks=256, ranks=8, epoch=3)
    previous_layer_tag = 5
    assert state.tag_after_reset(previous_layer_tag) == previous_layer_tag
    with pytest.raises(RuntimeError, match="before reset"):
        state.mailbox_tag(layer=4)

    for block in range(255):
        state.arrive_reset(block)
    assert not state.reset_complete
    state.arrive_reset(255)
    assert state.reset_complete and not state.peer_fence_complete
    assert state.tag_after_reset(previous_layer_tag) == 0

    for rank in range(7):
        state.arrive_peer_epoch(rank, 3)
    assert not state.publication_allowed
    with pytest.raises(ValueError, match="stale peer epoch"):
        state.arrive_peer_epoch(7, 2)
    state.arrive_peer_epoch(7, 3)
    assert state.publication_allowed
    assert state.mailbox_tag(layer=4) == 5

    # The same layer tag in a later step is unsafe until that step resets.
    next_step = StepResetState(blocks=256, ranks=8, epoch=4)
    with pytest.raises(RuntimeError, match="before reset"):
        next_step.mailbox_tag(layer=4)


def test_protocol_refuses_incomplete_projection_grid():
    with pytest.raises(ValueError, match="at least 253 CTAs"):
        _protocol(blocks=252)

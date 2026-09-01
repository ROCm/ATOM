import pytest
import torch

from atom.model_ops.fused_moe.routed_all2all import (
    build_routed_dispatch_plan,
    combine_routed_rows,
    pack_routed_payload,
    unpack_routed_payload,
)


def test_dispatch_plan_is_destination_major_and_deduplicates_rank():
    hidden = torch.tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])
    topk_ids = torch.tensor(
        [
            [0, 1, 4],
            [2, 3, -1],
            [5, 0, -1],
        ],
        dtype=torch.int32,
    )
    topk_weights = torch.tensor(
        [
            [0.5, 0.3, 0.2],
            [0.6, 0.4, 0.0],
            [0.8, 0.2, 0.0],
        ]
    )

    plan = build_routed_dispatch_plan(
        hidden,
        topk_ids,
        topk_weights,
        num_local_experts=2,
        world_size=3,
    )

    assert plan.send_counts.tolist() == [2, 1, 2]
    assert plan.token_indices.tolist() == [0, 2, 1, 0, 2]
    assert plan.hidden_states.tolist() == [
        [10.0, 11.0],
        [30.0, 31.0],
        [20.0, 21.0],
        [10.0, 11.0],
        [30.0, 31.0],
    ]
    assert plan.topk_ids.tolist() == [
        [0, 1, 4],
        [5, 0, 0],
        [2, 3, 0],
        [0, 1, 4],
        [5, 0, 0],
    ]
    assert torch.allclose(
        plan.topk_weights,
        torch.tensor(
            [
                [0.5, 0.3, 0.2],
                [0.8, 0.2, 0.0],
                [0.6, 0.4, 0.0],
                [0.5, 0.3, 0.2],
                [0.8, 0.2, 0.0],
            ]
        ),
    )


def test_dispatch_plan_handles_no_valid_routes():
    plan = build_routed_dispatch_plan(
        torch.empty((2, 4)),
        torch.full((2, 3), -1, dtype=torch.int32),
        torch.zeros((2, 3)),
        num_local_experts=2,
        world_size=4,
    )

    assert plan.send_counts.tolist() == [0, 0, 0, 0]
    assert plan.token_indices.numel() == 0
    assert plan.hidden_states.shape == (0, 4)
    assert plan.topk_ids.shape == (0, 3)


def test_dispatch_plan_rejects_expert_outside_physical_space():
    with pytest.raises(ValueError, match="outside dispatch space"):
        build_routed_dispatch_plan(
            torch.ones((1, 4)),
            torch.tensor([[6]], dtype=torch.int32),
            torch.ones((1, 1)),
            num_local_experts=2,
            world_size=3,
        )


def test_dispatch_plan_rejects_non_integer_expert_ids():
    with pytest.raises(ValueError, match="topk_ids must be int32 or int64"):
        build_routed_dispatch_plan(
            torch.ones((1, 4)),
            torch.tensor([[1.0]]),
            torch.ones((1, 1)),
            num_local_experts=2,
            world_size=2,
        )


def test_combine_routed_rows_restores_source_order_and_sums_destinations():
    returned = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    token_indices = torch.tensor([0, 2, 1, 0, 2])

    combined = combine_routed_rows(returned, token_indices, num_tokens=3)

    assert combined.tolist() == [[5.0], [3.0], [7.0]]


def test_combine_routed_rows_reuses_output_buffer():
    returned = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
    token_indices = torch.tensor([1, 1])
    output = torch.full((2, 2), 99.0)

    combined = combine_routed_rows(returned, token_indices, 2, output)

    assert combined.data_ptr() == output.data_ptr()
    assert combined.tolist() == [[0.0, 0.0], [7.0, 10.0]]


def test_mixed_dtype_payload_round_trip():
    # Odd BF16 width forces padding before the int32 metadata segment and
    # catches row-stride alignment bugs in dtype views.
    hidden = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.bfloat16)
    topk_ids = torch.tensor([[1, -1], [2, 3]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.75, 0.0], [0.6, 0.4]], dtype=torch.float32)

    payload, layout = pack_routed_payload(hidden, topk_ids, topk_weights)
    restored_hidden, restored_ids, restored_weights = unpack_routed_payload(
        payload, layout
    )

    assert payload.dtype == torch.uint8
    assert torch.equal(restored_hidden, hidden)
    assert torch.equal(restored_ids, topk_ids)
    assert torch.equal(restored_weights, topk_weights)


def test_two_rank_dispatch_compute_and_reverse_combine():
    hidden_by_rank = [
        torch.tensor([[1.0], [2.0]]),
        torch.tensor([[3.0]]),
    ]
    ids_by_rank = [
        torch.tensor([[0, 2], [1, 3]], dtype=torch.int32),
        torch.tensor([[0, 3]], dtype=torch.int32),
    ]
    weights_by_rank = [
        torch.tensor([[0.5, 0.5], [0.25, 0.75]]),
        torch.tensor([[0.25, 0.75]]),
    ]
    plans = [
        build_routed_dispatch_plan(
            hidden,
            ids,
            weights,
            num_local_experts=2,
            world_size=2,
        )
        for hidden, ids, weights in zip(
            hidden_by_rank, ids_by_rank, weights_by_rank, strict=True
        )
    ]

    # Simulate all-to-all: every destination concatenates source-major chunks.
    computed_by_destination: list[list[torch.Tensor]] = [[], []]
    for destination in range(2):
        lo, hi = destination * 2, (destination + 1) * 2
        for plan in plans:
            offsets = torch.cat(
                (torch.zeros(1, dtype=torch.int64), plan.send_counts.cumsum(0))
            )
            start, end = int(offsets[destination]), int(offsets[destination + 1])
            hidden = plan.hidden_states[start:end]
            ids = plan.topk_ids[start:end]
            weights = plan.topk_weights[start:end]
            owned = (ids >= lo) & (ids < hi)
            factors = torch.where(owned, ids + 1, torch.zeros_like(ids)).to(
                weights.dtype
            )
            computed_by_destination[destination].append(
                hidden * (weights * factors).sum(dim=1, keepdim=True)
            )

    # The reverse all-to-all restores destination-major order on each source.
    outputs = []
    for source, plan in enumerate(plans):
        returned = torch.cat(
            [computed_by_destination[dest][source] for dest in range(2)]
        )
        outputs.append(
            combine_routed_rows(
                returned,
                plan.token_indices,
                num_tokens=hidden_by_rank[source].shape[0],
            )
        )

    assert torch.allclose(outputs[0], torch.tensor([[2.0], [7.0]]))
    assert torch.allclose(outputs[1], torch.tensor([[9.75]]))

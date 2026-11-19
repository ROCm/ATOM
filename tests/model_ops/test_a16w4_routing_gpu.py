# SPDX-License-Identifier: MIT
"""Check route normalization on the device; no sorter or router mocks."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_repeated_routes_and_scheduled_rows_on_replay():
    from atom.model_ops.fused_moe.a16w4_routing import prepare_a16w4_routes

    ids = torch.tensor([[2, 2, 1], [3, 1, 2]], device="cuda", dtype=torch.int32)
    # Non-contiguous input, as when routing metadata comes from a wider buffer.
    w = torch.tensor([[0.25, 0.75], [0.75, 0.125], [-0.0, 0.125]], device="cuda").T
    count = torch.tensor([1], dtype=torch.int32, device=ids.device)
    prepare_a16w4_routes(ids, w, 4, count)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        oi, ow = prepare_a16w4_routes(ids, w, 4, count)
    graph.replay()
    assert oi.tolist() == [[2, -1, 1], [-1, -1, -1]]
    assert ow.tolist() == [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert torch.signbit(ow[0, 2]).item()  # untouched unique weight
    count.fill_(2)
    ids[0].copy_(torch.tensor([3, -1, 4], device=ids.device))
    graph.replay()
    assert oi.tolist() == [[3, -1, -1], [3, 1, 2]]
    assert ow.tolist() == [[0.25, 0.0, 0.0], [0.75, 0.125, 0.125]]
    count.zero_()
    graph.replay()
    assert (oi == -1).all() and (ow == 0).all()
    # No count: all input rows are live, including a three-way duplicate.
    ids[0].fill_(2)
    unscheduled_ids, unscheduled_weights = prepare_a16w4_routes(ids, w, 4)
    assert unscheduled_ids[0].tolist() == [2, -1, -1]
    assert unscheduled_weights[0].tolist() == [1.0, 0.0, 0.0]

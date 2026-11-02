# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from atom.models.kimi_k3.mono.external_collectives import (
    CollectiveKind,
    ExternalCollective,
    ExternalCollectivePlan,
    FrontPayloadLayout,
)
from atom.models.kimi_k3.mono.external_runner import (
    ExternalCollectiveK2Runner,
    MoeLocalPartials,
    PackedShardedFrontPartials,
)


@pytest.mark.parametrize("rows", (1, 8, 16, 48))
def test_packed_collective_plan(rows):
    plan = ExternalCollectivePlan(rows)
    payload = FrontPayloadLayout(rows)
    boundaries = plan.collectives

    assert tuple(item.name for item in boundaries) == (
        ExternalCollective.OPROJ,
        ExternalCollective.FRONT_PAYLOAD,
        ExternalCollective.ROUTED_LATENT,
        ExternalCollective.FINAL_HIDDEN,
    )
    assert boundaries[1].kind is CollectiveKind.ALL_GATHER
    assert boundaries[1].local_shape == payload.local_shape
    assert boundaries[1].global_shape == payload.gathered_shape
    assert all(
        item.kind is CollectiveKind.ALL_REDUCE
        for item in (boundaries[0], boundaries[2], boundaries[3])
    )


def test_packed_runner_orders_local_regions_and_collectives():
    rows = 2
    plan = ExternalCollectivePlan(rows)
    payload = FrontPayloadLayout(rows)
    events = []

    def all_reduce(tensor):
        events.append(("all_reduce", tuple(tensor.shape)))
        return tensor

    def all_gather(tensor):
        events.append(("all_gather", tuple(tensor.shape)))
        return torch.zeros(payload.gathered_shape, dtype=torch.uint8)

    runner = ExternalCollectiveK2Runner(plan, all_gather, all_reduce)
    retained = object()

    def oproj_local():
        events.append(("oproj",))
        return torch.zeros((rows, 7168), dtype=torch.bfloat16)

    def front_local(oproj):
        assert tuple(oproj.shape) == (rows, 7168)
        events.append(("front",))
        return PackedShardedFrontPartials(
            torch.zeros(payload.local_shape, dtype=torch.uint8),
            retained,
        )

    def finish_local(gathered, got_retained):
        assert tuple(gathered.shape) == payload.gathered_shape
        assert got_retained is retained
        events.append(("finish",))
        return MoeLocalPartials(
            torch.zeros((rows, 3584), dtype=torch.bfloat16),
            torch.zeros((rows, 7168), dtype=torch.bfloat16),
        )

    def finalize_local(routed, shared):
        assert tuple(routed.shape) == (rows, 3584)
        assert tuple(shared.shape) == (rows, 7168)
        events.append(("finalize",))
        return shared

    output = runner.run(
        oproj_local,
        front_local,
        finish_local,
        finalize_local,
    )

    assert tuple(output.shape) == (rows, 7168)
    assert [event[0] for event in events] == [
        "oproj",
        "all_reduce",
        "front",
        "all_gather",
        "finish",
        "all_reduce",
        "finalize",
        "all_reduce",
    ]


def test_runner_rejects_collective_shape_drift():
    rows = 1
    payload = FrontPayloadLayout(rows)
    runner = ExternalCollectiveK2Runner(
        ExternalCollectivePlan(rows),
        lambda _tensor: torch.zeros(payload.gathered_shape, dtype=torch.uint8),
        lambda _tensor: pytest.fail("all-reduce must not run"),
    )

    with pytest.raises(ValueError, match="oproj partial"):
        runner.run(
            lambda: torch.zeros((rows, 1), dtype=torch.bfloat16),
            lambda _oproj: pytest.fail("front must not run"),
            lambda _gathered, _retained: pytest.fail("finish must not run"),
            lambda _routed, _shared: pytest.fail("finalize must not run"),
        )

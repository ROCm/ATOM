# SPDX-License-Identifier: MIT
"""DP transport selection and host visibility, without a GPU or process group.

The fake stream delays the collective and host copy until synchronization;
reading an unsynchronized host buffer fails instead of returning stale data.
Real CPU tensors handle the payload, layout and output ownership assertions.
"""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.utils import envs
from atom.utils.tbo import ubatching as ub


@pytest.fixture
def transport(monkeypatch):
    runtime = SimpleNamespace(
        active=None,
        device=0,
        streams=[],
        allocations=[],
        gloo_calls=0,
        rccl_calls=0,
        compute=SimpleNamespace(pending=[]),
    )
    runtime.dp = SimpleNamespace(
        cpu_group=object(), device_group=object(), world_size=2
    )
    monkeypatch.setitem(
        sys.modules,
        "aiter.dist.parallel_state",
        SimpleNamespace(get_dp_group=lambda: runtime.dp),
    )
    monkeypatch.setattr(ub, "_DEVICE_SYNC_STATE", {})
    monkeypatch.setattr(ub, "_DEVICE_SYNC_CALLS", [0])
    monkeypatch.setattr(ub, "_DP_SYNC_CONFIG", None)
    monkeypatch.setenv("ATOM_DP_SYNC_VERIFY", "0")
    monkeypatch.setenv("ATOM_DP_SYNC_BACKEND", "rccl")

    class Stream:
        def __init__(self, device):
            self.pending = []
            runtime.streams.append(self)

        def __enter__(self):
            self.previous = runtime.active
            runtime.active = self
            return self

        def __exit__(self, *args):
            runtime.active = self.previous

        def synchronize(self):
            for operation in self.pending:
                operation()
            self.pending.clear()

        def wait_stream(self, compute):
            assert compute is runtime.compute
            prior = tuple(compute.pending)
            compute.pending.clear()
            self.pending.append(lambda: [operation() for operation in prior])

    class Buffer:
        def __init__(self, n):
            self.data = torch.full((n,), -1, dtype=torch.int32)

        def copy_(self, source, non_blocking=False):
            if non_blocking:
                runtime.active.pending.append(lambda: self.data.copy_(source.data))
            else:
                self.data.copy_(source)

        def view(self, *shape):
            assert all(not stream.pending for stream in runtime.streams)
            return self.data.view(*shape)

    empty = torch.empty

    def allocate(n, *, dtype, device, pin_memory=False):
        if torch.device(device).type == "cuda" or pin_memory:
            # Catches allocation on the caller's stream, even for an empty
            # tensor whose allocator storage could have pending writes.
            assert runtime.active is not None
            runtime.allocations.append((device, pin_memory, runtime.active))
            return Buffer(n)
        return empty(n, dtype=dtype, device=device)

    def rccl(out, local, group):
        assert group is runtime.dp.device_group
        runtime.rccl_calls += 1
        runtime.active.pending.append(
            lambda: out.data.copy_(torch.cat((local.data, local.data + 100)))
        )

    def gloo(out, local, group):
        assert group is runtime.dp.cpu_group
        runtime.gloo_calls += 1
        out[0].copy_(local)
        out[1].copy_(local + 100)

    monkeypatch.setattr(torch, "empty", allocate)
    monkeypatch.setattr(torch.cuda, "Stream", Stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: stream)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: runtime.device)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: runtime.compute)
    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", rccl)
    monkeypatch.setattr(torch.distributed, "all_gather", gloo)
    return runtime


@pytest.mark.parametrize("n", [5, 6, 9, 10])
def test_host_result_is_complete_and_survives_buffer_reuse(transport, n):
    local = torch.arange(n, dtype=torch.int32)
    first = ub._device_all_gather(local, 2, transport.dp.cpu_group)
    assert torch.equal(first, torch.stack((local, local + 100)))
    second = ub._device_all_gather(local + 7, 2, transport.dp.cpu_group)
    assert torch.equal(second, first + 7)
    assert torch.equal(first, torch.stack((local, local + 100)))
    assert len(transport.streams) == 1
    assert len(transport.allocations) == 3
    assert transport.allocations[-1][1] is True


def test_cache_changes_with_device_or_recreated_device_group(transport):
    local = torch.arange(5, dtype=torch.int32)
    ub._device_all_gather(local, 2, transport.dp.cpu_group)
    transport.device = 1
    ub._device_all_gather(local, 2, transport.dp.cpu_group)
    transport.dp.device_group = object()
    ub._device_all_gather(local, 2, transport.dp.cpu_group)
    assert len(transport.streams) == 3
    assert len(ub._DEVICE_SYNC_STATE) == 2


def test_previous_forward_collective_finishes_before_gather_on_every_call(
    transport, monkeypatch
):
    completed = [False]
    gather = torch.distributed.all_gather_into_tensor

    def ordered(out, local, group):
        # This assertion models GPU execution, not CPU launch order.
        def check():
            assert completed[0], "metadata raced the previous forward's collective"

        transport.active.pending.append(check)
        gather(out, local, group)

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", ordered)
    for step in range(2):
        completed[0] = False
        transport.compute.pending.append(lambda: completed.__setitem__(0, True))
        local = torch.arange(5, dtype=torch.int32) + step
        result = ub._device_all_gather(local, 2, transport.dp.cpu_group)
        assert torch.equal(result, torch.stack((local, local + 100)))
    assert len(transport.streams) == 1  # second call reuses buffers


@pytest.mark.parametrize("wrong_size", [False, True])
def test_group_mismatch_fails_before_posting_a_collective(transport, wrong_size):
    group = transport.dp.cpu_group if wrong_size else object()
    with pytest.raises(RuntimeError, match="same ranks"):
        ub._device_all_gather(
            torch.zeros(5, dtype=torch.int32), 3 if wrong_size else 2, group
        )
    assert transport.rccl_calls == transport.gloo_calls == 0


def test_verify_samples_device_calls_and_detects_mismatch(transport, monkeypatch):
    monkeypatch.setenv("ATOM_DP_SYNC_VERIFY", "2")
    local = torch.arange(5, dtype=torch.int32)
    for _ in range(3):
        ub._device_all_gather(local, 2, transport.dp.cpu_group)
    assert transport.gloo_calls == 1

    def different(out, local, group):
        out[0].copy_(local)
        out[1].copy_(local + 101)

    monkeypatch.setattr(torch.distributed, "all_gather", different)
    with pytest.raises(RuntimeError, match="DP sync mismatch"):
        ub._device_all_gather(local, 2, transport.dp.cpu_group)


def test_device_failure_propagates_without_switching_collectives(
    transport, monkeypatch
):
    def failed(*args, **kwargs):
        raise RuntimeError("collective failed")

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", failed)
    with pytest.raises(RuntimeError, match="collective failed"):
        ub._device_all_gather(
            torch.zeros(5, dtype=torch.int32), 2, transport.dp.cpu_group
        )
    assert transport.gloo_calls == 0


def test_default_uses_cpu_group_without_touching_cuda(transport, monkeypatch):
    monkeypatch.setenv("ATOM_DP_SYNC_BACKEND", "gloo")

    def refused(*args):
        raise AssertionError("gloo touched CUDA")

    monkeypatch.setattr(torch.cuda, "current_stream", refused)
    result = ub.sync_dp_metadata(
        dp_group=transport.dp.cpu_group,
        dp_size=2,
        scheduled_tokens=10,
        scheduled_bs=2,
        is_prefill=False,
        tbo_on=False,
        running_tokens_per_dp_rank=np.zeros(2, dtype=np.int32),
    )
    assert result.scheduled_tokens_per_dp_rank.tolist() == [10, 110]
    assert transport.gloo_calls == 1 and transport.rccl_calls == 0
    assert ub._DEVICE_SYNC_CALLS == [0]


def test_singleton_does_not_touch_cuda_or_a_collective(monkeypatch):
    monkeypatch.setenv("ATOM_DP_SYNC_BACKEND", "rccl")

    def refused(*args, **kwargs):
        raise AssertionError("singleton touched CUDA or a collective")

    monkeypatch.setattr(torch.cuda, "current_stream", refused)
    monkeypatch.setattr(torch.distributed, "all_gather", refused)
    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", refused)
    result = ub.sync_dp_metadata(
        dp_group=None,
        dp_size=1,
        scheduled_tokens=10,
        scheduled_bs=2,
        is_prefill=False,
        tbo_on=False,
        running_tokens_per_dp_rank=np.zeros(1, dtype=np.int32),
    )
    assert result.scheduled_tokens_per_dp_rank.tolist() == [10]


def test_backend_default_stays_gloo(monkeypatch):
    monkeypatch.delenv("ATOM_DP_SYNC_BACKEND", raising=False)
    assert envs.ATOM_DP_SYNC_BACKEND == "gloo"


@pytest.mark.parametrize(
    "tbo_on,dspark_on", [(False, False), (False, True), (True, False), (True, True)]
)
def test_device_metadata_matches_gloo_for_every_width(
    transport, monkeypatch, tbo_on, dspark_on
):
    results = []
    for backend in ("gloo", "rccl"):
        monkeypatch.setenv("ATOM_DP_SYNC_BACKEND", backend)
        running = np.zeros(2, dtype=np.int32)
        result = ub.sync_dp_metadata(
            dp_group=transport.dp.cpu_group,
            dp_size=2,
            scheduled_tokens=10,
            scheduled_bs=2,
            is_prefill=True,
            tbo_on=tbo_on,
            running_tokens_per_dp_rank=running,
            local_meets_min_tokens=True,
            local_can_split=True,
            local_ub_tokens=(4, 6),
            max_seqlen_q=3 if dspark_on else None,
        )
        results.append((result, running.copy()))
    a, b = (r[0] for r in results)
    assert np.array_equal(
        a.scheduled_tokens_per_dp_rank, b.scheduled_tokens_per_dp_rank
    )
    assert np.array_equal(results[0][1], results[1][1])
    assert a.max_bs_across_dp == b.max_bs_across_dp
    assert a.prefill_across_dp == b.prefill_across_dp
    assert a.any_rank_needs_eager == b.any_rank_needs_eager
    assert a.tbo_collective_active == b.tbo_collective_active
    assert a.ub_tokens_across_dp == b.ub_tokens_across_dp
    assert a.max_seqlen_q_across_dp == b.max_seqlen_q_across_dp
    assert a.max_final_requests_across_dp == b.max_final_requests_across_dp
    assert transport.rccl_calls == 1 and transport.gloo_calls == 1

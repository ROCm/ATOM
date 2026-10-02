# SPDX-License-Identifier: MIT
"""Real DMA/scatter correctness, asynchronous source reuse."""

import os

import pytest
import torch

from atom.utils.h2d import PublicationError, PublicationOwner
from tests.test_h2d_publication import buffer, setup

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_H2D_GPU_TESTS") != "1", reason="set RUN_H2D_GPU_TESTS=1"
)


def test_mixed_bytes_padding_dynamic_counts_and_graph_replay():
    owner = PublicationOwner("cuda", torch.cuda.Event())
    layouts = [
        ((5, 3), torch.float32, "rows", 2),
        ((1031,), torch.int64, "elements", 1027),
        ((19,), torch.bool, "bytes", 17),
        ((21,), torch.bfloat16, "bytes", 41),
    ]
    buffers = [
        buffer(*shape, dtype=dtype, device="cuda") for shape, dtype, _, _ in layouts
    ]
    members = [
        owner.bind(b, str(i), unit=layout[2])
        for i, (b, layout) in enumerate(zip(buffers, layouts))
    ]
    group = owner.group("mixed", members)
    assert group.use_transport("packed") == "packed"
    pointers = [b.gpu.data_ptr() for b in buffers]
    expected = [
        torch.full((b.cpu.numel() * b.cpu.element_size(),), 199, dtype=torch.uint8)
        for b in buffers
    ]
    owner.begin()
    group.publish([b.capacity for b in members])  # Compile before delayed work.
    owner.finish()
    for b in buffers:
        b.gpu.view(torch.uint8).fill_(199)
    outputs = [torch.empty_like(b.gpu.view(torch.uint8)) for b in buffers]
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for output, b in zip(outputs, buffers):
            output.copy_(b.gpu.view(torch.uint8))
    for step, counts in enumerate(
        (
            [x[3] for x in layouts],
            [0, 3, None, 1],
            [1, None, None, None],
            [None] * 4,
            [b.capacity for b in members],
        )
    ):
        owner.begin()
        for i, b in enumerate(buffers):
            b.cpu.view(torch.uint8).fill_(71 + step + i)
        torch.cuda._sleep(2_000_000)
        group.publish(counts)
        owner.finish()
        graph.replay()
        # begin() in the next step must wait before the same arena is packed.
        owner.completion.synchronize()
        torch.cuda.synchronize()
        for i, (b, member, count, output) in enumerate(
            zip(buffers, members, counts, outputs)
        ):
            expected[i][: (count or 0) * member.bytes_per_count] = 71 + step + i
            assert torch.equal(output.cpu().flatten(), expected[i])
            assert b.gpu.data_ptr() == pointers[i]
    assert len(group._backend.prefixes) <= 4


def test_packed_arena_reuse_and_atomic_validation():
    owner, a, b, _, _, group = setup("cuda", shape=(1031,))
    assert group.use_transport("packed") == "packed"
    owner.begin()
    group.publish((1031, 1031))
    owner.finish()
    observations = []
    for step in range(6):
        owner.begin()
        a.cpu.fill_(step + 11)
        b.cpu.fill_(step + 21)
        header = group._backend.header.copy()
        with pytest.raises(ValueError):
            group.publish((17, 1032))
        assert (header == group._backend.header).all()
        torch.cuda._sleep(3_000_000)
        group.publish((1031, 1031))
        observations.append((a.gpu.clone(), b.gpu.clone()))
        with pytest.raises(PublicationError, match="republish_reason"):
            group.publish((1, None))
        owner.finish()
    owner.completion.synchronize()
    for step, (first, second) in enumerate(observations):
        assert torch.all(first.cpu() == step + 11)
        assert torch.all(second.cpu() == step + 21)


def test_packed_failure_after_dma_poison_retains_arena(monkeypatch):
    owner, a, b, _, _, group = setup("cuda")
    assert group.use_transport("packed") == "packed"
    owner.begin()
    a.cpu.fill_(17)
    b.cpu.fill_(23)

    class FailingLaunch:
        def __getitem__(self, grid):
            def fail(*args, **kwargs):
                raise RuntimeError("scatter enqueue failed")

            return fail

    monkeypatch.setattr(group._backend, "launch", FailingLaunch())
    with pytest.raises(RuntimeError, match="scatter enqueue"):
        group.publish((16, 16))
    assert group._backend.host.is_pinned()
    owner.drain()
    with pytest.raises(PublicationError, match="failed"):
        owner.begin()


def test_packed_initializes_with_cuda_as_default_device():
    with torch.device("cuda"):
        owner, a, b, _, _, group = setup("cuda")
        assert group.use_transport("packed") == "packed"
        assert group._backend.host.device.type == "cpu"
        owner.begin()
        a.cpu.fill_(51)
        b.cpu.fill_(61)
        group.publish((16, 16))
        owner.finish()
        owner.completion.synchronize()
        assert torch.all(a.gpu.cpu() == 51) and torch.all(b.gpu.cpu() == 61)


@pytest.mark.parametrize("strided", [False, True])
def test_owner_packs_consumer_boundaries_and_keeps_supported_fallback(strided):
    owner, a, b, x, y, producer = setup("cuda")
    c = buffer(16, device="cuda")
    if strided:
        c.cpu, c.gpu = c.cpu[::2], c.gpu[::2]
    z = owner.bind(c, "c")
    consumer = owner.group("consumer", (x, y, z))
    owner.use_packed_transport()
    # Unsupported large groups must not disable packing a supported producer.
    assert consumer.transport == ("direct" if strided else "packed")
    assert producer.transport == ("packed" if strided else "direct")
    assert (producer._backend is None) != (consumer._backend is None)
    owner.begin()
    a.cpu.fill_(11)
    b.cpu.fill_(22)
    c.cpu.fill_(33)
    consumer.publish((16, 16, c.cpu.shape[0]))
    result = (a.gpu.clone(), b.gpu.clone(), c.gpu.clone())
    with pytest.raises(PublicationError, match="republish_reason"):
        producer.publish((16, 16))
    owner.finish()
    # The producer remains usable independently in the next epoch.
    owner.begin()
    a.cpu.fill_(44)
    b.cpu.fill_(55)
    producer.publish((16, 16))
    owner.finish()
    owner.completion.synchronize()
    assert all(torch.all(t.cpu() == v) for t, v in zip(result, (11, 22, 33)))
    assert torch.all(a.gpu.cpu() == 44) and torch.all(b.gpu.cpu() == 55)


def test_disjoint_single_member_publications_do_not_borrow_packed_header(monkeypatch):
    owner, a, b, _, _, group = setup("cuda", shape=(8,))
    assert group.use_transport("packed") == "packed"

    def unexpected_wait():
        pytest.fail("a fresh member must not wait for another member's direct DMA")

    monkeypatch.setattr(owner, "_wait_sources", unexpected_wait)
    owner.begin()
    try:
        a.cpu.fill_(17)
        torch.cuda._sleep(3_000_000)
        group.publish((8, None))
        first = a.gpu.clone()
        b.cpu.fill_(23)
        group.publish((None, 8))
        second = b.gpu.clone()
        with pytest.raises(PublicationError, match="republish_reason"):
            group.publish((8, None))
    finally:
        owner.finish()
        owner.completion.synchronize()
    assert first.cpu().tolist() == [17] * 8
    assert second.cpu().tolist() == [23] * 8


def test_direct_member_preserves_an_in_flight_packed_arena():
    owner = PublicationOwner("cuda", torch.cuda.Event())
    buffers = [buffer(8, device="cuda") for _ in range(5)]
    members = [owner.bind(buf, str(i)) for i, buf in enumerate(buffers)]
    group = owner.group("sparse", members)
    assert group.use_transport("packed") == "packed"
    owner.begin()
    group.publish((8, 8, None, None, None))  # Compile before delaying the queue.
    owner.finish()
    owner.begin()
    try:
        for i, buf in enumerate(buffers):
            buf.cpu.fill_(11 + i)
        torch.cuda._sleep(3_000_000)
        group.publish((8, 8, None, None, None))
        observed = [buf.gpu.clone() for buf in buffers[:2]]
        # Direct DMA can publish a fresh member while the arena is borrowed.
        group.publish((None, None, 8, None, None))
        observed.append(buffers[2].gpu.clone())
        # That direct copy must not release the preceding packed source.
        with pytest.raises(PublicationError, match="transport counts are in flight"):
            group.publish((None, None, None, 8, 8))
        members[0].acquire_write(republish_reason="finish the earlier packed read")
        group.publish((None, None, None, 8, 8))
        observed.extend(buf.gpu.clone() for buf in buffers[3:])
    finally:
        owner.finish()
        owner.completion.synchronize()
    assert [value.cpu().tolist() for value in observed] == [
        [11 + i] * 8 for i in range(5)
    ]

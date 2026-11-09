# SPDX-License-Identifier: MIT
"""Real DMA/scatter correctness, asynchronous source reuse."""

import os
from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_without_aiter

from atom.utils.h2d import HostArenaReader, PublicationError, PublicationOwner
from tests.test_h2d_publication import _publish, buffer, setup

gpu = pytest.mark.skipif(
    os.environ.get("RUN_H2D_GPU_TESTS") != "1", reason="set RUN_H2D_GPU_TESTS=1"
)


@gpu
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
    assert group.use_packed() == "packed"
    pointers = [b.gpu.data_ptr() for b in buffers]
    expected = [
        torch.full((b.cpu.numel() * b.cpu.element_size(),), 199, dtype=torch.uint8)
        for b in buffers
    ]
    owner.begin()
    _publish(group, [b.capacity for b in members])  # Compile before delayed work.
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
        _publish(group, counts)
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


@gpu
def test_packed_arena_reuse_and_atomic_validation():
    owner, a, b, _, _, group = setup("cuda", shape=(1031,))
    assert group.use_packed() == "packed"
    owner.begin()
    _publish(group, (1031, 1031))
    owner.finish()
    observations = []
    for step in range(6):
        owner.begin()
        a.cpu.fill_(step + 11)
        b.cpu.fill_(step + 21)
        header = group._backend.header.copy()
        with pytest.raises(ValueError):
            _publish(group, (17, 1032))
        assert (header == group._backend.header).all()
        torch.cuda._sleep(3_000_000)
        _publish(group, (1031, 1031))
        observations.append((a.gpu.clone(), b.gpu.clone()))
        with pytest.raises(PublicationError, match="republish_reason"):
            _publish(group, (1, None))
        owner.finish()
    owner.completion.synchronize()
    for step, (first, second) in enumerate(observations):
        assert torch.all(first.cpu() == step + 11)
        assert torch.all(second.cpu() == step + 21)


@gpu
def test_packed_failure_after_dma_poison_retains_arena(monkeypatch):
    owner, a, b, _, _, group = setup("cuda")
    assert group.use_packed() == "packed"
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
        _publish(group, (16, 16))
    assert group._backend.host.is_pinned()
    owner.drain()
    with pytest.raises(PublicationError, match="failed"):
        owner.begin()


@gpu
def test_packed_initializes_with_cuda_as_default_device():
    with torch.device("cuda"):
        owner, a, b, _, _, group = setup("cuda")
        assert group.use_packed() == "packed"
        assert group._backend.host.device.type == "cpu"
        owner.begin()
        a.cpu.fill_(51)
        b.cpu.fill_(61)
        _publish(group, (16, 16))
        owner.finish()
        owner.completion.synchronize()
        assert torch.all(a.gpu.cpu() == 51) and torch.all(b.gpu.cpu() == 61)


@gpu
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
    _publish(consumer, (16, 16, c.cpu.shape[0]))
    result = (a.gpu.clone(), b.gpu.clone(), c.gpu.clone())
    with pytest.raises(PublicationError, match="republish_reason"):
        _publish(producer, (16, 16))
    owner.finish()
    # The producer remains usable independently in the next epoch.
    owner.begin()
    a.cpu.fill_(44)
    b.cpu.fill_(55)
    _publish(producer, (16, 16))
    owner.finish()
    owner.completion.synchronize()
    assert all(torch.all(t.cpu() == v) for t, v in zip(result, (11, 22, 33)))
    assert torch.all(a.gpu.cpu() == 44) and torch.all(b.gpu.cpu() == 55)


@gpu
def test_a_boundary_another_covers_still_packs():
    """Two phases may publish the same members: the narrower boundary still
    gets its own arena, while a producer group they cover stays direct."""
    owner, a, b, x, y, producer = setup("cuda")
    z = owner.bind(buffer(16, device="cuda"), "c")
    wide = owner.group("prefill_metadata", (x, y, z))
    narrow = owner.group("decode_metadata", (x, y))
    owner.use_packed_transport({"prefill_metadata", "decode_metadata"})
    assert wide.transport == narrow.transport == "packed"
    assert producer.transport == "direct"
    owner.begin()
    a.cpu.fill_(7)
    b.cpu.fill_(8)
    _publish(narrow, (16, 16))
    owner.finish()
    owner.completion.synchronize()
    assert torch.all(a.gpu.cpu() == 7) and torch.all(b.gpu.cpu() == 8)


@gpu
def test_disjoint_single_member_publications_do_not_borrow_packed_header(monkeypatch):
    owner, a, b, _, _, group = setup("cuda", shape=(8,))
    assert group.use_packed() == "packed"

    def unexpected_wait():
        pytest.fail("a fresh member must not wait for another member's direct DMA")

    monkeypatch.setattr(owner, "_wait_sources", unexpected_wait)
    owner.begin()
    try:
        a.cpu.fill_(17)
        torch.cuda._sleep(3_000_000)
        _publish(group, (8, None))
        first = a.gpu.clone()
        b.cpu.fill_(23)
        _publish(group, (None, 8))
        second = b.gpu.clone()
        with pytest.raises(PublicationError, match="republish_reason"):
            _publish(group, (8, None))
    finally:
        owner.finish()
        owner.completion.synchronize()
    assert first.cpu().tolist() == [17] * 8
    assert second.cpu().tolist() == [23] * 8


@gpu
def test_direct_member_preserves_an_in_flight_packed_arena():
    owner = PublicationOwner("cuda", torch.cuda.Event())
    buffers = [buffer(8, device="cuda") for _ in range(5)]
    members = [owner.bind(buf, str(i)) for i, buf in enumerate(buffers)]
    group = owner.group("sparse", members)
    assert group.use_packed() == "packed"
    owner.begin()
    _publish(group, (8, 8, None, None, None))  # Compile before delaying the queue.
    owner.finish()
    owner.begin()
    try:
        for i, buf in enumerate(buffers):
            buf.cpu.fill_(11 + i)
        torch.cuda._sleep(3_000_000)
        _publish(group, (8, 8, None, None, None))
        observed = [buf.gpu.clone() for buf in buffers[:2]]
        # Direct DMA can publish a fresh member while the arena is borrowed.
        _publish(group, (None, None, 8, None, None))
        observed.append(buffers[2].gpu.clone())
        # That direct copy must not release the preceding packed source.
        with pytest.raises(PublicationError, match="transport counts are in flight"):
            _publish(group, (None, None, None, 8, 8))
        members[0].acquire_write(republish_reason="finish the earlier packed read")
        _publish(group, (None, None, None, 8, 8))
        observed.extend(buf.gpu.clone() for buf in buffers[3:])
    finally:
        owner.finish()
        owner.completion.synchronize()
    assert [value.cpu().tolist() for value in observed] == [
        [11 + i] * 8 for i in range(5)
    ]


class _Reader(HostArenaReader):
    """Reads every publication's host arena; with `takes`, records every
    scatter instead of launching it."""

    def __init__(self, takes=False):
        self.takes, self.taken = takes, []

    def reads(self, nbytes):
        return True

    def take(self, copy, arena, largest):
        if not self.takes:
            return False
        self.taken.append((arena, largest))
        return True


@gpu
def test_packed_scatter_reads_the_host_arena():
    # The scatter reads the pinned arena itself (`read_host_arena`); reuse
    # waits on the owner's event exactly as with the copy (the delay keeps
    # every launch queued).
    owner, a, b, _, _, group = setup("cuda", shape=(1031,))
    assert group.use_packed() == "packed"
    assert group.read_host_arena(_Reader())
    owner.begin()
    _publish(group, (1031, 1031))  # Compile before reading the arena.
    owner.finish()
    observations = []
    for step, counts in enumerate(((1031, 1031), (5, 1031), (1031, None), (1, 2))):
        owner.begin()
        a.cpu.fill_(step + 11)
        b.cpu.fill_(step + 21)
        torch.cuda._sleep(3_000_000)
        _publish(group, counts)
        observations.append((counts, a.gpu.clone(), b.gpu.clone()))
        owner.finish()
    owner.completion.synchronize()
    for step, (counts, first, second) in enumerate(observations):
        for count, got, value in zip(counts, (first, second), (step + 11, step + 21)):
            assert count is None or torch.all(got[:count].cpu() == value)


@gpu
def test_a_deferred_scatter_is_not_launched():
    # A reader takes the scatter (DeepSeek-V4.1's hi prologue launches it
    # later); declining, the publication launches as before.
    owner, a, b, _, _, group = setup("cuda", shape=(1031,))
    assert group.use_packed() == "packed"
    reader = _Reader(takes=True)
    assert group.read_host_arena(reader)
    owner.begin()
    _publish(group, (1031, 1031))
    owner.finish()
    owner.completion.synchronize()
    taken = len(reader.taken)
    owner.begin()
    a.cpu.fill_(5)
    b.cpu.fill_(6)
    before = a.gpu.clone()
    _publish(group, (1031, 1031))
    owner.finish()
    owner.completion.synchronize()
    assert len(reader.taken) == taken + 1
    assert reader.taken[-1] == (
        group._backend.host_arena.data_ptr(),
        1031 * a.cpu.element_size(),
    )
    assert torch.equal(a.gpu, before)  # not scattered
    reader.takes = False
    owner.begin()
    _publish(group, (1031, 1031))
    owner.finish()
    owner.completion.synchronize()
    assert torch.all(a.gpu.cpu() == 5) and torch.all(b.gpu.cpu() == 6)


def test_a_direct_group_does_not_read_a_host_arena():
    owner, _, _, _, _, group = setup("cpu")
    assert group.transport == "direct"
    assert group.read_host_arena(_Reader()) is False


def test_host_arena_reads_opt_in_at_initialization_only():
    owner, _, _, _, _, group = setup("cpu")
    owner.begin()
    with pytest.raises(PublicationError):
        group.read_host_arena(_Reader())


def test_the_default_reader_reads_and_takes_nothing():
    reader = HostArenaReader()
    assert not reader.reads(16) and not reader.take(None, 0, 0)


class _Kernel:
    def __init__(self, launches):
        self.launches = launches

    def __getitem__(self, grid):
        return lambda arena, destinations: self.launches.append(arena)


@pytest.fixture(scope="module")
def packed_h2d():
    """`atom.utils.packed_h2d`, its Triton kernel on the repo's stand-ins where
    AITER is not installed (the CPU job): the tests below drive its host
    logic only."""
    with atom_without_aiter() as load:
        yield load("atom.utils.packed_h2d")


def _copy(packed_h2d, reader, monkeypatch, *, capturing=False):
    """A `PackedCopy` of two 8-byte members on stand-ins (no GPU): the arena
    copies and the scatter launches recorded."""
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    members = [
        SimpleNamespace(
            bytes_per_count=1, source=torch.arange(8, dtype=torch.uint8) + 10 * i
        )
        for i in range(2)
    ]
    copy = object.__new__(packed_h2d.PackedCopy)
    copy.members = members
    copy.header_bytes = 32
    copy.host = torch.zeros(48, dtype=torch.uint8)
    copy.device = torch.zeros(48, dtype=torch.uint8)
    copy.header = copy.host[:32].view(torch.int64).numpy()
    copy.payload = memoryview(copy.host.numpy())
    copy.sources = tuple(memoryview(m.source.numpy()) for m in members)
    copy.destinations = torch.zeros(2, dtype=torch.int64)
    copy.prefixes = {}
    launches = []
    copy.kernel = _Kernel(launches)
    copy.reader = reader
    copy.host_arena = None if reader is None else packed_h2d.DeviceAddress(0x1000)
    return copy, launches


class _Declines(_Reader):
    def reads(self, nbytes):
        self.asked = nbytes
        return False


@pytest.mark.parametrize("capturing", [False, True])
@pytest.mark.parametrize("reader", [None, "declines", "reads"])
def test_only_an_accepted_eager_launch_reads_the_host_arena(
    monkeypatch, packed_h2d, capturing, reader
):
    """The copy then the scatter from the device arena, unless the reader
    accepts an eager launch: then the scatter reads the host arena (no H2D);
    a capture always copies."""
    reader = {None: None, "declines": _Declines(), "reads": _Reader()}[reader]
    copy, launches = _copy(packed_h2d, reader, monkeypatch, capturing=capturing)
    assert copy.submit([8, 8])
    (arena,) = launches
    if type(reader) is _Reader and not capturing:
        assert arena.data_ptr() == 0x1000 and not copy.device.any()  # no H2D
    else:
        assert arena is copy.device and torch.equal(copy.device, copy.host)
    if isinstance(reader, _Declines) and not capturing:
        assert reader.asked == 48  # header and payload


def test_a_taken_scatter_is_neither_copied_nor_launched(monkeypatch, packed_h2d):
    reader = _Reader(takes=True)
    copy, launches = _copy(packed_h2d, reader, monkeypatch)
    assert copy.submit([8, 3])
    assert reader.taken == [(0x1000, 8)] and launches == []
    assert not copy.device.any()


def test_a_device_address_without_a_hip_runtime(monkeypatch, packed_h2d):
    """`device_address` asks the HIP runtime torch runs; without one it raises, or (`identity_without_hip`: the PLE
    table's copies, ``qwen38.mono.ple``) gives the host address back."""
    monkeypatch.setattr(packed_h2d, "_host_get_device_pointer", lambda: None)
    with pytest.raises(RuntimeError, match="no HIP runtime"):
        packed_h2d.device_address(0x1000)
    assert packed_h2d.device_address(0x1040, identity_without_hip=True) == 0x1040

    def get(
        out, base, flags
    ):  # hipHostGetDevicePointer: the range at base is mapped at 0x9000
        out._obj.value = 0x9000 if base == 0x1000 else 0
        return 0

    monkeypatch.setattr(packed_h2d, "_host_get_device_pointer", lambda: get)
    assert packed_h2d.device_address(0x1040, 0x1000) == 0x9040
    assert packed_h2d.device_address(0x1000, identity_without_hip=True) == 0x9000
    with pytest.raises(RuntimeError, match="hipHostGetDevicePointer"):
        packed_h2d.device_address(0x2000)

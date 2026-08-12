# SPDX-License-Identifier: MIT
# The staging half of the KV path, now shared with the state tier. These are
# characterization tests: the extraction must not change behaviour.

import pytest
import torch

from atom.kv_transfer.offload.staged_transfer import StagedTransfer, _StagingBuffer

CPU = torch.device("cpu")


def test_buffer_is_allocated_once_and_reused():
    st = StagedTransfer(CPU, staging_buffer_bytes=1024)
    buf = _StagingBuffer()
    first = st.ensure_buffer(buf, 512)
    second = st.ensure_buffer(buf, 256)
    assert first.data_ptr() == second.data_ptr()
    assert int(second.numel()) == 256


def test_a_request_larger_than_the_buffer_is_an_error_not_a_realloc():
    """The buffer is bounded on purpose — silently growing it would put the
    HBM ceiling back in the hands of whatever the largest group happened to be."""
    st = StagedTransfer(CPU, staging_buffer_bytes=1024)
    with pytest.raises(RuntimeError, match="exceeds bounded GPU staging buffer"):
        st.ensure_buffer(_StagingBuffer(), 2048)


def test_release_drops_the_tensor_when_asked():
    st = StagedTransfer(CPU, staging_buffer_bytes=1024, release_after_transfer=True)
    buf = _StagingBuffer()
    st.ensure_buffer(buf, 512)
    st.release_buffer_if_requested(buf)
    assert buf.tensor is None


def test_release_is_a_no_op_by_default():
    st = StagedTransfer(CPU, staging_buffer_bytes=1024)
    buf = _StagingBuffer()
    st.ensure_buffer(buf, 512)
    st.release_buffer_if_requested(buf)
    assert buf.tensor is not None


def test_memory_tensor_rejects_a_non_uint8_object():
    st = StagedTransfer(CPU, staging_buffer_bytes=1024)

    class Obj:
        tensor = torch.zeros(64, dtype=torch.float16)

    with pytest.raises(TypeError, match="must be uint8"):
        st.memory_tensor(Obj(), 64)


def test_memory_tensor_rejects_an_object_that_is_too_small():
    st = StagedTransfer(CPU, staging_buffer_bytes=1024)

    class Obj:
        tensor = torch.zeros(16, dtype=torch.uint8)

    with pytest.raises(ValueError, match="too small"):
        st.memory_tensor(Obj(), 64)


def test_thread_state_is_per_device_and_cached():
    st = StagedTransfer(CPU, staging_buffer_bytes=1024)
    assert st.thread_state() is st.thread_state()


# ---------------------------------------------------------------------------
# pack / unpack -- the state tier's whole-entry transfer. GPU only: the Triton
# packer refuses a non-CUDA segment by design.
# ---------------------------------------------------------------------------

_needs_gpu = pytest.mark.skipif(
    not (hasattr(torch, "cuda") and torch.cuda.is_available()),
    reason="a ROCm GPU is required",
)


class _HostObj:
    """A MemoryObj whose bytes live on the host, which is the normal case."""

    def __init__(self, nbytes):
        self.tensor = torch.zeros(nbytes, dtype=torch.uint8)


class _DeviceObj:
    def __init__(self, nbytes, device):
        self.tensor = torch.zeros(nbytes, dtype=torch.uint8, device=device)


def _segments(device, sizes=(96, 32, 160)):
    """Segments of deliberately differing sizes -- GDN's k-views and v-views
    are not the same size, and `_build_meta` sums them per block."""
    return [
        torch.arange(n, dtype=torch.uint8, device=device).add_(i * 7)
        for i, n in enumerate(sizes)
    ]


def _nbytes(segments):
    return sum(int(s.numel()) * s.element_size() for s in segments)


@_needs_gpu
def test_pack_unpack_round_trips_bit_exactly_through_a_host_object():
    """The whole point of the state tier: what comes back must be the same
    bytes, across a D2H hop the KV path also makes."""
    device = torch.device("cuda")
    src = _segments(device)
    st = StagedTransfer(device, staging_buffer_bytes=1 << 16)
    dst = _HostObj(_nbytes(src))
    st.pack(src, dst)

    out = [torch.zeros_like(s) for s in src]
    st.unpack(dst, out)
    for a, b in zip(src, out, strict=True):
        assert torch.equal(a, b)


@_needs_gpu
def test_pack_writes_straight_into_a_device_object():
    """No staging hop is needed when the object already lives on our device."""
    device = torch.device("cuda")
    src = _segments(device)
    st = StagedTransfer(device, staging_buffer_bytes=1 << 16)
    dst = _DeviceObj(_nbytes(src), device)
    st.pack(src, dst)
    torch.cuda.synchronize()
    assert torch.equal(dst.tensor, torch.cat([s.reshape(-1) for s in src]))

    out = [torch.zeros_like(s) for s in src]
    st.unpack(dst, out)
    for a, b in zip(src, out, strict=True):
        assert torch.equal(a, b)


@_needs_gpu
def test_a_destination_too_small_raises_rather_than_truncating():
    """Silent truncation would store a half-entry that unpacks as garbage."""
    device = torch.device("cuda")
    src = _segments(device)
    st = StagedTransfer(device, staging_buffer_bytes=1 << 16)
    with pytest.raises(ValueError, match="too small"):
        st.pack(src, _HostObj(_nbytes(src) - 1))


@_needs_gpu
def test_pack_refuses_an_entry_larger_than_the_bounded_staging_buffer():
    device = torch.device("cuda")
    src = _segments(device)
    st = StagedTransfer(device, staging_buffer_bytes=64)
    with pytest.raises(RuntimeError, match="exceeds bounded GPU staging buffer"):
        st.pack(src, _HostObj(_nbytes(src)))

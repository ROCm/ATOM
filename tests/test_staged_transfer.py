# SPDX-License-Identifier: MIT
# The staging half of the KV path, now shared with the state tier. These are
# characterization tests: the extraction must not change behaviour.

import sys
import types

import pytest
import torch

from atom.kv_transfer.offload.staged_transfer import (
    StagedTransfer,
    _NullCtx,
    _StagingBuffer,
)

CPU = torch.device("cpu")


def test_buffer_is_allocated_once_and_reused():
    st = StagedTransfer(CPU, staging_buffer_bytes=1024)
    buf = _StagingBuffer(use_cuda=False)
    first = st.ensure_buffer(buf, 512)
    second = st.ensure_buffer(buf, 256)
    assert first.data_ptr() == second.data_ptr()
    assert int(second.numel()) == 256


def test_a_request_larger_than_the_buffer_is_an_error_not_a_realloc():
    """The buffer is bounded on purpose — silently growing it would put the
    HBM ceiling back in the hands of whatever the largest group happened to be."""
    st = StagedTransfer(CPU, staging_buffer_bytes=1024)
    with pytest.raises(RuntimeError, match="exceeds bounded GPU staging buffer"):
        st.ensure_buffer(_StagingBuffer(use_cuda=False), 2048)


def test_release_drops_the_tensor_when_asked():
    st = StagedTransfer(CPU, staging_buffer_bytes=1024, release_after_transfer=True)
    buf = _StagingBuffer(use_cuda=False)
    st.ensure_buffer(buf, 512)
    st.release_buffer_if_requested(buf)
    assert buf.tensor is None


def test_release_is_a_no_op_by_default():
    st = StagedTransfer(CPU, staging_buffer_bytes=1024)
    buf = _StagingBuffer(use_cuda=False)
    st.ensure_buffer(buf, 512)
    st.release_buffer_if_requested(buf)
    assert buf.tensor is not None


@pytest.mark.parametrize("value", ["", " ", "off ", " off", "0", "false", "no", "OFF"])
def test_env_flag_reads_empty_and_padded_values_as_off(monkeypatch, value):
    """`VAR=` must mean off, not on.

    Clearing a flag inline (`OFFLOAD_RELEASE_GPU_STAGING_AFTER_TRANSFER=`) is
    ordinary shell, and the original `not in ("0", "false", "no", "off")` test
    read the empty string as ON -- the opposite of what was written. A stray
    trailing space did the same.
    """
    from atom.kv_transfer.offload.staged_transfer import _env_flag

    monkeypatch.setenv("ATOM_TEST_STAGED_FLAG", value)
    assert _env_flag("ATOM_TEST_STAGED_FLAG") is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " on "])
def test_env_flag_still_reads_the_usual_on_spellings_as_on(monkeypatch, value):
    from atom.kv_transfer.offload.staged_transfer import _env_flag

    monkeypatch.setenv("ATOM_TEST_STAGED_FLAG", value)
    assert _env_flag("ATOM_TEST_STAGED_FLAG") is True


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
# pack / unpack -- the state tier's whole-entry transfer. Only the round-trips
# are GPU-gated: `_build_meta` refuses a non-CUDA segment by design. The guard
# and ordering tests below run everywhere, because CI has no GPU.
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


def test_a_destination_too_small_raises_rather_than_truncating():
    """Silent truncation would store a half-entry that unpacks as garbage.

    No GPU: `memory_tensor` sizes the destination before `pack` touches a
    device, so the guard is reachable with CPU segments.
    """
    src = _segments(CPU)
    st = StagedTransfer(CPU, staging_buffer_bytes=1 << 16)
    with pytest.raises(ValueError, match="too small"):
        st.pack(src, _HostObj(_nbytes(src) - 1))


def test_pack_refuses_an_entry_larger_than_the_bounded_staging_buffer():
    """Also pre-device: `ensure_buffer` refuses before any kernel launch."""
    src = _segments(CPU)
    st = StagedTransfer(CPU, staging_buffer_bytes=64)
    with pytest.raises(RuntimeError, match="exceeds bounded GPU staging buffer"):
        st.pack(src, _HostObj(_nbytes(src)))


# ---------------------------------------------------------------------------
# The intra-worker stage hand-off. Recorded stubs stand in for the CUDA streams
# and events, so the record/wait/synchronize *ordering* -- the part a skipped
# GPU test never covers -- is pinned on CPU. Only the kernels and the real
# events need a GPU; the sequence does not.
#
# Note this is NOT commit 7427e05e's fence. That one is recorded on the RPC
# thread and synchronized by the caller (`connector.py`'s `save_ready_event`,
# `state_tier._do_spill`'s `ready_event`); these events only order one worker
# thread's pack_stream against its own copy_stream.
# ---------------------------------------------------------------------------


class _RecordingStream:
    def __init__(self, log, name):
        self._log = log
        self._name = name

    def wait_event(self, event):
        self._log.append(f"wait:{event.name}@{self._name}")

    def synchronize(self):
        self._log.append(f"sync:{self._name}")


class _RecordingEvent:
    def __init__(self, log, name):
        self._log = log
        self.name = name

    def record(self, stream):
        self._log.append(f"record:{self.name}@{stream._name}")


def _instrument(st, monkeypatch, kernel_name):
    """Swap in recorded streams/events and a stub Triton module.

    The kernel module is replaced via `sys.modules` because `pack`/`unpack`
    import it inside the function; the stub keeps Triton (and thus a GPU) out
    of the picture entirely.
    """
    log = []
    state = st.thread_state()
    state.pack_stream = _RecordingStream(log, "pack_stream")
    state.copy_stream = _RecordingStream(log, "copy_stream")
    state.staging_buffer.ready_event = _RecordingEvent(log, "ready")
    state.staging_buffer.free_event = _RecordingEvent(log, "free")
    # Instance attribute shadows the method: the real one would build a
    # `torch.cuda.stream` context around our stub.
    state.stream_ctx = lambda stream: _NullCtx()

    module = types.ModuleType("atom.kv_transfer.offload.triton_kv_staging")

    def _kernel(*args, **kwargs):
        log.append(kernel_name)

    module.fused_pack_chunk_major = _kernel
    module.fused_unpack_chunk_major = _kernel
    monkeypatch.setitem(
        sys.modules, "atom.kv_transfer.offload.triton_kv_staging", module
    )
    return log, state


def test_pack_records_the_producer_event_before_the_consumer_waits(monkeypatch):
    """Pack -> record(ready) -> copy stream waits -> D2H -> record(free) ->
    synchronize. Whoever reads the MemoryObj next must not see an in-flight
    copy, which is exactly what dropping the wait would allow."""
    src = _segments(CPU)
    st = StagedTransfer(CPU, staging_buffer_bytes=1 << 16)
    log, state = _instrument(st, monkeypatch, "pack-kernel")
    dst = _HostObj(_nbytes(src))

    st.pack(src, dst)

    assert log == [
        "pack-kernel",
        "record:ready@pack_stream",
        "wait:ready@copy_stream",
        "record:free@copy_stream",
        "sync:copy_stream",
    ]
    assert state.staging_buffer.free_event_valid is True


def test_unpack_synchronizes_the_stream_that_writes_the_segments(monkeypatch):
    """The mirror: H2D on the copy stream, then the kernel stream waits, and
    it is the *kernel* stream that is synchronized -- it produces the
    observable result, so syncing the copy stream would hand the segments back
    while they are still being written."""
    src = _segments(CPU)
    st = StagedTransfer(CPU, staging_buffer_bytes=1 << 16)
    log, state = _instrument(st, monkeypatch, "unpack-kernel")

    st.unpack(_HostObj(_nbytes(src)), src)

    assert log == [
        "record:ready@copy_stream",
        "wait:ready@pack_stream",
        "unpack-kernel",
        "record:free@pack_stream",
        "sync:pack_stream",
    ]
    assert state.staging_buffer.free_event_valid is True


def test_a_failed_transfer_invalidates_the_free_event(monkeypatch):
    """A free event left valid after a transfer that never completed would let
    the next group's `run_pipeline` wait on it and overwrite the buffer early.

    The first pack is what makes this non-vacuous: it both validates the flag
    and leaves the buffer allocated, so `ensure_buffer` does not reset the flag
    on the second call and only the `except` clause can clear it.
    """
    src = _segments(CPU)
    st = StagedTransfer(CPU, staging_buffer_bytes=1 << 16)
    log, state = _instrument(st, monkeypatch, "pack-kernel")
    dst = _HostObj(_nbytes(src))

    st.pack(src, dst)
    assert state.staging_buffer.free_event_valid is True

    def _boom(*args, **kwargs):
        raise RuntimeError("kernel blew up")

    sys.modules["atom.kv_transfer.offload.triton_kv_staging"].fused_pack_chunk_major = (
        _boom
    )
    log.clear()

    with pytest.raises(RuntimeError, match="kernel blew up"):
        st.pack(src, dst)
    assert state.staging_buffer.free_event_valid is False
    assert log == []

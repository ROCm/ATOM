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
